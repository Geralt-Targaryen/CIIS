# tingis/event_management/event_snapshotter.py

import time
import logging
from datetime import datetime, timedelta

from tingis.utils.common_utils import timestamp_to_str, BEIJING_TZ
from tingis.knowledge_base.ob_db_manager.ob_db_operations import OBDatabaseManager
from tingis.global_vars import (
    risk_event_volume_timeline_table_name,
    risk_event_table_name,
    risk_event_voice_mapping_table_name,
    SNAPSHOT_HEARTBEAT_MINUTES,
    SNAPSHOT_ACTIVE_EVENT_LOOKBACK_HOURS,
    SNAPSHOT_INTERNAL_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

# 定义内部子批次大小，用于批量计算增量。
# 这是一个关键的性能调优参数，用于平衡查询次数和单次查询复杂度
INCREMENT_CALC_SUB_BATCH_SIZE = 200

def perform_snapshot_if_needed(db_manager: OBDatabaseManager, last_snapshot_ts: int) -> int:
    """
    对活跃事件进行分批快照。此版本为最终生产级实现，具备高性能和高可扩展性。

    核心逻辑：
    1. 心跳检测：判断是否到达快照时间点。
    2. 外层循环 (ID分页): 使用 `risk_event_id` 作为游标，处理海量的活跃事件，避免内存溢出。
    3. 内层循环 (子批次处理): 将外层循环获取到的一大批事件，再切分成更小的批次来计算增量声量，
       解决了N+1查询问题，同时避免了单次SQL查询因条件过长(OR...OR...)而导致的性能下降或超时。
    4. 批量写入: 将计算出的快照数据批量写入数据库。
    5. 容错: 整个过程包裹在try...except中，失败时返回旧的时间戳，以便下次重试。

    Args:
        db_manager (OBDatabaseManager): 数据库操作管理器实例。
        last_snapshot_ts (int): 上一次成功执行快照的Unix时间戳。

    Returns:
        int: 最新的快照时间戳（成功则为当前时间，失败则为旧时间）。
    """
    current_ts = int(time.time())
    snapshot_interval_seconds = SNAPSHOT_HEARTBEAT_MINUTES * 60

    if current_ts - last_snapshot_ts < snapshot_interval_seconds:
        return last_snapshot_ts

    logger.info("[Snapshotter] Heartbeat trigger. Performing a snapshot...")
    try:
        current_time_str = timestamp_to_str(current_ts)
        
        # 计算活跃事件的时间窗口起点
        active_lookback_hours = SNAPSHOT_ACTIVE_EVENT_LOOKBACK_HOURS
        time_cutoff_dt = datetime.now(BEIJING_TZ) - timedelta(hours=active_lookback_hours)
        time_cutoff_str = time_cutoff_dt.strftime('%Y-%m-%d %H:%M:%S')
        
        last_processed_risk_event_id = ""
        total_snapshots_created = 0
        
        # === 外层循环：ID分页处理所有活跃事件 ===
        while True:
            # Step 1: 获取一大批活跃事件 (e.g., 6000)
            sql_active_events_paginated = f"""
                SELECT risk_event_id, risk_event_volume, initial_timestamp
                FROM {risk_event_table_name}
                WHERE 
                    status = 'active' AND
                    risk_event_id > %s AND
                    GREATEST(
                        COALESCE(initial_timestamp, '1970-01-01'),
                        COALESCE(threshold_breach_timestamp, '1970-01-01'),
                        COALESCE(last_alerted_timestamp, '1970-01-01')
                    ) >= %s
                ORDER BY risk_event_id ASC
                LIMIT %s
            """
            active_events_batch = db_manager.execute_query(sql_active_events_paginated, (last_processed_risk_event_id, time_cutoff_str, SNAPSHOT_INTERNAL_BATCH_SIZE))

            if not active_events_batch:
                break
            last_processed_risk_event_id = active_events_batch[-1]['risk_event_id']
            
            # Step 2: 批量获取这一大批事件的上次快照时间
            event_ids_batch = [event['risk_event_id'] for event in active_events_batch]
            placeholders = ', '.join(['%s'] * len(event_ids_batch))
            sql_last_snapshot = f"""
                SELECT risk_event_id, MAX(snapshot_time) as last_snapshot
                FROM {risk_event_volume_timeline_table_name}
                WHERE risk_event_id IN ({placeholders})
                GROUP BY risk_event_id
            """
            last_snapshot_results = db_manager.execute_query(sql_last_snapshot, tuple(event_ids_batch))
            last_snapshot_map = {res['risk_event_id']: res['last_snapshot'] for res in last_snapshot_results} if last_snapshot_results else {}

            all_incremental_volumes = {}

            # === 内层循环：将大批次切分为小批次计算增量 ===
            for i in range(0, len(active_events_batch), INCREMENT_CALC_SUB_BATCH_SIZE):
                sub_batch_events = active_events_batch[i : i + INCREMENT_CALC_SUB_BATCH_SIZE]
                
                # Step 3.1: 为子批次构造动态 WHERE 条件
                conditions = []
                params = []
                for event in sub_batch_events:
                    event_id = event['risk_event_id']
                    last_snapshot_time_obj = last_snapshot_map.get(event_id)
                    
                    if last_snapshot_time_obj:
                        # 老事件：增量计算区间为 (last_snapshot_time, current_time]
                        conditions.append("(risk_event_id = %s AND voice_date > %s)")
                        params.extend([event_id, last_snapshot_time_obj.strftime('%Y-%m-%d %H:%M:%S')])
                    else:
                        # 新事件：增量计算区间为 [initial_timestamp, current_time]
                        conditions.append("(risk_event_id = %s AND voice_date >= %s)")
                        params.extend([event_id, event['initial_timestamp'].strftime('%Y-%m-%d %H:%M:%S')])
                
                if not conditions:
                    continue

                # Step 3.2: 对子批次执行合并查询
                sql_bulk_increment = f"""
                    SELECT risk_event_id, COUNT(voice_id) as increment_count
                    FROM {risk_event_voice_mapping_table_name}
                    WHERE ({' OR '.join(conditions)}) AND voice_date <= %s
                    GROUP BY risk_event_id
                """
                sub_batch_params = tuple(params) + (current_time_str,)
                
                increment_results = db_manager.execute_query(sql_bulk_increment, sub_batch_params)
                
                if increment_results:
                    for row in increment_results:
                        all_incremental_volumes[row['risk_event_id']] = row['increment_count']

            # Step 3.3: 组装这一大批事件的最终待插入记录
            records_to_insert = []
            for event in active_events_batch:
                event_id = event['risk_event_id']
                records_to_insert.append({
                    'risk_event_id': event_id,
                    'snapshot_time': current_time_str,
                    'volume_at_snapshot': event['risk_event_volume'],
                    'incremental_volume_at_snapshot': all_incremental_volumes.get(event_id, 0)
                })
            
            # Step 4: 批量插入这一大批事件的快照
            if records_to_insert:
                inserted_count = db_manager.execute_insert(risk_event_volume_timeline_table_name, records_to_insert)
                if inserted_count is not None:
                    total_snapshots_created += inserted_count
                logger.info(f"[Snapshotter] Inserted {inserted_count or 0} snapshot records for this page (total so far: {total_snapshots_created}).")

        logger.info(f"[Snapshotter] Snapshot cycle finished. Total snapshots created: {total_snapshots_created}.")
        return current_ts
        
    except Exception as e:
        logger.error(f"[Snapshotter] Error during snapshot cycle: {e}", exc_info=True)
        return last_snapshot_ts
