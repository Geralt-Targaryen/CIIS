# tingis/event_management/event_manager.py

import json
import time
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from tingis.utils.common_utils import dedup_list_of_dict, str_to_timestamp, timestamp_to_str, convert_beijing_time_str_to_timestamps
from tingis.knowledge_base.index_document.risk_event_summary_kb import VectorKnowledgeBaseRiskEventSummary
from tingis.knowledge_base.ob_db_manager.ob_db_operations import OBDatabaseManager
from tingis.event_management.event_alert_engine import EventAlertEngine
from tingis.global_vars import (
    risk_event_table_name,
    risk_event_voice_mapping_table_name,
    risk_event_alert_record_table_name,
    DEFAULT_SEARCH_DATA_SOURCE,
    ALERTING_WINDOW_HOURS,
    ALERT_SILENCE_WINDOW_MINUTES,
    ALERTING_MIN_INCREMENTAL_VOLUME,
    EVENT_MAX_AGE_HOURS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s: %(lineno)d - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 全局告警引擎实例
_alert_engine_instance = None

def get_alert_engine(ob_db_manager):
    global _alert_engine_instance
    if _alert_engine_instance is None:
        _alert_engine_instance = EventAlertEngine(ob_db_manager)
    return _alert_engine_instance


def extract_final_risk_event_id(voice_data: dict) -> str | None:
    """
    从voice数据中提取最终确定的风险事件ID
    这个ID可能是新创建的，也可能是合并到一个已经存在很久的历史事件ID。
    """
    agg_result = None
    # 查找语义聚合结果
    for detail in voice_data.get('algorithm_detail', []):
        if isinstance(detail, dict) and 'semantic_aggregation_result' in detail:
            agg_result = detail['semantic_aggregation_result']
            break

    if not agg_result:
        return voice_data.get('batch_cluster_id') # 如果没有聚合结果，则使用批内ID

    decision_result = agg_result.get('llm_decision_result', {})
    decision = decision_result.get('decision')

    if decision == 'merge':
        # 如果决策是合并，使用匹配上的历史事件ID
        return decision_result.get('matched_risk_event_id')
    elif decision == 'create_new':
        # 如果决策是新建，使用批内生成的聚类ID
        return voice_data.get('batch_cluster_id')
    else:
        # 对于 'suppress_as_false_positive' 或任何其他未知情况，都会触发警告并使用批内ID
        # logger.warning(f"Invalid decision '{decision}' found for voice {voice_data.get('voice_id')}. Defaulting to batch_cluster_id.")
        return voice_data.get('batch_cluster_id')


def persist_voice_to_risk_event_mappings(
    processed_voices: List[Dict],
    ob_db_manager: OBDatabaseManager,
    mapping_table_name: str
) -> int:
    """将voice与最终的risk_event_id的映射关系写入mapping数据库"""
    if not processed_voices:
        return 0

    mapping_records = []
    for v in processed_voices:
        risk_event_id = extract_final_risk_event_id(v)
        if not risk_event_id:
            continue

        # 确保元数据是JSON字符串
        source_meta = v.get('source_metadata')
        voice_date_str = source_meta.get('gmt_create')

        if isinstance(source_meta, dict):
            source_meta_str = json.dumps(source_meta, ensure_ascii=False)
        elif isinstance(source_meta, str):
            source_meta_str = source_meta
        else:
            source_meta_str = '{}'

        record = {
            'voice_id': v['voice_id'],
            'risk_event_id': risk_event_id,
            'data_source': DEFAULT_SEARCH_DATA_SOURCE,
            'batch_cluster_id': v.get('batch_cluster_id'),
            'voice_title': v.get('voice_title'),
            'initial_summary_text': v.get('initial_summary_text'),
            'batch_cluster_title': v.get('batch_cluster_title'),
            'source_metadata': source_meta_str, # 确保是字符串
            'algorithm_detail': json.dumps(v.get('algorithm_detail', []), ensure_ascii=False),
            'voice_date': voice_date_str,
        }
        mapping_records.append(record)

    if not mapping_records:
        return 0

    try:
        ob_db_manager.execute_insert(mapping_table_name, mapping_records)
        logger.info(f"Persisted {len(mapping_records)} voice-to-risk-event mappings.")
        return len(mapping_records)
    except Exception as e:
        logger.error(f"Failed to insert into {mapping_table_name}: {e}")
        raise


def _deduplicate_voices(expanded_voices: List[Dict]) -> List[Dict]:
    """根据voice_id对原声列表进行去重。只处理批次内重复，无法防止跨批次的重复"""
    if not expanded_voices:
        return []

    unique_map = {} 

    for v in expanded_voices:
        voice_id = v.get('voice_id')
        if not voice_id:
            continue

        unique_map[voice_id] = v

    return list(unique_map.values())


def _batch_update_event_volumes(db_manager: OBDatabaseManager, volume_deltas: Dict[str, int], current_time_str: str):
    """批量更新事件声量 & 更新其最后活跃时间。"""
    if not volume_deltas:
        return

    event_ids = list(volume_deltas.keys())
    case_sql_part = " ".join([f"WHEN '{eid}' THEN risk_event_volume + {delta}" for eid, delta in volume_deltas.items()])
   
    # 构建 ID 列表，用于 WHERE IN (...)
    id_placeholders = ', '.join(['%s'] * len(event_ids))

    sql = f"""
        UPDATE {risk_event_table_name}
        SET
            risk_event_volume = CASE risk_event_id
                {case_sql_part}
                ELSE risk_event_volume
            END,
            last_active_timestamp = %s
        WHERE risk_event_id IN ({id_placeholders})
    """
    
    # 构建参数列表
    # 参数列表的顺序很重要：第一个是 current_time_str, 后面是所有的 event_id
    params = [current_time_str] + event_ids
    
    try:
        affected_rows = db_manager.execute_update(sql, tuple(params))
        logger.info(f"Batch updated volumes for {affected_rows or 0} events in a single query.")
    except Exception as e:
        logger.error(f"Failed to batch update event volumes: {e}", exc_info=True)
        # 如果批量更新失败，可以降级回逐条更新
        logger.warning("Falling back to row-by-row update due to batch failure.")
        for event_id, delta in volume_deltas.items():
            fallback_sql = f"""
                UPDATE {risk_event_table_name} 
                SET risk_event_volume = risk_event_volume + %s, last_active_timestamp = %s
                WHERE risk_event_id = %s
            """
            db_manager.execute_update(fallback_sql, (delta, current_time_str, event_id))
        logger.info(f"Updated volumes and activity timestamps for {len(volume_deltas)} events.")


def _batch_get_incremental_volumes(db_manager: OBDatabaseManager, event_ids: List[str], window_start_str: str) -> Dict[str, int]:
    """使用单次 GROUP BY 查询批量获取增量声量。"""
    if not event_ids:
        return {}
    
    placeholders = ', '.join(['%s'] * len(event_ids))
    sql = f"""
        SELECT risk_event_id, COUNT(voice_id) as count 
        FROM {risk_event_voice_mapping_table_name} 
        WHERE risk_event_id IN ({placeholders}) 
        AND voice_date >= %s
        GROUP BY risk_event_id
    """
    params = tuple(event_ids) + (window_start_str,)
    results = db_manager.execute_query(sql, params)
    
    # 初始化所有事件的增量为0，然后用查询结果填充
    volume_map = {eid: 0 for eid in event_ids}
    if results:
        for row in results:
            volume_map[row['risk_event_id']] = row['count']
    return volume_map



def process_risk_events_batch(
    processed_voices: List[Dict],
    risk_vkb: VectorKnowledgeBaseRiskEventSummary,
    ob_db_manager: OBDatabaseManager,
    current_ts: Optional[int] = None,
) -> Dict[str, Any]:
    """
    处理一批已完成语义聚合的原声数据，批量进行事件的创建、更新和告警。
    """
    if not processed_voices:
        return {"new_events": 0, "updated_events": 0, "alerts_triggered": 0, "mappings": 0, "suppressed_voices": 0}

    total_start_time = time.time()

    # 确定当前逻辑时间
    if current_ts is None:
        current_ts = int(time.time())
    current_time_str = timestamp_to_str(current_ts)

    # Step 1: 数据准备与分组

    # Step 1.1: 将所有 voice 的映射关系写入数据库，用于算法解释性
    unique_voices = _deduplicate_voices(processed_voices)
    mapping_count = persist_voice_to_risk_event_mappings(unique_voices, ob_db_manager, risk_event_voice_mapping_table_name)

    # Step 1.2: 从声量计算流程中过滤掉被抑制的 voice
    voices_for_volume_calc = []
    suppressed_count = 0
    for v in unique_voices:
        decision = 'unknown'
        try:
            # 安全地提取决策结果
            agg_result = v.get('algorithm_detail', [{}])[0].get('semantic_aggregation_result', {})
            decision = agg_result.get('llm_decision_result', {}).get('decision')
        except (IndexError, KeyError, AttributeError):
            pass # 保持 decision 为 'unknown'

        if decision != 'suppress_as_false_positive':
            voices_for_volume_calc.append(v)
        else:
            suppressed_count += 1
            
    if suppressed_count > 0:
        logger.info(f"[EVENT_MANAGEMENT] {suppressed_count} voices were marked as 'suppressed' and will NOT contribute to event volume/alerts.")


    # Step 1.3: 应用记账窗口，并作用于过滤后的列表
    voice_time_cutoff_ts = current_ts - (EVENT_MAX_AGE_HOURS * 3600)
    
    valid_volume_voices = [
        v for v in voices_for_volume_calc # <--- 使用过滤后的列表
        if v.get('source_metadata', {}).get('gmt_create') and 
           str_to_timestamp(v['source_metadata']['gmt_create']) >= voice_time_cutoff_ts
    ]
    
    # 如果过滤后没有有效的 voice，直接返回统计结果
    if not valid_volume_voices:
        logger.info("[EVENT_MANAGEMENT] No non-suppressed voices within the active accounting window. Skipping volume updates and alerts.")
        return {"new_events": 0, "updated_events": 0, "alerts_triggered": 0, "mappings": mapping_count, "suppressed_voices": suppressed_count}

    # 后续所有步骤都自然地在 valid_volume_voices 这个干净的列表上进行

    # Step 1.4: 按 risk_event_id 对有效原声进行分组
    event_voice_groups = defaultdict(list)
    for v in valid_volume_voices:
        # 这里的 extract_final_risk_event_id 仍然会产生 WARNING，这是符合预期的
        risk_event_id = extract_final_risk_event_id(v)
        if risk_event_id:
            event_voice_groups[risk_event_id].append(v)


    # Step 2: 批量获取事件当前状态
    event_ids_in_batch = list(event_voice_groups.keys())
    existing_events_map = {}
    if event_ids_in_batch:
        placeholders = ', '.join(['%s'] * len(event_ids_in_batch))
        select_query = f"SELECT * FROM {risk_event_table_name} WHERE risk_event_id IN ({placeholders})"
        existing_events_results = ob_db_manager.execute_query(select_query, tuple(event_ids_in_batch))
        if existing_events_results:
            existing_events_map = {row['risk_event_id']: row for row in existing_events_results}

    # Step 3: 分离新/存量事件，并计算声量增量 (在内存中)
    new_event_tasks = {}
    update_event_tasks = {}
    
    for event_id, voices in event_voice_groups.items():
        # 此时 voices 列表里已经是过滤后的有效原声
        delta = len(voices)
        if event_id not in existing_events_map:
            new_event_tasks[event_id] = {'voices': voices, 'delta': delta}
        else:
            update_event_tasks[event_id] = {'delta': delta, 'existing_record': existing_events_map[event_id]}

    # Step 4: 批量执行数据库写入/更新
    # 4.1 批量更新存量事件的声量
    update_deltas = {eid: task['delta'] for eid, task in update_event_tasks.items()}
    if update_deltas:
        _batch_update_event_volumes(ob_db_manager, update_deltas, current_time_str)

    # 4.2 批量创建新事件
    new_event_records_to_insert = []
    new_kb_records_to_insert = []
    if new_event_tasks:
        for event_id, task in new_event_tasks.items():
            sample_voice = task['voices'][0]
            title = sample_voice.get('batch_cluster_title') or sample_voice.get('initial_summary_text') or 'Unknown Title'
            initial_ts_str = min(v['source_metadata']['gmt_create'] for v in task['voices'])
            
            new_event_records_to_insert.append({
                'risk_event_id': event_id, 'risk_event_title': title, 'status': 'active',
                'gm_code': sample_voice.get('context_gm_code', 'unknown'),
                'goc_product_line': sample_voice.get('context_goc_product_line', 'unknown'),
                'data_source': DEFAULT_SEARCH_DATA_SOURCE,
                'risk_event_volume': task['delta'], 
                'initial_timestamp': initial_ts_str,
                'last_active_timestamp': initial_ts_str # 新事件的初始活跃时间就是其创建时间
            })
            new_kb_records_to_insert.append({
                'risk_event_id': event_id, 'risk_event_title': title, 'status': 'active',
                'gm_code': sample_voice.get('context_gm_code', 'unknown'),
                'goc_product_line': sample_voice.get('context_goc_product_line', 'unknown'),
                'data_source': DEFAULT_SEARCH_DATA_SOURCE,
                'initial_timestamp': str_to_timestamp(initial_ts_str)
            })
    
    if new_event_records_to_insert:
        ob_db_manager.execute_insert(risk_event_table_name, new_event_records_to_insert)
        risk_vkb.insert_knowledge_base(risk_vkb.load_data(new_kb_records_to_insert))
        logger.info(f"Batch created {len(new_event_records_to_insert)} new risk events.")

    # Step 5: 统一进行告警检查 (数据准备)
    alert_engine = get_alert_engine(ob_db_manager)
    # 将 current_ts 传递给告警引擎，# 在告警检查部分，使用 current_ts
    alert_engine.set_current_time(current_ts)
    alert_window_start_str = timestamp_to_str(current_ts - (ALERTING_WINDOW_HOURS * 3600))
    
    # 5.1 批量获取更新事件的增量声量
    update_event_ids = list(update_event_tasks.keys())
    update_event_inc_volumes = _batch_get_incremental_volumes(ob_db_manager, update_event_ids, alert_window_start_str)

    # 5.2 组合所有需要检查告警的事件
    events_to_check_alert = []
    # 新事件
    for record in new_event_records_to_insert:
        events_to_check_alert.append({
            'event_id': record['risk_event_id'],
            'gm_code': record.get('gm_code'),
            'total_volume': record['risk_event_volume'],
            'incremental_volume': record['risk_event_volume'], # 新事件的增量即其初始声量
            'existing_record': None
        })
    # 更新事件
    for event_id, task in update_event_tasks.items():
        events_to_check_alert.append({
            'event_id': event_id,
            'gm_code': task['existing_record'].get('gm_code'),
            'total_volume': task['existing_record']['risk_event_volume'] + task['delta'],
            'incremental_volume': update_event_inc_volumes.get(event_id, 0),
            'existing_record': task['existing_record']
        })

    # Step 6: 统一执行告警判断与数据库更新
    alerts_to_insert = []
    ts_updates_to_perform = []
    silence_window_seconds = ALERT_SILENCE_WINDOW_MINUTES * 60

    for check_info in events_to_check_alert:
        event_id = check_info['event_id']
        gm_code = check_info.get('gm_code')

        should_alert, alert_context = alert_engine.should_alert_with_dynamic_baseline(
            event_id, gm_code, check_info['total_volume'], check_info['incremental_volume']
        )
        if not should_alert: continue

        can_trigger, alert_type = False, ""
        existing_record = check_info['existing_record']

        if not existing_record: # 是新事件
            can_trigger, alert_type = True, "first_alert"
            earliest_gmt = min(v['source_metadata']['gmt_create'] for v in new_event_tasks[event_id]['voices'])
            ts_updates_to_perform.append({'op': 'first_breach', 'id': event_id, 'time': earliest_gmt})
        else:
            if not existing_record.get('threshold_breach_timestamp'):
                can_trigger, alert_type = True, "first_alert"
                ts_updates_to_perform.append({'op': 'first_breach', 'id': event_id, 'time': current_time_str})
            elif existing_record.get('last_alerted_timestamp'):
                last_alert_ts = int(existing_record['last_alerted_timestamp'].timestamp())
                # if current_ts_int - last_alert_ts > silence_window_seconds:
                if current_ts - last_alert_ts > silence_window_seconds:
                    can_trigger, alert_type = True, "re_alert"
            else:
                can_trigger, alert_type = True, "re_alert"

        if can_trigger:
            alerts_to_insert.append({
                'risk_event_id': event_id, 'alert_timestamp': current_time_str,
                'volume_at_alert': check_info['total_volume'], 'alert_type': alert_type,
                'alert_context': json.dumps(alert_context, ensure_ascii=False)
            })
            ts_updates_to_perform.append({'op': 'last_alerted', 'id': event_id, 'time': current_time_str})

    # 6.2 批量执行告警相关的数据库写入
    if alerts_to_insert:
        ob_db_manager.execute_insert(risk_event_alert_record_table_name, alerts_to_insert)
    if ts_updates_to_perform:
        for upd in ts_updates_to_perform:
            if upd['op'] == 'first_breach':
                ob_db_manager.execute_update(f"UPDATE {risk_event_table_name} SET threshold_breach_timestamp = %s WHERE risk_event_id = %s", (upd['time'], upd['id']))
            elif upd['op'] == 'last_alerted':
                ob_db_manager.execute_update(f"UPDATE {risk_event_table_name} SET last_alerted_timestamp = %s WHERE risk_event_id = %s", (upd['time'], upd['id']))
    
    # Step 7: 最终日志
    total_latency = time.time() - total_start_time
    stats = {
        "new_events": len(new_event_tasks), 
        "updated_events": len(update_event_tasks), 
        "alerts_triggered": len(alerts_to_insert),
        "mappings": mapping_count,
        "suppressed_voices": suppressed_count
    }

    logger.info(f"[EVENT_MANAGEMENT] Summary: new={stats['new_events']}, updated={stats['updated_events']}, alerts={stats['alerts_triggered']}, suppressed={stats['suppressed_voices']}, mappings={stats['mappings']}. Total latency: {total_latency:.2f}s")

    return stats