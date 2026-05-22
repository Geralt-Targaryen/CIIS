# tingis/event_management/main_processor.py

import threading
import time
import logging
from typing import List, Dict, Tuple, Optional

from tingis.preprocessing.data_pipeline import route_and_enrich_voices, fetch_ting_tiyan_v3
from tingis.voice_router.voice_router_hybrid_search import initialize_search_resources
from tingis.knowledge_base.ob_db_manager.ob_db_operations import OBDatabaseManager
from tingis.semantic_aggregator.matching.matcher import initialize_kb_components
from tingis.semantic_aggregator.aggregation_pipeline import identify_and_reconcile_risk_events
from tingis.event_management.event_manager import process_risk_events_batch
from tingis.event_management.event_snapshotter import perform_snapshot_if_needed
from tingis.global_vars import TINGIS_DATABASE_CONFIG, TING_DATABASE_CONFIG, MAIN_THREAD_KEEP_ALIVE_MINUTES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s: %(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 将快照时间戳作为全局状态管理
_last_processed_id: int = 0
_last_snapshot_ts: int = 0
_start_time_override: Optional[str] = None
_components = {}

def initialize_components():
    global _components
    _components['milvus_client'], _components['risk_vkb'] = initialize_kb_components()
    _components['seg'], _components['stopwords'], _components['goc_keywords_mapping'], _ = initialize_search_resources()
    _components['ob_db_manager'] = OBDatabaseManager(TINGIS_DATABASE_CONFIG)
    _components['ting_ob_db_manager'] = OBDatabaseManager(TING_DATABASE_CONFIG)


def voice_processing_worker():
    global _last_processed_id, _start_time_override, _last_snapshot_ts

    if not _components:
        initialize_components()

    first_run = True

    while True:
        try:
            # === [ 1. DATA FETCH ] ===
            start_time = time.time()
            if first_run and _start_time_override:
                raw_voices, new_max_id = fetch_ting_tiyan_v3(db_manager=_components['ting_ob_db_manager'], start_time=_start_time_override, is_initial=True)
                first_run = False
            else:
                raw_voices, new_max_id = fetch_ting_tiyan_v3(db_manager=_components['ting_ob_db_manager'], last_processed_id=_last_processed_id)

            if not raw_voices:
                if first_run:
                    logger.info("[Init] No voices found from start time.")
                    first_run = False
                _last_snapshot_ts = perform_snapshot_if_needed(_components['ob_db_manager'], _last_snapshot_ts)
                time.sleep(2) # 在没有数据时短暂休眠，避免空轮询
                continue

            gmt_times = [v['source_metadata']['gmt_create'] for v in raw_voices if v.get('source_metadata', {}).get('gmt_create')]
            time_range = f"{min(gmt_times)} ~ {max(gmt_times)}" if gmt_times else "unknown"
            latency = time.time() - start_time
            logger.info(f"[DATA_FETCH] Time range: {time_range} | count={len(raw_voices)} | latency_s={latency:.2f}")

            # === [ 2. ROUTING & ENRICHMENT ] ===
            enriched_voices = route_and_enrich_voices(raw_voices, _components['seg'], _components['stopwords'], _components['goc_keywords_mapping'], _components['milvus_client'])

            if not enriched_voices:
                _last_processed_id = new_max_id
                logger.info("[ROUTING_ENRICHMENT] status=all_filtered")
                _last_snapshot_ts = perform_snapshot_if_needed(_components['ob_db_manager'], _last_snapshot_ts)
                continue

            # === [ 3. SEMANTIC AGGREGATION ] ===
            identified_voices = identify_and_reconcile_risk_events(
                _components['milvus_client'],
                _components['ob_db_manager'],
                enriched_voices, 
                search_start_cutoff=_start_time_override
            )

            # === [ 4. EVENT MANAGEMENT ] ===
            start_time = time.time()
            stats = process_risk_events_batch(identified_voices, _components['risk_vkb'], _components['ob_db_manager'])
            latency = time.time() - start_time
            logger.info(f"[EVENT_MANAGEMENT] new={stats['new_events']} updated={stats['updated_events']} alerts={stats['alerts_triggered']} mappings={stats['mappings']} latency_s={latency:.2f}")

            _last_processed_id = new_max_id

            # === [ 5. SNAPSHOT (串行执行) ] ===
            _last_snapshot_ts = perform_snapshot_if_needed(_components['ob_db_manager'], _last_snapshot_ts)

        except Exception as e:
            logger.error(f"[WORKER_ERROR] error='{e}'", exc_info=True)
            time.sleep(10)


def main():
    global _start_time_override
    _start_time_override = "2025-12-09 16:00:00"
    logger.info(f"System starting with Start Time: {_start_time_override}")
    
    initialize_components()

    # 只启动一个主数据处理线程
    voice_thread = threading.Thread(target=voice_processing_worker, daemon=True, name="VoiceStreamProcessor")
    voice_thread.start()
    
    logger.info("Event-driven risk processor started with serialized snapshotting.")

    keep_alive_seconds = MAIN_THREAD_KEEP_ALIVE_MINUTES * 60
    
    # 主线程保持存活以允许后台线程运行
    try:
        while voice_thread.is_alive():
            voice_thread.join(timeout=keep_alive_seconds) # 使用 join 来等待
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == '__main__':
    main()
