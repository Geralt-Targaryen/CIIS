# tingis/semantic_aggregator/aggregation_pipeline.py
import json
import logging
import copy
from typing import List, Dict, Optional
import time

from tingis.knowledge_base.ob_db_manager.ob_db_operations import OBDatabaseManager
from tingis.semantic_aggregator.voice_grouper import cluster_and_summarize_voices
from tingis.semantic_aggregator.event_matcher import match_and_decide_events
from tingis.utils.common_utils import generate_unclustered_goc_id, remove_key_from_list_of_dicts
from tingis.global_vars import UNCLUSTERED_ID

logger = logging.getLogger(__name__)

def _handle_unclustered_voices(final_voices: List[Dict]) -> List[Dict]:
    """
    对批内未聚类的voice去重和ID分配

    unclustered原因：
    没有提取到 GM Code。
    在分区内，因为 embedding 失败被跳过。
    本身语义非常独特，在 LSH 聚类时无法与任何其他 voice 形成簇。
    它们所在的粗粒度簇在LLM摘要阶段被判定为噪声，没有被包含在任何一个最终的子簇里。
    
    """
    unclustered_title_cache = {}
    for voice in final_voices:
        if not voice or voice.get('batch_cluster_id') != UNCLUSTERED_ID:
            continue

        title_candidate = voice.get('initial_summary_text') or voice.get('voice_title', '')
        title_clean = str(title_candidate).strip().strip('。,，.!')

        if title_clean in unclustered_title_cache:
            voice['batch_cluster_id'] = unclustered_title_cache[title_clean]
        else:
            new_event_id = generate_unclustered_goc_id()
            voice['batch_cluster_id'] = new_event_id
            unclustered_title_cache[title_clean] = new_event_id
        
        voice['batch_cluster_title'] = title_clean
        voice['batch_cluster_success'] = False
    return final_voices

def identify_and_reconcile_risk_events(
    milvus_client,
    db_manager: OBDatabaseManager,
    enriched_voices: List[Dict],
    search_start_cutoff: str | None,
    current_ts: Optional[int] = None
) -> List[Dict]:
    """
    语义聚合与事件决策的主流程
    """
    if not enriched_voices:
        return []

    total_start_time = time.time()
    voices_to_process = [v.copy() for v in enriched_voices if v]


    # Step 1: 批内聚类（先按GM_CODE分组）与批内摘要
    clustered_voices = cluster_and_summarize_voices(voices_to_process)

    # Step 2: 历史事件匹配与决策 (含向量检索、LLM) 
    decided_voices = match_and_decide_events(
        milvus_client,
        db_manager,
        clustered_voices,
        search_start_cutoff=search_start_cutoff,
        embedding_cache={}, # 传入一个空的缓存，让它自己填充
        current_ts=current_ts
    )

    # Step 3: Finalization
    final_voices = _handle_unclustered_voices(decided_voices)
    
    # 清理临时字段
    final_voices_clean = remove_key_from_list_of_dicts(final_voices, 'initial_summary_embedding')
    final_voices_clean = remove_key_from_list_of_dicts(final_voices_clean, 'top_k_risk_events')

    # print(f'决策结果：{final_voices_clean[0:5]}')
    total_voices = len(final_voices_clean)
    unclustered_count = sum(1 for v in final_voices_clean if v and v.get('batch_cluster_id', '').startswith('Unclustered'))

    total_latency = time.time() - total_start_time
    logger.info(f"[SEMANTIC_AGGREGATION]. Total: {total_voices}, Unclustered: {unclustered_count}. | total_latency_s={total_latency:.2f}")
    # 本批次待处理的 voice 总数，没有被成功分配到任何一个算法生成的簇的voice条数。

    return final_voices_clean

if __name__ == '__main__':

    from tingis.preprocessing.data_pipeline import route_and_enrich_voices
    from tingis.voice_router.voice_router_hybrid_search import initialize_search_resources
    from tingis.semantic_aggregator.matching.matcher import initialize_kb_components
    from tingis.global_vars import TINGIS_DATABASE_CONFIG
    from tingis.semantic_aggregator.test_data import voices_info_raw

    milvus_client, risk_vkb = initialize_kb_components()
    seg, stopwords, goc_keywords_mapping, _ = initialize_search_resources()
    ob_db_manager = OBDatabaseManager(TINGIS_DATABASE_CONFIG)

    # voices_info_raw = voices_info_raw_1
    print(f'原声量: {len(voices_info_raw)}')

    # 数据预处理与分发
    enriched_voices = route_and_enrich_voices(
        voices_info_raw=voices_info_raw,
        seg=seg,
        stopwords=stopwords,
        goc_keywords_mapping=goc_keywords_mapping,
        client=milvus_client
        )
    # print(f'预处理与分发: {len(enriched_voices)}')

    # print(f'预处理与分发样例：{enriched_voices[2:4]}')
    
    search_start_cutoff = "2025-11-20 16:35:00" 
    identified = identify_and_reconcile_risk_events(
        milvus_client,
        ob_db_manager,
        enriched_voices,
        search_start_cutoff
        )
    
    print(identified)
    print(f'聚类后样本量：{len(identified)}')
    # 本来 20条，过滤：18条，预处理与分发：15条，最后剩下 15条
