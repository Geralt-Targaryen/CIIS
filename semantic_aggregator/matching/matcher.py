# tingis/semantic_aggregator/matching/matcher.py

import json
import time
import logging
import traceback

from polygonmilvus import PolygonMilvusClient

from tingis.utils.common_utils import str_to_timestamp
from tingis.model_services.bge_reranker_v2_m3 import rerank_serve
from tingis.knowledge_base.index_document.risk_event_summary_kb import VectorKnowledgeBaseRiskEventSummary
from tingis.global_vars import (
    polygon_token_prod, 
    risk_event_summary_collection_name, 
    RISK_EVENT_RETRIEVAL_TOP_K, 
    RISK_EVENT_RERANKER_TOP_K,
    ENABLE_START_TIME_FILTER,
    EVENT_MATCHING_LOOKBACK_DAYS,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s: %(lineno)d - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_kb_components():
    """
    初始化Milvus客户端、OB数据库管理器和风险事件知识库
    """
    milvus_client = PolygonMilvusClient(token=polygon_token_prod)
    risk_vkb = VectorKnowledgeBaseRiskEventSummary(polygon_token_prod, risk_event_summary_collection_name)
    risk_vkb.initialize()
    ensure_collection_exists(milvus_client)
    return milvus_client, risk_vkb


def ensure_collection_exists(client):
    """
    确保风险事件集合存在（冷启动）
    """
    if not client.has_collection(risk_event_summary_collection_name):
        risk_vkb = VectorKnowledgeBaseRiskEventSummary(polygon_token_prod, risk_event_summary_collection_name)
        risk_vkb.initialize()
        risk_vkb.create_collection()
        risk_vkb.create_index()
        client.flush(risk_event_summary_collection_name)


def perform_vector_retrieval(client, embedding_data, gm_code=None, data_source=None, hard_start_time_limit=None):
    """
    向量召回：判断 批内聚类标题 是否与历史风险事件相关。
    基于 EVENT_MATCHING_LOOKBACK_DAYS 的粗粒度时间过滤。
    """
    filter_conditions = ["status == 'active'"]

    # 1. 确定时间计算的基准点 (base_ts)，默认使用当前时间
    base_ts = int(time.time())
    if hard_start_time_limit:
        try:
            base_ts = str_to_timestamp(hard_start_time_limit)
        except (ValueError, TypeError):
            logger.warning(f"Invalid hard_start_time_limit format: '{hard_start_time_limit}'. Falling back to current time.")

    # 2. 计算最终的时间下限
    final_start_ts = 0

    if EVENT_MATCHING_LOOKBACK_DAYS > 0:
        lookback_seconds = EVENT_MATCHING_LOOKBACK_DAYS * 86400
        final_start_ts = base_ts - lookback_seconds
        logger.debug(f"Applying coarse lookback filter of {EVENT_MATCHING_LOOKBACK_DAYS} days.")

    if ENABLE_START_TIME_FILTER and hard_start_time_limit:
        try:
            valid_data_start_ts = str_to_timestamp(hard_start_time_limit)
            final_start_ts = max(final_start_ts, valid_data_start_ts)
            logger.debug(f"Applying hard start time limit: {hard_start_time_limit}")
        except (ValueError, TypeError):
             pass

    # 3. 如果计算出了有效的时间下限，基于事件的创建时间 initial_timestamp 粗糙过滤
    if final_start_ts > 0:
        filter_conditions.append(f"initial_timestamp >= {final_start_ts}")

    # 4. 添加其他业务过滤条件
    if gm_code and gm_code != 'unknown':
        filter_conditions.append(f"gm_code in ['{gm_code}']")

    if data_source:
        filter_conditions.append(f"data_source == '{data_source}'")

    final_filter = " and ".join(filter_conditions)

    # logger.info(f">>> DEBUG: Performing Milvus search with final filter: \"{final_filter}\"")
    
    try:
        search_results = client.search(
            collection_name=risk_event_summary_collection_name,
            data=embedding_data,
            consistent_search=True,
            output_fields=["*"],
            limit=RISK_EVENT_RETRIEVAL_TOP_K,
            anns_field='risk_event_title_vector',
            filter=final_filter
        )
        # logger.info(f">>> DEBUG: Milvus search completed. Found {len(search_results[0]) if search_results else 0} results.")
        return search_results
    except Exception as e:
        logger.error(f">>> DEBUG: Milvus search FAILED with filter '{final_filter}'. Error: {e}", exc_info=True)
        raise


def _process_milvus_search_results(search_results):
    """
    处理 Milvus 搜索结果
    """
    retrieved_entities = []
    if search_results and search_results[0]:
        for hit in search_results[0]:
            try:
                entity = hit['entity']
                # Milvus 距离通常是越小越相似，这里转换为 score，0~2 越小越相似
                entity['score'] = hit.get('distance', 0.0)
                retrieved_entities.append(entity)
            except KeyError:
                continue
    return retrieved_entities


def _rerank_risk_events(batch_cluster_title, retrieved_entities):
    """
    对召回的风险事件进行重排序
    """
    rerank_scores = []
    if retrieved_entities:
        for _ in range(2):
            try:
                rerank_response = rerank_serve(
                    query=batch_cluster_title,
                    knowledge_list=[e['risk_event_title'] for e in retrieved_entities]
                )
                # reranked_knowledge_list = eval(rerank_response)['knowledge_list']
                reranked_data = json.loads(rerank_response)
                reranked_knowledge_list = reranked_data['knowledge_list']
                rerank_scores = [k.get('score', 0.0) for k in sorted(
                    reranked_knowledge_list, key=lambda x: x['index']
                )]
                # print(f'哈哈哈哈哈哈哈哈')
                # print(f"Rerank scores for query '{batch_cluster_title[:30]}...': {rerank_scores}")
                break
            except Exception:
                time.sleep(0.5)
    return rerank_scores



def batch_cluster_title_match_risk_event(
    milvus_client, 
    batch_cluster_title_vec, 
    batch_cluster_title, 
    gm_code, 
    data_source, 
    risk_event_reranker_top_k=None,
    hard_start_time_limit=None
):
    """
    批内聚类标题搜索历史风险事件并重排
    """
    if risk_event_reranker_top_k is None:
        risk_event_reranker_top_k = RISK_EVENT_RERANKER_TOP_K

    # Step 1: Vector Retrieval
    search_results = perform_vector_retrieval(
        milvus_client,
        [batch_cluster_title_vec],
        gm_code=gm_code,
        data_source=data_source,
        hard_start_time_limit=hard_start_time_limit
    )
    # print(f'搜索结果：{search_results}')
    # Step 2: Process & Rerank
    retrieved_entities = _process_milvus_search_results(search_results)
    rerank_scores = _rerank_risk_events(batch_cluster_title, retrieved_entities)

    # Step 3: Combine and Rank
    combined_results = []
    for i, entity in enumerate(retrieved_entities):
        entity['score'] = round(rerank_scores[i], 4)
        combined_results.append({
            "risk_event_id": entity.get('risk_event_id'),
            "risk_event_title": entity.get('risk_event_title'),
            "gm_code": entity.get('gm_code'),
            "goc_product_line": entity.get('goc_product_line'),
            "initial_timestamp": entity.get('initial_timestamp'),
            "score": entity['score']
        })
        
    # 按重排后的分数排序
    top_k_risk_events = sorted(combined_results, key=lambda x: x['score'], reverse=True)[:risk_event_reranker_top_k]

    return {"top_k_risk_events": top_k_risk_events}


if __name__ == '__main__':
    from tingis.model_services.bge_m3 import get_embeddings
    from tingis.voice_router.voice_router_hybrid_search import initialize_search_resources

    milvus_client, risk_vkb = initialize_kb_components()
    seg, stopwords, goc_keywords_mapping, _ = initialize_search_resources()

    batch_cluster_title = '花呗支付失败' # 批内聚类标题
    batch_cluster_title_vec = get_embeddings([batch_cluster_title])[0]
    gm_code = 'GC00002'
    initial_timestamp = '2025-06-02 10:00:01'
    data_source = 'hot_online'

    search_results = perform_vector_retrieval(milvus_client, batch_cluster_title_vec, gm_code, data_source, initial_timestamp)
    print(f'召回结果：{search_results}')
    
    # matched = batch_cluster_title_match_risk_event(milvus_client, batch_cluster_title_vec, batch_cluster_title, initial_timestamp, gm_code, data_source)
    # print(f'匹配结果：{matched}')
    # matched = {'top_k_risk_events': [{'risk_event_id': 'abc234', 'risk_event_title': '花呗无法支付', 'gm_code': 'GC00002', 'goc_product_line': '蚂蚁集团/数字金融/花呗', 'initial_timestamp': 1748587501, 'score': 0.994}]}


