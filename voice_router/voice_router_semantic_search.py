# -*- coding: utf-8 -*-

# tingis/voice_router/voice_router_semantic_search.py

from tqdm import tqdm
import concurrent.futures
import logging
import json
from typing import List, Dict, Optional, Tuple

from polygonmilvus import PolygonMilvusClient

from tingis.model_services.embedding_utils import get_embeddings
from tingis.model_services.bge_reranker_v2_m3 import rerank_serve
from tingis.global_vars import (
    polygon_token_prod,
    EMBEDDING_MAX_WORKERS,
    MILVUS_SEARCH_MAX_WORKERS,
    RERANKER_MAX_WORKERS,
    VOICE_ROUTER_SEMANTIC_SEARCH_RETRIEVAL_TOP_K,
    VOICE_ROUTER_SEMANTIC_SEARCH_RERANKER_TOP_K,
    kb_config_unified
)

logger = logging.getLogger(__name__)

def initialize_search_client():
    # 0. 创建连接
    client = PolygonMilvusClient(token=polygon_token_prod)
    return client

def semantic_search_atom(client, collection_name, data, retrieval_top_k, anns_field="voice_title_vector"):
    '''
    执行单次 Milvus 向量检索的原子操作。
    '''
    retrieval_res = client.search(
        collection_name=collection_name,
        data=data,
        output_fields=["*"],
        limit=retrieval_top_k,
        anns_field=anns_field
    )
    return retrieval_res

def deduplicate_results(results, key='gm_code'):
    seen = {}
    for item in results:
        k = item.get(key)
        # 增加对 k 为 None 的防御
        if k is None:
            continue
        if k not in seen or item.get('score', 0) > seen[k].get('score', 0):
            seen[k] = item
    return list(seen.values())

def _get_embedding_task(title: str) -> Tuple[str, Optional[List[float]]]:
    """单个embedding任务的封装"""
    try:
        embedding = get_embeddings([title])[0]
        return title, embedding
    except Exception as e:
        logger.warning(f"Embedding failed for title: '{title}'. Error: {e}")
        return title, None

def _retrieval_task(
    client: PolygonMilvusClient, 
    title: str, 
    embedding: List[float],
    collection_names: List[str],
    retrieval_top_k: int
) -> Tuple[str, List[Dict]]:
    """
    单个检索任务的封装，支持对一个或多个 collection 进行召回。
    """
    all_hits = []
    num_paths = len(collection_names)
    if num_paths == 0:
        return title, []
    
    # 均分 k 值
    retrieval_top_k_per_path = retrieval_top_k // num_paths if retrieval_top_k > num_paths else 1
    
    try:
        # 循环遍历所有指定的 collection
        for collection_name in collection_names:
            hits = semantic_search_atom(client, collection_name, [embedding], retrieval_top_k_per_path)[0]
            # 为每个结果打上来源标签
            for hit in hits:
                hit['entity']['collection_name'] = collection_name
            all_hits.extend(hits)
        return title, all_hits
    except Exception as e:
        logger.warning(f"Retrieval failed for title: '{title}' on collections: {collection_names}. Error: {e}")
        return title, []


def _rerank_task(
    title: str, 
    retrieved_items: List[Dict], 
    reranker_top_k: int
) -> Dict:
    """单个重排任务的封装"""
    if not retrieved_items:
        return {'voice_title': title, 'voice_router_result': []}
        
    entity_map = {}
    knowledge_list_for_rerank = []
    for item in retrieved_items:
        text_content = item['entity'].get('voice_title')
        if text_content:
            entity_map[text_content] = item['entity']
            knowledge_list_for_rerank.append(text_content)
    
    if not knowledge_list_for_rerank:
        return {'voice_title': title, 'voice_router_result': []}

    try:
        rerank_result_str = rerank_serve(query=title, knowledge_list=knowledge_list_for_rerank)
        reranked_data = json.loads(rerank_result_str)
        reranked_knowledge = reranked_data.get('knowledge_list', [])

        combined = []
        for item in reranked_knowledge:
            content = item.get('content')
            if content in entity_map:
                original_entity = entity_map[content]
                combined.append({
                    'gm_code': original_entity.get('gm_code'),
                    'goc_product_line': original_entity.get('goc_product_line'),
                    'score': round(item.get('score', 0.0), 4),
                    'collection_name': original_entity.get('collection_name'),
                    'document': content
                })
        
        sorted_res = sorted(combined, key=lambda x: x.get('score', 0), reverse=True)
        dedup_res = deduplicate_results(sorted_res, key='gm_code')
        top_res = dedup_res[:reranker_top_k]
        
        return {'voice_title': title, 'voice_router_result': top_res}

    except Exception as e:
        logger.warning(f"Rerank failed for title '{title}'. Error: {e}. Raw: '{str(rerank_result_str)[:200]}'")
        return {'voice_title': title, 'voice_router_result': []}


def run_semantic_search_pipeline(
    client: PolygonMilvusClient,
    voice_titles: List[str],
    retrieval_top_k: Optional[int] = None,
    reranker_top_k: Optional[int] = None,
    kb_config: Optional[Dict] = None # 通过 kb_config 控制召回策略
) -> List[Dict]:
    """
    为每个处理阶段（Embedding, Retrieval, Rerank）创建一个独立的、专用的线程池
    数据从一个池子处理完后，立即被投入下一个池子。
    """
    if not voice_titles:
        return []

    # 1、初始化参数
    retrieval_top_k = retrieval_top_k or VOICE_ROUTER_SEMANTIC_SEARCH_RETRIEVAL_TOP_K
    reranker_top_k = reranker_top_k or VOICE_ROUTER_SEMANTIC_SEARCH_RERANKER_TOP_K

    if kb_config is None:
        kb_config = kb_config_unified

    collection_names_to_search = kb_config.get('collections', [])
    # logger.info(f"Using kb_config: type='{kb_config.get('type')}', collections={collection_names_to_search}")

    total_items = len(voice_titles)
    final_results = {}

    # 2、为每个阶段创建独立的、专用的线程池
    embedding_executor = concurrent.futures.ThreadPoolExecutor(max_workers=EMBEDDING_MAX_WORKERS, thread_name_prefix="Embedding_")
    retrieval_executor = concurrent.futures.ThreadPoolExecutor(max_workers=MILVUS_SEARCH_MAX_WORKERS, thread_name_prefix="Retrieval_")
    rerank_executor = concurrent.futures.ThreadPoolExecutor(max_workers=RERANKER_MAX_WORKERS, thread_name_prefix="Rerank_")
    
    # 3、流水线处理
    embedding_futures = {embedding_executor.submit(_get_embedding_task, title): title for title in voice_titles}
    retrieval_futures = {}
    rerank_futures = {}

    try:
        # 第一层流水线：当 Embedding 完成时，立即提交 Retrieval 任务
        for future_emb in tqdm(concurrent.futures.as_completed(embedding_futures), total=total_items, desc="Embedding"):
            title, embedding = future_emb.result()
            if embedding is not None:
                # 将配置好的 collection 列表和 k 值传入 _retrieval_task
                future_ret = retrieval_executor.submit(
                    _retrieval_task, client, title, embedding, collection_names_to_search, retrieval_top_k
                )
                retrieval_futures[future_ret] = title
            else:
                final_results[title] = {'voice_title': title, 'voice_router_result': []}

        # 第二层流水线：当 Retrieval 完成时，立即提交 Rerank 任务
        for future_ret in tqdm(concurrent.futures.as_completed(retrieval_futures), total=len(retrieval_futures), desc="Retrieval"):
            title, retrieved_items = future_ret.result()
            future_rerank = rerank_executor.submit(_rerank_task, title, retrieved_items, reranker_top_k)
            rerank_futures[future_rerank] = title

        # 第三层流水线：收集 Rerank 结果
        for future_rerank in tqdm(concurrent.futures.as_completed(rerank_futures), total=len(rerank_futures), desc="Reranking"):
            title = rerank_futures[future_rerank]
            result = future_rerank.result()
            final_results[title] = result

    finally:
        # 确保所有线程池都被关闭
        embedding_executor.shutdown()
        retrieval_executor.shutdown()
        rerank_executor.shutdown()

    # 按原始顺序组装最终结果
    return [final_results.get(title, {'voice_title': title, 'voice_router_result': []}) for title in voice_titles]