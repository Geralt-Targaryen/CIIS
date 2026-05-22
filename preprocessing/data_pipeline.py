# tingis/preprocessing/pipeline.py

import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from tingis.global_vars import hotline_online_datasource_table_name
from tingis.utils.common_utils import BEIJING_TZ, str_to_timestamp
from tingis.preprocessing.preprocessor import clean_and_filter_voice
from tingis.voice_router.voice_router_hybrid_search import run_hybrid_search_pipeline
from tingis.preprocessing.summarizer import initial_summary_generation_modelops
from tingis.model_services.embedding_utils import batch_generate_embeddings_from_texts

from tingis.global_vars import DB_FETCH_BATCH_SIZE, VOICE_ROUTER_MAX_DISPATCH_COUNT, GM_CODE_FILTER_LIST, PIPELINE_MAX_WORKERS, PIPELINE_BATCH_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s: %(lineno)d - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 模块一：Query -> Preprocessor -> Router -> Initial Summary -> Embedding
# Embedding 对象是 Initial Summary，用于 聚类

def get_anchor_id_by_start_time(
    db_manager,
    start_time: str,
    table: Optional[str] = None
) -> int:
    """
    在 reset 模式下，用 start_time 找到对应的最小 id，作为增量游标锚点。
    返回：若查不到数据返回 0。
    """
    if table is None:
        table = hotline_online_datasource_table_name

    sql = f"""
        SELECT id
        FROM {table}
        WHERE gmt_create >= %s
        ORDER BY gmt_create ASC, id ASC
        LIMIT 1
    """
    try:
        rows = db_manager.execute_query(sql, (start_time,))
        if rows and rows[0] and rows[0].get('id') is not None:
            return int(rows[0]['id'])
    except Exception as e:
        logger.error(f"Failed to get anchor id by start_time={start_time}: {e}", exc_info=True)

    return 0

    
def fetch_ting_tiyan_v3(
    db_manager,
    last_processed_id: int = 0,
    start_time: Optional[str] = None,
    is_initial: bool = False,
    table: Optional[str] = None,
    limit: int = None, # 分批拉取数量
) -> Tuple[List[Dict[str, Any]], int]:
    """
    拉取用户原声

    id在ob表里默认主键

    - 如果 is_initial=True: 按 gmt_create >= start_time 查询（用于首次启动）
    - 否则: 按 id > last_processed_id 查询（增量）
    """
    if table is None: table = hotline_online_datasource_table_name
    if limit is None: limit = DB_FETCH_BATCH_SIZE
    
    if is_initial and start_time is not None:
    # if start_time is not None:
        # 首次拉取：按时间，按 gmt_create 排序
        sql = f"""
            SELECT id, voice_id, title, gmt_create, add_date, voice_channel, uid, voice_detail_link
            FROM {table}
            WHERE gmt_create >= %s
            ORDER BY gmt_create ASC, id ASC
            LIMIT {limit}
        """
        params = (start_time,)
    else:
        # 增量拉取：按 id
        # logger.info(f"[Incremental Fetch] Last ID: {last_processed_id}, Limit: {limit}")
        sql = f"""
            SELECT id, voice_id, title, gmt_create, add_date, voice_channel, uid, voice_detail_link
            FROM {table}
            WHERE id > %s
            ORDER BY id ASC
            LIMIT {limit}
        """
        params = (last_processed_id,)
        logger.debug(f"[Fetch] Mode: Incremental | Last ID: {last_processed_id}")
    
    try:
        query_results = db_manager.execute_query(sql, params)
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        return [], last_processed_id

    if not query_results:
        return [], last_processed_id

    processed_results = []

    # 用于本批次去重
    seen_voice_ids = set() 
    for i in query_results:
        v_id = i.get('voice_id')
        if v_id in seen_voice_ids:
            continue # 如果在本批次已经出现过，直接跳过
        seen_voice_ids.add(v_id)

        try:
            add_date_str = i['add_date'].strftime("%Y-%m-%d %H:%M:%S")
            gmt_create_str = i['gmt_create'].strftime("%Y-%m-%d %H:%M:%S")
            timestamp = str_to_timestamp(add_date_str)
            
            processed_results.append({
                'voice_id': i['voice_id'],
                'voice_title': i['title'],
                'source_metadata': {
                    'gmt_create': gmt_create_str,
                    'add_date': add_date_str,
                    'voice_channel': i['voice_channel'],
                    'uid': i['uid'],
                    'voice_detail_link': i['voice_detail_link'],
                },
                'id': i['id'],  # 临时保留，用于取 max_id
                '_timestamp': timestamp
            })
        except Exception as e:
            logger.warning(f"Error processing voice item: {i.get('voice_id', 'unknown')}, error: {e}")
            continue

    # 按业务时间排序（可选）
    processed_results.sort(key=lambda x: x['_timestamp'])
    
    # 提取 max_id
    max_id = max(item['id'] for item in processed_results) if processed_results else last_processed_id

    # logger.info(f"[Fetch Result] Retrieved {len(processed_results)} voices, Max ID: {max_id}")
    
    # 清理临时字段
    for item in processed_results:
        del item['id']
        del item['_timestamp']

    # logger.info(f"Fetched {len(processed_results)} voices (max_id={max_id})")
    return processed_results, max_id


def _process_fetched_results(query_results: List[Dict], last_known_id: int) -> Tuple[List[Dict[str, Any]], int]:
    """内部通用函数，用于处理从数据库查询出的原始行。"""
    if not query_results:
        return [], last_known_id

    processed_results = []
    seen_voice_ids = set() 
    for i in query_results:
        v_id = i.get('voice_id')
        if v_id in seen_voice_ids:
            continue
        seen_voice_ids.add(v_id)

        try:
            add_date_str = i['add_date'].strftime("%Y-%m-%d %H:%M:%S")
            gmt_create_str = i['gmt_create'].strftime("%Y-%m-%d %H:%M:%S")
            
            processed_results.append({
                'voice_id': i['voice_id'],
                'voice_title': i['title'],
                'source_metadata': {
                    'gmt_create': gmt_create_str,
                    'add_date': add_date_str,
                    'voice_channel': i['voice_channel'],
                    'uid': i['uid'],
                    'voice_detail_link': i['voice_detail_link'],
                },
                'id': i['id'],  # 临时保留，用于取 max_id
            })
        except Exception as e:
            logger.warning(f"Error processing voice item: {i.get('voice_id', 'unknown')}, error: {e}")
            continue

    # 按 gmt_create 排序以保证时序
    processed_results.sort(key=lambda x: x['source_metadata']['gmt_create'])
    
    max_id = max(item['id'] for item in processed_results) if processed_results else last_known_id
    
    # 清理临时字段
    for item in processed_results:
        del item['id']

    return processed_results, max_id


def _format_final_results(high_risk_items: list):
    final_results = []
    for risk_item in high_risk_items:
        voice_router_results = [
            {
                'gm_code': result['gm_code'],
                'goc_product_line': result['goc_product_line'],
                'voice_router_score': round(result.get('score', 0.0), 2),
                'collection_name': result['collection_name'],
                'document': result['document']
            }
            for result in risk_item.get('voice_router_result', [])
        ]
        
        final_results.append({
            'voice_id': risk_item['voice_id'],
            'voice_title': risk_item['voice_title'],
            'algorithm_detail': [
                {'voice_router_result': voice_router_results}
            ],
            'source_metadata': risk_item.get('source_metadata'),
            # todo: 兼容V1，后面可以去掉metadata
            'metadata': risk_item.get('metadata'),
        })
    return final_results

def route_and_enrich_voices(
    voices_info_raw: List[Dict],
    seg,
    stopwords,
    goc_keywords_mapping,
    client
) -> List[Dict]:
    """
    采用统一调度的大粒度并行流水线，最大化效率。
    """
    if not voices_info_raw:
        return []

    total_start_time = time.time()
    logger.info("[PARALLEL_PIPELINE] Starting...")

    # Step 1: 数据清洗 (串行)
    stage_start_time = time.time()
    voices_to_process = clean_and_filter_voice(voices_info_raw)
    if not voices_to_process:
        logger.info("[Pipeline] All voices filtered out after cleaning. Aborting.")
        return []
    logger.info(f"  -> Sub-Stage 1: [Initial Clean] count={len(voices_to_process)}")

    # Step 2: 统一调度下的并行路由与摘要
    stage_start_time = time.time()
    
    router_results_map = {}
    summary_results_map = {}
    
    with ThreadPoolExecutor(max_workers=PIPELINE_MAX_WORKERS) as executor:
        future_tasks = {}
        
        # 按批次提交任务
        for i in range(0, len(voices_to_process), PIPELINE_BATCH_SIZE):
            batch_voices = voices_to_process[i : i + PIPELINE_BATCH_SIZE]
            batch_titles = [v['voice_title'] for v in batch_voices]
            
            # 为每个批次提交一个路由任务
            future_r = executor.submit(run_hybrid_search_pipeline, batch_titles, seg, stopwords, goc_keywords_mapping, client)
            future_tasks[future_r] = ('router', i) # 标记任务类型和批次起始索引
            
            # 为每个批次提交一个摘要任务
            future_s = executor.submit(initial_summary_generation_modelops, batch_titles)
            future_tasks[future_s] = ('summary', i) # 标记任务类型和批次起始索引

        # 收集所有任务的结果
        for future in as_completed(future_tasks):
            task_type, batch_start_index = future_tasks[future]
            try:
                batch_results = future.result()
                original_batch_voices = voices_to_process[batch_start_index : batch_start_index + len(batch_results)]

                if task_type == 'router':
                    # 将路由结果与 voice_id 关联
                    for original_voice, result in zip(original_batch_voices, batch_results):
                        router_results_map[original_voice['voice_id']] = result.get('voice_router_result', [])
                
                elif task_type == 'summary':
                    # 将摘要结果与 voice_id 关联
                    for original_voice, result in zip(original_batch_voices, batch_results):
                        summary_results_map[original_voice['voice_id']] = result

            except Exception as e:
                logger.error(f"A parallel task of type '{task_type}' for batch starting at {batch_start_index} failed: {e}", exc_info=True)

        logger.info(f"  -> Sub-Stage 2: [Parallel Routing & Summary] latency_s={time.time() - stage_start_time:.2f}")

    # Step 3: 结果合并与过滤 (串行)
    stage_start_time = time.time()
    
    items_to_embed = []
    gm_code_blacklist = GM_CODE_FILTER_LIST or []
    max_dispatch_count = VOICE_ROUTER_MAX_DISPATCH_COUNT or 0

    for voice in voices_to_process:
        voice_id = voice['voice_id']
        
        # 必须同时有路由和摘要结果
        if voice_id in router_results_map and voice_id in summary_results_map:
            router_result_list = router_results_map[voice_id]
            summary_result_dict = summary_results_map[voice_id]

            # 执行路由后置过滤
            is_blacklisted = any(vr.get('gm_code') in gm_code_blacklist for vr in router_result_list)
            if is_blacklisted or len(router_result_list) > max_dispatch_count:
                continue

            # 格式化 router_result
            formatted_router_results = [
                {
                    'gm_code': r['gm_code'],
                    'goc_product_line': r['goc_product_line'],
                    'voice_router_score': round(r.get('score', 0.0), 2),
                    'collection_name': r['collection_name'],
                    'document': r['document']
                } for r in router_result_list
            ]

            # 合并所有信息
            final_item = {
                **voice,
                'initial_summary_text': summary_result_dict.get('initial_summary_text'),
                'initial_summary_id': summary_result_dict.get('initial_summary_id'),
                'algorithm_detail': [{'voice_router_result': formatted_router_results}]
            }
            items_to_embed.append(final_item)

    if not items_to_embed:
        logger.info("[ROUTING_ENRICHMENT] No items passed both routing and summary stages.")
        return []
    logger.info(f"  -> Sub-Stage 3: [Merging Results] count={len(items_to_embed)}")

    # Step 4: 最终 Embedding (并发)
    stage_start_time = time.time()
    
    texts_to_embed = [item['initial_summary_text'] for item in items_to_embed if item.get('initial_summary_text')]
    ids_to_embed = [item['voice_id'] for item in items_to_embed if item.get('initial_summary_text')]
    
    final_enriched_items = []
    if texts_to_embed:
        embeddings_map = batch_generate_embeddings_from_texts(texts_to_embed, ids_to_embed, max_workers=PIPELINE_MAX_WORKERS)
        
        for item in items_to_embed:
            embedding = embeddings_map.get(item['voice_id'])
            if embedding is not None:
                item['initial_summary_embedding'] = embedding
                final_enriched_items.append(item)
            else:
                logger.warning(f"Embedding not found or failed for voice_id {item['voice_id']}. This item will be dropped.")

    logger.info(f"  -> Sub-Stage 4: [Final Embedding] latency_s={time.time() - stage_start_time:.2f}")

    total_latency = time.time() - total_start_time
    logger.info(f"[ROUTING_ENRICHMENT] Final count={len(final_enriched_items)} | total_latency_s={total_latency:.2f}")

    return final_enriched_items


if __name__ == '__main__':
    from tingis.voice_router.voice_router_hybrid_search import initialize_search_resources
    from tingis.utils.common_utils import BEIJING_TZ
    from polygonmilvus import PolygonMilvusClient
    from tingis.global_vars import polygon_token_prod, TING_DATABASE_CONFIG
    from tingis.knowledge_base.ob_db_manager.ob_db_operations import OBDatabaseManager
    milvus_client = PolygonMilvusClient(token=polygon_token_prod)
    ob_db_manager = OBDatabaseManager(TING_DATABASE_CONFIG)

    seg, stopwords, goc_keywords_mapping, _ = initialize_search_resources()

    start_time_str = "2025-10-31 08:30:00"
    start_time_obj = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    start_time_str = start_time_obj.strftime("%Y-%m-%d %H:%M:%S")

    # 1. 查询
    query_start = datetime.now(BEIJING_TZ)
    voices_info_raw, new_max_id = fetch_ting_tiyan_v3(ob_db_manager, start_time = start_time_str, is_initial = True)
    # print(f'查询样例：\n{voices_info_raw[0:1]}')
    query_latency = (datetime.now(BEIJING_TZ) - query_start).total_seconds()
    logger.info(f"[Query] 时间范围: {start_time_str} | 原声数量: {len(voices_info_raw) if voices_info_raw else 0} | 耗时: {query_latency:.2f}s")

    # 2. 过滤 & 分发
    risk_to_initial_summary = route_and_enrich_voices(
        voices_info_raw=voices_info_raw,
        seg=seg,
        stopwords=stopwords,
        goc_keywords_mapping=goc_keywords_mapping,
        client=milvus_client
    )
    # print(risk_to_initial_summary[0:1])
