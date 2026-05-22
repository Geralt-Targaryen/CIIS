# tingis/semantic_aggregator/event_matcher.py
import time
import math
import logging
from collections import defaultdict, namedtuple
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tingis.knowledge_base.ob_db_manager.ob_db_operations import OBDatabaseManager
from tingis.semantic_aggregator.matching.matcher import batch_cluster_title_match_risk_event
from tingis.semantic_aggregator.matching.decider import compare_and_select_risk_event
from tingis.utils.common_utils import timestamp_to_str, to_timestamp, to_str
from tingis.model_services.embedding_utils import batch_generate_embeddings_from_texts
from tingis.global_vars import (
    MILVUS_SEARCH_MAX_WORKERS,
    DECAY_FACTOR_K,
    MIN_DECISION_SCORE_THRESHOLD,
    ENABLE_FALSE_POSITIVE_SUPPRESSION,
    UNCLUSTERED_ID
)

if ENABLE_FALSE_POSITIVE_SUPPRESSION:
    from tingis.global_vars import (
        false_positive_collection_name,
        FALSE_POSITIVE_SEARCH_TOP_K,
        FALSE_POSITIVE_DISTANCE_THRESHOLD
    )

logger = logging.getLogger(__name__)

SearchTaskID = namedtuple('SearchTaskID', ['batch_cluster_id', 'gm_code', 'title_text'])
TaskInfo = namedtuple('TaskInfo', ['task_id', 'search_type'])


def _prepare_and_enrich_search_tasks(
    clustered_voices: List[Dict], 
    embedding_cache: Dict
) -> Tuple[Dict, Dict, Dict]:
    """
    准备搜索任务，并动态生成缺失的 embedding
    """
    tasks = {}
    task_to_voices_map = defaultdict(list)
    titles_to_embed = set()
    
    # 第一次遍历：收集任务和需要 embedding 的标题
    for voice in clustered_voices:
        if not (voice and voice.get('batch_cluster_id') != 'unclustered'):
            continue
        
        batch_cluster_id = voice['batch_cluster_id']
        gm_code = voice.get('context_gm_code', 'unknown')
        title_text = voice.get('batch_cluster_title')

        if not (title_text and gm_code != 'unknown'):
            continue

        task_unique_id = SearchTaskID(batch_cluster_id, gm_code, title_text)
        task_to_voices_map[task_unique_id].append(voice)

        if task_unique_id not in tasks:
            tasks[task_unique_id] = {
                'batch_cluster_title': title_text,
                'search_gm_code': gm_code,
            }
            if title_text not in embedding_cache:
                titles_to_embed.add(title_text)

    # 如果有需要 embedding 的标题，批量生成一次
    if titles_to_embed:
        logger.info(f"Dynamically generating embeddings for {len(titles_to_embed)} new cluster titles...")
        title_list = list(titles_to_embed)
        new_embeddings = batch_generate_embeddings_from_texts(title_list, title_list)
        embedding_cache.update(new_embeddings) # 更新缓存

    # 第二次遍历：为任务注入 embedding
    final_tasks = {}
    for task_id, task_info in tasks.items():
        title = task_info['batch_cluster_title']
        embedding = embedding_cache.get(title)
        if embedding is not None:
            task_info['batch_cluster_title_vec'] = embedding
            final_tasks[task_id] = task_info
        else:
            logger.warning(f"Embedding still not found for title '{title}' after generation attempt. Skipping.")
            task_to_voices_map.pop(task_id, None)
            
    return final_tasks, task_to_voices_map, embedding_cache


def _calculate_decision_scores(
    candidates: List[Dict], 
    current_ts: int
) -> List[Dict]:
    """
    为候选事件计算带有时间衰减的综合决策分。
    """
    if not candidates:
        return []

    scored_candidates = []
    for cand in candidates:
        rerank_score = cand.get('score', 0.0)
        
        # 计算时间差 (天)
        delta_days = 0.0
        last_active_ts = to_timestamp(cand.get('last_active_timestamp'))
        
        if last_active_ts:
            delta_seconds = current_ts - last_active_ts
            delta_days = max(0, delta_seconds) / 86400.0
        else:
            # 如果没有活跃时间戳，给予一个较大的惩罚
            delta_days = 14.0

        # 计算时间衰减因子
        time_decay_factor = math.exp(-DECAY_FACTOR_K * delta_days)

        # 计算最终决策分
        decision_score = rerank_score * time_decay_factor
        
        # 将所有分数附加到候选对象上，供调试和决策
        cand['time_decay_factor'] = round(time_decay_factor, 4)
        cand['decision_score'] = round(decision_score, 4) # 精排得分（score） * 时间衰减因子（time_decay_factor）
        
        cand['last_active_timestamp'] = to_str(cand.get('last_active_timestamp'))
        cand['initial_timestamp'] = to_str(cand.get('initial_timestamp'))

        scored_candidates.append(cand)
        
    # 根据最终的 decision_score 进行降序排序
    return sorted(scored_candidates, key=lambda x: x.get('decision_score', 0.0), reverse=True)


def _run_parallel_searches(
    milvus_client, 
    search_tasks: Dict, 
    search_start_cutoff: str
) -> Tuple[Dict, Dict]:
    """并行执行主事件库和负样本库的搜索。"""
    main_search_results = {}
    fp_search_results = {}
    
    if not search_tasks:
        return main_search_results, fp_search_results

    search_start = time.time()

    with ThreadPoolExecutor(max_workers=MILVUS_SEARCH_MAX_WORKERS) as executor:
        future_to_task_info = {}
        
        for task_id, task_info in search_tasks.items():
            # 提交主事件库搜索任务
            future_main = executor.submit(
                batch_cluster_title_match_risk_event,
                milvus_client=milvus_client,
                batch_cluster_title_vec=task_info['batch_cluster_title_vec'],
                batch_cluster_title=task_info['batch_cluster_title'],
                gm_code=task_info['search_gm_code'],
                data_source=None,
                hard_start_time_limit=search_start_cutoff
            )
            future_to_task_info[future_main] = TaskInfo(task_id, 'main_search')

            # 如果启用了负样本抑制，提交负样本搜索任务
            if ENABLE_FALSE_POSITIVE_SUPPRESSION:
                future_fp = executor.submit(
                    milvus_client.search,
                    collection_name=false_positive_collection_name,
                    data=[task_info['batch_cluster_title_vec']],
                    limit=FALSE_POSITIVE_SEARCH_TOP_K,
                    filter=f"gm_code == '{task_info['search_gm_code']}'",
                    anns_field='risk_event_title_vector',
                    output_fields=["fp_id", "risk_event_title", "invalid_reason", "annotator"]
                )
                future_to_task_info[future_fp] = TaskInfo(task_id, 'fp_search')

        for future in as_completed(future_to_task_info):
            task_id, search_type = future_to_task_info[future]
            try:
                result = future.result()
                if search_type == 'main_search':
                    main_search_results[task_id] = result.get('top_k_risk_events', [])
                elif search_type == 'fp_search':
                    fp_search_results[task_id] = result
            except Exception as e:
                logger.error(f"A {search_type} task failed for {task_id}: {e}", exc_info=True)
                if search_type == 'main_search' and task_id not in main_search_results:
                    main_search_results[task_id] = []
                elif search_type == 'fp_search' and task_id not in fp_search_results:
                    fp_search_results[task_id] = []

    search_latency = time.time() - search_start
    logger.info(f"    ├─ [Vector Search] tasks={len(search_tasks)}, main_results={len(main_search_results)}, fp_results={len(fp_search_results)}, latency_s={search_latency:.2f}")
    
    return main_search_results, fp_search_results


def _fetch_active_timestamps(db_manager: OBDatabaseManager, event_ids: set) -> Dict:
    """根据事件ID批量查询数据库，获取 last_active_timestamp。"""
    if not event_ids:
        return {}
    
    ids_list = list(event_ids)
    placeholders = ', '.join(['%s'] * len(ids_list))
    from tingis.global_vars import risk_event_table_name
    sql = f"SELECT risk_event_id, last_active_timestamp FROM {risk_event_table_name} WHERE risk_event_id IN ({placeholders})"
    
    try:
        db_results = db_manager.execute_query(sql, tuple(ids_list))
        if db_results:
            return {row['risk_event_id']: row['last_active_timestamp'] for row in db_results}
    except Exception as e:
        logger.error(f"Failed to fetch active timestamps: {e}", exc_info=True)

    return {}


def _process_search_results(
    task_to_voices_map: Dict,
    main_search_results: Dict,
    fp_search_results: Dict,
    active_timestamps_map: Dict,
    current_ts: int
) -> Tuple[List[Dict], Dict]:
    '''
    处理搜索结果，进行负样本审核，准备LLM任务。
    '''
    process_start = time.time()

    decision_tasks = []
    llm_decision_map = {}
    suppressed_count = 0
    no_candidates_count = 0
    below_threshold_count = 0

    for task_id, voices in task_to_voices_map.items():
        batch_cluster_id = task_id.batch_cluster_id
        
        # 是否要做 负样本审核
        if ENABLE_FALSE_POSITIVE_SUPPRESSION:
            fp_hits = fp_search_results.get(task_id, [])
            if fp_hits and fp_hits[0]:
                top_fp_hit = fp_hits[0][0]
                fp_score = top_fp_hit.get('distance', 0.0)

                if fp_score is not None and fp_score <= FALSE_POSITIVE_DISTANCE_THRESHOLD:
                    matched_fp_event = top_fp_hit.get('entity', {})
                    reason = (
                        f"Suppressed by False Positive KB. "
                        f"Matched Title: '{matched_fp_event.get('risk_event_title', 'N/A')}' "
                        f"(Score: {fp_score:.4f}). "
                        f"Reason: '{matched_fp_event.get('invalid_reason', 'N/A')}', "
                        f"Annotator: {matched_fp_event.get('annotator', 'N/A')}."
                    )
                    logger.info(f"Cluster '{batch_cluster_id}' suppressed. Reason: {reason}")
                    
                    llm_decision_map[batch_cluster_id] = {
                        'decision': 'suppress_as_false_positive',
                        'matched_risk_event_id': None,
                        'reason': reason,
                        'trace_id': None  # 负样本抑制不涉及 LLM 调用
                    }
                    suppressed_count += 1
                    continue
        
        # 主事件匹配
        raw_candidates_with_ts = []
        for event in main_search_results.get(task_id, []):
            event['last_active_timestamp'] = active_timestamps_map.get(event.get('risk_event_id'))
            raw_candidates_with_ts.append(event)

        final_candidates = _calculate_decision_scores(raw_candidates_with_ts, current_ts)

        for v in voices:
            v['raw_reranked_candidates'] = raw_candidates_with_ts
            v['final_candidates'] = final_candidates

        if not final_candidates:
            llm_decision_map[batch_cluster_id] = {
                'decision': 'create_new', 
                'matched_risk_event_id': None, 
                'reason': "No candidates found.",
                'trace_id': None
            }
            no_candidates_count += 1
            continue
        
        top_candidate = final_candidates[0]
        if top_candidate['decision_score'] < MIN_DECISION_SCORE_THRESHOLD:
            llm_decision_map[batch_cluster_id] = {
                'decision': 'create_new', 
                'matched_risk_event_id': None, 
                'reason': f"Highest score ({top_candidate['decision_score']:.4f}) below threshold.",
                'trace_id': None
            }
            below_threshold_count += 1
            continue

        context_summaries = list(set(v.get('initial_summary_text', '') for v in voices if v.get('initial_summary_text')))[:5]
        if context_summaries:
            decision_tasks.append({
                "batch_cluster_id": batch_cluster_id,
                "batch_cluster_title": task_id.title_text,
                "context_summaries": context_summaries,
                "candidate_events": final_candidates
            })
        else:
            # 如果没有上下文摘要，也直接创建新事件，避免LLM在信息不足时误判
            llm_decision_map[batch_cluster_id] = {
                'decision': 'create_new', 
                'matched_risk_event_id': None, 
                'reason': "No context summaries available for LLM decision.",
                'trace_id': None
            }
    process_latency = time.time() - process_start
    logger.info(
        f"    ├─ [Result Processing] llm_tasks={len(decision_tasks)}, "
        f"suppressed={suppressed_count}, no_candidates={no_candidates_count}, "
        f"below_threshold={below_threshold_count}, latency_s={process_latency:.3f}"
    )

    return decision_tasks, llm_decision_map


def _inject_aggregation_results(clustered_voices, llm_decision_map):
    """
    将匹配和决策结果注入到 voice 对象中
    """
    for voice in clustered_voices:
        if not voice: continue
        
        batch_cluster_id = voice.get('batch_cluster_id', UNCLUSTERED_ID)
        decision = llm_decision_map.get(batch_cluster_id)
        
        # 即使没有决策结果，也构造一个默认的 agg_result
        agg_result = {
            # 最终用于决策的、经过时间衰减加权排序的候选列表
            'final_candidates': voice.get('final_candidates', []),

            # LLM的最终决策结果
            'llm_decision_result': decision or {
                'decision': 'create_new',
                'matched_risk_event_id': None,
                'reason': 'No candidates or LLM decision was made.'
            },
            
            # 原始的、未经时间衰减的rerank结果，用于对比分析
            'raw_reranked_candidates': voice.get('raw_reranked_candidates', [])
        }
        
        if 'algorithm_detail' not in voice or not isinstance(voice['algorithm_detail'], list):
            voice['algorithm_detail'] = []
        voice['algorithm_detail'].append({'semantic_aggregation_result': agg_result})

        # 清理不再需要的临时字段
        if 'final_candidates' in voice:
            del voice['final_candidates']
        if 'raw_reranked_candidates' in voice:
            del voice['raw_reranked_candidates']


def match_and_decide_events(
    milvus_client,
    db_manager: OBDatabaseManager,
    clustered_voices: List[Dict],
    search_start_cutoff: str | None,
    embedding_cache: Optional[Dict] = None,
    current_ts: Optional[int] = None
) -> List[Dict]:
    """
    主协调函数：为预聚类后的事件标题匹配历史事件 & 做出决策
    """
    if not clustered_voices: return []

    stage_start_time = time.time()
    embedding_cache = embedding_cache or {}
    if current_ts is None:
        current_ts = int(time.time())

    # Step 1: 准备搜索任务
    search_tasks, task_to_voices_map, _ = _prepare_and_enrich_search_tasks(clustered_voices, embedding_cache)

    # Step 2: 并行搜索
    main_search_results, fp_search_results = _run_parallel_searches(milvus_client, search_tasks, search_start_cutoff)
    
    # Step 3: 获取活跃事件的时间戳 (DB查询)
    all_retrieved_event_ids = {evt['risk_event_id'] for evts in main_search_results.values() for evt in evts}
    active_timestamps_map = _fetch_active_timestamps(db_manager, all_retrieved_event_ids)

    # Step 4: 处理搜索结果，进行负样本审核，并准备 LLM 任务
    decision_tasks, llm_decision_map = _process_search_results(
        task_to_voices_map, main_search_results, fp_search_results, active_timestamps_map, current_ts
    )
    
    # Step 5: 对需要进一步判断的任务，执行 LLM 决策
    if decision_tasks:
        llm_start = time.time()
        llm_results = compare_and_select_risk_event(decision_tasks)
        llm_latency = time.time() - llm_start
        # logger.info(f"    ├─ [LLM Decision] tasks={len(decision_tasks)}, latency_s={llm_latency:.2f}")
        
        trace_ids = []
        for res in llm_results:
            decision_result = res['decision_result']
            llm_decision_map[res['batch_cluster_id']] = decision_result
            if decision_result.get('trace_id'):
                trace_ids.append(decision_result['trace_id'])
        # 打印汇总日志
        trace_ids_str = ', '.join(trace_ids) if trace_ids else 'N/A'
        # logger.info(f"    ├─ [LLM Decision] trace_ids=[{trace_ids_str}]")

    # Step 6: 将所有决策结果注入回原始的 voice 对象
    _inject_aggregation_results(clustered_voices, llm_decision_map)

    total_latency = time.time() - stage_start_time
    logger.info(f"  -> [Matching & Decision] total_count={len(clustered_voices)}, total_latency_s={total_latency:.2f}")

    return clustered_voices
