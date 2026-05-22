# tingis/semantic_aggregator/voice_grouper.py

import copy
import logging
import time
from collections import defaultdict
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tingis.semantic_aggregator.tools.lsh_union_find import semantic_clustering
from tingis.semantic_aggregator.tools.summarizer import process_multiple_feedback_batches_concurrently
from tingis.utils.common_utils import generate_algo_goc_id
from tingis.global_vars import PARTITION_PROCESSING_MAX_WORKERS, UNCLUSTERED_ID


logger = logging.getLogger(__name__)


def _extract_primary_gm_info(voice: Dict) -> Tuple[str | None, str | None]:
    """从 voice 中提取主要的 GM Code 和 GOC Product Line。"""
    try:
        router_results = voice.get('algorithm_detail', [{}])[0].get('voice_router_result', [])
        if router_results:
            primary_result = router_results[0]
            gm_code = primary_result.get('gm_code')
            goc = primary_result.get('goc_product_line')
            if gm_code and gm_code not in ['unknown', None]:
                return gm_code, goc
    except (IndexError, KeyError):
        pass
    return None, None

def _process_partition(partition_voices: List[Dict], partition_original_indices: List[int]) -> Tuple[List[Dict], Dict]:
    """
    对单个GM分区进行批内聚类和摘要

    Returns:
        Tuple[List[Dict], Dict]: (final_sub_clusters, stats)
        - final_sub_clusters: 聚类结果列表
        - stats: 统计信息 {
            'partition_size': int,
            'batches': int,
            'latency_s': float,
            'trace_ids': List[str],
            'valid': bool
        }
    """
    empty_stats = {'partition_size': 0, 'batches': 0, 'latency_s': 0.0, 'trace_ids': [], 'valid': False}

    if not partition_voices:
        return [], empty_stats

    # 1. 直接使用传入的Embedding，过滤掉获取失败的 voice
    valid_voices = []
    valid_embeddings = []
    valid_original_indices = []

    for i, voice in enumerate(partition_voices):
        embedding = voice.get('initial_summary_embedding')
        if embedding is not None:
            valid_voices.append(voice)
            valid_embeddings.append(embedding)
            valid_original_indices.append(partition_original_indices[i])

    if not valid_embeddings:
        return [], empty_stats

    # 2. 粗粒度语义聚类 (Stage 1)
    clustering_start = time.time()
    local_clusters = semantic_clustering(valid_embeddings)
    clustering_latency = time.time() - clustering_start

    if not local_clusters:
        return [], empty_stats

    # 3. 准备摘要输入
    coarse_clusters_feedbacks = []
    coarse_clusters_indices_map = []
    for coarse_cluster_id in sorted(local_clusters.keys()):
        local_indices = local_clusters[coarse_cluster_id]
        coarse_clusters_feedbacks.append([partition_voices[idx]['voice_title'] for idx in local_indices])
        coarse_clusters_indices_map.append({
            local_idx: partition_original_indices[original_local_idx]
            for local_idx, original_local_idx in enumerate(local_indices)
        })

    # 4. LLM摘要与精细聚类 (Stage 2)
    summarization_start = time.time()
    concurrent_summaries = process_multiple_feedback_batches_concurrently(coarse_clusters_feedbacks)
    summarization_latency = time.time() - summarization_start

    # 收集 trace_ids（只记录前3个和总数，避免日志过长）
    trace_ids = [s.get('trace_id') for s in concurrent_summaries if s.get('trace_id')]

    stats = {
        'partition_size': len(valid_voices),
        'batches': len(coarse_clusters_feedbacks),
        'latency_s': summarization_latency,
        'trace_ids': trace_ids,
        'valid': True
    }

    # 5. 为每个最终的子簇生成ID并构建结果
    final_sub_clusters = []
    for i, summary_result in enumerate(concurrent_summaries):
        local_to_global_idx_map = coarse_clusters_indices_map[i]
        for summary_item in summary_result.get('summaries', []):
            sub_cluster_title = summary_item.get('text', 'Unknown Summary')
            sub_cluster_local_indices = summary_item.get('original_indices', [])
            sub_cluster_global_indices = [
                local_to_global_idx_map[local_idx]
                for local_idx in sub_cluster_local_indices if local_idx in local_to_global_idx_map
            ]

            if not sub_cluster_global_indices: continue

            final_sub_clusters.append({
                'batch_cluster_id': generate_algo_goc_id(),
                'batch_cluster_title': sub_cluster_title,
                'global_indices': sub_cluster_global_indices,
                'batch_cluster_success': summary_result.get('success', 0) == 1,
            })
    return final_sub_clusters, stats

def cluster_and_summarize_voices(enriched_voices: List[Dict]) -> List[Dict]:
    """
    1、将voices按GM Code分区
    2、并发地聚类和摘要
    3、并发处理大分区，串行处理小分区，这个先不考虑。
    """
    if not enriched_voices:
        return []

    # --- Step 1: 数据按GM Code分区 ---
    partition_start = time.time()
    gm_partitions = defaultdict(lambda: {'voices': [], 'indices': [], 'goc': 'unknown'})
    for i, voice in enumerate(enriched_voices):
        if not (voice and voice.get('voice_id')): continue
        gm_code, goc = _extract_primary_gm_info(voice)
        if gm_code:
            gm_partitions[gm_code]['voices'].append(voice)
            gm_partitions[gm_code]['indices'].append(i)
            if goc and goc != 'unknown':
                gm_partitions[gm_code]['goc'] = goc
    partition_latency = time.time() - partition_start
    logger.info(f"    ├─ [Partitioning] gm_codes={len(gm_partitions)}, latency_s={partition_latency:.3f}")
    
    # --- Step 2: 并行处理分区（包含LSH聚类和LLM摘要） ---
    parallel_start = time.time()
    processed_indices = set()
    all_stats = []  # 收集所有分区的统计信息

    with ThreadPoolExecutor(max_workers=PARTITION_PROCESSING_MAX_WORKERS) as executor:
        future_to_gm = {
            executor.submit(_process_partition, p_data['voices'], p_data['indices']): gm_code
            for gm_code, p_data in gm_partitions.items()
        }

        for future in as_completed(future_to_gm):
            gm_code = future_to_gm[future]
            try:
                final_sub_clusters, stats = future.result()
                if stats.get('valid'):
                    all_stats.append(stats)
                p_goc = gm_partitions[gm_code]['goc']
                for sub_cluster in final_sub_clusters:
                    for global_idx in sub_cluster['global_indices']:
                        if global_idx < len(enriched_voices) and enriched_voices[global_idx]:
                            enriched_voices[global_idx].update({
                                'batch_cluster_id': sub_cluster['batch_cluster_id'],
                                'batch_cluster_title': sub_cluster['batch_cluster_title'],
                                'batch_cluster_success': sub_cluster['batch_cluster_success'],
                                'context_gm_code': gm_code,
                                'context_goc_product_line': p_goc,
                            })
                            processed_indices.add(global_idx)
            except Exception as e:
                logger.error(f"Error processing partition for GM code {gm_code}: {e}", exc_info=True)

    # 打印 LLM Summarization 日志：第一条明细 + 最后一条明细 + 汇总
    if all_stats:
        # 按 latency 排序
        sorted_stats = sorted(all_stats, key=lambda s: s['latency_s'])

        # 第一条明细
        first = sorted_stats[0]
        trace_ids_str = ', '.join(first['trace_ids'][:3])
        if len(first['trace_ids']) > 3:
            trace_ids_str += f", ... ({len(first['trace_ids'])} total)"
        logger.info(
            f"    ├─ [LLM Summarization] partition_size={first['partition_size']}, "
            f"batches={first['batches']}, latency_s={first['latency_s']:.2f}, trace_ids=[{trace_ids_str}]"
        )

        # 如果有多条，打印最后一条明细
        if len(sorted_stats) > 1:
            last = sorted_stats[-1]
            trace_ids_str = ', '.join(last['trace_ids'][:3])
            if len(last['trace_ids']) > 3:
                trace_ids_str += f", ... ({len(last['trace_ids'])} total)"
            logger.info(
                f"    ├─ [LLM Summarization] partition_size={last['partition_size']}, "
                f"batches={last['batches']}, latency_s={last['latency_s']:.2f}, trace_ids=[{trace_ids_str}]"
            )

        # 汇总
        total_partitions = len(all_stats)
        total_voices = sum(s['partition_size'] for s in all_stats)
        total_batches = sum(s['batches'] for s in all_stats)
        logger.info(
            f"    ├─ [LLM Summarization] Summary: partitions={total_partitions}, voices={total_voices}, batches={total_batches}"
        )


    # --- Step 3: 处理未聚类的voices ---
    for i, voice in enumerate(enriched_voices):
        if i not in processed_indices and voice:
            gm, goc = _extract_primary_gm_info(voice)
            voice.update({
                'batch_cluster_id': UNCLUSTERED_ID,
                'batch_cluster_title': None,
                'batch_cluster_success': False,
                'context_gm_code': gm or 'unknown',
                'context_goc_product_line': goc or 'unknown',
            })

    parallel_latency = time.time() - parallel_start
    logger.info(f"    ├─ [Clustering & Summarization] processed_voices={len(processed_indices)}, latency_s={parallel_latency:.2f}")

    return enriched_voices

