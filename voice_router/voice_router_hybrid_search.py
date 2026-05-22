# -*- coding: utf-8 -*-

# tingis/voice_router/voice_router_hybrid_search.py

from polygonmilvus import PolygonMilvusClient

from tingis.voice_router.voice_router_semantic_search import run_semantic_search_pipeline
from tingis.voice_router.voice_router_text_search import run_text_search_pipeline, load_resources, load_goc_code_keywords
from tingis.global_vars import polygon_token_prod, DEFAULT_GOC_PRODUCT_LINE, DEFAULT_GM_CODE, VOICE_ROUTER_SEMANTIC_SEARCH_RERANKER_THRESHOLD


def initialize_search_resources():
    # 文本检索初始化
    seg, stopwords = load_resources()
    goc_keywords_mapping = load_goc_code_keywords()
    
    # 向量检索初始化
    client = PolygonMilvusClient(token=polygon_token_prod)
    
    return seg, stopwords, goc_keywords_mapping, client


def run_hybrid_search_pipeline(voice_titles, seg, stopwords, goc_keywords_mapping, client, thres=None):
    if not voice_titles:
        return []
        
    if thres is None:
        thres = VOICE_ROUTER_SEMANTIC_SEARCH_RERANKER_THRESHOLD

    # Step 1: 文本搜索
    all_text_results = run_text_search_pipeline(seg, stopwords, goc_keywords_mapping, voice_titles)

    results_map = {}
    semantic_search_candidates = []
    
    for text_result in all_text_results:
        title = text_result['voice_title']
        # 如果文本搜索有结果，直接存入最终 map
        if text_result.get('voice_router_result'):
            results_map[title] = text_result
        else:
            # 否则，加入待向量搜索的列表
            semantic_search_candidates.append(title)
            # 先用一个占位符，防止 key 不存在
            results_map[title] = None
    
    # Step 2: 只对未命中文本搜索的 candidates 执行向量搜索
    if semantic_search_candidates:
        semantic_results = run_semantic_search_pipeline(client, semantic_search_candidates)
        
        for result in semantic_results:
            title = result['voice_title']
            filtered_vr_result = [
                r for r in result.get('voice_router_result', [])
                if r.get('score', 0) >= thres
            ]
            
            if filtered_vr_result:
                results_map[title] = {'voice_title': title, 'voice_router_result': filtered_vr_result}
            else:
                results_map[title] = {
                    'voice_title': title,
                    'voice_router_result': [{
                        'gm_code': DEFAULT_GM_CODE,
                        'goc_product_line': DEFAULT_GOC_PRODUCT_LINE,
                        'score': -1,
                        'collection_name': 'x',
                        'document': 'x'
                    }]
                }

    final_results = [results_map[title] for title in voice_titles]
    
    return final_results