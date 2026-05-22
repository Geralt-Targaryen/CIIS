import re
import pkuseg
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from tingis.global_vars import KEYWORDS_FILE, STOPWORDS_FILE, GMCODE_KEYWORDS_FILE, VOICE_ROUTER_TEXT_SEARCH_TOP_K, goc_keyword_collection_name, TEXT_SEARCH_MAX_WORKERS


def load_resources():
    # 加载用户词典
    seg = pkuseg.pkuseg(user_dict=KEYWORDS_FILE)
    
    # 加载停用词
    with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f)
    
    return seg, stopwords


def split_words_pkuseg(seg, stopwords, voice_title):
    # 预处理
    # voice_title = voice_title.upper().replace("支付宝", "")
    voice_title = voice_title.upper()
    voice_title = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5]", "", voice_title)

    # 分词+过滤
    words = [
        word for word in seg.cut(voice_title) 
        if word not in stopwords 
        and len(word) > 1
    ]
    return words[:10]


def load_goc_code_keywords():
    # 返回: dict，{'gm_code":xx,"keywords": xx，用\n分割}
    d = {}
    regex = r'[^,^{^}]+'
    df = pd.read_excel(GMCODE_KEYWORDS_FILE)
    for index, raw in tqdm(df.iterrows(), total=len(df), desc="keywords load"):
        gm_code = str(raw['gm_code'])
        goc_product_line = str(raw['goc_product_line'])
        entity_str = str(raw['entity'])
        action_str = str(raw['action'])
        entity_keywords = {k.strip().upper() for k in re.findall(regex, entity_str) if k.strip()} # 去除首尾空格 & 过滤空字符串
        action_keywords = {k.strip().upper() for k in re.findall(regex, action_str) if k.strip()}
        key = f"{gm_code}_{goc_product_line}"
        d[key] = {'entity': entity_keywords, 'action': action_keywords}
    return d


def text_search(seg, stopwords, goc_keywords_mapping, voice_title, top_k=None):
    """
    基于实体(entity)和行为(action)关键词进行分层文本搜索：
    1. 优先匹配实体词，若命中则直接返回结果。
    2. 若未命中实体词，再匹配行为词。
    """
    if top_k is None:
        top_k = VOICE_ROUTER_TEXT_SEARCH_TOP_K
        
    voice_title_words = split_words_pkuseg(seg, stopwords, voice_title)
    voice_title_word_set = set(voice_title_words)
    
    entity_hits = []
    action_hits = []

    for key, mapped_keywords in goc_keywords_mapping.items():
        # 实体词匹配
        entity_keywords_set = mapped_keywords.get('entity', set())
        entity_matched_words = voice_title_word_set & entity_keywords_set
        if entity_matched_words:
            entity_hits.append({
                'key': key, 
                'match_count': len(entity_matched_words),
                'matched_words': entity_matched_words
            })

        # 行为词匹配
        action_keywords_set = mapped_keywords.get('action', set())
        action_matched_words = voice_title_word_set & action_keywords_set
        if action_matched_words:
            action_hits.append({
                'key': key, 
                'match_count': len(action_matched_words),
                'matched_words': action_matched_words # 保存匹配到的具体词汇
            })

    # 分层业务逻辑：
    if entity_hits:
        sorted_candidates = sorted(
            entity_hits,
            key=lambda x: (-x['match_count'], x['key'])
        )
        final_candidates = sorted_candidates
    else:
        sorted_candidates = sorted(
            action_hits,
            key=lambda x: (-x['match_count'], x['key'])
        )
        final_candidates = sorted_candidates

    result = []

    for item in final_candidates[:top_k]:
        key = item['key']
        gm_code, goc_product_line = key.split('_', 1)
        result.append({
            'gm_code': gm_code, 
            'goc_product_line': goc_product_line, 
            'score': 2,
            'collection_name': goc_keyword_collection_name,
            'document': ', '.join(item['matched_words']) # 将匹配到的关键词作为document
        })

    return result


def run_text_search_pipeline(seg, stopwords, goc_keywords_mapping, voice_titles, max_workers=None):

    if not voice_titles:
        return []
    
    if max_workers is None:
        max_workers = TEXT_SEARCH_MAX_WORKERS


    formatted_results = [None] * len(voice_titles)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(text_search, seg, stopwords, goc_keywords_mapping, voice_title): i
            for i, voice_title in enumerate(voice_titles)
        }

        # for future in tqdm(as_completed(future_to_index), total=len(voice_titles), desc="Text Searching"):
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                formatted_results[index] = {
                    'voice_title': voice_titles[index],
                    'voice_router_result': result
                }
            except Exception as e:
                formatted_results[index] = {
                    'voice_title': voice_titles[index],
                    'voice_router_result': [],
                    'error': str(e)
                }

    return formatted_results