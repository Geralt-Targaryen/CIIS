from tqdm import tqdm
import pandas as pd
import time

from tingis.voice_router.voice_router_semantic_search import initialize_search_client, run_semantic_search_pipeline
from tingis.voice_router.voice_router_text_search import load_resources, load_goc_code_keywords, run_text_search_pipeline
from tingis.voice_router.voice_router_hybrid_search import run_hybrid_search_pipeline

if __name__ == '__main__':
    client = initialize_search_client()
    seg, stopwords = load_resources()
    goc_keywords_mapping = load_goc_code_keywords()

    voice_titles = [
    #   '用户表示支付宝余额宝转出10*9元，但无法查看，需要查询另一个账户信息。',
    #   '用户支付宝自动转入余额宝，但无法查看余额资金去向。',
    #   '用户表示未参加阳光保险却被扣款，申请追回扣款',
    #   '用户未开通好医保却被扣费，询问是否可以报销，并询问扣费方式和退款方式。',
    #   '用户遇到了蚂蚁庄园，小鸡跑了',
    #   '果园里葡萄被偷吃了',
    #   '蚂蚁森林里小鸡跑了',
    #   '小鸡找不到了',
    #   '小鸡无法睡觉',
    #   '用户遇到了电费自动扣费的问题，要求取消代扣并联系谁。',
    #   '淘票票买不了票197372392303203',
    #   '支付宝首页打开啦',
    #   '高德打车不了',
    #   '用户淘宝挂了',
        
        '用户遇到了无法转账的问题',

        '用户遇到了碰一碰赏金有效期从12个月变成30天的问题，并且已经听别人说已经有结果了。',
        '用户遇到了碰一碰赏金规则突然改变的问题，参与活动不可靠，不说明有效期，导致用户不知道如何使用。',
        '用户遇到了支付宝碰一碰赏金活动有效期为12个月，但实际显示为30天的问题。',

        "团购券无法使用",
        "今天你快乐吗",
    ]
    
    # batch_results = run_semantic_search_pipeline(client, voice_titles)
    # batch_results = run_text_search_pipeline(seg, stopwords, goc_keywords_mapping, voice_titles)
    batch_results = run_hybrid_search_pipeline(voice_titles, seg, stopwords, goc_keywords_mapping, client)
    print(batch_results)
    