import re
import json

from tingis.preprocessing.filter_rules import *


# 清洗规则
KEYWORD_PII_PATTERN = r'((?:支付宝|对方|个人|实名)?(?:账号|手机号|手机号码|身份证号|身份证号码)(?:\s*(?:是|为))?)\s*([0-9A-Za-z\*\-@\.]+)'
NUMBER_PII_PATTERN = r'(?<!\d)\d{5,}(?!\d)'
AMOUNT_PATTERN = r'\d+(\.\d+)?元'


def clean_voice_title(title):
    """
    清洗标题：脱敏PII和金额
    """
    # PII
    processed_title = re.sub(KEYWORD_PII_PATTERN, r'\1*', title, flags=re.IGNORECASE)
    processed_title = re.sub(NUMBER_PII_PATTERN, '*', processed_title)
    processed_title = re.sub(AMOUNT_PATTERN, '*元', processed_title)

    return processed_title


def filter_queries(voice_list):
    """
    多重规则过滤
    """
    kept_data = []

    for item in voice_list:
        title = item['voice_title']

        # 规则1：长度过滤
        if len(title)<=12:
            continue

        if len(title) >= 80:
            continue

        # 规则2：匹配前缀 + 长度
        if any(title.startswith(prefix) and len(title) <= max_length for prefix, max_length in FILTER_RULES):
            continue

        # 规则3：任一关键词匹配 (OR Logic)
        if any(keyword in title for keyword in OR_KEYWORDS_FLAT_SET):
            continue

        # 规则4：所有关键词匹配 (AND Logic)
        is_and_filtered = any(
            all(keyword in title for keyword in rule_keywords)
            for category_rules in AND_KEYWORD_RULES.values()
            for rule_keywords in category_rules
        )
        if is_and_filtered:
            continue
        
        # 规则5：必须包含规则中的所有关键词，并且文本长度小于等于指定的长度
        if any(
            all(k in title for k in keywords_list) and len(title) <= max_length
            for keywords_list, max_length in KEYWORD_AND_LENGTH_FILTER_RULES
        ):
            continue

        # 规则6：必须包含 include_keywords 中的所有关键词，并且不包含 exclude_keywords 中的任何关键词，过滤
        if any(
            all(ik in title for ik in include_keywords) and not any(ek in title for ek in exclude_keywords)
            for include_keywords, exclude_keywords in INCLUDE_EXCLUDE_KEYWORDS_FILTER_RULES
        ):
            continue

        # 如果所有规则都未触发过滤，保留
        kept_data.append(item)

    return kept_data


def clean_and_filter_voice(raw_data):
    cleaned_data = []
    for item in raw_data:
        new_item = item.copy()
        new_item['voice_title'] = clean_voice_title(item['voice_title'])
        cleaned_data.append(new_item)

    final_data = filter_queries(cleaned_data)

    return final_data


if __name__ == '__main__':
    raw_data_1 = [
        {'id': 1, 'voice_title': '用户申请退款5元'},
        {'id': 2, 'voice_title': '为什么扣了我120元？'},
        {'id': 3, 'voice_title': '手续费是19.5元，太贵了！'},
        {'id': 4, 'voice_title': '用户咨询我的花呗额度'},
    ]
    raw_data_2 = [
        {'id': 1,
         'voice_title': '用户遇到支付宝无法扫码支付的问题,账号为******,支付密码为登录密码,无法解除,需要等待一段时间才能使用。用户希望得到解决方案。'},
        {'id': 5,
         'voice_title': '用户遇到了提现选择银开户行选不出来的问题，账号为*****,农村商业银行不支持。'},
        {'id': 6, 'voice_title': '用户支付宝扣费异常,无法查到这笔费用100元,对方账号为*****。'},
    ]
    raw_data_3 = [
        {'id': 1,
         'voice_title': '用户无法使用支付宝扫售货机购买水，账号是手机号码***,输入身份证后四位后仍无法解决，需要绑定银行卡才能使用。'},
        {'id': 2, 'voice_title': '用户注销了一个******的账户，但想查看账号是否有租赁合同。'},
    ]

    raw_data_4 = [
        {'id': 10, 'voice_title': '用户遇到了通话已转至语音留言，但尝试联系的用户无法接听的问题，需要录制留言。'},
        {'id': 11, 'voice_title': '客服电话怎么一直打不通啊'}]

    # 复合关键词规则过滤
    raw_data_5 = [{'id': 10, 'voice_title': '为何我的支付总是失败，请解决。'},
                  {'id': 11, 'voice_title': '我的花呗突然无法使用了，怎么回事？'},
                  {'id': 12, 'voice_title': '紧急！我的账户好像被盗用了！'},
                  {'id': 3,
                   'voice_title': '用户支付宝付款提示中断风险交易，恢复多次后仍提示验证失败，实名账号是本人在用的152***28406,担心申请审核后仍频繁出现，影响账户使用。'},
                  {'id': 4, 'voice_title': '举报诈骗啦啦啦'},
                  {"id": 5, 'voice_title': '啊啊啊询问。。。。。是否可以。事实上'},
                  {"id":7, 'voice_title': '哈哈协商还款,ll'},
                  {'id':8, 'voice_title': '用户询问的see非法饿的饿。。。查看 明 拉黑。兜底'},

                  {"id": 111, "voice_title": "车主。。。。。。。。。。。。。。。。。。 退款的未解决", },
                  {"id": 111, "voice_title": "车主。。。。。。。。。。。。。。。。。。 蚂蚁森林", },
                  {"id": 112, "voice_title": "用户咨询，，，企业支付宝 如何。。。哈哈。。。。哈哈哈哈。。的。", },
                  {"id": 113, "voice_title": "用户咨询，，，电脑，。企业支付宝哈哈。。。。哈哈哈哈。。的。", },
                  {'id':333,"voice_title": "用户遇到了支付宝退款一直弹窗，刷抖音边刷边弹的问题 扣费 咨询 之前 汇率"}
                  ]

    raw_data = raw_data_1 + raw_data_2 + raw_data_3 + raw_data_4 + raw_data_5

    filtered_data = clean_and_filter_voice(raw_data)

    print(json.dumps(filtered_data, indent=2, ensure_ascii=False))
    
    filter = False
    # filter = True
    if filter:
        import pandas as pd
        file_path_read = 'data/tingis/classifier/algo_res_0926_1009_flatten.xlsx'
        file_path_write = 'data/tingis/classifier/algo_res_0926_1009_flatten_filter.xlsx'
        df = pd.read_excel(file_path_read)
        raw_data_list = df.to_dict('records')
        filtered_data_list = clean_and_filter_voice(raw_data_list)
        filtered_df = pd.DataFrame(filtered_data_list)
        filtered_df = filtered_df[df.columns]
        print(filtered_df.tail())
        filtered_df.to_excel(file_path_write, index=False)

