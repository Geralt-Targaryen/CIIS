# 过滤规则

# 规则1: 基于前缀和长度
FILTER_RULES = [
    ('用户咨询', 12),
    ('用户询问', 12),
    ('用户表示', 12),
    ...
]

KEYWORD_AND_LENGTH_FILTER_RULES = [
    (["用户咨询", "如何"], 16),  # 必须包含 "用户咨询" 和 "如何"，且长度 <= 14
    (["用户询问", "如何"], 16),
    ...
]

INCLUDE_EXCLUDE_KEYWORDS_FILTER_RULES = [
    ({"电脑"}, {"账单"}),
    ({"二维码"}, {"支付宝"}),
    ...
]


# 规则2: 业务域分类 & 关键词过滤
# 规则组1：触发任一关键词即可过滤

or_security_risk = [        
    ["反诈", "诈骗", ...],
    
    ["欺诈", "被骗", ...],

    ['风控', '风险等级', ...],
]

or_digital_payment = [
    ["免密支付", '经营码', ...],
    ["被无故扣款", "用户遇到了退款未到账的问题",...],
]

or_digital_finance = [
    ["用户花呗被冻结，无法", "用户花呗被冻结，无法还款", ...],

    ["用户咨询芝麻信用如何", "用户咨询芝麻信用良好为何", ...],
]

or_general_system_issue = [
    ["不方便","不太方便", ...],

    ['失联', '还债', ...],

    ["家人", "父亲", "母亲", ...],
]

OR_KEYWORD_RULES = {
    "security_risk": or_security_risk,
    "digital_payment": or_digital_payment,
    "digital_finance": or_digital_finance,
    "general_system_issue": or_general_system_issue
}


and_general_system_issue = [
    ["需要联系", "客服", "解绑"], ["需要联系", "客服"],

    ["无法联系", "客服"], ["无法查询", "客服"],

    ...
]

and_security_risk = [
    ["被骗", "追回"], ["被骗", "封控"],
    ["被封", "投诉"], ["被封", "无法"], 
    ...
]


and_digital_payment = [

    ["咨询", "价格"], ["询问","价格"], 
    ["咨询", "分期"], ["询问", "分期"],
    ...
]


and_digital_finance = [
    ["花呗无法使用", "申请"], ["花呗无法使用", "希望"],

    ["未购买", "险", "扣费"], ["未购买", "险", "扣款"],

    ['用户咨询', '基金'], ['用户询问', '基金'],
    
    ...
]


AND_KEYWORD_RULES = {
    "security_risk": and_security_risk,
    "digital_payment": and_digital_payment,
    "digital_finance": and_digital_finance,
    "general_system_issue": and_general_system_issue
}

OR_KEYWORDS_FLAT_SET = {
    keyword
    for category_rules in OR_KEYWORD_RULES.values()
    for rule_list in category_rules
    for keyword in rule_list
}
