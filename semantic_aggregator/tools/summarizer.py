# tingis/semantic_aggregator/tools/summarizer.py

import json
from typing import List, Dict, Any
from string import Template
import re
import concurrent.futures
import logging

from tingis.model_services.llm_client import call_theta_llm
from tingis.global_vars import LLM_CLUSTER_SUMMARIZER_MAX_WORKERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s: %(lineno)d - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


custom_prompt_template_2 = '''
## 角色定位
您是一名经验丰富的支付宝技术风险分析专家，专注于从用户反馈数据中进行**批处理分析和模式识别**。您的任务是从一个包含多个用户反馈的批次（我们称之为“簇”）中，高度凝练并识别出**一个或多个核心风险问题**，并为每个核心问题生成简洁、客观、直接反映问题本质的短文本摘要。

## 任务目标
根据提供的**一个用户反馈文本簇 (cluster_feedbacks)**，生成**一个或多个核心问题摘要**，并**同时列出每个摘要所代表的原始反馈文本索引**。
这些摘要和它们关联的原始反馈将用于**一线应急人员**快速理解核心问题及受影响范围。

**核心挑战：**
输入的簇是通过算法初步聚合的，可能存在噪声，或者混杂了多个相关但不同的问题。你需要像人类专家一样审视整个簇，判断其一致性，并做出正确的“合并”或“拆分”决策。**你的摘要必须能代表其项下绝大多数反馈的核心语义，严禁以偏概全。**

## 核心问题定义与提取原则 (微观层面)
在决定了要为哪些反馈生成摘要后，请遵循以下原则进行提炼：

1.  **聚焦主体 (Crucial)**：
    *   主体是用户反馈的具体产品、功能、服务或系统。
    *   **优先具体明确**：若反馈明确提及具体活动、系统或合作平台（如“花呗”、“饿了么”、“飞猪”），优先以此为主体。
    *   **“支付宝”主体使用例外**：仅当反馈指向支付宝 App 整体问题，或无更具体主体时，才使用“支付宝”。
    *   **主体缺失时不强行设定**：若无明确主体，直接描述问题行为（如可直接输出“无法提现”）。

2.  **专注提取问题本质**：
    *   直接提取核心障碍或疑问。完全忽略用户诉求、情绪、赔偿要求及背景故事。
    *   **去除冗余表述**：删除“原因不明”、“突然”、“用户咨询”、“希望解决”等引导性或情绪化表述。
    *   **保留关键信息与专业术语**：保留涉及产品功能描述的原始专业术语。

3.  **抽象与概括**：
    *   **非关键信息普遍化**：对不影响问题识别的具体时间、金额数值、地点进行抽象概括（例如将具体金额概括为“金额不符”）。

## 针对“簇”的处理原则 (宏观层面 - 核心融合点)
1.  **评估簇内一致性与代表性**：
    *   首先通览全簇。判断它们是否都高度一致地指向同一个核心问题。
    *   **代表性检查（关键）**：当你生成一个摘要时，问自己：“这个摘要能准确代表归类到它下面的**超过 80%** 的原始反馈吗？”如果不能，说明你的归类太宽泛或摘要以偏概全了。

2.  **识别并隔离多个核心问题 (拆分策略)**：
    *   **拒绝以偏概全**：如果簇内混杂了大量的通用描述（如“莫名被扣款”）和极少数的具体场景描述（如“顺风车拼单扣款”），**绝不能**用少数派的具体场景来定义整个簇。必须将它们拆分为不同的摘要。
    *   **语义独立拆分**：如果发现簇内存在语义上明显独立的问题（例如一部分是“无法睡觉”，另一部分是“亲密度未增加”），必须拆分。

3.  **深层主体挖掘 (关键拆分点)**:
    *   在处理看似“通用”的反馈组时（如“扣款原因不明”），你必须再次审视，**找出并拆分出任何包含具体主体的反馈**。
    *   例如，`“用户询问工行扣款...”` 虽然也是扣款疑问，但其主体是明确的“工行”，**必须**从不含任何具体主体的通用疑问中独立出来，形成自己的簇。

4.  **识别并隔离噪声反馈**:
    *   仔细检查簇内是否存在无信息量、无法提炼出具体问题的反馈（例如：“用户表示最近不太有空”、“用户询问前面问的问题是否得到答复。”、“用户表示正在上课”等）。
    *   将这些噪声反馈归为一类，并为它们生成一个统一的摘要，例如 `"无明确问题描述"`。

5.  **摘要生成与关联策略**:
    *   为识别出的**每个**独立核心问题组（包括噪声组）分别生成一个摘要。
    *   **完整性与互斥性**: 确保输入列表中的**每一个**反馈索引都被精确地关联到且仅关联到一个摘要下。

## 输出原则 (严格约束)
*   **格式**：输出必须为 **JSON 格式字符串**，包含一个键 `"summaries"`，其值是 `SummaryItem` 数组。
*   **SummaryItem 结构**：`{ "text": "摘要字符串", "original_indices": [整数索引数组] }`
*   **摘要文本要求**：
    *   **极简**：严格控制在 **15 个汉字以内**。
    *   **无标点**：严禁包含逗号、顿号、句号等任何分句标点。
    *   **客观具体**：禁止编造内容，使用清晰的名词和动词。

## 思考步骤 (Chain-of-Thought)
1.  **通览与初判**：快速浏览所有文本，建立对簇内容多样性的初步印象。
2.  **模式识别与分组**：基于宏观原则，识别出簇内存在哪几种主要的问题模式。将相似反馈的索引暂存到不同的组。
3.  **代表性验证与噪声识别 (Critical Step)**：检查每个组，确认组内是否存在“以偏概全”的现象。同时，**找出并单独分组那些无信息量的噪声反馈**。确保所有索引都被分配完毕。
4.  **摘要提炼**：对每个确定下来的组，应用微观提取原则和输出约束，凝练成最终的短摘要。
5.  **构建 JSON**：组合最终输出。

## 示例分析 (Example Demonstration)

### 示例 1：混杂簇的处理  (展示“拒绝以偏概全”、“深层主体挖掘”与“噪声隔离”)
**输入:**
```json
{ "cluster_feedbacks": [
  "用户咨询订单扣款金额不对怎么办",                       (idx 0 - 通用)
  "用户咨询为何被扣款",                                   (idx 1 - 通用)
  "用户收到扣款*元的消息，但不知道扣款原因。",             (idx 2 - 通用)
  "用户表示顺风车拼单成功但支付宝被扣款，要求退款",         (idx 3 - **具体服务**)
  "用户遇到了多扣款的问题，希望得到解决。",                 (idx 4 - 通用)
  "用户询问工行扣款不知去向，咨询扣款原因。",               (idx 5 - **具体主体，混在通用中**)
  "你好在吗",                                           (idx 6 - **噪声**)
  "用户表示支付宝扣款不知去向，咨询扣款原因。"               (idx 7 - 通用)
]}
```

**【错误示范】 (以偏概全，用少数派定义了多数派):**
```json
{
  "summaries": [
    { "text": "顺风车拼单支付宝多扣款", "original_indices": [0, 1, 2, 3, 4, 5, 6] }
  ]
}
```
*错误原因：idx 3 是特例，不能代表其他通用的扣款疑问，更不能代表噪声。*

**【正确示范】 (精准拆分，各归其位):**
```json
{
  "summaries": [
    {
      "text": "用户对扣款原因存疑",
      "original_indices": [0, 1, 2, 4, 7]
    },
    {
      "text": "顺风车拼单成功被扣款",
      "original_indices": [3]
    },
    {
      "text": "工行扣款原因不明",
      "original_indices": [5]
    },
    {
      "text": "无明确问题描述",
      "original_indices": [6]
    }
  ]
}
```
***核心解读***：此例展示了最复杂的拆分。不仅将明确的“顺风车”和噪声“你好”分离，更关键的是，在剩余的“扣款疑问”中，进一步识别并拆分出了带有具体主体“工行”的反馈，只将**真正无具体主体的反馈**归为一类。这才是高质量的语义分拣。

### 示例 2：高度一致簇的处理
**输入:**
```json
{ "cluster_feedbacks": [
  "用户遇到了小鸡无法睡觉的问题，怀疑系统崩溃。",
  "用户咨询小鸡无法睡觉原因。",
  "用户遇到了支付宝蚂蚁庄园小鸡无法睡觉的问题。",
  "用户咨询蚂蚁庄园是否无法睡觉,用户表示无法更新手机存储权限"
]}
```
**【正确示范】:**
```json
{
  "summaries": [
    { "text": "蚂蚁庄园小鸡无法睡觉", "original_indices": [0, 1, 2, 3] }
  ]
}
```

## 输入数据 (请直接处理以下JSON数据，不带任何额外标记):
$clusters_to_process
'''


def extract_representative_fallback_summary(feedbacks):
    """
    从用户反馈列表中提取一个代表性的 fallback 摘要。
    默认策略：选择簇内最短的voice（先清洗）作为摘要
    """
    cleaned_list = []
    for fb in feedbacks:
        clean_fb = re.sub(r'^用户(?:表示|遇到|咨询|反映|说|称|需要|想|希望|尝试)?', '', fb)
        clean_fb = re.sub(r'[，,。!！？?；;\s]+', '', clean_fb).strip()
        if not clean_fb:
            clean_fb = fb
        cleaned_list.append(clean_fb)

    shortest = min(cleaned_list, key=len)
    return shortest


def _parse_and_validate_summaries(raw_output: str, original_feedbacks: List[str]) -> Dict[str, Any]:
    """
    解析、验证 LLM 返回的摘要结果，并在失败时执行 Fallback 逻辑。
    """
    try:
        parsed = json.loads(raw_output)
        
        if not isinstance(parsed, dict) or "summaries" not in parsed:
            raise ValueError("LLM response is missing 'summaries' key.")

        summaries = parsed["summaries"]
        if not isinstance(summaries, list) or not summaries:
            raise ValueError("'summaries' is not a non-empty list.")

        cleaned_summaries = []
        all_indices = set()
        for item in summaries:
            if not isinstance(item, dict): continue
            
            text = item.get("text", "").strip()
            indices = item.get("original_indices", [])
            
            if not isinstance(text, str) or not text: continue
            if not isinstance(indices, list): continue

            valid_indices = [idx for idx in indices if isinstance(idx, int) and 0 <= idx < len(original_feedbacks)]
            all_indices.update(valid_indices)
            
            if valid_indices:
                cleaned_summaries.append({
                    "text": text,
                    "original_indices": valid_indices
                })
        
        # 验证所有原始索引是否都被覆盖
        if len(all_indices) != len(original_feedbacks):
             logger.warning("LLM summaries did not cover all original indices. Fallback will be triggered.")
             raise ValueError("Incomplete index coverage from LLM.")

        return {"success": 1, "summaries": cleaned_summaries}

    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse or validate LLM summary response: {e}. Executing fallback.")
        fallback_text = extract_representative_fallback_summary(original_feedbacks)
        return {
            "success": 0,
            "summaries": [
                {
                    "text": fallback_text,
                    "original_indices": list(range(len(original_feedbacks)))
                }
            ]
        }


def get_representative_summary(cluster_feedbacks, llm_inference_func):
    """
    批内获取代表性摘要
    """
    if not cluster_feedbacks:
        return {"success": 0, "summaries": [{"text": "Empty input", "original_indices": []}]}

    # 构造 prompt
    json_str = json.dumps(cluster_feedbacks, ensure_ascii=False)
    # prompt = Template(custom_prompt_template_1).safe_substitute(clusters_to_process=json_str)
    prompt = Template(custom_prompt_template_2).safe_substitute(clusters_to_process=json_str)

    try:
        raw_output, trace_id = llm_inference_func(prompt, return_trace_id=True)
        result = _parse_and_validate_summaries(raw_output, cluster_feedbacks)
        result['trace_id'] = trace_id
        return result

    except Exception as e:
        logger.error(f"LLM inference function failed: {e}", exc_info=True)
        fallback_text = extract_representative_fallback_summary(cluster_feedbacks)
        return {
            "success": 0,
            "summaries": [
                {
                    "text": fallback_text,
                    "original_indices": list(range(len(cluster_feedbacks)))
                }
            ],
            "trace_id": None
        }


def process_multiple_feedback_batches_concurrently(
    list_of_cluster_feedbacks: List[List[str]],
    llm_inference_func = call_theta_llm,
    max_workers = None
) -> List[Dict[str, Any]]:
    """
    并发处理多个批次
    """

    if max_workers is None:
        max_workers = LLM_CLUSTER_SUMMARIZER_MAX_WORKERS

    num_batches = len(list_of_cluster_feedbacks)
    results = [None] * num_batches

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {}
        for i in range(num_batches):
            future = executor.submit(
                get_representative_summary,
                list_of_cluster_feedbacks[i],
                llm_inference_func
            )
            future_to_index[future] = i

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                results[index] = {
                    "success": 0,
                    "summaries": [{"text": f"Error during processing: {type(e).__name__}: {e}", "original_indices": []}],
                    "trace_id": None
                }
    
    return results


if __name__ == "__main__":

    voice_titles_1 = [
          "用户遇到了微剧不见了的问题，需要人工处理。",
          "用户遇到了微剧无法观看，显示没有任何动态刷新点不动的问题。",
          "用户无法观看微剧，更新软件后仍无法缓存，怀疑被下架。",
          "用户询问微剧无法播放和视频红包天数是否可以延长。",
          "用户咨询支付宝微剧为何无法观看",
          "用户咨询微剧领红包为何只能看10个就不能看",
          "用户遇到了主页没有微剧的问题，并询问如何查看朋友的微剧。",
          "用户遇到了最新5笔视频红包未入账的问题，金额未变，每个视频评论不到一分钟，用户询问微剧领红包的意义，并询问如何解锁。",
          "今天你快乐吗？"
          "卡券功能挂了吧"
          ]

    voice_titles_2 = [
          "用户无法登录企业支付宝账号,需要查询登录名和绑定信息,但营业执照未在手边,无法查询。",
          "用户遇到了企业名称下注册了两个支付宝账户，但无法登录或注销的问题。",
          "用户遇到了企业账号无法登录和提现的问题，需要找回账号。",
          "用户需要企业支付宝收款码，但无法登录前法人申请的账号，需要更改。",
          "用户忘记企业支付宝账户密码,无法重置,需要退保证金。",
          "用户表示企业支付宝账号无法登录,但显示有企业账户信息。用户尝试修改密码和邮箱,但无法成功。",
          "用户遇到了企业支付宝无法登录网商银行的问题，需要激活或注销账户才能开户。用户无法描述原来的密码，想重置密码需要很多流程，但法人不在家，用户希望得到帮助。",
          "用户在登录商家平台时,有账户和密码,但无法登录。",
          "用户想注销企业支付宝账号，但不知道支付密码，无法登录。"
          ]

    list_of_all_feedbacks = [voice_titles_1, voice_titles_2]
    concurrent_summaries = process_multiple_feedback_batches_concurrently(
        list_of_all_feedbacks,
    )
    print(concurrent_summaries)
    # print(json.dumps(concurrent_summaries, ensure_ascii=False, indent=2))

    concurrent_summaries = [
        {'success': 1, 'summaries': [{'text': '微剧无法观看或消失', 'original_indices': [0, 1, 2, 3, 4, 6]}, {'text': '微剧红包观看次数限制', 'original_indices': [5]}, {'text': '微剧红包未到账', 'original_indices': [7]}, {'text': '无明确问题描述', 'original_indices': [8]}]}, 
        {'success': 1, 'summaries': [{'text': '企业支付宝无法登录', 'original_indices': [0, 1, 2, 3, 4, 5, 6, 8]}, {'text': '商家平台登录失败', 'original_indices': [7]}]}
        ]
