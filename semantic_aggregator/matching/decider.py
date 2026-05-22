# tingis/semantic_aggregator/matching/decider.py

import math
import json
import logging
import traceback
import time
import re
from typing import List, Dict, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor,  as_completed

from tingis.model_services.llm_client import call_theta_llm
from tingis.global_vars import MIN_DECISION_SCORE_THRESHOLD, LLM_DECIDER_MAX_WORKERS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


multi_candidate_prompt_template = '''
## 角色定位
你是一名顶级的支付宝技术风险事件分析与仲裁专家。你的核心任务是精准地判断一个新出现的用户反馈簇是否可以归并到一个已知的历史风险事件中。

## 核心任务
根据“当前风险簇”的信息和一份“历史候选风险事件列表”，做出决策：
1.  如果“当前风险簇”与“历史候选风险事件列表”中的某一个事件高度匹配，则选择**合并（merge）**到该事件。
2.  如果没有任何一个候选事件是好的匹配，则决策为**新建（create_new）**一个独立的风险事件。

## 输入信息
你将收到一个 JSON 对象，包含：
1.  `batch_cluster_title`: 当前风险簇的高度概括摘要。
2.  `context_summaries`: 当前风险簇的原始反馈摘要样本（1-5条），这是理解 `batch_cluster_title` 内涵的关键上下文。
3.  `candidate_events`: 一个历史风险事件列表，每个事件包含 `risk_event_id` 和 `risk_event_title`。

## 决策原则（严格遵守）
1.  **主体一致性优先 (Highest Priority)**:
    *   **必须**首先检查“当前风险簇”（通过其样本 `context_summaries` 判断）与候选事件的主体是否一致。主体指具体的产品、功能、第三方平台（如“花呗”、“饿了么”、“酷狗音乐”、“ETC”）。
    *   如果主体明确且不同（例如，当前是“抖音”，候选是“小红书”），则**绝对不能合并**，即使问题现象相似。
    *   如果当前簇主体明确，而候选事件是通用描述（如“用户对扣款原因存疑”），要非常谨慎。只有当问题本质高度一致且没有更具体的候选时，才可考虑合并。优先选择与当前簇主体相同的候选。

2.  **问题本质匹配**:
    *   在主体一致的前提下，判断问题现象、用户遇到的核心障碍是否相同。例如，“无法支付”和“支付超时”可能指向同一根因，可以合并。但“无法支付”和“支付后未到账”是两个不同环节的问题，不能合并。

3.  **警惕通用摘要陷阱 (Anti-Generic Rule)**:
    *   如果当前簇有**明确的主体**（如“酷狗音乐”），而候选事件是一个非常**笼统、无具体主体**的描述（如“用户对扣款原因存疑”），请**极其谨慎**。只有在问题本质和范畴被通用描述完全覆盖，且无更具体的其他候选时，才能考虑合并。优先选择与当前簇主体相同的候选。

4.  **决策目标是精准，而非减少事件数量**:
    *   你的目标是**高质量的语义分类**，避免创建模糊不清的“万能事件”，也避免将不同根因的问题错误合并。
    *   **当匹配度存疑时，选择新建是更安全的选择。** 但如果存在高置信度的匹配，请果断合并，以促进事件收敛。

## 输出格式（必须严格遵守）
你必须返回一个**不含任何额外解释**的 JSON 字符串。该 JSON 对象必须包含以下字段：
*   `decision`: 你的最终决策，值为 `"merge"` 或 `"create_new"`。
*   `matched_risk_event_id`: 如果 `decision` 是 `"merge"`，此字段为匹配上的候选事件的 `risk_event_id` 字符串；如果是 `"create_new"`，此字段为 `null`。
*   `reason`: 一句简短的决策理由，用于人类审计。

## 思考步骤 (Chain-of-Thought)
1.  仔细阅读 `batch_cluster_title` 和 `context_summaries`，准确理解当前风险簇的核心问题和**关键主体**。
2.  逐一检查 `candidate_events` 列表中的每个候选事件。
3.  对于每个候选，与当前簇进行对比，问自己：
    *   主体是否一致？（“抖音” vs “小红书”？ -> 不一致，跳过）
    *   问题本质是否相同？（“支付失败” vs “支付后没优惠”？ -> 不相同，跳过）
4.  如果在所有候选者中找到一个**高度匹配**的，记录下它的 `risk_event_id`，决策为 `merge`。
5.  如果在所有候选者中都找不到合适的匹配，或者所有匹配都只是勉强相关，决策为 `create_new`。
6.  根据决策构建最终的 JSON 输出。

## 示例

### 示例 1: 精准匹配
**输入:**
```json
{
  "batch_cluster_title": "饿了么会员自动续费",
  "context_summaries": ["用户咨询饿了么会员为何自动扣费", "用户表示未同意饿了么自动续费"],
  "candidate_events": [
    {"risk_event_id": "evt_abc", "risk_event_title": "饿了么会员自动续费问题"},
    {"risk_event_id": "evt_def", "risk_event_title": "用户对扣款原因存疑"},
    {"risk_event_id": "evt_ghi", "risk_event_title": "美团会员无法开通"}
  ]
}
```
**输出:**
```json
{
  "decision": "merge",
  "matched_risk_event_id": "evt_abc",
  "reason": "当前簇与候选事件evt_abc主体(饿了么)和问题(会员自动续费)均高度匹配。"
}
```

### 示例 2: 无匹配，选择新建
**输入:**
```json
{
  "batch_cluster_title": "抖音支付扣款异常",
  "context_summaries": ["用户在抖音买东西被多扣钱", "抖音直播间付款失败"],
  "candidate_events": [
    {"risk_event_id": "evt_jkl", "risk_event_title": "小红书平台扣款问题"},
    {"risk_event_id": "evt_mno", "risk_event_title": "用户对扣款原因存疑"}
  ]
}
```
**输出:**
```json
{
  "decision": "create_new",
  "matched_risk_event_id": null,
  "reason": "当前簇主体为'抖音'，与所有候选事件主体均不匹配。"
}
```

## 请处理以下输入数据:
$task_input
'''


def parse_llm_json_output(raw_output: str, task_input: Dict) -> Dict:
    """
    1、健壮地解析LLM返回的JSON
    2、如果LLM决策失败：
        （1）如果召回的Top 1历史事件的reranker分数非常高（eg. > 0.95），那么认为是同一事件，merge
        （2）否则，降级为 create_new
    """
    try:
        data = json.loads(raw_output)
        if isinstance(data, dict) and 'decision' in data:
            if data['decision'] == 'merge' and 'matched_risk_event_id' in data:
                return data
            if data['decision'] == 'create_new':
                data['matched_risk_event_id'] = None
                return data
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning(
            f"!! FALLBACK !! reason=parse_failed batch_cluster_id={task_input.get('batch_cluster_id')}"
        )

    # --- Fallback Logic ---
    # 检查是否有高分候选
    candidate_events = task_input.get("candidate_events", [])
    if candidate_events:
        top_candidate = candidate_events[0]
        # top_score = top_candidate.get("score", 0.0)

        # 使用 decision_score 作为 fallback 的依据，而不是 reranker的score
        top_decision_score = top_candidate.get("decision_score", 0.0)
        
        if top_decision_score >= MIN_DECISION_SCORE_THRESHOLD:
            logger.warning(
                f"!! FALLBACK !! action=auto_merge batch_cluster_id={task_input.get('batch_cluster_id')} "
                f"score={top_decision_score:.4f}"
            )
            return {
                "decision": "merge",
                "matched_risk_event_id": top_candidate.get("risk_event_id"),
                "reason": f"Fallback: LLM failed. Auto-merged due to high decision score ({top_decision_score:.4f})."
            }


    # 如果没有高分候选，则默认新建
    return {
        "decision": "create_new",
        "matched_risk_event_id": None,
        "reason": f"Fallback: LLM failed and no high-confidence candidate found. Raw: {raw_output[:100]}"
    }


def _run_single_decision(task: Dict) -> Dict:
    """为单个聚类簇执行LLM决策"""
    batch_cluster_id = task['batch_cluster_id']
    
    # 清理候选事件，只保留必要字段
    clean_candidates = [
        {"risk_event_id": evt.get("risk_event_id"), "risk_event_title": evt.get("risk_event_title")}
        for evt in task.get("candidate_events", [])
    ]

    prompt_input = {
        "batch_cluster_title": task["batch_cluster_title"],
        "context_summaries": task["context_summaries"],
        "candidate_events": clean_candidates
    }

    prompt = multi_candidate_prompt_template.replace('$task_input', json.dumps(prompt_input, ensure_ascii=False))
    # print(f"Prompt for cluster {batch_cluster_id}:\n{prompt}")
    # print(f'prompt字符串长度：{len(prompt)}')

    trace_id = None

    try:
        # 调用 LLM 并获取 trace_id
        raw_response, trace_id = call_theta_llm(prompt, return_trace_id=True)
        decision_result = parse_llm_json_output(raw_response, task)
        decision_result['trace_id'] = trace_id  # 将 trace_id 添加到结果中
        
        logger.debug(f"LLM decision for cluster {batch_cluster_id}, trace_id: {trace_id}")
    except Exception as e:
        logger.error(
            f"!! FALLBACK !! reason=llm_call_failed cluster_id={task.get('batch_cluster_id')} "
            f"error='{type(e).__name__}'"
        )
        decision_result = parse_llm_json_output("", task) # 传入空字符串触发fallback
        decision_result['trace_id'] = trace_id

    return {"batch_cluster_id": batch_cluster_id, "decision_result": decision_result}


def compare_and_select_risk_event(tasks: List[Dict], max_workers: int = None) -> List[Dict]:
    """
    并发地为多个聚类簇执行多候选匹配决策
    """
    if not tasks:
        return []

    if max_workers is None:
        max_workers = LLM_DECIDER_MAX_WORKERS

    decision_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_run_single_decision, task): task for task in tasks}
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                task = future_to_task[future]
                logger.error(f"Task for cluster {task.get('batch_cluster_id')} generated an exception: {exc}")
                results.append({
                    "batch_cluster_id": task.get('batch_cluster_id'),
                    "decision_result": {
                        "decision": "create_new",
                        "matched_risk_event_id": None,
                        "reason": f"Fallback: Concurrency executor exception - {type(exc).__name__}",
                        "trace_id": None
                    }
                })

    decision_latency = time.time() - decision_start
    
    # 收集所有 trace_ids 用于日志
    trace_ids = [r['decision_result'].get('trace_id', 'N/A') for r in results if r.get('decision_result')]
    trace_ids_str = ', '.join(str(tid) for tid in trace_ids if tid)
    
    logger.info(
        f"[LLM Decision Batch] tasks={len(tasks)}, latency_s={decision_latency:.2f}, "
        f"trace_ids=[{trace_ids_str}]"
    )
    
    return results


if __name__ == '__main__':

    task_1 = {
        "batch_cluster_id": 'hhh_23',
        "batch_cluster_title": "饿了么会员自动续费",
        "context_summaries": ["用户咨询饿了么会员为何自动扣费", "用户表示未同意饿了么自动续费"],
        "candidate_events": [
            {"risk_event_id": "evt_abc", "risk_event_title": "饿了么会员自动续费问题"},
            {"risk_event_id": "evt_def", "risk_event_title": "用户对扣款原因存疑"},
            {"risk_event_id": "evt_ghi", "risk_event_title": "美团会员无法开通"}
        ]
        }
        
    task_2 = {
        "batch_cluster_id": 'dd_23',
        "batch_cluster_title": "江苏银行优惠券无法使用",
        "context_summaries": ["用户咨询江苏银行兑换的优惠券为何无法使用", "用户表示银行卡限额无法使用优惠券"],
        "candidate_events": [
            {"risk_event_id": "evt_abc", "risk_event_title": "异常扣款需退款"},
            {"risk_event_id": "evt_def", "risk_event_title": "信用卡扣款原因不明"},
            {"risk_event_id": "evt_ghi", "risk_event_title": "被扣款原因不明"}
        ]
        }
    output = _run_single_decision(task_2)

    # output = compare_and_select_risk_event([task_1, task_2])
    print(output)
