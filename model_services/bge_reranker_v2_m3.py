import json
import time
import logging
import ssl
import socket
import uuid
import ast
from typing import List, Optional, Dict, Any
from httpx import Client, RequestError, HTTPStatusError
from concurrent.futures import ThreadPoolExecutor, as_completed

from tingis.global_vars import RERANKER_MAX_WORKERS

logger = logging.getLogger(__name__)

# ==================== Trace ID 工具 ====================
def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
            return ip_address
        except Exception:
            return "127.0.0.1"

def ip_to_hex(ip):
    return ''.join(f'{int(part):02x}' for part in ip.split('.'))

def generate_trace_id():
    ip_hex = ip_to_hex(get_ip_address())
    timestamp = int(time.time() * 1000)
    sequence = uuid.uuid4().int & 0xFFFFFFFF
    return f"{ip_hex}{timestamp:013d}{sequence:08x}"

# ==================== SSL Context ====================
def create_ssl_context():
    ctx = ssl.create_default_context()
    ssl_option = getattr(ssl, 'OP_IGNORE_UNEXPECTED_EOF', 0x20000)
    ctx.options |= ssl_option
    return ctx

_httpx_client = None

def get_httpx_client():
    global _httpx_client
    if _httpx_client is None:
        ssl_ctx = create_ssl_context()
        _httpx_client = Client(
            verify=ssl_ctx,
            timeout=180,
            http2=False
        )
    return _httpx_client


# ==================== Reranker HTTP 调用 ====================
def maya_http_request(
    query,
    trace_id,
    service_group_unique_id="xxx",
    is_pre=True,
):
    if is_pre:
        url = f"https://xxx/inference/{service_group_unique_id}/v1"  # 预发域名
    else:
        url = f"https://xxx/inference/{service_group_unique_id}/v1"  # 生产域名

    headers = {
        "Content-Type": "application/json",
        "MPS-app-name": "test",
        "MPS-http-version": "1.0",
        "MPS-trace-id": trace_id,
    }

    data = {"features": query}
    client = get_httpx_client()

    try:
        response = client.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except (RequestError, HTTPStatusError, ValueError, json.JSONDecodeError) as e:
        logger.error(
            f"Reranker service call failed, trace_id: {trace_id}, error: {e}",
            exc_info=True
        )
        return None

# ==================== 单次 rerank 调用 ====================
def rerank_serve(
    query,
    knowledge_list,
    model="bge_reranker_v2_m3",
    service_group_unique_id="xxx",
    # is_pre=True,
    is_pre=False,
):
    '''对单个 query + knowledge_list 执行 rerank'''

    trace_id = generate_trace_id()

    request_data = {
        "model": model,
        "query": query,
        "knowledge_list": knowledge_list,
    }
    query_json = {"query": json.dumps(request_data)}

    response = maya_http_request(
        query_json,
        trace_id=trace_id,
        service_group_unique_id=service_group_unique_id,
        is_pre=is_pre,
    )

    if response is None:
        logger.warning(f"!! FALLBACK !! Reranker service returned None, trace_id: {trace_id}")
        return None

    try:
        result_map = response.get("resultMap")
        if not result_map:
            raise ValueError("Missing 'resultMap' in response")

        result_str = result_map.get("result")
        if result_str is None:
            raise ValueError("'result' is null")

        try:
            result_dict = json.loads(result_str)
        except json.JSONDecodeError:
            try:
                result_dict = ast.literal_eval(result_str)
            except (ValueError, SyntaxError) as e2:
                logger.error(f"Failed to parse result_str: {str(result_str)[:200]}..., error: {e2}")
                raise ValueError("Invalid result format")

        return json.dumps(result_dict, ensure_ascii=False)

    except (ValueError, KeyError, TypeError) as e:
        logger.error(
            f"Failed to parse reranker response, trace_id: {trace_id}, error: {e}",
            exc_info=True
        )
        return None

# ==================== 并发 rerank ====================
def parallel_rerank(requests, max_workers=None):
    """
    并发执行多个rerank请求
    """
    if max_workers is None:
        max_workers = RERANKER_MAX_WORKERS

    args_list = []
    for req in requests:
        args = {
            "query": req["query"],
            "knowledge_list": req["knowledge_list"],
            # "model": req.get("model", "bge_reranker_v2_m3"),
            # "service_group_unique_id": req.get("service_group_unique_id", "xxx"),
            # "is_pre": req.get("is_pre", False)
        }
        args_list.append(args)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(rerank_serve, **args): idx
            for idx, args in enumerate(args_list)
        }

        results = [None] * len(requests)
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                logger.error(f"Unexpected error in parallel rerank task: {e}", exc_info=True)
                results[idx] = None

    return results


if __name__ == "__main__":

    is_batch = True
    # is_batch = False
    if is_batch:
        test_requests = [
            {
                "query": "一带一路",
                "knowledge_list": ["一带一路建设", "国际经贸合作", "hi你好"]
            },
            {
                "query": "人工智能",
                "knowledge_list": ["机器学习", "深度学习框架", "自然语言处理"],
            }
        ]
        
        parallel_results = parallel_rerank(test_requests, max_workers=4)
        print(parallel_results) # 标准json
        # for i, result in enumerate(parallel_results):
        #     print(result)

    else:
        query = "一带一路"
        knowledge_list = ["一带一路", "hi呀"]
        result = rerank_serve(query, knowledge_list)
        print(result) # {'query': '一带一路', 'knowledge_list': [{'index': 0, 'score': 0.9953472018241882, 'content': '一带一路'}, {'index': 1, 'score': 0.05744408816099167, 'content': 'hi呀'}]}
        