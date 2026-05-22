import json
import time
import logging
import ssl
import socket
import uuid
import ast
from typing import List, Optional, Union
from httpx import Client, RequestError, HTTPStatusError

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

# ==================== SSL Context with OP_IGNORE_UNEXPECTED_EOF ====================
def create_ssl_context():
    ctx = ssl.create_default_context()
    ssl_option = getattr(ssl, 'OP_IGNORE_UNEXPECTED_EOF', 0x20000)
    ctx.options |= ssl_option
    # print(f"[DEBUG] SSL options set: 0x{ctx.options:x} (includes OP_IGNORE_UNEXPECTED_EOF=0x{ssl_option:x})")
    return ctx

# ==================== 全局复用的 HTTPX Client ====================
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

# ==================== Embedding 调用 ====================
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
        # response = requests.post(url, json=data, headers=headers)
        response = client.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except (RequestError, HTTPStatusError, ValueError, json.JSONDecodeError) as e:
        logger.error(
            f"Embedding service call failed, trace_id: {trace_id}, error: {e}",
            exc_info=True
        )
        return None


def safe_parse(item):
    if isinstance(item, str):
        try:
            return json.loads(item)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(item)
            except (ValueError, SyntaxError) as e:
                logger.error(f"Cannot parse string item: {item[:200]}..., error: {e}")
                raise ValueError("Item is not valid JSON or Python literal")
    return item


def get_embeddings(
    chunks,
    model="bge_m3",
    service_group_unique_id="xxx",
    is_pre=True,
):
    '''获取文本 embedding'''

    trace_id = generate_trace_id()

    query = {
        "model": model,
        "sents": chunks,
    }
    query_json = {"query": json.dumps(query)}

    response = maya_http_request(
        query_json,
        trace_id=trace_id,
        service_group_unique_id=service_group_unique_id,
        is_pre=is_pre,
    )

    if response is None:
        logger.warning(f"!! FALLBACK !! Embedding service returned None, trace_id: {trace_id}")
        return None
        
    try:
        result_map = response.get("resultMap")
        if not result_map:
            raise ValueError("Missing 'resultMap'")

        result = result_map.get("result")
        if result is None:
            raise ValueError("Embedding result is null")

        if len(chunks) > 1:
            object_value = result.get("objectValue") if isinstance(result, dict) else None
            if not object_value:
                raise ValueError("Missing 'objectValue'")
            embeddings = []
            for item in object_value:
                parsed = safe_parse(item)
                emb = parsed.get("embedding")
                if emb is None:
                    raise ValueError("Missing 'embedding'")
                embeddings.append(emb)
        else:
            parsed_result = safe_parse(result)
            emb = parsed_result.get("embedding")
            if emb is None:
                raise ValueError("Missing 'embedding'")
            embeddings = [emb]

        if any(e is None for e in embeddings):
            raise ValueError("Some embeddings are None")

        return embeddings

    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        logger.error(
            f"Failed to parse embedding response, trace_id: {trace_id}, error: {e}",
            exc_info=True
        )
        return None


if __name__ == "__main__":
    querys = [
    "支付宝转账失败，提示'账户异常'，但余额已扣除。",
    "扫码支付重复扣款两笔，金额未退回。",
    ]

    embedding = get_embeddings(querys)
    print(embedding) # listoflist or listoflist
    print(len(embedding))

