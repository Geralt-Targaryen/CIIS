import json
import time
import re
import logging
import ssl
import socket
import uuid
from typing import Union
from functools import partial
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
    # 获取 OP_IGNORE_UNEXPECTED_EOF 值（兼容旧版 Python）
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
            verify=ssl_ctx,      # 传入自定义 SSLContext
            timeout=180, # 网关端耗时就是180s
            http2=False          # 如不需要 HTTP/2，建议关闭以减少兼容问题
        )
    return _httpx_client

# ==================== ModelOps 调用 ====================
def modelops(messages: Union[list[dict], str], model: str, **kwargs):
    trace_id = generate_trace_id()
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]

    url = "url"

    # from urllib.parse import urlparse
    # import subprocess
    # parsed_url = urlparse(url)
    # host = parsed_url.hostname
    # try:
    #     cmd = ["dig", host]
    #     result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    #     print(result.stdout)
    # except subprocess.CalledProcessError as e:
    #     print(f"执行dig命令出错: {e.stderr}")
    # except FileNotFoundError:
    #     print("dig命令未找到，请确保系统已安装dig工具")

    headers = {
        'Content-Type': 'application/json;charset=utf-8',
        'Authorization': 'your_authorization',
        'SOFA-TraceId': trace_id,
        'SOFA-RpcId': '0.1'
    }
    json_data = {'model': model, 'messages': messages, 'stream': False, **kwargs}

    client = get_httpx_client()

    try:
        response = client.post(url, json=json_data, headers=headers)
        response.raise_for_status()

        resp_json = response.json()
        message = resp_json['choices'][0]['message']
        ret = message['content']
        if 'reasoning_content' in message:
            ret = f'{"<think>"}{message["reasoning_content"]}{"</think>"}{ret}'
        return ret

    except (RequestError, HTTPStatusError, ValueError, KeyError, json.JSONDecodeError) as e:
        logger.error(
            f"LLM call failed (no retry), trace_id: {trace_id}, error: {e}",
            exc_info=True
        )
        return None

# ==================== Qwen3 Handler ====================
def qwen3_handler(message: str, env: str = 'prod', model: str = 'Qwen3_8B_10K_Chat_20250627_BF16_vLLM_2A10V2-v1', is_think: bool = False) -> str:
    original_message = message

    # print(f'原始输入：{original_message}')
    if not is_think:
        message = f'{message}/no_thinking'

    qwen3 = partial(modelops, model=model, extraConfig={"modelEnv": env})
    res = qwen3(message)

    # 如果 LLM 调用失败（返回 None），回退到原始消息
    if res is None:
        logger.warning("!! FALLBACK !! LLM summarization failed, falling back to original message.")
        return original_message

    if not is_think:
        res = re.sub(r'</?think>', '', res)
        res = re.sub(r'\s+', ' ', res).strip()

    return res


if __name__ == '__main__':
    query = '今天你快乐吗？'
    result = qwen3_handler(query)
    print(result)

