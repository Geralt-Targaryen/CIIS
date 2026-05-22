"""
Theta平台的大模型
"""
import re
from openai import OpenAI
import socket
import time
import os
import threading
from typing import Tuple
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 线程安全
class TraceIdGenerator:
    def __init__(self):
        self.sequence = 1000
        self.ip_hex = self._get_ip_hex()
        self._lock = threading.Lock()
        # 添加一个线程本地存储，用于确保每个线程的随机性
        self._thread_local = threading.local()

    def _get_ip_hex(self) -> str:
        """Get local IP address and convert it to hex format."""
        try:
            # Create a socket to get the local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Doesn't need to be reachable
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            # Fallback to localhost if we can't get the IP
            ip = '127.0.0.1'

        # Convert IP address to hex
        hex_parts = [format(int(x), '02x') for x in ip.split('.')]
        return ''.join(hex_parts)

    def _get_timestamp(self) -> str:
        """Get current timestamp in milliseconds."""
        # return str(int(time.time() * 1000))
        # 使用纳秒级时间戳，确保并发时的唯一性
        return str(int(time.time() * 1000000))  # 微秒级

    def _get_sequence(self) -> str:
        """Get next sequence number in a thread-safe manner."""
        with self._lock:
            sequence = self.sequence
            self.sequence += 1
            # if self.sequence > 9000:
            if self.sequence > 9999:  # 扩大序列范围
                self.sequence = 1000
            # return str(sequence)
            return f"{sequence:04d}"  # 4位数字，补零

    def _get_thread_random(self) -> str:
        """Get a thread-local random number."""
        if not hasattr(self._thread_local, 'random_id'):
            # 为每个线程生成一个唯一的随机ID
            self._thread_local.random_id = random.randint(10000, 99999)
        return str(self._thread_local.random_id)

    def _get_process_id(self) -> str:
        """Get current process ID."""
        # return str(os.getpid())
        return f"{os.getpid():05d}"  # 5位数字，补零

    def generate_trace_id(self) -> str:
        """
        Generate a complete TraceId with enhanced uniqueness.
        Format: IP(8) + Timestamp_us(16) + Sequence(4) + ThreadRandom(5) + ProcessId(5)
        Total: 38 characters
        """
        timestamp = self._get_timestamp()
        sequence = self._get_sequence()
        thread_random = self._get_thread_random()
        process_id = self._get_process_id()
        
        # 添加额外的随机数，确保并发时的唯一性
        extra_random = f"{random.randint(100, 999)}"
        
        return f"{self.ip_hex}{timestamp}{sequence}{thread_random}{process_id}{extra_random}"


# 全局 TraceId 生成器实例
_trace_id_generator = TraceIdGenerator()


def call_theta_llm(query, model="Kimi-K2-Instruct", stream=True, return_trace_id=True):
    '''
    调用 Theta 平台的大模型
    '''

    # 生成 trace_id
    trace_id = _trace_id_generator.generate_trace_id()

    try:
        client = OpenAI(
            api_key="your_api_key",
            base_url="base_url"
        )
        completion = client.chat.completions.create(
            model = model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            top_p = 1,
            stream=stream,
            extra_headers = {
                "Content-Type": "application/json",
                "SOFA-TraceId": trace_id,
            },
            extra_body={
                "enable_sec_check": False,
                "timeout_ms": 600000  # 默认400000ms，最大为 600000ms = 10min
            }
        )

        # 处理响应
        if stream == False:
            response = completion.choices[0].message.content
        else:
            full_content = ""
            for chunk in completion:
                if chunk.choices:
                    if chunk.choices[0].delta.content is not None:
                        full_content += chunk.choices[0].delta.content
            response = full_content
        
        if model in ["Qwen3-8B", "Qwen3-32B"]:
            result = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            result = result.strip()
        else:
            result = response

        if return_trace_id:
            return result, trace_id
        return result

    except Exception as e:
        logger.error(
            f"Theta call failed, trace_id: {trace_id}, model: {model}, error: {type(e).__name__}: {e}",
            exc_info=True
        )
        if return_trace_id:
            return None, trace_id
        return None


if __name__ == '__main__':

    # model = "Qwen3-8B"
    # model = "Qwen3-32B"
    # model = "Qwen3-Next-80B-A3B-Instruct"
    # model = "Qwen3-235B-A22B-Instruct-2507"
    # model = "DeepSeek-V3.1-Terminus"
    model = "Kimi-K2-Instruct"
    # model = "Kimi-K2-Instruct-0905"
    # stream = False
    stream = True
    query = '今天你快乐吗？'
    result = call_theta_llm(query)
    print(f'结果: \n{result}')
