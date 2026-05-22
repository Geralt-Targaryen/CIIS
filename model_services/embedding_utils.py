import json
from tqdm import tqdm
import time
import logging
import traceback
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import numpy as np

from tingis.model_services.bge_m3  import get_embeddings
from tingis.global_vars import EMBEDDING_MAX_WORKERS, EMBEDDING_BATCH_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s: %(lineno)d - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def batch_generate_embeddings_from_texts(texts: List[str], ids: List[str], max_workers: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    通用批量 embedding 生成函数。
    
    :param texts: 要生成 embedding 的文本列表
    :param ids: 对应的唯一 ID 列表（与 texts 一一对应）
    :return: {id: embedding} 映射字典
    """
    if not texts or not ids:
        return {}

    assert len(texts) == len(ids), "texts 和 ids 长度必须一致"
    if max_workers is None:
        workers = EMBEDDING_MAX_WORKERS
    else:
        workers = max_workers
        
    # logger.info(f"Generating embeddings for {len(texts)} texts in batches...")

    embeddings_map = {}
    with ThreadPoolExecutor(max_workers=EMBEDDING_MAX_WORKERS) as executor:
        future_to_indices = {}
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch_texts = texts[i : i + EMBEDDING_BATCH_SIZE]
            batch_indices = list(range(i, min(i + EMBEDDING_BATCH_SIZE, len(texts))))
            future = executor.submit(get_embeddings, batch_texts)
            future_to_indices[future] = batch_indices

        # for future in tqdm(as_completed(future_to_indices), total=len(future_to_indices), desc="Generating embeddings"):
        for future in as_completed(future_to_indices):
            indices = future_to_indices[future]
            try:
                batch_embeddings = future.result()
                for i, orig_idx in enumerate(indices):
                    embeddings_map[ids[orig_idx]] = batch_embeddings[i]
            except Exception as e:
                logger.error(f"Error generating embeddings for batch starting at index {indices[0]}: {e}")
                for orig_idx in indices:
                    embeddings_map[ids[orig_idx]] = None

    return embeddings_map


if __name__ == "__main__":

    test_texts = ["用户反馈红包无法使用", "闲鱼无法解绑账号", "支付宝登录异常"]
    test_ids = ["xxxx", "https://tiyan.alipay.com/exp/exploration/xxxx", "id3"]

    result = batch_generate_embeddings_from_texts(test_texts, test_ids)

    for vid, emb in result.items():
        print(f"  {vid}: {emb[:3]}... (长度: {len(emb)})")
