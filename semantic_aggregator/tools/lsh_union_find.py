# tingis/semantic_aggregator/tools/lsh_union_find.py

import numpy as np
from collections import defaultdict

from tingis.model_services.embedding_utils import batch_generate_embeddings_from_texts
from tingis.utils.common_utils import generate_algo_goc_id
from tingis.global_vars import BATCH_SEMANTIC_CLUSTERING_THRESHOLD

class LSH:
    def __init__(self, dim=768, num_tables=20, hash_size=4):
        """
        局部敏感哈希索引
        :param dim: 向量维度
        :param num_tables: 哈希表数量，增加可以提高召回率，减少漏检
        :param hash_size: 每个哈希表的哈希函数数量，增加可以提高哈希桶的区分度，减少误报
        """
        self.dim = dim
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.hash_tables = [defaultdict(list) for _ in range(num_tables)]
        self.projections = [np.random.randn(hash_size, dim) for _ in range(num_tables)]

    def _get_hash(self, vec, table_idx):
        projection = self.projections[table_idx]
        return tuple((np.dot(vec, projection.T) > 0).astype(int).tolist())

    def insert(self, vec, idx):
        """插入单个向量"""
        for table_idx in range(self.num_tables):
            hash_key = self._get_hash(vec, table_idx)
            self.hash_tables[table_idx][hash_key].append(idx)

    def query(self, vec):
        """查询相似候选"""
        candidates = set()
        for table_idx in range(self.num_tables):
            hash_key = self._get_hash(vec, table_idx)
            candidates.update(self.hash_tables[table_idx].get(hash_key, []))
        return candidates


class UnionFind:
    """并查集实现"""

    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x


def cosine_distance(vec1, vec2, norm1, norm2, eps=1e-8):
    """
    自定义余弦距离计算
    :param vec1: 向量1
    :param vec2: 向量2
    :param norm1: 向量1的预计算模长
    :param norm2: 向量2的预计算模长
    :param eps: 防止除零的小量
    :return: 余弦距离 (1 - 余弦相似度)
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = norm1 * norm2
    # 处理零向量特殊情况
    if norm_product < eps:
        return 1.0
    similarity = dot_product / norm_product
    similarity = np.clip(similarity, -1.0, 1.0)
    return 1 - similarity


def semantic_clustering(embeddings, threshold=None):
    """
    相似度阈值（余弦距离）：
    1、降低阈值会使得聚类条件更加严格，只有语义上更接近的文本才会被合并
    2、太高（例如 0.2 甚至更高）：距离很大的文本仍然是相似的，导致簇内容混杂
    :return: 聚类结果字典 {簇ID: 成员索引列表}
    """
    if not embeddings:
        return {}

    if threshold is None:
        threshold = BATCH_SEMANTIC_CLUSTERING_THRESHOLD
    # 预计算所有向量的模长
    norms = [np.linalg.norm(vec) for vec in embeddings]
    n = len(embeddings)
    uf = UnionFind(n)

    lsh = LSH(dim=len(embeddings[0]))

    # 建立LSH索引
    for idx, vec in enumerate(embeddings):
        lsh.insert(vec, idx)

    # 近邻搜索与聚类
    for idx in range(n):
        candidates = lsh.query(embeddings[idx])
        for candidate in candidates:
            if candidate <= idx:
                continue
            # 使用预计算的模长
            distance = cosine_distance(embeddings[idx], embeddings[candidate],
                                       norms[idx], norms[candidate])
            if distance < threshold:
                uf.union(idx, candidate)

    # 整理聚类结果
    clusters = defaultdict(list)
    for idx in range(n):
        clusters[uf.find(idx)].append(idx)
        
    # 生成全局唯一 cluster_id
    final_clusters = {}
    for members in clusters.values():
        cluster_id = generate_algo_goc_id()
        final_clusters[cluster_id] = sorted(members)

    return final_clusters


if __name__ == "__main__":
    np.random.seed(42)

    queries_1 = [
        "碰一碰大牌补贴免单红包未抵扣导致实付金额无法退还",
        "大牌补贴碰一下红包无法抵扣",

        "用户在闲鱼小程序中无法解绑，点击客服也没有反应，现在不方便电话。",
        "用户在闲鱼app上查询退款是否到账，但账单中未找到，咸鱼称已打款，但交易订单扣款手续费为0.06和0.1,用户怀疑退款金额与实际不符，并询问是否入账*元。",
        "咸鱼上无法支付",

        "用户遇到了红包无法使用的问题，需要处理",
        "用户遇到了青春版无法关闭的问题",

        "哈哈哈",
    ]

    queries_2 = [
        "用户遇到了小鸡无法睡觉的问题，怀疑系统崩溃。",
        "用户咨询小鸡无法睡觉原因。",
        "用户点击蚂蚁庄园小鸡睡觉，但小鸡不睡，没有反馈和弹窗提示，点击之后没有弹窗，重启过手机和app也没有用，询问是否有解决办法。",

        "用户在小鸡庄园家庭任务中捐款*元以上给亲密度，但捐完后没有收到，询问原因。",
        "用户反馈小鸡捐款亲密度活动，但未收到退款，用户希望平台内联系，不要致电。",
        "用户在家庭捐款时，小鸡亲密度未加上，询问如何添加。",

        "用户遇到了蚂蚁庄园饲料积攒后消失的问题，希望得到解决。",
        "用户遇到了小鸡拿饲料的活动突然关闭的问题，导致几十万的饲料无法领取。",
        "用户遇到了小鸡饲料不见了的问题。"
    ]

    queries = queries_2
    query_ids = [str(i) for i in range(len(queries))]
    embeddings_map = batch_generate_embeddings_from_texts(queries, query_ids)
    embedding = [embeddings_map[str(i)] for i in range(len(queries))]

    clusters = semantic_clustering(embedding)
        
    cluster_texts = {}
    for cluster_id, indices in clusters.items():
        texts = [queries[i] for i in indices]
        cluster_texts[cluster_id] = texts

    for cluster_id, texts in cluster_texts.items():
        print(f"{cluster_id}: \n{texts}\n")
