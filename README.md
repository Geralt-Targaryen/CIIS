# CIIS - 风险事件检测与告警系统

## 项目简介

CIIS (Customer Incident Intelligent Service) 是对 ACL 2026 Industry (oral) 论文 TingIS 的复现。TingIS 是一个智能风险事件检测与告警系统，主要用于实时监控和分析用户反馈（原声/Voice）数据，通过 NLP 和向量检索技术自动识别、聚合风险事件，并触发告警。

**核心能力：**

- **数据采集**：从热在线用户原声库（ting_tiyan）拉取用户反馈数据
- **智能路由**：通过文本搜索 + 语义检索的混合策略，将原声分发至对应的业务线（GOC/GM Code）
- **事件聚合**：基于语义相似度进行批内聚类，并通过 LLM 决策判断是否合并至历史事件
- **告警管理**：支持动态基线告警，静默窗口控制，事件快照等功能

---

## 项目结构

```
tingis/
├── services/                    # 服务入口
│   ├── model.py                 # 线上部署入口（继承 MayaBaseHandler）
│   └── client_scheduler.py      # 调度客户端入口（定时调用 model 服务）
├── preprocessing/               # 数据预处理模块
│   ├── data_pipeline.py         # 数据拉取与处理流水线
│   ├── preprocessor.py          # 原声清洗与过滤
│   ├── filter_rules.py          # 过滤规则配置
│   ├── summarizer.py            # 原声摘要生成
│   └── read_data_source.py      # 数据源读取
├── voice_router/                # 原声路由模块
│   ├── voice_router_hybrid_search.py   # 混合搜索路由（文本 + 语义）
│   ├── voice_router_text_search.py     # 文本关键词搜索
│   └── voice_router_semantic_search.py # 语义向量搜索
├── semantic_aggregator/         # 语义聚合模块
│   ├── aggregation_pipeline.py  # 聚合主流程
│   ├── voice_grouper.py         # 批内聚类与摘要
│   ├── event_matcher.py         # 历史事件匹配
│   └── matching/
│       ├── matcher.py           # 向量检索与 Reranker
│       └── decider.py           # LLM 决策（merge/create/suppress）
├── event_management/            # 事件管理模块
│   ├── event_manager.py         # 事件创建、更新、告警
│   ├── event_alert_engine.py    # 告警引擎（动态基线）
│   └── event_snapshotter.py     # 事件快照
├── model_services/              # 模型服务模块
│   ├── bge_m3.py                # BGE-M3 Embedding 服务
│   ├── bge_reranker_v2_m3.py    # BGE-Reranker 服务
│   ├── embedding_utils.py       # Embedding 批量生成工具
│   ├── llm_client.py            # Theta 平台 LLM 调用
│   └── modelops.py              # 模型操作工具
├── knowledge_base/              # 知识库模块
│   ├── index_document/          # 向量知识库
│   │   ├── base_vector_kb.py    # 基础向量知识库类
│   │   ├── risk_event_summary_kb.py  # 风险事件摘要知识库
│   │   ├── voice_title_kb.py    # 原声标题知识库
│   │   └── false_positive_kb.py # 负样本知识库
│   └── ob_db_manager/           # OceanBase 数据库管理
│       ├── ob_db_operations.py  # 数据库操作封装
│       └── mist_auth.py         # 认证模块
├── utils/                       # 工具模块
│   └── common_utils.py          # 通用工具函数
├── global_vars.py               # 全局配置与变量
├── conf/                        # 配置文件
│   └── docker/                  # Docker 相关配置
│       ├── Dockerfile           # 容器构建文件
│       ├── build.yaml           # 构建配置
│       └── scripts/             # 启动/停止脚本
└── init_env.sh                  # 环境初始化脚本
```

---

## 环境依赖

### 系统要求

- **Python**: 3.10+
- **OS**: Linux (AliOS 7u2 推荐)
- **时区**: Asia/Shanghai

### 核心依赖

```txt
# 数据库
pymysql
polygonmilvus==1.1.2

# NLP
pkuseg==0.0.25
numpy==1.23.5

# LLM 服务
openai

# RPC 调度
layotto

# 工具库
pandas
openpyxl
requests==2.31.0
ws4py
urllib3==1.26.12

# 内部依赖
aii-pypai
aistudio-common
aistudio_serving
aeac-authn==1.2.2a1
```

> 详细依赖请参考 `init_env.sh` 或项目 `requirements.txt`

---

## 快速开始

### 1. 环境初始化

```bash
# 挂载 NAS（如需访问知识库文件）
sudo mount -t nfs -o xxxxx

# 安装依赖
pip install -i https://pypi.antfin-inc.com/simple/ -r requirements.txt
```

### 2. 配置数据库连接

编辑 `global_vars.py`，配置以下数据库连接信息：

- `TING_DATABASE_CONFIG`: 原声数据源数据库连接
- `TINGIS_DATABASE_CONFIG`: 风险事件存储数据库连接
- `polygon_token_prod`: Milvus 向量数据库 Token

---

## 部署说明

#### 处理流程

```
┌─────────────────────────────────────────────────────────────────┐
│                       model.py 主流程                            │
├─────────────────────────────────────────────────────────────────┤
│  1. 解析请求，加载状态（last_processed_id, cursor_origin_time）   │
│                        ↓                                        │
│  2. 数据拉取 (fetch_ting_tiyan_v3)                              │
│     - 增量模式：id > last_processed_id                           │
│     - 重置模式：gmt_create >= start_time_override               │
│                        ↓                                        │
│  3. 数据处理 (route_and_enrich_voices)                          │
│     - 清洗过滤                                                   │
│     - 混合搜索路由                                               │
│     - 原声摘要生成                                               │
│     - Embedding 生成                                            │
│                        ↓                                        │
│  4. 语义聚合 (identify_and_reconcile_risk_events)               │
│     - 批内聚类                                                   │
│     - 历史事件匹配                                               │
│     - LLM 决策                                                   │
│                        ↓                                        │
│  5. 事件管理 (process_risk_events_batch)                        │
│     - 持久化映射关系                                             │
│     - 更新事件声量                                               │
│     - 触发告警检查                                               │
│                        ↓                                        │
│  6. 状态保存 & 快照检查                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 调用说明

#### 功能说明

1. **单实例保护**：通过文件锁确保同一时间只有一个调度器运行
2. **手动触发**：支持设置 `start_time_str` 从指定时间点开始处理
3. **周期任务**：默认每 2 分钟检查一次是否有新数据

#### 调用流程

```
┌─────────────────────────────────────────────────────────────────┐
│                 client_scheduler.py 调度流程                     │
├─────────────────────────────────────────────────────────────────┤
│  1. 初始化 Layotto 客户端                                        │
│                        ↓                                        │
│  2. 手动触发（可选）process_all_pending_data_tingis()            │
│                        ↓                                        │
│  3. 注册周期任务 schedule.every(2).minutes.do(...)              │
│                        ↓                                        │
│  4. 循环调度：                                                    │
│     - 调用 call_maya_service() 发送请求                         │
│     - 解析响应 status                                           │
│     - 若 status == "has_more_to_process"，立即继续              │
│     - 若 status == "up_to_date"，等待下一周期                   │
└─────────────────────────────────────────────────────────────────┘
```

#### RPC 调用接口

```python
def call_maya_service(payload: dict) -> Optional[dict]:
    """
    通过 Layotto 调用 model 服务

    Args:
        payload: 请求数据
            - start_time_override: 可选，重置时间游标

    Returns:
        响应数据字典，包含 status、last_processed_id、process_stats 等
    """
```

---

## 配置说明

### 核心配置项（global_vars.py）

#### 1. 批处理与并发控制

| 配置项                          | 默认值 | 说明                         |
| ------------------------------- | ------ | ---------------------------- |
| `DB_FETCH_BATCH_SIZE`           | 400    | 每次从数据库拉取的最大记录数 |
| `DB_INSERT_BATCH_SIZE`          | 100    | 批量插入数据库的批次大小     |
| `PIPELINE_MAX_WORKERS`          | 10     | 数据处理流水线最大并发数     |
| `EMBEDDING_MAX_WORKERS`         | 50     | Embedding 服务并发数         |
| `LLM_VOICE_SUMMARY_MAX_WORKERS` | 50     | 摘要生成并发数               |

#### 2. 数据路由配置

#### 3. 事件聚合配置

| 配置项                                | 默认值 | 说明               |
| ------------------------------------- | ------ | ------------------ |
| `EVENT_MATCHING_LOOKBACK_DAYS`        | 14     | 事件匹配回溯天数   |
| `BATCH_SEMANTIC_CLUSTERING_THRESHOLD` | 0.15   | LSH 聚类相似度阈值 |
| `RISK_EVENT_RETRIEVAL_TOP_K`          | 10     | 向量召回 Top-K     |
| `RISK_EVENT_RERANKER_TOP_K`           | 3      | 重排 Top-K         |
| `RISK_EVENT_RERANKER_THRESHOLD`       | 0.85   | 精排相似度阈值     |

#### 4. 告警配置

| 配置项                            | 默认值 | 说明                   |
| --------------------------------- | ------ | ---------------------- |
| `EVENT_MAX_AGE_HOURS`             | 6      | 原声归属最大回溯小时数 |
| `ALERTING_WINDOW_HOURS`           | 1      | 增量声量计算窗口       |
| `ALERTING_MIN_TOTAL_VOLUME`       | 3      | 触发告警最小总声量     |
| `ALERTING_MIN_INCREMENTAL_VOLUME` | 3      | 触发告警最小增量声量   |
| `ALERT_SILENCE_WINDOW_MINUTES`    | 120    | 告警静默窗口（分钟）   |
| `SNAPSHOT_HEARTBEAT_MINUTES`      | 10     | 快照心跳间隔           |

#### 5. 知识库配置

---

## 常见问题

### Q: 如何添加新的过滤规则？

**A**: 编辑 `preprocessing/filter_rules.py`，添加新的过滤规则，支持：

- 前缀 + 长度过滤
- OR 关键词过滤
- AND 关键词过滤
- 包含/排除组合过滤

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Tingis 系统架构                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │ 用户原声库    │ ───> │  数据清洗     │ ───> │  原声路由     │              │
│  │ (ting_tiyan) │      │  (过滤/脱敏)  │      │ (文本+语义)   │              │
│  └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                      │                      │
│                                                      ↓                      │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │  风险事件库   │ <─── │  LLM 决策    │ <─── │  语义聚合     │              │
│  │ (risk_event) │      │(merge/create)│      │ (批内聚类)    │              │
│  └──────────────┘      └──────────────┘      └──────────────┘              │
│         │                    │                                            │
│         ↓                    ↓                                            │
│  ┌──────────────┐      ┌──────────────┐                                  │
│  │  告警引擎     │      │  向量知识库   │                                  │
│  │(动态基线)    │      │  (Milvus)    │                                  │
│  └──────────────┘      └──────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 引用

```
@article{DBLP:journals/corr/abs-2604-21889,
  author       = {Jun Wang and
                  Ziyin Zhang and
                  Rui Wang and
                  Hang Yu and
                  Peng Di and
                  Rui Wang},
  title        = {TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents
                  at Enterprise Scale},
  journal      = {CoRR},
  volume       = {abs/2604.21889},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2604.21889},
  doi          = {10.48550/ARXIV.2604.21889},
  eprinttype   = {arXiv},
  eprint       = {2604.21889}
}
```
