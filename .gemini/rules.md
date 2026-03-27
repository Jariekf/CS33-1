---
trigger: always_on
alwaysApply: true
---

# CS33-1 多智能体数据分析系统 - 项目开发规则

## 项目概述
本项目是一个基于 **LangGraph** 的多智能体数据分析与商业洞察系统，包含 6 个智能体节点协作完成从数据清洗到商业报告的全流水线。

## 🚨 不可修改的全局契约
以下文件定义了所有模块间的接口规范，**任何窗口/开发者不得擅自修改**：
1. **`core/state.py`** — `GraphState` 的字段定义（所有节点共享的状态字典）
2. **`core/graph.py`** — LangGraph 工作流拓扑（节点名称、边连接、条件路由映射）
3. **`agents/supervisor.py` 中的 `ReviewReport`** — Pydantic 结构化审查报告的字段定义
4. **`config.py`** — 全局常量（重试次数、评分阈值等）

如需修改以上任何文件，必须在主窗口中统一协调。

## 模块分工方案

| 开发窗口 | 负责文件 | 核心任务 |
|---|---|---|
| **窗口 1 (Manager + RAP)** | `agents/manager.py`, `tools/arxiv_tool.py` | 实现 RAP 检索增强计划生成：接入真实 LLM，将用户目标转为 arxiv 查询词，生成结构化 `global_plan` |
| **窗口 2 (数据处理)** | `agents/data_cleaning.py`, `agents/feature_engineering.py` | 实现代码生成与内部自救循环：接入 LLM 生成 Python 代码、解析报错并自动修复 |
| **窗口 3 (模型 + 审查)** | `agents/model_selection.py`, `agents/supervisor.py` | 实现模型训练代码生成（含 HPO/交叉验证）及 Supervisor 的 LLM 审查评分逻辑 |
| **窗口 4 (商业转化)** | `agents/business_translator.py`, `tools/file_reader.py` | 实现 Feature-Insight RAG：从落盘的特征重要性出发，检索文献，生成商业洞察报告 |

## 开发规范

### 节点函数签名约定
所有 Agent 节点函数必须遵循：
```python
def xxx_node(state: GraphState) -> dict:
    # 从 state 中读取本阶段所需字段
    # 执行逻辑
    # 返回需要更新的字段子集 (不要返回完整 state)
```

### 关键设计原则
1. **边界隔离**：每个 Agent 只读取 `global_plan` 中属于自己的子计划键（如 `data_cleaning_plan`），禁止访问其他阶段的计划。
2. **Memory Append-Only**：向 `memory` 写入时，只能返回 `{"memory": ["新条目"]}`，LangGraph 会自动追加。
3. **摘要入驻 + 详情落盘**：Supervisor 通过后，详细报告写入 `./workspace/` 目录下的 `.md` 文件，Memory 中只存 ≤100 字摘要 + 文件路径。
4. **内部自救优先**：Worker 节点的代码执行报错时，先在本节点内部重试修复（最多 `MAX_INNER_RETRY` 次），只有跑通后的结果才流转给 Supervisor。
5. **所有 TODO 标记**：代码中的 `TODO` 注释标识了需要替换为真实 LLM 调用的桩函数位置。

### LLM 调用统一方式
```python
from langchain_openai import ChatOpenAI
from config import LLM_CONFIG

llm = ChatOpenAI(
    model=LLM_CONFIG["model"],
    api_key=LLM_CONFIG["api_key"],
    base_url=LLM_CONFIG["base_url"],
    temperature=LLM_CONFIG["temperature"],
)
```
