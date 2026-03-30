---
trigger: always_on
alwaysApply: true
---

# CS33-1 多智能体数据分析系统 - 项目开发规则

## 项目概述
本项目是一个基于 **LangGraph** 的多智能体数据分析与商业洞察系统，包含 6 个智能体节点协作完成从数据清洗到商业报告的全流水线。

## 🚨 核心框架保护条款 (调优阶段必读)
本项目已完成各 Agent 的骨架实现。在后续的**Prompt 调优、工具扩展和逻辑优化**阶段，以下核心系统级文件属于**只读**的底层基建，**任何窗口/开发者严禁以优化为名进行修改**：

1. **`core/state.py`** — GraphState 状态机数据结构（全局状态流转的血脉）
2. **`core/graph.py`** — LangGraph 工作流拓扑（阶段转移逻辑与路由映射图）
3. **`core/llm.py`** — 统一 LLM 实例管理与解析库函数（代码抽取等基础机制）
4. **`main.py`** — 整体程序入口与图执行循环引擎
5. **`agents/supervisor.py` 中的 `ReviewReport` 数据类** — 决定 Pydantic 输出稳定性的核心 Schema
6. **`config.py` 中的框架常数** — 若非明确要求放宽或收紧，一律不可改动（如重试机制参数）

**调优建议范围**：后续的工作重点应完全聚焦在：修改 `Prompt 模板`（`_SYSTEM_PROMPT_*` 等变量）、增强或重构具体工具的具体逻辑 `tools/*`、调整传入 LLM 数据的拼装格式。如确实因为瓶颈需要动静核心框架，必须在主控或专门的架构讨论中发起请求。

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
