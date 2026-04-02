---
trigger: always_on
alwaysApply: true
---

# （示例）项目开发规则

将本文件复制为 `docs/rules.md` 后按团队约定修改；`docs/rules.md` 已被 git 忽略，请勿把含敏感或仅本地生效的内容提交进仓库。

```bash
cp docs/rules.example.md docs/rules.md
```

以下为默认模板内容，可按窗口分工自行增删。

## 项目概述

本项目是一个基于 **LangGraph** 的多智能体数据分析与商业洞察系统，包含多个智能体节点协作完成从数据清洗到商业报告的全流水线。

## 核心框架保护条款（调优阶段必读）

在后续的 **Prompt 调优、工具扩展和逻辑优化** 阶段，以下核心系统级文件属于**只读**的底层基建，**任何窗口/开发者严禁以优化为名进行修改**（除非经主控或架构讨论明确批准）：

1. **`core/state.py`** — GraphState 状态机数据结构
2. **`core/graph.py`** — LangGraph 工作流拓扑
3. **`core/llm.py`** — 统一 LLM 调用与代码抽取等基础机制
4. **`main.py`** — 程序入口与图执行循环
5. **`agents/supervisor.py` 中的 `ReviewReport` 数据类** — 审查输出 Schema
6. **`config.py` 中的框架常数** — 如重试次数等（非明确要求不可改）

**调优建议范围**：`Prompt` 模板（`_SYSTEM_PROMPT_*` 等）、`tools/*` 工具实现、传入 LLM 的数据拼装格式。

## 模块分工方案

| 开发窗口 | 负责文件 | 核心任务 |
|---|---|---|
| 窗口 1 | `agents/manager.py`, `tools/arxiv_tool.py` | RAP / 计划与检索 |
| 窗口 2 | `agents/data_cleaning.py`, `agents/feature_engineering.py` | 数据清洗与特征工程代码生成与自救 |
| 窗口 3 | `agents/model_selection.py`, `agents/supervisor.py` | 模型训练与 Supervisor 审查 |
| 窗口 4 | `agents/business_translator.py`, `tools/file_reader.py` | 商业洞察与 RAG |

（按实际课程或项目组分工表替换上表。）

## 开发规范

### 节点函数签名约定

```python
def xxx_node(state: GraphState) -> dict:
    # 从 state 中读取本阶段所需字段
    # 执行逻辑
    # 返回需要更新的字段子集（不要返回完整 state）
```

### 关键设计原则

1. **边界隔离**：每个 Agent 只读取 `global_plan` 中属于自己的子计划键。
2. **Memory Append-Only**：向 `memory` 写入时返回 `{"memory": ["新条目"]}`。
3. **摘要入驻 + 详情落盘**：Supervisor 通过后，详细报告写入 `./workspace/`，Memory 中存短摘要与路径。
4. **内部自救优先**：Worker 内重试至多 `MAX_INNER_RETRY` 次后再上交 Supervisor。
5. **TODO**：代码中 `TODO` 标记待接真实 LLM 或待完善处。

### LLM 调用统一方式

优先使用项目内封装（如 `core.llm.call_llm` / `get_llm`），避免在各 Agent 中重复构造客户端；若文档仍要求直接使用 `ChatOpenAI`，请与 `config` 中的 `LLM_CONFIG` 保持一致。
