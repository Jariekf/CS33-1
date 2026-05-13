---
trigger: always_on
alwaysApply: true
---

# (Example) project development rules

Copy this file to `docs/rules.md` and adjust per team agreement; `docs/rules.md` is git-ignored. Do not commit secrets or purely local-only content to the repository.

```bash
cp docs/rules.example.md docs/rules.md
```

Below is a default template; edit per workstream as needed.

## Project overview

This project is a **LangGraph**-based multi-agent data analysis and business-insight system. Multiple agent nodes collaborate on the full pipeline from data cleaning to business reports.

## Core framework protection (read before prompt tuning)

During **prompt tuning, tool extensions, and logic optimization**, the following system-level files are **read-only** infrastructure. **No workstream or developer may change them under the guise of optimization** unless explicitly approved by the lead or architecture discussion:

1. **`core/state.py`** — `GraphState` schema
2. **`core/graph.py`** — LangGraph workflow topology
3. **`core/llm.py`** — unified LLM calls and code extraction
4. **`main.py`** — entry point and graph execution loop
5. **`agents/supervisor.py`** — the `ReviewReport` dataclass (review output schema)
6. **`config.py`** framework constants — e.g. retry limits (do not change unless explicitly required)

**Suggested tuning scope**: prompt templates (`_SYSTEM_PROMPT_*`, etc.), `tools/*` implementations, and how data is assembled before LLM calls.

## Module ownership

| Workstream | Files | Main focus |
|---|---|---|
| 1 | `agents/manager.py`, `tools/arxiv_tool.py` | RAP / planning and retrieval |
| 2 | `agents/data_cleaning.py`, `agents/feature_engineering.py` | Data cleaning and feature-engineering code generation and self-heal |
| 3 | `agents/model_selection.py`, `agents/supervisor.py` | Model training and Supervisor review |
| 4 | `agents/business_translator.py`, `tools/file_reader.py` | Business insight and RAG |

(Replace the table with your course or team roster.)

## Development conventions

### Node function signatures

```python
def xxx_node(state: GraphState) -> dict:
    # Read fields needed for this stage from state
    # Run logic
    # Return a subset of fields to update (do not return the full state)
```

### Design principles

1. **Boundary isolation**: each agent only reads the sub-plan keys in `global_plan` that belong to it.
2. **Memory append-only**: when writing to `memory`, return `{"memory": ["new entry"]}`.
3. **Summary in memory, details on disk**: after Supervisor passes, write the full report under `./workspace/`; memory holds a short summary and paths.
4. **Prefer internal self-heal**: workers retry up to `MAX_INNER_RETRY` times before escalating to Supervisor.
5. **`TODO`**: marks places still wired to stub LLM or incomplete logic.

### Unified LLM usage

Prefer project helpers such as `core.llm.call_llm` / `get_llm` instead of constructing clients in every agent. If documentation still requires `ChatOpenAI` directly, keep it aligned with `LLM_CONFIG` in `config`.
