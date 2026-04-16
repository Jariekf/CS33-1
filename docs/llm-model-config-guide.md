# LLM Model Configuration Guide

This document explains how to update API keys, model versions, temperature values, and the mapping between agents and LLMs.

## 1) Fill API keys and model versions in `.env`

In the project root, edit `.env` and provide the values for the provider you want to use:

```env
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

You can also change `DEEPSEEK_MODEL` to another supported model, such as `deepseek-reasoner`.  
Likewise, OpenAI / Zhipu / Gemini / Qwen / MiniMax / Kimi each use their own `*_API_KEY`, `*_BASE_URL`, and `*_MODEL`.

## 2) Change model temperature

To update temperature, edit the corresponding `temperature` value in `core/my_llm.py`, for example:

- `llm_deepseek` temperature
- `llm_openai` temperature
- `llm_gemini` temperature

Each `llm_xxx` instance is configured independently.

## 3) Change which LLM each agent uses

To change the LLM assigned to each agent, edit `AGENT_PROFILE_MAP` in `core/llm.py` (currently around line 55):

```python
AGENT_PROFILE_MAP = {
    "manager": "deepseek",
    "data_cleaning": "qwen",
    "feature_engineering": "deepseek",
    "model_selection": "openai",
    "business_translator": "gemini",
    "supervisor": "openai",
}
```

Meaning: the left side is the agent name, and the right side is the profile name.  
That profile name is resolved in `PROFILE_LLM_MAP` to a concrete `llm_xxx` instance.

### Manager sub-stage routing

The manager agent now supports separate profiles for its 3 internal LLM calls:

- `manager_summary`: requirement summary + retrieval-slot extraction
- `manager_fusion`: multi-source RAP context fusion
- `manager_plan`: global plan generation

You do not need to edit Python for this. Set these optional environment variables in `.env`:

```env
MANAGER_SUMMARY_PROFILE=openai
MANAGER_FUSION_PROFILE=kimi
MANAGER_PLAN_PROFILE=qwen
```

Valid values are the profile names in `PROFILE_LLM_MAP`, for example:

- `openai`
- `qwen`
- `kimi`
- `deepseek`
- `gemini`
- `zhipu`
- `minimax`

If these variables are not set, manager sub-stages fall back to the normal `AGENT_PROFILE_MAP` defaults.

## 4) After making changes

- Save the files and rerun the pipeline.
- If a call fails, first check:
  - whether the corresponding `API_KEY` is set in `.env`
  - whether `BASE_URL` and `MODEL` match the selected provider
  - for manager combinations, whether `MANAGER_SUMMARY_PROFILE`, `MANAGER_FUSION_PROFILE`, and `MANAGER_PLAN_PROFILE` are set to valid profile names
  - whether each value in `AGENT_PROFILE_MAP` exists in `PROFILE_LLM_MAP`
