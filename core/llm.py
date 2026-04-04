"""
公共 LLM 调用模块
提供统一的 LLM 初始化、调用与代码提取功能，供所有 Agent 节点复用。
设计参照 AutoKaggle 的 llm.py + agent_base._generate_code_file 的代码提取逻辑。
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from core.my_llm import (
    llm_deepseek,
    llm_openai,
    llm_zhipu,
    llm_gemini,
    llm_gemini_flash,
    llm_qwen,
    llm_minimax,
    llm_kimi,
)

logger = logging.getLogger(__name__)


# Inject LLMs

PROFILE_LLM_MAP: dict[str, Any] = {
    "deepseek": llm_deepseek,
    "openai": llm_openai,
    "zhipu": llm_zhipu,
    "gemini": llm_gemini,
    "gemini_flash": llm_gemini_flash,
    "qwen": llm_qwen,
    "minimax": llm_minimax,
    "kimi": llm_kimi,
}
# PROFILE_LLM_MAP: dict[str, Any] = {
#     "deepseek": llm_deepseek,
#     "openai": llm_deepseek,
#     "zhipu": llm_deepseek,
#     "gemini": llm_deepseek,
#     "qwen": llm_deepseek,
#     "minimax": llm_deepseek,
#     "kimi": llm_deepseek,
# }

DEFAULT_PROFILE = "deepseek"

"""
LLM configuration mapping table.
Here you can define the llm models corresponding to different agents
"""

AGENT_PROFILE_MAP: dict[str, str] = {
    "manager": "deepseek", # Simply change the model name to the one you require
    "data_cleaning": "qwen",
    "feature_engineering": "deepseek",
    "model_selection": "openai",
    "business_translator": "gemini",
    "supervisor": "openai",
}

# AGENT_PROFILE_MAP = {
#     "manager": "gemini_flash",
#     "data_cleaning": "gemini_flash",
#     "feature_engineering": "gemini_flash",
#     "model_selection": "gemini_flash",
#     "business_translator": "gemini_flash",
#     "supervisor": "gemini_flash",
# }


def _resolve_profile(agent_name: str | None = None) -> str:
    """Resolve profile with priority: agent map > default."""
    if agent_name:
        return AGENT_PROFILE_MAP.get(agent_name.strip().lower(), DEFAULT_PROFILE)
    return DEFAULT_PROFILE


def get_llm(agent_name: str | None = None):
    """Return LLM instance by agent mapping."""
    if os.getenv("STUB_MODE", "false").lower() == "true":
        raise Exception("STUB_MODE 开启，强制生成带异常的 LLM 对象触发降级")

    resolved_profile = _resolve_profile(agent_name=agent_name)
    llm = PROFILE_LLM_MAP.get(resolved_profile)
    if llm is None:
        logger.warning(
            "未识别的 profile '%s'，自动回退到默认 profile '%s'。",
            resolved_profile,
            DEFAULT_PROFILE,
        )
        return PROFILE_LLM_MAP[DEFAULT_PROFILE]
    return llm


# ======================================================================
# 便捷调用函数
# ======================================================================

def call_llm(
    system_prompt: str,
    user_prompt: str,
    agent_name: str | None = None,
) -> str | None:
    """
    发起一次 LLM 调用。
    """
    if os.getenv("STUB_MODE", "false").lower() == "true":
        logger.info("STUB_MODE 开启，跳过 LLM 调用，直接返回 None 触发降级。")
        return None

    try:
        llm = get_llm(agent_name=agent_name)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error("LLM 调用失败: %s", e)
        return None


# ======================================================================
# 代码提取 (参照 AutoKaggle developer.py 的 regex 提取逻辑)
# ======================================================================

def extract_code(reply: str) -> str | None:
    """
    从 LLM 回复中提取 ```python ... ``` 代码块。
    参照 AutoKaggle developer._generate_code_file 中的:
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, raw_reply, re.DOTALL)

    若回复包含多个代码块，将它们拼接为一个完整脚本。

    Returns:
        提取到的代码字符串。若未找到代码块则返回 None。
    """
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, reply, re.DOTALL)

    if not matches:
        logger.warning("LLM 回复中未找到 ```python ... ``` 代码块。")
        return None

    # 多段代码拼接 (与 AutoKaggle 一致)
    code = "\n".join(match.strip() for match in matches)
    return code
