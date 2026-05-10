"""
公共 LLM 调用模块
提供统一的 LLM 初始化、调用与代码提取功能，供所有 Agent 节点复用。
设计参照 AutoKaggle 的 llm.py + agent_base._generate_code_file 的代码提取逻辑。
"""
from __future__ import annotations

import logging
import os
import re
import json
import time
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
from core.env_utils import DEBUG_TRACE_LOG_PATH, DEBUG_TRACE_SESSION_ID

logger = logging.getLogger(__name__)

# Debug trace configuration:
# - Disabled by default in shared environments.
# - Enable by setting DEBUG_TRACE_LOG_PATH (required) and optional DEBUG_TRACE_SESSION_ID.
_DEBUG_TRACE_LOG_PATH = DEBUG_TRACE_LOG_PATH
_DEBUG_TRACE_SESSION_ID = DEBUG_TRACE_SESSION_ID


def _write_debug_trace(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    """Append one NDJSON debug event for LLM-call diagnostics."""
    if not _DEBUG_TRACE_LOG_PATH:
        return
    payload = {
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    if _DEBUG_TRACE_SESSION_ID:
        payload["sessionId"] = _DEBUG_TRACE_SESSION_ID
    try:
        with open(_DEBUG_TRACE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Debug logging must never affect main flow.
        pass


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

# AGENT_PROFILE_MAP: dict[str, str] = {
#     "manager": "openai", # Simply change the model name to the one you require
#     "manager_summary": "openai",
#     "manager_fusion": "openai",
#     "manager_plan": "openai",
#     "data_cleaning": "openai",
#     "feature_engineering": "openai",
#     "model_selection": "openai",
#     "business_translator": "openai",
#     "supervisor": "openai",
# }

AGENT_PROFILE_MAP = {
    "manager": "gemini_flash",
    "manager_summary": "gemini_flash",
    "manager_fusion": "gemini_flash",
    "manager_plan": "gemini_flash",
    "data_cleaning": "gemini_flash",
    "feature_engineering": "gemini_flash",
    "eda_analysis": "gemini_flash",
    "model_selection": "gemini_flash",
    "business_translator": "gemini_flash",
    "supervisor": "gemini_flash",
}


def _resolve_profile(agent_name: str | None = None) -> str:
    """Resolve profile with priority: agent map > default."""
    if agent_name:
        normalized = agent_name.strip().lower()
        # Allow manager's 3 internal LLM calls to be configured independently via env.
        # Example:
        #   MANAGER_SUMMARY_PROFILE=qwen
        #   MANAGER_FUSION_PROFILE=kimi
        #   MANAGER_PLAN_PROFILE=openai
        env_key = f"{normalized.upper()}_PROFILE"
        env_profile = os.getenv(env_key, "").strip().lower()
        # region agent log
        _write_debug_trace(
            run_id="pre-fix",
            hypothesis_id="H5",
            location="core/llm.py:_resolve_profile",
            message="resolve_profile_env_override_check",
            data={
                "agent_name": normalized,
                "env_key": env_key,
                "has_env_profile": bool(env_profile),
                "mapped_profile": AGENT_PROFILE_MAP.get(normalized, ""),
                "default_profile": DEFAULT_PROFILE,
            },
        )
        # endregion
        if env_profile:
            return env_profile
        # EDA deep analysis: default to same model as feature_engineering if set
        if normalized == "eda_analysis":
            fe = os.getenv("FEATURE_ENGINEERING_PROFILE", "").strip().lower()
            if fe:
                return fe
        return AGENT_PROFILE_MAP.get(normalized, DEFAULT_PROFILE)
    return DEFAULT_PROFILE


def get_llm(agent_name: str | None = None):
    """Return LLM instance by agent mapping."""
    if os.getenv("STUB_MODE", "false").lower() == "true":
        # region agent log
        _write_debug_trace(
            run_id="pre-fix",
            hypothesis_id="H4",
            location="core/llm.py:get_llm",
            message="stub_mode_blocked_llm",
            data={"agent_name": agent_name or "", "stub_mode": True},
        )
        # endregion
        raise Exception("STUB_MODE is enabled; forcing LLM object failure to trigger fallback.")

    resolved_profile = _resolve_profile(agent_name=agent_name)
    llm = PROFILE_LLM_MAP.get(resolved_profile)
    # region agent log
    _write_debug_trace(
        run_id="pre-fix",
        hypothesis_id="H2",
        location="core/llm.py:get_llm",
        message="llm_profile_resolved",
        data={
            "agent_name": (agent_name or "").strip().lower(),
            "resolved_profile": resolved_profile,
            "profile_known": llm is not None,
        },
    )
    # endregion
    if llm is None:
        logger.warning(
            "Unknown profile '%s'; falling back to default profile '%s'.",
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
        logger.info("STUB_MODE is enabled; skipping LLM call and returning None to trigger fallback.")
        # region agent log
        _write_debug_trace(
            run_id="pre-fix",
            hypothesis_id="H4",
            location="core/llm.py:call_llm",
            message="call_llm_skipped_due_to_stub_mode",
            data={"agent_name": agent_name or "", "stub_mode": True},
        )
        # endregion
        return None

    try:
        # region agent log
        _write_debug_trace(
            run_id="pre-fix",
            hypothesis_id="H1",
            location="core/llm.py:call_llm",
            message="call_llm_enter",
            data={
                "agent_name": agent_name or "",
                "system_prompt_len": len(system_prompt or ""),
                "user_prompt_len": len(user_prompt or ""),
            },
        )
        # endregion
        llm = get_llm(agent_name=agent_name)
        llm_profile = _resolve_profile(agent_name=agent_name)
        key_env_name = f"{llm_profile.upper()}_API_KEY"
        base_env_name = f"{llm_profile.upper()}_BASE_URL"
        has_key = bool(os.getenv(key_env_name, ""))
        has_base = bool(os.getenv(base_env_name, ""))
        # region agent log
        _write_debug_trace(
            run_id="pre-fix",
            hypothesis_id="H1",
            location="core/llm.py:call_llm",
            message="llm_request_config_presence",
            data={
                "agent_name": agent_name or "",
                "llm_profile": llm_profile,
                "key_env_name": key_env_name,
                "base_env_name": base_env_name,
                "has_api_key": has_key,
                "has_base_url": has_base,
            },
        )
        # endregion
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        # region agent log
        _write_debug_trace(
            run_id="pre-fix",
            hypothesis_id="H3",
            location="core/llm.py:call_llm",
            message="llm_invoke_exception",
            data={
                "agent_name": agent_name or "",
                "error_type": type(e).__name__,
                "error_msg_preview": str(e)[:300],
            },
        )
        # endregion
        logger.error("LLM call failed: %s", e)
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
        logger.warning("No ```python ... ``` code block found in LLM reply.")
        return None

    # 多段代码拼接 (与 AutoKaggle 一致)
    code = "\n".join(match.strip() for match in matches)
    return code
