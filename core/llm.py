"""
公共 LLM 调用模块
提供统一的 LLM 初始化、调用与代码提取功能，供所有 Agent 节点复用。
设计参照 AutoKaggle 的 llm.py + agent_base._generate_code_file 的代码提取逻辑。
"""
from __future__ import annotations

import logging
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import LLM_CONFIG

logger = logging.getLogger(__name__)


# ======================================================================
# LLM 实例工厂
# ======================================================================

import os

def get_llm() -> ChatOpenAI:
    """
    根据 config.LLM_CONFIG 返回 ChatOpenAI 实例。
    若 API Key 缺失则记录警告。
    """
    if os.getenv("STUB_MODE", "false").lower() == "true":
        raise Exception("STUB_MODE 开启，强制生成带异常的 LLM 对象触发降级")

    if not LLM_CONFIG.get("api_key"):
        logger.warning("OPENAI_API_KEY 未设置，LLM 调用将会失败并降级为桩逻辑。")

    return ChatOpenAI(
        model=LLM_CONFIG["model"],
        api_key=LLM_CONFIG["api_key"],
        base_url=LLM_CONFIG["base_url"],
        temperature=LLM_CONFIG.get("temperature", 0.2),
    )


# ======================================================================
# 便捷调用函数
# ======================================================================

def call_llm(system_prompt: str, user_prompt: str) -> str | None:
    """
    发起一次 LLM 调用。
    """
    if os.getenv("STUB_MODE", "false").lower() == "true":
        logger.info("STUB_MODE 开启，跳过 LLM 调用，直接返回 None 触发降级。")
        return None

    try:
        llm = get_llm()
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
