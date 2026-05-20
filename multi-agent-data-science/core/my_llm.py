import os

from langchain_openai import ChatOpenAI

from core.env_utils import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASEURL,
    DEEPSEEK_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASEURL,
    OPENAI_MODEL,
    ZHIPU_API_KEY,
    ZHIPU_BASEURL,
    ZHIPU_MODEL,
    GEMINI_API_KEY,
    GEMINI_BASEURL,
    GEMINI_MODEL,
    GEMINI_FLASH_API_KEY,
    GEMINI_FLASH_BASEURL,
    GEMINI_FLASH_MODEL,
    QWEN_API_KEY,
    QWEN_BASEURL,
    QWEN_MODEL,
    MINIMAX_API_KEY,
    MINIMAX_BASEURL,
    MINIMAX_MODEL,
    KIMI_API_KEY,
    KIMI_BASEURL,
    KIMI_MODEL,
)

llm_deepseek = ChatOpenAI(
    model=DEEPSEEK_MODEL or "deepseek-chat",
    temperature=0.7,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASEURL,
)

llm_openai = ChatOpenAI(
    model=OPENAI_MODEL or "gpt-4o",
    temperature=0.7,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASEURL,
)

llm_zhipu = ChatOpenAI(
    model=ZHIPU_MODEL or "glm-4-plus",
    temperature=0.7,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASEURL,
)

llm_gemini = ChatOpenAI(
    model=GEMINI_MODEL or "gemini-2.5-pro",
    temperature=0.7,
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASEURL,
)

llm_gemini_flash = ChatOpenAI(
    model=GEMINI_FLASH_MODEL or GEMINI_MODEL or "gemini-2.0-flash",
    temperature=0.7,
    api_key=GEMINI_FLASH_API_KEY or GEMINI_API_KEY,
    base_url=GEMINI_FLASH_BASEURL or GEMINI_BASEURL,
)

llm_qwen = ChatOpenAI(
    model=QWEN_MODEL or "qwen-plus",
    temperature=0.7,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASEURL,
)

llm_minimax = ChatOpenAI(
    model=MINIMAX_MODEL or "MiniMax-Text-01",
    temperature=0.7,
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASEURL,
)

llm_kimi = ChatOpenAI(
    model=KIMI_MODEL or "moonshot-v1-8k",
    temperature=0.7,
    api_key=KIMI_API_KEY,
    base_url=KIMI_BASEURL,
)


def _normalize_pipeline_choice(raw: str | None) -> str:
    s = (raw or "deepseek").strip().lower().replace("-", "_")
    if s == "flash":
        return "gemini_flash"
    return s


# Set PIPELINE_LLM_SUPERVISOR env var to pick a cheaper/faster backend for the supervisor
# (e.g. "gemini_flash"). Falls back to PIPELINE_LLM if not set.
_PIPELINE_REGISTRY: dict[str, ChatOpenAI] = {
    "deepseek": llm_deepseek,
    "openai": llm_openai,
    "zhipu": llm_zhipu,
    "gemini": llm_gemini,
    "gemini_flash": llm_gemini_flash,
    "qwen": llm_qwen,
    "minimax": llm_minimax,
    "kimi": llm_kimi,
}


def _resolve_backend_for_agent(agent_key: str) -> str:
    """Map subgraph key → backend name (e.g. ``openai`` / ``deepseek``).

    Precedence:
      ``PIPELINE_LLM_<KEY>`` (KEY = agent token uppercased, e.g. ``MODEL``) >
      fallback ``PIPELINE_LLM``.
    Tokens ``default`` / ``pipeline`` / empty only read global ``PIPELINE_LLM``.
    """
    k = agent_key.strip().upper().replace("-", "_")
    if k in {"", "DEFAULT", "PIPELINE"}:
        return _normalize_pipeline_choice(os.getenv("PIPELINE_LLM"))
    specific = os.getenv(f"PIPELINE_LLM_{k}")
    if specific and specific.strip():
        return _normalize_pipeline_choice(specific)
    return _normalize_pipeline_choice(os.getenv("PIPELINE_LLM"))


def llm_for(agent_key: str) -> ChatOpenAI:
    """Chat model for a LangGraph subgraph. See ``PIPELINE_LLM_*`` env vars."""
    name = _resolve_backend_for_agent(agent_key)
    inst = _PIPELINE_REGISTRY.get(name)
    if inst is None:
        known = ", ".join(sorted(_PIPELINE_REGISTRY))
        raise ValueError(
            f"Resolved unknown backend {name!r} for agent {agent_key!r}. "
            f"PIPELINE_LLM must be one of: {known}"
        )
    return inst


llm_pipeline = llm_for("default")


def call_llm(system_prompt: str, user_prompt: str, agent_name: str = "") -> str:
    """Single Chat call (aligned with CS33-automl call_llm), used by BAT and other nodes."""
    from langchain_core.messages import HumanMessage, SystemMessage

    _ = agent_name  # reserved for tracing / logging
    msg = llm_for("bat").invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    content = getattr(msg, "content", None)
    return str(content) if content is not None else ""
