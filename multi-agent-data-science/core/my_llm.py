import os
import threading

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from core.env_utils import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASEURL,
    DEEPSEEK_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASEURL,
    OPENAI_MODEL,
    OPENAI_STRONG_MODEL,
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

# ── Token accumulator ────────────────────────────────────────────────────────

class _TokenAccumulator(BaseCallbackHandler):
    """Process-level singleton that counts tokens across all LLM calls.

    Works for any OpenAI-compatible provider (DeepSeek, Qwen, Kimi, Gemini via
    OpenAI proxy, etc.) because they all surface usage in ``llm_output`` or in
    ``usage_metadata`` on the AIMessage.  Both paths are tried so coverage is
    maximised even when one provider omits one format.
    """

    def __init__(self) -> None:
        self._lock       = threading.Lock()
        self.prompt      = 0
        self.completion  = 0
        self.total       = 0

    def reset(self) -> None:
        with self._lock:
            self.prompt = self.completion = self.total = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:  # type: ignore[override]
        p = c = t = 0

        # Path A: llm_output dict — most OpenAI-compatible providers
        usage = (response.llm_output or {}).get("token_usage") or {}
        p += int(usage.get("prompt_tokens",     0))
        c += int(usage.get("completion_tokens", 0))
        t += int(usage.get("total_tokens",      0))

        # Path B: usage_metadata on each AIMessage — newer LangChain / Gemini
        if t == 0:
            for gen_list in response.generations:
                for gen in gen_list:
                    meta = getattr(
                        getattr(gen, "message", None), "usage_metadata", None
                    ) or {}
                    p += int(meta.get("input_tokens",  0))
                    c += int(meta.get("output_tokens", 0))
                    t += int(meta.get("total_tokens",  0))

        with self._lock:
            self.prompt     += p
            self.completion += c
            self.total      += t


TOKEN_COUNTER = _TokenAccumulator()

# ── LLM instances ─────────────────────────────────────────────────────────────

_CB = [TOKEN_COUNTER]  # shared callback list for all instances

llm_deepseek = ChatOpenAI(
    model=DEEPSEEK_MODEL or "deepseek-chat",
    temperature=0.7,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASEURL,
    callbacks=_CB,
)

llm_openai = ChatOpenAI(
    model=OPENAI_MODEL or "gpt-4o-mini",
    temperature=0.5,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASEURL,
    callbacks=_CB,
)

llm_openai_strong = ChatOpenAI(
    model=OPENAI_STRONG_MODEL or "gpt-4o",
    temperature=0.3,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASEURL,
    callbacks=_CB,
)

llm_zhipu = ChatOpenAI(
    model=ZHIPU_MODEL or "glm-4-plus",
    temperature=0.7,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASEURL,
    callbacks=_CB,
)

llm_gemini = ChatOpenAI(
    model=GEMINI_MODEL or "gemini-2.5-pro",
    temperature=0.7,
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASEURL,
    callbacks=_CB,
)

llm_gemini_flash = ChatOpenAI(
    model=GEMINI_FLASH_MODEL or GEMINI_MODEL or "gemini-2.0-flash",
    temperature=0.7,
    api_key=GEMINI_FLASH_API_KEY or GEMINI_API_KEY,
    base_url=GEMINI_FLASH_BASEURL or GEMINI_BASEURL,
    callbacks=_CB,
)

llm_qwen = ChatOpenAI(
    model=QWEN_MODEL or "qwen-plus",
    temperature=0.7,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASEURL,
    callbacks=_CB,
)

llm_minimax = ChatOpenAI(
    model=MINIMAX_MODEL or "MiniMax-Text-01",
    temperature=0.7,
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASEURL,
    callbacks=_CB,
)

llm_kimi = ChatOpenAI(
    model=KIMI_MODEL or "moonshot-v1-8k",
    temperature=0.7,
    api_key=KIMI_API_KEY,
    base_url=KIMI_BASEURL,
    callbacks=_CB,
)


def _normalize_pipeline_choice(raw: str | None) -> str:
    s = (raw or "deepseek").strip().lower().replace("-", "_")
    if s == "flash":
        return "gemini_flash"
    return s


# Set PIPELINE_LLM_SUPERVISOR env var to pick a cheaper/faster backend for the supervisor
# (e.g. "gemini_flash"). Falls back to PIPELINE_LLM if not set.
_PIPELINE_REGISTRY: dict[str, ChatOpenAI] = {
    "deepseek":      llm_deepseek,
    "openai":        llm_openai,         # OPENAI_MODEL        (default: gpt-4o-mini)
    "openai_strong": llm_openai_strong,  # OPENAI_STRONG_MODEL (default: gpt-4o)
    "zhipu":         llm_zhipu,
    "gemini":        llm_gemini,
    "gemini_flash":  llm_gemini_flash,
    "qwen":          llm_qwen,
    "minimax":       llm_minimax,
    "kimi":          llm_kimi,
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
