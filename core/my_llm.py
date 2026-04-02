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
