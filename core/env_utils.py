import os
from dotenv import load_dotenv

load_dotenv(override=True)

# api keys
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
KIMI_API_KEY = os.getenv("KIMI_API_KEY")

# base url
DEEPSEEK_BASEURL = os.getenv("DEEPSEEK_BASE_URL")
OPENAI_BASEURL = os.getenv("OPENAI_BASE_URL")
ZHIPU_BASEURL = os.getenv("ZHIPU_BASE_URL")
GEMINI_BASEURL = os.getenv("GEMINI_BASE_URL")
QWEN_BASEURL = os.getenv("QWEN_BASE_URL")
MINIMAX_BASEURL = os.getenv("MINIMAX_BASE_URL")
KIMI_BASEURL = os.getenv("KIMI_BASE_URL")

# model
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
QWEN_MODEL = os.getenv("QWEN_MODEL")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL")
KIMI_MODEL = os.getenv("KIMI_MODEL")
