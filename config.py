"""
全局配置文件
定义 LLM 模型、API Key、重试策略等核心参数。
"""
import os
from dotenv import load_dotenv

load_dotenv()  # 从项目根目录的 .env 文件加载环境变量

# ============================================================
# LLM 配置
# ============================================================
LLM_CONFIG = {
    "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "temperature": 0.7,
}

# ============================================================
# 重试策略
# ============================================================
# Worker 节点内部语法自救的最大重试次数
MAX_INNER_RETRY = 3
# Supervisor 审查不通过后，全局重试的最大次数
MAX_OUTER_RETRY = 3
# Supervisor 通过的最低分数阈值 (1-5 分制)
SUPERVISOR_PASS_THRESHOLD = 3

# ============================================================
# 工作空间路径
# ============================================================
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")

# ============================================================
# Arxiv 检索配置
# ============================================================
ARXIV_MAX_RESULTS = 5
