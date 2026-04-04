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
# 工作空间与日志路径
# ============================================================
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# ============================================================
# Arxiv 检索配置
# ============================================================
ARXIV_MAX_RESULTS = 5

# ============================================================
# Manager RAP 多源轻量检索（可选依赖见 requirements.txt）
# Web：一般无需 API Key（duckduckgo-search）；Kaggle：~/.kaggle/kaggle.json；
# PWC：无 API，仅本地 JSON（见 PWC_DATA_DIR）；arXiv：无 Key，由 ARXIV_MAX_RESULTS 控制条数。
# ============================================================
# 默认关闭 optional 源；需要时在 .env 中设 RAP_WEB=true 等开启
RAP_WEB_ENABLED = os.getenv("RAP_WEB", "false").lower() in ("1", "true", "yes")
RAP_PWC_ENABLED = os.getenv("RAP_PWC", "false").lower() in ("1", "true", "yes")
RAP_KAGGLE_ENABLED = os.getenv("RAP_KAGGLE", "false").lower() in ("1", "true", "yes")
WEB_MAX_RESULTS = int(os.getenv("RAP_WEB_MAX_RESULTS", "5"))
RAP_FUSE_MAX_CHARS = int(os.getenv("RAP_FUSE_MAX_CHARS", "3500"))
PWC_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "paperswithcode")
