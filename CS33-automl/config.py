"""
Global configuration file.
Defines core parameters such as LLM models, API keys, and retry policy.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from the project-root .env file

# ============================================================
# LLM configuration
# ============================================================
LLM_CONFIG = {
    "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "temperature": 0.7,
}

# ============================================================
# Retry policy
# ============================================================
# Maximum retries for a worker node's internal syntax self-recovery
MAX_INNER_RETRY = 3
# Maximum global retries after the supervisor rejects a result
MAX_OUTER_RETRY = 3
# Minimum score threshold for the supervisor to pass (1-5 scale)
SUPERVISOR_PASS_THRESHOLD = 3

# ============================================================
# Workspace and log paths
# ============================================================
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
# Markdown report written during the EDA stage (kept in sync with tools.eda_charts / agents.eda_analysis)
EDA_REPORT_PATH = os.path.join(WORKSPACE_DIR, "eda_report.md")

# ============================================================
# Arxiv search configuration
# ============================================================
ARXIV_MAX_RESULTS = 5

# ============================================================
# Manager RAP lightweight multi-source retrieval (see requirements.txt for optional deps)
# Web: usually requires no API key (duckduckgo-search); Kaggle: ~/.kaggle/kaggle.json;
# PWC: no API; requires a local snapshot JSON (see PWC_DATA_DIR below). The directory may
# be empty or contain only .gitkeep before files are added — this does not break the run.
# ============================================================
RAP_ENABLED = os.getenv("RAP_ENABLED", "false").lower() in ("1", "true", "yes")
# Optional sources are disabled by default; enable them via .env, e.g. RAP_WEB=true
RAP_WEB_ENABLED = os.getenv("RAP_WEB", "false").lower() in ("1", "true", "yes")
RAP_PWC_ENABLED = os.getenv("RAP_PWC", "false").lower() in ("1", "true", "yes")
RAP_KAGGLE_ENABLED = os.getenv("RAP_KAGGLE", "false").lower() in ("1", "true", "yes")
WEB_MAX_RESULTS = int(os.getenv("RAP_WEB_MAX_RESULTS", "5"))
RAP_FUSE_MAX_CHARS = int(os.getenv("RAP_FUSE_MAX_CHARS", "3500"))
# Requires datasets.json and evaluation-tables.json (PapersWithCode export schema);
# if either is missing, the retrieval logic silently skips this source.
PWC_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "paperswithcode")
