import logging
import os
import time
from config import LOG_DIR

# Use a timestamp for this run as the dedicated folder name
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOG_DIR, RUN_ID)

# Ensure the per-run log directory exists
os.makedirs(RUN_LOG_DIR, exist_ok=True)

# Unified log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_agent_logger(agent_name: str) -> logging.Logger:
    """
    Return a Logger for the given agent name.
    If agent_name is `sys`, logs go to the system-wide aggregate file.
    The logger writes to both the console and a local file named after the agent.
    """
    logger = logging.getLogger(agent_name)

    # Avoid duplicate handlers from repeated registration
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)

    # 1. Console handler (StreamHandler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Per-agent file handler (FileHandler)
    log_file_path = os.path.join(RUN_LOG_DIR, f"{agent_name}.log")
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 3. For non-system agents, also tee logs into the aggregate sys log
    if agent_name != "sys":
        sys_log_path = os.path.join(RUN_LOG_DIR, "sys.log")
        sys_file_handler = logging.FileHandler(sys_log_path, encoding='utf-8')
        sys_file_handler.setFormatter(formatter)
        logger.addHandler(sys_file_handler)

    return logger
