import logging
import os
import time
from config import LOG_DIR

# 确定本次运行的时间戳作为专属文件夹名称
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOG_DIR, RUN_ID)

# 确保本次运行的日志专属目录存在
os.makedirs(RUN_LOG_DIR, exist_ok=True)

# 定义统一的日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_agent_logger(agent_name: str) -> logging.Logger:
    """
    获取指定智能体名称的 Logger。
    如果获取 `sys` 名称的 logger，则写入系统级聚合文件。
    该 logger 会将日志输出到控制台和对应名称的本地文件中。
    """
    logger = logging.getLogger(agent_name)
    
    # 避免多个 handler 重复注册导致重复输出
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)

    # 1. 终端输出处理器 (StreamHandler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 单独文件写入处理器 (FileHandler)
    log_file_path = os.path.join(RUN_LOG_DIR, f"{agent_name}.log")
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 3. 如果不是系统日志(sys)，额外将日志注入到 sys 日志中以实现聚合
    if agent_name != "sys":
        sys_log_path = os.path.join(RUN_LOG_DIR, "sys.log")
        sys_file_handler = logging.FileHandler(sys_log_path, encoding='utf-8')
        sys_file_handler.setFormatter(formatter)
        logger.addHandler(sys_file_handler)

    return logger
