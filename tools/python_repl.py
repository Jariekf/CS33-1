"""
Python 沙盒执行工具
在子进程中安全执行 Agent 生成的 Python 代码，并捕获 Stdout / Stderr 双流。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import os


def execute_code(code: str, timeout: int = 120) -> tuple[str, str]:
    """
    将代码写入临时文件并在子进程中执行。

    Args:
        code: 要执行的 Python 代码字符串
        timeout: 最大执行时间（秒）

    Returns:
        (stdout, stderr) 元组。
        若 stderr 为空字符串，表示执行成功。
    """
    # 写入临时 .py 文件
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="agent_exec_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.dirname(__file__)),  # 项目根目录
        )

        return result.stdout.strip(), result.stderr.strip()

    except subprocess.TimeoutExpired:
        return "", f"[TimeoutError] 代码执行超过 {timeout} 秒限制"
    except Exception as e:
        return "", f"[ExecutionError] {str(e)}"
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
