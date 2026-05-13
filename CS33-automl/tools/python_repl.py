"""
Python sandbox execution tool.
Runs agent-generated Python code in a subprocess and captures stdout and stderr.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import os


def execute_code(code: str, timeout: int = 300) -> tuple[str, str]:
    """
    Write code to a temporary file and run it in a subprocess.

    Args:
        code: Python source to execute
        timeout: Maximum runtime in seconds

    Returns:
        (stdout, stderr) tuple.
        An empty stderr string means success.
    """
    # Write a temporary .py file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="agent_exec_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(code)

        # Agents often emit pd.to_datetime etc.; pandas may write UserWarning/FutureWarning to stderr.
        # Returning stderr verbatim can make data_cleaning treat "warnings only" as failure.
        # Ignore common warning classes in the child process; real Tracebacks/exceptions still go to stderr.
        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "ignore::UserWarning",
                "-W",
                "ignore::FutureWarning",
                "-W",
                "ignore::DeprecationWarning",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.dirname(__file__)),  # project root
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        # Treat non-zero exit status as execution failure.
        if result.returncode != 0:
            return stdout, stderr

        # Successful execution may still emit warnings to stderr.
        # Do not trigger self-heal for warnings; keep them in the execution log instead.
        if stderr:
            combined_stdout = stdout
            warning_block = f"[stderr warnings]\n{stderr}"
            if combined_stdout:
                combined_stdout = f"{combined_stdout}\n\n{warning_block}"
            else:
                combined_stdout = warning_block
            return combined_stdout, ""

        return stdout, ""

    except subprocess.TimeoutExpired:
        return "", f"[TimeoutError] Code execution exceeded the {timeout}-second limit."
    except Exception as e:
        return "", f"[ExecutionError] {str(e)}"
    finally:
        # Remove temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
