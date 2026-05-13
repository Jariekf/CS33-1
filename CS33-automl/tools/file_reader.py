"""
File reading tool.
Lets agents read on-disk files on demand (Read_File_Tool).
Used by downstream nodes such as Business Translator to load detailed Supervisor reports from disk.
"""
from __future__ import annotations


def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read a local file and return its contents as a string.

    Args:
        file_path: Absolute or relative path to the file
        encoding: File encoding; default UTF-8

    Returns:
        File contents as a string

    Raises:
        FileNotFoundError: If the file does not exist
    """
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()
