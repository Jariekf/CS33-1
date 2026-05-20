"""
File reading tool.
Provides Agents with the ability to read local on-disk files on demand (Read_File_Tool).
Used by downstream nodes such as Business Translator to read reports or style skeletons from disk.
"""
from __future__ import annotations


def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read the contents of a local file and return them as a string.

    Args:
        file_path: absolute or relative path to the file
        encoding: file encoding; default UTF-8

    Returns:
        file contents as a string

    Raises:
        FileNotFoundError: file does not exist
    """
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()
