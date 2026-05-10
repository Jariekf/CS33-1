"""
文件读取工具
为 Agent 提供按需读取本地落盘文件的能力 (Read_File_Tool)。
用于 Business Translator 等后置节点从硬盘读取报告或 style skeletons。
"""
from __future__ import annotations


def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    读取本地文件内容并返回字符串。

    Args:
        file_path: 文件绝对或相对路径
        encoding: 文件编码，默认 UTF-8

    Returns:
        文件内容字符串

    Raises:
        FileNotFoundError: 文件不存在
    """
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()
