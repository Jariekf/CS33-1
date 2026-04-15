"""
全局状态定义 (GraphState)
所有 LangGraph 节点共享的状态字典。
采用 Annotated + operator.add 实现 Memory 的 Append-Only 机制。
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class GraphState(TypedDict):
    """LangGraph 全局状态，贯穿整个工作流。"""

    # ---- 用户输入 ----
    data_path: str                  # 用户上传数据集的路径
    user_goal: str                  # 用户的自然语言业务目标
    # Manager 检索/规划前可选：队友或用户提供的清洗规则、contract 等文字，会并入 user_requirement_summary
    manager_cleaning_hints: str

    # ---- Manager 产出 ----
    global_plan: dict               # Manager 通过 RAP 生成的结构化执行计划
    #   结构示例:
    #   {
    #     "data_cleaning_plan": {...},
    #     "feature_engineering_plan": {...},
    #     "model_selection_plan": {...}
    #   }

    # ---- 流水线运行时状态 ----
    current_phase: str              # 当前阶段名 ("data_cleaning" / "feature_engineering" / "model_selection")
    current_data_path: str          # 当前阶段的数据输入路径 (文件交接机制)
    current_code: str               # Worker 最近一次生成的代码
    current_execution_log: str      # Worker 代码执行后的 Stdout
    current_error: str              # Supervisor 打回时附带的反馈 / 内部自救时的 Stderr
    retry_count: int                # 当前阶段 Supervisor 打回的累计次数

    # ---- Supervisor 审查产出 ----
    current_review: dict | None     # Supervisor 最新一次的结构化审查报告 (ReviewReport)

    # ---- 运行评测指标 ----
    outer_retry_total: int          # Supervisor 打回导致的累计外部重试次数

    # ---- 全局记忆 (Append-Only) ----
    memory: Annotated[list[str], operator.add]
    #   每一项是一段精简摘要 + 落盘文件路径指针，如:
    #   "[data_cleaning] 移除15%缺失值, 详情见 ./workspace/clean_report.md"

    # ---- 最终交付 ----
    final_business_report: str      # Business Action Translator 产出的商业分析报告
