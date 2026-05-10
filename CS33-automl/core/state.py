"""
Global state definition (`GraphState`).

Shared dictionary for all LangGraph nodes. Uses `Annotated` plus `operator.add`
for append-only `memory`.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class GraphState(TypedDict):
    """Global LangGraph state for the workflow."""

    # ---- User input ----
    data_path: str                  # Path to user-uploaded dataset
    user_goal: str                  # Natural-language business goal
    # Optional hints for Manager retrieval/planning: cleaning rules / contract text, merged into user_requirement_summary
    manager_cleaning_hints: str
    # Authoritative dataset context (README, data dictionary, paper text, …), shared semantic anchor across agents
    dataset_description: str
    # Optional path to a user-held-out test CSV.
    #   - Empty string → single-CSV flow: ``data_path`` is the full dataset; data_cleaning does random 80/20 split.
    #   - Non-empty → pre-split flow: ``data_path`` is the train CSV; this field holds the separate test CSV;
    #     data_cleaning merges both with a hidden marker column, runs cleaning once, splits back into
    #     ``train_data.csv`` / ``test_data.csv``.
    # Only ``agents/data_cleaning.py`` reads this; other agents consume ``TRAIN_DATA_PATH`` / ``TEST_DATA_PATH`` in workspace.
    test_data_path: str
    # Supervised target column name (single source of truth).
    #   - Empty string = undetermined / unsupervised / Manager could not infer
    #   - Resolution order: user CLI --target > Manager inference > last column of data preview (CSV header)
    #   - Filled in ``agents/manager.py`` ``manager_node``; data_cleaning / feature_engineering / model_selection
    #     use ``state.get("target_column")`` so they stay aligned (no per-agent semantic guessing).
    target_column: str

    # ---- Manager output ----
    global_plan: dict               # Structured plan from Manager (RAP-assisted)
    #   Example shape:
    #   {
    #     "data_cleaning_plan": {...},
    #     "feature_engineering_plan": {...},
    #     "model_selection_plan": {...}
    #   }

    # ---- Runtime pipeline state ----
    current_phase: str              # e.g. "data_cleaning" / "feature_engineering" / "model_selection"
    current_data_path: str          # Raw input path for data_cleaning (first phase only).
    #   Downstream feature_engineering / model_selection always read their upstream's
    #   fixed output path (CLEANED_OUTPUT_PATH / FE_OUTPUT_PATH), so that a Supervisor
    #   retry does not re-feed a worker's own previous output back into itself.
    #   Worker nodes no longer mutate this field.
    current_code: str               # Latest worker-generated code
    current_execution_log: str      # Stdout after worker execution
    current_error: str              # Supervisor feedback on reject / stderr from inner retries
    retry_count: int                # Accumulated Supervisor retries this phase

    # ---- Supervisor output ----
    current_review: dict | None     # Latest structured ReviewReport from Supervisor

    # ---- Run-level metrics ----
    outer_retry_total: int          # Total outer retries triggered by Supervisor

    # ---- Global memory (append-only) ----
    memory: Annotated[list[str], operator.add]
    #   Each item: short summary + path pointer, e.g.
    #   "[data_cleaning] Handled high missing-rate columns; details in ./workspace/clean_report.md"

    # ---- Final deliverables ----
    eda_summary: str               # EDA narrative + chart references (markdown-like)
    eda_charts: list[dict]          # [{"name","rel_path","desc"}, ...] relative to workspace
    eda_conclusions: dict           # Deep EDA: task_type, candidate_models, recommended_metrics, warnings
    final_business_report: str       # Quarto/report output from Business Action Translator
