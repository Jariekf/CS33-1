"""Minimal demo2 workflow runner.

Usage:
    python main.py <csv_path> "<user_goal>" ["data_description"]
    python main.py --data <csv_path> --goal "<user_goal>" [--description-text "..."]
    python main.py --train <train_csv> --test <test_csv> --goal "<user_goal>" [--description-file path]
"""

from __future__ import annotations

import argparse
import sys
from time import perf_counter

from agents.business_translator import business_translator_node
from agents.data_cleaning import data_cleaning_node
from agents.eda_analysis import eda_analysis_node
from agents.feature_decision import feature_decision_node
from agents.feature_engineering import feature_engineering_node
from agents.manager import manager_node, manager_replanning_checkpoint
from agents.model_selection import model_selection_node
from agents.supervisor import supervisor_node
from core.progress import log_step
from core.state_schema import WorkflowState


def run_workflow(
    data_path: str,
    user_goal: str,
    data_description: str = "",
    test_data_path: str | None = None,
) -> WorkflowState:
    """Run the initial deterministic multi-agent workflow."""

    workflow_start = perf_counter()
    state: WorkflowState = {
        "data_path": data_path,
        "user_goal": user_goal,
        "data_description": data_description,
        "input_mode": "provided_train_test" if test_data_path else "single_file_split",
    }
    if test_data_path:
        state["test_data_path"] = test_data_path
    _run_step(state, "manager: initial planning", lambda: manager_node(state))
    _run_step(state, "data_cleaning", lambda: data_cleaning_node(state))
    _run_step(state, "supervisor: data_cleaning", lambda: supervisor_node(state, "data_cleaning"))
    _raise_if_failed(state)
    _run_step(state, "manager: replan after data_cleaning", lambda: manager_replanning_checkpoint(state, "data_cleaning"))
    _run_step(state, "eda_analysis", lambda: eda_analysis_node(state))
    _run_step(state, "supervisor: eda_analysis", lambda: supervisor_node(state, "eda_analysis"))
    _raise_if_failed(state)
    _run_step(state, "feature_decision", lambda: feature_decision_node(state))
    _run_step(state, "feature_engineering", lambda: feature_engineering_node(state))
    _run_step(state, "supervisor: feature_engineering", lambda: supervisor_node(state, "feature_engineering"))
    _raise_if_failed(state)
    _run_step(state, "manager: replan after feature_engineering", lambda: manager_replanning_checkpoint(state, "feature_engineering"))
    _run_step(state, "model_selection", lambda: model_selection_node(state))
    _run_step(state, "supervisor: model_selection", lambda: supervisor_node(state, "model_selection"))
    _raise_if_failed(state)
    _run_step(state, "business_translator", lambda: business_translator_node(state))
    _run_step(state, "supervisor: business_translator", lambda: supervisor_node(state, "business_translator"))
    _raise_if_failed(state)
    print(f"[DONE]  workflow total ({perf_counter() - workflow_start:.2f}s)", flush=True)
    return state


def _run_step(state: WorkflowState, name: str, fn) -> None:
    """Run one workflow step, merge updates, and log its duration."""

    with log_step(name):
        _merge(state, fn())


def _merge(state: WorkflowState, updates: dict) -> None:
    """Update state in place."""

    state.update(updates)


def _raise_if_failed(state: WorkflowState) -> None:
    """Stop workflow execution if Supervisor reports a validation failure."""

    if state.get("current_error"):
        raise RuntimeError(state["current_error"])


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse flag-based CLI while preserving the original positional interface."""

    if argv and not argv[0].startswith("-"):
        if len(argv) < 2:
            raise SystemExit('Usage: python main.py <csv_path> "<user_goal>" ["data_description"]')
        return argparse.Namespace(
            data=argv[0],
            train=None,
            test=None,
            goal=argv[1],
            description_text=argv[2] if len(argv) > 2 else "",
            description_file=None,
        )
    parser = argparse.ArgumentParser(description="Run the demo2 ML workflow.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data", help="Single CSV path. The workflow will create train/test split.")
    input_group.add_argument("--train", help="Training CSV path when train/test are provided separately.")
    parser.add_argument("--test", help="Test CSV path used with --train.")
    parser.add_argument("--goal", required=True, help="User modelling or analysis goal.")
    parser.add_argument("--description-text", default="", help="Optional dataset description text.")
    parser.add_argument("--description-file", help="Optional path to a dataset description file.")
    args = parser.parse_args(argv)
    if args.train and not args.test:
        parser.error("--train requires --test")
    if args.test and not args.train:
        parser.error("--test requires --train")
    return args


def _load_description(args: argparse.Namespace) -> str:
    """Load optional dataset description from text and/or file."""

    parts = []
    if args.description_text:
        parts.append(args.description_text)
    if args.description_file:
        with open(args.description_file, "r", encoding="utf-8") as f:
            parts.append(f.read())
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    final_state = run_workflow(
        args.train or args.data,
        args.goal,
        _load_description(args),
        test_data_path=args.test,
    )
    print("Workflow completed.")
    print(f"Plan version: {final_state.get('plan_version')}")
    print(f"Business report: {final_state.get('business_report_path')}")
    print("Artifacts written to ./workspace")
