"""Train/test split (no LLM): stratified sampling for classification, random split for regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from core.global_state import GlobalState
from core.report_writer import log_node


def _resolve_source_csv(state: GlobalState) -> Path | None:
    cleaned = (state.get("cleaned_data_path") or "").strip()
    if cleaned:
        p = Path(cleaned).expanduser().resolve()
        if p.is_file():
            return p
    raw = (state.get("data_path") or "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            return p
    return None


def _default_output_paths(source: Path) -> tuple[Path, Path]:
    out_dir = source.parent / "workspace"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    train_path = out_dir / f"train_{stem}.csv"
    test_path = out_dir / f"test_{stem}.csv"
    return train_path, test_path


def perform_train_test_file_split(
    *,
    source_path: Path,
    train_out: Path,
    test_out: Path,
    task_type: str,
    target_column: str | None,
    test_size: float = 0.2,
    random_state: int = 331,
) -> dict[str, Any]:
    df = pd.read_csv(source_path, sep=None, engine="python")
    stratify = None
    stratify_requested = task_type == "classification"
    stratify_fallback = False

    if stratify_requested:
        if not target_column or target_column not in df.columns:
            raise ValueError(
                "classification split requires a valid target_column that must exist in the CSV."
            )
        y = df[target_column]
        try:
            stratify = y
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )
            stratify_fallback = True
            stratify = None
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    train_out = Path(train_out).expanduser().resolve()
    test_out = Path(test_out).expanduser().resolve()
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    report: dict[str, Any] = {
        "source_path": str(source_path.resolve()),
        "task_type": task_type,
        "target_column": target_column or "",
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_path": str(train_out),
        "test_path": str(test_out),
        "stratify_requested": stratify_requested,
        "stratify_used": stratify is not None,
        "stratify_fallback_to_random": stratify_fallback,
    }
    return report


def split_dataset_node(state: GlobalState, *, test_size: float = 0.2, random_state: int = 331) -> GlobalState:
    """LangGraph node: reads task_type, writes training_data_path / testing_data_path and split_report."""
    task_type = (state.get("task_type") or "unknown").strip().lower()
    target_column = (state.get("target_column") or "").strip() or None

    if task_type not in {"classification", "regression", "unknown"}:
        task_type = "unknown"

    source = _resolve_source_csv(state)
    if source is None:
        err = {
            "stage": "split",
            "type": "FileNotFoundError",
            "message": "No usable dataset file found (prefers cleaned_data_path, falls back to data_path).",
        }
        log_node("split", success=False, detail="source file not found")
        return {
            "current_stage": "split_failed",
            "errors": (state.get("errors") or []) + [err],
            "split_report": {"error": err},
        }

    run_dir = (state.get("run_dir") or "").strip()
    if run_dir:
        out_dir = Path(run_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        train_out = out_dir / f"train_{source.stem}.csv"
        test_out = out_dir / f"test_{source.stem}.csv"
    else:
        train_out, test_out = _default_output_paths(source)

    try:
        if task_type == "classification":
            report = perform_train_test_file_split(
                source_path=source,
                train_out=train_out,
                test_out=test_out,
                task_type="classification",
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
            )
        else:
            # regression / unknown: random split of the full table
            report = perform_train_test_file_split(
                source_path=source,
                train_out=train_out,
                test_out=test_out,
                task_type="regression",
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
            )
    except Exception as exc:
        err = {"stage": "split", "type": exc.__class__.__name__, "message": str(exc)}
        log_node("split", success=False, detail=str(exc))
        return {
            "current_stage": "split_failed",
            "errors": (state.get("errors") or []) + [err],
            "split_report": {"error": err},
        }

    log_node(
        "split",
        success=True,
        detail=f"train={report['train_rows']} rows  test={report['test_rows']} rows",
    )
    return {
        "training_data_path": report["train_path"],
        "testing_data_path": report["test_path"],
        "split_report": report,
        "current_stage": "split_done",
        "next_action": "",
    }
