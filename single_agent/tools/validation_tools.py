"""Validation and JSON artifact helpers.

These functions provide deterministic checks before Supervisor-level review.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd


def ensure_workspace(path: str = "workspace") -> None:
    """Create the workspace directory if it does not exist."""

    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Any) -> None:
    """Write a JSON-serializable payload as formatted JSON."""

    ensure_workspace(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path: str) -> dict[str, Any]:
    """Read a JSON object from disk."""

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_csv(path: str, min_rows: int = 1) -> tuple[bool, str]:
    """Validate that a CSV exists, is readable, and has enough rows."""

    if not os.path.exists(path):
        return False, f"missing CSV artifact: {path}"
    if os.path.getsize(path) == 0:
        return False, f"empty CSV artifact: {path}"
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return False, f"unreadable CSV artifact {path}: {exc}"
    if len(df) < min_rows:
        return False, f"CSV artifact {path} has fewer than {min_rows} rows"
    if df.columns.empty:
        return False, f"CSV artifact {path} has no columns"
    return True, ""


def validate_json_object(path: str) -> tuple[bool, str]:
    """Validate that a JSON artifact exists and contains an object."""

    if not os.path.exists(path):
        return False, f"missing JSON artifact: {path}"
    if os.path.getsize(path) == 0:
        return False, f"empty JSON artifact: {path}"
    try:
        read_json(path)
    except Exception as exc:
        return False, f"invalid JSON artifact {path}: {exc}"
    return True, ""


def validate_matching_columns(train_path: str, test_path: str) -> tuple[bool, str]:
    """Validate that two CSV files have identical column order."""

    ok, msg = validate_csv(train_path)
    if not ok:
        return ok, msg
    ok, msg = validate_csv(test_path)
    if not ok:
        return ok, msg
    train_cols = list(pd.read_csv(train_path, nrows=0).columns)
    test_cols = list(pd.read_csv(test_path, nrows=0).columns)
    if train_cols != test_cols:
        return False, "train/test columns do not match"
    return True, ""
