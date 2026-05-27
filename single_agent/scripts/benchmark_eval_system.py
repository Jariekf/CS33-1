"""
Single-Agent Benchmark Evaluation Script
-----------------------------------------
Computes the cross-system performance metrics that are also tracked by
MAC-ADS / AutoGluon / AutoKaggle, so all four systems can be compared
on the same footing:

  SR  = 1 if pipeline produced a best_model with valid test metrics, else 0
  NPS = normalised primary metric in [0, 1]
        (lower-is-better metrics like RMSLE → max(0, 1 - val))
  CS  = 0.5 * SR + 0.5 * NPS

Runtime and token cost are accepted as optional CLI arguments because
Single-Agent does not currently track them internally. Provide them
manually (e.g. wall-clock from `time python ...` + Anthropic / OpenAI
usage from API logs).

By design this script does NOT compute Plan Adherence, Semantic
Alignment or BRM. Those metrics are specific to the MAC-ADS design
(retrieval-augmented planning, McKinsey-style report skeletons,
supervisor-driven artifact production) and are not part of the
single-agent baseline, so reporting them cross-system would bias the
comparison.

Usage:
    python scripts/benchmark_eval_system.py \
        --dataset bank_churn \
        --workspace workspace/bank_churn \
        --case-id 20260525-1200 \
        --runtime 412.5 \
        --tokens-prompt 152000 \
        --tokens-completion 6000

Outputs one row per call to single_agent/artifacts/experiment_records.csv
(creates if missing). Same (case_id, dataset, system_name) replaces the
existing row instead of duplicating.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


# ─── Per-dataset configuration (aligned with MAC-ADS conventions) ─────────────

COMPETITION_CONFIG: dict[str, dict[str, Any]] = {
    "bank_churn": {
        "primary_key":     "roc_auc",
        "fallback_keys":   ["f1_weighted", "accuracy"],
        "lower_is_better": False,
    },
    "academic_success": {
        "primary_key":     "accuracy",
        "fallback_keys":   ["f1_weighted"],
        "lower_is_better": False,
    },
    "obesity_risks": {   # NB: single_agent workspace folder is "obesity"
        "primary_key":     "accuracy",
        "fallback_keys":   ["f1_weighted"],
        "lower_is_better": False,
    },
    "student_study": {
        "primary_key":     "accuracy",
        "fallback_keys":   ["f1_weighted"],
        "lower_is_better": False,
    },
    "house_prices": {    # NB: single_agent workspace folder is "house_price"
        "primary_key":     "rmsle",
        "fallback_keys":   ["rmse", "mae"],
        "lower_is_better": True,
    },
}

DATASET_META = {
    "bank_churn": {
        "name":           "Bank Churn",
        "source":         "AutoKaggle / Kaggle Competition",
        "task_type":      "Tabular Binary Classification",
        "industry":       "Banking / Customer Retention",
        "primary_metric": "ROC-AUC",
    },
    "academic_success": {
        "name":           "Academic Success",
        "source":         "AutoKaggle / Kaggle Competition",
        "task_type":      "Tabular Multi-class Classification",
        "industry":       "Education / Student Support",
        "primary_metric": "Accuracy",
    },
    "obesity_risks": {
        "name":           "Obesity Risk",
        "source":         "AutoKaggle / Kaggle Competition",
        "task_type":      "Tabular Multi-class Classification",
        "industry":       "Healthcare / Public Health",
        "primary_metric": "Accuracy",
    },
    "student_study": {
        "name":           "Student Study Performance",
        "source":         "UCI Machine Learning Repository",
        "task_type":      "Tabular Multi-class Classification",
        "industry":       "Education / Student Performance",
        "primary_metric": "Accuracy",
    },
    "house_prices": {
        "name":           "House Prices",
        "source":         "AutoKaggle / Kaggle Competition",
        "task_type":      "Tabular Regression",
        "industry":       "Real Estate / Housing Valuation",
        "primary_metric": "RMSLE",
    },
}


# ─── SR / NPS / CS ────────────────────────────────────────────────────────────

def _compute_sr_nps_cs(model_report: dict,
                       primary_key: str,
                       fallback_keys: list[str],
                       lower_is_better: bool) -> dict:
    """Identical formula to MAC-ADS benchmark_eval_system.py:_compute_sr_nps_cs."""
    best_model = (model_report.get("best_model")
                  or model_report.get("selected_model_id"))
    test_metrics: dict = model_report.get("test_metrics") or {}

    sr = 1 if (best_model and test_metrics) else 0

    primary_val = test_metrics.get(primary_key)
    used_key = primary_key
    if primary_val is None:
        for fk in fallback_keys:
            if fk in test_metrics:
                primary_val = test_metrics[fk]
                used_key = fk
                break

    if primary_val is None:
        return {"SR":           sr,
                "NPS":          None,
                "CS":           None,
                "primary_val":  None,
                "metric_used":  None,
                "all_metrics":  test_metrics,
                "warning":      (f"none of {[primary_key] + fallback_keys}"
                                 " found in test_metrics")}

    if lower_is_better:
        nps = max(0.0, 1.0 - float(primary_val))
    else:
        nps = float(primary_val)

    nps = round(nps, 4)
    cs = round(0.5 * sr + 0.5 * nps, 4)
    warning = None if used_key == primary_key else (
        f"used fallback metric '{used_key}' because '{primary_key}' missing")

    return {"SR":           sr,
            "NPS":          nps,
            "CS":           cs,
            "primary_val":  float(primary_val),
            "metric_used":  used_key,
            "all_metrics":  test_metrics,
            "warning":      warning}


# ─── Orchestration ────────────────────────────────────────────────────────────

def evaluate_run(dataset_key: str,
                 workspace_dir: str | Path,
                 case_id: str | None = None,
                 runtime_total_s: float | None = None,
                 tokens_prompt: int | None = None,
                 tokens_completion: int | None = None) -> tuple[dict, dict]:
    workspace_dir = Path(workspace_dir)
    if not workspace_dir.exists():
        raise FileNotFoundError(f"workspace_dir does not exist: {workspace_dir}")
    if dataset_key not in COMPETITION_CONFIG:
        raise ValueError(f"unknown dataset_key: {dataset_key!r}. "
                         f"Known: {list(COMPETITION_CONFIG)}")

    cfg = COMPETITION_CONFIG[dataset_key]
    meta = DATASET_META[dataset_key]

    mr_path = workspace_dir / "model_report.json"
    if mr_path.exists():
        with mr_path.open(encoding="utf-8") as f:
            model_report = json.load(f)
    else:
        model_report = {}

    perf = _compute_sr_nps_cs(model_report,
                              cfg["primary_key"], cfg["fallback_keys"],
                              cfg["lower_is_better"])

    # Auto-load run_metrics.json (written by single_agent/main.py) when CLI
    # arguments were not provided. CLI args still take precedence.
    metrics_path = workspace_dir / "run_metrics.json"
    if metrics_path.exists():
        try:
            with metrics_path.open(encoding="utf-8") as f:
                rm = json.load(f)
            if runtime_total_s is None:
                runtime_total_s = rm.get("runtime_total_s")
            if tokens_prompt is None:
                tokens_prompt = rm.get("tokens_prompt")
            if tokens_completion is None:
                tokens_completion = rm.get("tokens_completion")
        except (OSError, json.JSONDecodeError):
            pass

    case_id = case_id or datetime.now().strftime("%Y%m%d-%H%M")
    tokens_total = None
    if tokens_prompt is not None and tokens_completion is not None:
        tokens_total = tokens_prompt + tokens_completion

    record = {
        "case_id":                  case_id,
        "timestamp":                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir":                  str(workspace_dir.resolve()),
        "competition":              dataset_key,
        "system_name":              "Single-Agent",
        "system_type":              "single-agent LLM baseline",
        "dataset":                  meta["name"],
        "source":                   meta["source"],
        "task_type":                meta["task_type"],
        "industry":                 meta["industry"],
        "primary_metric_name":      meta["primary_metric"],
        "primary_val":              perf["primary_val"],
        "metric_used":              perf.get("metric_used"),
        "SR":                       perf["SR"],
        "NPS":                      perf["NPS"],
        "CS":                       perf["CS"],
        "runtime_total_s":          runtime_total_s,
        "tokens_prompt_total":      tokens_prompt,
        "tokens_completion_total":  tokens_completion,
        "tokens_total":             tokens_total,
        # Surface every test metric the single-agent pipeline did track,
        # so downstream analysis can pick a shared metric across systems
        # if the cross-system primary metric is unavailable here.
        "all_test_metrics_json":    json.dumps(perf["all_metrics"],
                                               ensure_ascii=False),
    }

    return record, {"performance": perf}


def upsert_csv(record: dict, csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    key = (record["case_id"], record["competition"], record["system_name"])
    rows = [r for r in rows
            if (r.get("case_id"), r.get("competition"),
                r.get("system_name")) != key]
    rows.append(record)

    fieldnames = list(record.keys())
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(detail: dict, runtime_s: float | None,
                  tokens_total: int | None) -> None:
    perf = detail["performance"]
    line = "─" * 60
    print(line)
    print(" Single-Agent Benchmark Evaluation")
    print(line)
    print(f" SR  = {perf['SR']}")
    print(f" NPS = {perf['NPS']}        (primary_val = {perf['primary_val']}"
          f", metric_used = {perf.get('metric_used')})")
    print(f" CS  = {perf['CS']}")
    if perf.get("warning"):
        print(f"   warning: {perf['warning']}")
    print(f" runtime_total_s = {runtime_s}")
    print(f" tokens_total    = {tokens_total}")
    print(f" all test_metrics tracked: {list(perf['all_metrics'].keys())}")
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-Agent benchmark evaluation "
                    "(performance-only, cross-system compatible).")
    parser.add_argument("--dataset", required=True,
                        choices=list(COMPETITION_CONFIG.keys()),
                        help="Dataset key (matches MAC-ADS conventions).")
    parser.add_argument("--workspace", required=True,
                        help="Path to single_agent/workspace/<dataset-folder>.")
    parser.add_argument("--case-id", default=None,
                        help="Run identifier (default: timestamp now).")
    parser.add_argument("--runtime", type=float, default=None,
                        help="Total wall-clock runtime in seconds (optional).")
    parser.add_argument("--tokens-prompt", type=int, default=None,
                        help="Total prompt tokens consumed by the run (optional).")
    parser.add_argument("--tokens-completion", type=int, default=None,
                        help="Total completion tokens emitted by the run (optional).")
    parser.add_argument("--csv-out",
                        default="single_agent/artifacts/experiment_records.csv",
                        help="Cumulative CSV output path.")
    args = parser.parse_args()

    record, detail = evaluate_run(
        dataset_key=args.dataset,
        workspace_dir=args.workspace,
        case_id=args.case_id,
        runtime_total_s=args.runtime,
        tokens_prompt=args.tokens_prompt,
        tokens_completion=args.tokens_completion,
    )

    print_summary(detail,
                  runtime_s=args.runtime,
                  tokens_total=record["tokens_total"])
    upsert_csv(record, args.csv_out)
    print(f" [saved] {args.csv_out}")


if __name__ == "__main__":
    main()
