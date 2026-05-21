"""Our System Benchmark Evaluation Script
------------------------------------------
Reads a completed pipeline run-dir, extracts all metrics from the state
snapshot (*_state.json), computes ACS and BAT Report Maturity scores from
run artifacts, and upserts a structured record into the shared experiment
log (JSONL + CSV).  The same (run_dir, competition, system_name) triple
overwrites the previous row instead of duplicating.

All metrics are read from the state snapshot; no separate test-label CSV
is required.  Run ``python benchmark_eval_system.py --help`` for CLI usage.
"""

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# ── Shared output files (same paths as benchmark_eval.py) ────────────────────

_ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "workspace"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RECORDS_FILE = str(_ARTIFACTS_DIR / "experiment_records.jsonl")
RECORDS_CSV  = str(_ARTIFACTS_DIR / "experiment_records.csv")

# ── Competition / dataset config ─────────────────────────────────────────────

COMPETITION_CONFIG = {
    "bank_churn": {
        "label_col":   "Exited",
        "id_col":      "id",
        "metric":      "auc",
        "primary_key": "roc_auc",
    },
    "house_prices": {
        "label_col":   "SalePrice",
        "id_col":      "Id",
        "metric":      "rmse_log",
        "primary_key": "rmsle",
    },
    "obesity_risks": {
        "label_col":   "NObeyesdad",
        "id_col":      "id",
        "metric":      "accuracy",
        "primary_key": "accuracy",
    },
    "classification_with_an_academic_success_dataset": {
        "label_col":   "Target",
        "id_col":      "id",
        "metric":      "accuracy",
        "primary_key": "accuracy",
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
    "house_prices": {
        "name":           "House Prices",
        "source":         "AutoKaggle / Kaggle Competition",
        "task_type":      "Tabular Regression",
        "industry":       "Real Estate / Housing Valuation",
        "primary_metric": "RMSLE",
    },
    "obesity_risks": {
        "name":           "Obesity Risk",
        "source":         "AutoKaggle / Kaggle Competition",
        "task_type":      "Tabular Multi-class Classification",
        "industry":       "Healthcare / Public Health",
        "primary_metric": "Accuracy",
    },
    "classification_with_an_academic_success_dataset": {
        "name":           "Academic Success",
        "source":         "AutoKaggle / Kaggle Competition",
        "task_type":      "Tabular Multi-class Classification",
        "industry":       "Education / Student Support",
        "primary_metric": "Accuracy",
    },
}

# Per-agent timing keys → display names
# split_agent is intentionally omitted: its runtime is merged into Agent Manager.
# supervisor has no standalone timer (routing node); business_translator is the
# correct state key for the BAT subgraph.
AGENT_TIMING_DISPLAY = {
    "agent_manager":        "Agent Manager",
    "eda_agent":            "EDA Agent",
    "model_agent":          "Model Agent",
    "business_translator":  "Business Translator",
}

# Flat CSV phase abbreviations
PHASE_ABBREV_SYSTEM = {
    "Agent Manager":       "manager",
    "Data Cleaning":       "data_clean",
    "EDA Agent":           "eda",
    "Feature Engineering": "feat_eng",
    "Model Agent":         "model",
    "Business Translator": "bat",
}

PHASE_PRINT_ORDER = [
    "Agent Manager", "Data Cleaning",
    "EDA Agent", "Feature Engineering", "Model Agent", "Business Translator",
]

# ── Token-usage loader ────────────────────────────────────────────────────────

def _load_token_usage(run_dir: str) -> dict | None:
    """Read token_usage.json saved by main.py after each pipeline run.

    Returns a dict with keys prompt / completion / total, or None if the file
    does not exist (e.g. run was executed before token tracking was added).
    """
    path = Path(run_dir) / "reports" / "token_usage.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── State-snapshot helpers ────────────────────────────────────────────────────

def _find_state_snapshot(run_dir: str) -> dict:
    """Return the newest *_state.json from {run_dir}/reports/."""
    reports = Path(run_dir) / "reports"
    snaps = sorted(reports.glob("*_state.json"))
    if not snaps:
        raise FileNotFoundError(f"No *_state.json found in {reports}")
    with open(snaps[-1], "r", encoding="utf-8") as f:
        return json.load(f)


# ── Timing parser ─────────────────────────────────────────────────────────────

def _parse_timing(timing: dict | None) -> dict:
    """Aggregate state.timing into named group runtimes (seconds)."""
    if not timing:
        return {"total_seconds": None, "by_phase": {}}

    pipeline_start = timing.get("pipeline_start")
    groups: dict[str, float] = {}

    for key, label in AGENT_TIMING_DISPLAY.items():
        entry = timing.get(key)
        if entry and "start" in entry and "end" in entry:
            groups[label] = round(entry["end"] - entry["start"], 1)

    # Merge split_agent time into Agent Manager (split is a lightweight prereq)
    split_entry = timing.get("split_agent")
    if split_entry and "start" in split_entry and "end" in split_entry:
        split_s = round(split_entry["end"] - split_entry["start"], 1)
        groups["Agent Manager"] = round(groups.get("Agent Manager", 0) + split_s, 1)

    # Data cleaning: sum data_clean_train + data_clean_test
    dc_total = 0.0
    for dc_key in ("data_clean_train", "data_clean_test", "data_clean"):
        entry = timing.get(dc_key)
        if entry and "start" in entry and "end" in entry:
            dc_total += entry["end"] - entry["start"]
    if dc_total > 0:
        groups["Data Cleaning"] = round(dc_total, 1)

    # Feature engineering: sum all feature_* keys
    feat_total = 0.0
    for k, v in timing.items():
        if k.startswith("feature_") and isinstance(v, dict) and "start" in v and "end" in v:
            feat_total += v["end"] - v["start"]
    if feat_total > 0:
        groups["Feature Engineering"] = round(feat_total, 1)

    # Total wall clock
    all_entries = [
        v for k, v in timing.items()
        if k != "pipeline_start" and isinstance(v, dict) and "start" in v and "end" in v
    ]
    total_s: float | None = None
    if pipeline_start and all_entries:
        total_s = round(max(v["end"] for v in all_entries) - pipeline_start, 1)

    return {"total_seconds": total_s, "by_phase": groups}


# ── Metrics extraction ────────────────────────────────────────────────────────

def _extract_metrics(state: dict, primary_key: str, comp_metric: str) -> tuple[dict, dict]:
    """Return (primary_metrics, additional_metrics) from state.best_model.metrics."""
    best = state.get("best_model") or {}
    metrics: dict = best.get("metrics") or {}

    if not metrics:
        mr = state.get("model_results") or {}
        bm_name = best.get("model_name") or ""
        if bm_name and bm_name in mr:
            metrics = (mr[bm_name].get("final_test_metrics") or
                       mr[bm_name].get("metrics") or {})

    primary_val = metrics.get(primary_key)
    primary = {primary_key: primary_val}
    additional = {
        k: v for k, v in metrics.items()
        if k != primary_key and not k.endswith("_skipped_reason")
    }
    return primary, additional


# ── ACS ──────────────────────────────────────────────────────────────────────

def _compute_sr_anps_acs(state: dict, primary_key: str, primary_val) -> dict:
    """
    SR   = 1 if pipeline produced a best_model, else 0.
    ANPS = normalised primary metric (0–1, higher = better).
           AUC / Accuracy → ANPS = primary_val directly.
           RMSLE (lower-is-better) → ANPS = max(0, 1 – primary_val).
    ACS  = 0.5 × SR + 0.5 × ANPS.
    """
    sr = 1 if (state.get("best_model") and primary_val is not None) else 0

    if primary_val is None:
        return {"SR": sr, "ANPS": None, "ACS": None}

    if primary_key == "rmsle":
        anps = round(max(0.0, 1.0 - float(primary_val)), 4)
    else:
        anps = round(float(primary_val), 4)

    return {"SR": sr, "ANPS": anps, "ACS": round(0.5 * sr + 0.5 * anps, 4)}


# ── Interpretability metrics ──────────────────────────────────────────────────

# 1) Plan Adherence ─────────────────────────────────────────────────────────

_PA_STATE_CHECKS = [
    ("split_train_path",    "training_data_path",       True),
    ("split_test_path",     "testing_data_path",        True),
    ("clean_train_passed",  "train_cleaning_passed",    False),
    ("clean_test_passed",   "test_cleaning_passed",     False),
    ("candidate_models",    "candidate_models",         False),
    ("feature_eng_plan",    "feature_engineering_plan", False),
    ("feature_dataset_ids", "feature_dataset_ids",      False),
    ("model_results",       "model_results",            False),
    ("best_model",          "best_model",               False),
    ("business_report",     "business_report",          False),
]

_PA_REPORT_CHECKS = {
    "report_agent_manager": "agent_manager_report.md",
    "report_data_clean":    "data_clean_agent_report.md",
    "report_eda":           "eda_agent_report.md",
    "report_feature":       "feature_agent_report.md",
    "report_model":         "model_agent_report.md",
    "report_business":      "business_insight_report.md",
}


def _plan_adherence(state: dict, run_dir: str) -> dict:
    """
    Check whether each pipeline handoff produced the expected state fields
    and that expected report files exist.
    Score = passed / total (0–1).
    """
    details: dict[str, bool] = {}

    for label, field, req_file in _PA_STATE_CHECKS:
        val = state.get(field)
        if val is None or (isinstance(val, (dict, list)) and not val):
            details[label] = False
        elif req_file and isinstance(val, str):
            details[label] = Path(val).exists()
        else:
            details[label] = True

    reports_dir = Path(run_dir) / "reports"
    for label, fname in _PA_REPORT_CHECKS.items():
        details[label] = (reports_dir / fname).exists()

    passed = sum(1 for v in details.values() if v)
    total  = len(details)
    return {"score": round(passed / total, 4), "passed": passed, "total": total,
            "detail": details}


# 2) Semantic Alignment Score ────────────────────────────────────────────────

_COLUMN_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s*[:：]", re.MULTILINE)


def _extract_description_columns(description_file: str) -> list[str]:
    """Parse a data_description.txt and return unique column names."""
    if not description_file or not Path(description_file).exists():
        return []
    text = Path(description_file).read_text(encoding="utf-8", errors="ignore")
    cols = _COLUMN_RE.findall(text)
    seen: set[str] = set()
    unique: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _semantic_alignment(state: dict, run_dir: str, description_file: str) -> dict:
    """
    Fraction of data-description column names that are "correctly processed"
    by the system — mentioned in at least one agent report/state text field
    OR explicitly handled in the cleaning plan / report (meaning the system
    correctly identified the column even if it was later dropped).

    score = |mentioned ∪ cleaned_ok| / |all description columns|
    """
    columns = _extract_description_columns(description_file)
    if not columns:
        return {
            "score": None,
            "note": "description_file not provided or no columns parsed",
            "columns_found": 0,
            "columns_total": 0,
        }

    # Free-text corpus: all *.md reports + state RAP / description text
    corpus_parts: list[str] = []
    reports_dir = Path(run_dir) / "reports"
    for md_file in reports_dir.glob("*.md"):
        try:
            corpus_parts.append(md_file.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            pass
    for field in ("user_requirement_summary", "manager_rap_context", "description"):
        val = state.get(field)
        if isinstance(val, str) and val.strip():
            corpus_parts.append(val)
    corpus = "\n".join(corpus_parts).lower()

    mentioned_set = {c for c in columns if c.lower() in corpus}

    # Structured cleaning corpus: JSON-serialised cleaning plan / report state fields.
    # Columns that were correctly dropped in cleaning appear here even if absent
    # from later EDA / feature / model reports.
    cleaning_text = ""
    for field in (
        "cleaning_plan", "train_cleaning_plan", "test_cleaning_plan",
        "cleaning_report", "train_cleaning_report", "test_cleaning_report",
    ):
        val = state.get(field)
        if val:
            try:
                cleaning_text += json.dumps(val, ensure_ascii=False) + "\n"
            except (TypeError, ValueError):
                cleaning_text += str(val) + "\n"
    cleaning_lower = cleaning_text.lower()
    cleaned_ok_set = {c for c in columns if c.lower() in cleaning_lower}

    aligned = mentioned_set | cleaned_ok_set
    missing = [c for c in columns if c not in aligned]
    score   = round(len(aligned) / len(columns), 4)

    return {
        "score":                  score,
        "columns_found":          len(aligned),
        "columns_total":          len(columns),
        "via_report_mention":     len(mentioned_set),
        "via_cleaning_plan_only": len(cleaned_ok_set - mentioned_set),
        "missing_sample":         missing[:10],
    }


# 3) BAT Report Maturity ─────────────────────────────────────────────────────
# Weighted three-layer score: L1 Evidence=0.3, L2 Interpretation=0.3, L3 Action=0.4
# Each layer has N binary sub-checks; layer_score = passed_in_layer / N.
# Final: brm_score = L1*0.3 + L2*0.3 + L3*0.4

_MCKINSEY_SECTION_PATTERNS = [
    r"executive\s+summary", r"key\s+finding", r"strategic\s+(implication|recommendation)",
    r"business\s+(insight|implication|recommendation)", r"action\s+(item|plan|recommendation)",
    r"next\s+step", r"conclusion",
]
_MCKINSEY_RE = re.compile("|".join(_MCKINSEY_SECTION_PATTERNS), re.IGNORECASE)

# L1 — Evidence layer regexes (l1_external_evidence is state-based, not text-based)
_L1_EDA_CHART_RE   = re.compile(r"plots[\\/][\w\-]+\.png", re.IGNORECASE)
_L1_MODEL_CHART_RE = re.compile(
    r"explain[\\/][\w\-]+\.png|shap[\w\-]*\.png|importance[\w\-]*\.png|rf_[\w\-]+\.png",
    re.IGNORECASE,
)

# L2 — Interpretation layer regexes
# Match the "## Strategic Implications" section heading (LLM consistently generates this).
_L2_STRAT_IMPL_RE = re.compile(
    r"^#{1,3}\s+Strategic\s+Implications?", re.IGNORECASE | re.MULTILINE
)
_L2_NEXT_HEADING_RE = re.compile(r"^#{1,3}\s+", re.MULTILINE)

# Minimum non-whitespace chars in the Strategic Implications section body to pass.
_L2_IMPL_MIN_CHARS = 100

# SAS pass threshold: semantic alignment score ≥ this value counts as l2_semantic_alignment
_L2_SAS_THRESHOLD = 0.5

# L3 — Action layer regexes
_L3_SCQA_EXEC_RE   = re.compile(r"executive\s+summary", re.IGNORECASE)
_L3_SCQA_SIT_RE    = re.compile(r"situation|complication", re.IGNORECASE)
_L3_ACTION_SEC_RE  = re.compile(r"recommended\s+action|action\s+plan", re.IGNORECASE)
# Owner: any explicit ownership label (label-based, stable across LLM variants)
_L3_OWNER_RE       = re.compile(r"owner\s*:|lead\s*:|responsible\s*:", re.IGNORECASE)
# Time: duration pattern (number + time unit) — robust against label variation
# (catches "30 days", "6 months", "2 weeks", "90-day" etc.)
_L3_TIME_RE        = re.compile(
    r"\b\d+[\-\s]*(day|week|month|quarter|year)s?\b", re.IGNORECASE
)
# Metric: tracked but optional — not required for specificity pass
_L3_METRIC_RE      = re.compile(r"metric\s*:|kpi\s*:|target\s*:|measure\s*:", re.IGNORECASE)


def _bat_report_maturity(
    state: dict,
    run_dir: str,
    pa_score: float | None = None,
    sas_score: float | None = None,
) -> dict:
    """
    BAT Report Maturity — weighted four-component score (replaces [INTERPRETABILITY]).

    PA   Plan Adherence   (weight 0.1): pass-through of pre-computed pa_score
    L1   Evidence         (weight 0.2, 3 checks):
      l1_chart_eda        — ≥1 EDA chart (plots/*.png) embedded in report
      l1_chart_model      — ≥1 model-explanation chart (explain/*.png / shap / rf_)
      l1_external_evidence— RAP succeeded: manager_rap_context non-empty (Manager-level
                            arxiv+web fusion) OR bat_rag_buckets non-empty (BAT vector-DB)
                            OR bat_arxiv_papers non-empty (BAT arxiv fallback)
    L2   Interpretation   (weight 0.3, 2 checks):
      l2_implication_presence — '## Strategic Implications' section exists with ≥100 chars of body
      l2_semantic_alignment   — pre-computed SAS score ≥ _L2_SAS_THRESHOLD (0.5)
    L3   Action           (weight 0.4, 4 checks):
      l3_mckinsey_structure — ≥2 McKinsey-style section headings
      l3_scqa_structure     — both Executive Summary AND Situation/Complication headings
      l3_action_section     — Recommended Actions heading present
      l3_action_specificity — fractional score (owner + time + metric) / 3:
                              owner label (owner:/lead:/responsible:),
                              time-duration pattern (e.g. "30 days", "6 months"),
                              metric/KPI label (metric:/kpi:/target:)

    Final: brm_score = pa_score*0.1 + l1_score*0.2 + l2_score*0.3 + l3_score*0.4
    """
    detail: dict = {}

    # ── L1 external_evidence: state-level RAP check ───────────────────────────
    # Three independent RAP sources; any one succeeding counts.
    # 1. Manager-level RAP: arxiv + web retrieval fused into manager_rap_context.
    manager_ctx = state.get("manager_rap_context") or ""
    manager_hit = bool(manager_ctx.strip())

    # 2. BAT agent vector-DB buckets (McKinsey knowledge base).
    rag_buckets = state.get("bat_rag_buckets") or {}
    rag_hit = bool(
        rag_buckets and any(
            isinstance(v, list) and len(v) > 0 for v in rag_buckets.values()
        )
    )

    # 3. BAT agent ArXiv fallback papers.
    arxiv_papers = state.get("bat_arxiv_papers") or []
    arxiv_hit    = bool(arxiv_papers)

    rap_success  = manager_hit or rag_hit or arxiv_hit

    detail["l1_external_evidence"]    = rap_success
    detail["l1_ext_manager_rap_hit"]  = manager_hit
    detail["l1_ext_rag_hit"]          = rag_hit
    detail["l1_ext_arxiv_hit"]        = arxiv_hit

    # ── PA pass-through ───────────────────────────────────────────────────────
    pa_val = pa_score if pa_score is not None else 0.0
    detail["pa_score"] = pa_val

    bat_path = Path(run_dir) / "reports" / "business_insight_report.md"
    if not bat_path.exists():
        for k in (
            "l1_chart_eda", "l1_chart_model",
            "l2_implication_presence", "l2_semantic_alignment",
            "l3_mckinsey_structure", "l3_scqa_structure",
            "l3_action_section", "l3_action_specificity",
        ):
            detail[k] = False
        detail["note"] = "business_insight_report.md not found"
        l1_passed = int(rap_success)
        sas_pass = sas_score is not None and sas_score >= _L2_SAS_THRESHOLD
        detail["l2_implication_content_len"] = 0
        detail["l2_semantic_alignment"]      = sas_pass
        detail["l2_sas_score"]               = sas_score
        l2_passed = int(sas_pass)
        brm_score = round(pa_val * 0.1 + l1_passed / 3 * 0.2 + l2_passed / 2 * 0.3, 4)
        return {
            "score":    brm_score,
            "pa_score": pa_val,
            "l1_score": round(l1_passed / 3, 4), "l2_score": round(l2_passed / 2, 4), "l3_score": 0.0,
            "l1_passed": l1_passed, "l1_total": 3,
            "l2_passed": l2_passed, "l2_total": 2,
            "l3_passed": 0, "l3_total": 4,
            "detail": detail,
        }

    report_text = bat_path.read_text(encoding="utf-8", errors="ignore")

    # ── L1 Evidence (chart checks; external_evidence already set above) ───────
    eda_hits   = _L1_EDA_CHART_RE.findall(report_text)
    model_hits = _L1_MODEL_CHART_RE.findall(report_text)

    detail["l1_chart_eda"]         = len(eda_hits) >= 1
    detail["l1_chart_eda_count"]   = len(eda_hits)
    detail["l1_chart_model"]       = len(model_hits) >= 1
    detail["l1_chart_model_count"] = len(model_hits)

    l1_passed = sum(int(detail[k]) for k in ("l1_chart_eda", "l1_chart_model", "l1_external_evidence"))
    l1_score  = round(l1_passed / 3, 4)

    # ── L2 Interpretation ────────────────────────────────────────────────────
    # Check "## Strategic Implications" section exists and has substantive content.
    strat_match = _L2_STRAT_IMPL_RE.search(report_text)
    if strat_match:
        body_start = strat_match.end()
        next_h = _L2_NEXT_HEADING_RE.search(report_text, body_start)
        section_body = (
            report_text[body_start: body_start + next_h.start()]
            if next_h else report_text[body_start:]
        )
        impl_content_len = len(section_body.strip())
    else:
        impl_content_len = 0

    sas_pass = sas_score is not None and sas_score >= _L2_SAS_THRESHOLD

    detail["l2_implication_presence"]     = impl_content_len >= _L2_IMPL_MIN_CHARS
    detail["l2_implication_content_len"]  = impl_content_len
    detail["l2_semantic_alignment"]       = sas_pass
    detail["l2_sas_score"]               = sas_score

    l2_passed = sum(int(detail[k]) for k in ("l2_implication_presence", "l2_semantic_alignment"))
    l2_score  = round(l2_passed / 2, 4)

    # ── L3 Action ────────────────────────────────────────────────────────────
    mck_hits    = _MCKINSEY_RE.findall(report_text)
    has_exec    = bool(_L3_SCQA_EXEC_RE.search(report_text))
    has_sit     = bool(_L3_SCQA_SIT_RE.search(report_text))
    has_action  = bool(_L3_ACTION_SEC_RE.search(report_text))

    # action specificity: scan only the Recommended Actions section for the triple
    action_block = ""
    action_match = _L3_ACTION_SEC_RE.search(report_text)
    if action_match:
        action_block = report_text[action_match.start():]
    owner_ok   = bool(_L3_OWNER_RE.search(action_block))
    time_ok    = bool(_L3_TIME_RE.search(action_block))
    metric_ok  = bool(_L3_METRIC_RE.search(action_block))
    spec_score = round((int(owner_ok) + int(time_ok) + int(metric_ok)) / 3, 4)

    detail["l3_mckinsey_structure"]     = len(mck_hits) >= 2
    detail["l3_mckinsey_hits_count"]    = len(mck_hits)
    detail["l3_scqa_structure"]         = has_exec and has_sit
    detail["l3_scqa_has_exec"]          = has_exec
    detail["l3_scqa_has_situation"]     = has_sit
    detail["l3_action_section"]         = has_action
    detail["l3_action_specificity"]     = spec_score
    detail["l3_specificity_owner"]      = owner_ok
    detail["l3_specificity_time"]       = time_ok
    detail["l3_specificity_metric"]     = metric_ok

    l3_binary = sum(int(detail[k]) for k in (
        "l3_mckinsey_structure", "l3_scqa_structure", "l3_action_section",
    ))
    l3_passed = round(l3_binary + spec_score, 4)
    l3_score  = round(l3_passed / 4, 4)

    brm_score = round(pa_val * 0.1 + l1_score * 0.2 + l2_score * 0.3 + l3_score * 0.4, 4)

    return {
        "score":    brm_score,
        "pa_score": pa_val,
        "l1_score": l1_score,
        "l2_score": l2_score,
        "l3_score": l3_score,
        "l1_passed": l1_passed, "l1_total": 3,
        "l2_passed": l2_passed, "l2_total": 2,
        "l3_passed": l3_passed, "l3_total": 4,
        "detail":   detail,
    }


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _flatten_record(record: dict) -> dict:
    r: dict = {}
    r["case_id"]             = record["case_id"]
    r["timestamp"]           = record["timestamp"]
    r["run_dir"]             = record.get("run_dir", "")
    r["competition"]         = record.get("competition", "")
    r["system_name"]         = record["system"]["name"]
    r["system_type"]         = record["system"]["type"]
    r["dataset"]             = record["dataset"]["name"]
    r["source"]              = record["dataset"]["source"]
    r["task_type"]           = record["dataset"]["task_type"]
    r["industry"]            = record["dataset"]["industry"]
    r["primary_metric_name"] = record["dataset"]["primary_metric"]

    results = record["results"]

    for k, v in results["primary"].items():
        r[f"primary_{k}"] = v

    for k, v in results["additional"].items():
        if not k.endswith("_skipped_reason"):
            r[k] = v

    acs = results.get("acs", {})
    r["SR"]   = acs.get("SR")
    r["ANPS"] = acs.get("ANPS")
    r["ACS"]  = acs.get("ACS")

    res = results.get("resource") or {}
    rt  = res.get("runtime", {})
    tk  = res.get("tokens") or {}
    r["runtime_total_s"]         = rt.get("total_seconds")
    r["tokens_prompt_total"]     = tk.get("prompt")
    r["tokens_completion_total"] = tk.get("completion")
    r["tokens_total"]            = tk.get("total")

    by_rt = rt.get("by_phase", {})
    for phase, abbrev in PHASE_ABBREV_SYSTEM.items():
        r[f"rt_{abbrev}_s"] = by_rt.get(phase)

    interp = results.get("interpretability", {})
    brm = interp.get("bat_report_maturity") or {}
    r["brm_score"]    = brm.get("score")
    r["brm_pa_score"] = brm.get("pa_score")
    r["brm_l1_score"] = brm.get("l1_score")
    r["brm_l2_score"] = brm.get("l2_score")
    r["brm_l3_score"] = brm.get("l3_score")
    brm_detail = brm.get("detail") or {}
    for key in (
        "l1_chart_eda", "l1_chart_model", "l1_external_evidence",
        "l2_implication_presence", "l2_semantic_alignment",
        "l3_mckinsey_structure", "l3_scqa_structure",
        "l3_action_section", "l3_action_specificity",
    ):
        r[f"brm_{key}"] = int(brm_detail[key]) if isinstance(brm_detail.get(key), bool) else brm_detail.get(key)
    r["brm_l2_sas_score"] = brm_detail.get("l2_sas_score")

    return r


def _upsert_jsonl_record(record: dict, path: str) -> None:
    """Replace any prior line with the same (run_dir, competition, system.name), else append.

    Legacy lines without ``run_dir`` are kept verbatim so existing logs are not wiped.
    """
    run_dir = str(record.get("run_dir") or "")
    competition = str(record.get("competition") or "")
    sys_name = str((record.get("system") or {}).get("name") or "")
    key = (run_dir, competition, sys_name)

    kept: list[str] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                raw = line.rstrip("\n")
                if not raw.strip():
                    continue
                try:
                    old = json.loads(raw)
                except json.JSONDecodeError:
                    kept.append(raw)
                    continue
                old_key = (
                    str(old.get("run_dir") or ""),
                    str(old.get("competition") or ""),
                    str((old.get("system") or {}).get("name") or ""),
                )
                if old_key == key and run_dir:
                    continue
                kept.append(raw)

    kept.append(json.dumps(record, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")


def _read_records_csv(path: str) -> pd.DataFrame:
    """Load experiment CSV; tolerate ragged rows from legacy appends or embedded commas."""
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        import warnings

        warnings.warn(
            f"{path}: inconsistent column counts; skipping malformed rows. "
            "Consider deleting this CSV and re-running evals if results look wrong.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _upsert_csv(record: dict) -> None:
    """One row per (run_dir, competition, system_name); re-evaluating the same run overwrites."""
    row = _flatten_record(record)
    rd, comp, sn = str(row.get("run_dir") or ""), str(row.get("competition") or ""), str(row.get("system_name") or "")

    if not os.path.exists(RECORDS_CSV):
        pd.DataFrame([row]).to_csv(RECORDS_CSV, index=False)
        return

    df = _read_records_csv(RECORDS_CSV)
    has_key = all(c in df.columns for c in ("run_dir", "competition", "system_name"))
    if has_key and rd:
        same = (
            (df["run_dir"].fillna("").astype(str) == rd)
            & (df["competition"].fillna("").astype(str) == comp)
            & (df["system_name"].fillna("").astype(str) == sn)
        )
        df = df.loc[~same].copy()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RECORDS_CSV, index=False)


# ── Main evaluate function ────────────────────────────────────────────────────

def evaluate(
    run_dir: str,
    competition: str,
    description_file: str = "",
    system_name: str = "MAC-ADS",
    system_type: str = "multi-agent DS pipeline",
) -> dict:
    run_dir_key = str(Path(run_dir).expanduser().resolve())

    config      = COMPETITION_CONFIG[competition]
    meta        = DATASET_META[competition]
    primary_key = config["primary_key"]
    comp_metric = config["metric"]

    state = _find_state_snapshot(run_dir_key)

    primary, additional = _extract_metrics(state, primary_key, comp_metric)
    primary_val = primary.get(primary_key)
    acs         = _compute_sr_anps_acs(state, primary_key, primary_val)
    token_usage = _load_token_usage(run_dir_key)
    resource    = {"runtime": _parse_timing(state.get("timing")), "tokens": token_usage}

    pa  = _plan_adherence(state, run_dir_key)
    sas = _semantic_alignment(state, run_dir_key, description_file)
    brm = _bat_report_maturity(
        state, run_dir_key,
        pa_score=pa["score"],
        sas_score=sas.get("score"),
    )
    interpretability = {
        "plan_adherence":      pa,
        "semantic_alignment":  sas,
        "bat_report_maturity": brm,
    }

    SEP = "=" * 66
    now     = datetime.now()
    case_id = now.strftime("%Y%m%d-%H%M%S")

    # Build the full report in memory first, then print once to avoid
    # partial-flush artifacts in Windows terminals.
    lines: list[str] = []
    lines.append(f"\n{SEP}")
    lines.append(f"  EXPERIMENT RECORD  {case_id}")
    lines.append(SEP)
    lines.append(f"  System        : {system_name}  ({system_type})")
    lines.append(f"  Dataset       : {meta['name']}")
    lines.append(f"  Source        : {meta['source']}")
    lines.append(f"  Task type     : {meta['task_type']}")
    lines.append(f"  Industry      : {meta['industry']}")
    lines.append(f"  Primary metric: {meta['primary_metric']}")
    lines.append(f"{'─'*66}")

    lines.append("  [PRIMARY]")
    for k, v in primary.items():
        lines.append(f"    {k}: {v}")

    lines.append("  [ACS COMPONENTS]")
    lines.append(f"    SR={acs['SR']}  ANPS={acs['ANPS']}  ACS={acs['ACS']}")

    lines.append("  [ADDITIONAL]")
    for k, v in additional.items():
        lines.append(f"    {k}: {v}")

    lines.append("  [RESOURCE UTILIZATION]")
    rt       = resource["runtime"]
    by_phase = rt.get("by_phase", {})
    lines.append(f"    {'Phase':<30} {'Runtime':>9}")
    lines.append(f"    {'─'*30} {'─'*9}")
    for phase in PHASE_PRINT_ORDER:
        secs = by_phase.get(phase)
        if secs is not None:
            lines.append(f"    {phase:<30} {secs:>8.1f}s")
    lines.append(f"    {'─'*30} {'─'*9}")
    lines.append(f"    {'TOTAL':<30} {str(rt.get('total_seconds') or '-'):>8}s")
    tk = resource.get("tokens") or {}
    if tk:
        lines.append(f"    tokens  prompt={tk.get('prompt',0)}  "
                     f"completion={tk.get('completion',0)}  "
                     f"total={tk.get('total',0)}")
    else:
        lines.append("    (token_usage.json not found -- run with updated main.py)")

    def _ck(v: bool) -> str:
        return "✓" if v else "✗"

    brm_d = brm.get("detail") or {}
    lines.append("  [BAT REPORT MATURITY]")
    lines.append(f"    BRM Score : {brm['score']}  "
                 f"(PA×0.1 + L1×0.2 + L2×0.3 + L3×0.4)")
    lines.append(
        f"    Plan Adherence      (PA): {brm.get('pa_score', 0.0):.4f}"
        f"  ({pa['passed']}/{pa['total']} pipeline checks)"
    )
    lines.append(
        f"    Evidence            (L1): {brm.get('l1_score', 0.0):.4f}"
        f"  {brm.get('l1_passed', '?')}/{brm.get('l1_total', 3)}"
        f"  EDA-chart{_ck(brm_d.get('l1_chart_eda', False))}"
        f"  Model-chart{_ck(brm_d.get('l1_chart_model', False))}"
        f"  RAP{_ck(brm_d.get('l1_external_evidence', False))}"
    )
    lines.append(
        f"    Interpretation      (L2): {brm.get('l2_score', 0.0):.4f}"
        f"  {brm.get('l2_passed', '?')}/{brm.get('l2_total', 2)}"
        f"  Implication{_ck(brm_d.get('l2_implication_presence', False))}"
        f"  SAS{_ck(brm_d.get('l2_semantic_alignment', False))}"
        f"(score={brm_d.get('l2_sas_score')})"
    )
    spec_val = brm_d.get("l3_action_specificity", 0)
    spec_owner  = brm_d.get("l3_specificity_owner", False)
    spec_time   = brm_d.get("l3_specificity_time", False)
    spec_metric = brm_d.get("l3_specificity_metric", False)
    lines.append(
        f"    Action              (L3): {brm.get('l3_score', 0.0):.4f}"
        f"  {brm.get('l3_passed', '?')}/4"
        f"  McKinsey{_ck(brm_d.get('l3_mckinsey_structure', False))}"
        f"  SCQA{_ck(brm_d.get('l3_scqa_structure', False))}"
        f"  Actions{_ck(brm_d.get('l3_action_section', False))}"
        f"  Specificity({int(spec_owner)+int(spec_time)+int(spec_metric)}/3"
        f" owner{_ck(spec_owner)} time{_ck(spec_time)} metric{_ck(spec_metric)})"
    )
    lines.append(SEP)

    # Flush pending progress bars / logging so this block does not interleave.
    sys.stderr.flush()
    sys.stdout.flush()
    print("\n".join(lines), flush=True)

    record = {
        "case_id":     case_id,
        "timestamp":   now.isoformat(timespec="seconds"),
        "run_dir":     run_dir_key,
        "competition": competition,
        "system":      {"name": system_name, "type": system_type},
        "dataset":     meta,
        "results": {
            "primary":          primary,
            "additional":       additional,
            "acs":              acs,
            "resource":         resource,
            "interpretability": interpretability,
        },
    }

    _upsert_jsonl_record(record, RECORDS_FILE)
    print(f"\n  Record upserted in: {RECORDS_FILE}  (case_id={case_id}; key=run_dir+competition+system)")

    _upsert_csv(record)
    print(f"  CSV upserted: {RECORDS_CSV}")

    snap_path = Path(run_dir_key) / "reports" / "benchmark_metrics.json"
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"  Snapshot saved to:  {snap_path}\n")

    return record


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate our multi-agent DS pipeline run and record benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_eval_system.py \\
      --run-dir  artifacts/workspace/2026-05-16-00-39 \\
      --competition house_prices \\
      --description-file datasets/house_prices/data_description.txt

  python benchmark_eval_system.py \\
      --run-dir  artifacts/workspace/2026-05-16-01-00 \\
      --competition bank_churn \\
      --description-file datasets/bank_churn/data_description.txt \\
      --system-name "OurSystem-v1"
""",
    )
    parser.add_argument("--run-dir",    required=True,
                        help="Run output directory (contains reports/*_state.json)")
    parser.add_argument("--competition", required=True,
                        choices=list(COMPETITION_CONFIG.keys()))
    parser.add_argument("--description-file", default="", dest="description_file",
                        help="Data description .txt; enables the SAS check inside BAT Report Maturity (L2)")
    parser.add_argument("--system-name", default="MAC-ADS")
    parser.add_argument("--system-type", default="multi-agent DS pipeline")
    args = parser.parse_args()

    evaluate(
        run_dir=args.run_dir,
        competition=args.competition,
        description_file=args.description_file,
        system_name=args.system_name,
        system_type=args.system_type,
    )
