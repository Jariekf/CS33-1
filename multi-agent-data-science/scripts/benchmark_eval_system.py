"""Our System Benchmark Evaluation Script
------------------------------------------
Reads a completed pipeline run-dir, extracts all metrics from the state
snapshot, computes interpretability scores from run artifacts, and **upserts**
a structured record into the shared experiment log (JSONL + CSV): the same
``run_dir`` + competition + system_name overwrites the previous row instead of
duplicating.

Usage:
    python benchmark_eval_system.py \\
        --run-dir  artifacts/workspace/2026-05-16-00-39 \\
        --competition house_prices \\
        --test  path/to/split_test_with_labels.csv \\
        [--description-file datasets/house_prices/data_description.txt] \\
        [--system-name "OurSystem"] \\
        [--system-type "multi-agent DS pipeline"]

Required:
    --run-dir        Run output directory created by main.py
    --competition    One of: bank_churn | house_prices | obesity_risks |
                     classification_with_an_academic_success_dataset
    --test           Split test CSV that includes the true label column

Optional:
    --description-file   Data description .txt (for Semantic Alignment Score)
    --system-name        Label stored in experiment record (default: OurSystem)
    --system-type        Label stored in experiment record
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

_MCKINSEY_SECTION_PATTERNS = [
    r"executive\s+summary", r"key\s+finding", r"strategic\s+(implication|recommendation)",
    r"business\s+(insight|implication|recommendation)", r"action\s+(item|plan|recommendation)",
    r"next\s+step", r"conclusion",
    r"执行摘要", r"核心发现", r"关键发现", r"战略建议", r"业务洞察",
    r"行动建议", r"下一步", r"结论", r"结语",
]
_MCKINSEY_RE = re.compile("|".join(_MCKINSEY_SECTION_PATTERNS), re.IGNORECASE)


def _bat_report_maturity(state: dict, run_dir: str) -> dict:
    """
    3-component score (each 0 or 1 → total / 3):
      1. RAG loaded      – bat_rag_buckets in state is non-empty
      2. McKinsey struct  – business_insight_report.md contains ≥2 McKinsey-style headings
      3. Chart refs       – business_insight_report.md references ≥1 image
    """
    detail: dict = {}

    bats = state.get("bat_rag_buckets")
    rag_loaded = bool(
        bats and isinstance(bats, dict) and
        any(len(v) > 0 for v in bats.values() if isinstance(v, list))
    )
    detail["rag_loaded"] = rag_loaded

    bat_path = Path(run_dir) / "reports" / "business_insight_report.md"
    if not bat_path.exists():
        detail["mckinsey_structure"] = False
        detail["chart_references"]   = False
        detail["note"]               = "business_insight_report.md not found"
        return {"score": round((1 if rag_loaded else 0) / 3, 4),
                "passed": 1 if rag_loaded else 0, "total": 3, "detail": detail}

    report_text = bat_path.read_text(encoding="utf-8", errors="ignore")

    mckinsey_hits = _MCKINSEY_RE.findall(report_text)
    detail["mckinsey_structure"]  = len(mckinsey_hits) >= 2
    detail["mckinsey_hits_count"] = len(mckinsey_hits)

    chart_refs = re.findall(
        r"!\[.*?\]\(.*?\)|plots[\\/][\w\-]+\.png|explain[\\/][\w\-]+\.png|\.png",
        report_text, re.IGNORECASE,
    )
    detail["chart_references"] = len(chart_refs) >= 1
    detail["chart_ref_count"]  = len(chart_refs)

    passed = sum([
        1 if rag_loaded else 0,
        1 if detail["mckinsey_structure"] else 0,
        1 if detail["chart_references"] else 0,
    ])
    return {"score": round(passed / 3, 4), "passed": passed, "total": 3, "detail": detail}


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
    r["pa_score"]  = (interp.get("plan_adherence") or {}).get("score")
    r["sas_score"] = (interp.get("semantic_alignment") or {}).get("score")
    r["brm_score"] = (interp.get("bat_report_maturity") or {}).get("score")

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
    test_path: str,
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
    brm = _bat_report_maturity(state, run_dir_key)
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

    lines.append("  [INTERPRETABILITY]")
    lines.append(f"    Plan Adherence         : {pa['score']}  "
                 f"({pa['passed']}/{pa['total']} checks passed)")
    lines.append(f"    Semantic Alignment     : {sas.get('score')}  "
                 f"({sas.get('columns_found', 0)}/{sas.get('columns_total', 0)} columns)")
    lines.append(f"    BAT Report Maturity    : {brm['score']}  "
                 f"({brm.get('passed', '?')}/{brm.get('total', 3)} components ok)")
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
      --test  datasets/house_prices/split_test_with_labels.csv \\
      --description-file datasets/house_prices/data_description.txt

  python benchmark_eval_system.py \\
      --run-dir  artifacts/workspace/2026-05-16-01-00 \\
      --competition bank_churn \\
      --test  datasets/bank_churn/split_test_with_labels.csv \\
      --description-file datasets/bank_churn/data_description.txt \\
      --system-name "OurSystem-v1"
""",
    )
    parser.add_argument("--run-dir",    required=True,
                        help="Run output directory (contains reports/*_state.json)")
    parser.add_argument("--competition", required=True,
                        choices=list(COMPETITION_CONFIG.keys()))
    parser.add_argument("--test",        required=True,
                        help="Split test CSV with true label column")
    parser.add_argument("--description-file", default="", dest="description_file",
                        help="Data description .txt for Semantic Alignment Score")
    parser.add_argument("--system-name", default="MAC-ADS")
    parser.add_argument("--system-type", default="multi-agent DS pipeline")
    args = parser.parse_args()

    evaluate(
        run_dir=args.run_dir,
        competition=args.competition,
        test_path=args.test,
        description_file=args.description_file,
        system_name=args.system_name,
        system_type=args.system_type,
    )
