"""
AutoKaggle Benchmark Evaluation Script
---------------------------------------
Writes cumulative ``experiment_records.jsonl`` / ``experiment_records.csv`` in the
current working directory. Re-running evaluate for the same competition folder
+ system_name **replaces** the previous row (normalized AutoKaggle competition path,
competition key, system name) instead of appending duplicates.

Usage:
    # Step 1: replace AutoKaggle's input files with your project's split data
    python benchmark_eval.py prepare --competition bank_churn \
        --train path/to/split_train.csv \
        --test  path/to/split_test_with_labels.csv

    # Step 2: run AutoKaggle
    python framework.py --competition bank_churn --model gpt_4o

    # Step 3: evaluate and record results
    python benchmark_eval.py evaluate --competition bank_churn \
        --test path/to/split_test_with_labels.csv \
        [--system-name AutoKaggle] [--system-type "SOTA multi-agent"]
"""

import argparse
import shutil
import os
import math
import json
import re
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error,
    r2_score, max_error as sklearn_max_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
)

# ── Competition config ────────────────────────────────────────────────────────

COMPETITION_CONFIG = {
    'bank_churn': {
        'label_col': 'Exited',
        'id_col':    'id',
        'metric':    'auc',
    },
    'house_prices': {
        'label_col': 'SalePrice',
        'id_col':    'Id',
        'metric':    'rmse_log',
    },
    'obesity_risks': {
        'label_col': 'NObeyesdad',
        'id_col':    'id',
        'metric':    'accuracy',
    },
    'classification_with_an_academic_success_dataset': {
        'label_col': 'Target',
        'id_col':    'id',
        'metric':    'accuracy',
    },
}

DATASET_META = {
    'bank_churn': {
        'name':           'Bank Churn',
        'source':         'AutoKaggle / Kaggle Competition',
        'task_type':      'Tabular Binary Classification',
        'industry':       'Banking / Customer Retention',
        'primary_metric': 'ROC-AUC',
    },
    'house_prices': {
        'name':           'House Prices',
        'source':         'AutoKaggle / Kaggle Competition',
        'task_type':      'Tabular Regression',
        'industry':       'Real Estate / Housing Valuation',
        'primary_metric': 'RMSLE',
    },
    'obesity_risks': {
        'name':           'Obesity Risk',
        'source':         'AutoKaggle / Kaggle Competition',
        'task_type':      'Tabular Multi-class Classification',
        'industry':       'Healthcare / Public Health',
        'primary_metric': 'Accuracy',
    },
    'classification_with_an_academic_success_dataset': {
        'name':           'Academic Success',
        'source':         'AutoKaggle / Kaggle Competition',
        'task_type':      'Tabular Multi-class Classification',
        'industry':       'Education / Student Support',
        'primary_metric': 'Accuracy',
    },
}

COMPETITION_DIR   = 'multi_agents/competition'
RECORDS_FILE      = 'experiment_records.jsonl'
RECORDS_CSV       = 'experiment_records.csv'

# Short aliases for phase column names in CSV
PHASE_ABBREV = {
    'Understand Background':                          'ub',
    'Preliminary Exploratory Data Analysis':          'pre_eda',
    'Data Cleaning':                                  'data_clean',
    'In-depth Exploratory Data Analysis':              'deep_eda',
    'Feature Engineering':                            'feat_eng',
    'Model Building, Validation, and Prediction':     'model',
}

INTERMEDIATE_FILES = [
    'cleaned_train.csv', 'cleaned_test.csv',
    'processed_train.csv', 'processed_test.csv',
    'submission.csv', 'competition_info.txt', 'research_report.md',
]
INTERMEDIATE_DIRS = [
    'understand_background', 'pre_eda', 'deep_eda',
    'data_cleaning', 'feature_engineering', 'model_build_predict',
]

# ── Prepare ───────────────────────────────────────────────────────────────────

def prepare(competition: str, train_path: str, test_path: str):
    config    = COMPETITION_CONFIG[competition]
    label_col = config['label_col']
    id_col    = config['id_col']
    comp_dir  = os.path.join(COMPETITION_DIR, competition)

    test_df = pd.read_csv(test_path)
    assert label_col in test_df.columns, f"Label column '{label_col}' not found in test file."
    assert id_col    in test_df.columns, f"ID column '{id_col}' not found in test file."

    shutil.copy(train_path, os.path.join(comp_dir, 'train.csv'))
    print(f"[prepare] train.csv replaced ({len(pd.read_csv(train_path))} rows)")

    test_df.drop(columns=[label_col]).to_csv(os.path.join(comp_dir, 'test.csv'), index=False)
    print(f"[prepare] test.csv replaced ({len(test_df)} rows, label '{label_col}' removed)")

    sample = test_df[[id_col]].copy()
    sample[label_col] = 0.5
    sample.to_csv(os.path.join(comp_dir, 'sample_submission.csv'), index=False)
    print(f"[prepare] sample_submission.csv rebuilt")

    removed = []
    for f in INTERMEDIATE_FILES:
        fp = os.path.join(comp_dir, f)
        if os.path.exists(fp):
            os.remove(fp); removed.append(f)
    for d in INTERMEDIATE_DIRS:
        dp = os.path.join(comp_dir, d)
        if os.path.exists(dp):
            shutil.rmtree(dp); removed.append(d + '/')
    if removed:
        print(f"[prepare] Cleaned up: {removed}")

    print(f"\n[prepare] Done. Now run:\n  python framework.py --competition {competition} --model gpt_4o")


# ── Metrics ───────────────────────────────────────────────────────────────────

def _is_prob_column(series: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(series) and series.between(0, 1).all()


def _compute_classification_metrics(y_true, y_pred_raw) -> dict:
    metrics  = {}
    is_binary = y_true.nunique() == 2
    has_proba = _is_prob_column(pd.Series(y_pred_raw))

    if has_proba and is_binary:
        y_proba = y_pred_raw.values
        y_pred  = (y_proba >= 0.5).astype(int)
    else:
        y_pred  = y_pred_raw.values
        y_proba = None

    metrics['accuracy']    = round(float(accuracy_score(y_true, y_pred)), 4)
    metrics['f1_macro']    = round(float(f1_score(y_true, y_pred, average='macro',    zero_division=0)), 4)
    metrics['f1_weighted'] = round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4)
    metrics['precision']   = round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)), 4)
    metrics['recall']      = round(float(recall_score(y_true, y_pred, average='weighted',    zero_division=0)), 4)
    if is_binary:
        metrics['sensitivity'] = round(float(recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)), 4)

    if y_proba is not None:
        try:
            metrics['roc_auc'] = round(float(roc_auc_score(y_true, y_proba)), 4)
        except Exception as e:
            metrics['roc_auc'] = None
            metrics['roc_auc_skipped_reason'] = str(e)
        try:
            metrics['log_loss'] = round(float(log_loss(y_true, y_proba)), 4)
        except Exception as e:
            metrics['log_loss'] = None
            metrics['log_loss_skipped_reason'] = str(e)
    else:
        metrics['roc_auc']  = None
        metrics['log_loss'] = None

    return metrics


def _compute_regression_metrics(y_true, y_pred) -> dict:
    metrics = {}
    y_true  = y_true.values
    y_pred  = y_pred.values

    metrics['rmse']      = round(float(math.sqrt(mean_squared_error(y_true, y_pred))), 4)
    metrics['mae']       = round(float(mean_absolute_error(y_true, y_pred)), 4)
    metrics['r2']        = round(float(r2_score(y_true, y_pred)), 4)
    metrics['max_error'] = round(float(sklearn_max_error(y_true, y_pred)), 4)

    try:
        v = float(mean_absolute_percentage_error(y_true, y_pred))
        metrics['mape'] = round(v, 4) if np.isfinite(v) else None
        if not np.isfinite(v):
            metrics['mape_skipped_reason'] = 'y_true contains zero(s)'
    except Exception as e:
        metrics['mape'] = None
        metrics['mape_skipped_reason'] = str(e)

    try:
        metrics['rmsle'] = round(float(math.sqrt(
            mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))), 4)
    except Exception as e:
        metrics['rmsle'] = None
        metrics['rmsle_skipped_reason'] = str(e)

    return metrics


# ── ACS (aligned with benchmark_eval_system._compute_sr_anps_acs) ────────────

def _compute_sr_anps_acs_autokaggle(primary_key: str, primary_val, submission_rows: int) -> dict:
    """
    SR   = 1 if AutoKaggle produced a non-empty submission and primary metric is
           computed, else 0.
    ANPS = normalised primary metric (0–1, higher = better).
           AUC / Accuracy → ANPS = primary_val directly.
           RMSLE (lower-is-better) → ANPS = max(0, 1 − primary_val).
    ACS  = 0.5 × SR + 0.5 × ANPS.
    """
    sr = 1 if (submission_rows > 0 and primary_val is not None) else 0

    if primary_val is None:
        return {"SR": sr, "ANPS": None, "ACS": None}

    if primary_key == "rmsle":
        anps = round(max(0.0, 1.0 - float(primary_val)), 4)
    else:
        anps = round(float(primary_val), 4)

    return {"SR": sr, "ANPS": anps, "ACS": round(0.5 * sr + 0.5 * anps, 4)}


# ── Resource parsing ──────────────────────────────────────────────────────────

LOG_TS_FMT = '%Y-%m-%d %H:%M:%S,%f'
LOG_TS_RE  = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)')
PHASE_RE   = re.compile(r'Current State: State: (.+?), Current Step:')
TOKEN_RE   = re.compile(r'TOKEN_USAGE model=(\S+) prompt=(\d+) completion=(\d+) total=(\d+)')
SOP_START  = 'Start SOP for competition:'
SOP_END    = 'SOP is completed.'

PHASE_ORDER = [
    'Understand Background',
    'Preliminary Exploratory Data Analysis',
    'Data Cleaning',
    'In-depth Exploratory Data Analysis',
    'Feature Engineering',
    'Model Building, Validation, and Prediction',
]


def _parse_ts(line: str):
    m = LOG_TS_RE.match(line)
    return datetime.strptime(m.group(1), LOG_TS_FMT) if m else None


def parse_resource_usage(log_path: str) -> dict:
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Find the LAST completed run: scan backward for last SOP_END, then SOP_START before it
    end_idx   = next((i for i in range(len(lines)-1, -1, -1) if SOP_END   in lines[i]), None)
    start_idx = next((i for i in range(end_idx-1,   -1, -1) if SOP_START in lines[i]), None) if end_idx else None

    if start_idx is None or end_idx is None:
        return {'error': 'Could not find a completed run in the log.'}

    run_lines = lines[start_idx: end_idx + 1]
    t_start   = _parse_ts(run_lines[0])
    t_end     = _parse_ts(run_lines[-1])
    total_s   = (t_end - t_start).total_seconds() if t_start and t_end else None

    # Build intervals per phase (a phase may appear multiple times if retried)
    phase_intervals = defaultdict(list)
    cur_phase = None
    cur_start = None
    for line in run_lines:
        ts = _parse_ts(line)
        pm = PHASE_RE.search(line)
        if pm and ts:
            if cur_phase and cur_start:
                phase_intervals[cur_phase].append((cur_start, ts))
            cur_phase = pm.group(1).strip()
            cur_start = ts
    if cur_phase and cur_start:
        phase_intervals[cur_phase].append((cur_start, t_end))

    # Sum all intervals per phase (handles retries correctly)
    phase_runtime = {
        p: round(sum((e - s).total_seconds() for s, e in ivs), 1)
        for p, ivs in phase_intervals.items()
    }

    def _phase_at(ts):
        for p, ivs in phase_intervals.items():
            for s, e in ivs:
                if s <= ts <= e:
                    return p
        return None

    phase_tokens = defaultdict(lambda: {'prompt': 0, 'completion': 0, 'total': 0})
    total_tokens = {'prompt': 0, 'completion': 0, 'total': 0}
    for line in run_lines:
        tm = TOKEN_RE.search(line)
        if tm:
            ts  = _parse_ts(line)
            p, c, t = int(tm.group(2)), int(tm.group(3)), int(tm.group(4))
            total_tokens['prompt']     += p
            total_tokens['completion'] += c
            total_tokens['total']      += t
            if ts:
                ph = _phase_at(ts)
                if ph:
                    phase_tokens[ph]['prompt']     += p
                    phase_tokens[ph]['completion'] += c
                    phase_tokens[ph]['total']      += t

    return {
        'runtime': {'total_seconds': round(total_s, 1) if total_s else None, 'by_phase': phase_runtime},
        'tokens':  {'total': total_tokens, 'by_phase': dict(phase_tokens)},
    }


# ── CSV helpers ──────────────────────────────────────────────────────────────

def _flatten_record(record: dict) -> dict:
    """Flatten a nested experiment record into a single-level dict for CSV."""
    r = {}
    r['case_id']      = record['case_id']
    r['timestamp']    = record['timestamp']
    r['run_dir']      = record.get('run_dir', '')
    r['competition']  = record.get('competition', '')
    r['system_name']  = record['system']['name']
    r['system_type']  = record['system']['type']
    r['dataset']      = record['dataset']['name']
    r['source']       = record['dataset']['source']
    r['task_type']    = record['dataset']['task_type']
    r['industry']     = record['dataset']['industry']
    r['primary_metric_name'] = record['dataset']['primary_metric']

    results = record['results']

    # primary
    for k, v in results['primary'].items():
        r[f'primary_{k}'] = v

    # additional
    for k, v in results['additional'].items():
        r[k] = v

    acs = results.get("acs", {})
    r["SR"] = acs.get("SR")
    r["ANPS"] = acs.get("ANPS")
    r["ACS"] = acs.get("ACS")

    # resource — totals
    res = results.get('resource', {})
    rt  = res.get('runtime', {})
    tk  = res.get('tokens',  {})
    r['runtime_total_s']        = rt.get('total_seconds')
    r['tokens_prompt_total']    = tk.get('total', {}).get('prompt')
    r['tokens_completion_total']= tk.get('total', {}).get('completion')
    r['tokens_total']           = tk.get('total', {}).get('total')

    # resource — by phase
    by_rt  = rt.get('by_phase', {})
    by_tok = tk.get('by_phase', {})
    for phase, abbrev in PHASE_ABBREV.items():
        r[f'rt_{abbrev}_s']              = by_rt.get(phase)
        ph_tok = by_tok.get(phase, {})
        r[f'tok_{abbrev}_prompt']        = ph_tok.get('prompt')
        r[f'tok_{abbrev}_completion']    = ph_tok.get('completion')
        r[f'tok_{abbrev}_total']         = ph_tok.get('total')

    return r


def _upsert_jsonl_record(record: dict, path: str) -> None:
    """Replace prior line with same (run_dir, competition, system.name); keep legacy lines."""
    run_dir = str(record.get('run_dir') or '')
    competition = str(record.get('competition') or '')
    sys_name = str((record.get('system') or {}).get('name') or '')
    key = (run_dir, competition, sys_name)

    kept: list[str] = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                raw = line.rstrip('\n')
                if not raw.strip():
                    continue
                try:
                    old = json.loads(raw)
                except json.JSONDecodeError:
                    kept.append(raw)
                    continue
                old_key = (
                    str(old.get('run_dir') or ''),
                    str(old.get('competition') or ''),
                    str((old.get('system') or {}).get('name') or ''),
                )
                if old_key == key and run_dir:
                    continue
                kept.append(raw)

    kept.append(json.dumps(record, ensure_ascii=False))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(kept) + '\n')


def _read_records_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        import warnings

        warnings.warn(
            f"{path}: inconsistent column counts; skipping malformed rows.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.read_csv(path, engine='python', on_bad_lines='skip')


def _upsert_csv(record: dict) -> None:
    """One row per (run_dir, competition, system_name)."""
    row = _flatten_record(record)
    rd, comp, sn = str(row.get('run_dir') or ''), str(row.get('competition') or ''), str(row.get('system_name') or '')

    if not os.path.exists(RECORDS_CSV):
        pd.DataFrame([row]).to_csv(RECORDS_CSV, index=False)
        return

    df = _read_records_csv(RECORDS_CSV)
    has_key = all(c in df.columns for c in ('run_dir', 'competition', 'system_name'))
    if has_key and rd:
        same = (
            (df['run_dir'].fillna('').astype(str) == rd)
            & (df['competition'].fillna('').astype(str) == comp)
            & (df['system_name'].fillna('').astype(str) == sn)
        )
        df = df.loc[~same].copy()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RECORDS_CSV, index=False)


def evaluate(competition: str, test_path: str,
             system_name: str = 'AutoKaggle',
             system_type: str = 'SOTA multi-agent'):

    config    = COMPETITION_CONFIG[competition]
    meta      = DATASET_META[competition]
    label_col = config['label_col']
    id_col    = config['id_col']
    task_type = config['metric']
    comp_dir  = os.path.join(COMPETITION_DIR, competition)
    comp_dir_key = str(Path(comp_dir).resolve())

    submission_path = os.path.join(comp_dir, 'submission.csv')
    assert os.path.exists(submission_path), f"submission.csv not found: {submission_path}"

    submission = pd.read_csv(submission_path)
    true_df    = pd.read_csv(test_path)[[id_col, label_col]]
    merged     = submission.merge(true_df, on=id_col, suffixes=('_pred', '_true'))

    pred_col = label_col + '_pred' if label_col + '_pred' in merged.columns else label_col
    true_col = label_col + '_true' if label_col + '_true' in merged.columns else label_col

    if task_type == 'rmse_log':
        all_metrics = _compute_regression_metrics(merged[true_col], merged[pred_col])
        primary_key = 'rmsle'
    else:
        all_metrics = _compute_classification_metrics(merged[true_col], merged[pred_col])
        primary_key = 'roc_auc' if task_type == 'auc' else 'accuracy'

    primary_metrics    = {primary_key: all_metrics.get(primary_key)}
    additional_metrics = {k: v for k, v in all_metrics.items()
                          if k != primary_key and not k.endswith('_skipped_reason')}

    primary_val = primary_metrics.get(primary_key)
    acs = _compute_sr_anps_acs_autokaggle(primary_key, primary_val, len(submission))

    # Resource
    log_path = os.path.join(comp_dir, f'{competition}.log')
    resource = parse_resource_usage(log_path) if os.path.exists(log_path) else {}

    # ── Print ──
    SEP = '=' * 62
    now = datetime.now()
    case_id = now.strftime('%Y%m%d-%H%M%S')

    print(f"\n{SEP}")
    print(f"  EXPERIMENT RECORD  {case_id}")
    print(SEP)
    print(f"  System     : {system_name}  ({system_type})")
    print(f"  Dataset    : {meta['name']}")
    print(f"  Source     : {meta['source']}")
    print(f"  Task type  : {meta['task_type']}")
    print(f"  Industry   : {meta['industry']}")
    print(f"  Primary metric: {meta['primary_metric']}")
    print(f"{'─'*62}")

    print(f"  [PRIMARY]")
    for k, v in primary_metrics.items():
        print(f"    {k}: {v}")

    print(f"  [ACS COMPONENTS]")
    print(f"    SR={acs['SR']}  ANPS={acs['ANPS']}  ACS={acs['ACS']}")

    print(f"  [ADDITIONAL]")
    for k, v in additional_metrics.items():
        skipped = all_metrics.get(k + '_skipped_reason')
        suffix  = f"  ← skipped: {skipped}" if skipped else ''
        print(f"    {k}: {v}{suffix}")

    print(f"  [RESOURCE UTILIZATION]")
    if 'error' in resource:
        print(f"    {resource['error']}")
    else:
        rt  = resource.get('runtime', {})
        tk  = resource.get('tokens', {})
        ttl = tk.get('total', {})
        print(f"    {'Phase':<46} {'Runtime':>9}  {'Prompt':>8}  {'Completion':>10}  {'Token Cost':>9}")
        print(f"    {'─'*46} {'─'*9}  {'─'*8}  {'─'*10}  {'─'*9}")
        for phase in PHASE_ORDER:
            secs = rt.get('by_phase', {}).get(phase)
            tok  = tk.get('by_phase', {}).get(phase, {})
            if secs is not None:
                print(f"    {phase:<46} {secs:>8.1f}s"
                      f"  {tok.get('prompt',0):>8}"
                      f"  {tok.get('completion',0):>10}"
                      f"  {tok.get('total',0):>9}")
        print(f"    {'─'*46} {'─'*9}  {'─'*8}  {'─'*10}  {'─'*9}")
        print(f"    {'TOTAL':<46} {rt.get('total_seconds','—'):>8}s"
              f"  {ttl.get('prompt',0):>8}"
              f"  {ttl.get('completion',0):>10}"
              f"  {ttl.get('total',0):>9}")

    print(SEP)

    # ── Persist ──
    record = {
        'case_id':   case_id,
        'timestamp': now.isoformat(timespec='seconds'),
        'run_dir':   comp_dir_key,
        'competition': competition,
        'system': {
            'name': system_name,
            'type': system_type,
        },
        'dataset': meta,
        'results': {
            'primary':    primary_metrics,
            'additional': additional_metrics,
            'acs':        acs,
            'resource':   resource,
        },
    }

    _upsert_jsonl_record(record, RECORDS_FILE)
    print(f"\n  Record upserted in: {RECORDS_FILE}  (case_id={case_id}; key=run_dir+competition+system)")

    _upsert_csv(record)
    print(f"  CSV upserted: {RECORDS_CSV}")

    # Also save per-competition snapshot
    snap_path = os.path.join(comp_dir, 'benchmark_metrics.json')
    with open(snap_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"  Snapshot saved to:  {snap_path}\n")

    return record


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'evaluate'])
    parser.add_argument('--competition', required=True, choices=list(COMPETITION_CONFIG.keys()))
    parser.add_argument('--train',       help='Split train CSV with labels (prepare only)')
    parser.add_argument('--test',        required=True, help='Split test CSV with labels')
    parser.add_argument('--system-name', default='AutoKaggle',        help='System display name')
    parser.add_argument('--system-type', default='SOTA multi-agent',  help='System type description')
    args = parser.parse_args()

    if args.phase == 'prepare':
        assert args.train, '--train is required for prepare'
        prepare(args.competition, args.train, args.test)
    else:
        evaluate(args.competition, args.test,
                 system_name=args.system_name,
                 system_type=args.system_type)