"""Aggregate Benchmark Results
------------------------------
Reads ``experiment_records.jsonl`` and computes cross-run aggregates for
MAC-ADS (our system) and AutoKaggle.  Other baselines are out of scope.

Metrics aggregated per (competition, system_name) group:

  CS components  – SR (all runs), ANPS/ACS (successful runs only)
  Primary        – primary metric mean over successful runs
  Additional     – additional metrics mean over successful runs
  Interpretability – BRM / PA / SAS mean over runs that have the data
  Resource       – mean runtime and token usage over all runs

Records must use the current key schema (``results.cs`` with SR/NPS/CS).

Usage:
  python aggregate_results.py
  python aggregate_results.py --file artifacts/workspace/experiment_records.jsonl
  python aggregate_results.py --competition bank_churn
  python aggregate_results.py --competition bank_churn --system-name AutoKaggle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ── Default records path (same as benchmark_eval_system.py) ──────────────────

_DEFAULT_RECORDS = str(
    Path(__file__).resolve().parent.parent
    / "artifacts" / "workspace" / "experiment_records.jsonl"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_records(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Records file not found: {path}")
    records = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _mean(vals: list[float]) -> float | None:
    return round(sum(vals) / len(vals), 4) if vals else None


def _collect(records: list[dict], key_path: list[str]) -> list[float]:
    """Walk key_path into each record and collect non-None numeric values."""
    out = []
    for r in records:
        node = r
        for k in key_path:
            node = (node or {}).get(k)
        if node is not None:
            try:
                out.append(float(node))
            except (TypeError, ValueError):
                pass
    return out


def _aggregate_group(runs: list[dict]) -> dict:
    T  = len(runs)
    cs_vals = [r.get("results", {}).get("cs") or {} for r in runs]

    sr_vals  = [int(d.get("SR") or 0) for d in cs_vals]
    nps_vals = [float(d["NPS"]) for d in cs_vals
                if d.get("SR") == 1 and d.get("NPS") is not None]
    Ts   = len(nps_vals)
    SR   = round(sum(sr_vals) / T, 4)
    ANPS = _mean(nps_vals)
    ACS  = round(0.5 * SR + 0.5 * ANPS, 4) if ANPS is not None else None

    # Successful runs only (for model metrics)
    success = [r for r in runs
               if (r.get("results", {}).get("cs") or {}).get("SR") == 1]

    # Primary metric (varies by competition – just take whatever key is there)
    primary_agg: dict[str, float | None] = {}
    for r in success:
        for k, v in (r.get("results", {}).get("primary") or {}).items():
            if v is not None:
                primary_agg.setdefault(k, []).append(float(v))  # type: ignore[arg-type]
    primary_mean = {k: _mean(vs) for k, vs in primary_agg.items()}  # type: ignore[arg-type]

    # Additional metrics
    additional_agg: dict[str, list[float]] = {}
    for r in success:
        for k, v in (r.get("results", {}).get("additional") or {}).items():
            if v is not None:
                try:
                    additional_agg.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    pass
    additional_mean = {k: _mean(vs) for k, vs in additional_agg.items()}

    # Interpretability (use all runs that have the data)
    pa_vals  = _collect(runs, ["results", "interpretability", "plan_adherence",   "score"])
    sas_vals = _collect(runs, ["results", "interpretability", "semantic_alignment","score"])
    brm_vals = _collect(runs, ["results", "interpretability", "bat_report_maturity","score"])
    l1_vals  = _collect(runs, ["results", "interpretability", "bat_report_maturity","l1_score"])
    l2_vals  = _collect(runs, ["results", "interpretability", "bat_report_maturity","l2_score"])
    l3_vals  = _collect(runs, ["results", "interpretability", "bat_report_maturity","l3_score"])

    interp_mean = {
        "PA":  _mean(pa_vals),
        "SAS": _mean(sas_vals),
        "BRM": _mean(brm_vals),
        "L1":  _mean(l1_vals),
        "L2":  _mean(l2_vals),
        "L3":  _mean(l3_vals),
    }

    # Resource (all runs)
    rt_vals  = _collect(runs, ["results", "resource", "runtime", "total_seconds"])
    tok_vals = _collect(runs, ["results", "resource", "tokens", "total"])
    resource_mean = {
        "runtime_s": _mean(rt_vals),
        "tokens":    _mean(tok_vals),
    }

    return {
        "T": T, "Ts": Ts,
        "SR": SR, "ANPS": ANPS, "ACS": ACS,
        "primary":    primary_mean,
        "additional": additional_mean,
        "interp":     interp_mean,
        "resource":   resource_mean,
    }


# ── Printing ─────────────────────────────────────────────────────────────────

def _fmt(v: float | None, width: int = 6) -> str:
    return f"{v:.4f}" if v is not None else "   N/A"


def _print_group(comp: str, sys_: str, runs: list[dict], agg: dict) -> None:
    SEP2 = "-" * 50

    print(f"\n  Competition : {comp}")
    print(f"  System      : {sys_}")
    print(f"  {SEP2}")
    print(f"    Runs (T)  : {agg['T']}   (successful Ts={agg['Ts']})")

    print(f"\n  [CS COMPONENTS]")
    print(f"    SR    : {agg['SR']:.4f}   (mean over all T runs)")
    print(f"    ANPS  : {_fmt(agg['ANPS'])}   (mean NPS over Ts successful runs)")
    print(f"    ACS   : {_fmt(agg['ACS'])}   (0.5 x SR + 0.5 x ANPS)")

    if agg["primary"]:
        print(f"\n  [PRIMARY]  (mean over successful runs)")
        for k, v in agg["primary"].items():
            print(f"    {k:<20}: {_fmt(v)}")

    if agg["additional"]:
        print(f"\n  [ADDITIONAL]  (mean over successful runs)")
        for k, v in agg["additional"].items():
            print(f"    {k:<20}: {_fmt(v)}")

    interp = agg["interp"]
    if any(v is not None for v in interp.values()):
        print(f"\n  [INTERPRETABILITY]  (mean over runs with data)")
        print(f"    PA  (Plan Adherence)       : {_fmt(interp['PA'])}")
        print(f"    SAS (Semantic Alignment)   : {_fmt(interp['SAS'])}")
        print(f"    BRM (Report Maturity)      : {_fmt(interp['BRM'])}")
        print(f"      L1  Evidence             : {_fmt(interp['L1'])}")
        print(f"      L2  Interpretation       : {_fmt(interp['L2'])}")
        print(f"      L3  Action               : {_fmt(interp['L3'])}")

    res = agg["resource"]
    if res["runtime_s"] is not None or res["tokens"] is not None:
        print(f"\n  [RESOURCE]  (mean over all runs)")
        if res["runtime_s"] is not None:
            print(f"    Runtime : {res['runtime_s']:.1f}s")
        if res["tokens"] is not None:
            print(f"    Tokens  : {int(res['tokens'])}")

    # Per-run breakdown
    print(f"\n  [PER-RUN BREAKDOWN]")
    print(f"    {'#':<4} {'case_id':<18} {'SR':>4} {'NPS':>8} {'CS':>8} {'BRM':>8}")
    print(f"    {'----':<4} {'------------------':<18} {'----':>4} {'--------':>8} {'--------':>8} {'--------':>8}")
    for i, r in enumerate(runs, 1):
        cid   = r.get("case_id", "?")
        cs_d  = r.get("results", {}).get("cs") or {}
        sr    = int(cs_d.get("SR") or 0)
        nps   = cs_d.get("NPS")
        cs    = cs_d.get("CS")
        brm   = ((r.get("results", {})
                  .get("interpretability", {})
                  .get("bat_report_maturity") or {})
                 .get("score"))
        print(f"    {i:<4} {cid:<18} {sr:>4} {_fmt(nps):>8} {_fmt(cs):>8} {_fmt(brm):>8}")


# ── Main ─────────────────────────────────────────────────────────────────────

def aggregate(
    records_file: str = _DEFAULT_RECORDS,
    competition: str | None = None,
    system_name: str | None = None,
) -> dict[tuple[str, str], dict]:
    records = _load_records(records_file)

    if competition:
        records = [r for r in records if r.get("competition") == competition]
    if system_name:
        records = [r for r in records
                   if r.get("system", {}).get("name") == system_name]

    if not records:
        print("No matching records found.")
        return {}

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        comp = r.get("competition", "unknown")
        sys_ = r.get("system", {}).get("name", "unknown")
        groups.setdefault((comp, sys_), []).append(r)

    SEP = "=" * 66
    all_agg: dict[tuple[str, str], dict] = {}

    print(f"\n{SEP}")
    print(f"  AGGREGATE BENCHMARK RESULTS  ({len(records)} records)")
    print(SEP)

    for (comp, sys_), runs in sorted(groups.items()):
        agg = _aggregate_group(runs)
        all_agg[(comp, sys_)] = agg
        _print_group(comp, sys_, runs, agg)

    print(f"\n{SEP}\n")
    return all_agg


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark results across runs (ANPS/ACS + all metrics).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aggregate_results.py
  python aggregate_results.py --file artifacts/workspace/experiment_records.jsonl
  python aggregate_results.py --competition bank_churn
  python aggregate_results.py --competition bank_churn --system-name AutoKaggle
""",
    )
    parser.add_argument(
        "--file", default=_DEFAULT_RECORDS,
        help="Path to experiment_records.jsonl",
    )
    parser.add_argument("--competition", default=None, help="Filter by competition key")
    parser.add_argument("--system-name", default=None, dest="system_name",
                        help="Filter by system name")
    args = parser.parse_args()

    aggregate(
        records_file=args.file,
        competition=args.competition,
        system_name=args.system_name,
    )
