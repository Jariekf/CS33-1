"""
EDA Analysis node — deep deterministic statistics + LLM conclusions after data cleaning passes.

Writes ``workspace/eda_report.md`` with four sections: descriptive, correlation, diagnosis, conclusions.
"""
from __future__ import annotations

import os
import re
from typing import Any

from core.logger import get_agent_logger
from core.state import GraphState
from core.llm import call_llm
from agents.data_cleaning import CLEANED_OUTPUT_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH
from config import EDA_REPORT_PATH, WORKSPACE_DIR
from tools.eda_charts import generate_eda_charts
from tools.eda_statistics import compute_eda_statistics, stats_dict_to_json_for_prompt

logger = get_agent_logger("eda_analysis")

_SYSTEM_INTERPRET = """\
You are a senior data scientist. You receive **pre-computed** EDA statistics and facts in JSON/Markdown — \
do not re-calculate numbers. Your job is to interpret them and give actionable guidance for downstream \
feature engineering and model selection. Output in English. Be concise and evidence-based. \
"""

_USER_INTERPRET_TEMPLATE = """\
# User goal
{user_goal}

# Dataset description (authoritative, may be empty)
{dataset_description}

# Pre-computed EDA facts (JSON)
{facts_json}

# Instruction
Write the section below in Markdown exactly with these headings (fill content):

## EDA Conclusions

### Task Type
Choose one: classification | regression | clustering | time-series | unknown
Brief reasoning (2-4 sentences) referencing the facts above.

### Candidate Models
Bullet list of 2-4 model families suitable for this task and data scale (reference sample size and collinearity if relevant).

### Recommended Metrics
- Primary: name + one-line reason
- Secondary: name + one-line reason

### Key Warnings
Bullet list of data risks (missing values, drift, imbalance, multicollinearity, low samples per feature, etc.).

Do not invent statistics not present in the facts JSON.
"""


def _fmt3(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _format_facts_markdown(stats: dict[str, Any]) -> str:
    """Deterministic Markdown body: sections 1–3 for the report."""
    lines: list[str] = []

    # --- 1. Descriptive ---
    lines.append("## 1. Descriptive Analysis")
    lines.append("")
    ss = stats.get("sample_sufficiency") or {}
    train_shape = ss.get("train_shape") or {}
    test_shape = ss.get("test_shape") or {}
    lineage = ss.get("lineage_shapes") or {}
    lines.append(
        f"- **Shape (train):** inferred from pipeline  \n"
        f"- **Rows / features:** {ss.get('n_rows', '?')} / {ss.get('n_features', '?')}  \n"
        f"- **Rows per feature:** {ss.get('rows_per_feature', '?')}  "
        f"{'**(warning: low)**' if ss.get('warning_low_samples_per_feature') else ''}"
    )
    if train_shape:
        lines.append(f"- **Train shape:** {train_shape.get('rows', '?')} rows × {train_shape.get('columns', '?')} columns")
    if test_shape:
        lines.append(f"- **Test shape:** {test_shape.get('rows', '?')} rows × {test_shape.get('columns', '?')} columns")
    if lineage:
        lines.append("")
        lines.append("### Raw -> cleaned -> split shape summary")
        lines.append("")
        for label in ("raw", "cleaned", "train", "test"):
            shape = lineage.get(label)
            if shape:
                lines.append(f"- **{label}:** {shape.get('rows', '?')} rows × {shape.get('columns', '?')} columns")
        lines.append("")
    lines.append(f"- **Inferred target column:** `{stats.get('inferred_target')}`")
    lines.append(f"- **Target type hint (heuristic):** `{stats.get('target_type_hint')}`")
    if stats.get("target_type_warning"):
        lines.append(f"- **Target type warning:** {stats.get('target_type_warning')}")
    lines.append("")

    desc = stats.get("descriptive_stats") or {}
    if desc:
        lines.append("### Numeric summary (key columns)")
        lines.append("")
        lines.append("| Column | Mean | Std | Min | Median | Max |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for col, d in list(desc.items())[:12]:
            if isinstance(d, dict) and "mean" in d:
                lines.append(
                    f"| `{col}` | {_fmt3(d.get('mean'))} | {_fmt3(d.get('std'))} | "
                    f"{_fmt3(d.get('min'))} | {_fmt3(d.get('50%'))} | {_fmt3(d.get('max'))} |"
                )
        lines.append("")

    skew = stats.get("skewness") or {}
    if skew:
        lines.append("### Skewness (numeric)")
        lines.append("")
        for col, d in list(skew.items())[:12]:
            lines.append(f"- `{col}`: skew={d.get('skewness')} ({d.get('direction')})")
        lines.append("")

    cat = stats.get("category_summary") or {}
    if cat:
        lines.append("### Categorical columns (sample)")
        lines.append("")
        for col, d in list(cat.items())[:12]:
            lines.append(
                f"- `{col}`: nunique={d.get('nunique')}, "
                f"highly_imbalanced={d.get('is_highly_imbalanced')}"
            )
        lines.append("")

    lines.append(
        "*Charts:* `./eda_figs/numeric_distributions.png`, "
        "`./eda_figs/outlier_boxplots.png`, `./eda_figs/categorical_distributions.png`, "
        "`./eda_figs/class_balance.png`, `./eda_figs/target_distribution.png`, "
        "`./eda_figs/missing_values.png`"
    )
    lines.append("")

    # --- 2. Correlation ---
    lines.append("## 2. Correlation and relationship analysis")
    lines.append("")
    for name, key in (
        ("Pearson (top pairs)", "pearson_corr_top"),
        ("Spearman (top pairs)", "spearman_corr_top"),
    ):
        items = stats.get(key) or []
        lines.append(f"### {name}")
        lines.append("")
        for p in (items or [])[:10]:
            coeff = p.get("coeff")
            if isinstance(coeff, (int, float)):
                lines.append(f"- `{p.get('col_a')}` vs `{p.get('col_b')}`: {float(coeff):.4f}")
            else:
                lines.append(f"- `{p.get('col_a')}` vs `{p.get('col_b')}`: {coeff}")
        lines.append("")

    colin = stats.get("collinear_pairs") or []
    lines.append("### Highly collinear pairs (|r| > 0.95)")
    lines.append("")
    if colin:
        for p in colin[:20]:
            lines.append(f"- `{p.get('col_a')}` vs `{p.get('col_b')}`: r={p.get('coeff')}")
    else:
        lines.append("- (none detected among numeric features used)")
    lines.append("")

    tcorr = stats.get("target_correlation") or []
    lines.append("### Feature–target (Pearson) top 10")
    lines.append("")
    if tcorr:
        lines.append("| Feature | Pearson r |")
        lines.append("|---|---:|")
    for p in tcorr[:10]:
        lines.append(f"| `{p.get('feature')}` | {_fmt3(p.get('pearson_with_target'))} |")
    lines.append("")
    num_effects = stats.get("numeric_feature_class_effects") or []
    if num_effects:
        lines.append("### Numeric feature vs classification target effect size (Omega squared)")
        lines.append("")
        lines.append("| Feature | Omega squared |")
        lines.append("|---|---:|")
        for p in num_effects[:15]:
            lines.append(f"| `{p.get('feature')}` | {_fmt3(p.get('omega_squared'))} |")
        lines.append("")
    cat_assoc = stats.get("categorical_target_associations") or []
    if cat_assoc:
        lines.append("### Categorical feature vs classification target association (Cramer's V)")
        lines.append("")
        lines.append("| Feature | Cramer's V |")
        lines.append("|---|---:|")
        for p in cat_assoc[:15]:
            lines.append(f"| `{p.get('feature')}` | {_fmt3(p.get('cramers_v'))} |")
        lines.append("")
    lines.append(
        "*Charts:* `./eda_figs/corr_heatmap.png`, `./eda_figs/feature_target_corr.png`, "
        "`./eda_figs/numeric_by_target_density.png`, `./eda_figs/categorical_by_target_rate.png`, "
        "`./eda_figs/numeric_vs_target_regression.png`, `./eda_figs/categorical_vs_target_boxplot.png`"
    )
    lines.append("")

    # --- 3. Diagnosis ---
    lines.append("## 3. Problem diagnosis")
    lines.append("")
    cb = stats.get("class_balance") or {}
    if cb:
        lines.append("### Target / label distribution (relative frequencies)")
        lines.append("")
        for k, v in list(cb.items())[:25]:
            lines.append(f"- `{k}`: {float(v):.4f}")
        lines.append("")
    rf = stats.get("rf_importance_top15") or []
    lines.append("### Quick RandomForest importance (top 15, train-only exploratory fit)")
    lines.append("")
    if stats.get("rf_error"):
        lines.append(f"*RF note:* {stats.get('rf_error')}")
        lines.append("")
    for p in rf:
        lines.append(f"- `{p.get('feature')}`: {float(p.get('importance', 0)):.6f}")
    lines.append("")

    dt = stats.get("datetime_columns") or []
    lines.append(f"### Datetime-like columns (name heuristic): {dt or '(none)'}")
    lines.append("")

    drift = stats.get("train_test_drift") or []
    lines.append("### Train vs test distribution drift (KS / chi²)")
    lines.append("")
    if drift:
        for d in drift[:25]:
            flag = " **DRIFT**" if d.get("drift_warning") else ""
            lines.append(
                f"- `{d.get('col')}` ({d.get('test')}): p={d.get('p_value')}{flag}"
            )
    else:
        lines.append("- (no test file or no columns tested)")
    lines.append("")
    lines.append("*Charts:* `./eda_figs/rf_importance.png`, `./eda_figs/outlier_boxplots.png`")
    lines.append("")

    return "\n".join(lines)


def _semantic_task_context_from_state(state: GraphState) -> str:
    """Collect weak task-type hints without using Manager's evaluation_metric."""
    parts: list[str] = []
    user_goal = (state.get("user_goal") or "").strip()
    if user_goal:
        parts.append(user_goal)

    plan = state.get("global_plan") or {}
    if isinstance(plan, dict):
        for section_name in (
            "data_cleaning_plan",
            "feature_engineering_plan",
            "model_selection_plan",
        ):
            section = plan.get(section_name) or {}
            if not isinstance(section, dict):
                continue
            objectives = section.get("objectives") or []
            if isinstance(objectives, list):
                parts.extend(str(obj) for obj in objectives)
            elif objectives:
                parts.append(str(objectives))

    return "\n".join(parts)


def _heuristic_conclusions(stats: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Fallback when LLM unavailable."""
    hint = stats.get("target_type_hint") or "unknown"
    tgt = stats.get("inferred_target")
    dt_cols = stats.get("datetime_columns") or []
    warns: list[str] = []

    ss = stats.get("sample_sufficiency") or {}
    if ss.get("warning_low_samples_per_feature"):
        warns.append("Low rows-per-feature ratio; risk of overfitting.")

    colin = stats.get("collinear_pairs") or []
    if colin:
        warns.append(f"Multicollinearity: {len(colin)} pair(s) with |r|>0.95.")

    drift = stats.get("train_test_drift") or []
    nd = sum(1 for d in drift if d.get("drift_warning"))
    if nd:
        warns.append(f"Train/test drift flagged on {nd} column(s) (p<0.05).")

    cb = stats.get("class_balance") or {}
    if cb and len(cb) >= 2:
        mx = max(float(v) for v in cb.values())
        if mx > 0.85:
            warns.append("Possible class imbalance in target.")

    if not tgt:
        task = "clustering"
        models = ["KMeans", "GaussianMixture", "DBSCAN"]
        metrics = {"primary": "Silhouette (internal)", "secondary": "Davies-Bouldin"}
    elif dt_cols:
        task = "time_series"
        models = ["ARIMA / seasonal models", "Gradient boosting on lag features"]
        metrics = {"primary": "RMSE / MAPE", "secondary": "MAE"}
        warns.append("Datetime-like column names detected; validate temporal CV / leakage.")
    elif hint == "classification" or (hint == "unknown" and cb):
        task = "classification"
        models = ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"]
        metrics = {"primary": "F1 macro / balanced accuracy", "secondary": "ROC-AUC (binary/multiclass OVR)"}
    else:
        task = "regression"
        models = ["Ridge / ElasticNet", "RandomForestRegressor", "GradientBoostingRegressor"]
        metrics = {"primary": "RMSE / MAE", "secondary": "R²"}

    body = f"""## EDA Conclusions

### Task Type
**{task}** (heuristic fallback; inferred target: `{tgt}`; hint: `{hint}`)

### Candidate Models
{chr(10).join('- ' + m for m in models)}

### Recommended Metrics
- Primary: **{metrics['primary']}**
- Secondary: **{metrics['secondary']}**

### Key Warnings
{chr(10).join('- ' + w for w in warns) if warns else '- (none flagged by heuristics)'}
"""
    conclusions = {
        "task_type": task,
        "candidate_models": models,
        "recommended_metrics": metrics,
        "warnings": warns,
        "conclusion_markdown": body.strip(),
        "source": "heuristic",
    }
    return body, conclusions


def _llm_interpret_eda(
    facts_json: str,
    user_goal: str,
    dataset_description: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """Returns (markdown conclusions section or None, structured dict or None)."""
    user_prompt = _USER_INTERPRET_TEMPLATE.format(
        user_goal=user_goal or "(not specified)",
        dataset_description=dataset_description.strip()
        or "(not provided)",
        facts_json=facts_json,
    )
    reply = call_llm(_SYSTEM_INTERPRET, user_prompt, agent_name="eda_analysis")
    if not reply:
        return None, None
    text = reply.strip()
    # Structured extraction (best-effort)
    conclusions: dict[str, Any] = {
        "task_type": "unknown",
        "candidate_models": [],
        "recommended_metrics": {"primary": "", "secondary": ""},
        "warnings": [],
        "conclusion_markdown": text,
        "source": "llm",
    }
    tt = re.search(
        r"### Task Type\s*\n([^\n]+)",
        text,
        re.I | re.M,
    )
    if tt:
        line = tt.group(1).strip()
        for kw in ("classification", "regression", "clustering", "time-series", "unknown"):
            if kw in line.lower():
                conclusions["task_type"] = kw if kw != "time-series" else "time_series"
                break
    cm = re.findall(r"^-\s*(.+)$", text, re.M)
    if cm:
        conclusions["candidate_models"] = [c.strip() for c in cm[:8] if len(c) < 120]

    pm = re.search(r"Primary:\s*\*?\*?([^*\n]+)", text, re.I)
    sm = re.search(r"Secondary:\s*\*?\*?([^*\n]+)", text, re.I)
    if pm:
        conclusions["recommended_metrics"]["primary"] = pm.group(1).strip()
    if sm:
        conclusions["recommended_metrics"]["secondary"] = sm.group(1).strip()

    return text, conclusions


def _write_eda_report_md(
    train_path: str,
    sections_body: str,
    charts: list[dict],
) -> None:
    """Persist full four-part report."""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    lines = [
        "# EDA Report",
        "",
        f"**Train split:** `{train_path}`",
        "",
        "**Figures:** `./eda_figs/`",
        "",
        "---",
        "",
        sections_body.strip(),
        "",
    ]
    if charts:
        lines.extend(
            [
                "---",
                "",
                "## Chart file index",
                "",
                "| File | Path | Description |",
                "|------|------|-------------|",
            ]
        )
        for c in charts:
            rel = (c.get("rel_path") or "").strip()
            desc = (c.get("desc") or "").strip()
            name = (c.get("name") or "").strip()
            desc_safe = desc.replace("|", "/")
            lines.append(f"| `{name}` | `{rel}` | {desc_safe} |")
        chart_names = {str(c.get("name") or "").strip() for c in charts}
        if "missing_values.png" not in chart_names:
            lines.append(
                "| `missing_values.png` | (not generated) | Missing-value chart skipped because no missing values were detected in the training split. |"
            )

    with open(EDA_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def eda_analysis_node(state: GraphState) -> dict:
    """Deep EDA: statistics + charts + LLM conclusions."""
    logger.info("EDA analysis node started.")

    user_goal = (state.get("user_goal") or "").strip()
    dataset_description = (state.get("dataset_description") or "").strip()

    semantic_context = _semantic_task_context_from_state(state)
    stats = compute_eda_statistics(
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        target_col=None,
        raw_path=state.get("data_path"),
        cleaned_path=CLEANED_OUTPUT_PATH,
        semantic_context=semantic_context,
        dataset_description=dataset_description,
    )
    if stats.get("error"):
        logger.warning("EDA statistics error: %s", stats.get("error"))

    chart_result = generate_eda_charts(
        TRAIN_DATA_PATH,
        test_data_path=TEST_DATA_PATH,
        target_col=stats.get("inferred_target"),
        precomputed_stats=stats,
    )
    charts = chart_result.get("charts") or []
    chart_summary = (chart_result.get("text_summary") or "").strip()

    facts_md = _format_facts_markdown(stats)
    facts_json = stats_dict_to_json_for_prompt(stats, max_chars=14000)

    h_body, h_conclusions = _heuristic_conclusions(stats)
    interpret_md, llm_c = _llm_interpret_eda(facts_json, user_goal, dataset_description)
    if interpret_md and isinstance(llm_c, dict):
        section4 = interpret_md
        eda_conclusions = {**h_conclusions, **llm_c}
        eda_conclusions["conclusion_markdown"] = interpret_md.strip()
        eda_conclusions["source"] = "llm"
    elif interpret_md:
        section4 = interpret_md
        eda_conclusions = {**h_conclusions}
        eda_conclusions["conclusion_markdown"] = interpret_md.strip()
        eda_conclusions["source"] = "llm"
    else:
        logger.info("LLM interpret unavailable; using heuristic EDA conclusions.")
        section4 = h_body
        eda_conclusions = h_conclusions

    # Full markdown for downstream agents (facts + chart list + conclusions)
    full_summary_parts = [
        facts_md,
        "",
        "### Generated chart files",
        "",
        chart_summary,
        "",
        "---",
        "",
        section4,
    ]
    eda_summary_full = "\n".join(full_summary_parts)

    report_body = facts_md + "\n\n---\n\n" + section4
    try:
        _write_eda_report_md(TRAIN_DATA_PATH, report_body, charts)
        logger.info("EDA report written to %s", EDA_REPORT_PATH)
    except OSError as exc:
        logger.warning("Failed to write EDA report file: %s", exc)

    return {
        "current_phase": "eda_analysis",
        "eda_charts": charts,
        "eda_summary": eda_summary_full,
        "eda_conclusions": eda_conclusions,
    }
