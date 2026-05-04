"""
Static EDA chart generation (matplotlib + seaborn).

Produces PNG files under ``workspace/eda_figs/`` and a markdown-like text summary.
Designed to be deterministic and resilient: one failed chart does not block others.

Statistics used for optional charts (feature-target correlation, RF importance)
should be supplied via ``precomputed_stats`` from ``tools.eda_statistics.compute_eda_statistics``.
"""
from __future__ import annotations

import logging
import os
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from tools.eda_statistics import compute_eda_statistics, infer_target_column

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Matplotlib / font setup (Windows + Linux friendly fallbacks)
# -----------------------------------------------------------------------------


def _setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["axes.unicode_minus"] = False
    for fam in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
        try:
            plt.rcParams["font.sans-serif"] = [fam]
            break
        except Exception:
            continue


def generate_eda_charts(
    data_path: str,
    output_dir: str | None = None,
    target_col: str | None = None,
    precomputed_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Read cleaned train CSV, write standard EDA PNGs, return chart metadata and summary text.

    Returns:
        {
            "charts": [{"name": "...", "rel_path": "./eda_figs/...", "desc": "..."}, ...],
            "text_summary": "markdown-like summary string",
        }
    """
    _setup_plot_style()

    if output_dir is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(root, "workspace", "eda_figs")

    os.makedirs(output_dir, exist_ok=True)

    charts: list[dict[str, str]] = []
    summary_lines: list[str] = ["## EDA Summary (static charts)\n"]

    if not os.path.isfile(data_path):
        msg = f"Train data not found at `{data_path}`; skipping EDA charts."
        logger.warning(msg)
        summary_lines.append(msg)
        return {"charts": charts, "text_summary": "\n".join(summary_lines)}

    try:
        df = pd.read_csv(data_path)
    except Exception as exc:
        msg = f"Failed to read `{data_path}`: {exc}"
        logger.warning(msg)
        summary_lines.append(msg)
        return {"charts": charts, "text_summary": "\n".join(summary_lines)}

    if df.empty:
        summary_lines.append("Dataset is empty after load; no charts generated.")
        return {"charts": charts, "text_summary": "\n".join(summary_lines)}

    inferred = infer_target_column(df) if target_col is None else target_col
    if inferred is not None and inferred not in df.columns:
        inferred = None
    if target_col is not None and target_col in df.columns:
        inferred = target_col

    summary_lines.append(f"- **shape**: {df.shape[0]} rows × {df.shape[1]} columns")
    summary_lines.append(f"- **missing_cells_total**: {int(df.isna().sum().sum())}")
    summary_lines.append(
        f"- **inferred_target_col**: `{inferred}`" if inferred else "- **inferred_target_col**: (none)"
    )

    rel_prefix = "./eda_figs"

    def _save_chart(name: str, desc: str, fn: Any) -> None:
        path = os.path.join(output_dir, name)
        try:
            plt.close("all")
            fn()
            plt.tight_layout()
            plt.savefig(path, bbox_inches="tight", dpi=150)
            charts.append(
                {
                    "name": name,
                    "rel_path": f"{rel_prefix}/{name}",
                    "desc": desc,
                }
            )
        except Exception as exc:
            logger.warning("EDA chart %s failed: %s", name, exc)
        finally:
            plt.close("all")

    # 1) Missing values (top columns by missing rate)
    def _missing_plot() -> None:
        miss = df.isna().mean().sort_values(ascending=False)
        miss = miss[miss > 0].head(30)
        if miss.empty:
            ax = plt.gca()
            ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
            ax.axis("off")
            return
        plt.figure(figsize=(10, max(3, 0.35 * len(miss))))
        miss.plot(kind="barh", color="steelblue")
        plt.xlabel("Fraction missing")
        plt.ylabel("Column")
        plt.title("Missing value rate (top 30 columns)")

    _save_chart("missing_values.png", "Missing value rates by column", _missing_plot)

    # 2) Target distribution
    if inferred:
        s = df[inferred]

        def _target_plot() -> None:
            plt.figure(figsize=(8, 5))
            if pd.api.types.is_numeric_dtype(s):
                pd.to_numeric(s, errors="coerce").dropna().hist(bins=40, color="teal", edgecolor="white")
                plt.title(f"Target distribution: {inferred}")
                plt.xlabel(inferred)
            else:
                vc = s.astype(str).value_counts().head(20)
                vc.plot(kind="bar", color="coral", edgecolor="white")
                plt.title(f"Target category counts (top 20): {inferred}")
                plt.xticks(rotation=45, ha="right")
            plt.ylabel("Count")

        _save_chart(
            "target_distribution.png",
            f"Distribution of target `{inferred}`",
            _target_plot,
        )

        # Class balance (categorical / low-cardinality string)
        def _class_balance_plot() -> None:
            if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 25:
                ax = plt.gca()
                ax.text(0.5, 0.5, "Target is numeric with many unique values\n(class balance skipped)", ha="center", va="center")
                ax.axis("off")
                return
            vc = s.astype(str).value_counts()
            plt.figure(figsize=(8, 5))
            vc.plot(kind="bar", color="seagreen", edgecolor="white")
            plt.title(f"Class / category balance: {inferred}")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")

        _save_chart(
            "class_balance.png",
            "Category balance for target / key label column",
            _class_balance_plot,
        )

    # 3) Numeric distributions grid
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if inferred and inferred in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != inferred]
    take = numeric_cols[:6]

    def _numeric_grid() -> None:
        if not take:
            ax = plt.gca()
            ax.text(0.5, 0.5, "No numeric columns for histogram grid", ha="center", va="center")
            ax.axis("off")
            return
        n = len(take)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        plt.figure(figsize=(4 * cols, 3 * rows))
        for i, col in enumerate(take):
            ax = plt.subplot(rows, cols, i + 1)
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.axis("off")
                continue
            series.hist(bins=30, color="slateblue", edgecolor="white", ax=ax)
            ax.set_title(str(col)[:40])
        plt.suptitle("Numeric feature distributions (sample)")

    _save_chart(
        "numeric_distributions.png",
        "Histograms for up to six numeric features",
        _numeric_grid,
    )

    # 4) Correlation heatmap (top-N numeric by variance)
    def _corr_plot() -> None:
        num_df = df.select_dtypes(include=[np.number]).copy()
        if inferred and inferred in num_df.columns:
            num_df = num_df.drop(columns=[inferred])
        if num_df.shape[1] < 2:
            ax = plt.gca()
            ax.text(0.5, 0.5, "Need ≥2 numeric columns for correlation heatmap", ha="center", va="center")
            ax.axis("off")
            return
        variances = num_df.var(numeric_only=True).sort_values(ascending=False)
        top_cols = variances.head(min(15, len(variances))).index.tolist()
        sub = df[top_cols].apply(pd.to_numeric, errors="coerce")
        corr = sub.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=len(top_cols) <= 12, fmt=".2f", cmap="RdBu_r", center=0, square=False)
        plt.title("Correlation heatmap (top numeric columns by variance)")

    _save_chart(
        "corr_heatmap.png",
        "Pearson correlation heatmap for top numeric columns",
        _corr_plot,
    )

    # 5) Boxplots for outlier view
    def _box_plot() -> None:
        cols_box = numeric_cols[:6]
        if not cols_box:
            ax = plt.gca()
            ax.text(0.5, 0.5, "No numeric columns for boxplots", ha="center", va="center")
            ax.axis("off")
            return
        plt.figure(figsize=(12, 6))
        melt = df[cols_box].apply(pd.to_numeric, errors="coerce")
        std = melt.std()
        std = std.replace(0, np.nan)
        melt_norm = (melt - melt.mean()) / std
        melt_norm.boxplot(rot=45)
        plt.title("Standardized numeric columns (boxplots; sample)")

    _save_chart(
        "outlier_boxplots.png",
        "Boxplots for standardized numeric features (sample)",
        _box_plot,
    )

    # 6) Feature–target correlation (from pre-computed EDA stats)
    if precomputed_stats:
        tc_list = (precomputed_stats.get("target_correlation") or [])[:15]
        if tc_list:

            def _ftarget_plot() -> None:
                plt.figure(figsize=(9, max(4, 0.4 * len(tc_list))))
                feats = [str(d.get("feature", "")) for d in tc_list]
                vals = [float(d.get("pearson_with_target", 0)) for d in tc_list]
                y_pos = np.arange(len(feats))
                colors = ["coral" if v < 0 else "steelblue" for v in vals]
                plt.barh(y_pos, vals, color=colors)
                plt.yticks(y_pos, [f[:35] for f in feats], fontsize=8)
                plt.xlabel("Pearson correlation with target")
                plt.title("Top feature–target linear correlation")
                plt.axvline(0, color="gray", lw=0.6)
                plt.tight_layout()

            _save_chart(
                "feature_target_corr.png",
                "Top feature–target Pearson correlation (train)",
                _ftarget_plot,
            )

        rf_list = (precomputed_stats.get("rf_importance_top15") or [])[:15]
        if rf_list:

            def _rf_plot() -> None:
                plt.figure(figsize=(9, max(4, 0.4 * len(rf_list))))
                feats = [str(d.get("feature", "")) for d in rf_list]
                imp = [float(d.get("importance", 0)) for d in rf_list]
                y_pos = np.arange(len(feats))
                plt.barh(y_pos, imp, color="seagreen")
                plt.yticks(y_pos, [f[:35] for f in feats], fontsize=8)
                plt.xlabel("Importance")
                plt.title("RandomForest feature importance (quick n_estimators=50)")
                plt.tight_layout()

            _save_chart(
                "rf_importance.png",
                "RandomForest quick feature importance (train)",
                _rf_plot,
            )

    summary_lines.append("\n### Generated charts\n")
    if charts:
        for c in charts:
            summary_lines.append(f"- `{c['rel_path']}` — {c['desc']}")
    else:
        summary_lines.append("- (no chart files were produced)")

    return {"charts": charts, "text_summary": "\n".join(summary_lines)}


def purge_eda_figs_dir(workspace_root: str | None = None) -> None:
    """Remove ``workspace/eda_figs`` recursively (best-effort)."""
    if workspace_root is None:
        workspace_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace"
        )
    target = os.path.join(workspace_root, "eda_figs")
    if os.path.isdir(target):
        try:
            shutil.rmtree(target)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", target, exc)
