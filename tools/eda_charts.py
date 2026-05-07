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
import textwrap
import warnings
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


LOW_CARDINALITY_TARGET_MAX_CLASSES = 5
MAX_GRID_FEATURES = 6
MAX_CATEGORICAL_GRID_FEATURES = 8
MAX_CATEGORICAL_TARGET_FEATURES = 8
MAX_CATEGORY_LEVELS = 10


def _wrap_axis_label(label: object, width: int = 11) -> str:
    text = str(label)
    if len(text) <= width:
        return text
    wrapped = textwrap.wrap(text.replace("_", " "), width=width, break_long_words=False)
    return "\n".join(part.replace(" ", "_") for part in wrapped) if wrapped else text


def _numeric_variance_order(df: pd.DataFrame, columns: list[str]) -> list[str]:
    scored: list[tuple[float, str]] = []
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        variance = series.var()
        score = float(variance) if pd.notna(variance) else float("-inf")
        scored.append((score, str(col)))
    return [col for _, col in sorted(scored, key=lambda item: (-item[0], item[1]))]


def _select_numeric_distribution_columns(df: pd.DataFrame, columns: list[str], limit: int = MAX_GRID_FEATURES) -> list[str]:
    return _numeric_variance_order(df, columns)[:limit]


def _select_categorical_distribution_columns(
    df: pd.DataFrame,
    columns: list[str],
    limit: int = MAX_CATEGORICAL_GRID_FEATURES,
) -> list[str]:
    scored: list[tuple[int, str]] = []
    for col in columns:
        if col not in df.columns:
            continue
        nunique = int(df[col].nunique(dropna=True))
        if nunique <= 1:
            continue
        scored.append((nunique, str(col)))
    return [col for _, col in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]


def _rank_columns_by_stat_rows(
    candidates: list[str],
    rows: list[dict[str, Any]] | None,
    score_key: str,
    limit: int,
) -> list[str]:
    candidate_set = {str(c) for c in candidates}
    scored: list[tuple[float, str]] = []
    for row in rows or []:
        feature = str(row.get("feature", ""))
        if feature not in candidate_set:
            continue
        try:
            score = abs(float(row.get(score_key, 0)))
        except (TypeError, ValueError):
            continue
        scored.append((score, feature))
    ranked = [feature for _, feature in sorted(scored, key=lambda item: (-item[0], item[1]))]
    ranked_set = set(ranked)
    ranked.extend([col for col in candidates if col not in ranked_set])
    return ranked[:limit]


def _missing_overview_rows(df: pd.DataFrame, max_cols: int = 30) -> list[dict[str, Any]]:
    total = max(len(df), 1)
    rows = [
        {
            "column": str(col),
            "missing_count": int(df[col].isna().sum()),
            "missing_rate": float(df[col].isna().sum() / total),
        }
        for col in df.columns
    ]
    rows.sort(key=lambda row: (-int(row["missing_count"]), str(row["column"])))
    return rows[:max_cols]


def _is_low_cardinality_classification_target(s: pd.Series, target_type: str) -> bool:
    return target_type == "classification" and 2 <= int(s.nunique(dropna=True)) <= LOW_CARDINALITY_TARGET_MAX_CLASSES


def _resolve_target_type(df: pd.DataFrame, target_col: str | None, stats: dict[str, Any] | None) -> str:
    if stats and stats.get("target_type_hint"):
        return str(stats.get("target_type_hint"))
    if not target_col or target_col not in df.columns:
        return "unknown"
    s = df[target_col]
    return "regression" if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 25 else "classification"


def _top_categories(series: pd.Series, max_levels: int = MAX_CATEGORY_LEVELS) -> pd.Series:
    s = series.astype(str).fillna("__missing__")
    counts = s.value_counts()
    if len(counts) <= max_levels:
        return s
    keep = set(counts.head(max_levels - 1).index)
    return s.where(s.isin(keep), other="Other")


def _grid_shape(n: int, max_cols: int = 3) -> tuple[int, int]:
    cols = min(max_cols, max(1, n))
    rows = int(np.ceil(max(1, n) / cols))
    return rows, cols


def _sample_rows(df: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if pd.api.types.is_object_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(df[c])
    ]


def _target_display(stats: dict[str, Any] | None, fallback: str) -> str:
    if stats and stats.get("target_display_name"):
        return str(stats["target_display_name"])
    return fallback


def _class_label(value: object, stats: dict[str, Any] | None) -> str:
    raw = str(value)
    mapping = (stats or {}).get("target_class_labels") or {}
    label = mapping.get(raw, raw)
    return str(label)


def _title_set_label(title_prefix: str) -> str:
    return title_prefix.lower()


def generate_eda_charts(
    data_path: str,
    output_dir: str | None = None,
    target_col: str | None = None,
    precomputed_stats: dict[str, Any] | None = None,
    test_data_path: str | None = None,
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
    target_type = _resolve_target_type(df, inferred, precomputed_stats)
    target_label = _target_display(precomputed_stats, inferred or "target")

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
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Tight layout not applied.*", category=UserWarning)
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
        rows = _missing_overview_rows(df, max_cols=30)
        if not rows:
            ax = plt.gca()
            ax.text(0.5, 0.5, "No columns available", ha="center", va="center")
            ax.axis("off")
            return
        plot_rows = list(reversed(rows))
        labels = [row["column"] for row in plot_rows]
        counts = [int(row["missing_count"]) for row in plot_rows]
        rates = [float(row["missing_rate"]) for row in plot_rows]
        y_pos = np.arange(len(plot_rows))
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_rows))))
        ax.hlines(y_pos, 0, counts, color="#cfcfcf", linewidth=1.4)
        ax.scatter(counts, y_pos, color="black", s=22, zorder=3)
        for count, y, rate in zip(counts, y_pos, rates):
            offset = max(max(counts) * 0.015, 0.2)
            ax.text(count + offset, y, f"{rate * 100:.1f}%", va="center", fontsize=8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Number of Missing Values")
        ax.set_ylabel("Variable")
        ax.set_title("Missing Data Overview", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    if int(df.isna().sum().sum()) > 0:
        _save_chart("missing_values.png", "Missing value rates by column", _missing_plot)
    else:
        summary_lines.append("- **missing_values:** No missing values detected; missing-value chart skipped.")

    # 2) Target distribution / class balance
    if inferred:
        s = df[inferred]
        low_card_classification = _is_low_cardinality_classification_target(s, target_type)

        def _target_plot() -> None:
            plt.figure(figsize=(8, 5))
            if pd.api.types.is_numeric_dtype(s) and target_type == "regression":
                pd.to_numeric(s, errors="coerce").dropna().hist(bins=40, color="teal", edgecolor="white")
                plt.title(f"Training set target distribution: {target_label}")
                plt.xlabel(target_label)
            else:
                vc = s.astype(str).value_counts().head(20)
                vc.index = [_class_label(v, precomputed_stats) for v in vc.index]
                vc.plot(kind="bar", color="coral", edgecolor="white")
                plt.title(f"Training set target category counts: {target_label}")
                plt.xticks(rotation=45, ha="right")
            plt.ylabel("Count")

        if not low_card_classification:
            _save_chart(
                "target_distribution.png",
                f"Distribution of target `{inferred}`",
                _target_plot,
            )

        def _donut_plot(series: pd.Series, title_prefix: str) -> None:
            vc = series.astype(str).value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = sns.color_palette("Set2", n_colors=len(vc))
            wedges, _, autotexts = ax.pie(
                vc.values,
                labels=None,
                autopct=lambda pct: f"{pct:.1f}%",
                pctdistance=0.78,
                startangle=90,
                colors=colors,
                wedgeprops={"width": 0.42, "edgecolor": "white"},
            )
            for text in autotexts:
                text.set_color("white")
                text.set_fontsize(13)
                text.set_fontweight("bold")
            labels = [_class_label(x, precomputed_stats) for x in vc.index]
            for wedge, label, color in zip(wedges, labels, colors):
                theta = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
                ax.text(
                    0.43 * np.cos(theta),
                    0.43 * np.sin(theta),
                    label,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=color,
                )
            ax.text(0, 0, f"n={int(vc.sum())}", ha="center", va="center", fontsize=10)
            ax.set_title(
                f"{target_label} ({_title_set_label(title_prefix)})",
                fontsize=14,
                fontweight="bold",
            )
            ax.axis("equal")

        if low_card_classification:
            _save_chart(
                "class_balance.png",
                f"Training set donut chart for target balance: {target_label}",
                lambda: _donut_plot(s, "Training set"),
            )
            if test_data_path and os.path.isfile(test_data_path):
                try:
                    test_df = pd.read_csv(test_data_path)
                except Exception:
                    test_df = pd.DataFrame()
                if inferred in test_df.columns:
                    _save_chart(
                        "class_balance_test.png",
                        f"Test set donut chart for target balance: {target_label}",
                        lambda: _donut_plot(test_df[inferred], "Test set"),
                    )

    # 3) Numeric distributions grid
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if inferred and inferred in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != inferred]
    take = _select_numeric_distribution_columns(df, numeric_cols)

    def _numeric_grid() -> None:
        if not take:
            ax = plt.gca()
            ax.text(0.5, 0.5, "No numeric columns for histogram grid", ha="center", va="center")
            ax.axis("off")
            return
        n = len(take)
        rows, cols = _grid_shape(n)
        palette = sns.color_palette("colorblind", n_colors=n)
        plt.figure(figsize=(4 * cols, 3 * rows))
        for i, col in enumerate(take):
            ax = plt.subplot(rows, cols, i + 1)
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.axis("off")
                continue
            series.hist(bins=30, color=palette[i], edgecolor="white", ax=ax)
            ax.set_title(str(col)[:40])
        plt.suptitle("Numeric feature distributions (sample)")

    _save_chart(
        "numeric_distributions.png",
        "Histograms for up to six numeric features",
        _numeric_grid,
    )

    # 3b) Categorical feature distributions (not target-related)
    categorical_cols = _categorical_columns(df)
    if inferred and inferred in categorical_cols:
        categorical_cols = [c for c in categorical_cols if c != inferred]
    cat_take = _select_categorical_distribution_columns(df, categorical_cols)

    def _categorical_grid() -> None:
        if not cat_take:
            ax = plt.gca()
            ax.text(0.5, 0.5, "No categorical feature columns for distribution grid", ha="center", va="center")
            ax.axis("off")
            return
        rows, cols = _grid_shape(len(cat_take))
        palette = sns.color_palette("colorblind", n_colors=len(cat_take))
        plt.figure(figsize=(5 * cols, 3.5 * rows))
        for i, col in enumerate(cat_take):
            ax = plt.subplot(rows, cols, i + 1)
            s_cat = _top_categories(df[col])
            vc = (s_cat.value_counts(normalize=True).sort_values(ascending=False) * 100)
            vc.plot(kind="bar", color=palette[i], edgecolor="white", ax=ax)
            ax.set_title(str(col)[:40])
            ax.set_xlabel("")
            ax.set_ylabel("Percent of rows (%)")
            ax.tick_params(axis="x", labelrotation=45)
        plt.suptitle("Categorical feature distributions (top levels)", fontsize=14, fontweight="bold")

    _save_chart(
        "categorical_distributions.png",
        "Categorical feature distributions for up to six features",
        _categorical_grid,
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
        wrapped_labels = [_wrap_axis_label(col) for col in corr.columns]
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=False,
            xticklabels=wrapped_labels,
            yticklabels=wrapped_labels,
        )
        plt.title("Correlation heatmap (top numeric columns by variance)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=35, va="center")

    _save_chart(
        "corr_heatmap.png",
        "Pearson correlation heatmap for top numeric columns",
        _corr_plot,
    )

    # 5) Boxplots for outlier view
    def _box_plot() -> None:
        cols_box = numeric_cols[:MAX_GRID_FEATURES]
        if not cols_box:
            ax = plt.gca()
            ax.text(0.5, 0.5, "No numeric columns for boxplots", ha="center", va="center")
            ax.axis("off")
            return
        rows, cols = _grid_shape(len(cols_box))
        palette = sns.color_palette("colorblind", n_colors=len(cols_box))
        plt.figure(figsize=(4 * cols, 3.2 * rows))
        for i, col in enumerate(cols_box):
            ax = plt.subplot(rows, cols, i + 1)
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.axis("off")
                continue
            ax.boxplot(series, orientation="vertical", patch_artist=True, boxprops={"facecolor": palette[i]})
            ax.set_title(str(col)[:40])
            ax.set_xticks([])
            ax.set_ylabel("")
        plt.suptitle("Numeric feature boxplots")

    _save_chart(
        "outlier_boxplots.png",
        "Grid boxplots for numeric features on original scales",
        _box_plot,
    )

    # 5b) Feature-target relationship charts by task type
    if inferred:
        y = df[inferred]
        if target_type == "classification" and 2 <= y.nunique(dropna=True) <= 20:
            rel_num_cols = _rank_columns_by_stat_rows(
                _select_numeric_distribution_columns(df, numeric_cols, limit=len(numeric_cols)),
                (precomputed_stats or {}).get("numeric_feature_class_effects") or [],
                "omega_squared",
                MAX_GRID_FEATURES,
            )

            def _numeric_by_target_density() -> None:
                if not rel_num_cols:
                    ax = plt.gca()
                    ax.text(0.5, 0.5, "No numeric features for target density plots", ha="center", va="center")
                    ax.axis("off")
                    return
                work = _sample_rows(df[[*rel_num_cols, inferred]].dropna())
                rows, cols = _grid_shape(len(rel_num_cols))
                plt.figure(figsize=(4.5 * cols, 3.2 * rows))
                for i, col in enumerate(rel_num_cols):
                    ax = plt.subplot(rows, cols, i + 1)
                    plotted = False
                    for label, grp in work.groupby(inferred):
                        vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                        if len(vals) >= 3 and vals.nunique() > 1:
                            sns.kdeplot(vals, ax=ax, label=_class_label(label, precomputed_stats), fill=True, alpha=0.25, linewidth=1)
                            plotted = True
                    if not plotted:
                        sns.histplot(data=work, x=col, hue=inferred, stat="density", common_norm=False, ax=ax)
                    ax.set_title(str(col)[:40])
                    ax.set_xlabel("")
                    if i == 0:
                        ax.legend(title=target_label)
                plt.suptitle("Numeric feature distributions by target class")

            _save_chart(
                "numeric_by_target_density.png",
                "Density plots for numeric features grouped by target class",
                _numeric_by_target_density,
            )

            rel_cat_cols = _rank_columns_by_stat_rows(
                cat_take,
                (precomputed_stats or {}).get("categorical_target_associations") or [],
                "cramers_v",
                MAX_CATEGORICAL_TARGET_FEATURES,
            )

            def _categorical_by_target_rate() -> None:
                if not rel_cat_cols:
                    ax = plt.gca()
                    ax.text(0.5, 0.5, "No categorical features for target-rate plots", ha="center", va="center")
                    ax.axis("off")
                    return
                rows, cols = _grid_shape(len(rel_cat_cols), max_cols=2)
                palette = sns.color_palette("colorblind", n_colors=y.nunique(dropna=True))
                fig = plt.figure(figsize=(6 * cols, 4.1 * rows))
                legend_handles = None
                legend_labels = None
                for i, col in enumerate(rel_cat_cols):
                    ax = plt.subplot(rows, cols, i + 1)
                    feature = _top_categories(df[col])
                    table = pd.crosstab(feature, df[inferred].astype(str), normalize="index")
                    table = table.rename(columns={c: _class_label(c, precomputed_stats) for c in table.columns})
                    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
                    table.plot(kind="bar", stacked=True, color=palette[: table.shape[1]], ax=ax, legend=False)
                    if i == 0:
                        legend_handles, legend_labels = ax.get_legend_handles_labels()
                    ax.set_ylim(0, 1)
                    ax.set_title(str(col)[:40])
                    ax.set_xlabel("")
                    ax.set_ylabel("Class proportion")
                    ax.tick_params(axis="x", labelrotation=45)
                if legend_handles and legend_labels:
                    fig.legend(
                        legend_handles,
                        legend_labels,
                        title=target_label,
                        loc="lower center",
                        bbox_to_anchor=(0.5, -0.02),
                        ncol=min(len(legend_labels), 4),
                        frameon=True,
                    )
                plt.suptitle(
                    "Categorical feature composition by target class",
                    fontsize=14,
                    fontweight="bold",
                    y=0.99,
                )
                plt.subplots_adjust(top=0.88, bottom=0.20)

            _save_chart(
                "categorical_by_target_rate.png",
                "Normalized stacked bars for categorical features by target class",
                _categorical_by_target_rate,
            )

        elif target_type == "regression" and pd.api.types.is_numeric_dtype(y):
            rel_num_cols = _rank_columns_by_stat_rows(
                _select_numeric_distribution_columns(df, numeric_cols, limit=len(numeric_cols)),
                (precomputed_stats or {}).get("target_correlation") or [],
                "pearson_with_target",
                MAX_GRID_FEATURES,
            )

            def _numeric_vs_target_regression() -> None:
                if not rel_num_cols:
                    ax = plt.gca()
                    ax.text(0.5, 0.5, "No numeric features for target regression plots", ha="center", va="center")
                    ax.axis("off")
                    return
                work = _sample_rows(df[[*rel_num_cols, inferred]].dropna())
                rows, cols = _grid_shape(len(rel_num_cols))
                palette = sns.color_palette("colorblind", n_colors=len(rel_num_cols))
                plt.figure(figsize=(4.5 * cols, 3.2 * rows))
                for i, col in enumerate(rel_num_cols):
                    ax = plt.subplot(rows, cols, i + 1)
                    sns.regplot(
                        data=work,
                        x=col,
                        y=inferred,
                        ax=ax,
                        color=palette[i],
                        scatter_kws={"alpha": 0.25, "s": 12},
                        line_kws={"linewidth": 1.5},
                    )
                    ax.set_title(str(col)[:40])
                plt.suptitle("Numeric features vs continuous target")

            _save_chart(
                "numeric_vs_target_regression.png",
                "Scatter plots with regression lines for numeric features vs target",
                _numeric_vs_target_regression,
            )

            rel_cat_cols = cat_take[:MAX_CATEGORICAL_TARGET_FEATURES]

            def _categorical_vs_target_boxplot() -> None:
                if not rel_cat_cols:
                    ax = plt.gca()
                    ax.text(0.5, 0.5, "No categorical features for target boxplots", ha="center", va="center")
                    ax.axis("off")
                    return
                work = _sample_rows(df[[*rel_cat_cols, inferred]].dropna())
                rows, cols = _grid_shape(len(rel_cat_cols), max_cols=2)
                palette = sns.color_palette("colorblind", n_colors=len(rel_cat_cols))
                plt.figure(figsize=(6 * cols, 3.8 * rows))
                for i, col in enumerate(rel_cat_cols):
                    ax = plt.subplot(rows, cols, i + 1)
                    feature = _top_categories(work[col])
                    plot_df = pd.DataFrame({"feature": feature, "target": pd.to_numeric(work[inferred], errors="coerce")}).dropna()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", PendingDeprecationWarning)
                        sns.boxplot(data=plot_df, x="feature", y="target", ax=ax, color=palette[i])
                    ax.set_title(str(col)[:40])
                    ax.tick_params(axis="x", labelrotation=45)
                    ax.set_xlabel("")
                    ax.set_ylabel(target_label)
                plt.suptitle("Categorical features vs continuous target")

            _save_chart(
                "categorical_vs_target_boxplot.png",
                "Grid boxplots for categorical features vs continuous target",
                _categorical_vs_target_boxplot,
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
