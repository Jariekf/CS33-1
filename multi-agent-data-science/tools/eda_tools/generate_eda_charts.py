from __future__ import annotations

import json
import textwrap
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call

LOW_CARDINALITY_TARGET_MAX_CLASSES = 5
MAX_GRID_FEATURES = 6
MAX_CATEGORICAL_GRID_FEATURES = 8
MAX_CATEGORICAL_TARGET_FEATURES = 8
MAX_CATEGORY_LEVELS = 10

# Complete set of all chart filenames this tool can ever produce.
# Imported by the BAT agent as a static whitelist to validate LLM image references.
# Keep this in sync whenever save_chart() calls are added or renamed below.
EDA_CHART_FILENAMES: frozenset[str] = frozenset(
    {
        # --- always attempted (when data allows) ---
        "missing_values.png",
        "target_distribution.png",
        "numeric_distributions.png",
        "categorical_distributions.png",
        "corr_heatmap.png",
        "outlier_boxplots.png",
        "rf_importance.png",
        # --- classification-only ---
        "class_balance.png",
        "class_balance_test.png",
        "numeric_by_target_density.png",
        "categorical_by_target_rate.png",
        # --- regression-only ---
        "numeric_vs_target_regression.png",
        "categorical_vs_target_boxplot.png",
        "feature_target_corr.png",
    }
)


def _setup_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]


def _json_error(exc: Exception) -> str:
    return json.dumps(
        {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
        ensure_ascii=False,
    )


def _infer_target_column(df: pd.DataFrame) -> str | None:
    if df.empty or df.shape[1] == 0:
        return None
    semantic_exact = {
        "target",
        "label",
        "labels",
        "outcome",
        "response",
        "dependent",
        "y",
        "class",
    }
    semantic_contains = ("target", "label", "outcome", "response", "dependent")
    lowered = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    exact_hits = [c for c, lc in lowered.items() if lc in semantic_exact]
    contains_hits = [c for c, lc in lowered.items() if any(tok in lc for tok in semantic_contains)]
    if exact_hits:
        return str(exact_hits[0])
    if contains_hits:
        return str(contains_hits[0])
    return str(df.columns[-1])


def _resolve_target_type(df: pd.DataFrame, target_col: str | None, task_type: str = "") -> str:
    normalized = task_type.strip().lower()
    if normalized in {"classification", "regression"}:
        return normalized
    if not target_col or target_col not in df.columns:
        return "unknown"
    target = df[target_col]
    if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 25:
        return "regression"
    return "classification"


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(col)
        for col in df.columns
        if pd.api.types.is_object_dtype(df[col])
        or isinstance(df[col].dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(df[col])
    ]


def _numeric_variance_order(df: pd.DataFrame, columns: list[str]) -> list[str]:
    scored: list[tuple[float, str]] = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        variance = series.var()
        score = float(variance) if pd.notna(variance) else float("-inf")
        scored.append((score, str(col)))
    return [col for _, col in sorted(scored, key=lambda item: (-item[0], item[1]))]


def _select_numeric_columns(df: pd.DataFrame, columns: list[str], limit: int = MAX_GRID_FEATURES) -> list[str]:
    return _numeric_variance_order(df, columns)[:limit]


def _select_categorical_columns(
    df: pd.DataFrame,
    columns: list[str],
    limit: int = MAX_CATEGORICAL_GRID_FEATURES,
) -> list[str]:
    scored: list[tuple[int, str]] = []
    for col in columns:
        nunique = int(df[col].nunique(dropna=True))
        if nunique > 1:
            scored.append((nunique, str(col)))
    return [col for _, col in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]


def _grid_shape(n: int, max_cols: int = 3) -> tuple[int, int]:
    cols = min(max_cols, max(1, n))
    rows = int(np.ceil(max(1, n) / cols))
    return rows, cols


def _top_categories(series: pd.Series, max_levels: int = MAX_CATEGORY_LEVELS) -> pd.Series:
    values = series.astype(str).fillna("__missing__")
    counts = values.value_counts()
    if len(counts) <= max_levels:
        return values
    keep = set(counts.head(max_levels - 1).index)
    return values.where(values.isin(keep), other="Other")


def _sample_rows(df: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def _wrap_axis_label(label: object, width: int = 11) -> str:
    text = str(label)
    if len(text) <= width:
        return text
    wrapped = textwrap.wrap(text.replace("_", " "), width=width, break_long_words=False)
    return "\n".join(part.replace(" ", "_") for part in wrapped) if wrapped else text


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


def _relative_path(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir.parent).as_posix()
    except ValueError:
        return path.as_posix()


@tool
def generate_eda_charts(
    file_path: str,
    output_dir: str = "",
    target_column: str = "",
    task_type: str = "",
    test_file_path: str = "",
) -> str:
    """生成完整 EDA 图表并返回图表 metadata。

    参数：
      file_path：训练集 CSV 文件路径。
      output_dir：图片保存目录。新架构中应传入 {run_dir}/plots。
      target_column：目标列名。为空时用列名启发式推断。
      task_type：任务类型，优先使用 GlobalState 中的 classification/regression。
      test_file_path：测试集 CSV 路径，用于生成测试集类别平衡图。

    返回 JSON，包含 output_dir、plot_count、plot_paths、charts、text_summary。
    """
    tool_inputs = {
        "file_path": file_path,
        "output_dir": output_dir,
        "target_column": target_column,
        "task_type": task_type,
        "test_file_path": test_file_path,
    }
    try:
        _setup_plot_style()
        import matplotlib.pyplot as plt
        import seaborn as sns

        train_path = Path(file_path)
        if not train_path.is_file():
            raise FileNotFoundError(f"训练集文件不存在：{file_path}")

        df = pd.read_csv(train_path)
        if df.empty:
            raise ValueError("训练集为空，无法生成 EDA charts。")

        out = Path(output_dir) if output_dir else train_path.parent / "eda_plots"
        out.mkdir(parents=True, exist_ok=True)

        target_col = target_column if target_column and target_column in df.columns else _infer_target_column(df)
        if target_col and target_col not in df.columns:
            target_col = None
        target_type = _resolve_target_type(df, target_col, task_type=task_type)

        charts: list[dict[str, str]] = []
        summary_lines = [
            "## EDA Charts Summary",
            f"- training_data: `{train_path}`",
            f"- shape: {df.shape[0]} rows x {df.shape[1]} columns",
            f"- target_column: `{target_col}`" if target_col else "- target_column: (none)",
            f"- target_type: `{target_type}`",
        ]

        def save_chart(name: str, desc: str, draw: Callable[[], None]) -> None:
            chart_path = out / name
            try:
                plt.close("all")
                draw()
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Tight layout not applied.*", category=UserWarning)
                    plt.tight_layout()
                plt.savefig(chart_path, bbox_inches="tight", dpi=150)
                charts.append(
                    {
                        "name": name,
                        "path": str(chart_path),
                        "rel_path": _relative_path(chart_path, out),
                        "desc": desc,
                    }
                )
            except Exception as exc:
                summary_lines.append(f"- skipped `{name}`: {exc}")
            finally:
                plt.close("all")

        missing_total = int(df.isna().sum().sum())
        if missing_total > 0:
            def draw_missing() -> None:
                rows = list(reversed(_missing_overview_rows(df)))
                labels = [row["column"] for row in rows]
                counts = [int(row["missing_count"]) for row in rows]
                rates = [float(row["missing_rate"]) for row in rows]
                y_pos = np.arange(len(rows))
                _, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(rows))))
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

            save_chart("missing_values.png", "Missing value rates by column", draw_missing)

        if target_col:
            y = df[target_col]
            is_low_card_classification = (
                target_type == "classification"
                and 2 <= int(y.nunique(dropna=True)) <= LOW_CARDINALITY_TARGET_MAX_CLASSES
            )

            if not is_low_card_classification:
                def draw_target_distribution() -> None:
                    plt.figure(figsize=(8, 5))
                    if pd.api.types.is_numeric_dtype(y) and target_type == "regression":
                        pd.to_numeric(y, errors="coerce").dropna().hist(bins=40, color="teal", edgecolor="white")
                        plt.xlabel(str(target_col))
                        plt.title(f"Training set target distribution: {target_col}")
                    else:
                        y.astype(str).value_counts().head(20).plot(kind="bar", color="coral", edgecolor="white")
                        plt.xticks(rotation=45, ha="right")
                        plt.title(f"Training set target category counts: {target_col}")
                    plt.ylabel("Count")

                save_chart("target_distribution.png", f"Distribution of target `{target_col}`", draw_target_distribution)

            def draw_donut(series: pd.Series, title_prefix: str) -> None:
                vc = series.astype(str).value_counts()
                _, ax = plt.subplots(figsize=(8, 5))
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
                for wedge, label, color in zip(wedges, vc.index.astype(str), colors):
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
                ax.set_title(f"{target_col} ({title_prefix.lower()})", fontsize=14, fontweight="bold")
                ax.axis("equal")

            if is_low_card_classification:
                save_chart(
                    "class_balance.png",
                    f"Training set donut chart for target balance: {target_col}",
                    lambda: draw_donut(y, "Training set"),
                )
                test_path = Path(test_file_path) if test_file_path else None
                if test_path and test_path.is_file():
                    test_df = pd.read_csv(test_path)
                    if target_col in test_df.columns:
                        save_chart(
                            "class_balance_test.png",
                            f"Test set donut chart for target balance: {target_col}",
                            lambda: draw_donut(test_df[target_col], "Test set"),
                        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.astype(str).tolist()
        if target_col in numeric_cols:
            numeric_cols = [col for col in numeric_cols if col != target_col]
        selected_numeric = _select_numeric_columns(df, numeric_cols)

        def draw_numeric_distributions() -> None:
            if not selected_numeric:
                raise ValueError("没有可绘制的数值特征列。")
            rows, cols = _grid_shape(len(selected_numeric))
            palette = sns.color_palette("colorblind", n_colors=len(selected_numeric))
            plt.figure(figsize=(4 * cols, 3 * rows))
            for index, col in enumerate(selected_numeric):
                ax = plt.subplot(rows, cols, index + 1)
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                series.hist(bins=30, color=palette[index], edgecolor="white", ax=ax)
                ax.set_title(str(col)[:40])
            plt.suptitle("Numeric feature distributions")

        save_chart("numeric_distributions.png", "Histograms for numeric features", draw_numeric_distributions)

        categorical_cols = _categorical_columns(df)
        if target_col in categorical_cols:
            categorical_cols = [col for col in categorical_cols if col != target_col]
        selected_categorical = _select_categorical_columns(df, categorical_cols)

        def draw_categorical_distributions() -> None:
            if not selected_categorical:
                raise ValueError("没有可绘制的类别特征列。")
            rows, cols = _grid_shape(len(selected_categorical))
            palette = sns.color_palette("colorblind", n_colors=len(selected_categorical))
            plt.figure(figsize=(5 * cols, 3.5 * rows))
            for index, col in enumerate(selected_categorical):
                ax = plt.subplot(rows, cols, index + 1)
                vc = (_top_categories(df[col]).value_counts(normalize=True).sort_values(ascending=False) * 100)
                vc.plot(kind="bar", color=palette[index], edgecolor="white", ax=ax)
                ax.set_title(str(col)[:40])
                ax.set_xlabel("")
                ax.set_ylabel("Percent of rows (%)")
                ax.tick_params(axis="x", labelrotation=45)
            plt.suptitle("Categorical feature distributions (top levels)", fontsize=14, fontweight="bold")

        save_chart(
            "categorical_distributions.png",
            "Categorical feature distributions for selected features",
            draw_categorical_distributions,
        )

        def draw_corr_heatmap() -> None:
            num_df = df.select_dtypes(include=[np.number]).copy()
            if target_col in num_df.columns:
                num_df = num_df.drop(columns=[target_col])
            if num_df.shape[1] < 2:
                raise ValueError("至少需要 2 个数值列才能绘制相关热力图。")
            variances = num_df.var(numeric_only=True).sort_values(ascending=False)
            top_cols = variances.head(min(15, len(variances))).index.tolist()
            corr = df[top_cols].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
            plt.figure(figsize=(10, 8))
            wrapped_labels = [_wrap_axis_label(col) for col in corr.columns]
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                xticklabels=wrapped_labels,
                yticklabels=wrapped_labels,
            )
            plt.title("Correlation heatmap (top numeric columns by variance)", fontsize=14, fontweight="bold")
            plt.xticks(rotation=35, ha="right")
            plt.yticks(rotation=35, va="center")

        save_chart("corr_heatmap.png", "Pearson correlation heatmap for top numeric columns", draw_corr_heatmap)

        def draw_outlier_boxplots() -> None:
            if not selected_numeric:
                raise ValueError("没有可绘制箱线图的数值特征列。")
            rows, cols = _grid_shape(len(selected_numeric))
            palette = sns.color_palette("colorblind", n_colors=len(selected_numeric))
            plt.figure(figsize=(4 * cols, 3.2 * rows))
            for index, col in enumerate(selected_numeric):
                ax = plt.subplot(rows, cols, index + 1)
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                ax.boxplot(series, orientation="vertical", patch_artist=True, boxprops={"facecolor": palette[index]})
                ax.set_title(str(col)[:40])
                ax.set_xticks([])
            plt.suptitle("Numeric feature boxplots")

        save_chart("outlier_boxplots.png", "Grid boxplots for numeric features", draw_outlier_boxplots)

        if target_col:
            y = df[target_col]
            if target_type == "classification" and 2 <= y.nunique(dropna=True) <= 20:
                rel_num_cols = selected_numeric[:MAX_GRID_FEATURES]

                def draw_numeric_by_target_density() -> None:
                    if not rel_num_cols:
                        raise ValueError("没有数值特征用于按目标类别绘图。")
                    work = _sample_rows(df[[*rel_num_cols, target_col]].dropna())
                    rows, cols = _grid_shape(len(rel_num_cols))
                    plt.figure(figsize=(4.5 * cols, 3.2 * rows))
                    for index, col in enumerate(rel_num_cols):
                        ax = plt.subplot(rows, cols, index + 1)
                        plotted = False
                        for label, group in work.groupby(target_col):
                            vals = pd.to_numeric(group[col], errors="coerce").dropna()
                            if len(vals) >= 3 and vals.nunique() > 1:
                                sns.kdeplot(vals, ax=ax, label=str(label), fill=True, alpha=0.25, linewidth=1)
                                plotted = True
                        if not plotted:
                            sns.histplot(data=work, x=col, hue=target_col, stat="density", common_norm=False, ax=ax)
                        ax.set_title(str(col)[:40])
                        ax.set_xlabel("")
                        if index == 0:
                            ax.legend(title=str(target_col))
                    plt.suptitle("Numeric feature distributions by target class")

                save_chart(
                    "numeric_by_target_density.png",
                    "Density plots for numeric features grouped by target class",
                    draw_numeric_by_target_density,
                )

                rel_cat_cols = selected_categorical[:MAX_CATEGORICAL_TARGET_FEATURES]

                def draw_categorical_by_target_rate() -> None:
                    if not rel_cat_cols:
                        raise ValueError("没有类别特征用于按目标类别绘图。")
                    rows, cols = _grid_shape(len(rel_cat_cols), max_cols=2)
                    palette = sns.color_palette("colorblind", n_colors=y.nunique(dropna=True))
                    fig = plt.figure(figsize=(6 * cols, 4.1 * rows))
                    legend_handles = None
                    legend_labels = None
                    for index, col in enumerate(rel_cat_cols):
                        ax = plt.subplot(rows, cols, index + 1)
                        table = pd.crosstab(_top_categories(df[col]), df[target_col].astype(str), normalize="index")
                        table.plot(kind="bar", stacked=True, color=palette[: table.shape[1]], ax=ax, legend=False)
                        if index == 0:
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
                            title=str(target_col),
                            loc="lower center",
                            bbox_to_anchor=(0.5, -0.02),
                            ncol=min(len(legend_labels), 4),
                            frameon=True,
                        )
                    plt.suptitle("Categorical feature composition by target class", fontsize=14, fontweight="bold")
                    plt.subplots_adjust(top=0.88, bottom=0.20)

                save_chart(
                    "categorical_by_target_rate.png",
                    "Normalized stacked bars for categorical features by target class",
                    draw_categorical_by_target_rate,
                )

            elif target_type == "regression" and pd.api.types.is_numeric_dtype(y):
                rel_num_cols = selected_numeric[:MAX_GRID_FEATURES]

                def draw_numeric_vs_target_regression() -> None:
                    if not rel_num_cols:
                        raise ValueError("没有数值特征用于目标回归关系图。")
                    work = _sample_rows(df[[*rel_num_cols, target_col]].dropna())
                    rows, cols = _grid_shape(len(rel_num_cols))
                    palette = sns.color_palette("colorblind", n_colors=len(rel_num_cols))
                    plt.figure(figsize=(4.5 * cols, 3.2 * rows))
                    for index, col in enumerate(rel_num_cols):
                        ax = plt.subplot(rows, cols, index + 1)
                        sns.regplot(
                            data=work,
                            x=col,
                            y=target_col,
                            ax=ax,
                            color=palette[index],
                            scatter_kws={"alpha": 0.25, "s": 12},
                            line_kws={"linewidth": 1.5},
                        )
                        ax.set_title(str(col)[:40])
                    plt.suptitle("Numeric features vs continuous target")

                save_chart(
                    "numeric_vs_target_regression.png",
                    "Scatter plots with regression lines for numeric features vs target",
                    draw_numeric_vs_target_regression,
                )

                rel_cat_cols = selected_categorical[:MAX_CATEGORICAL_TARGET_FEATURES]

                def draw_categorical_vs_target_boxplot() -> None:
                    if not rel_cat_cols:
                        raise ValueError("没有类别特征用于目标箱线图。")
                    work = _sample_rows(df[[*rel_cat_cols, target_col]].dropna())
                    rows, cols = _grid_shape(len(rel_cat_cols), max_cols=2)
                    palette = sns.color_palette("colorblind", n_colors=len(rel_cat_cols))
                    plt.figure(figsize=(6 * cols, 3.8 * rows))
                    for index, col in enumerate(rel_cat_cols):
                        ax = plt.subplot(rows, cols, index + 1)
                        plot_df = pd.DataFrame(
                            {
                                "feature": _top_categories(work[col]),
                                "target": pd.to_numeric(work[target_col], errors="coerce"),
                            }
                        ).dropna()
                        sns.boxplot(data=plot_df, x="feature", y="target", ax=ax, color=palette[index])
                        ax.set_title(str(col)[:40])
                        ax.tick_params(axis="x", labelrotation=45)
                        ax.set_xlabel("")
                        ax.set_ylabel(str(target_col))
                    plt.suptitle("Categorical features vs continuous target")

                save_chart(
                    "categorical_vs_target_boxplot.png",
                    "Grid boxplots for categorical features vs continuous target",
                    draw_categorical_vs_target_boxplot,
                )

        if target_col and target_type == "regression" and pd.api.types.is_numeric_dtype(df[target_col]):
            target_corr = []
            for col in numeric_cols:
                value = pd.to_numeric(df[col], errors="coerce").corr(pd.to_numeric(df[target_col], errors="coerce"))
                if pd.notna(value):
                    target_corr.append((abs(float(value)), str(col), float(value)))
            target_corr.sort(reverse=True)
            target_corr = target_corr[:15]

            if target_corr:
                def draw_feature_target_corr() -> None:
                    labels = [item[1] for item in target_corr]
                    values = [item[2] for item in target_corr]
                    y_pos = np.arange(len(labels))
                    colors = ["coral" if value < 0 else "steelblue" for value in values]
                    plt.figure(figsize=(9, max(4, 0.4 * len(labels))))
                    plt.barh(y_pos, values, color=colors)
                    plt.yticks(y_pos, [label[:35] for label in labels], fontsize=8)
                    plt.xlabel("Pearson correlation with target")
                    plt.title("Top feature-target linear correlation")
                    plt.axvline(0, color="gray", lw=0.6)

                save_chart(
                    "feature_target_corr.png",
                    "Top feature-target Pearson correlation",
                    draw_feature_target_corr,
                )

        if target_col:
            def draw_rf_importance() -> None:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.preprocessing import LabelEncoder

                feature_df = df.drop(columns=[target_col])
                x_num = feature_df.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors="coerce")
                x_num = x_num.fillna(x_num.median(numeric_only=True)).fillna(0)
                x_cat = pd.get_dummies(feature_df[_categorical_columns(feature_df)], dummy_na=True)
                x_all = pd.concat([x_num, x_cat], axis=1).fillna(0)
                x_all = x_all.loc[:, x_all.nunique(dropna=False) > 1]
                if x_all.empty:
                    raise ValueError("编码后没有可用于 RF importance 的特征。")

                if target_type == "regression" and pd.api.types.is_numeric_dtype(df[target_col]):
                    y_rf = pd.to_numeric(df[target_col], errors="coerce")
                    mask = y_rf.notna()
                    model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
                    model.fit(x_all.loc[mask], y_rf.loc[mask])
                else:
                    encoder = LabelEncoder()
                    y_rf = encoder.fit_transform(df[target_col].astype(str).fillna("__na__"))
                    if len(np.unique(y_rf)) < 2:
                        raise ValueError("目标列只有一个类别，无法计算 RF importance。")
                    model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
                    model.fit(x_all, y_rf)

                pairs = sorted(zip(x_all.columns.astype(str), model.feature_importances_), key=lambda item: -item[1])[:15]
                labels = [item[0] for item in pairs]
                values = [float(item[1]) for item in pairs]
                y_pos = np.arange(len(labels))
                plt.figure(figsize=(9, max(4, 0.4 * len(labels))))
                plt.barh(y_pos, values, color="seagreen")
                plt.yticks(y_pos, [label[:35] for label in labels], fontsize=8)
                plt.xlabel("Importance")
                plt.title("RandomForest feature importance (quick n_estimators=50)")

            save_chart(
                "rf_importance.png",
                "RandomForest quick feature importance",
                draw_rf_importance,
            )

        summary_lines.append("")
        summary_lines.append("### Generated charts")
        if charts:
            for chart in charts:
                summary_lines.append(f"- `{chart['rel_path']}` - {chart['desc']}")
        else:
            summary_lines.append("- (no chart files were produced)")

        result = {
            "output_dir": str(out),
            "plot_count": len(charts),
            "plot_paths": [chart["path"] for chart in charts],
            "charts": charts,
            "text_summary": "\n".join(summary_lines),
        }
        output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        output = _json_error(exc)

    trace_tool_call("generate_eda_charts", tool_inputs, output)
    return output
