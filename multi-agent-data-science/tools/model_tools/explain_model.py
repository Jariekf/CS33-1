"""explain_model tool — SHAP-based model interpretability.

Pipeline position: evaluate_model → **explain_model** → business_translator_agent

Reads the already-preprocessed test CSV (testing_feature_dataset_ids[model_name])
and the trained model pkl, then computes SHAP values to produce:
  - Global feature importance ranking (mean |SHAP|)
  - Per-feature direction (positive/negative impact on prediction)
  - A SHAP summary bar-chart saved to run_dir/plots/

Results are stored in best_model["shap_explanation"] by model_agent._handle_explain.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call


@tool
def explain_model(
    model_path: str,
    featured_test_path: str,
    target_column: str,
    model_name: str,
    task_type: str,
    output_dir: str = "",
    max_samples: int = 500,
    top_n_features: int = 20,
) -> str:
    """使用 SHAP 对最终模型进行可解释性分析，返回全局特征重要性和方向。

    工作流位置：evaluate_model → **explain_model** → business_translator_agent

    参数：
      model_path (str)：train_model 返回的模型 pkl 路径。
      featured_test_path (str)：feature agent 输出的预处理后测试集 CSV 路径
        (state.testing_feature_dataset_ids[model_name])，与 evaluate_model 所用路径相同。
      target_column (str)：目标列名，用于从测试集中分离 X。
      model_name (str)：模型名称，用于输出文件命名。
      task_type (str)："classification" 或 "regression"。
      output_dir (str)：图表输出目录；留空时自动推断为 model_path 同 run_dir 下的 plots/ 目录。
      max_samples (int)：用于 SHAP 解释的最大样本数，默认 500。
        对树模型（TreeExplainer）无性能瓶颈，可适当增大。
        对线性模型（LinearExplainer）速度也快，可增大。
        对 KernelExplainer（SVM/KNN 等）计算慢，建议保持 ≤ 500。
      top_n_features (int)：在特征重要性列表中返回的 top 特征数，默认 20。

    返回 JSON，包含：
      model_name：模型名称。
      explainer_type：实际使用的 SHAP explainer 类名。
      samples_explained：实际用于 SHAP 计算的样本数。
      feature_importance：特征重要性列表（按 mean |SHAP| 降序），每项含：
        - rank：排名（从 1 开始）
        - feature：特征名
        - mean_abs_shap：该特征的 mean(|SHAP value|)，值越大影响越大
        - mean_shap：该特征的 mean(SHAP value)，正值表示正向影响预测，负值反之
        - direction："positive" | "negative" | "mixed"（|mean_shap| < 0.05 * mean_abs_shap 时为 mixed）
      top_features：top_n_features 个最重要特征的名称列表（按重要性降序）。
      shap_summary_plot_path：SHAP bar chart PNG 路径（生成失败时为 null）。
    """
    tool_inputs = {
        "model_path": model_path,
        "featured_test_path": featured_test_path,
        "target_column": target_column,
        "model_name": model_name,
        "task_type": task_type,
        "max_samples": max_samples,
        "top_n_features": top_n_features,
    }
    try:
        import shap
    except ImportError:
        output = json.dumps(
            {"error": {"type": "ImportError", "message": "shap 未安装，请运行 pip install shap"}},
            ensure_ascii=False,
        )
        trace_tool_call("explain_model", tool_inputs, output)
        return output

    try:
        # ── 加载模型和特征数据 ──────────────────────────────────────────────
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        test_df = pd.read_csv(featured_test_path)
        if target_column not in test_df.columns:
            raise ValueError(f"目标列 '{target_column}' 不在测试集中。")

        feature_cols = [c for c in test_df.columns if c != target_column]
        X = test_df[feature_cols]

        # ── 采样（针对慢速 explainer）──────────────────────────────────────
        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42).reset_index(drop=True)
        samples_explained = len(X)

        # ── 构建 SHAP Explainer（auto-dispatch）────────────────────────────
        # shap.Explainer 会自动选择：
        #   树模型     → TreeExplainer  (exact, fast)
        #   线性模型   → LinearExplainer (exact, fast)
        #   其他       → PermutationExplainer / KernelExplainer (approx, slow)
        explainer = shap.Explainer(model, X)
        explainer_type = type(explainer).__name__

        shap_values = explainer(X)

        # ── 提取 SHAP 矩阵 ─────────────────────────────────────────────────
        # shap_values.values shape:
        #   回归         → (n_samples, n_features)
        #   二分类       → (n_samples, n_features) or (n_samples, n_features, 2)
        #   多分类       → (n_samples, n_features, n_classes)
        vals: np.ndarray = shap_values.values
        if vals.ndim == 3:
            # 取正类（index=1）做二分类；多分类取均值
            if task_type == "classification" and vals.shape[2] == 2:
                vals = vals[:, :, 1]
            else:
                vals = vals.mean(axis=2)

        # ── 计算特征重要性 ──────────────────────────────────────────────────
        mean_abs = np.abs(vals).mean(axis=0)   # mean |SHAP| per feature
        mean_shap = vals.mean(axis=0)          # mean SHAP per feature (direction)

        sorted_idx = np.argsort(mean_abs)[::-1]
        feature_importance: list[dict[str, Any]] = []
        for rank, idx in enumerate(sorted_idx[:top_n_features], start=1):
            m_abs = float(mean_abs[idx])
            m_shap = float(mean_shap[idx])
            # direction: "mixed" when mean SHAP magnitude is < 5% of mean |SHAP|
            if abs(m_shap) < 0.05 * m_abs:
                direction = "mixed"
            elif m_shap > 0:
                direction = "positive"
            else:
                direction = "negative"
            feature_importance.append({
                "rank": rank,
                "feature": feature_cols[idx],
                "mean_abs_shap": round(m_abs, 6),
                "mean_shap": round(m_shap, 6),
                "direction": direction,
            })

        top_features = [fi["feature"] for fi in feature_importance]

        # ── 生成 SHAP bar chart ────────────────────────────────────────────
        shap_summary_plot_path: str | None = None
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend for headless environments
            import matplotlib.pyplot as plt

            plot_dir = Path(output_dir) if output_dir else Path(model_path).parent.parent / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10, max(4, top_n_features * 0.4)))
            top_idx = sorted_idx[:top_n_features][::-1]  # reversed for horizontal bar (bottom→top)
            feat_names = [feature_cols[i] for i in top_idx]
            feat_vals = [float(mean_abs[i]) for i in top_idx]
            colors = [
                "#e74c3c" if float(mean_shap[i]) > 0 else "#3498db"
                for i in top_idx
            ]
            ax.barh(feat_names, feat_vals, color=colors)
            ax.set_xlabel("mean(|SHAP value|)")
            ax.set_title(f"SHAP Feature Importance — {model_name}")
            # legend
            from matplotlib.patches import Patch
            ax.legend(
                handles=[
                    Patch(color="#e74c3c", label="positive impact"),
                    Patch(color="#3498db", label="negative impact"),
                ],
                loc="lower right",
                fontsize=8,
            )
            plt.tight_layout()
            plot_path = plot_dir / f"{model_name}_shap_importance.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            shap_summary_plot_path = str(plot_path)
        except Exception as _plot_exc:
            shap_summary_plot_path = None  # non-fatal

        result = {
            "model_name": model_name,
            "explainer_type": explainer_type,
            "samples_explained": samples_explained,
            "feature_importance": feature_importance,
            "top_features": top_features,
            "shap_summary_plot_path": shap_summary_plot_path,
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as exc:
        output = json.dumps(
            {"error": {"type": type(exc).__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("explain_model", tool_inputs, output)
    return output
