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
import os
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
    """Use SHAP to perform interpretability analysis on the final model and return global feature importance and direction.

    Pipeline position: evaluate_model -> **explain_model** -> business_translator_agent

    Args:
      model_path (str): path to the model pkl returned by train_model.
      featured_test_path (str): path to the preprocessed test-set CSV output by the feature agent
        (state.testing_feature_dataset_ids[model_name]); the same path used by evaluate_model.
      target_column (str): target column name; used to separate X from the test set.
      model_name (str): model name; used for naming output files.
      task_type (str): "classification" or "regression".
      output_dir (str): chart output directory; when empty, automatically inferred as the plots/ directory
        under the same run_dir as model_path.
      max_samples (int): maximum number of samples used for SHAP explanation; default 500.
        No performance bottleneck for tree models (TreeExplainer); can be increased.
        Also fast for linear models (LinearExplainer); can be increased.
        Slow for KernelExplainer (SVM/KNN etc.); recommended to keep <= 500.
      top_n_features (int): number of top features to return in the feature importance list; default 20.

    Returns JSON containing:
      model_name: name of the model.
      explainer_type: class name of the SHAP explainer actually used.
      samples_explained: actual number of samples used for SHAP computation.
      feature_importance: list of feature importance entries (descending by mean |SHAP|), each containing:
        - rank: rank (starting from 1)
        - feature: feature name
        - mean_abs_shap: mean(|SHAP value|) for this feature; larger means greater influence
        - mean_shap: mean(SHAP value) for this feature; positive means positive influence on predictions, negative means the opposite
        - direction: "positive" | "negative" | "mixed" (mixed when |mean_shap| < 0.05 * mean_abs_shap)
      top_features: list of the top_n_features most important feature names (descending by importance).
      shap_summary_plot_path: path to the SHAP bar chart PNG (null if generation fails).
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
        # Avoid tqdm progress lines interleaving with downstream benchmark prints on Windows.
        os.environ.setdefault("TQDM_DISABLE", "1")
        import shap
    except ImportError:
        output = json.dumps(
            {"error": {"type": "ImportError", "message": "shap is not installed; please run: pip install shap"}},
            ensure_ascii=False,
        )
        trace_tool_call("explain_model", tool_inputs, output)
        return output

    try:
        max_samples = int(max_samples)
        top_n_features = int(top_n_features)

        # -- Load model and feature data --
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        test_df = pd.read_csv(featured_test_path)
        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in test set.")

        # Align X with train_model / evaluate_model using feature_columns JSON
        # (same directory as the .pkl). Avoids passing stray string columns into
        # sklearn + SHAP (can raise confusing TypeErrors inside shap).
        feat_path = Path(model_path).parent / f"{model_name}_feature_columns.json"
        if feat_path.is_file():
            feature_cols = json.loads(
                feat_path.read_text(encoding="utf-8")
            )
            X = pd.DataFrame(index=test_df.index)
            for col in feature_cols:
                s = (
                    test_df[col]
                    if col in test_df.columns
                    else pd.Series(0, index=test_df.index, dtype=float)
                )
                X[col] = pd.to_numeric(s, errors="coerce")
            X = X.fillna(0.0)
        else:
            feature_cols = [c for c in test_df.columns if c != target_column]
            X = test_df[feature_cols]
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
            X = X.fillna(0.0)

        feature_cols = list(X.columns)
        if X.shape[1] == 0:
            raise ValueError(
                "No feature columns left for SHAP after alignment; "
                "check featured test CSV and feature_columns.json."
            )

        # -- Sample (for slow explainers) --
        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=331).reset_index(drop=True)
        samples_explained = len(X)

        # -- Build SHAP Explainer (auto-dispatch) --
        # shap.Explainer auto-selects:
        #   tree models   -> TreeExplainer  (exact, fast)
        #   linear models -> LinearExplainer (exact, fast)
        #   others        -> PermutationExplainer / KernelExplainer (approx, slow)
        explainer = shap.Explainer(model, X)
        explainer_type = type(explainer).__name__

        shap_values = explainer(X)

        # -- Extract SHAP matrix --
        # shap_values.values shape:
        #   regression       -> (n_samples, n_features)
        #   binary classification  -> (n_samples, n_features) or (n_samples, n_features, 2)
        #   multi-class classification -> (n_samples, n_features, n_classes)
        vals: np.ndarray = shap_values.values
        if vals.ndim == 3:
            # Use positive class (index=1) for binary classification; average over classes for multi-class.
            if task_type == "classification" and vals.shape[2] == 2:
                vals = vals[:, :, 1]
            else:
                vals = vals.mean(axis=2)

        # -- Compute feature importance --
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

        # -- Generate SHAP bar chart --
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
