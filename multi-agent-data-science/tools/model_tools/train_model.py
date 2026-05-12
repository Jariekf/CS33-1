import json
import math
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.env_utils import PROJECT_ROOT
from tools.tool_trace import trace_tool_call


def _compute_train_score(
    model: Any, X: Any, y: Any, scoring: str
) -> tuple[float | None, str]:
    """Compute in-sample score using the specified metric name (human-readable).

    Returns (score, metric_name). Falls back to (None, "skipped") for metrics
    that require probability estimates (roc_auc) when the model does not support
    predict_proba, or for unrecognised scoring strings.
    """
    try:
        y_pred = model.predict(X)
        if scoring in ("f1_weighted", "f1_macro"):
            from sklearn.metrics import f1_score
            avg = "weighted" if scoring == "f1_weighted" else "macro"
            return round(float(f1_score(y, y_pred, average=avg, zero_division=0)), 4), scoring
        if scoring == "accuracy":
            from sklearn.metrics import accuracy_score
            return round(float(accuracy_score(y, y_pred)), 4), scoring
        if scoring == "recall":
            from sklearn.metrics import recall_score
            return round(float(recall_score(y, y_pred, average="weighted", zero_division=0)), 4), scoring
        if scoring == "precision":
            from sklearn.metrics import precision_score
            return round(float(precision_score(y, y_pred, average="weighted", zero_division=0)), 4), scoring
        if scoring == "roc_auc":
            if hasattr(model, "predict_proba"):
                from sklearn.metrics import roc_auc_score
                y_prob = model.predict_proba(X)
                n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 1
                if n_classes == 2:
                    score = roc_auc_score(y, y_prob[:, 1])
                else:
                    score = roc_auc_score(y, y_prob, multi_class="ovr", average="weighted")
                return round(float(score), 4), scoring
        if scoring == "r2":
            from sklearn.metrics import r2_score
            return round(float(r2_score(y, y_pred)), 4), scoring
        if scoring == "rmse":
            from sklearn.metrics import mean_squared_error
            return round(float(math.sqrt(mean_squared_error(y, y_pred))), 4), scoring
        if scoring == "mae":
            from sklearn.metrics import mean_absolute_error
            return round(float(mean_absolute_error(y, y_pred)), 4), scoring
    except Exception:
        pass
    return None, "skipped"


def _get_model(model_name: str, model_params: dict[str, Any] | None = None) -> Any:
    params = model_params or {}

    # ── Classification ────────────────────────────────────────────────────────
    if model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**{"max_iter": 1000, "random_state": 42, **params})
    if model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**{"n_estimators": 100, "random_state": 42, **params})
    if model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(**{"random_state": 42, "eval_metric": "logloss", "verbosity": 0, **params})
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**{"random_state": 42, **params})
    if model_name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**{"random_state": 42, "verbosity": -1, **params})
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**{"random_state": 42, **params})
    if model_name == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**{"random_state": 42, **params})
    if model_name == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**{"n_estimators": 100, "random_state": 42, **params})
    if model_name == "AdaBoost":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(**{"random_state": 42, **params})
    if model_name == "SVM":
        from sklearn.svm import SVC
        return SVC(**{"probability": True, "random_state": 42, **params})
    if model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**params)
    if model_name == "GaussianNB":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(**params)
    if model_name == "MLPClassifier":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(**{"hidden_layer_sizes": (100,), "max_iter": 500, "random_state": 42, **params})

    # ── Regression ────────────────────────────────────────────────────────────
    if model_name == "LinearRegression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**params)
    if model_name == "Ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**{"random_state": 42, **params})
    if model_name == "Lasso":
        from sklearn.linear_model import Lasso
        return Lasso(**{"random_state": 42, **params})
    if model_name == "ElasticNet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(**{"random_state": 42, **params})
    if model_name == "HuberRegressor":
        from sklearn.linear_model import HuberRegressor
        return HuberRegressor(**params)
    if model_name == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**{"n_estimators": 100, "random_state": 42, **params})
    if model_name == "XGBoostRegressor":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(**{"random_state": 42, "verbosity": 0, **params})
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**{"random_state": 42, **params})
    if model_name == "LightGBMRegressor":
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**{"random_state": 42, "verbosity": -1, **params})
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**{"random_state": 42, **params})
    if model_name == "GradientBoostingRegressor":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**{"random_state": 42, **params})
    if model_name == "ExtraTreesRegressor":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(**{"n_estimators": 100, "random_state": 42, **params})
    if model_name == "SVR":
        from sklearn.svm import SVR
        return SVR(**params)
    if model_name == "MLPRegressor":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(**{"hidden_layer_sizes": (100,), "max_iter": 500, "random_state": 42, **params})

    raise ValueError(f"未知模型名称: {model_name}")


@tool
def train_model(
    featured_train_path: str,
    model_name: str,
    task_type: str,
    target_column: str,
    output_dir: str = "",
    model_params: dict[str, Any] | None = None,
    scoring: str = "",
) -> str:
    """在特征工程完成后的训练集上训练指定模型，保存模型和特征列表。

    参数：
      featured_train_path (str)：特征工程后的训练集 CSV 路径（来自 feature_dataset_ids[model_name]）。
      model_name (str)：模型名称，必须是候选池中的名称（如 XGBoost、Ridge）。
      task_type (str)："classification" 或 "regression"。
      target_column (str)：目标列名，训练时会从特征中排除。
      output_dir (str)：模型保存目录，默认 artifacts/models/。
      model_params (dict)：可选，模型参数（通常来自 tune_and_select_model 的 best_params）。
      scoring (str)：与 CV 调参一致的评分指标名（即 primary_metric，如 f1_weighted、roc_auc、r2、rmse）。
        传入后 train_score 将使用该指标计算，便于与 cv_best_score 和 test 指标做三点对比。
        留空则跳过计算，train_score 返回 null。

    返回 JSON，包含：
      model_path：保存的模型 pkl 路径。
      feature_columns_path：训练特征列名 JSON 路径（评估测试集时用于对齐列）。
      model_name：模型名称。
      train_score：训练集上使用 scoring 指标的得分；若 scoring 留空或不支持则为 null。
      train_score_metric：实际使用的指标名，"skipped" 表示未计算。
      rows_trained：训练样本数。
    """
    tool_inputs = {
        "featured_train_path": featured_train_path,
        "model_name": model_name,
        "task_type": task_type,
        "target_column": target_column,
        "model_params": model_params or {},
        "scoring": scoring,
    }
    try:
        df = pd.read_csv(featured_train_path)
        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 不在训练集中。")

        feature_cols = [c for c in df.columns if c != target_column]
        X = df[feature_cols].select_dtypes(include="number")
        feature_cols = X.columns.tolist()
        y = df[target_column]

        model = _get_model(model_name, model_params=model_params)
        model.fit(X, y)
        train_score, train_score_metric = _compute_train_score(model, X, y, scoring)

        out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "artifacts" / "models"
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = out_dir / f"{model_name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        feat_cols_path = out_dir / f"{model_name}_feature_columns.json"
        feat_cols_path.write_text(json.dumps(feature_cols, ensure_ascii=False))

        result = {
            "model_path": str(model_path),
            "feature_columns_path": str(feat_cols_path),
            "model_name": model_name,
            "train_score": train_score,
            "train_score_metric": train_score_metric,
            "rows_trained": int(len(df)),
            "model_params": model_params or {},
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("train_model", tool_inputs, output)
    return output
