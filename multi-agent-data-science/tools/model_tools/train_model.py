import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.env_utils import PROJECT_ROOT
from tools.tool_trace import trace_tool_call


def _get_model(model_name: str) -> Any:
    if model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=42)
    if model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=42)
    if model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0)
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(random_state=42)
    if model_name == "SVM":
        from sklearn.svm import SVC
        return SVC(probability=True, random_state=42)
    if model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier()
    if model_name == "LinearRegression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    if model_name == "Ridge":
        from sklearn.linear_model import Ridge
        return Ridge(random_state=42)
    if model_name == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=42)
    if model_name == "XGBoostRegressor":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(random_state=42, verbosity=0)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(random_state=42)
    if model_name == "SVR":
        from sklearn.svm import SVR
        return SVR()
    raise ValueError(f"未知模型名称: {model_name}")


@tool
def train_model(
    featured_train_path: str,
    model_name: str,
    task_type: str,
    target_column: str,
    output_dir: str = "",
) -> str:
    """在特征工程完成后的训练集上训练指定模型，保存模型和特征列表。

    参数：
      featured_train_path (str)：特征工程后的训练集 CSV 路径（来自 feature_dataset_ids[model_name]）。
      model_name (str)：模型名称，必须是候选池中的名称（如 XGBoost、Ridge）。
      task_type (str)："classification" 或 "regression"。
      target_column (str)：目标列名，训练时会从特征中排除。
      output_dir (str)：模型保存目录，默认 artifacts/models/。

    返回 JSON，包含：
      model_path：保存的模型 pkl 路径。
      feature_columns_path：训练特征列名 JSON 路径（评估测试集时用于对齐列）。
      model_name：模型名称。
      train_score：训练集上的评分（分类为 accuracy，回归为 R²）。
      rows_trained：训练样本数。
    """
    tool_inputs = {
        "featured_train_path": featured_train_path,
        "model_name": model_name,
        "task_type": task_type,
        "target_column": target_column,
    }
    try:
        df = pd.read_csv(featured_train_path)
        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 不在训练集中。")

        feature_cols = [c for c in df.columns if c != target_column]
        X = df[feature_cols].select_dtypes(include="number")
        feature_cols = X.columns.tolist()
        y = df[target_column]

        model = _get_model(model_name)
        model.fit(X, y)
        train_score = round(float(model.score(X, y)), 4)

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
            "rows_trained": int(len(df)),
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("train_model", tool_inputs, output)
    return output
