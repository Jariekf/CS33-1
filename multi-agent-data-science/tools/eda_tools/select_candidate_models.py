import json
from typing import Any

from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call

CLASSIFICATION_POOL = ["LogisticRegression", "RandomForest", "XGBoost", "SVM", "KNN"]
REGRESSION_POOL = ["LinearRegression", "Ridge", "RandomForestRegressor", "XGBoostRegressor", "SVR"]


@tool
def select_candidate_models(
    task_type: str,
    selected_models: list[str],
    feature_engineering_plan: dict[str, Any],
    model_selection_plan: str,
) -> str:
    """提交 EDA 分析后选定的三个候选模型、各模型特征工程方案和模型选择理由。

    参数：
      task_type (str)："classification" 或 "regression"。

      selected_models (list[str])：从候选池中选出的恰好 3 个模型名称，大小写敏感。
        分类候选池：LogisticRegression、RandomForest、XGBoost、SVM、KNN
        回归候选池：LinearRegression、Ridge、RandomForestRegressor、XGBoostRegressor、SVR
        错误示例：xgboost、random_forest、XGboost（均会被拒绝）。

      feature_engineering_plan (dict)：各模型的特征工程方案，key 为模型名，value 为方案 dict。
        示例：
        {
          "XGBoost":            {"scaling": false, "encoding": "label",  "notes": "树模型无需缩放"},
          "RandomForest":       {"scaling": false, "encoding": "label",  "notes": "集成树，无需缩放"},
          "LogisticRegression": {"scaling": true,  "encoding": "onehot", "notes": "线性模型需标准化"}
        }

      model_selection_plan (str)：选择这三个模型的理由，结合数据特征、任务类型和目标列说明。
    """
    tool_inputs = {
        "task_type": task_type,
        "selected_models": selected_models,
        "feature_engineering_plan": feature_engineering_plan,
        "model_selection_plan": model_selection_plan,
    }
    try:
        pool = CLASSIFICATION_POOL if task_type == "classification" else REGRESSION_POOL
        invalid = [m for m in selected_models if m not in pool]
        if invalid:
            raise ValueError(f"模型不在候选池中: {invalid}。候选池: {pool}")
        if len(selected_models) != 3:
            raise ValueError(f"必须选择恰好 3 个模型，收到 {len(selected_models)} 个。")

        result = {
            "selected_models": selected_models,
            "feature_engineering_plan": feature_engineering_plan,
            "model_selection_plan": model_selection_plan,
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("select_candidate_models", tool_inputs, output)
    return output
