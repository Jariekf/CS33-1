import json
from typing import Any

from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call

# ── Metrics 白名单 ────────────────────────────────────────────────────────────
# primary_metric / evaluation_metrics 均使用人类可读名（rmse/mae 而非 neg_* 前缀）
# tune_and_select_model 内部会做 rmse→neg_root_mean_squared_error 的映射
CLASSIFICATION_METRICS = {"accuracy", "f1_macro", "f1_weighted", "roc_auc", "precision", "recall"}
REGRESSION_METRICS = {"r2", "rmse", "mae"}

CLASSIFICATION_POOL = [
    "LogisticRegression",    # 线性基线，可解释性强
    "RandomForest",          # 集成树，稳定、抗噪
    "XGBoost",               # 梯度提升，竞赛级性能
    "LightGBM",              # 更快的梯度提升，大数据集优先
    "GradientBoosting",      # sklearn 原生梯度提升，无需额外依赖
    "ExtraTrees",            # 极端随机树，速度快、方差低
    "AdaBoost",              # 自适应提升，对不平衡数据有效
    "SVM",                   # 核方法，中小数据集表现好
    "KNN",                   # 实例学习，无参数假设
    "GaussianNB",            # 朴素贝叶斯，极快的概率基线
    "MLPClassifier",         # 多层感知机，数值特征多/样本量大时有优势
]

REGRESSION_POOL = [
    "LinearRegression",          # 线性基线
    "Ridge",                     # L2 正则，防共线性
    "Lasso",                     # L1 正则，内置特征选择
    "ElasticNet",                # L1+L2，兼顾稀疏与稳定
    "HuberRegressor",            # 鲁棒回归，对异常值不敏感
    "RandomForestRegressor",     # 集成树
    "XGBoostRegressor",          # 梯度提升
    "LightGBMRegressor",         # 更快的梯度提升
    "GradientBoostingRegressor", # sklearn 原生梯度提升
    "ExtraTreesRegressor",       # 极端随机树
    "SVR",                       # 支持向量回归
    "MLPRegressor",              # 多层感知机，复杂非线性关系时可选
]


@tool
def select_candidate_models_and_metrics(
    task_type: str,
    selected_models: list[str],
    feature_engineering_plan: dict[str, Any],
    model_selection_plan: str,
    primary_metric: str,
    evaluation_metrics: list[str],
    metrics_rationale: str,
) -> str:
    """提交 EDA 分析后选定的三个候选模型、特征工程方案、模型选择理由，以及评估指标决策。

    参数：
      task_type (str)："classification" 或 "regression"。

      selected_models (list[str])：从候选池中选出的恰好 3 个模型名称，大小写敏感。
        分类候选池：LogisticRegression、RandomForest、XGBoost、LightGBM、GradientBoosting、
                    ExtraTrees、AdaBoost、SVM、KNN、GaussianNB、MLPClassifier
        回归候选池：LinearRegression、Ridge、Lasso、ElasticNet、HuberRegressor、
                    RandomForestRegressor、XGBoostRegressor、LightGBMRegressor、
                    GradientBoostingRegressor、ExtraTreesRegressor、SVR、MLPRegressor
        错误示例：xgboost、random_forest、XGboost（均会被拒绝）。

      feature_engineering_plan (dict)：各模型的特征工程方案，key 为模型名，value 为方案 dict。
        示例：
        {
          "XGBoost":            {"scaling": false, "encoding": "label",  "notes": "树模型无需缩放"},
          "RandomForest":       {"scaling": false, "encoding": "label",  "notes": "集成树，无需缩放"},
          "LogisticRegression": {"scaling": true,  "encoding": "onehot", "notes": "线性模型需标准化"}
        }

      model_selection_plan (str)：选择这三个模型的理由，结合数据特征、任务类型和目标列说明。

      primary_metric (str)：CV 参数调优使用的单一优化指标（决定"最优参数"的判断标准）。
        分类可选：accuracy、f1_macro、f1_weighted、roc_auc、precision、recall
        回归可选：r2、rmse、mae
        选择建议：
          - 类别均衡时：f1_weighted（分类）、r2（回归）
          - 类别严重不平衡时：roc_auc 或 recall（关注少数类）
          - 回归存在明显异常值时：mae（比 r2 对离群点更鲁棒）
          - 业务目标是排序/概率校准时：roc_auc

      evaluation_metrics (list[str])：最终 test set 综合评估时报告的指标列表（2~4 个）。
        分类可选：accuracy、f1_macro、f1_weighted、roc_auc、precision、recall
        回归可选：r2、rmse、mae
        建议：primary_metric 必须包含在此列表中；可额外加入 1~2 个互补指标。
        示例（不平衡分类）：["roc_auc", "f1_weighted", "recall"]
        示例（回归）：["r2", "rmse", "mae"]

      metrics_rationale (str)：选择上述指标的理由，结合任务类型、数据特征和业务目标说明。
    """
    tool_inputs = {
        "task_type": task_type,
        "selected_models": selected_models,
        "feature_engineering_plan": feature_engineering_plan,
        "model_selection_plan": model_selection_plan,
        "primary_metric": primary_metric,
        "evaluation_metrics": evaluation_metrics,
        "metrics_rationale": metrics_rationale,
    }
    try:
        # ── 模型名校验 ────────────────────────────────────────────────────────
        pool = CLASSIFICATION_POOL if task_type == "classification" else REGRESSION_POOL
        invalid_models = [m for m in selected_models if m not in pool]
        if invalid_models:
            raise ValueError(f"模型不在候选池中: {invalid_models}。候选池: {pool}")
        if len(selected_models) != 3:
            raise ValueError(f"必须选择恰好 3 个模型，收到 {len(selected_models)} 个。")

        # ── Metrics 校验 ──────────────────────────────────────────────────────
        allowed = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
        if primary_metric not in allowed:
            raise ValueError(
                f"primary_metric '{primary_metric}' 不合法。{task_type} 可选：{sorted(allowed)}"
            )
        invalid_metrics = [m for m in evaluation_metrics if m not in allowed]
        if invalid_metrics:
            raise ValueError(
                f"evaluation_metrics 中存在不合法项: {invalid_metrics}。{task_type} 可选：{sorted(allowed)}"
            )
        if not evaluation_metrics:
            raise ValueError("evaluation_metrics 不能为空列表。")
        if primary_metric not in evaluation_metrics:
            raise ValueError(
                f"primary_metric '{primary_metric}' 必须包含在 evaluation_metrics 中。"
            )

        result = {
            "selected_models": selected_models,
            "feature_engineering_plan": feature_engineering_plan,
            "model_selection_plan": model_selection_plan,
            "primary_metric": primary_metric,
            "evaluation_metrics": evaluation_metrics,
            "metrics_rationale": metrics_rationale,
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("select_candidate_models_and_metrics", tool_inputs, output)
    return output
