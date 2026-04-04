import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.manager import _generate_global_plan, _generate_search_queries
from agents.business_translator import _build_rag_queries, _generate_business_report


def test_manager():
    print("--- Testing Manager Agent ---")
    user_goal = "Predict customer churn to improve retention."
    df_preview = (
        "Columns (3): ['id', 'age', 'churn']\n"
        "Shape: (100, 3)\n"
        "Dtypes:\n id int64\n age int64\n churn int64\n"
        "Head:\n   id  age  churn\n0   1   25      0\n1   2   35      1\n2   3   45      0"
    )

    queries = _generate_search_queries(user_goal, df_preview)
    print(f"Generated Search Queries: {queries}")

    docs = [
        {
            "title": "Churn Prediction using xgboost",
            "summary": "A comprehensive study on predicting customer churn using gradient boosting methods.",
        }
    ]
    rap_context = "\n".join(
        f"- {d.get('title', 'N/A')}: {(d.get('summary') or '')[:200]}" for d in docs
    )
    plan = _generate_global_plan(
        user_goal,
        df_preview,
        rap_context,
        user_requirement_summary="Predict churn from tabular customer data.",
    )
    print(f"Generated Global Plan: {plan}")


def test_business_translator():
    print("\n--- Testing Business Translator Agent ---")
    user_goal = "Predict customer churn to improve retention."
    top_features = ["age", "account_balance", "num_transactions"]

    queries = _build_rag_queries(user_goal, top_features)
    print(f"Generated RAG Queries: {queries}")

    model_summary = "XGBoost achieved an F1 score of 0.85 with good precision."
    feature_importance = {"age": 0.4, "account_balance": 0.35, "num_transactions": 0.25}
    docs = [
        {
            "title": "Age and Churn",
            "summary": "Study showing younger demographics have higher churn risk.",
        }
    ]

    report = _generate_business_report(
        user_goal, model_summary, feature_importance, top_features, docs
    )
    print(f"Generated Business Report:\n{report[:500]}...\n(Truncated for preview)")


if __name__ == "__main__":
    test_manager()
    test_business_translator()
