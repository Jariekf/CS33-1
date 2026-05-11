from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GlobalState(TypedDict, total=False):
    """全系统共享状态。

    这个 State 不应该作为模块级全局变量保存运行时数据，
    而是作为 LangGraph 的 StateGraph schema。
    """

    # conversation
    messages: Annotated[list[BaseMessage], add_messages]
    user_request: str

    # dataset
    data_path: str
    dataset_name: str
    description: str
    target_column: str
    task_type: str

    # profile
    dataset_profile: dict[str, Any]

    # cleaning
    dataset_id: str
    cleaning_plan: dict[str, Any]
    cleaned_data_path: str
    cleaning_report: dict[str, Any]
    cleaning_passed: bool

    # split
    is_split: bool
    training_data_path: str
    testing_data_path: str
    split_report: dict[str, Any]

    # split / cleaning originals (never overwritten by save_dataset handlers)
    raw_training_data_path: str
    raw_testing_data_path: str

    # per-target cleaning
    cleaning_target: str  # "train" or "test"
    train_cleaning_plan: dict[str, Any]
    test_cleaning_plan: dict[str, Any]
    train_cleaned_data_path: str
    test_cleaned_data_path: str
    train_cleaning_report: dict[str, Any]
    test_cleaning_report: dict[str, Any]
    train_cleaning_passed: bool
    test_cleaning_passed: bool

    # eda
    eda_report: dict[str, Any]
    eda_charts: list[dict[str, Any]]
    candidate_models: list[str]
    feature_engineering_plan: dict[str, Any]
    model_selection_plan: str

    # feature engineering
    feature_dataset_ids: dict[str, str]
    feature_preprocessors: dict[str, Any]

    # feature selection
    feature_selection_plan: dict[str, Any]
    feature_sets: dict[str, list[str]]
    feature_selection_report: dict[str, Any]

    # modeling
    modeling_plan: dict[str, Any]
    model_results: dict[str, Any]
    best_model: dict[str, Any]
    model_report: dict[str, Any]

    # final report
    business_report: str

    # business translator (BAT) intermediate context
    bat_rag_buckets: dict[str, Any]
    bat_arxiv_papers: list[dict[str, Any]]
    bat_style_skeletons: str
    bat_tool_rounds: int

    # workflow control
    current_stage: str
    next_action: str
    retry_counts: dict[str, int]
    max_retries: dict[str, int]
    errors: list[dict[str, Any]]

    # run isolation
    run_dir: str