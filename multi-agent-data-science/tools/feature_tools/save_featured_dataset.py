import json
from pathlib import Path

from langchain_core.tools import tool

from core.dataset_store import get_dataset
from core.env_utils import PROJECT_ROOT
from tools.tool_trace import trace_tool_call


@tool
def save_featured_dataset(
    dataset_id: str,
    model_name: str,
    output_dir: str = "",
    split: str = "train",
) -> str:
    """Save the feature-engineered dataset as a CSV file.

    Called once for each model's training set and once for its test set, distinguished by the split parameter:
      - split="train": filename is {model_name}_train_featured.csv; result is written to feature_dataset_ids.
      - split="test": filename is {model_name}_test_featured.csv; result is written to testing_feature_dataset_ids.

    Args:
      dataset_id (str): ID of the in-memory dataset that has completed feature engineering (training or test set).
      model_name (str): corresponding model name (e.g. "XGBoost"); used to name the output file.
      output_dir (str): save directory; defaults to artifacts/features/.
      split (str): "train" (default) or "test"; determines the file naming and the state field written to.

    Returns JSON containing:
      output_path: full path to the saved CSV file.
      model_name: corresponding model name.
      split: dataset type ("train" or "test").
      rows: number of saved rows.
      cols: number of saved columns.
    """
    tool_inputs = {"dataset_id": dataset_id, "model_name": model_name,
                   "output_dir": output_dir, "split": split}
    try:
        df = get_dataset(dataset_id)
        out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "artifacts" / "features"
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{model_name}_train_featured.csv" if split == "train" else f"{model_name}_test_featured.csv"
        out_path = out_dir / filename
        df.to_csv(out_path, index=False)
        result = {
            "output_path": str(out_path),
            "model_name": model_name,
            "split": split,
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("save_featured_dataset", tool_inputs, output)
    return output
