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
    """将特征工程完成后的数据集保存为 CSV 文件。

    每个模型的训练集和测试集各调用一次，通过 split 参数区分：
      - split="train"：文件名为 {model_name}_train_featured.csv，结果写入 feature_dataset_ids。
      - split="test"：文件名为 {model_name}_test_featured.csv，结果写入 testing_feature_dataset_ids。

    参数：
      dataset_id (str)：内存中已完成特征工程的数据集 ID（训练集或测试集均可）。
      model_name (str)：对应的模型名称（如 "XGBoost"），用于命名输出文件。
      output_dir (str)：保存目录，默认为 artifacts/features/。
      split (str)："train"（默认）或 "test"，决定文件命名和写入的 state 字段。

    返回 JSON，包含：
      output_path：保存的 CSV 文件完整路径。
      model_name：对应模型名称。
      split：数据集类型（"train" 或 "test"）。
      rows：保存的行数。
      cols：保存的列数。
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
