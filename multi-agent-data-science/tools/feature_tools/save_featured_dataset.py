import json
from pathlib import Path

from langchain_core.tools import tool

from core.dataset_store import get_dataset
from core.env_utils import PROJECT_ROOT
from tools.tool_trace import trace_tool_call


@tool
def save_featured_dataset(dataset_id: str, model_name: str, output_dir: str = "") -> str:
    """将特征工程完成后的训练集保存为 CSV 文件。

    参数：
      dataset_id (str)：内存中已完成特征工程的数据集 ID。
      model_name (str)：对应的模型名称（如 "XGBoost"），用于命名输出文件。
        文件名格式：{model_name}_train_featured.csv
      output_dir (str)：保存目录，默认为 artifacts/features/。

    返回 JSON，包含：
      output_path：保存的 CSV 文件完整路径，此路径会被写入 state.feature_dataset_ids[model_name]。
      model_name：对应模型名称。
      rows：保存的行数。
      cols：保存的列数。
    """
    tool_inputs = {"dataset_id": dataset_id, "model_name": model_name, "output_dir": output_dir}
    try:
        df = get_dataset(dataset_id)
        out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "artifacts" / "features"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name}_train_featured.csv"
        df.to_csv(out_path, index=False)
        result = {
            "output_path": str(out_path),
            "model_name": model_name,
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
