import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call


@tool
def plot_distributions(file_path: str, output_dir: str = "") -> str:
    """为训练集数值列绘制分布直方图，保存为 PNG 文件。

    参数：
      file_path (str)：训练集 CSV 文件路径。
      output_dir (str)：图片保存目录。默认为训练集同级目录下的 eda_plots/ 子目录。

    返回 JSON，包含：
      output_dir：图片保存目录路径。
      plot_count：生成图片数量。
      plot_paths：每张图片的完整路径列表。
    """
    tool_inputs = {"file_path": file_path, "output_dir": output_dir}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise ValueError("训练集中没有数值列，无法绘制分布图。")

        out = Path(output_dir) if output_dir else Path(file_path).parent / "eda_plots"
        out.mkdir(parents=True, exist_ok=True)

        plot_paths = []
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[col].dropna().hist(ax=ax, bins=30, edgecolor="black")
            ax.set_title(f"Distribution: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            save_path = out / f"{col}_distribution.png"
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            plot_paths.append(str(save_path))

        result = {
            "output_dir": str(out),
            "plot_count": len(plot_paths),
            "plot_paths": plot_paths,
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("plot_distributions", tool_inputs, output)
    return output
