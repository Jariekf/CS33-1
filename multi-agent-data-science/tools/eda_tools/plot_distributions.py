import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call


@tool
def plot_distributions(file_path: str, output_dir: str = "") -> str:
    """Plot distribution histograms for numeric columns in the training set and save them as PNG files.

    Args:
      file_path (str): path to the training-set CSV file.
      output_dir (str): directory in which to save the images. Defaults to the eda_plots/ subdirectory in the same directory as the training set.

    Returns JSON containing:
      output_dir: path to the image save directory.
      plot_count: number of images generated.
      plot_paths: list of full paths to each image.
    """
    tool_inputs = {"file_path": file_path, "output_dir": output_dir}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns in the training set; cannot plot distributions.")

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
