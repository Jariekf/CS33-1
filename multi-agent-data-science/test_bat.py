import os
import sys
from pathlib import Path
from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))

from agent.business_translator_agent import build_business_translator_agent

agent = build_business_translator_agent(show_graph=False)
demo_run = PROJECT_ROOT / "artifacts" / "workspace" / "bat_standalone_test"
demo_run.mkdir(parents=True, exist_ok=True)

events = agent.stream(
    {
        "messages": [HumanMessage(content="提升用户留存率并给出相应的营销策略")],
        "user_request": "提升用户留存率并给出相应的营销策略",
        "description": "这是关于用户留存的数据集，包含营销和技术手段的特征。",
        "task_type": "classification",
        "target_column": "retained",
        "run_dir": str(demo_run),
        "best_model": {
            "model_name": "LightGBM",
            "metrics": {"accuracy": 0.95},
            "task_type": "classification",
        },
    },
    stream_mode="values",
    config={"recursion_limit": 80},
)

for event in events:
    if "current_stage" in event and event["current_stage"] == "business_report_done":
        print("\n\n--- GENERATED REPORT ---\n")
        print(event.get("business_report", ""))
        break
