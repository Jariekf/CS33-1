"""
多智能体数据分析系统 - 主入口
基于 LangGraph 构建的全流水线自动化数据分析与商业洞察系统。
"""
from datetime import datetime

from core.graph import build_graph
from core.manager_prompt_eval import write_manager_prompt_eval_report
from core.state import GraphState
from core.logger import get_agent_logger

sys_logger = get_agent_logger("sys")


def run(data_path: str, user_goal: str, manager_cleaning_hints: str = "") -> str:
    """
    启动多智能体分析流水线。

    Args:
        data_path: 数据集文件路径
        user_goal: 用户的自然语言业务目标
        manager_cleaning_hints: 可选；清洗规则/contract 等文字，写入检索摘要并与规划对齐

    Returns:
        最终的商业洞察分析报告文本
    """
    run_started_at = datetime.now()
    # 构建 LangGraph 工作流
    sys_logger.info(f"Main execution started with data_path: '{data_path}', goal: '{user_goal}'")
    app = build_graph()

    # 初始化输入状态
    initial_state: GraphState = {
        "data_path": data_path,
        "user_goal": user_goal,
        "manager_cleaning_hints": manager_cleaning_hints,
        "global_plan": {},
        "current_phase": "",
        "current_data_path": data_path,
        "current_code": "",
        "current_execution_log": "",
        "current_error": "",
        "retry_count": 0,
        "current_review": None,
        "outer_retry_total": 0,
        "memory": [],
        "final_business_report": "",
    }

    # 执行工作流
    sys_logger.info("Starting graph invocation...")
    final_state = app.invoke(initial_state)
    sys_logger.info("Graph invocation finished successfully.")

    try:
        write_manager_prompt_eval_report(
            initial_state=initial_state,
            final_state=final_state,
            run_started_at=run_started_at,
        )
    except Exception as e:
        sys_logger.warning("Manager prompt eval report generation skipped due to error: %s", e)

    return final_state.get("final_business_report", "未生成报告")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python main.py <data_path> <user_goal>")
        print('Example: python main.py ./data/train.csv "预测客户流失并给出挽留策略"')
        sys.exit(1)

    report = run(sys.argv[1], sys.argv[2])
    
    # 【修复】先落盘保存，防止终端打印遇到 Unicode 报错导致没保存上
    import os
    os.makedirs("./workspace", exist_ok=True)
    report_path = "./workspace/business_insight_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[+] 最终报告已成功生成并安全保存至: {report_path}")

    print("\n" + "=" * 60)
    print("最终商业分析报告")
    print("=" * 60)
    try:
        print(report)
    except UnicodeEncodeError:
        print("[!] 报告包含终端无法显示的特殊字符，请直接打开 workspace 里的文件查看！")
