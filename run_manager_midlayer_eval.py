"""
Run manager summary/fusion evaluation without full pipeline execution.
"""
from __future__ import annotations

import sys

from core.manager_prompt_eval import write_manager_midlayer_eval_report


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python run_manager_midlayer_eval.py <data_path> <user_goal> [cleaning_hints]")
        return 1
    data_path = sys.argv[1]
    user_goal = sys.argv[2]
    cleaning_hints = sys.argv[3] if len(sys.argv) > 3 else ""
    metrics = write_manager_midlayer_eval_report(
        data_path=data_path,
        user_goal=user_goal,
        manager_cleaning_hints=cleaning_hints,
    )
    print("[+] manager midlayer eval done")
    print(f"summary_json_parse_ok={metrics['summary_json_parse_ok']}")
    print(f"summary_required_fields_ok={metrics['summary_required_fields_ok']}")
    print(f"summary_empty_rate={metrics['summary_empty_rate']}")
    print(f"fusion_empty_rate={metrics['fusion_empty_rate']}")
    print(f"fusion_repetition_rate={metrics['fusion_repetition_rate']}")
    print(f"fusion_code_leakage_flag={metrics['fusion_code_leakage_flag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
