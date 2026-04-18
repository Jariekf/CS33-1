import sys
from pathlib import Path
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from agents import manager_retrieval as mr
from core import manager_prompt_eval as mpe


class TestManagerMidlayerEval(unittest.TestCase):
    def test_summary_meta_fields_and_faithfulness_warning(self):
        reply = (
            '{"user_requirement_summary":"根据 feature1 和 feature2 预测 churn，遵守去重约束",'
            '"downstream_task":"churn prediction","application_domain":"telecom","area":"tabular classification"}'
        )
        with patch("agents.manager_retrieval.call_llm", return_value=reply):
            summary, slots, meta = mr.summarize_and_extract_slots_with_meta(
                user_goal="Predict customer churn",
                df_preview=(
                    "Columns (5): ['id', 'feature1', 'feature2', 'feature3', 'target']\n"
                    "Shape: (5, 5)\nDtypes:\nfeature1 float64\nfeature2 float64"
                ),
                cleaning_hints="需要去重并控制删除比例",
            )
        self.assertTrue(summary)
        self.assertIn("downstream_task", slots)
        self.assertTrue(meta["json_parse_ok"])
        self.assertTrue(meta["required_fields_ok"])
        self.assertEqual(meta["empty_rate"], 0.0)
        self.assertIn("faithfulness_warning_count", meta)

    def test_fusion_meta_fields(self):
        with patch("agents.manager_retrieval.call_llm", return_value="- use robust scaling\n- use robust scaling"):
            context, meta = mr.retrieve_and_fuse_rap_context_with_meta(
                user_requirement_summary="Predict churn from tabular data",
                slots={
                    "downstream_task": "churn prediction",
                    "application_domain": "telecom",
                    "area": "tabular classification",
                },
                user_goal="Predict churn",
            )
        self.assertTrue(context)
        self.assertIn("empty_rate", meta)
        self.assertIn("repetition_rate", meta)
        self.assertIn("code_leakage_flag", meta)

    def test_failure_attribution_prefers_manager_input_issue(self):
        label, owner, reason = mpe._infer_failure_attribution(
            json_parse_ok=False,
            schema_ok=False,
            used_fallback=True,
            pipeline_success=False,
            summary_json_parse_ok=True,
            summary_required_fields_ok=True,
            summary_empty_rate=0.0,
            fusion_empty_rate=0.0,
            fusion_code_leakage_flag=False,
        )
        self.assertEqual(label, "manager_input_issue")
        self.assertEqual(owner, "manager")
        self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main()
import sys
from pathlib import Path
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from agents import manager_retrieval as mr
from core import manager_prompt_eval as mpe


class TestManagerMidlayerEval(unittest.TestCase):
    def test_summary_meta_fields_and_faithfulness_warning(self):
        reply = (
            '{"user_requirement_summary":"根据 feature1 和 feature2 预测 churn，遵守去重约束",'
            '"downstream_task":"churn prediction","application_domain":"telecom","area":"tabular classification"}'
        )
        with patch("agents.manager_retrieval.call_llm", return_value=reply):
            summary, slots, meta = mr.summarize_and_extract_slots_with_meta(
                user_goal="Predict customer churn",
                df_preview=(
                    "Columns (5): ['id', 'feature1', 'feature2', 'feature3', 'target']\n"
                    "Shape: (5, 5)\nDtypes:\nfeature1 float64\nfeature2 float64"
                ),
                cleaning_hints="需要去重并控制删除比例",
            )
        self.assertTrue(summary)
        self.assertIn("downstream_task", slots)
        self.assertTrue(meta["json_parse_ok"])
        self.assertTrue(meta["required_fields_ok"])
        self.assertEqual(meta["empty_rate"], 0.0)
        self.assertIn("faithfulness_warning_count", meta)

    def test_fusion_meta_fields(self):
        with patch("agents.manager_retrieval.call_llm", return_value="- use robust scaling\n- use robust scaling"):
            context, meta = mr.retrieve_and_fuse_rap_context_with_meta(
                user_requirement_summary="Predict churn from tabular data",
                slots={
                    "downstream_task": "churn prediction",
                    "application_domain": "telecom",
                    "area": "tabular classification",
                },
                user_goal="Predict churn",
            )
        self.assertTrue(context)
        self.assertIn("empty_rate", meta)
        self.assertIn("repetition_rate", meta)
        self.assertIn("code_leakage_flag", meta)

    def test_failure_attribution_prefers_manager_input_issue(self):
        label, owner, reason = mpe._infer_failure_attribution(
            json_parse_ok=False,
            schema_ok=False,
            used_fallback=True,
            pipeline_success=False,
            summary_json_parse_ok=True,
            summary_required_fields_ok=True,
            summary_empty_rate=0.0,
            fusion_empty_rate=0.0,
            fusion_code_leakage_flag=False,
        )
        self.assertEqual(label, "manager_input_issue")
        self.assertEqual(owner, "manager")
        self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main()
