import os
import sys
import tempfile
from datetime import datetime
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
            '{"user_requirement_summary":"Predict churn from feature1 and feature2 while respecting deduplication constraints",'
            '"downstream_task":"churn prediction","application_domain":"telecom","area":"tabular classification"}'
        )
        with patch("agents.manager_retrieval.call_llm", return_value=reply):
            summary, slots, meta = mr.summarize_and_extract_slots_with_meta(
                user_goal="Predict customer churn",
                df_preview=(
                    "Columns (5): ['id', 'feature1', 'feature2', 'feature3', 'target']\n"
                    "Shape: (5, 5)\nDtypes:\nfeature1 float64\nfeature2 float64"
                ),
                cleaning_hints="deduplicate rows and control the row-drop ratio",
            )
        self.assertTrue(summary)
        self.assertIn("downstream_task", slots)
        self.assertTrue(meta["json_parse_ok"])
        self.assertTrue(meta["required_fields_ok"])
        self.assertEqual(meta["empty_rate"], 0.0)
        self.assertIn("faithfulness_warning_count", meta)

    def test_fusion_meta_contract(self):
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
        # Fusion meta includes health metrics and RAP status fields.
        self.assertIn("empty_rate", meta)
        self.assertIn("repetition_rate", meta)
        self.assertIn("code_leakage_flag", meta)
        self.assertIn("rap_enabled", meta)
        self.assertIn("arxiv_raw_chars", meta)
        self.assertIn("web_raw_chars", meta)
        self.assertIn("pwc_raw_chars", meta)
        self.assertIn("kaggle_raw_chars", meta)

    def test_disabled_and_empty_fusion_messages_stay_distinct(self):
        disabled_text = "No external retrieval results; RAP is disabled, so planning uses only the user goal and data preview."
        empty_text = "No external retrieval results; planning uses only the user goal and data preview."
        self.assertNotEqual(disabled_text, empty_text)
        self.assertIn("disabled", disabled_text.lower())
        self.assertNotIn("disabled", empty_text.lower())

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

    def test_midlayer_eval_writes_only_under_manager_eval_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = os.fspath(Path(tmpdir))
            latest_dir = os.fspath(Path(tmpdir) / "manager_eval" / "latest")
            history_dir = os.fspath(Path(tmpdir) / "manager_eval" / "history")
            with (
                patch.object(mpe, "WORKSPACE_DIR", workspace_dir),
                patch.object(mpe, "MANAGER_EVAL_DIR", os.fspath(Path(tmpdir) / "manager_eval")),
                patch.object(mpe, "MANAGER_EVAL_LATEST_DIR", latest_dir),
                patch.object(mpe, "MANAGER_EVAL_HISTORY_DIR", history_dir),
                patch("agents.manager._get_data_preview", return_value="Columns (2): ['x', 'y']"),
                patch(
                    "agents.manager_retrieval.summarize_and_extract_slots_with_meta",
                    return_value=(
                        "Summary text",
                        {
                            "downstream_task": "classification",
                            "application_domain": "banking",
                            "area": "tabular classification",
                        },
                        {
                            "json_parse_ok": True,
                            "required_fields_ok": True,
                            "empty_rate": 0.0,
                            "faithfulness_warning_count": 0,
                        },
                    ),
                ),
                patch(
                    "agents.manager_retrieval.retrieve_and_fuse_rap_context_with_meta",
                    return_value=(
                        "No external retrieval results; planning uses only the user goal and data preview.",
                        {
                            "empty_rate": 0.0,
                            "repetition_rate": 0.0,
                            "code_leakage_flag": False,
                            "rap_enabled": True,
                            "arxiv_raw_chars": 0,
                            "web_raw_chars": 0,
                            "pwc_raw_chars": 0,
                            "kaggle_raw_chars": 0,
                        },
                    ),
                ),
            ):
                mpe.write_manager_midlayer_eval_report(
                    data_path="./data/test_data.csv",
                    user_goal="Predict churn",
                )

            self.assertTrue(Path(latest_dir, "manager_midlayer_eval_latest.md").exists())
            self.assertTrue(Path(latest_dir, "manager_midlayer_eval_latest.csv").exists())
            self.assertFalse(Path(workspace_dir, "manager_midlayer_eval_latest.md").exists())
            self.assertFalse(Path(workspace_dir, "manager_midlayer_eval_latest.csv").exists())


class TestManagerPromptEvalReport(unittest.TestCase):
    def test_prompt_eval_report_writes_only_under_manager_eval_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = os.fspath(Path(tmpdir))
            latest_dir = os.fspath(Path(tmpdir) / "manager_eval" / "latest")
            history_dir = os.fspath(Path(tmpdir) / "manager_eval" / "history")
            with (
                patch.object(mpe, "WORKSPACE_DIR", workspace_dir),
                patch.object(mpe, "MANAGER_EVAL_DIR", os.fspath(Path(tmpdir) / "manager_eval")),
                patch.object(mpe, "MANAGER_EVAL_LATEST_DIR", latest_dir),
                patch.object(mpe, "MANAGER_EVAL_HISTORY_DIR", history_dir),
            ):
                mpe.write_manager_prompt_eval_report(
                    initial_state={
                        "data_path": "./data/test_data.csv",
                        "user_goal": "Predict churn",
                        "manager_cleaning_hints": "",
                    },
                    final_state={
                        "global_plan": {
                            "_manager_plan_eval": {
                                "prompt_hash": "abc123",
                                "system_prompt": "prompt",
                                "user_prompt_preview": "preview",
                                "raw_reply_preview": "reply",
                                "json_parse_ok": True,
                                "schema_ok": True,
                                "used_fallback": False,
                            },
                            "_manager_summary_eval": {
                                "json_parse_ok": True,
                                "required_fields_ok": True,
                                "empty_rate": 0.0,
                                "faithfulness_warning_count": 0,
                            },
                            "_manager_fusion_eval": {
                                "empty_rate": 0.0,
                                "repetition_rate": 0.0,
                                "code_leakage_flag": False,
                            },
                        },
                        "outer_retry_total": 0,
                    },
                    run_started_at=datetime.now(),
                )

            self.assertTrue(Path(latest_dir, "manager_prompt_eval_latest.md").exists())
            self.assertTrue(Path(latest_dir, "manager_prompt_eval_latest.csv").exists())
            self.assertFalse(Path(workspace_dir, "manager_prompt_eval_latest.md").exists())
            self.assertFalse(Path(workspace_dir, "manager_prompt_eval_latest.csv").exists())


if __name__ == "__main__":
    unittest.main()
