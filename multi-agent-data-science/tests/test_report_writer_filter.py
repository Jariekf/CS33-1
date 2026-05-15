"""Unit tests for the non-BAT business-advice filter in report_writer."""

from __future__ import annotations

from pathlib import Path

from core.report_writer import _strip_non_bat_recommendations, save_agent_report


class _Msg:
    """Lightweight stand-in for a LangChain message object."""

    def __init__(self, content: str) -> None:
        self.content = content


def test_strip_removes_chinese_business_advice_section() -> None:
    report = (
        "## 模型对比\n"
        "RandomForest 表现最佳。\n"
        "\n"
        "### 业务建议\n"
        "1. 建议加大营销投放。\n"
        "2. 优先联系高 balance 客户。\n"
        "\n"
        "### 评估指标\n"
        "f1_macro=0.78\n"
    )
    out = _strip_non_bat_recommendations(report)
    assert "## 模型对比" in out
    assert "RandomForest 表现最佳" in out
    assert "### 评估指标" in out
    assert "f1_macro=0.78" in out
    assert "业务建议" not in out
    assert "建议加大营销投放" not in out


def test_strip_removes_english_next_steps_at_eof() -> None:
    report = (
        "# Model Report\n"
        "\n"
        "## Results\n"
        "Accuracy = 0.91\n"
        "\n"
        "## Next Steps\n"
        "- Tune threshold\n"
        "- Deploy to production\n"
    )
    out = _strip_non_bat_recommendations(report)
    assert "## Results" in out
    assert "Accuracy = 0.91" in out
    assert "Next Steps" not in out
    assert "Deploy to production" not in out


def test_strip_handles_recommendations_plural() -> None:
    report = (
        "## Findings\n"
        "Class imbalance detected.\n"
        "\n"
        "### Recommendations\n"
        "Use SMOTE.\n"
        "\n"
        "### Recommended Actions\n"
        "Adjust class weights.\n"
        "\n"
        "## Conclusion\n"
        "Model is ready.\n"
    )
    out = _strip_non_bat_recommendations(report)
    assert "## Findings" in out
    assert "Class imbalance detected" in out
    assert "## Conclusion" in out
    assert "Model is ready" in out
    assert "Recommendations" not in out
    assert "Recommended Actions" not in out
    assert "SMOTE" not in out
    assert "Adjust class weights" not in out


def test_strip_skip_when_heading_inside_code_fence() -> None:
    report = (
        "## Code Sample\n"
        "```\n"
        "## 业务建议\n"
        "print('this is inside a code block, must not be stripped')\n"
        "```\n"
        "\n"
        "## Findings\n"
        "OK.\n"
    )
    out = _strip_non_bat_recommendations(report)
    assert "## Findings" in out
    assert "print('this is inside a code block" in out
    # The fenced "## 业务建议" line should survive because it's inside ```...```.
    assert "## 业务建议" in out


def test_strip_drops_nested_subsections_until_same_level() -> None:
    report = (
        "## Modeling Results\n"
        "ok\n"
        "\n"
        "## 业务建议\n"
        "first paragraph\n"
        "### 客户分群\n"
        "details\n"
        "### ROI 优化\n"
        "details\n"
        "\n"
        "## Final Notes\n"
        "tail\n"
    )
    out = _strip_non_bat_recommendations(report)
    assert "## Modeling Results" in out
    assert "## Final Notes" in out
    assert "tail" in out
    assert "客户分群" not in out
    assert "ROI 优化" not in out
    assert "业务建议" not in out


def test_strip_idempotent_on_clean_text() -> None:
    report = (
        "## EDA Findings\n"
        "All numeric.\n"
        "\n"
        "## Correlation\n"
        "Weak signals.\n"
    )
    out1 = _strip_non_bat_recommendations(report)
    out2 = _strip_non_bat_recommendations(out1)
    assert out1 == out2
    assert "EDA Findings" in out1
    assert "Correlation" in out1


def test_save_agent_report_filters_non_bat_agent(tmp_path: Path) -> None:
    msg = _Msg(
        "## Result\nF1=0.78\n\n### 业务建议\n- 建议加大营销\n\n### Next Steps\n- Deploy\n"
    )
    out_path = save_agent_report(
        agent_name="model_agent",
        message=msg,
        model="RandomForest",
        run_dir=str(tmp_path),
    )
    text = Path(out_path).read_text(encoding="utf-8")
    assert "F1=0.78" in text
    assert "业务建议" not in text
    assert "Next Steps" not in text
    assert "建议加大营销" not in text
    assert "Deploy" not in text


def test_save_agent_report_preserves_business_translator(tmp_path: Path) -> None:
    msg = _Msg(
        "## Executive Summary\nWin.\n\n## Recommended Actions\n- Pilot 90-day campaign\n"
    )
    out_path = save_agent_report(
        agent_name="business_translator",
        message=msg,
        run_dir=str(tmp_path),
    )
    text = Path(out_path).read_text(encoding="utf-8")
    assert "Executive Summary" in text
    assert "Recommended Actions" in text
    assert "Pilot 90-day campaign" in text


def test_save_agent_report_empty_after_filter_inserts_placeholder(tmp_path: Path) -> None:
    msg = _Msg("## 业务建议\n- 营销\n- 推广\n")
    out_path = save_agent_report(
        agent_name="model_agent",
        message=msg,
        run_dir=str(tmp_path),
    )
    text = Path(out_path).read_text(encoding="utf-8")
    # Original "## 业务建议" heading must be gone (any heading level)
    for level in ("# ", "## ", "### ", "#### "):
        assert f"\n{level}业务建议" not in text
        assert not text.startswith(f"{level}业务建议")
    assert "营销" not in text
    assert "推广" not in text
    assert "业务建议章节已被过滤" in text
