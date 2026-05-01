"""
Supervisor Agent (质量把控与验证智能体)
职责：
  1. 接收 Worker 的代码产出与执行日志
  2. 利用 LLM 进行结构化审查打分 (Pydantic ReviewReport)
  3. 决策路由：通过 → 落盘报告 + 摘要入 Memory + 推进；不通过 → 反馈 + 重试
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime

from core.logger import get_agent_logger
from pydantic import BaseModel, Field, ValidationError, field_validator

from core.state import GraphState
from core.llm import call_llm
from config import SUPERVISOR_PASS_THRESHOLD, MAX_OUTER_RETRY, WORKSPACE_DIR

logger = get_agent_logger("supervisor")

# ======================================================================
# 结构化审查报告定义
# ======================================================================

class ReviewReport(BaseModel):
    """Supervisor 产出的标准化审查报告。"""
    score: int = Field(description="Quality score from 1 to 5; 3 is the passing threshold.")
    passed: bool = Field(description="Whether the current phase is allowed to pass.")
    feedback_for_retry: str = Field(default="", description="Specific fix guidance when the phase does not pass.")
    summary_for_memory: str = Field(default="", description="Concise memory summary written after pass, within 100 words.")
    detailed_report: str = Field(default="", description="Detailed Markdown review report written after pass.")

    # LLMs often fill fields that are not meaningful in the current context with
    # null (e.g. setting ``feedback_for_retry`` to null when ``passed`` is true).
    # In Pydantic v2 a field default only applies when the key is missing; an
    # explicit null still runs type validation and fails on ``str``. Coerce
    # None -> "" here so that a legitimate LLM reply is not rejected by the
    # schema and forced into the degraded-pass branch.
    @field_validator(
        "feedback_for_retry",
        "summary_for_memory",
        "detailed_report",
        mode="before",
    )
    @classmethod
    def _none_to_empty_string(cls, v):
        return "" if v is None else v


# ======================================================================
# Prompt 模板
# ======================================================================

_SYSTEM_PROMPT_REVIEW = """\
You are an expert Data Science Supervisor for a multi-agent analytics pipeline.
Your job is to decide whether a worker agent's code output is good enough to
advance to the next phase. Review the worker's implementation against the
manager's plan, the execution log, and data science correctness.

Be evidence-based and practical: pass work that is broadly correct, runnable,
and useful for the next phase, but fail clear blockers. Score quality from 1 to
5. A score of {threshold} or above is passing only when there are no blocker
issues. Return a structured review report in strict JSON."""

_USER_PROMPT_REVIEW = """\
# PHASE #
{phase}

# PLAN #
{plan}

# WORKER CODE #
```python
{code}
```

# EXECUTION LOG #
```text
{execution_log}
```

# INSTRUCTIONS #
1. Review the work using this checklist:
   - Plan coverage: Did the code complete the core requirements in PLAN?
   - Execution evidence: Does the log show the code ran successfully and produced usable outputs?
   - Data science correctness: Are target variables, train/test splits, metrics, transformations, and evaluation logic appropriate?
   - Robustness: Are obvious edge cases, missing values, column availability, and file paths handled well enough for this phase?
   - Next-phase readiness: Are outputs, artifacts, or summaries sufficient for the next agent to continue?

2. Fail the phase if any blocker is present:
   - Runtime failure or traceback in the execution log.
   - A core PLAN requirement is missing.
   - Output needed by the next phase is missing, empty, or unusable.
   - Obvious data leakage, wrong target variable, wrong evaluation metric, or invalid train/test split.
   - Code appears not to have been executed, or the log does not support the claimed result.

3. Scoring rubric:
   - 5: Complete, correct, robust, and clearly evidenced by the log.
   - 4: Solid and passable, with only minor non-blocking issues.
   - 3: Acceptable and passable, but has meaningful risks or gaps that do not block the next phase.
   - 2: Major issue; should retry.
   - 1: Runtime failure, unusable output, or clearly wrong logic.

4. If it passes, set `passed=true`, provide `summary_for_memory` within 100 words, and write a useful `detailed_report`.

5. If it fails, set `passed=false` and write actionable `feedback_for_retry` using this structure:
   - Root cause:
   - Fix steps:
   - Re-run check:

6. The `detailed_report` must be concise Markdown and include:
   - Plan coverage: key items met and missing.
   - Execution evidence: quote or summarize 1-3 relevant log signals.
   - Quality assessment: correctness, robustness, and next-phase readiness.
   - Risks: 1-3 remaining risks, even if passing.

7. YOU MUST RETURN ONLY VALID JSON: a single object, no prose, no markdown fences, no trailing commas. Follow this format exactly:
{{
  "score": <int 1-5>,
  "passed": <boolean>,
  "feedback_for_retry": "<string>",
  "summary_for_memory": "<string>",
  "detailed_report": "<string markdown format>"
}}
8. Inside string values, especially `detailed_report`, ALL double quotes must be escaped as `\\"`, and newlines as `\\n`. Do NOT emit raw/unescaped quotes or raw line breaks inside a string, because they will break JSON parsing.
"""


# Stricter re-ask template used when the first reply is not parseable JSON.
_USER_PROMPT_REPAIR = """\
Your previous reply could not be parsed as strict JSON. The parser error was:
{parse_error}

Re-emit the review as a SINGLE JSON object matching the schema below, with
NO prose, NO markdown fences, NO trailing commas, and ALL double quotes /
newlines inside string values properly escaped (\\" and \\n).

Schema:
{{
  "score": <int 1-5>,
  "passed": <boolean>,
  "feedback_for_retry": "<string>",
  "summary_for_memory": "<string>",
  "detailed_report": "<string markdown format>"
}}

Your previous (invalid) reply was:
---
{previous_reply}
---
"""


# ======================================================================
# Supervisor 节点
# ======================================================================

def supervisor_node(state: GraphState) -> dict:
    """
    Supervisor 的 LangGraph 节点函数。
    读取 Worker 的执行日志，调用 LLM 进行评审，并更新 State。
    """
    phase = state["current_phase"]
    code = state.get("current_code", "")
    execution_log = state.get("current_execution_log", "")
    current_error = (state.get("current_error") or "").strip()
    logger.info(f"Supervisor node started for phase: {phase}.")
    plan = state.get("global_plan", {}).get(f"{phase}_plan", {})

    # If the worker already exhausted its own retries and surfaced a runtime
    # error, do not let the Supervisor's JSON/LLM degradation path mask that
    # failure as a pass. Bounce the phase back immediately with the concrete
    # worker error so the outer retry loop can react deterministically.
    if current_error:
        logger.warning(
            "Worker phase %s entered Supervisor with current_error present; forcing review failure.",
            phase,
        )
        review = ReviewReport(
            score=1,
            passed=False,
            feedback_for_retry=current_error,
            summary_for_memory="",
            detailed_report="",
        )
        updates: dict = {"current_review": review.model_dump()}
        updates["current_error"] = review.feedback_for_retry
        updates["retry_count"] = state.get("retry_count", 0) + 1
        updates["outer_retry_total"] = state.get("outer_retry_total", 0) + 1
        return updates

    # ------------------------------------------------------------------
    # 调用 LLM 进行结构化审查
    # ------------------------------------------------------------------
    review = _llm_review(phase, plan, code, execution_log)
    logger.info(f"LLM review completed. Score: {review.score}, Passed: {review.passed}")

    # ------------------------------------------------------------------
    # 根据审查结果，构造状态更新
    # ------------------------------------------------------------------
    updates: dict = {"current_review": review.model_dump()}

    if review.passed:
        logger.info(f"Review passed for phase: {phase}. Writing report to output log.")
        # ---- 通过：落盘详情 + 摘要入 Memory ----
        report_path = os.path.join(WORKSPACE_DIR, f"{phase}_report.md")
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(review.detailed_report)

        memory_entry = f"[{phase}] {review.summary_for_memory} Details: {report_path}"
        updates["memory"] = [memory_entry]
        updates["current_error"] = ""
        updates["retry_count"] = 0
    else:
        logger.warning(f"Review failed for phase: {phase}. Feedback: {review.feedback_for_retry}")
        # ---- 不通过：附带反馈信息，累加重试计数 ----
        updates["current_error"] = review.feedback_for_retry
        updates["retry_count"] = state.get("retry_count", 0) + 1
        updates["outer_retry_total"] = state.get("outer_retry_total", 0) + 1

    return updates


# ======================================================================
# 条件路由函数 (供 graph.py 的 add_conditional_edges 调用)
# ======================================================================

def supervisor_router(state: GraphState) -> str:
    """
    根据 Supervisor 审查结果决定路由方向。
    返回:
        "pass"  — 审查通过，推进至下一阶段
        "retry" — 审查不通过，退回 Worker 重修
        "error" — 超出最大重试次数，终止流程
    """
    review = state.get("current_review")
    if review and review.get("passed"):
        return "pass"

    if state.get("retry_count", 0) >= MAX_OUTER_RETRY:
        return "error"

    return "retry"


# ======================================================================
# 私有辅助函数
# ======================================================================

def _extract_balanced_json_object(text: str) -> str | None:
    """
    Find the first top-level balanced ``{...}`` object inside arbitrary LLM
    prose. Tracks string state so that ``{`` / ``}`` inside JSON string values
    do not throw off the match.

    Returns the raw substring (not yet parsed) or None if no balanced object
    is found.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _strip_markdown_fences(text: str) -> str:
    """
    Drop leading/trailing ``` / ```json fences if the LLM wrapped its JSON in
    a code block despite being told not to. Untouched if no fence is present.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```(?:json|JSON)?\s*", "", stripped, count=1)
    stripped = re.sub(r"\s*```\s*$", "", stripped, count=1)
    return stripped.strip()


def _parse_review_reply(reply: str) -> tuple[ReviewReport | None, str | None]:
    """
    Try to parse an LLM reply into a :class:`ReviewReport`.

    Returns ``(report, None)`` on success, or ``(None, err_msg)`` on failure.
    Applies a couple of cheap repairs (fence stripping, balanced-brace
    extraction, trailing-comma removal) before giving up. Does NOT try to
    fix unescaped quotes inside string values — those come from the model
    and are better handled by a repair re-ask.
    """
    if not reply:
        return None, "empty LLM reply"

    text = _strip_markdown_fences(reply)
    payload = _extract_balanced_json_object(text)
    if payload is None:
        return None, "no balanced JSON object found in reply"

    last_err: str | None = None
    for candidate in (payload, re.sub(r",(\s*[}\]])", r"\1", payload)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = f"JSONDecodeError: {e}"
            continue
        try:
            return ReviewReport(**data), None
        except ValidationError as e:
            return None, f"ReviewReport validation failed: {e}"
    return None, last_err or "unknown JSON parse failure"


def _persist_raw_reply_for_debug(phase: str, reply: str, reason: str) -> str | None:
    """Dump the raw LLM reply so we can inspect what broke parsing. Best-effort."""
    try:
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            WORKSPACE_DIR, f"supervisor_raw_reply_{phase}_{stamp}.txt"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# reason: {reason}\n# phase: {phase}\n\n{reply}")
        return path
    except Exception as e:
        logger.warning("Failed to persist raw Supervisor reply: %s", e)
        return None


def _llm_review(
    phase: str, plan: dict, code: str, execution_log: str
) -> ReviewReport:
    """
    Call the LLM to produce a structured review.

    Failure tiers (unlike the previous implementation, we no longer collapse
    "parse failure" and "LLM unreachable" into a single degrade-pass path):
      1. LLM returns parseable JSON -> normal flow.
      2. LLM returns JSON but parsing fails -> persist the raw reply and make
         one "please re-emit strict JSON" repair call; if it parses, flow on;
         otherwise fall through to tier 3.
      3. LLM unreachable or both attempts fail to parse -> degrade-pass so
         the pipeline does not stall, with ``detailed_report`` clearly marked
         as a placeholder and pointing at the persisted raw reply for debug.
    """
    system_prompt = _SYSTEM_PROMPT_REVIEW.format(threshold=SUPERVISOR_PASS_THRESHOLD)
    user_prompt = _USER_PROMPT_REVIEW.format(
        phase=phase,
        plan=json.dumps(plan, ensure_ascii=False, indent=2),
        code=code,
        execution_log=execution_log,
    )

    # ---- First call ----
    reply = call_llm(system_prompt, user_prompt, agent_name="supervisor")
    if not reply:
        return _degrade_pass(
            phase,
            execution_log,
            reason="empty LLM reply (call failed or was blocked)",
            raw_dump_path=None,
        )

    report, err = _parse_review_reply(reply)
    if report is not None:
        return _apply_hard_review_guards(report, execution_log)

    # ---- Second call (repair re-ask) ----
    dump_path = _persist_raw_reply_for_debug(
        phase, reply, reason=f"initial parse failed: {err}"
    )
    logger.warning(
        "[SupervisorJsonParseFailed] phase=%s reason=%s; attempting repair re-ask. "
        "raw reply dumped to %s",
        phase,
        err,
        dump_path,
    )

    repair_prompt = _USER_PROMPT_REPAIR.format(
        parse_error=err,
        previous_reply=reply,
    )
    repaired = call_llm(system_prompt, repair_prompt, agent_name="supervisor")
    if repaired:
        report, err2 = _parse_review_reply(repaired)
        if report is not None:
            logger.info("[SupervisorJsonParseFailed] repair re-ask succeeded.")
            return _apply_hard_review_guards(report, execution_log)
        # Keep the second-attempt raw reply too; it usually differs from the
        # first and helps diagnose whether the model simply cannot escape.
        _persist_raw_reply_for_debug(
            phase, repaired, reason=f"repair re-ask parse failed: {err2}"
        )
        logger.error(
            "[SupervisorJsonParseFailed] repair re-ask also failed: %s", err2
        )

    # ---- Tier 3: degrade-pass ----
    return _degrade_pass(
        phase,
        execution_log,
        reason=f"[SupervisorJsonParseFailed] {err}",
        raw_dump_path=dump_path,
    )


def _degrade_pass(
    phase: str,
    execution_log: str,
    *,
    reason: str,
    raw_dump_path: str | None,
) -> ReviewReport:
    """
    Fallback degrade-pass.

    ``detailed_report`` is explicitly marked as a placeholder rather than a
    real review, so downstream consumers (Memory / ``*_report.md``) can tell
    at a glance that the Supervisor did not produce a meaningful judgement
    this phase.
    """
    if _log_has_fatal_error(execution_log):
        return ReviewReport(
            score=max(1, SUPERVISOR_PASS_THRESHOLD - 1),
            passed=False,
            feedback_for_retry=(
                "Supervisor review was unavailable or unparsable, and the execution log "
                "contains fatal error signals such as Traceback/Error/Exception. "
                "Fix the runtime error and re-run this phase before continuing."
            ),
            summary_for_memory="",
            detailed_report="",
        )

    dump_line = (
        f"Raw LLM reply persisted at: `{raw_dump_path}`"
        if raw_dump_path
        else "(raw reply was not persisted)"
    )
    detailed = (
        f"# {phase} review report (degraded pass / placeholder)\n\n"
        f"> ⚠️ No valid Supervisor review was produced this phase. Reason: {reason}\n"
        f"> {dump_line}\n\n"
        "## Execution log\n"
        "```\n"
        f"{execution_log}\n"
        "```\n"
    )
    return ReviewReport(
        score=SUPERVISOR_PASS_THRESHOLD + 1,
        passed=True,
        feedback_for_retry="",
        summary_for_memory=f"{phase} review degraded-pass ({reason}).",
        detailed_report=detailed,
    )


def _apply_hard_review_guards(
    review: ReviewReport,
    execution_log: str,
) -> ReviewReport:
    """
    Enforce deterministic safety checks after the LLM review is parsed.

    The LLM can be lenient or internally inconsistent, but routing decisions
    must still obey the numeric threshold and obvious runtime failure signals.
    """
    if review.score < SUPERVISOR_PASS_THRESHOLD:
        review.passed = False
        if not review.feedback_for_retry:
            review.feedback_for_retry = (
                f"Quality score {review.score} is below the passing threshold "
                f"{SUPERVISOR_PASS_THRESHOLD}. Complete the missing requirements "
                "and re-run validation."
            )

    if _log_has_fatal_error(execution_log):
        review.passed = False
        review.score = min(review.score, max(1, SUPERVISOR_PASS_THRESHOLD - 1))
        if not review.feedback_for_retry:
            review.feedback_for_retry = (
                "Execution log contains fatal error signals such as "
                "Traceback/Error/Exception. Fix the runtime error and re-run "
                "this phase before continuing."
            )

    return review


def _log_has_fatal_error(execution_log: str) -> bool:
    """Detect obvious runtime failures that should never be supervisor-approved."""
    if not execution_log:
        return False

    log = execution_log.lower()
    fatal_markers = (
        "traceback (most recent call last)",
        "error:",
        "exception",
        "fatal",
        "segmentation fault",
        "module not found",
        "modulenotfounderror",
        "importerror",
        "keyerror",
        "valueerror",
        "typeerror",
        "indexerror",
        "zerodivisionerror",
        "filenotfounderror",
    )
    return any(marker in log for marker in fatal_markers)
