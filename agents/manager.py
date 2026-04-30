"""
Manager Agent (coordination and planning agent)
Responsibilities:
  1. Read the user goal plus optional hints / dataset description, then
     generate a user_requirement_summary with an LLM.
  2. Build a RAP (Retrieval-Augmented Planning) context through lightweight
     multi-source retrieval and fusion.
  3. Combine the summary, dataset description, RAP context, and data preview
     into a structured global execution plan.
  4. Initialize the global state and start the workflow.

Execution order inside manager_node:
  Step 1a - Data profile: read CSV metadata such as columns, dtypes, and sample rows without an LLM.
  Step 1b - Requirement understanding: generate user_requirement_summary and retrieval slots.
  Step 2 - RAP: retrieve and fuse lightweight multi-source context.
  Step 3 - Planning: generate global_plan and attach read-only metadata.
  Step 4 - Initialize workspace/runtime state and return.
"""
from __future__ import annotations

import os
import json
import re
import hashlib
import pandas as pd

from core.state import GraphState
from core.llm import call_llm
from config import WORKSPACE_DIR
from core.logger import get_agent_logger

from agents.manager_retrieval import (
    generate_arxiv_search_queries,
    retrieve_and_fuse_rap_context_with_meta,
    summarize_and_extract_slots_with_meta,
)

logger = get_agent_logger("manager")

_MANAGER_PLAN_ALLOWED_TOP_LEVEL_KEYS = {
    "data_cleaning_plan",
    "feature_engineering_plan",
    "model_selection_plan",
}


def manager_node(state: GraphState) -> dict:
    """
    LangGraph node function for the Manager Agent.
    Input: initial state with ``data_path``, ``user_goal``, and optional
    Manager-side context such as ``manager_cleaning_hints`` /
    ``dataset_description``.
    Output: updated ``global_plan`` plus runtime fields for the first worker
    phase.
    """

    user_goal = state["user_goal"]
    data_path = state["data_path"]
    cleaning_hints = (state["manager_cleaning_hints"] or "").strip()
    dataset_description = (state.get("dataset_description") or "").strip()

    logger.info(
        "Starting manager_node with user_goal: '%s', data_path: '%s', dataset_description: %s",
        user_goal,
        data_path,
        f"{len(dataset_description)} chars" if dataset_description else "none",
    )

    # ------------------------------------------------------------------
    # Step 1a - Data profile (non-LLM).
    # Read CSV columns, dtypes, shape, and sample rows as structured facts for later LLM steps.
    # ------------------------------------------------------------------
    df_preview = _get_data_preview(data_path)
    logger.info("Data preview generated.")

    # ------------------------------------------------------------------
    # Step 1b - Requirement summary and retrieval slots (LLM).
    # This step consumes optional hints / dataset description and produces the
    # narrative summary plus retrieval slots used downstream.
    # ------------------------------------------------------------------
    req_summary, slots, summary_eval_meta = summarize_and_extract_slots_with_meta(
        user_goal,
        df_preview,
        cleaning_hints,
        dataset_description,
    )
    logger.info("User requirement summary and retrieval slots ready.")

    # ------------------------------------------------------------------
    # Step 2 - RAP retrieval and fusion, separated from Step 1b for readability and testing.
    # ------------------------------------------------------------------
    rap_context, fusion_eval_meta = retrieve_and_fuse_rap_context_with_meta(req_summary, slots, user_goal)
    logger.info("RAP context prepared for planning.")

    # ------------------------------------------------------------------
    # Step 3 - Structured global_plan generation (LLM).
    # ------------------------------------------------------------------
    global_plan = _generate_global_plan(
        user_goal,
        df_preview,
        rap_context,
        user_requirement_summary=req_summary,
        dataset_description=dataset_description,
    )
    global_plan["_manager_summary_eval"] = summary_eval_meta
    global_plan["_manager_fusion_eval"] = fusion_eval_meta
    logger.info("Global structured plan generated successfully.")

    # ------------------------------------------------------------------
    # Step 4 - Initialize runtime state and return.
    # ------------------------------------------------------------------
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    return {
        "global_plan": global_plan,
        "current_phase": "data_cleaning",
        "current_data_path": data_path,
        "current_code": "",
        "current_execution_log": "",
        "current_error": "",
        "retry_count": 0,
        "current_review": None,
        "memory": [],
        "final_business_report": "",
    }


# ======================================================================
# Private helper functions.
# ======================================================================

def _get_data_preview(data_path: str) -> str:
    """
    Read a structured CSV profile (columns, dtypes, shape, and sample rows)
    and render it as plain text for later LLM steps.
    """
    try:
        df = pd.read_csv(data_path, nrows=5)
        info = (
            f"Columns ({len(df.columns)}): {list(df.columns)}\n"
            f"Shape: {df.shape}\n"
            f"Dtypes:\n{df.dtypes.to_string()}\n"
            f"Head:\n{df.head(3).to_string()}"
        )
        return info
    except Exception as e:
        return f"[Warning] Unable to preview data: {e}"


def _generate_search_queries(user_goal: str, df_preview: str) -> list[str]:
    """Compatibility helper for tests and external callers: generate arXiv search queries."""
    return generate_arxiv_search_queries(user_goal, df_preview)


def _generate_global_plan(
    user_goal: str,
    df_preview: str,
    rap_context: str,
    *,
    user_requirement_summary: str = "",
    dataset_description: str = "",
) -> dict:
    plan, _ = _generate_global_plan_with_meta(
        user_goal,
        df_preview,
        rap_context,
        user_requirement_summary=user_requirement_summary,
        dataset_description=dataset_description,
    )
    return plan


def _build_global_plan_prompts(
    user_goal: str,
    df_preview: str,
    rap_context: str,
    *,
    user_requirement_summary: str = "",
    dataset_description: str = "",
) -> tuple[str, str, str, str]:
    """Build prompts for structured execution plan generation (JSON).

    Relationship with the older `retrieved_docs: list[dict]` parameter:
    - Previously, this function joined several {title, summary} items into a literature context string.
    - Now, `retrieve_and_fuse_rap_context` produces `rap_context` first, and this function consumes that string.
    - `plan["_rap_context"]` stores the RAP text injected into the planning prompt for audit and reproduction.

    Planning sees `user_goal`, the summary block, optional
    `dataset_description`, `df_preview`, and `rap_context`; each serves a
    different role.

    `data_cleaning_plan.cleaning_contract` - Minimal general mode (keys are optional; default or `{}` indicates no validation):

    - `max_row_drop_ratio`: A floating-point number from 0 to 1, representing the maximum percentage of rows allowed to be deleted/dropped relative to the number of rows in the input CSV.

    Column retention is not enforced: cleaning may drop or rename columns per objectives.

    Differences from `rules`:

    - `rules`: Plans which cleaning steps (processes) to perform, for subsequent rule-based execution or suggestions.

    - `cleaning_contract`: Task-level safety boundaries (results), hard-validated by the Data Cleaning node after data is written to disk.
    """
    summary_block = user_requirement_summary.strip() or user_goal
    docs_context = rap_context.strip() if rap_context.strip() else "No retrieval results"
    desc_text = dataset_description.strip() if dataset_description else ""
    dataset_description_block = (
        desc_text
        if desc_text
        else "(not provided; infer cautiously from column names and sample rows)"
    )

    sys_prompt = """You are a Data Science Engineering Director generating an execution plan."""
    user_prompt = f"""
User Goal: {user_goal}
User requirement summary (for alignment):
{summary_block}

Dataset Description (authoritative; use this over guesses from column names):
{dataset_description_block}

Data Preview:
{df_preview}

Literature / external knowledge (RAP):
{docs_context}

Please formulate a high-level Data Science global execution plan tailored to the objective. 
You must output ONLY a valid JSON object with EXACTLY the following structure (no markdown fences around it if possible, or strictly enclosed in ```json...```):
{{
    "data_cleaning_plan": {{
        "objectives": ["obj1", "obj2"],
        "constraints": "string constraint",
        "rules": [],
        "cleaning_contract": {{}}
    }},
    "feature_engineering_plan": {{
        "objectives": ["obj1"],
        "constraints": "string constraint"
    }},
    "model_selection_plan": {{
        "objectives": ["model1", "model2"],
        "evaluation_metric": "e.g., f1, accuracy"
    }}
}}

Notes for data_cleaning_plan:
- "rules": procedural steps to perform (what to run). Optional list of objects like {{"rule": "drop_duplicates", "params": {{}}}}. Use [] if none. Not validated as bounds.
- "cleaning_contract": post-execution guardrails. Prefer {{}} or only "max_row_drop_ratio". Do not list "protected_columns" or imply all columns must survive cleaning - dropping columns is allowed.
    * "max_row_drop_ratio" (number 0..1, optional): only enforced key; max allowed fraction of input rows removed, i.e. (rows_in - rows_out) / rows_in.

Notes for feature_engineering_plan:
- Ground all objectives in columns that are visible in the Data Preview.
- Do not mention invented fields such as `product_id`, `country`, `period`, `date`, or forecast targets unless the preview clearly supports them.
- If time columns or grouping keys are unclear, prefer simple safe feature engineering over grouped time-series designs.
"""
    return sys_prompt, user_prompt, summary_block, docs_context


def _generate_global_plan_with_meta(
    user_goal: str,
    df_preview: str,
    rap_context: str,
    *,
    user_requirement_summary: str = "",
    dataset_description: str = "",
) -> tuple[dict, dict]:
    sys_prompt, user_prompt, summary_block, docs_context = _build_global_plan_prompts(
        user_goal,
        df_preview,
        rap_context,
        user_requirement_summary=user_requirement_summary,
        dataset_description=dataset_description,
    )
    reply = call_llm(sys_prompt, user_prompt, agent_name="manager_plan")
    parsed_obj: dict | None = None
    json_parse_ok = False

    if reply:
        try:
            match = re.search(r'\{.*\}', reply, re.DOTALL)
            if match:
                plan = json.loads(match.group(0))
                if isinstance(plan, dict):
                    parsed_obj = plan
                    json_parse_ok = True
                    if "data_cleaning_plan" in plan:
                        _normalize_data_cleaning_plan(plan)
                        plan["_rap_context"] = docs_context
                        plan["_user_requirement_summary"] = summary_block
                        meta = _build_manager_plan_eval_meta(
                            system_prompt=sys_prompt,
                            user_prompt=user_prompt,
                            raw_reply=reply,
                            parsed_obj=parsed_obj,
                            json_parse_ok=json_parse_ok,
                            used_fallback=False,
                        )
                        plan["_manager_plan_eval"] = meta
                        return plan, meta
        except Exception:
            pass

    # Fallback
    fallback_plan = {
        "data_cleaning_plan": {
            "objectives": ["Check and handle missing values", "Remove duplicate rows", "Correct data types"],
            "constraints": "Do not remove more than 5% of total rows",
            "rules": [],
            "cleaning_contract": {},
        },
        "feature_engineering_plan": {
            "objectives": ["Create key derived features based on RAP context", "Encode categorical features", "Scale features"],
            "constraints": "Keep feature dimensionality within a reasonable range",
        },
        "model_selection_plan": {
            "objectives": ["Test XGBoost / LightGBM / RandomForest", "Use cross-validation"],
            "evaluation_metric": "f1",
        },
        "_rap_context": docs_context,
        "_user_requirement_summary": summary_block,
    }
    _normalize_data_cleaning_plan(fallback_plan)
    meta = _build_manager_plan_eval_meta(
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        raw_reply=reply or "",
        parsed_obj=parsed_obj,
        json_parse_ok=json_parse_ok,
        used_fallback=True,
    )
    fallback_plan["_manager_plan_eval"] = meta
    return fallback_plan, meta


def _build_manager_plan_eval_meta(
    *,
    system_prompt: str,
    user_prompt: str,
    raw_reply: str,
    parsed_obj: dict | None,
    json_parse_ok: bool,
    used_fallback: bool,
) -> dict:
    schema = _evaluate_plan_schema(parsed_obj)
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]
    return {
        "prompt_hash": prompt_hash,
        "system_prompt": system_prompt,
        "user_prompt_preview": user_prompt[:1200],
        "raw_reply_preview": raw_reply[:1500],
        "json_parse_ok": json_parse_ok,
        "schema_ok": schema["schema_ok"],
        "used_fallback": used_fallback,
    }


def _evaluate_plan_schema(plan: dict | None) -> dict:
    if not isinstance(plan, dict):
        return {
            "schema_ok": False,
        }

    keys = set(plan.keys())
    missing_required_keys = sorted(_MANAGER_PLAN_ALLOWED_TOP_LEVEL_KEYS - keys)

    data_cleaning_plan = plan.get("data_cleaning_plan")
    feature_engineering_plan = plan.get("feature_engineering_plan")
    model_selection_plan = plan.get("model_selection_plan")

    rules_type_ok = isinstance(
        data_cleaning_plan.get("rules"),
        list,
    ) if isinstance(data_cleaning_plan, dict) else False
    cleaning_contract_type_ok = isinstance(
        data_cleaning_plan.get("cleaning_contract"),
        dict,
    ) if isinstance(data_cleaning_plan, dict) else False
    cleaning_contract_allowed_keys_only = False
    if cleaning_contract_type_ok:
        allowed_contract_keys = {"max_row_drop_ratio"}
        cleaning_contract_allowed_keys_only = set(
            data_cleaning_plan.get("cleaning_contract", {}).keys()
        ).issubset(allowed_contract_keys)

    stage_boundary_ok = _stage_boundary_ok(
        data_cleaning_plan=data_cleaning_plan,
        feature_engineering_plan=feature_engineering_plan,
        model_selection_plan=model_selection_plan,
    )

    schema_ok = (
        not missing_required_keys
        and isinstance(data_cleaning_plan, dict)
        and isinstance(feature_engineering_plan, dict)
        and isinstance(model_selection_plan, dict)
        and rules_type_ok
        and cleaning_contract_type_ok
    )

    return {"schema_ok": schema_ok}


def _stage_boundary_ok(
    *,
    data_cleaning_plan: dict | None,
    feature_engineering_plan: dict | None,
    model_selection_plan: dict | None,
) -> bool:
    if not all(
        isinstance(section, dict)
        for section in (data_cleaning_plan, feature_engineering_plan, model_selection_plan)
    ):
        return False

    dc_text = json.dumps(data_cleaning_plan, ensure_ascii=False).lower()
    fe_text = json.dumps(feature_engineering_plan, ensure_ascii=False).lower()
    ms_text = json.dumps(model_selection_plan, ensure_ascii=False).lower()

    model_keywords = (
        "cross-validation",
        "cross validation",
        "hyperparameter",
        "randomforest",
        "xgboost",
        "lightgbm",
        "logistic regression",
    )
    cleaning_keywords = (
        "duplicate",
        "missing value",
        "dtype",
        "data type",
        "dropna",
    )

    dc_has_model_leakage = any(keyword in dc_text for keyword in model_keywords)
    fe_has_model_leakage = any(keyword in fe_text for keyword in model_keywords)
    ms_has_cleaning_leakage = any(keyword in ms_text for keyword in cleaning_keywords)

    return not (dc_has_model_leakage or fe_has_model_leakage or ms_has_cleaning_leakage)


def _normalize_data_cleaning_plan(plan: dict) -> None:
    """Ensure data_cleaning_plan has rules + cleaning_contract with safe defaults."""
    dcp = plan.get("data_cleaning_plan")
    if not isinstance(dcp, dict):
        plan["data_cleaning_plan"] = {
            "objectives": [],
            "constraints": "",
            "rules": [],
            "cleaning_contract": {},
        }
        return
    if not isinstance(dcp.get("rules"), list):
        dcp["rules"] = []
    if not isinstance(dcp.get("cleaning_contract"), dict):
        dcp["cleaning_contract"] = {}
    # Drop legacy / LLM-hallucinated key: we do not enforce "must keep these columns".
    dcp["cleaning_contract"].pop("protected_columns", None)
