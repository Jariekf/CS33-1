"""
Manager RAP 轻量级检索层（对标 AutoML retrieve_knowledge 的形态，无 PDF/BM25）。

- 单次 LLM：生成 user_requirement_summary + 结构化槽位（任务/领域/area），可选并入 cleaning 队友提供的 hints。
- 多源仅收集摘要级文本：arXiv abstract、DuckDuckGo 网页摘要、本地 PapersWithCode JSON、Kaggle kernel 列表元数据。
- 单次 LLM：融合各源为一段 plan 可用的知识文本。

依赖：arxiv 已有；可选 `duckduckgo-search`、`kaggle` 及 Kaggle API 凭证。
PWC：仅当 `config.PWC_DATA_DIR` 下存在 `datasets.json` 与 `evaluation-tables.json` 才参与检索；未充实该库前目录可空（或仅 .gitkeep），逻辑上等同关闭该源。

--------------------------------------------------------------------
数据流（与 manager.py 的衔接）
  user_goal, df_preview, cleaning_hints
      → summarize_and_extract_slots → (user_requirement_summary, slots)
      → retrieve_and_fuse_rap_context(summary, slots, user_goal) → rap_context: str
      → manager._generate_global_plan(..., rap_context, user_requirement_summary=...)
      → global_plan（业务 JSON + 元数据 _rap_context / _user_requirement_summary）

本模块不读、不写LangGraph的GraphState；仅由 manager_node 调用，返回值留在manager局部再写入global_plan。
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from config import ARXIV_MAX_RESULTS, PWC_DATA_DIR, RAP_FUSE_MAX_CHARS, RAP_KAGGLE_ENABLED, RAP_PWC_ENABLED, RAP_WEB_ENABLED, WEB_MAX_RESULTS
from core.llm import call_llm
from core.logger import get_agent_logger
from tools.arxiv_tool import search_arxiv

logger = get_agent_logger("manager")

"""
摘要 vs 槽位:
- user_requirement_summary：一段自然语言，给人和后续 LLM 读（Web 检索 query 生成、_fuse_sources、规划 prompt 里的「需求摘要」块）。
- slots：短结构化短语（downstream_task / application_domain / area），给「关键词型」检索用（arXiv 查询串、PWC 文本过滤、Kaggle kernel search）。
二者来自同一次 LLM JSON，分工是「叙述对齐」与「检索接口」分离；df_preview 只作为该 LLM 的输入上下文，不是槽位本身。
"""
def generate_arxiv_search_queries(user_goal: str, df_preview: str) -> list[str]:
    """根据目标与数据预览生成 arXiv 检索词（与 summarize 共用槽位，避免重复 LLM 时可改用 build_rap_context 内部结果）。"""
    _, slots = summarize_and_extract_slots(user_goal, df_preview, cleaning_hints="")
    return _queries_from_slots(slots, user_goal)


def summarize_and_extract_slots(
    user_goal: str,
    df_preview: str,
    cleaning_hints: str,
) -> tuple[str, dict[str, str]]:
    """
    生成 user_requirement_summary，并抽取 downstream_task / application_domain / area。
    cleaning_hints：队友或用户预先给出的 data_cleaning rules/contract 等文字，写入摘要供检索与规划对齐。
    """
    hints_block = ""
    if cleaning_hints and cleaning_hints.strip():
        hints_block = f"""
Additional cleaning / contract hints from the team (must be reflected in the summary and kept consistent for retrieval):
{cleaning_hints.strip()}
"""
    sys_prompt = "You are a senior data science project manager. Output only valid JSON, no markdown."
    user_prompt = f"""
User goal:
{user_goal}

Data preview:
{df_preview}
{hints_block}

Return a single JSON object with exactly these string fields:
- "user_requirement_summary": one concise paragraph (English or Chinese matching the user goal) describing the ML task, data, and objectives; if cleaning hints were given, weave them in as constraints.
- "downstream_task": short phrase for search (e.g. churn prediction, demand forecasting).
- "application_domain": short phrase (e.g. telecom, e-commerce).
- "area": ML angle for search (e.g. tabular classification, gradient boosting).

Example: {{"user_requirement_summary":"...","downstream_task":"churn prediction","application_domain":"telecom","area":"tabular supervised learning"}}
"""
    reply = call_llm(sys_prompt, user_prompt, agent_name="manager")
    default_slots = {
        "downstream_task": user_goal[:120],
        "application_domain": "general",
        "area": "machine learning",
    }
    if not reply:
        summary = user_goal if not cleaning_hints else f"{user_goal}\nConstraints: {cleaning_hints}"
        return summary.strip(), default_slots

    try:
        match = re.search(r"\{[^{}]*\}", reply, re.DOTALL)
        if not match:
            match = re.search(r"\{.*\}", reply, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            summary = str(obj.get("user_requirement_summary", user_goal)).strip()
            slots = {
                "downstream_task": str(obj.get("downstream_task", default_slots["downstream_task"])).strip(),
                "application_domain": str(obj.get("application_domain", default_slots["application_domain"])).strip(),
                "area": str(obj.get("area", default_slots["area"])).strip(),
            }
            return summary or user_goal, slots
    except (json.JSONDecodeError, TypeError):
        pass

    summary = user_goal if not cleaning_hints else f"{user_goal}\nConstraints: {cleaning_hints}"
    return summary.strip(), default_slots


def _queries_from_slots(slots: dict[str, str], user_goal: str) -> list[str]:
    t = slots.get("downstream_task", "").strip()
    d = slots.get("application_domain", "").strip()
    a = slots.get("area", "").strip()
    q1 = f"{t} {d}".strip()
    q2 = f"{a} {t}".strip()
    out: list[str] = []
    if q1:
        out.append(q1)
    if q2 and q2 not in out:
        out.append(q2)
    return out[:2] if out else [f"{user_goal} machine learning"]


def _format_arxiv_context(queries: list[str], max_per_query: int | None) -> str:
    max_per_query = max_per_query or ARXIV_MAX_RESULTS
    lines: list[str] = []
    seen: set[str] = set()
    for q in queries:
        for doc in search_arxiv(q, max_results=max_per_query):
            title = doc.get("title", "N/A")
            key = title
            if key in seen:
                continue
            seen.add(key)
            summ = (doc.get("summary") or "")[:600].replace("\n", " ")
            lines.append(f"- {title}: {summ}")
    return "\n".join(lines) if lines else ""


def _retrieve_web_light(user_requirement_summary: str) -> str:
    if not RAP_WEB_ENABLED:
        return ""
    query_reply = call_llm(
        "You generate short web search queries for ML engineering. Reply with a single line, at most 12 words, no quotes.",
        f"Problem summary:\n{user_requirement_summary}\n\nGive one English search query.",
        agent_name="manager",
    )
    search_query = (query_reply or user_requirement_summary)[:200].strip()
    try:
        from duckduckgo_search import DDGS
        """- Web 检索选用 DuckDuckGo（duckduckgo-search）：通常无需 Google 官方 Search API Key，实现轻量；若需可替换为其它 search 后端。"""

        rows: list[str] = []
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, max_results=WEB_MAX_RESULTS):
                title = r.get("title") or ""
                body = (r.get("body") or "")[:400]
                href = r.get("href") or ""
                if title or body:
                    rows.append(f"- {title} | {body} | {href}")
        return "\n".join(rows)
    except ImportError:
        logger.warning("duckduckgo-search 未安装，跳过网页检索。pip install duckduckgo-search")
        return ""
    except Exception as e:
        logger.warning("网页检索失败: %s", e)
        return ""


def _retrieve_pwc_light(slots: dict[str, str], user_requirement_summary: str, top_k: int = 8) -> str:
    """本地 PWC 数据可选：目录未同步或缺 JSON 时返回空串，不视为错误。"""
    if not RAP_PWC_ENABLED:
        return ""
    base = Path(PWC_DATA_DIR)
    ds_path = base / "datasets.json"
    tb_path = base / "evaluation-tables.json"
    if not ds_path.is_file() or not tb_path.is_file():
        logger.info("PapersWithCode 本地数据不存在，跳过（路径: %s）", base)
        return ""

    user_task = slots.get("downstream_task", "").lower().replace("-", " ").replace("_", " ")
    user_area = slots.get("area", "").lower().replace("-", " ").replace("_", " ")

    try:
        datasets = json.loads(ds_path.read_text(encoding="utf-8"))
        datasets = [x for x in datasets if isinstance(x, dict) and len(x.get("data_loaders") or []) > 0]
        loaders: list[str] = []
        for dataset in datasets:
            desc = (dataset.get("description") or "").lower()
            tasks = [str(t.get("task", "")).lower() for t in (dataset.get("tasks") or []) if isinstance(t, dict)]
            if user_task and (user_task in desc or user_task in tasks or user_area in desc):
                name = dataset.get("name", "")
                loaders.append(
                    f"DATASET: {name}\nDESC: {dataset.get('description', '')[:500]}\nTASKS: {tasks[:5]}\nLOADERS: {dataset.get('data_loaders', [])[:3]}"
                )

        tables = json.loads(tb_path.read_text(encoding="utf-8"))
        tables = [t for t in tables if isinstance(t, dict) and len(t.get("datasets") or []) > 0]
        bench: list[str] = []
        for table in tables:
            desc = (table.get("description") or "").lower()
            task = str(table.get("task", "")).lower()
            cats = [str(c).lower() for c in (table.get("categories") or [])]
            if user_task and (
                user_task in desc
                or user_task == task
                or user_area in desc
                or any(user_area in c for c in cats)
            ):
                bench.append(
                    f"TASK: {table.get('task')}\nDESC: {table.get('description', '')[:400]}\nDATASETS: {str(table.get('datasets'))[:800]}"
                )

        if not loaders and not bench:
            return ""

        if len(loaders) > top_k:
            loaders = random.sample(loaders, top_k)
        if len(bench) > top_k:
            bench = random.sample(bench, top_k)

        blob = "\n---\n".join(loaders + bench)
        return blob[: RAP_FUSE_MAX_CHARS]
    except Exception as e:
        logger.warning("PWC 轻量检索失败: %s", e)
        return ""


def _retrieve_kaggle_light(slots: dict[str, str], top_k: int = 6) -> str:
    if not RAP_KAGGLE_ENABLED:
        return ""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.warning("kaggle 包未安装，跳过 Kaggle 检索。pip install kaggle")
        return ""

    task = slots.get("downstream_task", "").replace("-", " ").replace("_", " ").strip()
    domain = slots.get("application_domain", "").replace("-", " ").replace("_", " ").strip()
    search = f"{task} {domain}".strip() or "machine learning"

    try:
        api = KaggleApi()
        api.authenticate()
        kernels = api.kernels_list(search=search, sort_by="relevance", page=1, page_size=top_k)
        lines: list[str] = []
        for k in kernels:
            title = getattr(k, "title", None) or str(k)
            ref = getattr(k, "ref", "")
            lines.append(f"- {title} ({ref})")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Kaggle 检索跳过（需 ~/.kaggle/kaggle.json 等配置）: %s", e)
        return ""


def _fuse_sources(
    user_requirement_summary: str,
    parts: dict[str, str],
) -> str:
    """将各源原始片段融合为一段规划可用的知识（单次 LLM）。"""
    non_empty = {k: v for k, v in parts.items() if v and v.strip()}
    if not non_empty:
        return ""

    if len(non_empty) == 1:
        only = next(iter(non_empty.values()))
        if len(only) <= 2500:
            return only.strip()

    blocks = []
    for name, text in non_empty.items():
        clipped = text[: RAP_FUSE_MAX_CHARS]
        blocks.append(f"### {name}\n{clipped}")

    bundle = "\n\n".join(blocks)
    sys_prompt = (
        "You are an ML/AI research consultant. Merge the following noisy snippets into one coherent list of "
        "actionable suggestions for downstream planning (data cleaning, features, modeling). "
        "No code. Prefer concrete method names and task-relevant practices. Use the same language as the user summary when possible."
    )
    user_prompt = f"""
User requirement summary:
{user_requirement_summary}

Sources:
{bundle}

Produce a bullet list of key insights (5–12 bullets) the planning agent can use.
"""
    fused = call_llm(sys_prompt, user_prompt, agent_name="manager")
    if fused and fused.strip():
        return fused.strip()
    return bundle[:6000]


def retrieve_and_fuse_rap_context(
    user_requirement_summary: str,
    slots: dict[str, str],
    user_goal: str,
) -> str:
    """
    Step 2（RAP）：基于 Step 1b 产出的摘要与槽位做多源检索并融合为一段文本。
    与「需求理解」解耦，便于单独测试或替换检索策略。
    """
    queries = _queries_from_slots(slots, user_goal)

    arxiv_raw = _format_arxiv_context(queries, ARXIV_MAX_RESULTS)
    web_raw = _retrieve_web_light(user_requirement_summary)
    pwc_raw = _retrieve_pwc_light(slots, user_requirement_summary)
    kaggle_raw = _retrieve_kaggle_light(slots)

    parts = {
        "arXiv (abstracts)": arxiv_raw,
        "Web search snippets": web_raw,
        "PapersWithCode (local)": pwc_raw,
        "Kaggle kernels (titles)": kaggle_raw,
    }
    rap_context = _fuse_sources(user_requirement_summary, parts)
    if not rap_context.strip():
        rap_context = arxiv_raw or "无外部检索结果；仅基于用户目标与数据预览进行规划。"

    return rap_context


def build_rap_context(
    user_goal: str,
    df_preview: str,
    *,
    cleaning_hints: str = "",
) -> tuple[str, str]:
    """
    便捷封装：顺序执行 1b（摘要+槽位）与 Phase 2（检索+融合）。

    Returns:
        user_requirement_summary: 供日志、后续节点或再检索复用。
        rap_context: 融合后的 RAP 文本，可直接注入 _generate_global_plan。
    """
    summary, slots = summarize_and_extract_slots(user_goal, df_preview, cleaning_hints)
    rap_context = retrieve_and_fuse_rap_context(summary, slots, user_goal)
    return summary, rap_context
