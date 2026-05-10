"""Minimal OpenAI-compatible LLM client.

The workflow treats LLM output as optional for business narrative generation.
If the API key or SDK is unavailable, callers can fall back to deterministic
report generation without breaking the data science pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs into environment variables if unset."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call an OpenAI-compatible chat model and return text content.

    Required environment:
        OPENAI_API_KEY

    Optional environment:
        OPENAI_MODEL, default ``gpt-4o-mini``
        OPENAI_BASE_URL, for OpenAI-compatible providers
        LLM_TEMPERATURE, default ``0.2``
    """

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed") from exc

    kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned empty content")
    return content.strip()
