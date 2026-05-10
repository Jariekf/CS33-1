"""
Lightweight cleaning for markdown produced by Firecrawl / Jina Reader / pymupdf4llm.

Goal: strip elements that pollute embeddings (image refs, navigation TOC,
subscribe / copyright CTAs, residual HTML) while preserving headings, lists
and prose so the markdown header splitter still works downstream.

Public entry point: ``clean_markdown(text)``.
"""
from __future__ import annotations

import re

# Regexes used by clean_markdown. Compiled at module import for reuse.
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_HTML_TAG_RE = re.compile(r"</?(?:div|span|sub|sup|figure|figcaption|svg|nav|button|aside|iframe)[^>]*>", re.IGNORECASE)
_INLINE_ANCHOR_TOC_RE = re.compile(r"^\s*[-*]\s+\[[^\]]+\]\(#[^)]+\)\s*$")
_HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Lines whose content matches any of these (case-insensitive) are dropped.
# Keep this list short and conservative - we only strip obvious boilerplate.
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*subscribe\b.*$", re.IGNORECASE),
    re.compile(r"^\s*sign\s*up\b.*$", re.IGNORECASE),
    re.compile(r"^\s*sign\s*in\b.*$", re.IGNORECASE),
    re.compile(r"^\s*©\s*(?:\d{4}\s*)?(?:mckinsey|company|all rights reserved).*$", re.IGNORECASE),
    re.compile(r"^\s*copyright\s+©.*$", re.IGNORECASE),
    re.compile(r"^\s*all rights reserved.*$", re.IGNORECASE),
    re.compile(r"^\s*share\s+(?:on|this|article).*$", re.IGNORECASE),
    re.compile(r"^\s*(?:download|print)\s+(?:pdf|article|report).*$", re.IGNORECASE),
    re.compile(r"^\s*follow\s+(?:us|mckinsey)\b.*$", re.IGNORECASE),
    re.compile(r"^\s*about\s+the\s+author\(s\)\s*$", re.IGNORECASE),
]


def _is_boilerplate(line: str) -> bool:
    return any(p.match(line) for p in _BOILERPLATE_PATTERNS)


def clean_markdown(text: str) -> str:
    """Return a cleaned copy of ``text`` suitable for header-aware chunking.

    The transformation is intentionally conservative: when in doubt, keep the
    line. Heading lines (``#``, ``##``, ...) are NEVER dropped because the
    downstream ``MarkdownHeaderTextSplitter`` relies on them for section
    boundaries.
    """
    if not text:
        return ""

    text = _IMAGE_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if line.lstrip().startswith("#"):
            cleaned_lines.append(line)
            continue

        if _INLINE_ANCHOR_TOC_RE.match(line):
            continue
        if _HORIZONTAL_RULE_RE.match(line):
            continue
        if _is_boilerplate(line):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = _MULTI_BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip() + "\n"
