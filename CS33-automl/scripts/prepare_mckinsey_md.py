"""
Resolve every entry in ``data/mckinsey_kb/manifest.yaml`` into a local
markdown file and emit ``manifest.lock.yaml`` for the upload step.

Each manifest entry exposes exactly one of three source fields:

  - ``url``        -> fetch via Jina Reader (https://r.jina.ai/<url>)
  - ``local_pdf``  -> convert via pymupdf4llm
  - ``local_path`` -> already a .md / .markdown / .txt, pass-through

Outputs:

  - ``data/mckinsey_kb/raw/<slug>.md``     (only for url + local_pdf)
  - ``data/mckinsey_kb/manifest.lock.yaml`` (entries + resolved local_path)
  - ``data/mckinsey_kb/prepare_errors.log`` (failed entries; non-fatal)

Usage:

    python scripts/prepare_mckinsey_md.py            # incremental
    python scripts/prepare_mckinsey_md.py --force    # rebuild everything
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KB_DIR = PROJECT_ROOT / "data" / "mckinsey_kb"
MANIFEST_PATH = KB_DIR / "manifest.yaml"
LOCK_PATH = KB_DIR / "manifest.lock.yaml"
RAW_DIR = KB_DIR / "raw"
ERROR_LOG = KB_DIR / "prepare_errors.log"

JINA_BASE = "https://r.jina.ai"
JINA_TIMEOUT = 60
ALLOWED_SOURCE_TYPES = {"mckinsey", "equity_research", "textbook"}
ALLOWED_PASS_THROUGH_SUFFIXES = {".md", ".markdown", ".txt"}

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("prepare_mckinsey_md")


def _slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", (text or "").strip()).strip("-")
    return (text[:max_len] or "untitled").lower()


def _stable_slug(source_key: str, title: str | None) -> str:
    digest = hashlib.md5(source_key.encode("utf-8")).hexdigest()[:8]
    return f"{digest}_{_slugify(title or source_key)}"


def _fetch_via_jina(url: str) -> str:
    """Fetch a URL through Jina Reader and return markdown."""
    target = f"{JINA_BASE}/{url}"
    headers = {
        "Accept": "text/markdown",
        "X-Return-Format": "markdown",
    }
    api_key = os.getenv("JINA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(target, headers=headers, timeout=JINA_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _convert_local_pdf(pdf_path: Path) -> str:
    """Convert a local PDF file to markdown via pymupdf4llm."""
    try:
        import pymupdf4llm  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pymupdf4llm is required for local_pdf entries. Install with: "
            "pip install pymupdf4llm"
        ) from e
    return pymupdf4llm.to_markdown(str(pdf_path))


def _validate_entry(entry: dict, idx: int) -> tuple[str, str]:
    source_keys = [k for k in ("url", "local_pdf", "local_path") if entry.get(k)]
    if len(source_keys) == 0:
        raise ValueError(f"entry #{idx}: must provide one of url / local_pdf / local_path")
    if len(source_keys) > 1:
        logger.warning(
            "entry #%d: multiple source fields %s; using '%s' (priority url > local_pdf > local_path)",
            idx,
            source_keys,
            source_keys[0],
        )
    source_kind = source_keys[0]
    source_value = str(entry[source_kind])

    source_type = entry.get("source_type")
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise ValueError(
            f"entry #{idx}: source_type='{source_type}' is not one of {sorted(ALLOWED_SOURCE_TYPES)}"
        )
    return source_kind, source_value


def _process_entry(
    entry: dict,
    idx: int,
    *,
    force: bool,
) -> dict:
    source_kind, source_value = _validate_entry(entry, idx)
    title = entry.get("title")

    if source_kind == "local_path":
        path = Path(source_value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"entry #{idx}: local_path not found: {path}")
        if path.suffix.lower() not in ALLOWED_PASS_THROUGH_SUFFIXES:
            raise ValueError(
                f"entry #{idx}: local_path must be one of {sorted(ALLOWED_PASS_THROUGH_SUFFIXES)}, got {path.suffix}"
            )
        local_path = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
        logger.info("entry #%d: pass-through local_path -> %s", idx, local_path)
        return {**entry, "local_path": local_path, "source_kind": source_kind}

    slug = _stable_slug(source_value, title)
    out_path = RAW_DIR / f"{slug}.md"

    if out_path.exists() and not force:
        logger.info("entry #%d: cached %s (skip; pass --force to rebuild)", idx, out_path.name)
        return {**entry, "local_path": str(out_path.relative_to(PROJECT_ROOT)), "source_kind": source_kind}

    if source_kind == "url":
        logger.info("entry #%d: fetching url via Jina Reader: %s", idx, source_value)
        markdown = _fetch_via_jina(source_value)
    elif source_kind == "local_pdf":
        pdf_path = Path(source_value)
        if not pdf_path.is_absolute():
            pdf_path = PROJECT_ROOT / pdf_path
        if not pdf_path.exists():
            raise FileNotFoundError(f"entry #{idx}: local_pdf not found: {pdf_path}")
        logger.info("entry #%d: converting local PDF via pymupdf4llm: %s", idx, pdf_path)
        markdown = _convert_local_pdf(pdf_path)
    else:  # pragma: no cover - guarded by _validate_entry
        raise AssertionError(f"unhandled source_kind: {source_kind}")

    if not markdown.strip():
        raise ValueError(f"entry #{idx}: produced empty markdown")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    logger.info("entry #%d: wrote %s (%d chars)", idx, out_path.name, len(markdown))
    return {**entry, "local_path": str(out_path.relative_to(PROJECT_ROOT)), "source_kind": source_kind}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild already-cached entries")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        logger.error("Manifest not found: %s", MANIFEST_PATH)
        return 1

    raw = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        logger.error("Manifest must be a YAML list, got: %s", type(raw).__name__)
        return 1
    entries = [e for e in raw if isinstance(e, dict)]
    if not entries:
        logger.warning("Manifest is empty; nothing to do.")
        return 0

    resolved: list[dict] = []
    errors: list[str] = []
    for idx, entry in enumerate(entries, start=1):
        try:
            resolved.append(_process_entry(entry, idx, force=args.force))
        except Exception as e:  # noqa: BLE001 - we want to keep going
            msg = f"entry #{idx} failed: {e}"
            logger.error(msg)
            errors.append(msg)

    LOCK_PATH.write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Wrote lock file: %s (%d entries)", LOCK_PATH, len(resolved))

    if errors:
        ERROR_LOG.write_text("\n".join(errors) + "\n", encoding="utf-8")
        logger.warning("Wrote %d error(s) to %s", len(errors), ERROR_LOG)
    elif ERROR_LOG.exists():
        ERROR_LOG.unlink()

    by_kind: dict[str, int] = {}
    for e in resolved:
        by_kind[e["source_kind"]] = by_kind.get(e["source_kind"], 0) + 1
    logger.info("Summary by source_kind: %s", by_kind)
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
