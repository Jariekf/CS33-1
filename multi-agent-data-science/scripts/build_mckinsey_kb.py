#!/usr/bin/env python3
"""One-shot ingest: McKinsey KB markdown under data/mckinsey_kb/raw -> Chroma ``mckinsey_kb``.

Reads ``manifest.lock.yaml`` (preferred) for per-file metadata (source_type, title, year).
Run from ``multi-agent-data-science/``:

    python scripts/build_mckinsey_kb.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.service import upload_knowledge_base

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logger = logging.getLogger(__name__)

MANIFEST_LOCK = PROJECT_ROOT / "data" / "mckinsey_kb" / "manifest.lock.yaml"


def _normalize_meta(entry: dict) -> dict[str, object]:
    meta: dict[str, object] = {
        "source_type": entry.get("source_type", "mckinsey"),
        "title": entry.get("title", "Untitled"),
    }
    if "year" in entry and entry["year"] is not None:
        meta["year"] = entry["year"]
    return meta


def _collect_paths_and_metadata() -> tuple[list[str], dict[str, dict[str, object]]]:
    if not MANIFEST_LOCK.is_file():
        raise FileNotFoundError(
            f"Missing {MANIFEST_LOCK}. Copy from CS33-automl or run prepare_mckinsey_md.py first."
        )
    raw = MANIFEST_LOCK.read_text(encoding="utf-8")
    entries = yaml.safe_load(raw)
    if not isinstance(entries, list):
        raise ValueError("manifest.lock.yaml must be a YAML list")

    file_paths: list[str] = []
    file_metadata: dict[str, dict[str, object]] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        lp = entry.get("local_path")
        if not lp:
            continue
        path = Path(str(lp))
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.is_file():
            logger.warning("Skip missing file: %s", path)
            continue
        meta = _normalize_meta(entry)
        abs_str = str(path)
        file_paths.append(abs_str)
        file_metadata[abs_str] = meta
        # Also register forward-slash variant for loader lookup
        alt = str(path.as_posix())
        if alt != abs_str:
            file_metadata[alt] = meta

    return file_paths, file_metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    os.chdir(PROJECT_ROOT)

    collection = os.getenv("MCKINSEY_COLLECTION", "mckinsey_kb")
    file_paths, file_metadata = _collect_paths_and_metadata()
    if not file_paths:
        raise SystemExit("No markdown files to ingest; check manifest.lock.yaml local_path entries.")

    logger.info("Ingesting %d file(s) into collection=%r", len(file_paths), collection)
    result = upload_knowledge_base(
        file_paths=file_paths,
        drop_old=True,
        collection_name=collection,
        file_metadata=file_metadata,
        markdown_aware=None,
    )
    logger.info("Done: %s", result)


if __name__ == "__main__":
    main()
