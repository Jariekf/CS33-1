"""
Upload the prepared McKinsey-style markdown corpus into the Chroma collection
``mckinsey_kb`` (independent of the generic ``rag_collection``).

Inputs:
  - data/mckinsey_kb/manifest.lock.yaml  (produced by prepare_mckinsey_md.py)

Behaviour:
  - Each chunk is tagged with metadata: source_type, title, year, section_path
  - Markdown structure is preserved by header-aware splitting (H1-H4)
  - The collection is rebuilt from scratch by default (drop_old=True)

Usage:
    python scripts/upload_mckinsey_kb.py
    python scripts/upload_mckinsey_kb.py --append          # do not drop existing
    python scripts/upload_mckinsey_kb.py --collection foo  # override target collection
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from pprint import pprint

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.service import upload_knowledge_base

KB_DIR = PROJECT_ROOT / "data" / "mckinsey_kb"
LOCK_PATH = KB_DIR / "manifest.lock.yaml"
DEFAULT_COLLECTION = "mckinsey_kb"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("upload_mckinsey_kb")


def _build_payload(entries: list[dict]) -> tuple[list[str], dict[str, dict]]:
    file_paths: list[str] = []
    file_metadata: dict[str, dict] = {}
    for entry in entries:
        local_path = entry.get("local_path")
        if not local_path:
            logger.warning("Skipping entry without local_path: %s", entry)
            continue
        abs_path = (PROJECT_ROOT / local_path) if not Path(local_path).is_absolute() else Path(local_path)
        if not abs_path.exists():
            logger.warning("Skipping missing file: %s", abs_path)
            continue
        file_paths.append(str(abs_path))
        meta: dict[str, object] = {"source_type": entry["source_type"]}
        if entry.get("title"):
            meta["title"] = entry["title"]
        if entry.get("year") is not None:
            meta["year"] = entry["year"]
        file_metadata[str(abs_path)] = meta
    return file_paths, file_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append", action="store_true", help="append instead of rebuilding")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="target Chroma collection")
    args = parser.parse_args()

    if not LOCK_PATH.exists():
        logger.error("Lock file missing: %s", LOCK_PATH)
        logger.error("Run scripts/prepare_mckinsey_md.py first.")
        return 1

    entries = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8")) or []
    if not entries:
        logger.warning("Lock file is empty; nothing to upload.")
        return 0

    file_paths, file_metadata = _build_payload(entries)
    if not file_paths:
        logger.error("No usable files to ingest.")
        return 1

    counts = Counter(meta["source_type"] for meta in file_metadata.values())
    logger.info(
        "Ingesting %d file(s) into collection '%s' (drop_old=%s). By source_type: %s",
        len(file_paths),
        args.collection,
        not args.append,
        dict(counts),
    )

    result = upload_knowledge_base(
        file_paths=file_paths,
        drop_old=not args.append,
        collection_name=args.collection,
        file_metadata=file_metadata,
    )

    print("\n[McKinsey KB Upload Result]")
    pprint({**result, "by_source_type": dict(counts)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
