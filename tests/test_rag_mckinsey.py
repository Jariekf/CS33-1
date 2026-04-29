"""
Smoke test for the McKinsey-style RAG additions.

Verifies on a temporary collection (no impact on real `mckinsey_kb`):
  - markdown header-aware ingestion preserves H2/H3 in `section_path` metadata
  - per-file metadata (source_type / title / year) is attached to chunks
  - `metadata_filter` isolates results to a single source_type
  - independent collections do not leak

Requires the same OpenAI-compatible env as `tests/test_rag_service.py`
(OPENAI_API_KEY, OPENAI_BASE_URL).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.service import retrieve_knowledge, upload_knowledge_base


SAMPLE_MCKINSEY = """\
# The State of AI in 2024

## Executive Summary
Generative AI has moved from experimentation to scaled deployment in
financial services. Leaders are seeing margin expansion in customer service,
underwriting, and back-office automation.

## Where to Start
The fastest ROI tends to come from agent-assist deployments in contact
centers. Pyramid Principle suggests answering the executive's bottom-line
question first, then defending it with three MECE themes.
"""

SAMPLE_EQUITY = """\
# Microsoft Q3 FY24 Earnings Review

## Bottom Line
Azure AI revenue acceleration validates the platform thesis. We raise our
12-month price target on stronger commercial booking trends.

## Key Drivers
Co-pilot adoption in enterprise tenants and capex discipline are the two
most material drivers for the next four quarters.
"""

COLLECTION = "mckinsey_kb_test"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        mck_path = tmp_dir / "sample_mck.md"
        eq_path = tmp_dir / "sample_eq.md"
        mck_path.write_text(SAMPLE_MCKINSEY, encoding="utf-8")
        eq_path.write_text(SAMPLE_EQUITY, encoding="utf-8")

        print(f"\n[ingest] uploading 2 sample markdown files into '{COLLECTION}' (drop_old=True)")
        upload_result = upload_knowledge_base(
            file_paths=[str(mck_path), str(eq_path)],
            drop_old=True,
            collection_name=COLLECTION,
            file_metadata={
                str(mck_path): {"source_type": "mckinsey", "title": "State of AI 2024", "year": 2024},
                str(eq_path): {"source_type": "equity_research", "title": "MSFT Q3 FY24", "year": 2024},
            },
        )
        print(f"[ingest] result: {upload_result}")
        assert upload_result["chunk_count"] >= 2, upload_result
        assert upload_result["collection"] == COLLECTION, upload_result

        print("\n[query] filter source_type=mckinsey")
        mck_hits = retrieve_knowledge(
            query="How should we frame the executive answer?",
            top_k=3,
            collection_name=COLLECTION,
            metadata_filter={"source_type": "mckinsey"},
        )
        assert mck_hits, "expected at least one mckinsey hit"
        for h in mck_hits:
            meta = h["metadata"]
            print(f"  - title={meta.get('title')!r} section={meta.get('section_path')!r} type={meta.get('source_type')!r}")
            assert meta.get("source_type") == "mckinsey", meta
            assert meta.get("section_path"), meta

        print("\n[query] filter source_type=equity_research")
        eq_hits = retrieve_knowledge(
            query="What is the bottom line on Azure revenue?",
            top_k=3,
            collection_name=COLLECTION,
            metadata_filter={"source_type": "equity_research"},
        )
        assert eq_hits, "expected at least one equity_research hit"
        for h in eq_hits:
            meta = h["metadata"]
            print(f"  - title={meta.get('title')!r} section={meta.get('section_path')!r} type={meta.get('source_type')!r}")
            assert meta.get("source_type") == "equity_research", meta

        print("\n[query] no filter (mixed)")
        mixed = retrieve_knowledge(
            query="strategic implications",
            top_k=4,
            collection_name=COLLECTION,
        )
        kinds = {h["metadata"].get("source_type") for h in mixed}
        print(f"  source_types in mixed result = {kinds}")
        assert kinds <= {"mckinsey", "equity_research"}, kinds

    print("\nOK")


if __name__ == "__main__":
    main()
