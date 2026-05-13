"""
arXiv search tool.
Wraps the arxiv Python library for literature search used by Manager (RAP) and Business Translator (RAG).
"""
from __future__ import annotations

from config import ARXIV_MAX_RESULTS


def search_arxiv(query: str, max_results: int | None = None) -> list[dict]:
    """
    Search arXiv papers.

    Args:
        query: Search keywords
        max_results: Maximum number of results; defaults to config

    Returns:
        List of papers, each with title, summary, url, etc.
    """
    import os
    if os.getenv("STUB_MODE", "false").lower() == "true":
        print(f"[Stub Mode] Skipping arXiv retrieval. Query: {query}")
        return [{
            "title": "Stub Paper on " + query, 
            "summary": "This is a dummy paper for testing the pipeline early stage.", 
            "url": "http://arxiv.org/abs/0000.0000", 
            "published": "2024-01-01", 
            "authors": ["Stub Author"]
        }]

    if max_results is None:
        max_results = ARXIV_MAX_RESULTS

    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "summary": paper.summary,
                "url": paper.entry_id,
                "published": str(paper.published),
                "authors": [a.name for a in paper.authors[:3]],
            })
        return results

    except ImportError:
        print("[Warning] arxiv package is not installed; returning empty results. Run: pip install arxiv")
        return []
    except Exception as e:
        print(f"[Warning] arXiv retrieval failed: {e}")
        return []
