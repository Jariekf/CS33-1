from __future__ import annotations

from typing import Iterable, Mapping

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline


def _normalize_file_paths(file_paths: str | Iterable[str]) -> list[str]:
    if isinstance(file_paths, str):
        return [file_paths]
    return list(file_paths)


def upload_knowledge_base(
    file_paths: str | Iterable[str],
    drop_old: bool = False,
    *,
    collection_name: str | None = None,
    file_metadata: Mapping[str, Mapping[str, object]] | None = None,
    markdown_aware: bool | None = None,
    config: RAGConfig | None = None,
) -> dict:
    """
    Upload local text/markdown files into a RAG knowledge base.

    This is the public ingestion API. It handles document loading, chunking,
    embedding generation, and vector-store storage.

    Args:
        file_paths: A single file path string or an iterable of file path strings.
            Paths can be absolute, or relative to the current working directory.
        drop_old: Whether to drop the existing collection before uploading.
            Use ``True`` when rebuilding the knowledge base from scratch,
            ``False`` when appending new documents.
        collection_name: Optional Chroma collection name. When provided, this
            takes precedence over ``config.chroma_collection`` / the
            ``RAG_CHROMA_COLLECTION`` env var. Use this to keep specialised
            knowledge bases (e.g. ``mckinsey_kb``) isolated from the default
            ``rag_collection``.
        file_metadata: Optional ``{path: metadata_dict}`` mapping. Each
            ``metadata_dict`` is merged into every chunk produced from that
            file (e.g. ``{"source_type": "mckinsey", "title": "...", "year": 2024}``).
        markdown_aware: If ``None`` (default), markdown files (``.md`` /
            ``.markdown``) are auto-detected and processed with header-aware
            splitting; other files use the plain-text path. Pass ``True`` /
            ``False`` to force a single mode.
        config: Optional ``RAGConfig`` instance. When omitted, configuration is
            loaded from environment variables / defaults.

    Returns:
        Ingestion statistics: ``loaded_documents``, ``chunk_count``,
        ``collection``, ``persist_directory``.

    Examples:
        # generic plain-text usage (existing behaviour, unchanged)
        upload_knowledge_base("docs/knowledge.txt", drop_old=True)

        # McKinsey knowledge base with metadata + isolated collection
        upload_knowledge_base(
            file_paths=["data/mckinsey_kb/raw/a.md", "data/mckinsey_kb/raw/b.md"],
            drop_old=True,
            collection_name="mckinsey_kb",
            file_metadata={
                "data/mckinsey_kb/raw/a.md": {"source_type": "mckinsey", "title": "...", "year": 2024},
                "data/mckinsey_kb/raw/b.md": {"source_type": "equity_research", "title": "...", "year": 2024},
            },
        )
    """
    pipeline = RAGPipeline(config=config, collection_name=collection_name)
    return pipeline.ingest_text_files(
        file_paths=_normalize_file_paths(file_paths),
        drop_old=drop_old,
        file_metadata=file_metadata,
        markdown_aware=markdown_aware,
    )


def retrieve_knowledge(
    query: str,
    top_k: int | None = None,
    *,
    collection_name: str | None = None,
    metadata_filter: Mapping[str, object] | None = None,
    config: RAGConfig | None = None,
) -> list[dict]:
    """
    Retrieve relevant contexts from a RAG knowledge base.

    Args:
        query: Search query text.
        top_k: Number of final contexts to return. Falls back to
            ``RAGConfig.top_k`` when ``None``.
        collection_name: Optional Chroma collection name (same precedence as
            in :func:`upload_knowledge_base`).
        metadata_filter: Optional Chroma metadata filter, e.g.
            ``{"source_type": "mckinsey"}``. Passed directly to Chroma's
            ``similarity_search(filter=...)``.
        config: Optional ``RAGConfig`` instance.

    Returns:
        A list of dicts. Each item contains:
            - ``content``: retrieved chunk text
            - ``metadata``: chunk metadata (source, source_type, section_path, ...)

    Example:
        contexts = retrieve_knowledge(
            "How should we frame strategic implications?",
            top_k=3,
            collection_name="mckinsey_kb",
            metadata_filter={"source_type": "mckinsey"},
        )
    """
    pipeline = RAGPipeline(config=config, collection_name=collection_name)
    result = pipeline.query(
        query_text=query,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )
    documents = result.get("documents") or []
    return [
        {
            "content": doc.page_content,
            "metadata": dict(doc.metadata or {}),
        }
        for doc in documents
    ]
