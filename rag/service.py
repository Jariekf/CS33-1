from __future__ import annotations

from typing import Iterable

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline


def _normalize_file_paths(file_paths: str | Iterable[str]) -> list[str]:
    if isinstance(file_paths, str):
        return [file_paths]
    return list(file_paths)


def upload_knowledge_base(
    file_paths: str | Iterable[str],
    drop_old: bool = False,
    config: RAGConfig | None = None,
) -> dict:
    """
    Upload local text files into the RAG knowledge base.

    This is the public ingestion API for other modules or agents. It handles
    document loading, chunking, embedding generation, and vector-store storage.

    Args:
        file_paths: A single file path string or an iterable of file path strings.
            Paths can be absolute paths, or relative paths based on the current
            process working directory.
        drop_old: Whether to drop the existing vector-store collection before uploading.
            Use True when rebuilding the knowledge base from scratch. Use False
            when appending new documents to the existing collection.
        config: Optional RAGConfig instance. If omitted, configuration is loaded
            from environment variables and defaults in rag.config.RAGConfig.

    Returns:
        A dictionary with ingestion statistics:
            - loaded_documents: Number of source documents loaded.
            - chunk_count: Number of chunks written to the vector database.
            - collection: Vector-store collection name.
            - persist_directory: Local Chroma persistence directory.

    Example:
        upload_knowledge_base("docs/knowledge.txt", drop_old=True)
        upload_knowledge_base(["docs/a.txt", "docs/b.txt"], drop_old=False)
    """
    pipeline = RAGPipeline(config=config)
    return pipeline.ingest_text_files(
        file_paths=_normalize_file_paths(file_paths),
        drop_old=drop_old,
    )


def retrieve_knowledge(
    query: str,
    top_k: int | None = None,
    config: RAGConfig | None = None,
) -> list[dict]:
    """
    Retrieve relevant contexts from the RAG knowledge base.

    This is the public retrieval API for other modules or agents. It handles
    query embedding, vector-store retrieval, optional reranking, and returns
    the final context snippets.

    Args:
        query: The user question or search query.
        top_k: Optional number of final contexts to return. If omitted, the
            default value from RAGConfig.top_k is used. This argument has higher
            priority than config.top_k when provided.
        config: Optional RAGConfig instance. If omitted, configuration is loaded
            from environment variables and defaults in rag.config.RAGConfig.

    Returns:
        A list of dictionaries. Each item contains:
            - content: The retrieved context text.

    Example:
        contexts = retrieve_knowledge("What is RAG?", top_k=3)
        for item in contexts:
            print(item["content"])
    """
    pipeline = RAGPipeline(config=config)
    result = pipeline.query(query_text=query, top_k=top_k)
    return [
        {
            "content": content,
        }
        for content in result["contexts"]
    ]
