from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RAGConfig:
    """Centralized runtime configuration for the RAG system."""

    # Data processing - plain text
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

    # Data processing - markdown-aware path (used when ingesting structured
    # markdown such as McKinsey / equity research / textbook content). Larger
    # chunks better preserve narrative paragraphs under each heading.
    markdown_chunk_size: int = int(os.getenv("RAG_MD_CHUNK_SIZE", "1200"))
    markdown_chunk_overlap: int = int(os.getenv("RAG_MD_CHUNK_OVERLAP", "200"))

    # Retrieval
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    retrieve_fetch_k: int = int(os.getenv("RAG_FETCH_K", "10"))

    # Chroma
    chroma_persist_directory: str = os.getenv("RAG_CHROMA_DIR", "./workspace/chroma_rag")
    chroma_collection: str = os.getenv("RAG_CHROMA_COLLECTION", "rag_collection")

    # Embedding model (OpenAI-compatible endpoint).
    # ``RAG_EMBEDDING_*`` takes precedence so the RAG embedding provider can be
    # configured independently of the main LLM routing (which uses
    # ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` for the OpenAI profile in
    # ``core/my_llm.py``). Falls back to ``OPENAI_*`` for backward
    # compatibility with the original setup.
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_api_key: str = os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    embedding_base_url: str = (
        os.getenv("RAG_EMBEDDING_BASE_URL")
        or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    # Batch size sent per /embeddings request. Provider limits vary widely:
    #   - OpenAI official: up to 2048
    #   - DashScope (Qwen text-embedding-v3): 10
    #   - Zhipu embedding-3: 64
    # Default to 10 because that is the strictest common limit and the value
    # is purely an HTTP-batching concern (it does not affect retrieval
    # quality). Increase via env when on a more permissive provider.
    embedding_batch_size: int = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "10"))

    # Optional reranker
    reranker_model: str = os.getenv(
        "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker_enabled: bool = _env_bool("RAG_ENABLE_RERANKER", True)
