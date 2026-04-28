from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)

_RESERVED_MILVUS_URI = os.getenv("MILVUS_URI", "").strip()
if _RESERVED_MILVUS_URI and not _RESERVED_MILVUS_URI.startswith(("http://", "https://")):
    os.environ.pop("MILVUS_URI", None)


def _resolve_milvus_uri() -> str:
    """
    Resolve Milvus URI without leaking local Milvus Lite paths into pymilvus'
    reserved MILVUS_URI environment variable.
    """
    return os.getenv("RAG_MILVUS_URI") or _RESERVED_MILVUS_URI or "./milvus_rag.db"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RAGConfig:
    """Centralized runtime configuration for the RAG system."""

    # Data processing
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

    # Retrieval
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    retrieve_fetch_k: int = int(os.getenv("RAG_FETCH_K", "10"))

    # Milvus
    milvus_uri: str = _resolve_milvus_uri()
    milvus_collection: str = os.getenv("RAG_MILVUS_COLLECTION", "rag_collection")

    # Embedding model (OpenAI-compatible endpoint)
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Optional reranker
    reranker_model: str = os.getenv(
        "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker_enabled: bool = _env_bool("RAG_ENABLE_RERANKER", True)
