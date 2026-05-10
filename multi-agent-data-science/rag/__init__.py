"""RAG package: Chroma + embeddings + retrieval for BAT."""

from rag.config import RAGConfig
from rag.service import retrieve_knowledge, upload_knowledge_base

__all__ = ["RAGConfig", "retrieve_knowledge", "upload_knowledge_base"]
