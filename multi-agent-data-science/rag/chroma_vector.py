from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.config import RAGConfig


def _ensure_chroma_parent(persist_directory: str) -> None:
    Path(persist_directory).expanduser().mkdir(parents=True, exist_ok=True)


def create_chroma_store(
    documents: list[Document],
    embeddings: Embeddings,
    config: RAGConfig,
    drop_old: bool = False,
) -> Chroma:
    """Create or rebuild a Chroma vector store from documents."""
    _ensure_chroma_parent(config.chroma_persist_directory)
    vector_store = Chroma(
        collection_name=config.chroma_collection,
        embedding_function=embeddings,
        persist_directory=config.chroma_persist_directory,
    )

    if drop_old:
        vector_store.delete_collection()
        vector_store = Chroma(
            collection_name=config.chroma_collection,
            embedding_function=embeddings,
            persist_directory=config.chroma_persist_directory,
        )

    if documents:
        vector_store.add_documents(documents)
    return vector_store


def connect_chroma_store(embeddings: Embeddings, config: RAGConfig) -> Chroma:
    """Connect to an existing Chroma collection for retrieval."""
    _ensure_chroma_parent(config.chroma_persist_directory)
    return Chroma(
        collection_name=config.chroma_collection,
        embedding_function=embeddings,
        persist_directory=config.chroma_persist_directory,
    )
