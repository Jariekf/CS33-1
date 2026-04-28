from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.config import RAGConfig


def _import_milvus_client():
    """
    Import pymilvus after protecting it from local .db values
    accidentally stored in its reserved MILVUS_URI environment variable.
    """
    import os

    reserved_uri = os.getenv("MILVUS_URI", "").strip()
    if not reserved_uri or not reserved_uri.startswith(("http://", "https://")):
        os.environ["MILVUS_URI"] = "http://localhost:19530"

    from pymilvus import MilvusClient

    return MilvusClient


def _document_id(doc: Document, index: int) -> int:
    raw = f"{doc.page_content}:{doc.metadata}:{index}".encode("utf-8")
    return int(hashlib.sha1(raw).hexdigest()[:15], 16)


class MilvusVectorStore:
    """Minimal Milvus vector store wrapper with LangChain-like search API."""

    def __init__(self, embeddings: Embeddings, config: RAGConfig) -> None:
        _ensure_local_milvus_parent(config.milvus_uri)
        MilvusClient = _import_milvus_client()
        self.embeddings = embeddings
        self.config = config
        self.client = MilvusClient(uri=config.milvus_uri)

    def add_documents(self, documents: list[Document], drop_old: bool = False) -> None:
        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)
        dimension = len(vectors[0]) if vectors else 0

        if drop_old and self.client.has_collection(self.config.milvus_collection):
            self.client.drop_collection(self.config.milvus_collection)

        if not self.client.has_collection(self.config.milvus_collection):
            self.client.create_collection(
                collection_name=self.config.milvus_collection,
                dimension=dimension,
                metric_type="COSINE",
            )

        rows = [
            {
                "id": _document_id(doc, index),
                "vector": vector,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            for index, (doc, vector) in enumerate(zip(documents, vectors))
        ]
        if rows:
            self.client.insert(collection_name=self.config.milvus_collection, data=rows)

    def similarity_search(self, query: str, k: int) -> list[Document]:
        query_vector = self.embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.config.milvus_collection,
            data=[query_vector],
            limit=k,
            output_fields=["text", "metadata"],
        )

        docs: list[Document] = []
        for hit in results[0] if results else []:
            entity = hit.get("entity", {})
            docs.append(
                Document(
                    page_content=entity.get("text", ""),
                    metadata=entity.get("metadata", {}),
                )
            )
        return docs


def _ensure_local_milvus_parent(uri: str) -> None:
    """Create parent directory for local Milvus Lite database files."""
    if uri.startswith(("http://", "https://")):
        return
    parent = Path(uri).expanduser().parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def create_milvus_store(
    documents: list[Document],
    embeddings: Embeddings,
    config: RAGConfig,
    drop_old: bool = False,
) -> MilvusVectorStore:
    """
    Create or rebuild a Milvus vector store from documents.
    """
    vector_store = MilvusVectorStore(embeddings=embeddings, config=config)
    vector_store.add_documents(documents, drop_old=drop_old)
    return vector_store


def connect_milvus_store(embeddings: Embeddings, config: RAGConfig) -> MilvusVectorStore:
    """
    Connect to an existing Milvus collection for retrieval.
    """
    return MilvusVectorStore(embeddings=embeddings, config=config)
