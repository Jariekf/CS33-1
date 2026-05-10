from __future__ import annotations

import logging
from dataclasses import asdict, replace
from typing import Mapping

from rag.chroma_vector import connect_chroma_store, create_chroma_store
from rag.config import RAGConfig
from rag.document_processor import build_embeddings, prepare_documents
from rag.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


def _safe_config_dict(config: RAGConfig) -> dict:
    config_dict = asdict(config)
    if config_dict.get("embedding_api_key"):
        config_dict["embedding_api_key"] = "***REDACTED***"
    return config_dict


def _resolve_config(config: RAGConfig | None, collection_name: str | None) -> RAGConfig:
    """Build the effective RAGConfig honoring an optional collection override.

    ``RAGConfig`` is a frozen dataclass; ``replace`` clones it with the new
    collection name without mutating the caller's instance.
    """
    cfg = config or RAGConfig()
    if collection_name and collection_name != cfg.chroma_collection:
        cfg = replace(cfg, chroma_collection=collection_name)
    return cfg


class RAGPipeline:
    """
    End-to-end RAG pipeline:
    1) Load docs (text or markdown-aware)
    2) Chunk docs (with optional markdown header awareness + metadata)
    3) Embed and upsert into Chroma
    4) Retrieve top-k (with optional metadata filter)
    5) Optionally rerank
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        *,
        collection_name: str | None = None,
    ) -> None:
        logger.info("Initializing RAG pipeline")
        self.config = _resolve_config(config, collection_name)
        logger.info(
            "RAG config loaded: chroma_dir=%s, collection=%s, chunk_size=%s, top_k=%s",
            self.config.chroma_persist_directory,
            self.config.chroma_collection,
            self.config.chunk_size,
            self.config.top_k,
        )
        self.embeddings = build_embeddings(self.config)
        logger.info("Embedding model initialized: %s", self.config.embedding_model)
        self.vector_store = None
        self.reranker = (
            CrossEncoderReranker(self.config.reranker_model)
            if self.config.reranker_enabled
            else None
        )
        if self.reranker is None:
            logger.info("Reranker disabled")
        elif self.reranker.available:
            logger.info("Reranker initialized: %s", self.config.reranker_model)
        else:
            logger.warning("Reranker unavailable, retrieval will use original vector order")

    def ingest_text_files(
        self,
        file_paths: list[str],
        drop_old: bool = False,
        *,
        file_metadata: Mapping[str, Mapping[str, object]] | None = None,
        markdown_aware: bool | None = None,
    ) -> dict:
        logger.info("Starting ingestion for %d file(s): %s", len(file_paths), file_paths)
        chunks = prepare_documents(
            file_paths=file_paths,
            config=self.config,
            file_metadata=file_metadata,
            markdown_aware=markdown_aware,
        )
        logger.info("Prepared %d chunk(s) ready for embedding", len(chunks))

        logger.info(
            "Writing chunks to Chroma: persist_directory=%s, collection=%s, drop_old=%s",
            self.config.chroma_persist_directory,
            self.config.chroma_collection,
            drop_old,
        )
        self.vector_store = create_chroma_store(
            documents=chunks,
            embeddings=self.embeddings,
            config=self.config,
            drop_old=drop_old,
        )
        logger.info("Chroma ingestion completed")

        return {
            "loaded_documents": len(file_paths),
            "chunk_count": len(chunks),
            "collection": self.config.chroma_collection,
            "persist_directory": self.config.chroma_persist_directory,
        }

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        *,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> dict:
        if self.vector_store is None:
            logger.info(
                "Vector store not initialized in memory; connecting to collection=%s",
                self.config.chroma_collection,
            )
            self.vector_store = connect_chroma_store(self.embeddings, self.config)

        k = top_k or self.config.top_k
        fetch_k = max(k, self.config.retrieve_fetch_k)
        logger.info(
            "Retrieving documents: query=%s, fetch_k=%d, final_top_k=%d, filter=%s",
            query_text,
            fetch_k,
            k,
            metadata_filter,
        )
        search_kwargs = {"k": fetch_k}
        if metadata_filter:
            search_kwargs["filter"] = dict(metadata_filter)
        retrieved_docs = self.vector_store.similarity_search(query=query_text, **search_kwargs)
        logger.info("Retrieved %d candidate document(s)", len(retrieved_docs))

        if self.reranker is not None:
            logger.info("Reranking candidate documents")
            final_docs = self.reranker.rerank(query_text, retrieved_docs, top_k=k)
        else:
            final_docs = retrieved_docs[:k]
        logger.info("Returning %d final context(s)", len(final_docs))

        return {
            "query": query_text,
            "config": _safe_config_dict(self.config),
            "retrieved_count": len(retrieved_docs),
            "final_count": len(final_docs),
            "contexts": [doc.page_content for doc in final_docs],
            "documents": final_docs,
        }
