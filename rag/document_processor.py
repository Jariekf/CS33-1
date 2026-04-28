from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import RAGConfig


def load_text_documents(file_paths: Iterable[str], encoding: str = "utf-8") -> list[Document]:
    """Load one or more plain-text files into LangChain Document objects."""
    documents: list[Document] = []
    for raw_path in file_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        loader = TextLoader(str(path), encoding=encoding)
        documents.extend(loader.load())
    return documents


def chunk_documents(documents: list[Document], config: RAGConfig) -> list[Document]:
    """Split long documents into retrieval-friendly chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return splitter.split_documents(documents)


def prepare_documents(file_paths: Iterable[str], config: RAGConfig) -> list[Document]:
    """Load and chunk files for vector-store ingestion."""
    documents = load_text_documents(file_paths)
    return chunk_documents(documents, config)


def build_embeddings(config: RAGConfig) -> OpenAIEmbeddings:
    """Build an embedding client using OpenAI-compatible configuration."""
    kwargs = {
        "model": config.embedding_model,
        "api_key": config.embedding_api_key,
    }
    if config.embedding_base_url:
        kwargs["base_url"] = config.embedding_base_url
    return OpenAIEmbeddings(**kwargs)
