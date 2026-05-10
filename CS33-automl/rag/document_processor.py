from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.config import RAGConfig
from rag.preprocess import clean_markdown

# Header levels passed to MarkdownHeaderTextSplitter. We capture H1..H4 so a
# section path like "Executive Summary > Key Findings > Subsection" can be
# reconstructed from chunk metadata.
_MD_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

_MARKDOWN_SUFFIXES = {".md", ".markdown"}


def _is_markdown_file(path: Path) -> bool:
    return path.suffix.lower() in _MARKDOWN_SUFFIXES


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


def _build_section_path(meta: Mapping[str, str]) -> str:
    parts = [meta.get(level) for _, level in _MD_HEADERS]
    return " > ".join(p for p in parts if p)


def load_markdown_documents(
    file_paths: Iterable[str],
    file_metadata: Mapping[str, Mapping[str, object]] | None = None,
    encoding: str = "utf-8",
) -> list[Document]:
    """Load markdown files, split by headings, attach user metadata + section path.

    For each input path we:
      1. Read the raw text
      2. Strip image refs, residual HTML, and obvious boilerplate via
         ``clean_markdown`` (so they do not pollute embeddings)
      3. Split on H1-H4 with ``MarkdownHeaderTextSplitter`` (each output
         Document has heading levels stored in ``metadata``)
      4. Merge user-provided metadata (source_type / title / year / ...) and
         a flattened ``section_path`` string into each chunk

    The output is *header-level* sections; downstream callers should still
    run :func:`chunk_documents` to enforce a maximum chunk size.
    """
    file_metadata = file_metadata or {}
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS,
        strip_headers=False,
    )

    documents: list[Document] = []
    for raw_path in file_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        text = path.read_text(encoding=encoding)
        cleaned = clean_markdown(text)

        per_file_meta = dict(file_metadata.get(str(path), {}) or file_metadata.get(raw_path, {}) or {})
        per_file_meta.setdefault("source", str(path))

        sections = splitter.split_text(cleaned)
        if not sections:
            # File had no headers at all: keep it as one Document so the user
            # does not silently lose content.
            doc = Document(page_content=cleaned, metadata={**per_file_meta, "section_path": ""})
            documents.append(doc)
            continue

        for sec in sections:
            section_path = _build_section_path(sec.metadata)
            merged_meta = {**per_file_meta, **sec.metadata, "section_path": section_path}
            documents.append(Document(page_content=sec.page_content, metadata=merged_meta))

    return documents


def chunk_documents(
    documents: list[Document],
    config: RAGConfig,
    *,
    markdown_aware: bool = False,
) -> list[Document]:
    """Split long documents into retrieval-friendly chunks.

    ``markdown_aware=True`` uses the configured markdown chunk size/overlap
    (typically larger to keep narrative paragraphs intact under each heading);
    plain-text path keeps the original 700/120 default to preserve backward
    compatibility with the existing generic RAG collection.
    """
    if markdown_aware:
        size, overlap = config.markdown_chunk_size, config.markdown_chunk_overlap
    else:
        size, overlap = config.chunk_size, config.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
    )
    return splitter.split_documents(documents)


def prepare_documents(
    file_paths: Iterable[str],
    config: RAGConfig,
    *,
    file_metadata: Mapping[str, Mapping[str, object]] | None = None,
    markdown_aware: bool | None = None,
) -> list[Document]:
    """Load and chunk files for vector-store ingestion.

    When ``markdown_aware`` is None it is auto-detected per-file from the
    extension: ``.md`` / ``.markdown`` go through the header-aware loader,
    everything else falls back to ``TextLoader``.
    """
    file_paths = list(file_paths)

    if markdown_aware is None:
        md_paths = [p for p in file_paths if _is_markdown_file(Path(p))]
        plain_paths = [p for p in file_paths if not _is_markdown_file(Path(p))]
    elif markdown_aware:
        md_paths, plain_paths = file_paths, []
    else:
        md_paths, plain_paths = [], file_paths

    documents: list[Document] = []

    if md_paths:
        md_docs = load_markdown_documents(md_paths, file_metadata=file_metadata)
        documents.extend(chunk_documents(md_docs, config, markdown_aware=True))

    if plain_paths:
        plain_docs = load_text_documents(plain_paths)
        if file_metadata:
            for d in plain_docs:
                src = d.metadata.get("source")
                meta = file_metadata.get(src) if src else None
                if meta:
                    d.metadata.update(dict(meta))
        documents.extend(chunk_documents(plain_docs, config, markdown_aware=False))

    return documents


def build_embeddings(config: RAGConfig) -> OpenAIEmbeddings:
    """Build an embedding client using OpenAI-compatible configuration.

    ``check_embedding_ctx_length`` is disabled so the client sends raw strings
    instead of pre-tokenised token-id arrays. The latter is supported by
    OpenAI's official endpoint but rejected by most third-party
    OpenAI-compatible providers (DashScope, Zhipu, Kimi, ...) with
    ``InvalidParameter: contents is neither str nor list of str``. We already
    enforce chunk size at the splitter layer, so we do not need tiktoken to
    re-truncate batches here.
    """
    kwargs = {
        "model": config.embedding_model,
        "api_key": config.embedding_api_key,
        "check_embedding_ctx_length": False,
        # Per-request batch size; see ``RAGConfig.embedding_batch_size`` for
        # the rationale. Without this, langchain defaults to 1000 which
        # exceeds DashScope (10) / Zhipu (64) per-request limits and yields
        # ``batch size is invalid`` 400s.
        "chunk_size": config.embedding_batch_size,
    }
    if config.embedding_base_url:
        kwargs["base_url"] = config.embedding_base_url
    return OpenAIEmbeddings(**kwargs)
