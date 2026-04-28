# RAG Configuration and Usage Guide

This folder provides a lightweight Retrieval-Augmented Generation (RAG) service based on:

- OpenAI-compatible embeddings
- Chroma vector storage
- Optional CrossEncoder reranking

Other modules and agents should use the public functions in `rag.service` instead of directly calling lower-level files.

## Environment Variables

Add or update the following variables in the project root `.env` file.

### RAG Chunking and Retrieval

```env
RAG_CHUNK_SIZE=700
RAG_CHUNK_OVERLAP=120
RAG_TOP_K=4
RAG_FETCH_K=10
```

- `RAG_CHUNK_SIZE`: Maximum size of each text chunk.
- `RAG_CHUNK_OVERLAP`: Overlap between neighbouring chunks to preserve context.
- `RAG_TOP_K`: Default number of final contexts returned by retrieval.
- `RAG_FETCH_K`: Number of initial candidates fetched from Chroma before reranking.

### Embedding Model

```env
RAG_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

- `RAG_EMBEDDING_MODEL`: Embedding model used to convert documents and queries into vectors.
- `OPENAI_API_KEY`: API key used by the embedding client.
- `OPENAI_BASE_URL`: OpenAI-compatible base URL used by the embedding client.

Currently, RAG embeddings reuse the existing OpenAI-compatible variables.

### Chroma

```env
RAG_CHROMA_DIR=./workspace/chroma_rag
RAG_CHROMA_COLLECTION=rag_collection
```

- `RAG_CHROMA_DIR`: Local directory used by Chroma to persist vector data.
- `RAG_CHROMA_COLLECTION`: Collection name used to store RAG vectors.

Chroma is persistent in this project because `persist_directory` is configured. Uploaded vectors are stored under `RAG_CHROMA_DIR` and can be reused by later retrieval calls as long as the directory is kept.

### Reranker

```env
RAG_ENABLE_RERANKER=true
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

- `RAG_ENABLE_RERANKER`: Whether to rerank retrieved candidates.
- `RAG_RERANKER_MODEL`: CrossEncoder model name used by `sentence-transformers`.

If the reranker is unavailable, retrieval falls back to the original vector search order.

## Public APIs

Import public functions from `rag.service`.

```python
from rag.service import upload_knowledge_base, retrieve_knowledge
```

### Upload Knowledge Base

Use `upload_knowledge_base()` to load local text files, split them into chunks, embed them, and store them in Chroma.

```python
upload_result = upload_knowledge_base(
    file_paths="tests/sample_knowledge.txt",
    drop_old=True,
)
```

You can also upload multiple files:

```python
upload_result = upload_knowledge_base(
    file_paths=[
        "docs/knowledge_a.txt",
        "docs/knowledge_b.txt",
    ],
    drop_old=False,
)
```

Parameters:

- `file_paths`: A single path string or a list of path strings.
- `drop_old`: If `True`, rebuild the Chroma collection from scratch. If `False`, append to the existing collection.
- `config`: Optional `RAGConfig` object. Usually not needed.

Return value example:

```python
{
    "loaded_documents": 1,
    "chunk_count": 3,
    "collection": "rag_collection",
    "persist_directory": "./workspace/chroma_rag",
}
```

### Retrieve Knowledge

Use `retrieve_knowledge()` to retrieve relevant context snippets for a query.

```python
contexts = retrieve_knowledge(
    query="What is RAG?",
    top_k=3,
)
```

Parameters:

- `query`: Query text.
- `top_k`: Optional number of final contexts to return. If omitted, `RAG_TOP_K` from config is used.
- `config`: Optional `RAGConfig` object. Usually not needed.

Return value example:

```python
[
    {
        "content": "Retrieved context text..."
    }
]
```

## Path Notes

Relative paths are resolved based on the current process working directory. If a script is run from an IDE and cannot find a file, use an absolute path or build the path from the project root:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
knowledge_file = PROJECT_ROOT / "tests" / "sample_knowledge.txt"
upload_knowledge_base(str(knowledge_file), drop_old=True)
```

## Quick Test

Run the service-level test from the project root:

```bash
./.venv/bin/python tests/test_rag_service.py
```

This uploads `tests/sample_knowledge.txt` and retrieves context for a sample query.
