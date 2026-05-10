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
RAG_EMBEDDING_API_KEY=your_api_key
RAG_EMBEDDING_BASE_URL=https://api.openai.com/v1
```

- `RAG_EMBEDDING_MODEL`: Embedding model used to convert documents and queries into vectors.
- `RAG_EMBEDDING_API_KEY`: API key for the embedding client. Falls back to
  `OPENAI_API_KEY` when unset (backward compatible with earlier setups).
- `RAG_EMBEDDING_BASE_URL`: OpenAI-compatible base URL. Falls back to
  `OPENAI_BASE_URL` when unset.

Decoupling RAG from `OPENAI_*` lets you keep the main LLM routing (DeepSeek /
Zhipu / Qwen / ... in `core/my_llm.py`) independent from the embedding
provider. Common provider examples:

| Provider | base URL | model |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-small` |
| DashScope (Qwen) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `text-embedding-v3` |
| Zhipu | `https://open.bigmodel.cn/api/paas/v4` | `embedding-3` |

### Chroma

```env
RAG_CHROMA_DIR=./artifacts/chroma_rag
RAG_CHROMA_COLLECTION=rag_collection
```

- `RAG_CHROMA_DIR`: Local directory used by Chroma to persist vector data (default under `multi-agent-data-science/artifacts/`).
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
    "persist_directory": "./artifacts/chroma_rag",
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

---

## McKinsey Knowledge Base Usage

A specialised, isolated knowledge base for the BAT agent that learns to write
McKinsey-style business reports. Lives in collection `mckinsey_kb`, separate
from the generic `rag_collection`.

### 1) Register your sources in `data/mckinsey_kb/manifest.yaml`

Each entry must provide **exactly one** of the source fields:

```yaml
# Online webpage / online PDF -> fetched via Jina Reader
- url: https://www.mckinsey.com/.../the-state-of-ai-in-2024
  source_type: mckinsey
  title: "The state of AI in 2024"
  year: 2024

# Local PDF -> converted via pymupdf4llm (no network)
- local_pdf: data/mckinsey_kb/source_pdf/eq_msft_2024q3.pdf
  source_type: equity_research
  title: "Microsoft Q3 FY24 Earnings Review"
  year: 2024

# Already in markdown -> pass-through (no conversion)
- local_path: data/mckinsey_kb/source_md/pyramid_principle.md
  source_type: textbook
  title: "The Pyramid Principle"
  year: 1985
```

`source_type` must be one of `mckinsey | equity_research | textbook`.

### 2) Prepare markdown (URL + PDF -> local md)

Use CS33-automl `scripts/prepare_mckinsey_md.py` if available, or place `.md` files under `data/mckinsey_kb/raw/`.

### 3) Upload into Chroma collection `mckinsey_kb`

```bash
python scripts/build_mckinsey_kb.py
```

Each chunk receives metadata: `source_type`, `title`, `year`, `section_path`
(reconstructed from H1>H2>H3 of the original markdown).

### 4) Query the collection programmatically

```python
from rag.service import retrieve_knowledge

hits = retrieve_knowledge(
    query="How should we frame strategic implications?",
    top_k=3,
    collection_name="mckinsey_kb",
    metadata_filter={"source_type": "mckinsey"},
)
for h in hits:
    print(h["metadata"].get("title"), "->", h["metadata"].get("section_path"))
    print(h["content"][:200])
```

### Tunable env vars

| Variable | Default | Effect |
|---|---|---|
| `RAG_MD_CHUNK_SIZE` | 1200 | Chunk size for markdown ingestion |
| `RAG_MD_CHUNK_OVERLAP` | 200 | Overlap between markdown chunks |
| `MCKINSEY_COLLECTION` | mckinsey_kb | Collection name BAT retrieves from |
| `MCKINSEY_PER_BUCKET_TOP_K` | 3 | Hits per source_type bucket per query |
| `MCKINSEY_MIN_HITS` | 4 | If aggregated McKinsey hits < N, BAT falls back to arxiv |
| `JINA_API_KEY` | (none) | Optional, sent as Bearer token to Jina Reader |
