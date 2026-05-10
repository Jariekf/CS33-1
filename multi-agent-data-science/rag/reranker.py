from __future__ import annotations

from langchain_core.documents import Document


class CrossEncoderReranker:
    """
    Lightweight wrapper around sentence-transformers CrossEncoder.
    If dependency is missing, rerank() falls back to original order.
    """

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self._model = None
        self._import_error = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name, max_length=max_length)
        except Exception as exc:  # pragma: no cover - optional dependency
            self._import_error = exc

    def rerank(self, query: str, docs: list[Document], top_k: int | None = None) -> list[Document]:
        """
        Score (query, doc) pairs and reorder docs by descending relevance.
        """
        if not docs:
            return docs
        if self._model is None:
            return docs if top_k is None else docs[:top_k]

        import numpy as np

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self._model.predict(pairs)
        sorted_indices = np.argsort(scores)[::-1]
        reranked = [docs[idx] for idx in sorted_indices]
        return reranked if top_k is None else reranked[:top_k]

    @property
    def available(self) -> bool:
        return self._model is not None
