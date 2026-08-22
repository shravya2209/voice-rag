"""Optional cross-encoder reranker — configurable via USE_RERANKER."""

from __future__ import annotations

from typing import Optional

from app.config import get_settings
from app.data.models import RetrievedChunk
from app.utils.logging import get_logger
from app.utils.timing import Timer

log = get_logger("retrieval.reranker")


class Reranker:
    """Cross-encoder reranker for candidate re-scoring."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None

    @property
    def model(self):
        """Lazy-load cross-encoder model."""
        if self._model is None:
            with Timer("Load reranker model"):
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.settings.reranker_model,
                    max_length=512,
                )
            log.info(f"Loaded reranker: {self.settings.reranker_model}")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Re-score candidates with cross-encoder and return top-k.

        Args:
            query: User query
            candidates: Pre-retrieved chunks to rerank
            top_k: Number of results after reranking

        Returns:
            Reranked list of RetrievedChunk with rerank_score populated
        """
        if not candidates:
            return []

        k = top_k or self.settings.top_k

        pairs = [(query, rc.chunk.text) for rc in candidates]

        with Timer(f"Rerank {len(pairs)} candidates"):
            scores = self.model.predict(pairs)

        for rc, score in zip(candidates, scores):
            rc.rerank_score = float(score)

        reranked = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)
        return reranked[:k]

    def warmup(self) -> None:
        """Pre-load model."""
        _ = self.model
        log.info("Reranker warmed up")
