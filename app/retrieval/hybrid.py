"""Hybrid retrieval — combines dense (FAISS) + lexical (BM25) with RRF."""

from __future__ import annotations

from typing import Optional
import numpy as np

from app.config import get_settings
from app.data.models import RetrievedChunk
from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import FAISSVectorStore
from app.utils.logging import get_logger
from app.utils.timing import Timer

log = get_logger("retrieval.hybrid")


class BM25Index:
    """Lightweight BM25 index built in-memory from chunk texts."""

    def __init__(self) -> None:
        self._index = None
        self._built = False

    @property
    def is_built(self) -> bool:
        return self._built

    def build(self, texts: list[str]) -> None:
        """Build BM25 index from tokenized texts."""
        from rank_bm25 import BM25Okapi

        with Timer(f"BM25 index build ({len(texts)} docs)"):
            tokenized = [t.lower().split() for t in texts]
            self._index = BM25Okapi(tokenized)
        self._built = True
        log.info(f"BM25 index built with {len(texts)} documents")

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Return (doc_index, score) tuples."""
        if not self._built:
            raise RuntimeError("BM25 index not built")
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


class HybridRetriever:
    """Combines dense and BM25 retrieval using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector_store: FAISSVectorStore | None = None,
        embedder: Embedder | None = None,
    ):
        self.settings = get_settings()
        self.vector_store = vector_store or FAISSVectorStore.get_instance()
        self.embedder = embedder or Embedder.get_instance()
        self.bm25 = BM25Index()
        self._bm25_ready = False

    def ensure_bm25(self) -> None:
        """Build BM25 index from vector store chunks if not already built."""
        if self._bm25_ready:
            return
        if not self.vector_store.is_loaded:
            return
        texts = self.vector_store.get_all_texts()
        if texts:
            self.bm25.build(texts)
            self._bm25_ready = True

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using configured mode.

        Args:
            query: User query text
            top_k: Final number of results
            mode: 'dense', 'bm25', or 'hybrid' (default from config)
        """
        k = top_k or self.settings.top_k
        retrieval_mode = mode or self.settings.retrieval_mode
        candidates_k = self.settings.retrieval_candidates

        if retrieval_mode == "dense":
            return self._dense_retrieve(query, k)
        elif retrieval_mode == "bm25":
            return self._bm25_retrieve(query, k)
        else:
            return self._hybrid_retrieve(query, k, candidates_k)

    def _dense_retrieve(
        self, query: str, top_k: int
    ) -> list[RetrievedChunk]:
        query_emb = self.embedder.embed_text(query)
        results = self.vector_store.search(query_emb, top_k=top_k)
        return [
            RetrievedChunk(chunk=chunk, dense_score=score, fused_score=score)
            for chunk, score in results
        ]

    def _bm25_retrieve(
        self, query: str, top_k: int
    ) -> list[RetrievedChunk]:
        self.ensure_bm25()
        all_chunks = self.vector_store.get_all_chunks()
        bm25_results = self.bm25.search(query, top_k=top_k)
        return [
            RetrievedChunk(
                chunk=all_chunks[idx],
                bm25_score=score,
                fused_score=score,
            )
            for idx, score in bm25_results
            if idx < len(all_chunks)
        ]

    def _hybrid_retrieve(
        self, query: str, top_k: int, candidates_k: int
    ) -> list[RetrievedChunk]:
        """Hybrid retrieval with Reciprocal Rank Fusion."""
        self.ensure_bm25()
        all_chunks = self.vector_store.get_all_chunks()

        # Dense retrieval
        with Timer("Dense retrieval"):
            query_emb = self.embedder.embed_text(query)
            dense_results = self.vector_store.search(query_emb, top_k=candidates_k)

        # BM25 retrieval
        with Timer("BM25 retrieval"):
            bm25_results = self.bm25.search(query, top_k=candidates_k)

        # Reciprocal Rank Fusion
        rrf_k = 60  # standard RRF constant
        chunk_scores: dict[str, dict] = {}

        for rank, (chunk, score) in enumerate(dense_results):
            cid = chunk.chunk_id
            if cid not in chunk_scores:
                chunk_scores[cid] = {
                    "chunk": chunk,
                    "dense_score": score,
                    "bm25_score": 0.0,
                    "rrf": 0.0,
                }
            chunk_scores[cid]["dense_score"] = score
            chunk_scores[cid]["rrf"] += self.settings.dense_weight / (rrf_k + rank + 1)

        for rank, (idx, score) in enumerate(bm25_results):
            if idx >= len(all_chunks):
                continue
            chunk = all_chunks[idx]
            cid = chunk.chunk_id
            if cid not in chunk_scores:
                chunk_scores[cid] = {
                    "chunk": chunk,
                    "dense_score": 0.0,
                    "bm25_score": 0.0,
                    "rrf": 0.0,
                }
            chunk_scores[cid]["bm25_score"] = score
            chunk_scores[cid]["rrf"] += self.settings.bm25_weight / (rrf_k + rank + 1)

        # Sort by RRF score, take top-k
        sorted_results = sorted(
            chunk_scores.values(),
            key=lambda x: x["rrf"],
            reverse=True,
        )[:top_k]

        return [
            RetrievedChunk(
                chunk=r["chunk"],
                dense_score=r["dense_score"],
                bm25_score=r["bm25_score"],
                fused_score=r["rrf"],
            )
            for r in sorted_results
        ]
