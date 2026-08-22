"""Dense vector retriever wrapping the FAISS vector store."""

from __future__ import annotations

import numpy as np

from app.data.models import Chunk, RetrievedChunk
from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import FAISSVectorStore
from app.utils.logging import get_logger

log = get_logger("retrieval.retriever")


class DenseRetriever:
    """Pure dense retrieval using FAISS."""

    def __init__(
        self,
        vector_store: FAISSVectorStore | None = None,
        embedder: Embedder | None = None,
    ):
        self.vector_store = vector_store or FAISSVectorStore.get_instance()
        self.embedder = embedder or Embedder.get_instance()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks by dense similarity."""
        query_emb = self.embedder.embed_text(query)
        results = self.vector_store.search(query_emb, top_k=top_k)

        return [
            RetrievedChunk(
                chunk=chunk,
                dense_score=score,
                fused_score=score,
            )
            for chunk, score in results
        ]
