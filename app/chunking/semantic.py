"""Semantic chunking strategy — groups related sentences by embedding similarity."""

from __future__ import annotations

import re
import numpy as np
from typing import TYPE_CHECKING

from app.chunking.base import BaseChunker
from app.data.models import DatasetRecord, Chunk
from app.config import get_settings

if TYPE_CHECKING:
    from app.embeddings.embedder import Embedder


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    sentences = _SENT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


class SemanticChunker(BaseChunker):
    """Strategy C: Semantic chunking.

    Groups semantically related sentences by computing pairwise
    embedding similarity and merging adjacent sentences that exceed
    a similarity threshold.
    """

    strategy_name = "semantic"

    def __init__(
        self,
        embedder: "Embedder | None" = None,
        similarity_threshold: float | None = None,
        max_chunk_chars: int | None = None,
    ):
        settings = get_settings()
        self.similarity_threshold = (
            similarity_threshold or settings.semantic_similarity_threshold
        )
        self.max_chunk_chars = (max_chunk_chars or settings.chunk_size) * 4
        self._embedder = embedder

    @property
    def embedder(self) -> "Embedder":
        if self._embedder is None:
            from app.embeddings.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def chunk(self, record: DatasetRecord) -> list[Chunk]:
        text = record.text.strip()
        if not text:
            return []

        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [self._make_chunk(record, text, 0)]

        # Embed all sentences
        embeddings = self.embedder.embed_batch(sentences)

        # Compute cosine similarities between adjacent sentences
        groups: list[list[int]] = [[0]]
        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            combined_len = sum(len(sentences[j]) for j in groups[-1]) + len(sentences[i])

            if sim >= self.similarity_threshold and combined_len <= self.max_chunk_chars:
                groups[-1].append(i)
            else:
                groups.append([i])

        # Build chunks from groups
        chunks: list[Chunk] = []
        for idx, group in enumerate(groups):
            chunk_text = " ".join(sentences[i] for i in group)
            chunks.append(self._make_chunk(
                record, chunk_text, idx,
                extra_meta={
                    "num_sentences": len(group),
                    "sentence_indices": group,
                },
            ))

        return chunks
