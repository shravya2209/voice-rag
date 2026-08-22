"""Metadata-aware chunking — preserves document boundaries and metadata."""

from __future__ import annotations

from app.chunking.base import BaseChunker
from app.chunking.sentence import SentenceChunker
from app.data.models import DatasetRecord, Chunk
from app.config import get_settings


class MetadataChunker(BaseChunker):
    """Strategy D: Metadata-aware chunking.

    Preserves meaningful boundaries from the dataset structure.
    For MSMARCO-XI, each passage is already a self-contained unit,
    so this strategy keeps passages intact when they fit within size
    limits, and falls back to sentence chunking for oversized passages.
    Enriches chunks with full metadata from the source record.
    """

    strategy_name = "metadata"

    def __init__(self, max_chunk_chars: int | None = None):
        settings = get_settings()
        # max_chunk_chars is in raw characters
        self.max_chars = max_chunk_chars or (settings.chunk_size * 4)
        # Sentence chunker needs token count: divide by 4
        sentence_tokens = max(self.max_chars // 4, 32)
        self._sentence_fallback = SentenceChunker(chunk_size=sentence_tokens)

    def chunk(self, record: DatasetRecord) -> list[Chunk]:
        text = record.text.strip()
        if not text:
            return []

        # If the passage fits within limits, keep it as one chunk
        if len(text) <= self.max_chars:
            return [self._make_chunk(
                record, text, 0,
                extra_meta={
                    "preserved_boundary": True,
                    "original_length": len(text),
                },
            )]

        # Oversized passages: fall back to sentence chunking
        # but re-tag with metadata strategy name
        fallback_chunks = self._sentence_fallback.chunk(record)
        for c in fallback_chunks:
            c.strategy = self.strategy_name
            c.metadata["preserved_boundary"] = False
            c.metadata["fallback"] = "sentence"

        return fallback_chunks
