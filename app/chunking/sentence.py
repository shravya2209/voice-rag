"""Sentence-aware chunking strategy."""

from __future__ import annotations

import re

from app.chunking.base import BaseChunker
from app.data.models import DatasetRecord, Chunk
from app.config import get_settings


# Simple sentence splitter regex — handles common abbreviations
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = _SENT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


class SentenceChunker(BaseChunker):
    """Strategy A: Sentence-aware chunking.

    Splits on sentence boundaries and combines sentences until
    a target character budget is reached.
    """

    strategy_name = "sentence"

    def __init__(self, chunk_size: int | None = None):
        settings = get_settings()
        self.max_chars = (chunk_size or settings.chunk_size) * 4

    def chunk(self, record: DatasetRecord) -> list[Chunk]:
        text = record.text.strip()
        if not text:
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return [self._make_chunk(record, text, 0)]

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_len = 0
        idx = 0

        for sent in sentences:
            sent_len = len(sent)

            if current_len + sent_len > self.max_chars and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(self._make_chunk(
                    record, chunk_text, idx,
                    extra_meta={"num_sentences": len(current_sentences)},
                ))
                idx += 1
                current_sentences = []
                current_len = 0

            current_sentences.append(sent)
            current_len += sent_len + 1  # +1 for space

        # Remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(self._make_chunk(
                record, chunk_text, idx,
                extra_meta={"num_sentences": len(current_sentences)},
            ))

        return chunks
