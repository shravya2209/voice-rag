"""Fixed-size overlapping chunking strategy."""

from __future__ import annotations

from app.chunking.base import BaseChunker
from app.data.models import DatasetRecord, Chunk
from app.config import get_settings


class FixedSizeChunker(BaseChunker):
    """Strategy B: Fixed-size chunks with configurable overlap."""

    strategy_name = "fixed"

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk(self, record: DatasetRecord) -> list[Chunk]:
        text = record.text.strip()
        if not text:
            return []

        # Use character-based chunking (approx 4 chars/token)
        char_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4

        if len(text) <= char_size:
            return [self._make_chunk(record, text, 0)]

        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + char_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(self._make_chunk(
                    record, chunk_text, idx,
                    extra_meta={"char_start": start, "char_end": min(end, len(text))},
                ))
                idx += 1

            start += char_size - char_overlap
            if start >= len(text):
                break

        return chunks
