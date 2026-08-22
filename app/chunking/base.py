"""Abstract base class for all chunking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from app.data.models import DatasetRecord, Chunk


class BaseChunker(ABC):
    """Interface that all chunking strategies must implement."""

    strategy_name: str = "base"

    @abstractmethod
    def chunk(self, record: DatasetRecord) -> list[Chunk]:
        """Split a dataset record into chunks.

        Args:
            record: A single dataset record with text and metadata.

        Returns:
            List of Chunk objects with full metadata.
        """
        ...

    def chunk_batch(self, records: list[DatasetRecord]) -> list[Chunk]:
        """Chunk multiple records."""
        chunks = []
        for record in records:
            chunks.extend(self.chunk(record))
        return chunks

    def _make_chunk(
        self,
        record: DatasetRecord,
        text: str,
        chunk_index: int,
        extra_meta: dict | None = None,
    ) -> Chunk:
        """Helper to create a Chunk with standard metadata."""
        meta = {
            "query_id": record.query_id,
            "is_selected": record.is_selected,
        }
        if extra_meta:
            meta.update(extra_meta)

        return Chunk(
            chunk_id=f"{record.id}_c{chunk_index}",
            document_id=record.id,
            text=text,
            chunk_index=chunk_index,
            strategy=self.strategy_name,
            source=record.source,
            language=record.language,
            metadata=meta,
        )
