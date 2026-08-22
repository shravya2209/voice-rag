"""Tests for all chunking strategies."""

import pytest
from app.data.models import DatasetRecord
from app.chunking.fixed import FixedSizeChunker
from app.chunking.sentence import SentenceChunker
from app.chunking.metadata import MetadataChunker
from app.chunking import get_chunker


def make_record(text: str, id: str = "test_0") -> DatasetRecord:
    return DatasetRecord(id=id, text=text, source="test")


SHORT_TEXT = "This is a short passage."
LONG_TEXT = (
    "The Earth orbits the Sun. It takes approximately 365 days to complete one orbit. "
    "The Earth's axis is tilted at about 23.5 degrees. This tilt causes the seasons. "
    "During summer, one hemisphere is tilted toward the Sun. During winter, it tilts away. "
    "The equinoxes occur when neither hemisphere is tilted toward the Sun. "
    "The Earth also rotates on its axis once every 24 hours. "
    "This rotation causes day and night. The side facing the Sun experiences day. "
    "The opposite side experiences night. The Earth's atmosphere protects life from harmful radiation."
)


class TestFixedSizeChunker:
    def test_short_text_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(make_record(SHORT_TEXT))
        assert len(chunks) == 1
        assert chunks[0].text == SHORT_TEXT
        assert chunks[0].strategy == "fixed"

    def test_long_text_multiple_chunks(self):
        chunker = FixedSizeChunker(chunk_size=30, chunk_overlap=5)
        chunks = chunker.chunk(make_record(LONG_TEXT))
        assert len(chunks) > 1

    def test_metadata_preserved(self):
        chunker = FixedSizeChunker()
        chunks = chunker.chunk(make_record(SHORT_TEXT, id="doc_42"))
        assert chunks[0].document_id == "doc_42"
        assert chunks[0].chunk_id.startswith("doc_42_c")

    def test_empty_text(self):
        chunker = FixedSizeChunker()
        chunks = chunker.chunk(make_record(""))
        assert len(chunks) == 0


class TestSentenceChunker:
    def test_sentence_splitting(self):
        chunker = SentenceChunker(chunk_size=30)
        chunks = chunker.chunk(make_record(LONG_TEXT))
        assert len(chunks) >= 1
        assert all(c.strategy == "sentence" for c in chunks)

    def test_short_text_single_chunk(self):
        chunker = SentenceChunker(chunk_size=200)
        chunks = chunker.chunk(make_record(SHORT_TEXT))
        assert len(chunks) == 1

    def test_metadata_contains_sentence_count(self):
        chunker = SentenceChunker(chunk_size=30)
        chunks = chunker.chunk(make_record(LONG_TEXT))
        for c in chunks:
            assert "num_sentences" in c.metadata

    def test_empty_text(self):
        chunker = SentenceChunker()
        chunks = chunker.chunk(make_record(""))
        assert len(chunks) == 0


class TestMetadataChunker:
    def test_short_text_preserved(self):
        chunker = MetadataChunker(max_chunk_chars=2000)
        chunks = chunker.chunk(make_record(SHORT_TEXT))
        assert len(chunks) == 1
        assert chunks[0].metadata.get("preserved_boundary") is True

    def test_long_text_fallback(self):
        chunker = MetadataChunker(max_chunk_chars=50)
        chunks = chunker.chunk(make_record(LONG_TEXT))
        assert len(chunks) > 1
        assert all(c.strategy == "metadata" for c in chunks)

    def test_empty_text(self):
        chunker = MetadataChunker()
        chunks = chunker.chunk(make_record(""))
        assert len(chunks) == 0


class TestChunkerFactory:
    def test_get_fixed(self):
        c = get_chunker("fixed")
        assert isinstance(c, FixedSizeChunker)

    def test_get_sentence(self):
        c = get_chunker("sentence")
        assert isinstance(c, SentenceChunker)

    def test_get_metadata(self):
        c = get_chunker("metadata")
        assert isinstance(c, MetadataChunker)

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            get_chunker("nonexistent")

    def test_batch_chunking(self):
        chunker = get_chunker("sentence")
        records = [make_record(f"Sentence {i}. Another sentence.") for i in range(5)]
        chunks = chunker.chunk_batch(records)
        assert len(chunks) >= 5
