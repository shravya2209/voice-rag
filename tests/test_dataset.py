"""Tests for dataset loading and cleaning."""

import pytest
from app.data.models import DatasetRecord
from app.data.cleaner import clean_text, is_quality_passage


class TestCleaner:
    def test_clean_text_normalizes_whitespace(self):
        assert clean_text("hello   world") == "hello world"

    def test_clean_text_strips(self):
        assert clean_text("  hello  ") == "hello"

    def test_clean_text_empty(self):
        assert clean_text("") == ""

    def test_clean_text_none(self):
        assert clean_text(None) == ""

    def test_clean_text_control_chars(self):
        result = clean_text("hello\x00world")
        assert "\x00" not in result

    def test_quality_passage_valid(self):
        assert is_quality_passage("This is a valid passage with enough text content for the check.") is True

    def test_quality_passage_too_short(self):
        assert is_quality_passage("Hi") is False

    def test_quality_passage_empty(self):
        assert is_quality_passage("") is False

    def test_quality_passage_mostly_numbers(self):
        assert is_quality_passage("12345 67890 12345 67890 12345 67890") is False


class TestDatasetRecord:
    def test_create_record(self):
        r = DatasetRecord(
            id="p_0",
            text="Test passage text",
            language="en",
            source="msmarco-xi",
        )
        assert r.id == "p_0"
        assert r.text == "Test passage text"
        assert r.language == "en"

    def test_record_defaults(self):
        r = DatasetRecord(id="p_1", text="Test")
        assert r.source == "msmarco-xi"
        assert r.metadata == {}
        assert r.is_selected is False

    def test_record_serialization(self):
        r = DatasetRecord(id="p_0", text="Test", metadata={"key": "val"})
        json_str = r.model_dump_json()
        r2 = DatasetRecord.model_validate_json(json_str)
        assert r2.id == r.id
        assert r2.metadata == {"key": "val"}


class TestMalformedRecords:
    def test_empty_text_rejected(self):
        assert is_quality_passage("") is False

    def test_null_handling(self):
        assert clean_text(None) == ""

    def test_whitespace_only(self):
        result = clean_text("   \n\t  ")
        assert result == ""

    def test_unicode_normalization(self):
        # NFKC normalization
        result = clean_text("\uff21\uff22\uff23")
        assert result == "ABC"
