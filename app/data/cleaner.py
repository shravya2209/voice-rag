"""Text cleaning utilities for passage preprocessing."""

from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """Clean and normalize a passage text for embedding quality."""
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove control characters (keep newlines for structure)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Strip leading/trailing
    text = text.strip()
    return text


def is_quality_passage(text: str, min_chars: int = 30, max_chars: int = 10000) -> bool:
    """Filter out too-short, too-long, or garbage passages."""
    if not text:
        return False
    n = len(text)
    if n < min_chars or n > max_chars:
        return False
    # Reject if mostly non-alpha (URLs, code dumps, etc.)
    alpha_ratio = sum(c.isalpha() for c in text) / max(n, 1)
    if alpha_ratio < 0.5:
        return False
    return True
