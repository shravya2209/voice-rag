"""Safety guardrail — detects and rejects unsafe/inappropriate input."""

from __future__ import annotations

import re

from app.utils.logging import get_logger

log = get_logger("guardrails.safety")

# Patterns for detecting clearly unsafe requests
_UNSAFE_PATTERNS = [
    r"\b(how to|ways to|help me|teach me|show me)\b.*\b(hack|exploit|attack|phish)\b",
    r"\b(hack|exploit|attack)\b.*\b(into|system|server|computer|network)\b",
    r"\b(how to|ways to)\b.*\b(harm|kill|injure|poison|bomb|hurt)\b",
    r"\b(make|create|build|construct)\b.*\b(weapon|bomb|explosive|drug)\b",
    r"\b(steal|forge|counterfeit)\b.*\b(identity|credit card|passport)\b",
    r"\b(malware|ransomware|trojan|keylogger|spyware)\b.*\b(create|make|build|deploy|install)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _UNSAFE_PATTERNS]


def check_safety(query: str) -> dict:
    """Check if a query is safe to process.

    Returns:
        dict with 'safe' (bool) and 'reason' (str if unsafe)
    """
    if not query or not query.strip():
        return {"safe": False, "reason": "Empty query"}

    query_lower = query.lower().strip()

    # Check against unsafe patterns
    for pattern in _COMPILED:
        if pattern.search(query_lower):
            log.warning(f"Unsafe query detected: '{query[:50]}...'")
            return {
                "safe": False,
                "reason": "This query appears to request potentially harmful information.",
            }

    return {"safe": True, "reason": ""}
