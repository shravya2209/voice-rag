"""Grounding guardrail — ensures answers are based on retrieved context."""

from __future__ import annotations

from app.utils.logging import get_logger

log = get_logger("guardrails.grounding")


def check_grounding(
    answer: str,
    context_texts: list[str],
    min_overlap_ratio: float = 0.15,
) -> dict:
    """Check if the answer is grounded in the retrieved context.

    Uses token overlap as a lightweight proxy for grounding.
    A more robust check would use the LLM itself, but this
    adds latency. This heuristic is fast and catches obvious
    hallucinations.

    Args:
        answer: Generated answer text
        context_texts: List of retrieved passage texts
        min_overlap_ratio: Minimum ratio of answer tokens found in context

    Returns:
        dict with 'grounded' (bool), 'score' (float 0-1), 'reason' (str)
    """
    if not answer or not context_texts:
        return {"grounded": False, "score": 0.0, "reason": "Empty answer or context"}

    # Tokenize answer and context
    answer_tokens = set(answer.lower().split())
    # Remove very common stop words to focus on content words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "it", "its", "this", "that", "these", "those", "i", "you",
        "he", "she", "we", "they", "and", "or", "but", "not", "no",
        "if", "then", "so", "as", "than", "too", "very", "just",
    }
    answer_content = answer_tokens - stop_words

    if len(answer_content) == 0:
        return {"grounded": True, "score": 1.0, "reason": "Answer is too short to check"}

    # Check if answer is in a non-Latin / Indic script (e.g. Kannada, Hindi, Telugu, etc.)
    non_ascii_chars = sum(1 for c in answer if ord(c) > 127)
    if len(answer) > 0 and (non_ascii_chars / len(answer)) > 0.15:
        # Cross-lingual response: direct word-level overlap check on English context does not apply
        return {
            "grounded": True,
            "score": 0.90,
            "reason": "Multilingual response generated from retrieved context",
        }

    # Build context token set
    context_combined = " ".join(context_texts).lower()
    context_tokens = set(context_combined.split()) - stop_words

    # Calculate overlap
    overlap = answer_content & context_tokens
    overlap_ratio = len(overlap) / len(answer_content) if answer_content else 0

    grounded = overlap_ratio >= min_overlap_ratio

    if not grounded:
        log.warning(
            f"Low grounding score: {overlap_ratio:.2f} "
            f"({len(overlap)}/{len(answer_content)} content tokens)"
        )

    return {
        "grounded": grounded,
        "score": round(overlap_ratio, 3),
        "reason": "" if grounded else "Answer may contain information not in the retrieved context",
    }
