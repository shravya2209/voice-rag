"""Relevance guardrail — detects off-topic queries and low-confidence retrieval."""

from __future__ import annotations

from app.config import get_settings
from app.data.models import RetrievedChunk
from app.utils.logging import get_logger

log = get_logger("guardrails.relevance")

INSUFFICIENT_CONTEXT_MSG = (
    "I don't have enough relevant information in the provided "
    "knowledge base to answer that."
)


def check_retrieval_confidence(
    chunks: list[RetrievedChunk],
    threshold: float | None = None,
) -> dict:
    """Check if retrieved chunks have sufficient relevance scores.

    Args:
        chunks: Retrieved chunks with scores
        threshold: Minimum score threshold (uses config default)

    Returns:
        dict with 'confident' (bool), 'max_score', 'avg_score', 'reason'
    """
    settings = get_settings()
    min_score = threshold or settings.min_relevance_score

    if not chunks:
        return {
            "confident": False,
            "max_score": 0.0,
            "avg_score": 0.0,
            "reason": "No relevant passages found",
        }

    scores = [_get_best_score(c) for c in chunks]
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)

    if max_score < min_score:
        log.info(
            f"Low confidence: max_score={max_score:.3f} < threshold={min_score}"
        )
        return {
            "confident": False,
            "max_score": max_score,
            "avg_score": avg_score,
            "reason": INSUFFICIENT_CONTEXT_MSG,
        }

    return {
        "confident": True,
        "max_score": max_score,
        "avg_score": avg_score,
        "reason": "",
    }


def check_query_relevance(query: str) -> dict:
    """Basic check for meaningless or too-short queries.

    Returns:
        dict with 'relevant' (bool) and 'reason' (str)
    """
    if not query or len(query.strip()) < 3:
        return {"relevant": False, "reason": "Query is too short or empty"}

    # Check for nonsense input (all same character, etc.)
    unique_chars = set(query.lower().replace(" ", ""))
    if len(unique_chars) < 3:
        return {"relevant": False, "reason": "Query appears to be meaningless"}

    return {"relevant": True, "reason": ""}


def _get_best_score(chunk: RetrievedChunk) -> float:
    """Get the best available score for a chunk."""
    if chunk.rerank_score > 0:
        return chunk.rerank_score
    if chunk.dense_score > 0:
        return chunk.dense_score
    if chunk.fused_score > 0:
        return chunk.fused_score
    return chunk.bm25_score
