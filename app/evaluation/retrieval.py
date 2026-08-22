"""Retrieval evaluation metrics."""

from __future__ import annotations

from app.data.models import RetrievedChunk


def precision_at_k(retrieved: list[RetrievedChunk], relevant_ids: set[str], k: int) -> float:
    """Calculate precision@k."""
    if not retrieved or k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for rc in top_k if rc.chunk.document_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved: list[RetrievedChunk], relevant_ids: set[str], k: int) -> float:
    """Calculate recall@k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for rc in top_k if rc.chunk.document_id in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved: list[RetrievedChunk], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank."""
    for i, rc in enumerate(retrieved, 1):
        if rc.chunk.document_id in relevant_ids:
            return 1.0 / i
    return 0.0


def average_score(retrieved: list[RetrievedChunk]) -> float:
    """Average relevance score of retrieved chunks."""
    if not retrieved:
        return 0.0
    scores = []
    for rc in retrieved:
        s = rc.rerank_score if rc.rerank_score > 0 else rc.fused_score
        if s <= 0:
            s = rc.dense_score
        scores.append(s)
    return sum(scores) / len(scores) if scores else 0.0
