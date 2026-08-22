"""Data models for the RAG pipeline."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class DatasetRecord(BaseModel):
    """Reusable dataset record model — maps raw HuggingFace rows."""
    id: str
    text: str
    metadata: dict = Field(default_factory=dict)
    language: str = "en"
    source: str = "msmarco-xi"
    query_id: Optional[int] = None
    query_text: Optional[str] = None
    is_selected: bool = False


class Chunk(BaseModel):
    """A chunk produced by any chunking strategy."""
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int = 0
    strategy: str = "fixed"
    source: str = "msmarco-xi"
    language: str = "en"
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk augmented with retrieval scores."""
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0


class RAGResponse(BaseModel):
    """Structured RAG pipeline output."""
    query: str
    answer: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
    grounding_score: float = 0.0
    guardrail_flags: dict = Field(default_factory=dict)
    timings: dict = Field(default_factory=dict)
