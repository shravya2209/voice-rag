"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class TextQueryRequest(BaseModel):
    """POST /api/query"""
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = None
    retrieval_mode: Optional[str] = None


class TranscribeRequest(BaseModel):
    """Metadata for POST /api/transcribe (audio in form-data)."""
    language: str = "en"


class SourceDisplay(BaseModel):
    """A source passage for UI display."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    strategy: str = ""
    language: str = "en"


class LatencyBreakdown(BaseModel):
    """Latency timing breakdown."""
    stt: Optional[float] = None
    embedding: Optional[float] = None
    retrieval: Optional[float] = None
    reranking: Optional[float] = None
    generation: Optional[float] = None
    total: Optional[float] = None


class QueryResponse(BaseModel):
    """Structured JSON response for all query endpoints."""
    transcript: Optional[str] = None
    answer: str
    sources: list[SourceDisplay] = Field(default_factory=list)
    latency_ms: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    grounding_score: float = 0.0
    guardrail_flags: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """GET /api/health"""
    status: str
    index_loaded: bool
    index_size: int
    embedding_model: str
    llm_provider: str


class MetricsResponse(BaseModel):
    """GET /api/metrics"""
    total_queries: int = 0
    total_voice_queries: int = 0
    total_text_queries: int = 0
    total_errors: int = 0
    guardrail_blocks: int = 0
    avg_latency_ms: float = 0.0
    uptime_seconds: float = 0.0


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
