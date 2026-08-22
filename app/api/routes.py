"""FastAPI route definitions."""

from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.api.schemas import (
    TextQueryRequest,
    QueryResponse,
    SourceDisplay,
    LatencyBreakdown,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
)
from app.config import get_settings
from app.orchestration.pipeline import RAGPipeline
from app.retrieval.vector_store import FAISSVectorStore
from app.evaluation.metrics import metrics
from app.utils.logging import get_logger

log = get_logger("api.routes")

router = APIRouter()

# Lazy-initialized pipeline
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@router.post("/api/query", response_model=QueryResponse)
async def text_query(req: TextQueryRequest):
    """Process a text query through the RAG pipeline."""
    try:
        pipeline = get_pipeline()
        result = await pipeline.process_text_query(req.query)

        sources = [
            SourceDisplay(
                chunk_id=rc.chunk.chunk_id,
                document_id=rc.chunk.document_id,
                text=rc.chunk.text[:500],
                score=round(rc.rerank_score if rc.rerank_score > 0 else rc.fused_score, 4),
                strategy=rc.chunk.strategy,
                language=rc.chunk.language,
            )
            for rc in result.sources
        ]

        latency = LatencyBreakdown(**{
            k: round(v, 2) for k, v in result.timings.items()
        })

        metrics.record_query(result.timings.get("total", 0), is_voice=False)

        return QueryResponse(
            answer=result.answer,
            sources=sources,
            latency_ms=latency,
            grounding_score=result.grounding_score,
            guardrail_flags=result.guardrail_flags,
        )
    except Exception as e:
        metrics.record_error()
        log.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/transcribe", response_model=QueryResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("auto"),
):
    """Transcribe audio without running the RAG pipeline."""
    settings = get_settings()
    try:
        audio_data = await file.read()

        max_bytes = settings.max_audio_size_mb * 1024 * 1024
        if len(audio_data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large (max {settings.max_audio_size_mb}MB)",
            )

        pipeline = get_pipeline()
        transcript = await pipeline.stt.transcribe(
            audio_data,
            filename=file.filename or "audio.webm",
            language=language,
        )

        return QueryResponse(
            transcript=transcript,
            answer="",
            sources=[],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        metrics.record_error()
        log.error(f"Transcribe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/voice-query", response_model=QueryResponse)
async def voice_query(
    file: UploadFile = File(...),
    language: str = Form("auto"),
):
    """Full voice pipeline: audio → STT → RAG → answer."""
    settings = get_settings()
    try:
        audio_data = await file.read()

        max_bytes = settings.max_audio_size_mb * 1024 * 1024
        if len(audio_data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large (max {settings.max_audio_size_mb}MB)",
            )

        pipeline = get_pipeline()
        transcript, result = await pipeline.process_voice_query(
            audio_data,
            filename=file.filename or "audio.webm",
            language=language,
        )

        sources = [
            SourceDisplay(
                chunk_id=rc.chunk.chunk_id,
                document_id=rc.chunk.document_id,
                text=rc.chunk.text[:500],
                score=round(rc.rerank_score if rc.rerank_score > 0 else rc.fused_score, 4),
                strategy=rc.chunk.strategy,
                language=rc.chunk.language,
            )
            for rc in result.sources
        ]

        latency = LatencyBreakdown(**{
            k: round(v, 2) for k, v in result.timings.items()
        })

        metrics.record_query(result.timings.get("total", 0), is_voice=True)

        return QueryResponse(
            transcript=transcript,
            answer=result.answer,
            sources=sources,
            latency_ms=latency,
            grounding_score=result.grounding_score,
            guardrail_flags=result.guardrail_flags,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        metrics.record_error()
        log.error(f"Voice query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    store = FAISSVectorStore.get_instance()
    return HealthResponse(
        status="ok" if store.is_loaded else "degraded",
        index_loaded=store.is_loaded,
        index_size=store.size,
        embedding_model=settings.embedding_model,
        llm_provider=settings.llm_provider,
    )


@router.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Application metrics endpoint."""
    return MetricsResponse(**metrics.to_dict())
