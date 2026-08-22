"""FastAPI application — entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.routes import router
from app.config import get_settings
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.hybrid import HybridRetriever
from app.embeddings.embedder import Embedder
from app.utils.logging import get_logger

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models and index once. Shutdown: cleanup."""
    settings = get_settings()
    log.info("=" * 60)
    log.info("Voice-RAG starting up...")
    log.info(f"  Embedding model : {settings.embedding_model}")
    log.info(f"  LLM provider    : {settings.llm_provider}")
    log.info(f"  Retrieval mode  : {settings.retrieval_mode}")
    log.info(f"  Chunking strat. : {settings.chunking_strategy}")
    log.info(f"  Reranker        : {'ON' if settings.use_reranker else 'OFF'}")
    log.info(f"  Top-K           : {settings.top_k}")
    log.info("=" * 60)

    # ── Load embedding model (warmup) ──────────────────────────
    embedder = Embedder.get_instance()
    embedder.warmup()

    # ── Load FAISS index ───────────────────────────────────────
    store = FAISSVectorStore.get_instance()
    try:
        store.load()
        log.info(f"FAISS index loaded: {store.size} vectors")
    except FileNotFoundError:
        log.warning(
            "FAISS index not found — run 'python scripts/build_index.py' first. "
            "API will return errors for queries until index is built."
        )

    # ── Build BM25 index if hybrid mode ────────────────────────
    if settings.retrieval_mode in ("hybrid", "bm25") and store.is_loaded:
        retriever = HybridRetriever(vector_store=store, embedder=embedder)
        retriever.ensure_bm25()

    log.info("Startup complete!")
    yield
    log.info("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Voice-RAG Assistant",
        description="Voice-enabled Retrieval-Augmented Generation system",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(router)

    # Serve frontend static files
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/")
        async def serve_frontend():
            return FileResponse(str(frontend_dir / "index.html"))

    return app


app = create_app()
