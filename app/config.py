"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings — all configurable via .env or environment variables."""

    # ── Paths ──────────────────────────────────────────────────────
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data_dir: Optional[Path] = Field(default=None)
    index_dir: Optional[Path] = Field(default=None)

    # ── Dataset ────────────────────────────────────────────────────
    dataset_name: str = "ai4bharat/MSMARCO-XI"
    dataset_subset: str = "default"
    dataset_split: str = "train"
    dataset_max_rows: int = 5000

    # ── Embeddings ─────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 256

    # ── Chunking ───────────────────────────────────────────────────
    chunking_strategy: str = "sentence"  # fixed | sentence | semantic | metadata
    chunk_size: int = 256
    chunk_overlap: int = 50
    semantic_similarity_threshold: float = 0.75

    # ── Retrieval ──────────────────────────────────────────────────
    retrieval_mode: str = "hybrid"  # dense | bm25 | hybrid
    top_k: int = 5
    retrieval_candidates: int = 20  # pre-rerank pool
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    min_relevance_score: float = 0.25

    # ── Reranker ───────────────────────────────────────────────────
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── LLM ────────────────────────────────────────────────────────
    llm_provider: str = "gemini"  # gemini | openai
    llm_api_key: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_model: str = ""
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024
    llm_timeout: int = 15

    # ── ElevenLabs ─────────────────────────────────────────────────
    elevenlabs_api_key: str = ""
    elevenlabs_stt_model: str = "scribe_v1"
    elevenlabs_tts_voice: str = "JBFqnCBsd6RMkjVDRZzb"
    elevenlabs_tts_model: str = "eleven_flash_v2_5"
    elevenlabs_timeout: int = 30
    max_audio_size_mb: int = 25

    # ── Guardrails ─────────────────────────────────────────────────
    guardrails_enabled: bool = True
    offtopic_threshold: float = 0.30
    safety_enabled: bool = True

    # ── Caching ────────────────────────────────────────────────────
    cache_embeddings: bool = True
    cache_max_size: int = 1000

    # ── Server ─────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    max_request_size_mb: int = 30

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def model_post_init(self, __context) -> None:
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.index_dir is None:
            self.index_dir = self.project_root / "indexes"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        # Resolve LLM key/model aliases
        if not self.llm_api_key:
            if self.llm_provider == "gemini":
                self.llm_api_key = self.gemini_api_key
            elif self.llm_provider == "openai":
                self.llm_api_key = self.openai_api_key
        if not self.llm_model:
            if self.llm_provider == "gemini":
                self.llm_model = self.gemini_model
            elif self.llm_provider == "openai":
                self.llm_model = self.openai_model


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
