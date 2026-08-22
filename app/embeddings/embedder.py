"""Embedding abstraction — loads model once, provides embed_text / embed_batch."""

from __future__ import annotations

import hashlib
import numpy as np
from collections import OrderedDict
from typing import Optional

from app.config import get_settings
from app.utils.logging import get_logger
from app.utils.timing import Timer

log = get_logger("embeddings")


class Embedder:
    """Sentence-transformer embedder with LRU cache for latency optimization.

    The model is loaded once on first use and reused for all subsequent calls.
    Embeddings are L2-normalized for cosine similarity via inner product.
    """

    _instance: Optional["Embedder"] = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_max = self.settings.cache_max_size

    @classmethod
    def get_instance(cls) -> "Embedder":
        """Return singleton embedder instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            with Timer("Load embedding model"):
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.settings.embedding_model,
                    device="cpu",
                )
            log.info(
                f"Loaded {self.settings.embedding_model} "
                f"(dim={self.settings.embedding_dim})"
            )
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string. Uses cache if available."""
        if self.settings.cache_embeddings:
            key = self._cache_key(text)
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vec = np.asarray(embedding, dtype=np.float32)

        if self.settings.cache_embeddings:
            self._put_cache(self._cache_key(text), vec)

        return vec

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Embed a batch of texts efficiently. Returns (N, dim) array."""
        bs = batch_size or self.settings.embedding_batch_size
        with Timer(f"Embed batch ({len(texts)} texts)"):
            embeddings = self.model.encode(
                texts,
                batch_size=bs,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,
            )
        result = np.asarray(embeddings, dtype=np.float32)

        # Populate cache
        if self.settings.cache_embeddings:
            for i, text in enumerate(texts):
                self._put_cache(self._cache_key(text), result[i])

        return result

    def warmup(self) -> None:
        """Pre-load model for startup warmup."""
        _ = self.model
        _ = self.embed_text("warmup query")
        log.info("Embedder warmed up")

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _put_cache(self, key: str, value: np.ndarray) -> None:
        self._cache[key] = value
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
