"""Embedding abstraction — lazy-loads model with ONNX/FastEmbed and SentenceTransformer fallback."""

from __future__ import annotations

import hashlib
import gc
import numpy as np
from collections import OrderedDict
from typing import Optional

from app.config import get_settings
from app.utils.logging import get_logger
from app.utils.timing import Timer

log = get_logger("embeddings")


class Embedder:
    """Sentence-transformer embedder with LRU cache and low-memory execution.

    Supports FastEmbed (ONNX runtime for ~40MB RAM) with SentenceTransformer fallback.
    Embeddings are L2-normalized for cosine similarity via inner product.
    """

    _instance: Optional["Embedder"] = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._model_type = None  # 'fastembed' | 'sentence_transformers'
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_max = self.settings.cache_max_size

    @classmethod
    def get_instance(cls) -> "Embedder":
        """Return singleton embedder instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_model(self) -> None:
        """Lazy-load the embedding model with low-memory configuration."""
        if self._model is not None:
            return

        with Timer("Load embedding model"):
            # Try fastembed first (ONNX runtime, ~40MB RAM on Linux)
            try:
                from fastembed import TextEmbedding
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self._model = TextEmbedding(model_name=model_name)
                self._model_type = "fastembed"
                log.info(
                    f"Loaded FastEmbed ONNX model {model_name} "
                    f"(low-memory mode, dim={self.settings.embedding_dim})"
                )
                return
            except Exception as e:
                log.info(f"FastEmbed initialization note: {e}, using SentenceTransformer...")

            # Fallback to SentenceTransformer with single-thread CPU constraints
            try:
                import torch
                torch.set_num_threads(1)
                if hasattr(torch, "set_num_interop_threads"):
                    try:
                        torch.set_num_interop_threads(1)
                    except RuntimeError:
                        pass
            except ImportError:
                pass

            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.settings.embedding_model,
                device="cpu",
            )
            self._model_type = "sentence_transformers"
            log.info(
                f"Loaded SentenceTransformer {self.settings.embedding_model} "
                f"(dim={self.settings.embedding_dim})"
            )

    @property
    def model(self):
        if self._model is None:
            self._init_model()
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string. Uses cache if available."""
        if self.settings.cache_embeddings:
            key = self._cache_key(text)
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        if self._model is None:
            self._init_model()

        if self._model_type == "fastembed":
            embeddings = list(self.model.embed([text]))
            vec = np.asarray(embeddings[0], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        else:
            try:
                import torch
                with torch.inference_mode():
                    embedding = self.model.encode(
                        text,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
            except Exception:
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
        if self._model is None:
            self._init_model()

        bs = batch_size or self.settings.embedding_batch_size
        with Timer(f"Embed batch ({len(texts)} texts)"):
            if self._model_type == "fastembed":
                embeddings = list(self.model.embed(texts, batch_size=bs))
                result = np.asarray(embeddings, dtype=np.float32)
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                result = result / norms
            else:
                try:
                    import torch
                    with torch.inference_mode():
                        embeddings = self.model.encode(
                            texts,
                            batch_size=bs,
                            normalize_embeddings=True,
                            show_progress_bar=len(texts) > 100,
                        )
                except Exception:
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

        gc.collect()
        return result

    def warmup(self) -> None:
        """Optional embedder warmup."""
        _ = self.model
        _ = self.embed_text("warmup query")
        log.info("Embedder warmed up")

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _put_cache(self, key: str, value: np.ndarray) -> None:
        self._cache[key] = value
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

