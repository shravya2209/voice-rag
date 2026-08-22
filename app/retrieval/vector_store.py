"""FAISS vector store — build offline, load once at startup."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.data.models import Chunk
from app.utils.logging import get_logger
from app.utils.timing import Timer

log = get_logger("retrieval.vector_store")

INDEX_FILENAME = "faiss.index"
META_FILENAME = "chunks_meta.jsonl"


class FAISSVectorStore:
    """FAISS-backed vector store with metadata sidecar.

    Index is built offline and saved to disk.
    At runtime, it is loaded once and used for all queries.
    """

    _instance: Optional["FAISSVectorStore"] = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self._index = None
        self._chunks: list[Chunk] = []
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "FAISSVectorStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def index_path(self) -> Path:
        return self.settings.index_dir / INDEX_FILENAME

    @property
    def meta_path(self) -> Path:
        return self.settings.index_dir / META_FILENAME

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._index is not None

    @property
    def size(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    def build_index(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        """Build FAISS index from chunks and their embeddings.

        Args:
            chunks: List of Chunk objects
            embeddings: (N, dim) float32 array, L2-normalized
        """
        import faiss

        assert len(chunks) == embeddings.shape[0], (
            f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
        )

        dim = embeddings.shape[1]
        log.info(f"Building FAISS index: {len(chunks)} vectors, dim={dim}")

        with Timer("FAISS index build"):
            # Use IndexFlatIP for exact cosine similarity (embeddings are normalized)
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings.astype(np.float32))

        self._index = index
        self._chunks = chunks
        self._loaded = True
        log.info(f"Index built: {index.ntotal} vectors")

    def save(self) -> None:
        """Persist index and metadata to disk."""
        import faiss

        if self._index is None:
            raise RuntimeError("No index to save — build first")

        self.settings.index_dir.mkdir(parents=True, exist_ok=True)

        with Timer("Save FAISS index"):
            faiss.write_index(self._index, str(self.index_path))

        with Timer("Save chunk metadata"):
            with open(self.meta_path, "w", encoding="utf-8") as f:
                for chunk in self._chunks:
                    f.write(chunk.model_dump_json() + "\n")

        log.info(
            f"Saved index ({self.index_path}) and "
            f"metadata ({self.meta_path})"
        )

    def load(self) -> None:
        """Load pre-built index and metadata from disk."""
        import faiss

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata not found at {self.meta_path}")

        with Timer("Load FAISS index"):
            self._index = faiss.read_index(str(self.index_path))

        with Timer("Load chunk metadata"):
            self._chunks = []
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self._chunks.append(Chunk.model_validate_json(line))

        self._loaded = True
        log.info(
            f"Loaded index with {self._index.ntotal} vectors, "
            f"{len(self._chunks)} chunks"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for nearest neighbors.

        Args:
            query_embedding: (dim,) float32 vector, L2-normalized
            top_k: Number of results to return

        Returns:
            List of (Chunk, score) tuples, sorted by score descending
        """
        if not self.is_loaded:
            raise RuntimeError("Index not loaded — call load() first")

        k = top_k or self.settings.top_k
        k = min(k, self._index.ntotal)

        query = query_embedding.reshape(1, -1).astype(np.float32)

        with Timer(f"FAISS search (k={k})"):
            scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((self._chunks[idx], float(score)))

        return results

    def get_all_texts(self) -> list[str]:
        """Return all chunk texts (for BM25 index building)."""
        return [c.text for c in self._chunks]

    def get_all_chunks(self) -> list[Chunk]:
        """Return all chunk objects."""
        return list(self._chunks)
