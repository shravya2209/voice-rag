"""Build FAISS index from prepared dataset records."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import json
from pathlib import Path

from app.config import get_settings
from app.data.models import DatasetRecord
from app.chunking import get_chunker
from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import FAISSVectorStore
from app.utils.logging import get_logger

log = get_logger("build_index")


def main():
    settings = get_settings()
    t_total = time.perf_counter()

    # ── Load records ───────────────────────────────────────────
    records_path = settings.data_dir / "records.jsonl"
    if not records_path.exists():
        print(f"ERROR: {records_path} not found. Run prepare_dataset.py first.")
        sys.exit(1)

    records: list[DatasetRecord] = []
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(DatasetRecord.model_validate_json(line))
    print(f"Loaded {len(records)} records")

    # ── Chunk ──────────────────────────────────────────────────
    strategy = settings.chunking_strategy
    print(f"\nChunking with strategy: {strategy}")
    t0 = time.perf_counter()
    chunker = get_chunker(strategy)
    chunks = chunker.chunk_batch(records)
    chunk_time = (time.perf_counter() - t0) * 1000
    print(f"Produced {len(chunks)} chunks in {chunk_time:.0f}ms")

    # Chunk stats
    chunk_lengths = [len(c.text) for c in chunks]
    print(f"  Avg chunk size: {sum(chunk_lengths)/len(chunk_lengths):.0f} chars")
    print(f"  Min chunk size: {min(chunk_lengths)} chars")
    print(f"  Max chunk size: {max(chunk_lengths)} chars")

    # ── Embed ──────────────────────────────────────────────────
    print(f"\nEmbedding {len(chunks)} chunks...")
    t0 = time.perf_counter()
    embedder = Embedder.get_instance()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_batch(texts)
    embed_time = (time.perf_counter() - t0) * 1000
    print(f"Embedded in {embed_time:.0f}ms ({embed_time/len(chunks):.1f}ms/chunk)")
    print(f"Embeddings shape: {embeddings.shape}")

    # ── Build FAISS index ──────────────────────────────────────
    print(f"\nBuilding FAISS index...")
    t0 = time.perf_counter()
    store = FAISSVectorStore.get_instance()
    store.build_index(chunks, embeddings)
    build_time = (time.perf_counter() - t0) * 1000
    print(f"Index built in {build_time:.0f}ms")

    # ── Save ───────────────────────────────────────────────────
    store.save()
    total_time = (time.perf_counter() - t_total) * 1000

    # ── Summary ────────────────────────────────────────────────
    index_size = store.index_path.stat().st_size / 1024 / 1024
    meta_size = store.meta_path.stat().st_size / 1024 / 1024

    print(f"\n--- Index Build Summary ---")
    print(f"Strategy:        {strategy}")
    print(f"Records:         {len(records)}")
    print(f"Chunks:          {len(chunks)}")
    print(f"Embedding dim:   {embeddings.shape[1]}")
    print(f"Index size:      {index_size:.2f} MB")
    print(f"Metadata size:   {meta_size:.2f} MB")
    print(f"Chunking time:   {chunk_time:.0f}ms")
    print(f"Embedding time:  {embed_time:.0f}ms")
    print(f"Build time:      {build_time:.0f}ms")
    print(f"Total time:      {total_time:.0f}ms")
    print(f"\nIndex saved to: {settings.index_dir}")


if __name__ == "__main__":
    main()
