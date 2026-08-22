"""Demo mode — runs predefined queries and shows results without needing a microphone."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force UTF-8 on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import asyncio
import json
from pathlib import Path

from app.config import get_settings
from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.hybrid import HybridRetriever
from app.orchestration.pipeline import RAGPipeline
from app.utils.logging import get_logger

log = get_logger("demo")


DEMO_QUERIES = [
    "What is the purpose of the immune system?",
    "How does solar energy work?",
    "What are the health benefits of drinking water?",
    "What causes climate change?",
    "How do computers process information?",
]


async def main():
    settings = get_settings()

    print("=" * 70)
    print("  Voice-RAG Demo Mode")
    print("=" * 70)
    print()

    # ── Setup ──────────────────────────────────────────────────
    embedder = Embedder.get_instance()
    embedder.warmup()

    store = FAISSVectorStore.get_instance()
    try:
        store.load()
    except FileNotFoundError:
        print("ERROR: FAISS index not found. Run build_index.py first.")
        sys.exit(1)

    retriever = HybridRetriever(vector_store=store, embedder=embedder)
    retriever.ensure_bm25()

    pipeline = RAGPipeline()
    print(f"Index loaded: {store.size} vectors\n")

    # ── Load custom questions if available ─────────────────────
    eval_path = Path(__file__).parent.parent / "evaluation" / "questions.json"
    if eval_path.exists():
        with open(eval_path) as f:
            data = json.load(f)
        queries = [q["query"] for q in data["questions"][:5]]
    else:
        queries = DEMO_QUERIES

    # ── Run queries ────────────────────────────────────
    for i, query in enumerate(queries, 1):
        print(f"{'-' * 70}")
        print(f"Query {i}: {query}")
        print(f"{'-' * 70}")

        result = await pipeline.process_text_query(query)

        print(f"\nAnswer: {result.answer}\n")

        if result.sources:
            print(f"Retrieved Sources ({len(result.sources)}):")
            for j, rc in enumerate(result.sources[:3], 1):
                score = rc.rerank_score if rc.rerank_score > 0 else rc.fused_score
                print(f"  {j}. [{rc.chunk.chunk_id}] (score: {score:.4f})")
                print(f"     {rc.chunk.text[:120]}...")
            print()

        print(f"Latency:")
        for component, ms in result.timings.items():
            print(f"  {component:<15} {ms:>7.1f}ms")
        print()

    print("=" * 70)
    print("  Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
