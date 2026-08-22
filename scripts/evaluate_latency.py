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
from app.evaluation.latency import LatencyTracker
from app.utils.logging import get_logger

log = get_logger("evaluate_latency")


def load_questions() -> list[str]:
    """Load evaluation questions."""
    eval_path = Path(__file__).parent.parent / "evaluation" / "questions.json"
    if eval_path.exists():
        with open(eval_path, "r") as f:
            data = json.load(f)
        return [q["query"] for q in data["questions"]]

    # Fallback queries if evaluation set doesn't exist
    return [
        "What is machine learning?",
        "How does climate change affect oceans?",
        "What are the benefits of exercise?",
        "How do vaccines work?",
        "What is photosynthesis?",
        "What causes earthquakes?",
        "How does the internet work?",
        "What is artificial intelligence?",
        "What are renewable energy sources?",
        "How does the human immune system work?",
        "What is quantum computing?",
        "How do antibiotics work?",
        "What is blockchain technology?",
        "What causes inflation?",
        "How does DNA replication work?",
    ]


async def main():
    settings = get_settings()
    tracker = LatencyTracker()

    print("=" * 60)
    print("Voice-RAG Latency Evaluation")
    print("=" * 60)
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Retrieval mode:  {settings.retrieval_mode}")
    print(f"Top-K:           {settings.top_k}")
    print(f"Reranker:        {'ON' if settings.use_reranker else 'OFF'}")
    print()

    # ── Setup ──────────────────────────────────────────────────
    print("Loading models and index...")
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

    questions = load_questions()
    print(f"Running {len(questions)} test queries...\n")

    # ── Run queries ────────────────────────────────────────────
    for i, query in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {query[:60]}...", end=" ")
        try:
            result = await pipeline.process_text_query(query)
            tracker.record_from_timings(result.timings)
            total = result.timings.get("total", 0)
            print(f"✓ {total:.0f}ms")
        except Exception as e:
            print(f"✗ Error: {e}")

    # ── Report ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LATENCY RESULTS")
    print("=" * 60)
    print(tracker.print_table())

    # ── Check target ───────────────────────────────────────────
    stats = tracker.get_stats()
    if "total" in stats:
        total_p50 = stats["total"].p50
        total_p100 = stats["total"].p100
        print(f"\n{'Target':>15}: <200ms (retrieval + generation)")
        print(f"{'Measured P50':>15}: {total_p50:.1f}ms")
        print(f"{'Measured P100':>15}: {total_p100:.1f}ms")

        # Identify bottleneck
        print(f"\nBottleneck analysis:")
        for name, s in sorted(stats.items(), key=lambda x: x[1].p50, reverse=True):
            pct = (s.p50 / total_p50 * 100) if total_p50 > 0 else 0
            print(f"  {name:<15} {s.p50:>7.1f}ms ({pct:>5.1f}%)")

    # Save results
    results_path = Path(__file__).parent.parent / "evaluation" / "latency_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(tracker.to_dict_list(), f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
