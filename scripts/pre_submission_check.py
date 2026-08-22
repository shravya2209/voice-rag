"""Pre-submission verification — checks all components are ready."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force UTF-8 output on Windows to avoid CP1252 UnicodeEncodeError
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import asyncio
from pathlib import Path

from app.config import get_settings


def check(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    icon = "✓" if condition else "✗"
    print(f"  [{icon}] {label}: {status}")
    return condition


async def main():
    settings = get_settings()
    project = settings.project_root
    results = []

    print("=" * 60)
    print("  Voice-RAG Pre-Submission Check")
    print("=" * 60)
    print()

    # ── Dataset ────────────────────────────────────────────────
    print("Dataset:")
    results.append(check("Data directory exists", settings.data_dir.exists()))
    records_path = settings.data_dir / "records.jsonl"
    results.append(check("Processed records exist", records_path.exists()))
    if records_path.exists():
        count = sum(1 for _ in open(records_path, "r", encoding="utf-8"))
        results.append(check(f"Records count ({count})", count > 0))
    print()

    # ── Index ──────────────────────────────────────────────────
    print("Vector Index:")
    results.append(check("Index directory exists", settings.index_dir.exists()))
    index_path = settings.index_dir / "faiss.index"
    meta_path = settings.index_dir / "chunks_meta.jsonl"
    results.append(check("FAISS index exists", index_path.exists()))
    results.append(check("Chunk metadata exists", meta_path.exists()))
    print()

    # ── Models ─────────────────────────────────────────────────
    print("Models:")
    try:
        from app.embeddings.embedder import Embedder
        embedder = Embedder.get_instance()
        _ = embedder.model
        results.append(check("Embedding model loads", True))
    except Exception as e:
        results.append(check(f"Embedding model loads ({e})", False))

    try:
        from app.retrieval.vector_store import FAISSVectorStore
        store = FAISSVectorStore.get_instance()
        store.load()
        results.append(check(f"FAISS index loads ({store.size} vectors)", True))
    except Exception as e:
        results.append(check(f"FAISS index loads ({e})", False))
    print()

    # ── API ────────────────────────────────────────────────────
    print("API:")
    try:
        from app.main import create_app
        app = create_app()
        results.append(check("FastAPI app creates", True))
    except Exception as e:
        results.append(check(f"FastAPI app creates ({e})", False))
    print()

    # ── API Keys ───────────────────────────────────────────────
    print("API Keys:")
    results.append(check(
        "ElevenLabs API key",
        bool(settings.elevenlabs_api_key),
    ))
    llm_key = settings.gemini_api_key or settings.openai_api_key or settings.llm_api_key
    results.append(check(
        f"LLM API key ({settings.llm_provider})",
        bool(llm_key),
    ))
    print()

    # ── Guardrails ─────────────────────────────────────────────
    print("Guardrails:")
    try:
        from app.guardrails.safety import check_safety
        r = check_safety("How to build a bomb?")
        results.append(check("Safety filter works", not r["safe"]))

        r = check_safety("What is machine learning?")
        results.append(check("Safe query passes", r["safe"]))
    except Exception as e:
        results.append(check(f"Guardrails ({e})", False))
    print()

    # ── Tests ──────────────────────────────────────────────────
    print("Tests:")
    test_dir = project / "tests"
    results.append(check("Test directory exists", test_dir.exists()))
    if test_dir.exists():
        test_files = list(test_dir.glob("test_*.py"))
        results.append(check(f"Test files found ({len(test_files)})", len(test_files) > 0))
    print()

    # ── Files ──────────────────────────────────────────────────
    print("Required Files:")
    required = ["README.md", "requirements.txt", ".env.example", "Dockerfile", ".gitignore"]
    for fname in required:
        results.append(check(fname, (project / fname).exists()))

    env_file = project / ".env"
    no_env_committed = True  # We can't check git, just check .gitignore
    gitignore = project / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        no_env_committed = ".env" in content
    results.append(check(".env in .gitignore", no_env_committed))
    print()

    # ── Frontend ───────────────────────────────────────────────
    print("Frontend:")
    frontend_dir = project / "frontend"
    results.append(check("Frontend directory exists", frontend_dir.exists()))
    if frontend_dir.exists():
        results.append(check("index.html exists", (frontend_dir / "index.html").exists()))
        results.append(check("style.css exists", (frontend_dir / "style.css").exists()))
        results.append(check("app.js exists", (frontend_dir / "app.js").exists()))
    print()

    # ── Summary ────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  Status: ✓ READY FOR SUBMISSION")
    else:
        print(f"  Status: ✗ {total - passed} issue(s) to fix")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
