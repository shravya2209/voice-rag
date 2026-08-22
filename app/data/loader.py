"""Dataset loader — downloads and caches MSMARCO-XI from HuggingFace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

from app.config import get_settings
from app.data.models import Passage
from app.utils.logging import get_logger

log = get_logger("data.loader")


def load_dataset_streaming(max_rows: int | None = None) -> Generator[dict, None, None]:
    """Stream dataset rows from HuggingFace without loading all into memory."""
    from datasets import load_dataset

    settings = get_settings()
    n = max_rows or settings.dataset_max_rows

    log.info(f"Streaming {n} rows from {settings.dataset_name}...")
    ds = load_dataset(
        settings.dataset_name,
        settings.dataset_subset,
        split=settings.dataset_split,
        streaming=True,
    )
    for i, row in enumerate(ds):
        if i >= n:
            break
        yield row
    log.info(f"Streamed {min(i + 1, n)} rows")


def extract_passages(max_rows: int | None = None) -> list[Passage]:
    """Extract unique English passages from the dataset."""
    settings = get_settings()
    cache_path = settings.data_dir / "passages.jsonl"

    # Return cached if available
    if cache_path.exists():
        log.info(f"Loading cached passages from {cache_path}")
        return _load_cached_passages(cache_path)

    log.info("Extracting passages from dataset...")
    seen_texts: set[str] = set()
    passages: list[Passage] = []
    passage_counter = 0

    for row in load_dataset_streaming(max_rows):
        query_id = row.get("query_id")
        passage_data = row.get("passages", {})

        eng_passages = passage_data.get("English_passages", [])
        is_selected_list = passage_data.get("is_selected", [])

        for idx, text in enumerate(eng_passages):
            if not text or not text.strip():
                continue
            clean = text.strip()
            if clean in seen_texts:
                continue
            seen_texts.add(clean)

            selected = bool(is_selected_list[idx]) if idx < len(is_selected_list) else False
            passages.append(Passage(
                passage_id=f"p_{passage_counter}",
                text=clean,
                source_query_id=query_id,
                is_selected=selected,
                language="en",
            ))
            passage_counter += 1

    log.info(f"Extracted {len(passages)} unique passages")

    # Cache to disk
    _save_passages_cache(passages, cache_path)
    return passages


def _save_passages_cache(passages: list[Passage], path: Path) -> None:
    """Save passages as JSONL for fast reloading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(p.model_dump_json() + "\n")
    log.info(f"Cached {len(passages)} passages to {path}")


def _load_cached_passages(path: Path) -> list[Passage]:
    """Load passages from JSONL cache."""
    passages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                passages.append(Passage.model_validate_json(line))
    log.info(f"Loaded {len(passages)} passages from cache")
    return passages
