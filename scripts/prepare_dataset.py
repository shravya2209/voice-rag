"""Prepare dataset — extract, clean, deduplicate, and save passages from parquet."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pyarrow.parquet as pq

from app.config import get_settings
from app.data.cleaner import clean_text, is_quality_passage
from app.data.models import DatasetRecord
from app.utils.logging import get_logger

log = get_logger("prepare_dataset")


def main():
    settings = get_settings()

    # ── Load manifest ──────────────────────────────────────────
    manifest_path = settings.data_dir / "download_manifest.json"
    if not manifest_path.exists():
        print("ERROR: download_manifest.json not found. Run download_dataset.py first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    parquet_path = manifest["parquet_path"]
    print(f"Loading parquet: {parquet_path}")

    # ── Read parquet in streaming batches ───────────────────────
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows_in_file = parquet_file.metadata.num_rows
    target_rows = min(settings.dataset_max_rows, total_rows_in_file)
    print(f"Processing up to {target_rows} rows from {total_rows_in_file} total rows")

    # ── Extract passages ───────────────────────────────────────
    seen_texts: set[str] = set()
    records: list[DatasetRecord] = []
    passage_counter = 0
    rejected = 0
    rows_processed = 0

    batch_size = 500
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["query_id", "passages"]):
        batch_dict = batch.to_pydict()
        query_ids = batch_dict.get("query_id", [])
        passages_list = batch_dict.get("passages", [])
        num_in_batch = len(query_ids)

        for i in range(num_in_batch):
            if rows_processed >= target_rows:
                break
            rows_processed += 1

            query_id = query_ids[i]
            passages = passages_list[i]

            if not isinstance(passages, dict):
                continue

            eng_passages = passages.get("English_passages", [])
            is_selected_list = passages.get("is_selected", [])

            if not eng_passages:
                continue

            for idx, text in enumerate(eng_passages):
                if not text or not isinstance(text, str) or not text.strip():
                    continue

                cleaned = clean_text(text)

                if not is_quality_passage(cleaned):
                    rejected += 1
                    continue

                if cleaned in seen_texts:
                    continue
                seen_texts.add(cleaned)

                selected = bool(is_selected_list[idx]) if idx < len(is_selected_list) else False

                records.append(DatasetRecord(
                    id=f"p_{passage_counter}",
                    text=cleaned,
                    metadata={
                        "query_id": query_id,
                        "is_selected": selected,
                    },
                    language="en",
                    source="msmarco-xi",
                    query_id=query_id,
                    is_selected=selected,
                ))
                passage_counter += 1

        if rows_processed >= target_rows:
            break

    print(f"\nExtracted: {passage_counter} passages")
    print(f"Rejected (low quality): {rejected}")
    print(f"Final unique records: {len(records)}")

    # ── Save processed records ─────────────────────────────────
    out_path = settings.data_dir / "records.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    print(f"\nSaved to: {out_path}")

    # ── Print statistics ───────────────────────────────────────
    text_lengths = [len(r.text) for r in records]
    print(f"\n--- Dataset Statistics ---")
    print(f"Total records: {len(records)}")
    print(f"Avg text length: {sum(text_lengths) / len(text_lengths):.0f} chars")
    print(f"Min text length: {min(text_lengths)} chars")
    print(f"Max text length: {max(text_lengths)} chars")
    print(f"Selected passages: {sum(1 for r in records if r.is_selected)}")
    print(f"Storage: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
