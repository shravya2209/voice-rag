"""Download MSMARCO-XI dataset — uses HuggingFace Hub parquet files directly (no torch needed)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import json

from app.config import get_settings


def main():
    settings = get_settings()
    print(f"Dataset: {settings.dataset_name}")
    print(f"Max rows: {settings.dataset_max_rows}")
    print()

    # ── Download first parquet shard ───────────────────────────
    print("Discovering parquet files...")
    api = HfApi()
    files = api.list_repo_files(
        settings.dataset_name,
        repo_type="dataset",
        revision="refs/convert/parquet",
    )

    parquet_files = [f for f in files if f.endswith(".parquet") and "/train/" in f]
    print(f"Found {len(parquet_files)} train parquet shards")

    if not parquet_files:
        print("ERROR: No parquet files found!")
        sys.exit(1)

    # Download first shard (enough for 5000 rows)
    shard = parquet_files[0]
    print(f"\nDownloading: {shard}...")
    local_path = hf_hub_download(
        settings.dataset_name,
        filename=shard,
        repo_type="dataset",
        revision="refs/convert/parquet",
        cache_dir=str(settings.data_dir / ".cache"),
    )
    print(f"Downloaded to: {local_path}")

    # ── Inspect schema with pyarrow (memory-efficient) ─────────
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(local_path)
    print(f"\n--- Schema Inspection ---")
    print(f"Total rows in shard: {parquet_file.metadata.num_rows}")
    print(f"Total row groups: {parquet_file.metadata.num_row_groups}")
    print(f"Columns: {parquet_file.schema.names}")
    print()

    # Read just the first small batch (e.g. 5 rows) for preview
    first_batch = next(parquet_file.iter_batches(batch_size=5))
    table_sample = first_batch.to_pydict()

    # ── Preview first few rows ─────────────────────────────────
    print("--- First 5 rows preview ---")
    num_sample = len(table_sample.get("query_id", []))
    for i in range(num_sample):
        query_id = table_sample.get("query_id", ["?"])[i]
        eng_query = str(table_sample.get("Eng_Query", [""])[i])[:60]
        passages = table_sample.get("passages", [{}])[i]
        eng_passages = passages.get("English_passages", []) if isinstance(passages, dict) else []
        print(f"  Row {i}: query_id={query_id}, eng_passages={len(eng_passages)}, query='{eng_query}'")

    # ── Save raw parquet path for prepare_dataset ──────────────
    manifest = {"parquet_path": local_path, "total_rows": parquet_file.metadata.num_rows}
    manifest_path = settings.data_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to: {manifest_path}")
    print("Run 'python scripts/prepare_dataset.py' next.")


if __name__ == "__main__":
    main()

