"""
Run this first to inspect the LiveCodeBench dataset structure
before kicking off the full pipeline.

Usage:
    python explore_dataset.py [--write-preview]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from config.settings import LCB_DATASET, LCB_VERSION, RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect the LiveCodeBench dataset")
    parser.add_argument(
        "--write-preview",
        action="store_true",
        help="Write the first ten records to RESULTS_DIR/dataset_preview.json",
    )
    return parser.parse_args()


def write_preview(dataset) -> Path:
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_preview.json"
    records = []
    for sample in dataset.select(range(min(10, len(dataset)))):
        records.append({
            key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in sample.items()
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> None:
    args = parse_args()
    dataset = load_dataset(LCB_DATASET, name=LCB_VERSION, split="test")

    print(f"\nTotal samples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}\n")
    if not len(dataset):
        return

    sample = dataset[min(20, len(dataset) - 1)]
    for key, value in sample.items():
        if isinstance(value, (list, dict)):
            preview = json.dumps(value, indent=2)[:300] + "..."
        else:
            preview = str(value)[:300]
        print(f"--- {key} ---")
        print(preview)
        print()

    if args.write_preview:
        out_path = write_preview(dataset)
        print(f"Saved {min(10, len(dataset))} samples -> {out_path}")


if __name__ == "__main__":
    main()
