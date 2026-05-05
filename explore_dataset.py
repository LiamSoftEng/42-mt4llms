"""
Run this first to inspect the LiveCodeBench dataset structure
before kicking off the full pipeline.

Usage:
    python explore_dataset.py
"""

from datasets import load_dataset
from config.settings import LCB_DATASET, LCB_VERSION
import json
from datetime import datetime
from pathlib import Path


ds = load_dataset(
    LCB_DATASET,
    name=LCB_VERSION,
    split="test",
)

print(f"\nTotal samples: {len(ds)}")
print(f"Columns: {ds.column_names}\n")

# Show one sample in detail
sample = ds[20]
for key, value in sample.items():
    if isinstance(value, (list, dict)):
        preview = json.dumps(value, indent=2)[:300] + "..."
    else:
        preview = str(value)[:300]
    print(f"--- {key} ---")
    print(preview)
    print()

# Write first 10 samples to results/dataset_preview.json
Path("results").mkdir(exist_ok=True)
out_path = Path("results") / "dataset_preview.json"
 
records = []
for sample in ds.select(range(10)):
    record = {}
    for key, value in sample.items():
        if isinstance(value, datetime):
            record[key] = value.isoformat()
        else:
            record[key] = value
    records.append(record)
 
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
 
print(f"Saved {len(records)} samples -> {out_path}")
