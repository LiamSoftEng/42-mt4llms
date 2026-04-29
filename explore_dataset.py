"""
Run this first to inspect the LiveCodeBench dataset structure
before kicking off the full pipeline.

Usage:
    python explore_dataset.py
"""

from datasets import load_dataset
import json

ds = load_dataset(
    "bzantium/livecodebench",
    name="release_v2",
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