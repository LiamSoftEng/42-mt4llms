#!/usr/bin/env python3
"""
Combine per-repeat metrics CSVs into one CSV per difficulty.

For each difficulty (easy / medium / hard), this finds all
repeats-XX.metrics.csv files under that difficulty's folder, tags each
row with which repeat it came from, and stacks them so that all repeats
for the same question `id` sit next to each other (sorted by id, then
by repeat number).

No third-party packages required - standard library only.

Usage:
    python combine_repeats.py --base results/gpt-5-6-sol/base --out ./output

By default it searches recursively under --base for files matching
'repeats-*.metrics.csv' and buckets them by difficulty based on
'easy' / 'medium' / 'hard' appearing anywhere in the file's path.
Adjust DIFFICULTY_KEYWORDS below if your folder names differ.
"""

import argparse
import csv
import re
from pathlib import Path

DIFFICULTY_KEYWORDS = ["easy", "medium", "hard"]


def find_difficulty(path: Path) -> str | None:
    path_str = str(path).lower()
    for kw in DIFFICULTY_KEYWORDS:
        if kw in path_str:
            return kw
    return None


def extract_repeat_number(filename: str) -> str:
    m = re.search(r"repeats?-(\d+)\.metrics\.csv$", filename)
    return m.group(1) if m else filename


def read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="results", help="Base directory to search")
    parser.add_argument("--out", default="output", help="Output directory for CSVs")
    parser.add_argument("--pattern", default="repeat-*.metrics.csv", help="Glob pattern for repeat files")
    args = parser.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(base.rglob(args.pattern))
    if not files:
        print(f"No files matching '{args.pattern}' found under {base}")
        return

    buckets: dict[str, list[Path]] = {}
    for f in files:
        diff = find_difficulty(f)
        if diff is None:
            print(f"WARNING: could not determine difficulty for {f}, skipping")
            continue
        buckets.setdefault(diff, []).append(f)

    for diff, diff_files in buckets.items():
        all_rows = []
        fieldnames = None

        for f in sorted(diff_files):
            repeat_label = extract_repeat_number(f.name)
            file_fieldnames, rows = read_csv_rows(f)

            if fieldnames is None:
                fieldnames = ["source_file", "repeat"] + file_fieldnames
            elif file_fieldnames != fieldnames[2:]:
                print(f"WARNING: {f.name} columns differ from earlier files in '{diff}' bucket")

            for row in rows:
                row["source_file"] = f.name
                row["repeat"] = repeat_label
                all_rows.append(row)

        # group repeats of the same question together, ordered by repeat within each
        all_rows.sort(key=lambda r: (r.get("id", ""), r.get("repeat", "")))

        out_path = out_dir / f"{diff}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"Wrote {out_path} ({len(all_rows)} rows, {len(diff_files)} files)")


if __name__ == "__main__":
    main()