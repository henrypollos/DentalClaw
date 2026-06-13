#!/usr/bin/env python3
"""Summarize file types and directory layout for a path."""

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize_files(path: Path):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        return {
            "success": True,
            "root": str(path),
            "file_count": 1,
            "directory_count": 0,
            "total_bytes": path.stat().st_size,
            "suffix_counts": [{"suffix": path.suffix.lower() or "<no_ext>", "count": 1}],
            "top_level_entries": [],
        }

    file_count = 0
    directory_count = 0
    total_bytes = 0
    suffix_counter = Counter()
    top_level_entries = []

    for entry in sorted(path.iterdir(), key=lambda item: item.name):
        top_level_entries.append(
            {
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
            }
        )

    for item in path.rglob("*"):
        if item.is_dir():
            directory_count += 1
            continue
        if item.is_file():
            file_count += 1
            total_bytes += item.stat().st_size
            suffix_counter[item.suffix.lower() or "<no_ext>"] += 1

    return {
        "success": True,
        "root": str(path),
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
        "suffix_counts": [
            {"suffix": suffix, "count": count}
            for suffix, count in suffix_counter.most_common(50)
        ],
        "top_level_entries": top_level_entries,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize files under a path.")
    parser.add_argument("path", help="Target file or directory.")
    args = parser.parse_args()

    result = summarize_files(Path(args.path))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
