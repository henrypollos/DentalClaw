#!/usr/bin/env python3
"""Scan a directory tree and emit a structured listing."""

import argparse
import json
from pathlib import Path


def scan_tree(path: Path, max_depth=None):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        return {
            "success": True,
            "root": str(path),
            "entries": [
                {
                    "path": path.name,
                    "type": "file",
                    "depth": 0,
                    "size_bytes": path.stat().st_size,
                }
            ],
        }

    entries = []
    for item in sorted(path.rglob("*")):
        rel_path = item.relative_to(path)
        depth = len(rel_path.parts)
        if max_depth is not None and depth > max_depth:
            continue
        entry = {
            "path": rel_path.as_posix(),
            "type": "directory" if item.is_dir() else "file",
            "depth": depth,
        }
        if item.is_file():
            entry["size_bytes"] = item.stat().st_size
        entries.append(entry)

    return {
        "success": True,
        "root": str(path),
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser(description="Scan a file or directory tree.")
    parser.add_argument("path", help="Target path.")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum relative depth to include.",
    )
    args = parser.parse_args()

    result = scan_tree(Path(args.path), max_depth=args.max_depth)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
