#!/usr/bin/env python3
"""Resolve the effective dataset root inside an extracted archive tree."""

import argparse
import json
from pathlib import Path


def resolve_archive_root(root: Path):
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    chain = [str(root)]
    current = root
    collapsed_levels = 0

    while True:
        entries = sorted(current.iterdir(), key=lambda item: item.name)
        dirs = [entry for entry in entries if entry.is_dir()]
        files = [entry for entry in entries if entry.is_file()]

        if len(dirs) == 1 and not files:
            current = dirs[0]
            chain.append(str(current))
            collapsed_levels += 1
            continue
        break

    return {
        "success": True,
        "input_root": str(root),
        "resolved_root": str(current),
        "collapsed_levels": collapsed_levels,
        "chain": chain,
    }


def main():
    parser = argparse.ArgumentParser(description="Resolve the effective root of an extracted archive.")
    parser.add_argument("root", help="Directory to inspect.")
    args = parser.parse_args()

    result = resolve_archive_root(Path(args.root))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
