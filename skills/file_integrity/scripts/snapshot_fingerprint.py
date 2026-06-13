#!/usr/bin/env python3
"""Generate a compact fingerprint report for a file or directory snapshot."""

import argparse
import json
from collections import Counter
from pathlib import Path

from compute_checksum import compute_checksum


def _iter_files(path: Path):
    if path.is_file():
        yield path
        return
    for item in sorted(path.rglob("*")):
        if item.is_file():
            yield item


def snapshot_fingerprint(path: Path, algorithm: str = "sha256"):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    checksum_info = compute_checksum(path, algorithm=algorithm)
    files = list(_iter_files(path))
    suffix_counter = Counter((file_path.suffix.lower() or "<no_ext>") for file_path in files)
    total_bytes = sum(file_path.stat().st_size for file_path in files)

    top_extensions = [
        {"suffix": suffix, "count": count}
        for suffix, count in suffix_counter.most_common(20)
    ]

    return {
        "success": True,
        "path": str(path),
        "type": checksum_info["type"],
        "algorithm": algorithm,
        "checksum": checksum_info["checksum"],
        "file_count": len(files),
        "total_bytes": total_bytes,
        "top_extensions": top_extensions,
    }


def main():
    parser = argparse.ArgumentParser(description="Create a compact fingerprint for a file or directory.")
    parser.add_argument("path", help="Target file or directory.")
    parser.add_argument(
        "--algorithm",
        default="sha256",
        help="Hash algorithm accepted by hashlib. Default: sha256",
    )
    args = parser.parse_args()

    result = snapshot_fingerprint(Path(args.path), algorithm=args.algorithm)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
