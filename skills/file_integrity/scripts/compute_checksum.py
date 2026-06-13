#!/usr/bin/env python3
"""Compute checksums for files or directory trees."""

import argparse
import hashlib
import json
from pathlib import Path

CHUNK_SIZE = 1024 * 1024


def _new_hasher(algorithm: str):
    try:
        return hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc


def _hash_file(path: Path, algorithm: str) -> str:
    hasher = _new_hasher(algorithm)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _iter_files(path: Path):
    for item in sorted(path.rglob("*")):
        if item.is_file():
            yield item


def _hash_directory(path: Path, algorithm: str) -> str:
    manifest_hasher = _new_hasher(algorithm)
    for file_path in _iter_files(path):
        relative = file_path.relative_to(path).as_posix()
        file_digest = _hash_file(file_path, algorithm)
        manifest_hasher.update(relative.encode("utf-8"))
        manifest_hasher.update(bytes([0]))
        manifest_hasher.update(file_digest.encode("ascii"))
        manifest_hasher.update(b"\n")
    return manifest_hasher.hexdigest()


def compute_checksum(path: Path, algorithm: str = "sha256"):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        digest = _hash_file(path, algorithm)
        kind = "file"
    elif path.is_dir():
        digest = _hash_directory(path, algorithm)
        kind = "directory"
    else:
        raise ValueError(f"Unsupported path type: {path}")

    return {
        "success": True,
        "path": str(path),
        "type": kind,
        "algorithm": algorithm,
        "checksum": digest,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute a checksum for a file or directory.")
    parser.add_argument("path", help="Target file or directory.")
    parser.add_argument(
        "--algorithm",
        default="sha256",
        help="Hash algorithm accepted by hashlib. Default: sha256",
    )
    args = parser.parse_args()

    result = compute_checksum(Path(args.path), algorithm=args.algorithm)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
