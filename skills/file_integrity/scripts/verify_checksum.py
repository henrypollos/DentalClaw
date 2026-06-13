#!/usr/bin/env python3
"""Verify a file or directory checksum against an expected digest."""

import argparse
import json
from pathlib import Path

from compute_checksum import compute_checksum


def verify_checksum(path: Path, expected_checksum: str, algorithm: str = "sha256"):
    result = compute_checksum(path, algorithm=algorithm)
    actual = result["checksum"].lower()
    expected = expected_checksum.strip().lower()
    result["expected_checksum"] = expected
    result["match"] = actual == expected
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify a checksum for a file or directory.")
    parser.add_argument("path", help="Target file or directory.")
    parser.add_argument("expected_checksum", help="Expected digest string.")
    parser.add_argument(
        "--algorithm",
        default="sha256",
        help="Hash algorithm accepted by hashlib. Default: sha256",
    )
    args = parser.parse_args()

    result = verify_checksum(Path(args.path), args.expected_checksum, algorithm=args.algorithm)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["match"] else 1)


if __name__ == "__main__":
    main()
