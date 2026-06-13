#!/usr/bin/env python3
"""Copy or move the effective archive root contents into a clean destination."""

import argparse
import json
import shutil
from pathlib import Path


def _copy_entry(src: Path, dst: Path):
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _move_entry(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def flatten_nested_dirs(source_dir: Path, dest_dir: Path, mode: str = "copy", clear_dest: bool = False):
    source_dir = source_dir.resolve()
    dest_dir = dest_dir.resolve()

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {source_dir}")

    if source_dir == dest_dir:
        raise ValueError("Source and destination directories must be different.")

    if clear_dest and dest_dir.exists():
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    for entry in sorted(source_dir.iterdir(), key=lambda item: item.name):
        target = dest_dir / entry.name
        if mode == "copy":
            _copy_entry(entry, target)
        elif mode == "move":
            _move_entry(entry, target)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        moved.append(entry.name)

    return {
        "success": True,
        "source_dir": str(source_dir),
        "dest_dir": str(dest_dir),
        "mode": mode,
        "entries": moved,
    }


def main():
    parser = argparse.ArgumentParser(description="Flatten a resolved dataset root into a clean destination.")
    parser.add_argument("source_dir", help="Resolved source directory.")
    parser.add_argument("dest_dir", help="Destination directory.")
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="Whether to copy or move source entries.",
    )
    parser.add_argument(
        "--clear-dest",
        action="store_true",
        help="Delete the destination directory before flattening.",
    )
    args = parser.parse_args()

    result = flatten_nested_dirs(
        Path(args.source_dir),
        Path(args.dest_dir),
        mode=args.mode,
        clear_dest=args.clear_dest,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
