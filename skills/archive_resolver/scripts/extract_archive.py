#!/usr/bin/env python3
"""Extract common archive formats with safe path checks."""

import argparse
import json
import os
import tarfile
import zipfile
from pathlib import Path


def _safe_join(base: Path, name: str) -> Path:
    target = (base / name).resolve()
    base_resolved = base.resolve()
    if os.path.commonpath([str(base_resolved), str(target)]) != str(base_resolved):
        raise ValueError(f"Unsafe archive member path: {name}")
    return target


def _archive_members(path: Path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            return [info.filename for info in zf.infolist() if info.filename and not info.is_dir()]
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            return [member.name for member in tf.getmembers() if member.isfile()]
    raise ValueError(f"Unsupported archive format: {path}")


def extract_archive(archive_path: Path, output_dir: Path, overwrite: bool = False):
    archive_path = archive_path.resolve()
    output_dir = output_dir.resolve()

    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = 0
    format_name = None

    if zipfile.is_zipfile(archive_path):
        format_name = "zip"
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                _safe_join(output_dir, info.filename)
            zf.extractall(output_dir)
            extracted_files = sum(1 for info in zf.infolist() if not info.is_dir())
    elif tarfile.is_tarfile(archive_path):
        format_name = "tar"
        with tarfile.open(archive_path) as tf:
            for member in tf.getmembers():
                _safe_join(output_dir, member.name)
            tf.extractall(output_dir)
            extracted_files = sum(1 for member in tf.getmembers() if member.isfile())
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    members = _archive_members(archive_path)
    top_level_entries = sorted({Path(name).parts[0] for name in members if Path(name).parts})

    return {
        "success": True,
        "archive_path": str(archive_path),
        "output_dir": str(output_dir),
        "format": format_name,
        "file_count": extracted_files,
        "top_level_entries": top_level_entries,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract an archive into a target directory.")
    parser.add_argument("archive_path", help="Path to the archive file.")
    parser.add_argument("output_dir", help="Directory to extract into.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow extracting into a non-empty output directory.",
    )
    args = parser.parse_args()

    result = extract_archive(Path(args.archive_path), Path(args.output_dir), overwrite=args.overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
