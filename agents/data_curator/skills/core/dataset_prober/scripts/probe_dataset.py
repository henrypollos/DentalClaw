#!/usr/bin/env python3
"""Probe dataset structure and likely task roles."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
CURRENT_FILE = Path(__file__).resolve()
LIB_DIR = CURRENT_FILE.parents[2] / '_lib'
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from curation_core import build_probe_report, write_json


def write_markdown(report: dict, path: Path):
    lines = [
        '# Dataset Probe Report',
        '',
        f"- Dataset root: `{report['dataset_root']}`",
        f"- Primary image dir: `{report['primary_image_dir']}`" if report['primary_image_dir'] else '- Primary image dir: None',
        f"- Primary case count: {report['primary_case_count']}",
        f"- Task candidates: {', '.join(report['task_candidates']) if report['task_candidates'] else 'None'}",
        '',
        '## Raster Label Dirs',
        '',
    ]
    for entry in report['raster_label_dirs']:
        lines.append(f"- `{entry['role']}` -> `{entry['path']}` ({entry['matched_case_count']} matched, overlap {entry['overlap_ratio']:.3f})")
    if not report['raster_label_dirs']:
        lines.append('- None')
    lines.extend(['', '## JSON Annotations', ''])
    for entry in report['json_annotations']:
        lines.append(f"- `{entry['role_hint']}` / {entry['detected_format']} -> `{entry['path']}`")
    if not report['json_annotations']:
        lines.append('- None')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Probe a dataset directory.')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, default=None)
    args = parser.parse_args()

    report = build_probe_report(args.dataset_root)
    write_json(report, args.output_json)
    if args.output_md:
        write_markdown(report, args.output_md)
    print(report)


if __name__ == '__main__':
    main()
