#!/usr/bin/env python3
"""Create deterministic train/val/test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
CURRENT_FILE = Path(__file__).resolve()
LIB_DIR = CURRENT_FILE.parents[2] / '_lib'
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from curation_core import read_jsonl, split_case_ids, write_json


def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits for a canonical dataset.')
    parser.add_argument('--canonical-root', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cases = read_jsonl(args.canonical_root / 'cases.jsonl')
    case_ids = [row['case_id'] for row in cases]
    splits = split_case_ids(case_ids, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    payload = {
        'canonical_root': str(args.canonical_root.resolve()),
        'seed': args.seed,
        'ratios': {
            'train': args.train_ratio,
            'val': args.val_ratio,
            'test': args.test_ratio,
        },
        'splits': splits,
    }
    write_json(payload, args.output_json)
    print(payload)


if __name__ == '__main__':
    main()
