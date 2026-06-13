#!/usr/bin/env python3
"""
Compatibility wrapper for legacy TDD -> nnUNet export calls.

This script preserves the old CLI shape used by historical `main` sessions,
but delegates the actual work to the maintained `tdd-nnunet-export` skill.
That means stale prompts that still call this file will now get:

- `teeth_32class` polygon-based export
- proper `imagesTs/labelsTs` holdout generation
- dataset QC report emission
- standard DentalClaw artifact locations
"""

import argparse
import subprocess
import sys
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[3]
EXPORTER_SCRIPT = (
    REPO_ROOT
    / "agents"
    / "data_curator"
    / "skills"
    / "datasets"
    / "tdd-nnunet-export"
    / "scripts"
    / "export_tdd_to_nnunet.py"
)
DEFAULT_PREPROCESSED_ROOT = REPO_ROOT / "artifacts" / "datasets" / "nnUNet" / "nnUNet_preprocessed"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "artifacts" / "models" / "nnUNet" / "nnUNet_results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Legacy compatibility wrapper for exporting TDD to nnUNetv2 32-class 2D format."
    )
    parser.add_argument("--tdd_root", required=True, help="TDD dataset root")
    parser.add_argument(
        "--nnunet_raw",
        required=True,
        help="Legacy nnUNet_raw path. If this points at a `nnUNet_raw/` folder, the wrapper will use its parent as the output root.",
    )
    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_output_root(nnunet_raw_arg: str) -> Path:
    path = Path(nnunet_raw_arg).resolve()
    return path.parent if path.name == "nnUNet_raw" else path


def main():
    args = parse_args()
    if not EXPORTER_SCRIPT.is_file():
        raise FileNotFoundError("Maintained exporter script not found: {}".format(EXPORTER_SCRIPT))

    output_root = resolve_output_root(args.nnunet_raw)
    command = [
        sys.executable,
        str(EXPORTER_SCRIPT),
        "--dataset-root",
        str(Path(args.tdd_root).resolve()),
        "--output-root",
        str(output_root),
        "--preprocessed-root",
        str(DEFAULT_PREPROCESSED_ROOT),
        "--results-root",
        str(DEFAULT_RESULTS_ROOT),
        "--task",
        "teeth_32class",
        "--dataset-id",
        str(args.dataset_id),
        "--dataset-name",
        str(args.dataset_name),
        "--test-ratio",
        str(args.test_frac),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        command.append("--overwrite")

    print(
        "[legacy-wrapper] Redirecting to maintained TDD exporter with task=teeth_32class, "
        "test_ratio={}, output_root={}".format(args.test_frac, output_root)
    )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
