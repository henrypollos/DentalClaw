#!/usr/bin/env python3
"""Score CBCT QC findings against an injected error manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def findings_by_case(normalized_cases_path: Path) -> Dict[str, List[dict]]:
    rows = read_jsonl(normalized_cases_path)
    mapping: Dict[str, List[dict]] = {}
    for row in rows:
        mapping[str(row["case_id"])] = list(row.get("findings") or [])
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Score injected CBCT QC errors against audit findings.")
    parser.add_argument("--injection-manifest", type=Path, required=True)
    parser.add_argument("--normalized-cases", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.injection_manifest)
    case_findings = findings_by_case(args.normalized_cases)

    records = []
    category_stats: Dict[str, Dict[str, int]] = {}
    detected_total = 0
    total = 0

    for record in manifest.get("injections", []):
        category = str(record["category"])
        expected_any = set(record.get("expected_findings_any") or [])
        expected_all = set(record.get("expected_findings_all") or [])
        case_ids = [str(case_id) for case_id in (record.get("case_ids") or [])]
        observed_codes = {
            finding["code"]
            for case_id in case_ids
            for finding in case_findings.get(case_id, [])
        }
        matched_any = sorted(observed_codes & expected_any)
        matched_all = sorted(expected_all & observed_codes)
        detected = (not expected_all or expected_all.issubset(observed_codes)) and (not expected_any or bool(matched_any))
        if not expected_all and expected_any:
            detected = bool(matched_any)
        elif expected_all and not expected_any:
            detected = expected_all.issubset(observed_codes)

        total += 1
        if detected:
            detected_total += 1
        stats = category_stats.setdefault(category, {"injected": 0, "detected": 0})
        stats["injected"] += 1
        if detected:
            stats["detected"] += 1

        records.append({
            "category": category,
            "case_ids": case_ids,
            "expected_findings_any": sorted(expected_any),
            "expected_findings_all": sorted(expected_all),
            "observed_codes": sorted(observed_codes),
            "matched_any": matched_any,
            "matched_all": matched_all,
            "detected": detected,
            "details": record.get("details"),
        })

    summary = {
        "injection_manifest": str(args.injection_manifest.resolve()),
        "normalized_cases": str(args.normalized_cases.resolve()),
        "total_injected_records": total,
        "detected_records": detected_total,
        "detected_fraction": (detected_total / total) if total else None,
        "category_stats": {
            category: {
                **stats,
                "detected_fraction": (stats["detected"] / stats["injected"]) if stats["injected"] else None,
            }
            for category, stats in sorted(category_stats.items())
        },
        "records": records,
    }
    write_json(args.output_json, summary)
    print(json.dumps({
        "total_injected_records": total,
        "detected_records": detected_total,
        "detected_fraction": summary["detected_fraction"],
        "category_stats": summary["category_stats"],
    }, indent=2))


if __name__ == "__main__":
    main()
