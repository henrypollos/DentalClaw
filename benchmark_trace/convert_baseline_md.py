#!/usr/bin/env python3
"""
把用户手工汇总的 baseline 回答 md（按顺序排列的 JSON 块）拆成 <intent_id>.json 规范文件。

用法:
    python benchmark_trace/convert_baseline_md.py --md benchmark_results/baselines/qwen/1.md \
        --out benchmark_results/baselines/qwen --model "Qwen3.8-Max" --date 2026-08-21
"""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def split_json_blocks(text: str):
    """按花括号平衡切分文本中的多个 JSON 对象（支持单行与多行块、转义引号）。"""
    blocks = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] != "{":
            i += 1
        if i >= n:
            break
        depth, in_str, esc, j = 0, False, False, i
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if depth == 0 and j < n:
            blocks.append(text[i:j + 1])
            i = j + 1
        else:
            i += 1
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--date", required=True)
    args = ap.parse_args()

    text = Path(args.md).read_text(encoding="utf-8")
    blocks = split_json_blocks(text)
    print(f"切出 {len(blocks)} 个 JSON 块")

    intents_file = Path(__file__).resolve().parents[1] / "benchmark_intents" / "intents.platform_mvp_30.jsonl"
    intents = [json.loads(l) for l in open(intents_file, encoding="utf-8") if l.strip()]
    assert len(blocks) == len(intents), f"块数 {len(blocks)} != 意图数 {len(intents)}"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for intent, block in zip(intents, blocks):
        try:
            blob = json.loads(block)
        except Exception as e:
            print(f"⚠️ {intent['id']}: 解析失败 {e}\n{block[:120]}")
            continue
        blob["model"] = args.model
        blob["date"] = args.date
        (out_dir / f"{intent['id']}.json").write_text(
            json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
        n_ok += 1
    print(f"写出 {n_ok}/{len(blocks)} 个文件 → {out_dir}")


if __name__ == "__main__":
    main()
