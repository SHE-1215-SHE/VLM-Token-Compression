"""MMBench: 选择题（dev 子集），报 Accuracy。
子集 500 条 -> data/mmbench_sample.jsonl
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def load_sample(data_dir: str, n: int = 500, seed: int = 42):
    path = Path(data_dir) / "mmbench_sample.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prompt_of(item: dict) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in item["choices"].items())
    return f"{item['question']}\n{opts}\nAnswer with the option's letter only."


def parse_answer(text: str) -> str:
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else ""


def score(preds: list[str], gts: list[str]) -> dict:
    return {"acc": sum(p == g for p, g in zip(preds, gts)) / max(len(gts), 1)}
