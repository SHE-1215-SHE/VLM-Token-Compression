"""GQA: 组合视觉问答（短答案），报 Accuracy（宽松匹配）。
子集 500 条 -> data/gqa_sample.jsonl
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def load_sample(data_dir: str, n: int = 500, seed: int = 42):
    path = Path(data_dir) / "gqa_sample.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prompt_of(item: dict) -> str:
    return item["question"] + " Answer in one or two words."


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def score(preds: list[str], gts: list[str]) -> dict:
    hit = sum(normalize(g) in normalize(p) for p, g in zip(preds, gts))
    return {"acc": hit / max(len(gts), 1)}
