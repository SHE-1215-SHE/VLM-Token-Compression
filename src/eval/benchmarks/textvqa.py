"""TextVQA: 图中文字读取（OCR 类），报 ANLS。
预期：token 压缩在低预算下最先在此崩——失败案例分析主素材。
子集 300 条 -> data/textvqa_sample.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path


def load_sample(data_dir: str, n: int = 300, seed: int = 42):
    path = Path(data_dir) / "textvqa_sample.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prompt_of(item: dict) -> str:
    return item["question"]


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls_one(pred: str, gts: list[str], tau: float = 0.5) -> float:
    best = 1.0
    for g in gts:
        d = _levenshtein(pred.lower().strip(), g.lower().strip())
        nd = min(d / max(len(pred), len(g), 1), 1.0)
        best = min(best, nd if nd < tau else 1.0)
    return 1.0 - best


def score(preds: list[str], gts: list[list[str]]) -> dict:
    return {"anls": sum(anls_one(p, g) for p, g in zip(preds, gts)) / max(len(gts), 1)}
