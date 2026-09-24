"""POPE: 二元存在性问答（yes/no），报 F1 / Accuracy。
数据: https://github.com/RUCAIBox/POPE （coco 划分）
子集: scripts/sample_benchmarks.py 固定 seed 抽 500 条存 data/pope_sample.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path


def load_sample(data_dir: str, n: int = 500, seed: int = 42):
    path = Path(data_dir) / "pope_sample.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prompt_of(item: dict) -> str:
    return item["question"] + " Answer the question with Yes or No."


def parse_answer(text: str) -> str:
    t = text.strip().lower()
    return "yes" if t.startswith("yes") else "no"


def score(preds: list[str], gts: list[str]) -> dict:
    tp = sum(p == "yes" and g == "yes" for p, g in zip(preds, gts))
    fp = sum(p == "yes" and g == "no" for p, g in zip(preds, gts))
    fn = sum(p == "no" and g == "yes" for p, g in zip(preds, gts))
    acc = sum(p == g for p, g in zip(preds, gts)) / len(gts)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"acc": acc, "f1": f1}
