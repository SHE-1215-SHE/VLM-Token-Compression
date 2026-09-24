"""各 benchmark 固定 seed 抽样为 jsonl（保证结果可复现、可对比）。

数据下载见 README「Data」；此处只做抽样与格式统一。
输出: data/{pope,gqa,textvqa,mmbench}_sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

N = {"pope": 500, "gqa": 500, "textvqa": 300, "mmbench": 500}


def sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()

    out = Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    # W1 任务 5：按 raw 文件格式读取 -> sample -> 写 jsonl
    # 每条统一字段: {id, image, question, answer, choices?}
    print("TODO(W1-5): 实现四个数据集的 reader，字段对齐 runner.py")


if __name__ == "__main__":
    main()
