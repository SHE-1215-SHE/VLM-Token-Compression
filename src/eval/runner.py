"""评测主入口：method x budget x benchmark 三重循环。

W1 只要求 budget=1.0（baseline）能跑通；方法插桩在 W2 接入。
运行示例：
    python -m src.eval.runner --config configs/qwen25vl_3b.yaml \
        --method baseline --budget 1.0 --benchmarks pope,gqa --limit 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.eval.benchmarks import gqa, mmbench, pope, textvqa
from src.models.loader import load_from_config

BENCH_REGISTRY = {"pope": pope, "gqa": gqa, "textvqa": textvqa, "mmbench": mmbench}


def run_one(model, processor, bench_name: str, data_dir: str, limit: int, method=None):
    """单 benchmark 评测。method=None 即 baseline。
    W2 起：在 generate 前后挂 method.compress()。
    """
    raise NotImplementedError("W1 任务 6：写推理循环（process_vision_info + generate + 解析）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen25vl_3b.yaml")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--benchmarks", default="pope,gqa,textvqa,mmbench")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=0, help="调试用样本数上限，0=全量")
    args = ap.parse_args()

    Path(args.out).mkdir(exist_ok=True, parents=True)
    model, processor = None, None  # 延迟加载，W1 填 load_from_config(args.config)
    print("runner skeleton; implement run_one in W1 task 6")


if __name__ == "__main__":
    main()
