"""汇总各 benchmark 指标为统一结果行。"""
from __future__ import annotations

BENCH_METRICS = {
    "pope": "f1",
    "gqa": "acc",
    "textvqa": "anls",
    "mmbench": "acc",
}
