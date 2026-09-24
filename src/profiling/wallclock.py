"""CUDA events 分段计时：prefill（首 token 前）与 decode（逐 token）。

本仓库核心差异化：论文报 FLOPs，我们报真实 ms。
用法：
    with Timer() as t: model.generate(...)
    t.elapsed_ms
"""
from __future__ import annotations

import torch


class Timer:
    def __init__(self):
        self.start = None
        self.end = None
        self.elapsed_ms = 0.0

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()
        torch.cuda.synchronize()
        self.elapsed_ms = self.start.elapsed_time(self.end)


def measure_generate(model, processor, inputs, n_warmup=2, n_runs=5) -> dict:
    """warmup 后多次取均值；分别记录 TTFT 与总时长。W1 任务 7 实现。"""
    raise NotImplementedError
