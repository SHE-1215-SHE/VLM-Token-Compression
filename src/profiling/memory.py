"""显存峰值测量。"""
from __future__ import annotations

import torch


def peak_memory_gb(fn, *args, **kwargs) -> tuple[object, float]:
    """执行 fn 并返回 (fn结果, 峰值显存GB)。调用前需 reset_peak_memory_stats。"""
    torch.cuda.reset_peak_memory_stats()
    result = fn(*args, **kwargs)
    return result, torch.cuda.max_memory_allocated() / 1024**3
