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
    """warmup 后多次取均值；分别记录 prefill（TTFT）与 decode 总时长。

    TTFT 用 first-token 计时近似：generate 内部无分段钩子，
    这里通过 streamer 逐 token 记录时间戳拆分 prefill / decode。
    """
    from transformers import TextStreamer

    class _Stamp(TextStreamer):
        """在收到第一个生成 token（单 token 块）时记录时间。"""
        first_ts = None

        def put(self, value, stream_delay=None):  # noqa: D401
            # 第一块是整个 prompt（len>1），之后的块才是逐个生成 token
            if self.first_ts is None and value is not None and value.shape[-1] == 1:
                torch.cuda.synchronize()
                self.first_ts = torch.cuda.Event(enable_timing=True)
                self.first_ts.record()

        def end(self):
            pass

    results = {"ttft_ms": [], "total_ms": []}
    for i in range(n_warmup + n_runs):
        streamer = _Stamp(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        model.generate(**inputs, max_new_tokens=32, do_sample=False, streamer=streamer)
        t1.record()
        torch.cuda.synchronize()
        if i < n_warmup:
            continue
        total = t0.elapsed_time(t1)
        ttft = t0.elapsed_time(streamer.first_ts) if streamer.first_ts is not None else total
        results["ttft_ms"].append(ttft)
        results["total_ms"].append(total)
    return {
        "ttft_ms": sum(results["ttft_ms"]) / len(results["ttft_ms"]),
        "total_ms": sum(results["total_ms"]) / len(results["total_ms"]),
        "n_runs": n_runs,
    }
