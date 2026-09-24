"""FastV (ECCV'24): decoder 第 K 层后按最后一行文本 token 对视觉 token 的
attention 分数排序，保留 top-ratio 视觉 token。

W2 实现。参考: https://github.com/pkunlp-icler/FastV
"""
from __future__ import annotations

import torch

from .base import CompressionMethod


class FastV(CompressionMethod):
    name = "fastv"

    def __init__(self, budget: float = 0.33, k_layer: int = 2, **kwargs):
        super().__init__(budget, **kwargs)
        self.k_layer = k_layer

    def compress(self, visual_tokens: torch.Tensor, attn: torch.Tensor | None = None) -> torch.Tensor:
        # attn: (B, H, L, L) 第 k_layer 层注意力；取最后一个文本 token 行对视觉列的分数
        raise NotImplementedError("W2 任务：在此实现 attention rank + topk gather")
