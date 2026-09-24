"""所有压缩方法的统一接口。

runner.py 只认这个接口，新增方法 = 新写一个文件实现 compress()。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class CompressionMethod(ABC):
    name: str = "base"

    def __init__(self, budget: float = 1.0, **kwargs):
        # budget: 保留比例 (0~1]，1.0 即 baseline 不压缩
        self.budget = budget

    @abstractmethod
    def compress(
        self,
        visual_tokens: torch.Tensor,
        attn: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """输入 (B, N, D) 视觉 token，输出 (B, N', D)，N' = ceil(N * budget)。"""
