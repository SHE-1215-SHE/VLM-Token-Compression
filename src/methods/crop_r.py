"""CROP-R (arXiv'25): MLLM 上的 training-free token 压缩——
先 re-weight 抑制细粒度冗余 token，再 prune。

本地已有官方代码 c:\Users\13764\Desktop\test\cropr-main，
W2 优先做：读懂其核心 200 行 -> 适配 Qwen2.5-VL（原码可能基于 LLaVA）。
"""
from __future__ import annotations

import torch

from .base import CompressionMethod


class CropR(CompressionMethod):
    name = "crop_r"

    def compress(self, visual_tokens: torch.Tensor, attn: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("W2 任务：从 cropr-main 移植核心逻辑")
