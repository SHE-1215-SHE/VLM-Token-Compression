"""VisionZip (CVPR'24): 视觉编码器输出侧——用 [CLS] 对 patch 的 attention
选 dominant tokens，其余上下文 token 加权合并。

W2 实现。参考: https://github.com/dvlab-research/VisionZip
注意：Qwen2.5-VL 无 [CLS]，需用 attention 列均值或 last-row 近似，属适配点。
"""
from __future__ import annotations

import torch

from .base import CompressionMethod


class VisionZip(CompressionMethod):
    name = "visionzip"

    def compress(self, visual_tokens: torch.Tensor, attn: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("W2 任务：dominant selection + contextual merging")
