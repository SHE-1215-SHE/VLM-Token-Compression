"""在 decoder 第 K 层裁剪视觉 token 的基类（FastV 风格插桩点）。

W2 的 fastv.py / visionzip.py / crop_r.py 都继承 TokenPruner，
只需实现 rank_tokens()；本基类负责：
1. 定位 input_ids 中视觉 token 的索引区间（image_pad token id）；
2. 在第 K 层 forward 前按 rank 结果 gather hidden_states / position_ids / attention_mask；
3. 记录被裁 token 数，供 profiling 对齐。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class TokenPruner(ABC):
    def __init__(self, k_layer: int = 2, keep_ratio: float = 0.33):
        self.k_layer = k_layer
        self.keep_ratio = keep_ratio
        self.removed: List[int] = []

    @abstractmethod
    def rank_tokens(self, attn: torch.Tensor, visual_idx: torch.Tensor) -> torch.Tensor:
        """返回视觉 token 的重要性分数 (batch, n_visual)。"""

    def visual_token_positions(self, input_ids: torch.Tensor, image_pad_id: int) -> torch.Tensor:
        return (input_ids == image_pad_id).nonzero(as_tuple=True)[1]
