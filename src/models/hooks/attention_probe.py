"""Forward hook 抓取 decoder 每层 attention 权重。

用法（W1 任务 4）：
    probe = AttentionProbe(model.model.layers[K])
    out = model.generate(**inputs)
    attn = probe.last   # (batch, heads, L, L)

注意：
- 只有 attn_implementation="eager" 时 output_attentions=True 才真正返回矩阵；
- L = 视觉 token 数 + 文本 token 数，视觉 token 在前（image token 位于 input_ids 中
  由 <|image_pad|> 标记，位置可用 grid_thw 推出）。
"""
from __future__ import annotations

from typing import Optional

import torch


class AttentionProbe:
    def __init__(self, layer_module: torch.nn.Module):
        self.last: Optional[torch.Tensor] = None
        self.handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        # Qwen2.5-VL decoder layer 返回 (hidden_states, attn_weights?) 或 (hidden_states, ...)
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self.last = output[1].detach()

    def remove(self):
        self.handle.remove()
