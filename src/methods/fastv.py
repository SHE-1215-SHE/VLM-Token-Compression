"""FastV (ECCV'24)：decoder 第 K 层后按 attention 分数剪视觉 token 的 KV cache。

流程（对 Qwen2.5-VL dynamic 分辨率的适配实现）：
1. 手动 prefill（带 output_attentions），hook 抓第 K 层 attention (B,H,L,L)；
2. 取最后一个输入 token 的 query 行对视觉 token 列的平均 attention 分数排序，
   保留 top-budget 比例的视觉 token（文本 token 全保留）；
3. 从 DynamicCache 的每层 K/V 中删掉被剪位置（RoPE 已烘焙进 K，删列即可）；
4. 手动贪心 decode 循环续写。

budget=1.0 时不剪枝，输出应与 baseline generate 逐 token 一致（自检用）。
参考: https://github.com/pkunlp-icler/FastV
"""
from __future__ import annotations

import torch

from .base import CompressionMethod
from ..models.hooks.attention_probe import AttentionProbe


class FastV(CompressionMethod):
    name = "fastv"

    def __init__(self, budget: float = 0.33, k_layer: int = 2, **kwargs):
        super().__init__(budget, **kwargs)
        self.k_layer = k_layer  # 0-indexed decoder 层号
        self.last_kept = 0  # 记录实际保留的视觉 token 数，供 profiling

    def rank_visual(self, attn: torch.Tensor, visual_pos: torch.Tensor) -> torch.Tensor:
        """返回按分数降序的视觉 token 局部索引。attn: (1,H,L,L)。"""
        # FastV 官方：最后一个输入 token 的 attention 行（含 head 平均）
        scores = attn[0, :, -1, visual_pos].mean(dim=0)  # (Nv,)
        return scores.argsort(descending=True)

    def compress(self, visual_tokens: torch.Tensor, attn: torch.Tensor | None = None) -> torch.Tensor:
        """接口兼容实现：FastV 作用在 KV cache 上，真正的入口是 generate()。"""
        if attn is None:
            return visual_tokens
        n = visual_tokens.shape[1]
        n_keep = max(1, int(n * self.budget))
        keep = self.rank_visual(attn, torch.arange(n, device=visual_tokens.device))[:n_keep].sort().values
        return visual_tokens[:, keep, :]

    @torch.no_grad()
    def generate(self, model, inputs: dict, max_new_tokens: int = 32) -> torch.Tensor:
        """剪枝版贪心解码，返回完整生成序列 ids（不含 prompt）。"""
        k_layer = min(self.k_layer, len(model.model.language_model.layers) - 1)
        image_pad_id = model.config.image_token_id

        # --- 1. 手动 prefill：只让目标层物化 attention（其余层不产出权重，
        #        避免 36 层全量权重导致的显存churn，实测 5 倍减速）---
        probe = AttentionProbe(model.model.language_model.layers[k_layer])
        layer = model.model.language_model.layers[k_layer]
        orig_forward = layer.forward

        def _force_attn(*args, **kwargs):
            kwargs["output_attentions"] = True
            return orig_forward(*args, **kwargs)

        layer.forward = _force_attn
        try:
            out = model(**inputs, use_cache=True)
        finally:
            layer.forward = orig_forward
            probe.remove()
        past = out.past_key_values
        input_ids = inputs["input_ids"]
        B, L = input_ids.shape

        # --- 2. 排序 + 决定保留集合 ---
        visual_pos = (input_ids[0] == image_pad_id).nonzero(as_tuple=True)[0]
        if self.budget < 1.0 and visual_pos.numel() > 0:
            order = self.rank_visual(probe.last, visual_pos)
            n_keep = max(1, int(visual_pos.numel() * self.budget))
            keep_global = visual_pos[order[:n_keep]].sort().values
            drop_global = visual_pos[order[n_keep:]]
            self.last_kept = int(n_keep)
            keep_mask = torch.ones(L, dtype=torch.bool, device=input_ids.device)
            keep_mask[drop_global] = False  # 只标记被剪的低分视觉 token
            keep_mask = keep_mask.tolist()
        else:
            self.last_kept = int(visual_pos.numel())
            keep_mask = None

        if keep_mask is not None:
            # --- 3. 剪 KV cache：每层 K/V 在序列维 gather（4.55 的 DynamicCache
            #        key_cache 是只读 property，因此构建新 cache 写回）---
            from transformers.cache_utils import DynamicCache

            keep_idx = torch.tensor([i for i, k in enumerate(keep_mask) if k], device=input_ids.device)
            pruned = DynamicCache()
            for li in range(len(past)):
                pruned.update(past.key_cache[li][:, :, keep_idx, :],
                              past.value_cache[li][:, :, keep_idx, :], li)
            past = pruned

        # --- 4. 贪心 decode 循环 ---
        next_id = out.logits[:, -1:].argmax(dim=-1)
        generated = [next_id]
        for step in range(max_new_tokens - 1):
            # 文本 token 的 M-RoPE 三维位置相同；从原 prompt 末位续接（RoPE 相对编码允许空洞）
            position_ids = torch.full((3, B, 1), L + step, device=input_ids.device, dtype=torch.long)
            out = model(
                input_ids=next_id,
                past_key_values=past,
                use_cache=True,
                position_ids=position_ids,
                attention_mask=None,  # 无 padding，默认全可见
            )
            next_id = out.logits[:, -1:].argmax(dim=-1)
            generated.append(next_id)
        return torch.cat(generated, dim=1)  # (B, T)
