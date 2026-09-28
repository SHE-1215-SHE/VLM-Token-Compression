"""VisionZip (CVPR'24) 的 Qwen2.5-VL 适配实现：视觉编码器输出侧压缩。

原版用 CLIP 的 [CLS] attention 选 dominant tokens；Qwen2.5-VL 无 [CLS]，
且 PatchMerger 要求 2x2 网格对齐，因此适配点放在**merge 后的视觉嵌入**上：
1. model.visual 输出 E (N, D)，以 ||E||2 作为 saliency（常见替代，README 需注明）；
2. 保留 top-budget 的 dominant tokens（按原网格顺序）；
3. 其余 token 按位置连续分块（每 4 个）做平均，生成少量 context tokens（加权池化思想）；
4. 视觉序列 = dominant（原位）+ context（均值位置），替换 input_ids 中的 image_pad 嵌入；
5. M-RoPE：用 model.get_rope_index 拿全序列三维位置，kept token 保留原位置
   （RoPE 相对编码允许位置空洞），context token 用块内均值位置。

参考: https://github.com/dvlab-research/VisionZip
"""
from __future__ import annotations

import torch

from .base import CompressionMethod


class VisionZip(CompressionMethod):
    name = "visionzip"

    def __init__(self, budget: float = 0.5, context_group: int = 4, **kwargs):
        super().__init__(budget, **kwargs)
        self.context_group = context_group  # 被剪 token 按 x 个一组合并为 context token

    # ------------------------------------------------------------------
    def _compress_visual(self, embeds: torch.Tensor, pos3: torch.Tensor):
        """embeds: (N, D) merge 后视觉嵌入；pos3: (3, N) 其 M-RoPE 位置。
        返回 (new_embeds (N', D), new_pos3 (3, N'))。
        """
        n = embeds.shape[0]
        n_keep = max(1, int(n * self.budget))
        if n_keep >= n:
            return embeds, pos3

        scores = embeds.float().norm(dim=-1)  # (N,)
        dominant = scores.topk(n_keep).indices.sort().values  # 保持原顺序
        dropped_mask = torch.ones(n, dtype=torch.bool, device=embeds.device)
        dropped_mask[dominant] = False
        dropped = dropped_mask.nonzero(as_tuple=True)[0]

        # context tokens：连续分块均值
        g = self.context_group
        n_ctx = (dropped.numel() + g - 1) // g
        ctx_embeds, ctx_pos = [], []
        for gi in range(n_ctx):
            idx = dropped[gi * g: (gi + 1) * g]
            ctx_embeds.append(embeds[idx].mean(dim=0))
            ctx_pos.append(pos3[:, idx].float().mean(dim=1).round().long())
        new_embeds = torch.cat([embeds[dominant], torch.stack(ctx_embeds)], dim=0)
        new_pos = torch.cat([pos3[:, dominant], torch.stack(ctx_pos, dim=1)], dim=1)
        return new_embeds, new_pos

    def compress(self, visual_tokens: torch.Tensor, attn: torch.Tensor | None = None) -> torch.Tensor:
        """接口兼容实现：VisionZip 作用在编码器输出嵌入上，真正的入口是 generate()。"""
        new_vis, _ = self._compress_visual(visual_tokens[0],
                                           torch.arange(3).unsqueeze(1).expand(3, visual_tokens.shape[1]).to(visual_tokens.device))
        return new_vis.unsqueeze(0)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, model, inputs: dict, max_new_tokens: int = 32) -> torch.Tensor:
        """编码器侧压缩 + 手动贪心解码。"""
        image_pad_id = model.config.image_token_id
        input_ids = inputs["input_ids"]  # (1, L)
        B, L = input_ids.shape

        # --- 1. 视觉编码（merge 后）---
        embed_tokens = model.get_input_embeddings()
        full_embeds = embed_tokens(input_ids)[0]  # (L, D)
        grid = inputs.get("image_grid_thw", None)
        visual = model.visual(inputs["pixel_values"], grid_thw=grid)  # (N, D) merge 后
        n_vis = visual.shape[0]

        vis_start = (input_ids[0] == image_pad_id).nonzero(as_tuple=True)[0]
        assert vis_start.numel() == n_vis, f"image_pad({vis_start.numel()}) != visual({n_vis})"
        is_vis = torch.zeros(L, dtype=torch.bool, device=input_ids.device)
        is_vis[vis_start] = True
        text_pos = (~is_vis).nonzero(as_tuple=True)[0]

        # --- 2. M-RoPE 全序列位置 ---
        pos3, _ = model.model.get_rope_index(
            input_ids, grid, None, inputs.get("attention_mask", None))
        pos3 = pos3[:, 0, :]  # (3, L)，batch=1

        # --- 3. 压缩视觉段 ---
        vis_pos3 = pos3[:, vis_start]  # (3, N)
        new_vis, new_vpos = self._compress_visual(visual, vis_pos3)
        self.last_kept = int(new_vis.shape[0])

        # --- 4. 组装新序列：保持原顺序 [图前文本] + [压缩视觉段] + [图后文本]，
        #     否则生成会条件于视觉 token 而非 prompt 文本 ---
        text_pos = text_pos.cpu()
        first_vis, last_vis = int(vis_start[0]), int(vis_start[-1])
        before = text_pos[text_pos < first_vis]
        after = text_pos[text_pos > last_vis]
        emb_list, pos_list = [], []
        for i in before.tolist():
            emb_list.append(full_embeds[i:i+1]); pos_list.append(pos3[:, i:i+1])
        emb_list.append(new_vis);            pos_list.append(new_vpos)
        for i in after.tolist():
            emb_list.append(full_embeds[i:i+1]); pos_list.append(pos3[:, i:i+1])
        seq_embeds = torch.cat(emb_list, dim=0)
        seq_pos3 = torch.cat(pos_list, dim=1)  # (3, L')
        L2 = seq_embeds.shape[0]

        attn_mask = torch.ones(B, L2, dtype=torch.long, device=input_ids.device)

        # --- 5. prefill + 贪心 decode ---
        out = model(inputs_embeds=seq_embeds.unsqueeze(0).to(input_ids.device),
                    position_ids=seq_pos3.unsqueeze(1).to(input_ids.device),
                    attention_mask=attn_mask, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(dim=-1)
        generated = [next_id]
        for step in range(max_new_tokens - 1):
            # 文本续写位置：从原 prompt 的最大文本位置 L-1 之后接续
            position_ids = torch.full((3, B, 1), L + step, device=input_ids.device, dtype=torch.long)
            out = model(input_ids=next_id, past_key_values=past, use_cache=True,
                        position_ids=position_ids,
                        attention_mask=torch.ones(B, L2 + step + 1, device=input_ids.device, dtype=torch.long))
            next_id = out.logits[:, -1:].argmax(dim=-1)
            generated.append(next_id)
        return torch.cat(generated, dim=1)
