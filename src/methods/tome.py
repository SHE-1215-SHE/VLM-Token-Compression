"""ToMe-VLM：把 ToMe (NeurIPS'23) 的 bipartite soft matching 迁移到 VLM 的 KV cache。

与 ViT 版的差异（迁移适配点）：
1. 作用位置：不在编码器内逐层合并，而在 LLM 第 K 层 prefill 后一次性合并视觉 token
   的 KV（保留 token 数 = budget 比例）；
2. 相似度来源：hook 第 K 层 k_proj 输出（GQA: num_kv_heads×head_dim），视觉 token
   两两余弦相似度，head 取均值——不需要 output_attentions，prefill 更快；
3. 匹配：ToMe 二分匹配——视觉序列按位置奇偶分 A/B 两集，每个 A 找最相似的 B，
   按 sim 降序贪心取互不相交的 r 对，合并到较早位置；
4. KV 合并：获胜槽的 K/V = 两 token 均值（K 已含 RoPE 旋转，取均值近似"平均位置"，
   文献中 ToMe-on-KV 的通用做法，README 注明近似误差）；
5. 位置：获胜槽沿用较早 token 的烘焙位置，无需重排 M-RoPE。

r 调度：budget<1 时 r = N_vis - N_keep；--adaptive 预留（内容自适应配额，见 W3）。
参考: https://github.com/facebookresearch/ToMe
"""
from __future__ import annotations

import torch

from .base import CompressionMethod


class ToMeVLM(CompressionMethod):
    name = "tome"

    def __init__(self, budget: float = 0.5, k_layer: int = 2, **kwargs):
        super().__init__(budget, **kwargs)
        self.k_layer = k_layer
        self.last_kept = 0
        self.last_merged = 0

    # ------------------------------------------------------------------
    def _merge_plan(self, k_vis: torch.Tensor, r: int):
        """k_vis: (Hkv, Nv, D) 第 K 层视觉 token 的 key。
        返回 {最终根槽位: [被并入它的 token 列表]}。

        注意：单轮二分匹配的互不相交配对上限是 ⌈Nv/2⌉，而一次合并 r≈Nv 时
        r 远超该上限，因此官方 ToMe"逐层小 r"的做法在这里必须改为多轮迭代：
        每轮在存活 token 上重算相似度配对，直到移除满 r 个。跨轮会出现
        "获胜方又被合并"的链，因此用 parent 指针记账、最后统一解析到根。"""
        Hkv, Nv, D = k_vis.shape
        parent: dict[int, int] = {}
        removed: set[int] = set()
        alive = list(range(Nv))
        while len(removed) < r and len(alive) > 1:
            a_glob = alive[0::2]  # 偶数位 A 集
            b_glob = alive[1::2]  # 奇数位 B 集
            if not a_glob or not b_glob:
                break
            ia = torch.tensor(a_glob, device=k_vis.device)
            ib = torch.tensor(b_glob, device=k_vis.device)
            ka = torch.nn.functional.normalize(k_vis[:, ia].mean(0), dim=-1)  # (nA, D)
            kb = torch.nn.functional.normalize(k_vis[:, ib].mean(0), dim=-1)  # (nB, D)
            sim = ka @ kb.T  # (nA, nB)
            best_b = sim.argmax(dim=1)  # (nA,)
            best_s = sim.gather(1, best_b[:, None]).squeeze(1)  # (nA,)
            order = best_s.argsort(descending=True)

            chosen: set[int] = set()
            for ai in order.tolist():
                if len(removed) + len(chosen) >= r:
                    break
                b_g = b_glob[int(best_b[ai])]
                if b_g in removed or b_g in chosen:
                    continue  # 本轮该目的地已被占用
                parent[b_g] = a_glob[ai]
                chosen.add(b_g)
            if not chosen:
                break
            removed |= chosen
            alive = [g for g in alive if g not in removed]

        def _root(x: int) -> int:
            while x in parent:
                x = parent[x]
            return x

        groups: dict[int, list[int]] = {}
        for g in removed:
            groups.setdefault(_root(g), []).append(g)
        return groups

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, model, inputs: dict, max_new_tokens: int = 32) -> torch.Tensor:
        k_layer = min(self.k_layer, len(model.model.language_model.layers) - 1)
        image_pad_id = model.config.image_token_id
        input_ids = inputs["input_ids"]
        B, L = input_ids.shape

        # --- 1. prefill + hook 第 K 层 k_proj ---
        cache_box = {}
        handle = model.model.language_model.layers[k_layer].self_attn.k_proj.register_forward_hook(
            lambda mod, inp, out: cache_box.__setitem__("k", out.detach()))
        out = model(**inputs, use_cache=True)
        handle.remove()
        past = out.past_key_values
        visual_pos = (input_ids[0] == image_pad_id).nonzero(as_tuple=True)[0]
        Nv = visual_pos.numel()

        n_keep = Nv if self.budget >= 1.0 else max(1, int(Nv * self.budget))
        r = max(0, Nv - n_keep)
        self.last_kept = n_keep

        if r > 0 and Nv >= 2:
            Hkv = model.config.text_config.num_key_value_heads
            D = cache_box["k"].shape[-1] // Hkv
            k_vis = cache_box["k"][0][visual_pos].reshape(Nv, Hkv, D).permute(1, 0, 2)  # (Hkv, Nv, D)
            groups = self._merge_plan(k_vis, r)
            losers = {j for js in groups.values() for j in js}
            self.last_merged = len(losers)

            # --- 2. 重建 KV：合并组均值，其余照抄 ---
            src = [v for v in range(Nv) if v not in losers]  # 根槽位，按原顺序排列
            src_t = torch.tensor(src, device=input_ids.device)
            win_local = [src.index(i) for i in groups.keys()]
            grp = [torch.tensor([i] + js, device=input_ids.device) for i, js in groups.items()]

            from transformers.cache_utils import DynamicCache
            pruned = DynamicCache()
            L2 = L - len(losers)
            for li in range(len(past)):
                nk = past.key_cache[li][:, :, src_t, :].clone()  # (B,H,L2,D)
                nv = past.value_cache[li][:, :, src_t, :].clone()
                for pos, idxs in zip(win_local, grp):
                    nk[:, :, pos, :] = past.key_cache[li][:, :, idxs, :].mean(dim=2)
                    nv[:, :, pos, :] = past.value_cache[li][:, :, idxs, :].mean(dim=2)
                pruned.update(nk, nv, li)

            # --- 3. 解码循环（与 FastV 相同，位置从原 prompt 末位续接）---
            next_id = out.logits[:, -1:].argmax(dim=-1)
            generated = [next_id]
            for step in range(max_new_tokens - 1):
                position_ids = torch.full((3, B, 1), L + step, device=input_ids.device, dtype=torch.long)
                out = model(input_ids=next_id, past_key_values=pruned, use_cache=True,
                            position_ids=position_ids,
                            attention_mask=torch.ones(B, L2 + step + 1, device=input_ids.device, dtype=torch.long))
                next_id = out.logits[:, -1:].argmax(dim=-1)
                generated.append(next_id)
            return torch.cat(generated, dim=1)

        # budget=1.0 或无法合并：走与 FastV 一致的手动解码
        next_id = out.logits[:, -1:].argmax(dim=-1)
        generated = [next_id]
        for step in range(max_new_tokens - 1):
            position_ids = torch.full((3, B, 1), L + step, device=input_ids.device, dtype=torch.long)
            out = model(input_ids=next_id, past_key_values=past, use_cache=True,
                        position_ids=position_ids,
                        attention_mask=torch.ones(B, L + step + 1, device=input_ids.device, dtype=torch.long))
            next_id = out.logits[:, -1:].argmax(dim=-1)
            generated.append(next_id)
        return torch.cat(generated, dim=1)

    def compress(self, visual_tokens: torch.Tensor, attn: torch.Tensor | None = None) -> torch.Tensor:
        """接口兼容实现：ToMe 作用在 KV cache 上，真正的入口是 generate()。"""
        return visual_tokens
