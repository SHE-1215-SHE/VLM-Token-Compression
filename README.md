# VLM-Token-Compression-Bench

A unified benchmark for **training-free visual token compression** in Multimodal LLMs, with real wall-clock measurements on consumer GPUs.

> 状态：进行中（W1）。本 README 为骨架初稿，结果表将随评测推进填充。

## TL;DR

- 评测对象：Qwen2.5-VL-3B / 7B
- 方法：FastV、VisionZip、CROP-R（training-free 视觉 token 压缩）
- 指标：任务精度（POPE / GQA / TextVQA / MMBench 子集）+ **真实延迟/显存**（非 FLOPs 代理）
- 核心发现（占位）：token 预算降到 XX 以下时，wall-clock 收益与精度损失如何权衡；OCR/细粒度任务为何崩

## Why

多数论文只报 FLOPs / MACs，但视觉 token 压缩在 decode 阶段收益有限、prefill 收益受 kernel 影响，**消费级 GPU 上 wall-clock 经常不降反升**——这是公开痛点，本仓库给出系统实测。

## Methods

| Method | 压缩位置 | 策略 | 来源 |
|---|---|---|---|
| FastV | LLM decoder 第 K 层 | 按 attention 分数剪视觉 token | ECCV'24 |
| VisionZip | 视觉编码器输出 | 主导 token 选择 + 上下文合并 | CVPR'24 |
| CROP-R | MLLM | 细粒度冗余抑制（re-weight + prune） | arXiv'25 |

## Results（模板，待填）

| Method | Budget | POPE-F1 | GQA-Acc | TextVQA | MMBench | Prefill (ms) | Decode (ms) | Peak Mem (GB) |
|---|---|---|---|---|---|---|---|---|
| baseline (576) | 100% | | | | | | | |
| FastV | 33% | | | | | | | |
| ... | | | | | | | | |

## 文献脉络：ViT Token 选择 → VLM Token 压缩

（面试/讲述用主线）DynamicViT / EViT / NAP / MAT / ToMe / FasterViT / LVTP / MPM 等在纯分类 ViT 上做 token 剪枝/合并；FastV 本质是 DynamicViT 的 decoder 版（用 [CLS]→最后文本 token 的 attention 替代）。本仓库把这条线统一放到 VLM 评测下检验。

## Quick Start

```bash
pip install -r requirements.txt
python scripts/check_env.py          # W1 自检：推理 + token 数 + attention hook
python scripts/run_baseline.py       # 无压缩基线
```

## Roadmap

- [ ] W1: 推理 harness + hook 验证 + 基线
- [ ] W2: FastV / VisionZip 插桩
- [ ] W3: 全量 sweep + wall-clock 表
- [ ] W4: 结论图 + Gradio demo
- [ ] 想法：视觉编码器侧压缩 × LLM 侧压缩 = 双端压缩（结合轻量 ViT 经验）

## License / Citation

TBD
