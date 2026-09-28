"""统一加载 Qwen2.5-VL。

W1 验收点：
- attn_implementation 必须是 "eager"，否则 hook 拿不到 attention 权重；
- 返回 (model, processor)，processor 负责图像预处理与 token 展开。
"""
from __future__ import annotations

import torch
import yaml
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def load_from_config(config_path: str):
    """返回 (model, processor, cfg)。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    keys = {"model_path", "dtype", "attn_implementation", "device_map", "load_in_8bit", "max_pixels"}
    model, processor = load_model(**{k: v for k, v in cfg.items() if k in keys})
    return model, processor, cfg


def load_model(
    model_path: str,
    dtype: str = "bfloat16",
    attn_implementation: str = "eager",
    device_map: str = "cuda:0",
    load_in_8bit: bool = False,
    max_pixels: int | None = None,
):
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        load_in_8bit=load_in_8bit if load_in_8bit else None,
    )
    # eager attention 显存随分辨率平方涨，需封顶视觉 token 数
    proc_kwargs = {"max_pixels": max_pixels} if max_pixels else {}
    processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)
    return model, processor
