"""W1 自检脚本：一条命令验证技术路线是否走得通。

依次检查：
  [1] CUDA 环境与显存
  [2] Qwen2.5-VL-3B 加载（eager attention）
  [3] 单图单问推理
  [4] dynamic resolution 下真实视觉 token 数（关键地基）
  [5] attention hook 能否拿到 (1, H, L, L) 权重矩阵  <-- 最大风险点

全部 PASS 即可进入 W2；[5] 失败则降级为拷贝 HF modeling 文件自改 forward。
"""
from __future__ import annotations

import sys

import torch

MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"


def check(n: int, ok: bool, msg: str):
    print(f"[{n}] {'PASS' if ok else 'FAIL'} - {msg}")
    return ok


def main():
    ok = True

    # [1] 环境
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        ok &= check(1, True, f"GPU={torch.cuda.get_device_name(0)} 显存 {free/1024**3:.1f}/{total/1024**3:.1f}GB free")
    else:
        return check(1, False, "无 CUDA，后续检查跳过")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # [2] 加载
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    n_layers = len(model.model.language_model.layers)
    ok &= check(2, True, f"模型加载成功，decoder 层数={n_layers}")

    # [3] 推理
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": IMAGE_URL},
            {"type": "text", "text": "Describe this image in one sentence."},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    reply = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    ok &= check(3, bool(reply.strip()), f"输出: {reply[:80].strip()!r}")

    # [4] 真实视觉 token 数
    grid = getattr(inputs, "image_grid_thw", None)
    merge = model.config.vision_config.spatial_merge_size ** 2
    if grid is not None:
        raw = int(grid.prod())
        tokens = raw // merge
        ok &= check(4, tokens > 0, f"grid_thw={grid.tolist()} -> 原始patch={raw} -> merge后视觉token={tokens}（参考576）")
    else:
        ok &= check(4, False, "未取到 image_grid_thw")

    # [5] attention hook
    from src.models.hooks.attention_probe import AttentionProbe

    probe = AttentionProbe(model.model.language_model.layers[2])
    out = model(**inputs, output_attentions=True)
    probe.remove()
    a = probe.last
    ok &= check(5, a is not None and a.dim() == 4, f"attn shape={tuple(a.shape) if a is not None else None}")

    print("\n结论:", "W1 技术路线打通，可进 W2" if ok else "有 FAIL 项，按提示降级处理")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
