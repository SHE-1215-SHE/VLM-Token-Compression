"""W4 Gradio demo：上传图片 + 提问，对比 baseline 与三种 token 压缩方法的回答与耗时。

运行（需 GPU，单卡 7.7GB 峰值，与训练进程并存时注意显存）：
    python scripts/demo.py
然后浏览器打开 http://127.0.0.1:7860
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# 模型全部在本地缓存，强制离线，避免 transformers 启动时连 huggingface.co 重试卡住
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 任意目录可直接启动

import gradio as gr
import torch

# gradio_client 1.3.0 已知 bug：schema 中 additionalProperties 为 bool 时
# json_schema_to_python_type 抛 TypeError/APIInfoParseError，导致 /info 500、
# launch 误判 localhost 不可访问。此处打补丁兼容（blocks.py 经模块属性访问，可生效）。
import gradio_client.utils as _gc_utils

_orig_get_type = _gc_utils.get_type
_gc_utils.get_type = lambda schema: (_orig_get_type(schema)
                                     if isinstance(schema, dict) else "Any")
_orig_json2py = _gc_utils.json_schema_to_python_type
def _safe_json2py(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    try:
        return _orig_json2py(schema, defs)
    except Exception:
        return "Any"
_gc_utils.json_schema_to_python_type = _safe_json2py

from src.methods import build_method
from src.models.loader import load_from_config

ROOT = Path(__file__).resolve().parent.parent
MODEL, PROCESSOR, CFG = load_from_config(str(ROOT / "configs" / "qwen25vl_3b.yaml"))
METHODS = ["baseline", "fastv", "visionzip", "tome"]


def _build_inputs(image, question: str) -> dict:
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return PROCESSOR(text=[text], images=image_inputs, padding=True,
                     return_tensors="pt").to("cuda")


@torch.no_grad()
def predict(image, question: str, method: str, budget: float):
    if image is None:
        return "请先上传图片", ""
    inputs = _build_inputs(image, question)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if method == "baseline":
        ids = MODEL.generate(**inputs, max_new_tokens=32, do_sample=False)
        # generate 返回含 prompt 的完整序列，截掉只留新 token
        ids = ids[:, inputs["input_ids"].shape[1]:]
        extra = ""
    else:
        m = build_method(method, float(budget))
        ids = m.generate(MODEL, inputs, max_new_tokens=32)
        kept = getattr(m, "last_kept", None)
        merged = getattr(m, "last_merged", None)
        extra = f"  kept={kept} merged={merged}" if kept is not None else ""
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30

    answer = PROCESSOR.batch_decode(ids, skip_special_tokens=True)[0]
    info = (f"method={method} budget={budget}{extra}  |  "
            f"time={elapsed:.2f}s  peak_mem={peak:.1f}GB")
    return answer, info


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VLM Token Compression Demo") as ui:
        gr.Markdown(
            "## VLM Token Compression Demo (Qwen2.5-VL-3B)\n"
            "对比 baseline 与 FastV / VisionZip / ToMe-VLM 的回答、耗时与显存。\n\n"
            "**预期行为**（README 发现 1/3 的直观呈现）：\n"
            "- baseline：正常回答；\n"
            "- **开放式描述**（多 token 生成）下，压缩方法可能退化（role-marker 循环 / 字符粘连 / 答非所问），"
            "KV 侧方法（FastV/ToMe）尤其明显；\n"
            "- **yes/no 或选择题**（单 token 答案）下，FastV/ToMe 的答案与 baseline 完全一致（首 token 来自未修改的 prefill）；\n"
            "- VisionZip 是唯一随预算真实掉精度的方法（编码器侧压缩）。")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="输入图片")
                question = gr.Textbox(label="问题",
                                      value="Describe this image briefly.")
                method = gr.Radio(METHODS, value="baseline", label="方法")
                budget = gr.Slider(0.01, 1.0, value=0.33, step=0.01,
                                   label="budget（保留的视觉 token 比例）")
                btn = gr.Button("生成", variant="primary")
            with gr.Column():
                answer = gr.Textbox(label="回答", lines=4)
                info = gr.Textbox(label="运行信息", lines=1)
        btn.click(predict, [image, question, method, budget], [answer, info])
    return ui


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
