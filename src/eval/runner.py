"""评测主入口：method x budget x benchmark 三重循环。

W1：baseline（budget=1.0）；方法插桩在 W2 接入。
运行示例：
    python -m src.eval.runner --config configs/qwen25vl_3b.yaml \
        --method baseline --budget 1.0 --benchmarks pope --limit 50
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import torch

from src.eval.benchmarks import gqa, mmbench, pope, textvqa
from src.models.loader import load_from_config
from src.methods import build_method
from src.profiling.wallclock import Timer

BENCH_REGISTRY = {"pope": pope, "gqa": gqa, "textvqa": textvqa, "mmbench": mmbench}


def build_messages(item: dict) -> list[dict]:
    """统一把 jsonl 条目转成 Qwen-VL 消息格式。image 支持本地路径或 base64。"""
    import os

    from PIL import Image

    img = item["image"]
    if isinstance(img, str) and len(img) < 256 and os.path.exists(img):
        pass  # 本地路径
    elif isinstance(img, str) and img:
        img = Image.open(io.BytesIO(base64.b64decode(img))).convert("RGB")  # base64
    else:
        img = None
    content = [{"type": "image", "image": img}] if img is not None else []
    content.append({"type": "text", "text": BENCH_REGISTRY[item["_bench"]].prompt_of(item)})
    return [{"role": "user", "content": content}]


@torch.no_grad()
def run_one(model, processor, bench_name: str, items: list[dict], limit: int, method=None) -> dict:
    """单 benchmark 评测：逐条 generate（或压缩方法） -> 解析 -> 打分。"""
    bench = BENCH_REGISTRY[bench_name]
    from qwen_vl_utils import process_vision_info

    preds, gts, timings = [], [], []
    if limit:
        items = items[:limit]
    for k, item in enumerate(items):
        item = {**item, "_bench": bench_name}
        messages = build_messages(item)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

        with Timer() as t:
            if method is None or method.name == "baseline":
                ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
                # generate 返回含 prompt 的完整序列，截掉只留新 token（与压缩方法口径一致）
                ids = ids[:, inputs["input_ids"].shape[1]:]
            else:
                # 压缩方法：内部做 prefill/剪枝/贪心解码，返回 (B,T) 生成 ids
                ids = method.generate(model, inputs, max_new_tokens=32)
        reply = processor.batch_decode(ids, skip_special_tokens=True)[0]

        if bench_name == "textvqa":
            gts.append(item["answers"] if "answers" in item else [item["answer"]])
        else:
            gts.append(item["answer"])
        preds.append(reply)
        timings.append(t.elapsed_ms)
        if (k + 1) % 50 == 0:
            print(f"  [{bench_name}] {k + 1}/{len(items)}  mean={sum(timings)/len(timings):.0f}ms")

    if bench_name == "textvqa":
        scores = bench.score(preds, gts)
    elif hasattr(bench, "parse_answer"):
        scores = bench.score([bench.parse_answer(p) for p in preds], gts)
    else:  # gqa: 宽松匹配直接用原始输出
        scores = bench.score(preds, gts)
    return {"scores": scores, "mean_gen_ms": sum(timings) / max(len(timings), 1),
            "preds": preds, "gts": gts, "timings": timings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen25vl_3b.yaml")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--benchmarks", default="pope,mmbench")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=0, help="调试用样本数上限，0=全量")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)

    model, processor, cfg = load_from_config(args.config)
    method = build_method(args.method, args.budget) if args.method != "baseline" else None
    print(f"模型已加载: {args.config}  method={args.method} budget={args.budget}")

    for bench_name in args.benchmarks.split(","):
        bench_name = bench_name.strip()
        sample_path = Path(args.data_dir) / f"{bench_name}_sample.jsonl"
        if not sample_path.exists():
            print(f"跳过 {bench_name}: 缺 {sample_path}（先跑 scripts/sample_benchmarks.py）")
            continue
        items = [json.loads(l) for l in open(sample_path, encoding="utf-8")]
        print(f"[{bench_name}] 共 {len(items)} 条，开始 baseline 评测...")
        t0 = time.time()
        result = run_one(model, processor, bench_name, items, args.limit, method)
        result["wall_s"] = time.time() - t0
        result["method"], result["budget"] = args.method, args.budget
        result["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3

        tag = f"{args.method}_{args.budget}"
        with open(out_dir / f"{bench_name}_{tag}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print(f"[{bench_name}] 指标: {result['scores']}  用时 {result['wall_s']:.0f}s  峰值显存 {result['peak_mem_gb']:.1f}GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
