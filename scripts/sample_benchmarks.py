"""各 benchmark 固定 seed 抽样为 jsonl（保证结果可复现、可对比）。

输出: data/{pope,mmbench}_sample.jsonl，统一字段: {id, image, question, answer, choices?, split?}
- pope:      基于本地 COCO2017（--coco-root 指定路径）自建，POPE 官方逻辑
             （random/popular/adversarial 三种负采样策略，官方基于 val2014，
              此处用 val2017 重建，README 中需注明）
- mmbench:   官方 dev TSV（图片内嵌 base64），随机抽 500 条

用法:
    python scripts/sample_benchmarks.py --coco-root /path/to/coco
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

N = {"pope": 500, "mmbench": 500}


def write_jsonl(rows: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  写出 {len(rows)} 条 -> {path}")


# ---------------------------------------------------------------------------
# POPE: 自建（本地 COCO2017）
# ---------------------------------------------------------------------------
def load_coco_instances(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pope(coco_root: Path, n: int, seed: int) -> list[dict]:
    val = load_coco_instances(coco_root / "annotations" / "instances_val2017.json")
    train = load_coco_instances(coco_root / "annotations" / "instances_train2017.json")

    cat_name = {c["id"]: c["name"] for c in val["categories"]}
    img_file = {im["id"]: im["file_name"] for im in val["images"]}

    # 图片 -> GT 类别集合
    img_cats = defaultdict(set)
    for a in val["annotations"]:
        img_cats[a["image_id"]].add(a["category_id"])

    # popular: train2017 类别频次；adversarial: train2017 类别共现
    cat_freq = Counter()
    co_occur = defaultdict(Counter)
    anns = train["annotations"]
    img_cats_train = defaultdict(set)
    for a in anns:
        img_cats_train[a["image_id"]].add(a["category_id"])
        cat_freq[a["category_id"]] += 1
    for cats in img_cats_train.values():
        for c1 in cats:
            for c2 in cats:
                if c1 != c2:
                    co_occur[c1][c2] += 1

    # 只保留有标注的图
    eligible = sorted(img_cats.keys())
    rng = random.Random(seed)
    sampled_imgs = rng.sample(eligible, min(n // 2, len(eligible)))

    def sample_negative(gt_cats: set[int], strategy: str) -> int | None:
        candidates = set(cat_name.keys()) - gt_cats
        if strategy == "random":
            return rng.choice(sorted(candidates))
        if strategy == "popular":
            # 高频非 GT 类别
            pool = sorted(candidates, key=lambda c: -cat_freq[c])
            return pool[rng.randrange(min(25, len(pool)))]  # 取 Top-25 内随机
        if strategy == "adversarial":
            # 与 GT 类别共现最高的非 GT 类别
            score = Counter()
            for gt in gt_cats:
                for c, v in co_occur[gt].items():
                    if c not in gt_cats:
                        score[c] += v
            if not score:
                return None
            pool = sorted(score.keys(), key=lambda c: -score[c])
            return pool[rng.randrange(min(5, len(pool)))]
        return None

    rows = []
    strategies = ["random", "popular", "adversarial"]
    for i, img_id in enumerate(sampled_imgs):
        gt_cats = img_cats[img_id]
        fname = img_file[img_id]
        # 正样本：随机一个 GT 类别
        yes_cat = cat_name[rng.choice(sorted(gt_cats))]
        rows.append({
            "id": f"pope_yes_{img_id}",
            "image": str(coco_root / "images" / "val2017" / fname),
            "question": f"Is there a {yes_cat} in the image?",
            "answer": "yes",
            "split": strategies[i % 3],
        })
        # 负样本：轮换三种策略
        strat = strategies[i % 3]
        neg = sample_negative(gt_cats, strat)
        if neg is None:  # 无共现统计时退化为 random
            strat = "random"
            neg = sample_negative(gt_cats, "random")
        rows.append({
            "id": f"pope_no_{img_id}",
            "image": str(coco_root / "images" / "val2017" / fname),
            "question": f"Is there a {cat_name[neg]} in the image?",
            "answer": "no",
            "split": strat,
        })
    return rows


# ---------------------------------------------------------------------------
# MMBench: 官方 dev TSV（图片内嵌 base64）
# ---------------------------------------------------------------------------
def build_mmbench(tsv_path: Path, n: int, seed: int) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(tsv_path, sep="\t")
    rng = random.Random(seed)
    idx = list(df.index)
    rng.shuffle(idx)
    rows = []
    for i in idx[:n]:
        r = df.loc[i]
        choices = {k: str(r[k]) for k in "ABCD" if k in df.columns and pd.notna(r[k])}
        img_b64 = r["image"] if "image" in df.columns and pd.notna(r["image"]) else None
        rows.append({
            "id": str(r["index"]),
            "image": img_b64,  # base64，runner 内解码
            "question": str(r["question"]),
            "answer": str(r["answer"]),
            "choices": choices,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--coco-root", default="data/coco")
    ap.add_argument("--mmbench-tsv", default="data/raw/mmbench_dev_20230712.tsv")
    args = ap.parse_args()

    out = Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/2] 自建 POPE（本地 COCO2017）...")
    pope_rows = build_pope(Path(args.coco_root), N["pope"], args.seed)
    write_jsonl(pope_rows, out / "pope_sample.jsonl")

    print("[2/2] MMBench dev 抽样...")
    tsv = Path(args.mmbench_tsv)
    if tsv.exists():
        mmb_rows = build_mmbench(tsv, N["mmbench"], args.seed)
        write_jsonl(mmb_rows, out / "mmbench_sample.jsonl")
    else:
        print(f"  跳过: 未找到 {tsv}（先下载 MMBench dev TSV）")


if __name__ == "__main__":
    main()
