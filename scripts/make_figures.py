"""W4 结论图：从 results/*.json 汇总生成精度-预算曲线与实测时延图。

输出（results/figures/）：
- accuracy_vs_budget.png：POPE F1 / MMBench Acc 随预算变化（三方法 + baseline 参考线）
- latency.png：各方法 x 预算的实测端到端时延（对比 baseline），凸显"理论加速 != 实测加速"

运行：python scripts/make_figures.py --out results
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["fastv", "visionzip", "tome"]
BUDGETS = [0.33, 0.11, 0.055]
BUDGET_LABEL = {1.0: "100%", 0.33: "33%", 0.11: "11%", 0.055: "5.5%"}
COLOR = {"fastv": "tab:blue", "visionzip": "tab:red", "tome": "tab:green"}
FIRST_LETTER = re.compile(r"^\(?([A-D])\)?[.。:]?", re.I)


def load(out_dir: Path, bench: str, method: str, budget: float) -> dict | None:
    p = out_dir / f"{bench}_{method}_{budget}.json"
    if not p.exists():
        return None
    return json.load(open(p, encoding="utf-8"))


def first_letter_acc(result: dict) -> float:
    """按回复首字母解析的 acc（结构性口径：首 token 来自未修改的 prefill logits）。"""
    hits = 0
    for p, g in zip(result["preds"], result["gts"]):
        m = FIRST_LETTER.match(p.strip())
        hits += bool(m and m.group(1).upper() == g)
    return hits / max(len(result["gts"]), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    base = {b: load(out_dir, b, "baseline", 1.0) for b in ("pope", "mmbench")}
    assert all(base.values()), "缺 baseline 结果，先跑 runner"

    # ---------- 图 1：精度 vs 预算 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, bench, metric, ylab in [
        (axes[0], "pope", "f1", "POPE F1"),
        (axes[1], "mmbench", "acc", "MMBench Acc"),
    ]:
        ax.axhline(base[bench]["scores"][metric], color="gray", ls="--", lw=1,
                   label=f"baseline ({BUDGET_LABEL[1.0]})")
        for m in METHODS:
            xs, ys = [], []
            for bud in BUDGETS:
                r = load(out_dir, bench, m, bud)
                if r is None:
                    continue
                xs.append(bud * 100)
                ys.append(first_letter_acc(r) if (bench == "mmbench" and m == "tome")
                          else r["scores"][metric])
            ax.plot(xs, ys, "o-", color=COLOR[m], label=m)

        # ToMe 标准解析口径（退化文本解析失败的"假崩"）单独标出
        if bench == "mmbench":
            xs, ys = [], []
            for bud in BUDGETS:
                r = load(out_dir, bench, "tome", bud)
                if r:
                    xs.append(bud * 100)
                    ys.append(r["scores"]["acc"])
            if xs:
                ax.plot(xs, ys, "x--", color=COLOR["tome"], alpha=0.55,
                        label="tome (strict parse, artifact)")

        ax.set_xscale("log")
        ax.set_xticks([100, 33, 11, 5.5])
        ax.set_xticklabels(["100%", "33%", "11%", "5.5%"])
        ax.invert_xaxis()
        ax.set_xlabel("visual token budget (kept %)")
        ax.set_ylabel(ylab)
        ax.set_ylim(0.3, 1.0) if bench == "mmbench" else ax.set_ylim(0.85, 0.98)
        ax.set_title(bench.upper())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Qwen2.5-VL-3B: accuracy vs visual token budget")
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_vs_budget.png", dpi=150)
    plt.close(fig)

    # ---------- 图 2：实测时延 ----------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels, vals, cols = [], [], []
    pairs = [("baseline", 1.0, "gray")]
    for m in METHODS:
        pairs += [(m, b, COLOR[m]) for b in BUDGETS]
    for m, b, c in pairs:
        r = load(out_dir, "pope", m, b)  # 两榜时延相近，取 POPE
        if r is None:
            continue
        labels.append(f"{m}\n{BUDGET_LABEL[b]}")
        vals.append(r["mean_gen_ms"] / 1000)
        cols.append(c)
    bars = ax.bar(range(len(vals)), vals, color=cols, alpha=0.85)
    base_s = base["pope"]["mean_gen_ms"] / 1000
    ax.axhline(base_s, color="gray", ls="--", lw=1)
    ax.text(len(vals) - 0.5, base_s + 0.06, f"baseline {base_s:.2f}s", ha="right", fontsize=8, color="gray")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}s", ha="center", fontsize=8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean generation time per item (s)")
    ax.set_title("Wall-clock per item (POPE, 4090, batch=1) — compression is a net slowdown")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "latency.png", dpi=150)
    plt.close(fig)

    print(f"图已生成: {fig_dir / 'accuracy_vs_budget.png'}")
    print(f"          {fig_dir / 'latency.png'}")


if __name__ == "__main__":
    main()
