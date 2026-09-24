"""method x budget 扫描（W3 主实验）。"""
from __future__ import annotations

import itertools
import subprocess
import sys

METHODS = ["baseline", "fastv", "visionzip", "crop_r"]
BUDGETS = [1.0, 0.33, 0.11, 0.055]

if __name__ == "__main__":
    for m, b in itertools.product(METHODS, BUDGETS):
        if m == "baseline" and b != 1.0:
            continue
        cmd = [sys.executable, "-m", "src.eval.runner",
               "--config", "configs/qwen25vl_3b.yaml", "--method", m, "--budget", str(b)]
        print(">>", " ".join(cmd))
        subprocess.call(cmd)
