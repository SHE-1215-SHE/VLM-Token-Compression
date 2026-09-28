"""method x budget 扫描（W3 主实验）。"""
from __future__ import annotations

import itertools
import subprocess
import sys

METHODS = ["fastv", "visionzip"]
BUDGETS = [0.33, 0.11, 0.055]

if __name__ == "__main__":
    for m, b in itertools.product(METHODS, BUDGETS):
        cmd = [sys.executable, "-m", "src.eval.runner",
               "--config", "configs/qwen25vl_3b.yaml", "--method", m, "--budget", str(b)]
        print(">>", " ".join(cmd), flush=True)
        subprocess.call(cmd)
