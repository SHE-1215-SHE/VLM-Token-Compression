"""无压缩基线全量跑（W1 任务 6 的入口）。"""
from __future__ import annotations

import subprocess
import sys

CMD = [
    sys.executable, "-m", "src.eval.runner",
    "--config", "configs/qwen25vl_3b.yaml",
    "--method", "baseline",
    "--budget", "1.0",
]

if __name__ == "__main__":
    raise SystemExit(subprocess.call(CMD + sys.argv[1:]))
