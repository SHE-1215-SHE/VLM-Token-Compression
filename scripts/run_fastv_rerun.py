"""等待 tome 队列结束后，用修复版 fastv 重跑全量（验证 + 3 budgets）。"""
from __future__ import annotations

import subprocess
import sys
import time

WAIT_PID = sys.argv[1]  # tome 队列进程号

if __name__ == "__main__":
    while True:
        try:
            with open(f"/proc/{WAIT_PID}/stat"):
                pass
        except FileNotFoundError:
            print(">> tome 队列已结束，开始 fastv 重跑", flush=True)
            break
        time.sleep(120)

    # 修复验证：20 条，确认剪枝方向与速度恢复
    subprocess.call([sys.executable, "-m", "src.eval.runner", "--method", "fastv",
                     "--budget", "0.33", "--benchmarks", "pope", "--limit", "20"])
    # 全量
    for b in ["0.33", "0.11", "0.055"]:
        subprocess.call([sys.executable, "-m", "src.eval.runner", "--method", "fastv",
                         "--budget", b, "--benchmarks", "pope,mmbench"])
    print(">> fastv 重跑完成", flush=True)
