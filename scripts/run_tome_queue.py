"""等待主 sweep 结束后，追加跑 tome 的验证与全量（避免双进程 OOM）。"""
from __future__ import annotations

import subprocess
import sys
import time

MAIN_SWEEP_PID = sys.argv[1]  # 运行中的 run_sweep.py 进程号

if __name__ == "__main__":
    # 1. 等主 sweep 退出
    while True:
        try:
            with open(f"/proc/{MAIN_SWEEP_PID}/stat") as f:
                pass
        except FileNotFoundError:
            print(">> 主 sweep 已结束", flush=True)
            break
        time.sleep(120)

    # 2. tome 快速验证（各 20 条，确认剪枝逻辑正常）
    for b in ["0.33", "0.11"]:
        subprocess.call([sys.executable, "-m", "src.eval.runner", "--method", "tome",
                         "--budget", b, "--benchmarks", "pope", "--limit", "20"])
    # 3. tome 全量
    for b in ["0.33", "0.11", "0.055"]:
        subprocess.call([sys.executable, "-m", "src.eval.runner", "--method", "tome",
                         "--budget", b, "--benchmarks", "pope,mmbench"])
    print(">> tome 队列完成", flush=True)
