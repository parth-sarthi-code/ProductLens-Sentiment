#!/usr/bin/env python3
"""
Monitors parthsarthi2803/productlens-3-gold-eval until completion,
then automatically downloads the results.
"""

import subprocess
import sys
import time
from pathlib import Path

KERNEL_REF = "parthsarthi2803/productlens-3-gold-eval"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "gold_eval"


def get_status() -> str:
    res = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL_REF],
        capture_output=True,
        text=True,
    )
    for line in res.stdout.splitlines():
        if "has status" in line:
            return line.split('"')[1]
    return "UNKNOWN"


def download_outputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading outputs to {OUTPUT_DIR}...")
    subprocess.run(
        ["kaggle", "kernels", "output", KERNEL_REF, "-p", str(OUTPUT_DIR)],
        check=True,
    )
    print("✅ Outputs downloaded successfully.")


def main():
    print(f"👀 Monitoring {KERNEL_REF}...")
    max_wait_sec = 3600  # 60 min max
    t0 = time.time()
    last_status = None

    while time.time() - t0 < max_wait_sec:
        status = get_status()
        if status != last_status:
            elapsed = int(time.time() - t0)
            print(f"[{elapsed}s] Status changed: {status}")
            last_status = status

        if status == "KernelWorkerStatus.COMPLETE":
            print("🎉 Kernel completed successfully!")
            download_outputs()
            sys.exit(0)
        elif status == "KernelWorkerStatus.ERROR":
            print("❌ Kernel failed with ERROR.")
            download_outputs()
            sys.exit(1)

        time.sleep(20)

    print("⚠️ Timed out waiting for kernel.")
    sys.exit(2)


if __name__ == "__main__":
    main()
