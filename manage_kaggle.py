#!/usr/bin/env python3
"""
ProductLens 3.0: Kaggle CLI Orchestrator
Pushes notebooks, monitors running sessions, and downloads evaluation artifacts.
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

KERNEL_CONFIGS = {
    "gold_eval": {
        "ref": "parthsarthi2803/productlens-3-gold-eval",
        "dir": BASE_DIR / "kaggle" / "stage2_gold_eval",
    },
    "silver_label": {
        "ref": "parthsarthi2803/productlens-3-silver-label",
        "dir": BASE_DIR / "kaggle" / "stage2_silver_label",
    },
}
OUTPUTS_DIR = BASE_DIR / "outputs"


def run_cmd(cmd, check=True):
    print(f"🚀 Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if check and res.returncode != 0:
        print(f"❌ Command failed with code {res.returncode}")
        sys.exit(res.returncode)
    return res.returncode


def test_auth():
    print("🔑 Verifying Kaggle API credentials...")
    res = subprocess.run(["kaggle", "datasets", "list", "-m"], capture_output=True, text=True)
    if res.returncode == 0:
        print("✅ Authentication successful! Kaggle connected.")
    else:
        print("❌ Authentication failed. Please check ~/.kaggle/access_token or ~/.kaggle/kaggle.json")
        print(res.stderr)
        sys.exit(1)


def push_kernel(target="gold_eval"):
    test_auth()
    cfg = KERNEL_CONFIGS.get(target, KERNEL_CONFIGS["gold_eval"])
    print(f"📦 Pushing kernel to Kaggle ({cfg['ref']}) from {cfg['dir']}...")
    run_cmd(["kaggle", "kernels", "push", "-p", str(cfg["dir"])])
    print(f"\n✅ Kernel pushed successfully! Running in headless GPU background worker.")
    print(f"Track status with: python manage_kaggle.py status --target {target}")


def check_status(target="gold_eval"):
    cfg = KERNEL_CONFIGS.get(target, KERNEL_CONFIGS["gold_eval"])
    run_cmd(["kaggle", "kernels", "status", cfg["ref"]], check=False)


def download_outputs(target="gold_eval"):
    cfg = KERNEL_CONFIGS.get(target, KERNEL_CONFIGS["gold_eval"])
    out_dir = OUTPUTS_DIR / target
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading outputs from {cfg['ref']} to {out_dir}...")
    run_cmd(["kaggle", "kernels", "output", cfg["ref"], "-p", str(out_dir)])
    print(f"✅ Outputs downloaded to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="ProductLens 3.0 Kaggle Orchestrator")
    parser.add_argument(
        "--target",
        choices=["gold_eval", "silver_label"],
        default="gold_eval",
        help="Target kernel to operate on",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    for cmd in ["auth", "push", "status", "download"]:
        sub = subparsers.add_parser(cmd)
        sub.add_argument("--target", choices=["gold_eval", "silver_label"], default="gold_eval")

    args = parser.parse_args()
    target = getattr(args, "target", None) or "gold_eval"

    if args.command == "auth":
        test_auth()
    elif args.command == "push":
        push_kernel(target)
    elif args.command == "status":
        check_status(target)
    elif args.command == "download":
        download_outputs(target)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
