#!/usr/bin/env python3
"""S1: 监督学习 — 8×8 / 10 雷

学会基础模式识别：1-2-1、边角数字、flood fill 触发条件等。
这是整个训练流程里唯一有"正确答案"的阶段。

用法:
    python scripts/train_s1.py                    # 完整流程
    python scripts/train_s1.py --skip_data          # 跳过数据生成
    python scripts/train_s1.py --eval_only           # 仅评估已有 checkpoint
"""

import argparse
import subprocess
import sys
from pathlib import Path

STAGE = "S1"
DEFAULTS = {
    "width": 8, "height": 8, "mines": 10,
    "n_samples": 10000, "epochs": 20,
    "data_dir": "data/S1", "save_dir": "checkpoints/S1",
}


def run(cmd, desc=""):
    print(f"\n── {desc}")
    print(f"   $ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"❌ {desc}")
        sys.exit(1)
    print(f"✅ {desc}")


def main():
    p = argparse.ArgumentParser(description="S1: Supervised training")
    p.add_argument("--skip_data", action="store_true", help="Skip data generation")
    p.add_argument("--eval_only", action="store_true", help="Only evaluate existing checkpoint")
    p.add_argument("--device", default="auto")
    p.add_argument("--n_games", type=int, default=500, help="Eval games")
    args = p.parse_args()

    if args.device == "auto":
        import torch
        args.device = "mps" if torch.backends.mps.is_available() else \
                      "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    # 1) Generate
    if not args.skip_data and not args.eval_only:
        run([
            sys.executable, "scripts/generate_data.py",
            "--width", str(DEFAULTS["width"]),
            "--height", str(DEFAULTS["height"]),
            "--mines", str(DEFAULTS["mines"]),
            "--n_samples", str(DEFAULTS["n_samples"]),
            "--output", DEFAULTS["data_dir"],
            "--require_win",  # Phase 1: 只保留 solver 全通的棋盘
        ], "S1: Generate data")

    # 2) Train
    if not args.eval_only:
        run([
            sys.executable, "scripts/train.py",
            "--data_dir", DEFAULTS["data_dir"],
            "--epochs", str(DEFAULTS["epochs"]),
            "--save_dir", DEFAULTS["save_dir"],
            "--device", args.device,
        ], f"S1: Train ({DEFAULTS['epochs']} epochs)")

    # 3) Evaluate
    ckpt = Path(DEFAULTS["save_dir"]) / "best_model.pt"
    if ckpt.exists():
        run([
            sys.executable, "scripts/evaluate.py",
            str(ckpt),
            "--width", str(DEFAULTS["width"]),
            "--height", str(DEFAULTS["height"]),
            "--mines", str(DEFAULTS["mines"]),
            "--n_games", str(args.n_games),
            "--device", args.device,
        ], "S1: Evaluate")
    else:
        print(f"⚠ No checkpoint at {ckpt}")


if __name__ == "__main__":
    main()
