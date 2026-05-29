#!/usr/bin/env python3
"""S1: 监督学习 (概率蒸馏) — 8×8 / 10 雷

学会基础模式识别：1-2-1、边角数字、flood fill 触发条件等。
使用 ProbabilitySolver 计算精确 P(mine) 软标签 + MSE loss。

用法:
    python scripts/train_s1.py                    # 完整流程（数据已有则跳过生成）
    python scripts/train_s1.py --force_data        # 强制重新生成数据
    python scripts/train_s1.py --resume            # 从 checkpoint 续训
    python scripts/train_s1.py --eval_only         # 仅评估已有 checkpoint
"""

import argparse
import subprocess
import sys
from pathlib import Path

STAGE = "S1"
DEFAULTS = {
    "width": 8, "height": 8, "mines": 10,
    "n_samples": 10000, "epochs": 50,
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
    p = argparse.ArgumentParser(description="S1: Supervised training (prob distillation)")
    p.add_argument("--force_data", action="store_true", help="Force regenerate training data")
    p.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    p.add_argument("--eval_only", action="store_true", help="Only evaluate existing checkpoint")
    p.add_argument("--device", default="auto")
    p.add_argument("--n_games", type=int, default=500, help="Eval games")
    args = p.parse_args()

    if args.device == "auto":
        import torch
        args.device = "mps" if torch.backends.mps.is_available() else \
                      "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    # 1) Generate data (auto-skip if exists)
    if not args.eval_only:
        cmd = [
            sys.executable, "scripts/generate_data.py",
            "--width", str(DEFAULTS["width"]),
            "--height", str(DEFAULTS["height"]),
            "--mines", str(DEFAULTS["mines"]),
            "--n_samples", str(DEFAULTS["n_samples"]),
            "--output", DEFAULTS["data_dir"],
        ]
        if args.force_data:
            cmd.append("--force")
        run(cmd, "S1: Generate data (prob distillation)")

    # 2) Train
    if not args.eval_only:
        cmd = [
            sys.executable, "scripts/train.py",
            "--data_dir", DEFAULTS["data_dir"],
            "--epochs", str(DEFAULTS["epochs"]),
            "--save_dir", DEFAULTS["save_dir"],
            "--device", args.device,
        ]
        if args.resume:
            resume_ckpt = Path(DEFAULTS["save_dir"]) / "final_model.pt"
            if resume_ckpt.exists():
                cmd.extend(["--resume", str(resume_ckpt)])
            else:
                print(f"⚠ No checkpoint to resume from: {resume_ckpt}")
                print(f"   Starting fresh training")
        run(cmd, f"S1: Train ({DEFAULTS['epochs']} epochs)")

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
