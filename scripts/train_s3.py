#!/usr/bin/env python3
"""S3: 监督学习 (概率蒸馏) — 12×12 / 40 雷 (继承 S2 权重)

更大棋盘，训练模型应对尺度变化。
用法:
    python scripts/train_s3.py                    # 完整流程
    python scripts/train_s3.py --force_data        # 强制重新生成数据
    python scripts/train_s3.py --eval_only         # 仅评估
"""

import argparse
import subprocess
import sys
from pathlib import Path

STAGE = "S3"
DEFAULTS = {
    "width": 12, "height": 12, "mines": 40,
    "n_samples": 10000, "epochs": 30,
    "data_dir": "data/S3", "save_dir": "checkpoints/S3",
    "pretrained": "checkpoints/S2/best_model.pt",
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
    p = argparse.ArgumentParser(description="S3: Supervised training (prob distillation, 12×12)")
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

    pretrained_path = Path(DEFAULTS["pretrained"])
    if not pretrained_path.exists() and not args.eval_only:
        print(f"❌ Pretrained checkpoint not found: {pretrained_path}")
        print(f"   Run train_s2.py first")
        sys.exit(1)

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
        run(cmd, "S3: Generate data")

    if not args.eval_only:
        cmd = [
            sys.executable, "scripts/train.py",
            "--data_dir", DEFAULTS["data_dir"],
            "--epochs", str(DEFAULTS["epochs"]),
            "--save_dir", DEFAULTS["save_dir"],
            "--device", args.device,
            "--pretrained", DEFAULTS["pretrained"],
            "--lr", "3e-4",
            "--weight_decay", "1e-4",
        ]
        if args.resume:
            resume_ckpt = Path(DEFAULTS["save_dir"]) / "final_model.pt"
            if resume_ckpt.exists():
                cmd.extend(["--resume", str(resume_ckpt)])
            else:
                print(f"⚠ No checkpoint to resume from: {resume_ckpt}")
        run(cmd, f"S3: Train ({DEFAULTS['epochs']} epochs)")

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
        ], "S3: Evaluate")
    else:
        print(f"⚠ No checkpoint at {ckpt}")


if __name__ == "__main__":
    main()
