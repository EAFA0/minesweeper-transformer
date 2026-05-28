#!/usr/bin/env python3
"""S3: 监督学习 — 12×12 / 40 雷

继承 S2 权重，适应 12×12 大棋盘。PE 自动 bilinear 插值适配新尺寸。

用法: python scripts/train_s3.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

STAGE = "S3"
DEFAULTS = {
    "width": 12, "height": 12, "mines": 40,
    "n_samples": 20000, "epochs": 15,
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
    p = argparse.ArgumentParser(description="S3: Supervised training 12×12 / 40 mines")
    p.add_argument("--skip_data", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--n_games", type=int, default=500)

    args = p.parse_args()
    if args.device == "auto":
        import torch
        args.device = "mps" if torch.backends.mps.is_available() else \
                      "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    pretrained = DEFAULTS["pretrained"]
    if not Path(pretrained).exists():
        print(f"⚠ Pretrained model not found: {pretrained}")
        pretrained = ""

    if not args.skip_data and not args.eval_only:
        run([
            sys.executable, "scripts/generate_data.py",
            "--width", str(DEFAULTS["width"]),
            "--height", str(DEFAULTS["height"]),
            "--mines", str(DEFAULTS["mines"]),
            "--n_samples", str(DEFAULTS["n_samples"]),
            "--output", DEFAULTS["data_dir"],
        ], "S3: Generate data")

    if not args.eval_only:
        cmd = [
            sys.executable, "scripts/train.py",
            "--data_dir", DEFAULTS["data_dir"],
            "--epochs", str(DEFAULTS["epochs"]),
            "--save_dir", DEFAULTS["save_dir"],
            "--device", args.device,
        ]
        if pretrained:
            cmd += ["--pretrained", pretrained]
        run(cmd, f"S3: Train")

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


if __name__ == "__main__":
    main()
