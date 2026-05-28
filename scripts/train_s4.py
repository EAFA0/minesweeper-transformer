#!/usr/bin/env python3
"""S4: 监督学习 — 16×16 / 80 雷

终极规格。继承 S3 权重。PE 原生 16×16（无需插值）。
80 雷是 ms-toollib 无猜生成的上限。

用法: python scripts/train_s4.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

STAGE = "S4"
DEFAULTS = {
    "width": 16, "height": 16, "mines": 80,
    "n_samples": 30000, "epochs": 10,
    "data_dir": "data/S4", "save_dir": "checkpoints/S4",
    "pretrained": "checkpoints/S3/best_model.pt",
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
    p = argparse.ArgumentParser(description="S4: Supervised training 16×16 / 80 mines")
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
        ], "S4: Generate data")

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
        run(cmd, f"S4: Train")

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
        ], "S4: Evaluate")


if __name__ == "__main__":
    main()
