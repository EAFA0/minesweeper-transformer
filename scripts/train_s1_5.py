#!/usr/bin/env python3
"""S1.5: 监督学习 — 8×8 / 15 雷

S1(10雷) 和 S2(20雷) 之间的过渡阶梯。
15 雷密度 23%，模型能平滑过渡而不是从 16% 跳到 39%。

用法:
    python scripts/train_s1_5.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULTS = {
    "width": 8, "height": 8, "mines": 15,
    "n_samples": 10000, "epochs": 10,
    "data_dir": "data/S1_5", "save_dir": "checkpoints/S1_5",
    "pretrained": "checkpoints/S1/best_model.pt",
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
    p = argparse.ArgumentParser(description="S1.5: Supervised training 8×8 / 15 mines")
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
        ], "S1.5: Generate data")

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
        run(cmd, f"S1.5: Train ({DEFAULTS['epochs']} epochs)")

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
        ], "S1.5: Evaluate")


if __name__ == "__main__":
    main()
