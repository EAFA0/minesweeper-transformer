#!/usr/bin/env python3
"""S2: 监督学习 — 8×8 / 20 雷

继承 S1 权重，在更高密度下微调。
ms-toollib 提供无猜棋盘，每个训练样本都有确定的正确答案。

用法:
    python scripts/train_s2.py                    # 完整流程
    python scripts/train_s2.py --skip_data          # 跳过数据生成
    python scripts/train_s2.py --eval_only           # 仅评估
"""

import argparse
import subprocess
import sys
from pathlib import Path

STAGE = "S2"
DEFAULTS = {
    "width": 8, "height": 8, "mines": 20,
    "n_samples": 40000, "epochs": 30,
    "data_dir": "data/S2", "save_dir": "checkpoints/S2",
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
    p = argparse.ArgumentParser(description="S2: Supervised training 8×8 / 20 mines")
    p.add_argument("--skip_data", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--n_games", type=int, default=500, help="Eval games")
    p.add_argument("--pretrained", default=DEFAULTS["pretrained"])
    args = p.parse_args()

    if args.device == "auto":
        import torch
        args.device = "mps" if torch.backends.mps.is_available() else \
                      "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    pretrained = args.pretrained
    if not Path(pretrained).exists():
        print(f"⚠ Pretrained model not found: {pretrained}")
        print("  Run S1 first: python scripts/train_s1.py")
        pretrained = ""

    # 1) Generate
    if not args.skip_data and not args.eval_only:
        run([
            sys.executable, "scripts/generate_data.py",
            "--width", str(DEFAULTS["width"]),
            "--height", str(DEFAULTS["height"]),
            "--mines", str(DEFAULTS["mines"]),
            "--n_samples", str(DEFAULTS["n_samples"]),
            "--output", DEFAULTS["data_dir"],
        ], "S2: Generate data")

    # 2) Train
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
        run(cmd, f"S2: Train ({DEFAULTS['epochs']} epochs)")

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
        ], "S2: Evaluate")
    else:
        print(f"⚠ No checkpoint at {ckpt}")


if __name__ == "__main__":
    main()
