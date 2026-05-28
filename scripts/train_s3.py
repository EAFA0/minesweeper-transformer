#!/usr/bin/env python3
"""S3: PPO 强化学习 — 12×12 / 40 雷

继承 S2 的 PPO 权重，适应 12×12 大棋盘。PE 自动 bilinear 插值适配新尺寸。

用法:
    python scripts/train_s3.py
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.ppo_train import PPOConfig, train_ppo


def main():
    p = argparse.ArgumentParser(description="S3: PPO RL 12×12 / 40 mines")
    p.add_argument("--pretrained", default="checkpoints/S2/best_model.pt")
    p.add_argument("--total_games", type=int, default=20000)
    p.add_argument("--save_dir", default="checkpoints/S3")
    p.add_argument("--device", default="auto")

    args = p.parse_args()
    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else \
                      "cuda" if torch.cuda.is_available() else "cpu"

    config = PPOConfig(
        width=12, height=12, total_mines=40,
        pretrained=args.pretrained,
        total_games=args.total_games,
        save_dir=args.save_dir,
        device=args.device,
    )
    train_ppo(config)


if __name__ == "__main__":
    main()
