#!/usr/bin/env python3
"""S4: PPO 强化学习 — 16×16 / 99 雷

终极规格。继承 S3 的 PPO 权重，PE 原生支持 16×16（无需插值）。

用法:
    python scripts/train_s4.py
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.ppo_train import PPOConfig, train_ppo


def main():
    p = argparse.ArgumentParser(description="S4: PPO RL 16×16 / 99 mines")
    p.add_argument("--pretrained", default="checkpoints/S3/best_model.pt")
    p.add_argument("--total_games", type=int, default=30000)
    p.add_argument("--save_dir", default="checkpoints/S4")
    p.add_argument("--device", default="auto")

    args = p.parse_args()
    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else \
                      "cuda" if torch.cuda.is_available() else "cpu"

    config = PPOConfig(
        width=16, height=16, total_mines=99,
        pretrained=args.pretrained,
        total_games=args.total_games,
        save_dir=args.save_dir,
        device=args.device,
    )
    train_ppo(config)


if __name__ == "__main__":
    main()
