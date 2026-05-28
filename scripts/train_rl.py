#!/usr/bin/env python3
"""Phase 2 RL Training CLI.

Usage:
    python scripts/train_rl.py --total_games 5000 --device mps
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from training.rl_train import RLConfig, train_rl


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: REINFORCE policy gradient fine-tuning"
    )
    parser.add_argument("--pretrained", default="checkpoints/best_model.pt",
                        help="Path to Phase 1 pretrained model")
    parser.add_argument("--total_games", type=int, default=5000,
                        help="Total games to play (default: 5000)")
    parser.add_argument("--games_per_batch", type=int, default=16,
                        help="Games per REINFORCE gradient step")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (should be low for fine-tuning)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Exploration temperature (0.1=greedy, 1.0=random)")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discount factor")
    parser.add_argument("--save_dir", default="checkpoints/rl",
                        help="Output directory")
    parser.add_argument("--device", default="auto",
                        help="Device: cpu, cuda, mps, or auto")

    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    config = RLConfig(
        pretrained_path=args.pretrained,
        total_games=args.total_games,
        games_per_batch=args.games_per_batch,
        lr=args.lr,
        temperature=args.temperature,
        gamma=args.gamma,
        save_dir=args.save_dir,
        device=device,
    )

    train_rl(config)


if __name__ == "__main__":
    main()
