#!/usr/bin/env python3
"""Supervised distillation training — model learns solver-computed probabilities.

Usage:
    # 1. Generate data first:
    python scripts/generate_distill_data.py --width 6 --height 6 --mines 18 --n_games 50000

    # 2. Train:
    python scripts/train_distill.py --data_dir data/distill/6x6_18 --epochs 20 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.training.distill import DistillConfig, train_distill


def main():
    p = argparse.ArgumentParser(
        description="Supervised distillation training (solver soft labels)")
    p.add_argument("--data_dir", default="data/distill/6x6_18",
                   help="Directory with .npz training data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--refinement_steps", type=int, default=4)
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_transformer_layers", type=int, default=3)
    p.add_argument("--n_attention_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--eval_games", type=int, default=200)
    p.add_argument("--save_dir", default="checkpoints/distill_6x6_18")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    # These must match the data
    p.add_argument("--board_width", type=int, default=6)
    p.add_argument("--board_height", type=int, default=6)
    p.add_argument("--board_mines", type=int, default=18)

    args = p.parse_args()

    config = DistillConfig(
        data_dir=args.data_dir,
        board_width=args.board_width,
        board_height=args.board_height,
        board_mines=args.board_mines,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        refinement_steps=args.refinement_steps,
        hidden_channels=args.hidden_channels,
        d_model=args.d_model,
        num_transformer_layers=args.n_transformer_layers,
        num_attention_heads=args.n_attention_heads,
        d_ff=args.d_ff,
        eval_interval_epochs=args.eval_interval,
        eval_games=args.eval_games,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
    )

    train_distill(config)


if __name__ == "__main__":
    main()
