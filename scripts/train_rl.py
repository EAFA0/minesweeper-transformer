#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""RL fine-tuning entry point — thin wrapper around training.rl_train.

用法:
    python scripts/train_rl.py \
        --pretrained checkpoints/S3/best_model.pt \
        --width 10 --height 10 --mines 40 \
        --total_games 5000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.rl_board_pool import default_pool_path
from training.rl_train import RLConfig, train_rl


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="REINFORCE policy gradient fine-tuning for Minesweeper"
    )
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--pretrained", default="",
                        help="Path to supervised checkpoint for warm-start")
    parser.add_argument("--total_games", type=int, default=5000,
                        help="Total games for RL training")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="checkpoints/rl")
    parser.add_argument("--refine", type=int, default=4, dest="refine_steps",
                        help="Iterative refinement steps (default: 4)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Action selection temperature (default: 1.0)")
    parser.add_argument("--board_pool", default="",
                        help="Path to pre-built board pool (.npz). Default is based on board size.")
    parser.add_argument("--mixed", action="store_true",
                        help="Use mixed-size/mixed-density board pool")
    parser.add_argument("--no_board_pool", action="store_true",
                        help="Disable board pool and generate boards online (slow; debug only)")
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = args.device

    board_pool = ""
    if not args.no_board_pool:
        board_pool = args.board_pool or default_pool_path(
            args.width, args.height, args.mines, args.mixed
        )

    config = RLConfig(
        width=args.width, height=args.height,
        total_mines=args.mines,
        mine_continue=True,
        board_pool_path=board_pool,
        pretrained_path=args.pretrained,
        total_games=args.total_games,
        lr=args.lr,
        save_dir=args.save_dir,
        refine_steps=args.refine_steps,
        temperature=args.temperature,
        mixed_env=args.mixed,
        device=dev,
    )
    train_rl(config)


if __name__ == "__main__":
    main()
