#!/usr/bin/env python3
"""RL fine-tuning entry point — thin wrapper around training.rl_train.

用法:
    python scripts/train_rl.py --pretrained checkpoints/S2_5/best_model.pt

    python scripts/train_rl.py \
        --pretrained checkpoints/S2_5/best_model.pt \
        --width 10 --height 10 --mines 30 \
        --board_mode self_validated --mine_continue \
        --total_games 5000 --refine 3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.rl_train import RLConfig, train_rl


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="REINFORCE policy gradient fine-tuning for Minesweeper"
    )
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--mine_continue", action="store_true",
                        help="Continue game after mine hit — denser training signal")
    parser.add_argument("--warmup", type=int, default=0, dest="warmup_clicks",
                        help="Random safe reveals before model takes over (default: 0)")
    parser.add_argument("--pretrained", default="",
                        help="Path to supervised checkpoint for warm-start")
    parser.add_argument("--total_games", type=int, default=5000,
                        help="Total games for RL training")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="checkpoints/rl")
    parser.add_argument("--refine", type=int, default=8, dest="refine_steps",
                        help="Iterative refinement steps (default: 8)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Action selection temperature (default: 1.0)")
    parser.add_argument("--entropy_coef", type=float, default=0.05,
                        help="Entropy bonus coefficient — prevents policy collapse (default: 0.05)")
    parser.add_argument("--board_pool", default="rl_boards.npz",
                        help="Pre-generate boards to .npz (speeds up RL)")
    parser.add_argument("--no_board_pool", action="store_true",
                        help="Disable board pooling")
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

    config = RLConfig(
        width=args.width, height=args.height,
        total_mines=args.mines,
        mine_continue=args.mine_continue,
        warmup_clicks=args.warmup_clicks,
        board_pool_path="" if args.no_board_pool else args.board_pool,
        pretrained_path=args.pretrained,
        total_games=args.total_games,
        lr=args.lr,
        save_dir=args.save_dir,
        refine_steps=args.refine_steps,
        temperature=args.temperature,
        entropy_coef=args.entropy_coef,
        device=dev,
    )
    train_rl(config)


if __name__ == "__main__":
    main()
