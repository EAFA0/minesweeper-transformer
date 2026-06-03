"""Online BCE Training Entry Point.

Self-validated boards from disk-backed pool, BCE loss on frontier cells,
full BPTT refinement.

Usage:
  python scripts/train.py --board_width 8 --board_height 8 --board_mines 10 --n_games 5000
  python scripts/train.py --pretrained checkpoints/S1/best_model.pt --n_games 500
  python scripts/train.py --resume checkpoints/S1/final_model.pt
"""

import argparse
import torch

from training.train import TrainingConfig, train


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser(description="Minesweeper Transformer — Online BCE Training")

    # Board
    p.add_argument("--board_width", type=int, default=8)
    p.add_argument("--board_height", type=int, default=8)
    p.add_argument("--board_mines", type=int, default=10)
    p.add_argument("--max_game_steps", type=int, default=200)

    # Training
    p.add_argument("--n_games", type=int, default=5000)
    p.add_argument("--eval_interval_games", type=int, default=50)
    p.add_argument("--eval_games", type=int, default=100)
    p.add_argument("--pool_size", type=int, default=64)
    p.add_argument("--pool_workers", type=int, default=0,
                   help="Multiprocessing workers (0=serial)")

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4, dest="learning_rate")
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)

    # Checkpoint
    p.add_argument("--pretrained", default="")
    p.add_argument("--resume_from", default="")
    p.add_argument("--save_dir", default="checkpoints/S1")
    p.add_argument("--device", default="auto")
    p.add_argument("--board_pool_path", default="")

    args = p.parse_args()

    device = args.device if args.device != "auto" else auto_device()
    print(f"Device: {device}")

    config = TrainingConfig(
        board_width=args.board_width,
        board_height=args.board_height,
        board_mines=args.board_mines,
        max_game_steps=args.max_game_steps,
        board_pool_size=args.pool_size,
        pool_workers=args.pool_workers,
        n_games=args.n_games,
        eval_interval_games=args.eval_interval_games,
        eval_games=args.eval_games,
        board_pool_path=args.board_pool_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        save_dir=args.save_dir,
        device=device,
        pretrained=args.pretrained,
        resume_from=args.resume_from,
    )

    train(config)


if __name__ == "__main__":
    main()
