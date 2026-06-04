"""Online Training Entry Point.

Self-validated boards from disk-backed pool, BCE loss on frontier cells
(or MSE on all covered cells), full BPTT refinement.

Usage:
  python scripts/train.py --board_width 8 --board_height 8 --board_mines 10 --n_games 5000
  python scripts/train.py --pretrained checkpoints/S1/best_model.pt --n_games 500
  python scripts/train.py --loss_type mse --n_games 5000
"""

import argparse
import torch

from config import TrainingConfig
from training.train import train


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser(description="Minesweeper Transformer — Online Training")
    default_cfg = TrainingConfig()

    # Board
    p.add_argument("--board_width", type=int, default=default_cfg.board_width)
    p.add_argument("--board_height", type=int, default=default_cfg.board_height)
    p.add_argument("--board_mines", type=int, default=default_cfg.board_mines)
    p.add_argument("--max_game_steps", type=int, default=default_cfg.max_game_steps)

    # Training
    p.add_argument("--n_games", type=int, default=default_cfg.n_games)
    p.add_argument("--eval_interval_games", type=int, default=default_cfg.eval_interval_games)
    p.add_argument("--eval_games", type=int, default=default_cfg.eval_games)
    p.add_argument("--pool_size", type=int, default=default_cfg.board_pool_size)
    p.add_argument("--pool_workers", type=int, default=default_cfg.pool_workers,
                   help="Multiprocessing workers (0=serial)")
    p.add_argument("--loss_type", type=str, default="bce", choices=["bce", "mse"],
                   help="Loss type: bce (frontier) or mse (all covered cells)")

    # Optimizer
    p.add_argument("--lr", type=float, default=default_cfg.learning_rate, dest="learning_rate")
    p.add_argument("--weight_decay", type=float, default=default_cfg.weight_decay)
    p.add_argument("--grad_clip_norm", type=float, default=default_cfg.grad_clip_norm)

    # Checkpoint
    p.add_argument("--pretrained", default=default_cfg.pretrained)
    p.add_argument("--resume_from", default=default_cfg.resume_from)
    p.add_argument("--save_dir", default=default_cfg.save_dir)
    p.add_argument("--device", default="auto")
    p.add_argument("--board_pool_path", default=default_cfg.board_pool_path)

    args = p.parse_args()

    device = args.device if args.device != "auto" else auto_device()
    print(f"Device: {device}")

    lr = args.learning_rate

    # Auto-reduce lr when fine-tuning (pretrained model is already good)
    if args.pretrained and args.learning_rate == 3e-4:
        lr = 1e-4
        print(f"Fine-tuning mode: auto-lowering lr {3e-4:.0e} → {lr:.0e}")
    elif args.pretrained and lr >= 3e-4:
        print(f"⚠ Fine-tuning with lr={lr:.0e} — consider using --lr 1e-5 for stability")

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
        loss_type=args.loss_type,
        learning_rate=lr,
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
