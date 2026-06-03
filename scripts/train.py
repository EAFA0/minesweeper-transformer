"""Unified training entry point.

Supports two modes:
- supervised: MSE loss on pre-generated .npz probability distillation data
- online:     BCE loss on self-validated boards (frontier cells only)

Examples:
  python scripts/train.py --mode supervised --data_dir data/S1 --epochs 5
  python scripts/train.py --mode online --board_width 8 --board_height 8 --board_mines 10 --n_games 5000
"""

import argparse
import torch

from minesweeper_transformer.training.train import TrainingConfig, train, train_online


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser(
        description="Minesweeper Transformer — Unified Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    p.add_argument("--mode", choices=["supervised", "online"], default="supervised",
                   help="Training mode: supervised (MSE on .npz) or online (BCE on self-validated)")

    # Shared
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4, dest="learning_rate")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--refinement_steps", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--lr_scheduler", choices=["cosine", "plateau", "none"], default="cosine")
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--save_dir", default="checkpoints/S1")
    p.add_argument("--device", default="auto")
    p.add_argument("--pretrained", default="", help="Curriculum transfer: load weights only")
    p.add_argument("--resume_from", default="", help="Resume: weights + optimizer + metrics")
    p.add_argument("--augment", action="store_true", default=True, help="D4 augmentation (default on)")
    p.add_argument("--no_augment", action="store_false", dest="augment", help="Disable D4 augmentation")

    # Supervised-specific
    p.add_argument("--data_dir", default="data/S1", help="Path to .npz data directory")
    p.add_argument("--val_ratio", type=float, default=0.2)

    # Online-specific
    p.add_argument("--n_games", type=int, default=5000)
    p.add_argument("--board_width", type=int, default=8)
    p.add_argument("--board_height", type=int, default=8)
    p.add_argument("--board_mines", type=int, default=10)
    p.add_argument("--max_game_steps", type=int, default=200)
    p.add_argument("--eval_interval_games", type=int, default=50)
    p.add_argument("--eval_games", type=int, default=100)
    p.add_argument("--board_pool_path", default="")

    args = p.parse_args()

    # Device resolution
    device = args.device if args.device != "auto" else auto_device()
    print(f"Device: {device}")

    config = TrainingConfig(
        mode=args.mode,
        data_dir=args.data_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        refinement_steps=args.refinement_steps,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        grad_clip_norm=args.grad_clip_norm,
        save_dir=args.save_dir,
        device=device,
        pretrained=args.pretrained,
        resume_from=args.resume_from,
        augment=args.augment,
        val_ratio=args.val_ratio,
        # Online BCE
        n_games=args.n_games,
        board_width=args.board_width,
        board_height=args.board_height,
        board_mines=args.board_mines,
        max_game_steps=args.max_game_steps,
        eval_interval_games=args.eval_interval_games,
        eval_games=args.eval_games,
        board_pool_path=args.board_pool_path,
    )

    if args.mode == "online":
        train_online(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
