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
from training.train_supervised import train_supervised


def main():
    p = argparse.ArgumentParser(description="Minesweeper Transformer — Training")
    default_cfg = TrainingConfig()

    p.add_argument("--mode", type=str, default="online", choices=["online", "supervised"],
                   help="Training mode: online (self-play) or supervised (offline npz)")
    p.add_argument("--data_dir", type=str, default="data",
                   help="Directory for offline npz data (used in supervised mode)")
    p.add_argument("--epochs", type=int, default=default_cfg.epochs,
                   help="Number of epochs for supervised mode")

    # Board
    p.add_argument("--board_width", type=int, default=default_cfg.board_width)
    p.add_argument("--board_height", type=int, default=default_cfg.board_height)
    p.add_argument("--board_mines", type=int, default=default_cfg.board_mines)
    p.add_argument("--max_game_steps", type=int, default=default_cfg.max_game_steps)

    # Training
    p.add_argument("--n_games", type=int, default=default_cfg.n_games)
    p.add_argument("--eval_interval_games", type=int, default=default_cfg.eval_interval_games)
    p.add_argument("--eval_games", type=int, default=default_cfg.eval_games)
    p.add_argument("--pool_size", type=int, default=default_cfg.pool_size)
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
    p.add_argument("--save_dir", default=default_cfg.save_dir, help="Directory to save checkpoints. Defaults to checkpoints/{stage} or checkpoints/run_{timestamp}")
    p.add_argument("--device", default="auto")
    p.add_argument("--board_pool_path", default=default_cfg.board_pool_path)
    p.add_argument("--stage", type=str, default=None, choices=["S1", "S2", "S3"],
                   help="Training stage (e.g. S1, S2, S3). Will use checkpoints/{stage} as save_dir if specified.")
    p.add_argument("--arch", type=str, default="V4", choices=["V1", "V1_5", "V4"],
                   help="Architecture version to use (V1 = Single Pass, V1_5 = Early Refine (w/ Conf Head), V4 = CNN Once + Transformer Loop)")

    args = p.parse_args()

    # Apply stage config if specified (overrides defaults but respects CLI overrides)
    from config.stage_config import apply_stage_config
    apply_stage_config(args, default_cfg)

    # Determine save_dir / run_dir
    if args.save_dir == default_cfg.save_dir and not args.stage:
        from datetime import datetime
        save_dir = datetime.now().strftime("checkpoints/run_%Y%m%d_%H%M%S")
    else:
        save_dir = args.save_dir

    from utils.device import get_device
    device = get_device(args.device)
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
        pool_size=args.pool_size,
        pool_workers=args.pool_workers,
        n_games=args.n_games,
        eval_interval_games=args.eval_interval_games,
        eval_games=args.eval_games,
        board_pool_path=args.board_pool_path,
        loss_type=args.loss_type,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        save_dir=save_dir,
        device=device,
        pretrained=args.pretrained,
        resume_from=args.resume_from,
        data_dir=args.data_dir,
        epochs=args.epochs,
    )

    if args.mode == "supervised":
        from model.architecture import ModelConfig
        train_supervised(config, ModelConfig(), arch=args.arch, run_dir=save_dir)
    else:
        train(config, arch=args.arch)


if __name__ == "__main__":
    main()
