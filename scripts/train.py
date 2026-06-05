import argparse
import torch

from config import TrainingConfig
from training.train import train
from training.train_supervised import train_supervised
from utils.device import get_device


def main():
    p = argparse.ArgumentParser(description="Minesweeper Transformer — Training")
    
    # High-level configuration
    p.add_argument("--mode", type=str, default="online", choices=["online", "supervised"],
                   help="Training mode: online (self-play) or supervised (offline npz)")
    p.add_argument("--stage", type=str, default=None, choices=["S1", "S2", "S3"],
                   help="Training stage (e.g. S1, S2, S3). Applies stage-specific board/optimizer configs.")
    p.add_argument("--arch", type=str, default="V4", choices=["V1", "V1_5", "V4"],
                   help="Architecture version to use")
    p.add_argument("--loss_type", type=str, default="bce", choices=["bce", "mse"],
                   help="Loss function: bce (binary cross-entropy on ground-truth mines) or mse (probability distillation)")
    p.add_argument("--device", default="auto")

    # Explicit overrides (optional)
    p.add_argument("--n_games", type=int, default=None, help="Override number of games")
    p.add_argument("--lr", type=float, default=None, dest="learning_rate", help="Override learning rate")
    p.add_argument("--pretrained", default=None, help="Override pretrained checkpoint path")
    p.add_argument("--resume_from", default=None, help="Resume training from checkpoint")
    p.add_argument("--save_dir", default=None, help="Override save directory")
    p.add_argument("--data_dir", type=str, default=None, help="Directory for offline npz data (supervised mode)")
    p.add_argument("--refinement_steps", type=int, default=None, help="Refinement steps during training/inference (default: 4)")
    p.add_argument("--board_width", type=int, default=None, help="Override board width")
    p.add_argument("--board_height", type=int, default=None, help="Override board height")
    p.add_argument("--board_mines", type=int, default=None, help="Override mine count")

    args = p.parse_args()

    config = TrainingConfig()

    # 1. Apply stage defaults if specified
    if args.stage:
        from config.stage_config import apply_stage_config
        apply_stage_config(config, args.stage)

    # 2. Apply explicit CLI overrides
    if args.n_games is not None:
        config.n_games = args.n_games
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.pretrained is not None:
        config.pretrained = args.pretrained
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.loss_type is not None:
        config.loss_type = args.loss_type
    if args.refinement_steps is not None:
        config.refinement_steps = args.refinement_steps
    if args.board_width is not None:
        config.board_width = args.board_width
    if args.board_height is not None:
        config.board_height = args.board_height
    if args.board_mines is not None:
        config.board_mines = args.board_mines

    # 3. Dynamic save_dir fallback
    if config.save_dir == "checkpoints" and not args.stage:
        from datetime import datetime
        config.save_dir = datetime.now().strftime("checkpoints/run_%Y%m%d_%H%M%S")

    device = get_device(args.device)
    print(f"Device: {device}")

    # Auto-reduce lr when fine-tuning
    if config.pretrained and config.learning_rate == 3e-4:
        config.learning_rate = 1e-4
        print(f"Fine-tuning mode: auto-lowering lr 3e-4 → 1e-4")
    elif config.pretrained and config.learning_rate >= 3e-4:
        print(f"⚠ Fine-tuning with lr={config.learning_rate:.0e} — consider using lower lr for stability")

    config.device = str(device)

    if args.mode == "supervised":
        from model.architecture import ModelConfig
        train_supervised(config, ModelConfig(), arch=args.arch, run_dir=config.save_dir)
    else:
        train(config, arch=args.arch)


if __name__ == "__main__":
    main()
