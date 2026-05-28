# Minesweeper Transformer — Phase 1 Training Script
# Usage: python scripts/train.py [--epochs 50] [--batch_size 64]

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.train import train, TrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train Minesweeper Transformer (Phase 1: supervised learning)"
    )
    parser.add_argument("--data_dir", default="data/training",
                        help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--save_dir", default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--device", default="auto",
                        help="Device: cpu, cuda, mps, or auto")
    parser.add_argument("--lr_scheduler", default="cosine",
                        choices=["cosine", "plateau", "none"],
                        help="LR scheduler type")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable D4 data augmentation")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--pretrained", default="",
                        help="Path to pretrained checkpoint for curriculum transfer")

    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        device=args.device,
        lr_scheduler=args.lr_scheduler,
        augment=not args.no_augment,
        grad_clip_norm=args.grad_clip,
        pretrained=args.pretrained,
    )

    train(config)


if __name__ == "__main__":
    main()
