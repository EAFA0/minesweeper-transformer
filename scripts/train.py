# Minesweeper Transformer — Supervised Training Script (Probability Distillation)
# Usage:
#   python scripts/train.py --epochs 50 --device auto
#   python scripts/train.py --resume checkpoints/final_model.pt --epochs 70 --lr 3e-4

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.train import train, TrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train Minesweeper Transformer (probability distillation, MSE loss)"
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
                        help="Path to pretrained checkpoint (weights only, for curriculum transfer)")
    parser.add_argument("--resume", default="", dest="resume_from",
                        help="Path to checkpoint to resume training (loads weights + optimizer + metrics)")
    parser.add_argument("--refine", type=int, default=1, dest="refinement_steps",
                        help="Iterative refinement steps during training (default: 1 = single-pass)")

    args = parser.parse_args()

    if args.resume_from and args.pretrained:
        print("Error: --resume and --pretrained are mutually exclusive")
        sys.exit(1)

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
        resume_from=args.resume_from,
        refinement_steps=args.refinement_steps,
    )

    train(config)


if __name__ == "__main__":
    main()
