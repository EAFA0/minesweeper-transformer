"""Standalone evaluation script.

Evaluates a trained model by playing N games on self-validated boards.
Uses the shared training.evaluate module for consistency with training-time eval.

Examples:
  python scripts/evaluate.py checkpoints/S1/best_model.pt --n_games 1000
  python scripts/evaluate.py checkpoints/S3/best_model.pt --width 12 --height 12 --mines 40
"""

import argparse
import torch
from pathlib import Path

from config import TrainingConfig
from training.evaluate import evaluate_model, load_model


def main():
    p = argparse.ArgumentParser(description="Evaluate Minesweeper Transformer model")
    default_cfg = TrainingConfig()

    p.add_argument("checkpoint", help="Path to model checkpoint (.pt)")
    p.add_argument("--n_games", type=int, default=1000)
    p.add_argument("--width", type=int, default=default_cfg.board_width)
    p.add_argument("--height", type=int, default=default_cfg.board_height)
    p.add_argument("--mines", type=int, default=default_cfg.board_mines)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refine_steps", type=int, default=None,
                   help="Override eval refinement steps (default: from policy)")
    p.add_argument("--board_pool", default=None,
                   help="Board pool .npz path (auto: eval_boards_WxH_M.npz)")
    p.add_argument("--device", default="auto")

    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Board: {args.width}×{args.height}/{args.mines} mines")
    print(f"Eval games: {args.n_games}")

    model = load_model(args.checkpoint, device)
    print(f"Model loaded: {model.num_parameters:,} parameters")

    board_pool_path = Path(args.board_pool) if args.board_pool else None

    result = evaluate_model(
        model, device,
        n_games=args.n_games,
        width=args.width, height=args.height,
        total_mines=args.mines,
        seed=args.seed,
        board_pool_path=board_pool_path,
        refine_steps=args.refine_steps,
    )

    print(f"\n═══ Results ═══")
    print(f"Games: {result['n_games']} (gen_failed: {result['gen_failed']})")
    print(f"Win:  {result['won']:4d} ({result['win_rate']:.2%})")
    print(f"Loss: {result['lost']:4d}")
    print(f"Stuck:{result['stuck']:4d}")
    print(f"Action accuracy: {result['action_accuracy']:.4f}")
    print(f"Avg game steps: {result['avg_steps']:.1f}")
    print(f"Avg refine steps: {result['avg_refine_steps']:.1f} (early stop savings)")
    print(f"Time: {result['elapsed']:.0f}s")


if __name__ == "__main__":
    main()
