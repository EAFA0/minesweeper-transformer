# Minesweeper Transformer — Training Data Generation (Probability Distillation)
# Usage: python scripts/generate_data.py [--n_samples 1000] [--output data/training]
#
# Generates probability-distilled training data from no-guess boards.
# Each step records (board_state → solver-computed P(mine) matrix).
# Model learns to estimate probabilities via MSE loss.
#
# Skips generation if data already exists (use --force to override).

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.generator import generate_training_data


def data_exists(output_dir: Path, expected_n_samples: int) -> bool:
    """Check if valid probability-distilled data already exists."""
    stats_file = output_dir / "stats.json"
    if not stats_file.exists():
        return False

    data_files = sorted(output_dir.glob("data_*.npz"))
    if not data_files:
        return False

    # Verify format: check first file has 'probs' key (not old 'labels')
    try:
        d = np.load(data_files[0])
        if "probs" not in d:
            return False
    except Exception:
        return False

    # Check sample count
    try:
        with open(stats_file) as f:
            stats = json.load(f)
        n_generated = stats.get("generated", 0)
        if n_generated < expected_n_samples:
            print(f"⚠ Existing data has {n_generated} games, need {expected_n_samples}. Regenerating...")
            return False
    except Exception:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate probability-distilled training data for Minesweeper Transformer"
    )
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of game trajectories to generate (default: 10000)")
    parser.add_argument("--output", type=Path, default="data/training",
                        help="Output directory (default: data/training)")
    parser.add_argument("--width", type=int, default=8, help="Board width (default: 8)")
    parser.add_argument("--height", type=int, default=8, help="Board height (default: 8)")
    parser.add_argument("--mines", type=int, default=10, help="Number of mines (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bar")
    parser.add_argument("--samples_per_file", type=int, default=100,
                        help="Number of games per .npz file (default: 100)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if data already exists")

    args = parser.parse_args()

    output_dir = Path(args.output)

    if not args.force and data_exists(output_dir, args.n_samples):
        print(f"📦 Data already exists: {output_dir}")
        data_files = sorted(output_dir.glob("data_*.npz"))
        total = sum(1 for _ in data_files)
        print(f"   {total} files, target={args.n_samples} games")
        print(f"   Use --force to regenerate")
        return

    stats = generate_training_data(
        output_dir=output_dir,
        n_samples=args.n_samples,
        width=args.width,
        height=args.height,
        total_mines=args.mines,
        seed=args.seed,
        samples_per_file=args.samples_per_file,
        show_progress=not args.no_progress,
    )

    print(f"\n📊 Generation complete!")
    print(f"   Generated: {stats['generated']} games from {stats['attempts']} attempts")
    print(f"   Avg steps per game: {stats['avg_steps_per_game']:.1f}")
    print(f"   Total training steps: {stats['total_steps']}")
    print(f"   Avg ambiguous cells/step: {stats['avg_ambig_per_game']:.1f}")
    print(f"   Output files: {stats['output_files']}")
    print(f"   Time: {stats['elapsed_seconds']:.1f}s")
    print(f"   Output dir: {output_dir.resolve()}")
    print(f"\n   Label type: probability distillation (MSE)")
    print(f"   Data format: (channels, probs, masks) per sample")


if __name__ == "__main__":
    main()
