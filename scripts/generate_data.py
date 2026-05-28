# Minesweeper Transformer — Training Data Generation Script
# Usage: python scripts/generate_data.py [--n_samples 1000] [--output data/training]

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.generator import generate_training_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate supervised training data for Minesweeper Transformer"
    )
    parser.add_argument(
        "--n_samples", type=int, default=10000,
        help="Number of solvable game trajectories to generate (default: 10000)"
    )
    parser.add_argument(
        "--output", type=Path, default="data/training",
        help="Output directory (default: data/training)"
    )
    parser.add_argument(
        "--width", type=int, default=8,
        help="Board width (default: 8)"
    )
    parser.add_argument(
        "--height", type=int, default=8,
        help="Board height (default: 8)"
    )
    parser.add_argument(
        "--mines", type=int, default=10,
        help="Number of mines (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no_progress", action="store_true",
        help="Disable progress bar"
    )

    args = parser.parse_args()

    stats = generate_training_data(
        output_dir=args.output,
        n_samples=args.n_samples,
        width=args.width,
        height=args.height,
        total_mines=args.mines,
        seed=args.seed,
        show_progress=not args.no_progress,
    )

    print(f"\n📊 Generation complete!")
    print(f"   Generated: {stats['generated']} games from {stats['attempts']} attempts")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Avg steps per game: {stats['avg_steps_per_game']:.1f}")
    print(f"   Total training steps: {stats['total_steps']}")
    print(f"   Output files: {stats['output_files']}")
    print(f"   Time: {stats['elapsed_seconds']:.1f}s")
    print(f"   Output dir: {args.output.resolve()}")


if __name__ == "__main__":
    main()
