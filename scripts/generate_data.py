# pyright: reportMissingImports=false
# Minesweeper Transformer — Training Data Generation (Probability Distillation)
# Usage: python scripts/generate_data.py --stage S1

import argparse
from pathlib import Path

from config import STAGE_DATASETS
from data.pipeline import (
    DEFAULT_N_SAMPLES,
    DEFAULT_SAMPLES_PER_FILE,
    generate_from_args,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate probability-distilled training data for Minesweeper Transformer"
    )
    parser.add_argument("--stage", choices=list(STAGE_DATASETS.keys()),
                        help="Generate one canonical stage dataset into data/{stage}")
    parser.add_argument("--all_stages", action="store_true",
                        help="Generate all canonical stage datasets S1-S5")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES,
                        help="Number of game trajectories to generate (default: 10000)")
    parser.add_argument("--output", type=Path, default="data",
                        help="Output root for stage mode, or direct output dir without --stage")
    parser.add_argument("--width", type=int, default=8, help="Board width (default: 8)")
    parser.add_argument("--height", type=int, default=8, help="Board height (default: 8)")
    parser.add_argument("--mines", type=int, default=10, help="Number of mines (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bar")
    parser.add_argument("--samples_per_file", type=int, default=DEFAULT_SAMPLES_PER_FILE,
                        help="Number of games per .npz file (default: 2000)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if data already exists")
    parser.add_argument("--mixed", action="store_true",
                        help="Experimental: generate mixed variable-size/density dataset")
    parser.add_argument("--min_size", type=int, default=4,
                        help="Min board size for mixed mode (default: 4)")
    parser.add_argument("--max_size", type=int, default=8,
                        help="Max board size for mixed mode (default: 8)")
    parser.add_argument("--min_density", type=float, default=0.1,
                        help="Min mine density for mixed mode (default: 0.1)")
    parser.add_argument("--max_density", type=float, default=0.5,
                        help="Max mine density for mixed mode (default: 0.5)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Worker processes for fixed-size mode (0=auto, 1=single). Default: 0")

    args = parser.parse_args()

    try:
        generate_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
