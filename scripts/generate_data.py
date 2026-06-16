#!/usr/bin/env python3
"""Generate training data for a specific board configuration."""

import argparse

from data.pipeline import generate_from_args


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for minesweeper AI"
    )
    parser.add_argument(
        "--width", type=int, default=8, help="Board width (default: 8)"
    )
    parser.add_argument(
        "--height", type=int, default=8, help="Board height (default: 8)"
    )
    parser.add_argument(
        "--mines", type=int, default=10, help="Number of mines (default: 10)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of games to generate (default: 10000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--samples_per_file",
        type=int,
        default=2000,
        help="Games per .npz file (default: 2000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration, ignore existing data",
    )
    args = parser.parse_args()
    generate_from_args(args)


if __name__ == "__main__":
    main()
