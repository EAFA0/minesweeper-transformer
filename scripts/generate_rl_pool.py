#!/usr/bin/env python3
"""Generate a pool of self-validated mixed boards for RL training.

Usage:
    python scripts/generate_rl_pool.py --target_size 5000 --workers 8
"""

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.self_validated import generate_self_validated_board
from minesweeper.constants import GameStatus


def generate_one_board(args_tuple):
    """Worker function to generate a single valid board."""
    seed, min_size, max_size, min_density, max_density = args_tuple
    rng = np.random.default_rng(seed)

    while True:
        w = rng.integers(min_size, max_size + 1)
        h = rng.integers(min_size, max_size + 1)
        density = rng.uniform(min_density, max_density)
        mines = max(1, int(w * h * density))

        game = generate_self_validated_board(
            width=w, height=h, total_mines=mines,
            rng=rng,
        )
        if game is not None and game.status == GameStatus.PLAYING:
            return (game.get_mine_mask(), game.visible.copy(), w, h)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate RL boards pool with multiprocessing."
    )
    parser.add_argument("--output", default="rl_boards.npz", help="Output .npz file")
    parser.add_argument("--target_size", type=int, default=5000, help="Number of boards")
    parser.add_argument("--min_size", type=int, default=6)
    parser.add_argument("--max_size", type=int, default=10)
    parser.add_argument("--min_density", type=float, default=0.1)
    parser.add_argument("--max_density", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of CPU workers (default: all cores)")
    args = parser.parse_args()

    workers = args.workers or cpu_count()
    print(f"Generating {args.target_size} RL boards using {workers} workers...")

    # Generate unique seeds for each worker task
    master_rng = np.random.default_rng(args.seed)
    seeds = master_rng.integers(0, 2**31 - 1, size=args.target_size)

    tasks = [
        (s, args.min_size, args.max_size, args.min_density, args.max_density)
        for s in seeds
    ]

    results = []
    with Pool(workers) as pool:
        for res in pool.imap_unordered(generate_one_board, tasks):
            results.append(res)
            if len(results) % 100 == 0:
                print(f"  {len(results)}/{args.target_size} boards generated...")

    print(f"Saving to {args.output}...")
    save_dict = {}
    for i, (mask, vis, w, h) in enumerate(results):
        save_dict[f"mask_{i}"] = mask
        save_dict[f"vis_{i}"] = vis
        save_dict[f"w_{i}"] = np.array(w)
        save_dict[f"h_{i}"] = np.array(h)

    np.savez_compressed(args.output, **save_dict)
    print(f"Done! RL pool saved with {len(results)} boards.")


if __name__ == "__main__":
    main()
