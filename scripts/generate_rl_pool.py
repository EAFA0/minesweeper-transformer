#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Generate a pool of self-validated boards for RL training.

Supports both fixed-size and mixed boards.

Usage:
    # Mixed: random sizes 6-10, density 10-40%
    python scripts/generate_rl_pool.py --target_size 5000 --workers 16

    # Fixed: 10×10/40 mines only
    python scripts/generate_rl_pool.py --target_size 5000 --width 10 --height 10 --mines 40
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
from training.rl_board_pool import default_pool_path


def generate_one_mixed(args_tuple):
    """Worker: generate one random-size/random-density board."""
    seed, min_size, max_size, min_density, max_density = args_tuple
    rng = np.random.default_rng(seed)
    while True:
        w = rng.integers(min_size, max_size + 1)
        h = rng.integers(min_size, max_size + 1)
        density = rng.uniform(min_density, max_density)
        mines = max(1, int(w * h * density))
        game = generate_self_validated_board(w, h, mines, rng=rng)
        if game is not None and game.status == GameStatus.PLAYING:
            return (game.get_mine_mask(), game.visible.copy(), w, h)


def generate_one_fixed(args_tuple):
    """Worker: generate one fixed-size board."""
    seed, width, height, mines = args_tuple
    rng = np.random.default_rng(seed)
    while True:
        game = generate_self_validated_board(width, height, mines, rng=rng)
        if game is not None and game.status == GameStatus.PLAYING:
            return (game.get_mine_mask(), game.visible.copy(), width, height)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate RL boards pool with multiprocessing."
    )
    parser.add_argument("--output", default="", help="Output .npz file")
    parser.add_argument("--target_size", type=int, default=5000, help="Number of boards")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of CPU workers (default: all cores)")

    # Fixed mode
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--mines", type=int, default=None)

    # Mixed mode (used when width/height/mines not specified)
    parser.add_argument("--min_size", type=int, default=6)
    parser.add_argument("--max_size", type=int, default=10)
    parser.add_argument("--min_density", type=float, default=0.1)
    parser.add_argument("--max_density", type=float, default=0.4)

    args = parser.parse_args()

    workers = args.workers or cpu_count()
    fixed_mode = args.width is not None and args.height is not None and args.mines is not None
    output = args.output or default_pool_path(
        args.width, args.height, args.mines, mixed=not fixed_mode
    )

    if fixed_mode:
        print(f"Generating {args.target_size} fixed boards "
              f"({args.width}×{args.height}/{args.mines}) with {workers} workers...")
        master_rng = np.random.default_rng(args.seed)
        seeds = master_rng.integers(0, 2**31 - 1, size=args.target_size)
        tasks = [(int(s), args.width, args.height, args.mines) for s in seeds]
        worker_func = generate_one_fixed
    else:
        print(f"Generating {args.target_size} mixed boards "
              f"({args.min_size}-{args.max_size}, {args.min_density}-{args.max_density}) "
              f"with {workers} workers...")
        master_rng = np.random.default_rng(args.seed)
        seeds = master_rng.integers(0, 2**31 - 1, size=args.target_size)
        tasks = [(int(s), args.min_size, args.max_size, args.min_density, args.max_density)
                 for s in seeds]
        worker_func = generate_one_mixed

    results = []
    with Pool(workers) as pool:
        for res in pool.imap_unordered(worker_func, tasks):
            results.append(res)
            if len(results) % 100 == 0:
                print(f"  {len(results)}/{args.target_size} boards generated...")

    print(f"Saving to {output}...")
    save_dict = {}
    for i, (mask, vis, w, h) in enumerate(results):
        save_dict[f"mask_{i}"] = mask
        save_dict[f"vis_{i}"] = vis
        save_dict[f"w_{i}"] = np.array(w)
        save_dict[f"h_{i}"] = np.array(h)

    np.savez_compressed(output, **save_dict)
    print(f"Done! RL pool saved with {len(results)} boards.")


if __name__ == "__main__":
    main()
