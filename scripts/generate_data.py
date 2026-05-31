# pyright: reportMissingImports=false
# Minesweeper Transformer — Training Data Generation (Probability Distillation)
# Usage: python scripts/generate_data.py [--n_samples 1000] [--output data/training]
#
# Generates probability-distilled training data from no-guess boards.
# Each step records (board_state → solver-computed P(mine) matrix).
# Model learns to estimate probabilities via MSE loss.
#
# Supports:
# - Fixed-size generation (--width --height --mines), with optional multiprocessing
# - Mixed data generation (--mixed, --min_size, --max_size, --min_density, --max_density)
#
# Skips generation if data already exists (use --force to override).

import argparse
import json
import multiprocessing
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.generator import (
    generate_training_data,
    record_game_trajectory,
    save_trajectory_buffer,
)
from data.mixed_generator import generate_mixed_data


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


def _parallel_worker(seed, width, height, total_mines):
    """Wrapper for multiprocessing: seed → record_game_trajectory."""
    rng = np.random.default_rng(seed)
    return record_game_trajectory(width=width, height=height,
                                  total_mines=total_mines, rng=rng)


def generate_training_data_parallel(
    output_dir: Path,
    n_samples: int = 10000,
    width: int = 8, height: int = 8, total_mines: int = 10,
    seed: int = 42, samples_per_file: int = 100, workers: int = 16,
) -> dict:
    """Multi-process training data generation for fixed-size boards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # Prepare seeds (over-generate by 50% to handle failed trajectories)
    rng = np.random.default_rng(seed)
    n_seeds = int(n_samples * 1.5)
    seeds = rng.integers(0, 2**31 - 1, size=n_seeds)

    worker_func = partial(_parallel_worker, width=width, height=height,
                          total_mines=total_mines)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_samples, desc=f"Generating ({workers} workers)")
    except ImportError:
        pbar = None

    buffer = []
    file_idx = 0
    total_attempts = 0
    total_generated = 0
    total_steps = 0
    total_ambiguous = 0
    chunksize = max(1, n_samples // (workers * 10))

    with multiprocessing.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(worker_func, seeds, chunksize=chunksize):
            total_attempts += 1
            if result is not None:
                total_generated += 1
                total_steps += result["n_steps"]
                for step in result["trajectory"]:
                    total_ambiguous += step["n_ambiguous"]
                buffer.append(result)
                if pbar:
                    pbar.update(1)
                if len(buffer) >= samples_per_file:
                    save_trajectory_buffer(
                        buffer, output_dir, file_idx, include_counts=False
                    )
                    buffer = []
                    file_idx += 1
                if total_generated >= n_samples:
                    pool.terminate()
                    break

    if buffer:
        save_trajectory_buffer(buffer, output_dir, file_idx, include_counts=False)
        file_idx += 1

    elapsed = time.time() - start
    if pbar:
        pbar.close()

    stats = {
        "params": {
            "width": width, "height": height, "total_mines": total_mines,
            "n_samples_target": n_samples, "workers": workers,
            "label_type": "probability_distillation",
        },
        "attempts": total_attempts,
        "generated": total_generated,
        "total_steps": total_steps,
        "total_ambiguous_cells": total_ambiguous,
        "elapsed_seconds": elapsed,
        "avg_steps_per_game": total_steps / max(1, total_generated),
        "avg_ambig_per_game": total_ambiguous / max(1, total_generated),
        "output_files": file_idx,
        "games_per_second": total_generated / elapsed if elapsed > 0 else 0,
    }

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    return stats


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
    parser.add_argument("--mixed", action="store_true",
                        help="Generate mixed dataset (variable sizes + densities)")
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

    output_dir = Path(args.output)

    if not args.force and data_exists(output_dir, args.n_samples):
        print(f"📦 Data already exists: {output_dir}")
        data_files = sorted(output_dir.glob("data_*.npz"))
        total = sum(1 for _ in data_files)
        print(f"   {total} files, target={args.n_samples} games")
        print(f"   Use --force to regenerate")
        return

    if args.mixed:
        stats = generate_mixed_data(
            output_dir=output_dir,
            n_samples=args.n_samples,
            min_size=args.min_size,
            max_size=args.max_size,
            min_density=args.min_density,
            max_density=args.max_density,
            seed=args.seed,
            samples_per_file=args.samples_per_file,
            show_progress=not args.no_progress,
        )
        print(f"\n📊 Mixed generation complete!")
        print(f"   Generated: {stats['generated']} games from {stats['attempts']} attempts")
        print(f"   Size range: {args.min_size}-{args.max_size}, density: {args.min_density}-{args.max_density}")
        print(f"   Avg steps per game: {stats['avg_steps_per_game']:.1f}")
        print(f"   Total training steps: {stats['total_steps']}")
        print(f"   Output files: {stats['output_files']}")
        print(f"   Time: {stats['elapsed_seconds']:.1f}s")
        return

    workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
    if workers <= 1:
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
    else:
        stats = generate_training_data_parallel(
            output_dir=output_dir,
            n_samples=args.n_samples,
            width=args.width,
            height=args.height,
            total_mines=args.mines,
            seed=args.seed,
            samples_per_file=args.samples_per_file,
            workers=workers,
        )

    print(f"\n📊 Generation complete!")
    print(f"   Generated: {stats['generated']} games from {stats['attempts']} attempts")
    print(f"   Avg steps per game: {stats['avg_steps_per_game']:.1f}")
    print(f"   Total training steps: {stats['total_steps']}")
    print(f"   Avg ambiguous cells/step: {stats.get('avg_ambig_per_game', 0):.1f}")
    print(f"   Output files: {stats['output_files']}")
    print(f"   Time: {stats['elapsed_seconds']:.1f}s")
    print(f"   Speed: {stats.get('games_per_second', 0):.1f} games/s")
    print(f"   Output dir: {output_dir.resolve()}")
    print(f"\n   Label type: probability distillation (MSE)")
    print(f"   Data format: (channels, probs, masks) per sample")


if __name__ == "__main__":
    main()
