# pyright: reportMissingImports=false
# Minesweeper Transformer — Training Data Generation (Probability Distillation)
# Usage: python scripts/generate_data.py [--n_samples 1000] [--output data/training]

import argparse
import json
import multiprocessing
import sys
import time
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np

from data.generator import (
    generate_training_data,
    generate_trajectory,
)
from data.writer import TrajectoryWriter
from data.mixed_generator import generate_mixed_data


def _parallel_worker(seed, width, height, total_mines):
    """Wrapper for multiprocessing: seed → generate_trajectory."""
    rng = np.random.default_rng(seed)
    return generate_trajectory(width=width, height=height,
                               total_mines=total_mines, rng=rng)


def generate_training_data_parallel(
    output_dir: Path,
    n_samples: int = 10000,
    width: int = 8, height: int = 8, total_mines: int = 10,
    seed: int = 42, samples_per_file: int = 100, workers: int = 16,
    start_file_idx: int = 0, existing_stats: Optional[dict] = None,
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

    total_attempts = 0
    total_generated = 0
    total_steps = 0
    total_ambiguous = 0
    chunksize = max(1, n_samples // (workers * 10))

    writer = TrajectoryWriter(
        output_dir=output_dir,
        prefix=f"{width}x{height}_{total_mines}",
        samples_per_file=samples_per_file,
        start_file_idx=start_file_idx
    )

    with multiprocessing.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(worker_func, seeds, chunksize=chunksize):
            total_attempts += 1
            if result is not None:
                total_generated += 1
                total_steps += len(result["actions"])
                writer.append(result)
                if pbar:
                    pbar.update(1)
                    
                if total_generated >= n_samples:
                    break

    if pool:
        pool.terminate()

    writer.flush()
    file_idx = writer.file_idx

    elapsed = time.time() - start
    if pbar:
        pbar.close()

    if existing_stats:
        total_attempts += existing_stats.get("attempts", 0)
        total_generated += existing_stats.get("generated", 0)
        total_steps += existing_stats.get("total_steps", 0)
        total_ambiguous += existing_stats.get("total_ambiguous_cells", 0)
        elapsed += existing_stats.get("elapsed_seconds", 0.0)

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
    parser.add_argument("--samples_per_file", type=int, default=2000,
                        help="Number of games per .npz file (default: 2000)")
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

    start_file_idx = 0
    existing_stats = None

    if output_dir.exists() and not args.force:
        stats_file = output_dir / "stats.json"
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    existing_stats = json.load(f)
                data_files = list(output_dir.glob(f"{args.width}x{args.height}_{args.mines}_*.npz"))
                if data_files:
                    indices = [int(p.stem.split('_')[-1]) for p in data_files]
                    start_file_idx = max(indices) + 1
                
                already_generated = existing_stats.get("generated", 0)
                print(f"📦 Found existing data: {already_generated} games.")
                
                if already_generated >= args.n_samples:
                    print(f"   Target of {args.n_samples} games already reached. Skipping generation.")
                    print(f"   (Use --force to regenerate, or specify a larger --n_samples to append more)")
                    return
                
                # Calculate how many MORE we need to reach the target
                needed_samples = args.n_samples - already_generated
                print(f"   Appending {needed_samples} new games starting from file index {start_file_idx}...")
                args.n_samples = needed_samples
                
            except Exception as e:
                print(f"⚠ Could not read existing stats: {e}. Starting fresh...")

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
            start_file_idx=start_file_idx,
            existing_stats=existing_stats,
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
            start_file_idx=start_file_idx,
            existing_stats=existing_stats,
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
            start_file_idx=start_file_idx,
            existing_stats=existing_stats,
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
