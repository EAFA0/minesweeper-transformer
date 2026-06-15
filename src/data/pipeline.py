"""Canonical training-data build pipeline."""

import argparse
import json
import multiprocessing
import re
import time
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np

from config import DATA_SCHEMA_VERSION, STAGE_DATASETS, get_stage_dataset
from data.generator import generate_training_data, generate_trajectory
from data.writer import TrajectoryWriter

DEFAULT_N_SAMPLES = 10000
DEFAULT_SAMPLES_PER_FILE = 2000


def _parallel_worker(seed, width, height, total_mines):
    """Wrapper for multiprocessing: seed -> generate_trajectory."""
    rng = np.random.default_rng(seed)
    return generate_trajectory(
        width=width, height=height, total_mines=total_mines, rng=rng
    )


def generate_training_data_parallel(
    output_dir: Path,
    n_samples: int = DEFAULT_N_SAMPLES,
    width: int = 8,
    height: int = 8,
    total_mines: int = 10,
    seed: int = 42,
    samples_per_file: int = DEFAULT_SAMPLES_PER_FILE,
    workers: int = 16,
    start_file_idx: int = 0,
    existing_stats: Optional[dict] = None,
    file_prefix: str | None = None,
    dataset_name: str = "",
) -> dict:
    """Multi-process training data generation for fixed-size boards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    file_prefix = file_prefix or f"train_{width}x{height}_{total_mines}"

    rng = np.random.default_rng(seed)
    n_seeds = int(n_samples * 1.5)
    seeds = rng.integers(0, 2**31 - 1, size=n_seeds)

    worker_func = partial(
        _parallel_worker, width=width, height=height, total_mines=total_mines
    )

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
        prefix=file_prefix,
        samples_per_file=samples_per_file,
        start_file_idx=start_file_idx,
    )

    with multiprocessing.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(worker_func, seeds, chunksize=chunksize):
            total_attempts += 1
            if result is None:
                continue

            total_generated += 1
            total_steps += len(result["actions"])
            total_ambiguous += int(result.get("ambiguous_steps", 0))
            writer.append(result)
            if pbar:
                pbar.update(1)

            if total_generated >= n_samples:
                break

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

    stats = _build_stats(
        width=width,
        height=height,
        total_mines=total_mines,
        n_samples=n_samples,
        workers=workers,
        dataset_name=dataset_name,
        file_prefix=file_prefix,
        total_attempts=total_attempts,
        total_generated=total_generated,
        total_steps=total_steps,
        total_ambiguous=total_ambiguous,
        elapsed=elapsed,
        output_files=file_idx,
    )
    write_stats(output_dir, stats)
    return stats


def generate_from_args(args) -> None:
    """Generate one or all canonical datasets from parsed CLI args."""
    if args.all_stages:
        for stage in STAGE_DATASETS:
            stage_args = argparse.Namespace(**vars(args))
            stage_args.stage = stage
            generate_one_dataset(stage_args)
        return

    generate_one_dataset(args)


def generate_one_dataset(args) -> None:
    """Generate one fixed-size dataset."""
    dataset_name = args.stage or ""
    if args.stage:
        dataset = get_stage_dataset(args.stage)
        args.width = dataset.width
        args.height = dataset.height
        args.mines = dataset.mines
        if args.n_samples == DEFAULT_N_SAMPLES:
            args.n_samples = dataset.n_samples
        if args.samples_per_file == DEFAULT_SAMPLES_PER_FILE:
            args.samples_per_file = dataset.samples_per_file
        output_dir = stage_output_dir(Path(args.output), args.stage)
        file_prefix = dataset.file_prefix
        print(
            f"Stage {args.stage}: {args.width}x{args.height}/{args.mines} "
            f"-> {output_dir}"
        )
    else:
        output_dir = Path(args.output)
        file_prefix = f"train_{args.width}x{args.height}_{args.mines}"

    start_file_idx = 0
    existing_stats = None

    if output_dir.exists() and not args.force:
        stats_file = output_dir / "stats.json"
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    existing_stats = json.load(f)
                start_file_idx = next_file_index(output_dir)

                already_generated = existing_stats.get("generated", 0)
                print(f"📦 Found existing data: {already_generated} games.")

                if already_generated >= args.n_samples:
                    print(
                        f"   Target of {args.n_samples} games already reached. "
                        "Skipping generation."
                    )
                    print(
                        "   (Use --force to regenerate, or specify a larger "
                        "--n_samples to append more)"
                    )
                    return

                needed_samples = args.n_samples - already_generated
                print(
                    f"   Appending {needed_samples} new games starting from "
                    f"file index {start_file_idx}..."
                )
                args.n_samples = needed_samples

            except Exception as e:
                print(f"⚠ Could not read existing stats: {e}. Starting fresh...")

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
            file_prefix=file_prefix,
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
            file_prefix=file_prefix,
            dataset_name=dataset_name,
        )

    update_stats_metadata(output_dir, stats, args, file_prefix, dataset_name)
    print_generation_summary(stats, output_dir, args)


def next_file_index(output_dir: Path) -> int:
    """Return the next chunk index for any canonical or legacy npz in a dir."""
    max_idx = -1
    for path in output_dir.glob("*.npz"):
        if path.name.startswith("eval_boards"):
            continue
        match = re.search(r"_(\d{4})\.npz$", path.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def stage_output_dir(output: Path, stage: str) -> Path:
    """Treat --output as root unless it already points at the stage dir."""
    return output if output.name == stage else output / stage


def update_stats_metadata(
    output_dir: Path,
    stats: dict,
    args,
    file_prefix: str,
    dataset_name: str,
) -> None:
    """Add canonical schema metadata to stats.json."""
    stats.setdefault("params", {})
    stats["params"].update(
        {
            "width": args.width,
            "height": args.height,
            "total_mines": args.mines,
            "n_samples_target": args.n_samples,
            "workers": args.workers,
            "label_type": "probability_distillation",
            "schema_version": DATA_SCHEMA_VERSION,
            "dataset_name": dataset_name,
            "file_prefix": file_prefix,
            "layout": "data/{stage}/train_{stage}_{width}x{height}_{mines}_{index}.npz",
        }
    )
    write_stats(output_dir, stats)


def write_stats(output_dir: Path, stats: dict) -> None:
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)


def print_generation_summary(stats: dict, output_dir: Path, args) -> None:
    print("\n📊 Generation complete!")
    print(f"   Generated: {stats['generated']} games from {stats['attempts']} attempts")

    print(f"   Avg steps per game: {stats['avg_steps_per_game']:.1f}")
    print(f"   Total training steps: {stats['total_steps']}")
    print(f"   Avg ambiguous cells/step: {stats.get('avg_ambig_per_game', 0):.1f}")
    print(f"   Output files: {stats['output_files']}")
    print(f"   Time: {stats['elapsed_seconds']:.1f}s")
    print(f"   Speed: {stats.get('games_per_second', 0):.1f} games/s")
    print(f"   Output dir: {output_dir.resolve()}")
    print("\n   Label type: probability distillation (MSE)")
    print("   Data format: (channels, probs, masks) per sample")


def _build_stats(
    width: int,
    height: int,
    total_mines: int,
    n_samples: int,
    workers: int,
    dataset_name: str,
    file_prefix: str,
    total_attempts: int,
    total_generated: int,
    total_steps: int,
    total_ambiguous: int,
    elapsed: float,
    output_files: int,
) -> dict:
    return {
        "params": {
            "width": width,
            "height": height,
            "total_mines": total_mines,
            "n_samples_target": n_samples,
            "workers": workers,
            "label_type": "probability_distillation",
            "schema_version": DATA_SCHEMA_VERSION,
            "dataset_name": dataset_name,
            "file_prefix": file_prefix,
            "layout": "data/{stage}/train_{stage}_{width}x{height}_{mines}_{index}.npz",
        },
        "attempts": total_attempts,
        "generated": total_generated,
        "total_steps": total_steps,
        "total_ambiguous_cells": total_ambiguous,
        "elapsed_seconds": elapsed,
        "avg_steps_per_game": total_steps / max(1, total_generated),
        "avg_ambig_per_game": total_ambiguous / max(1, total_generated),
        "output_files": output_files,
        "games_per_second": total_generated / elapsed if elapsed > 0 else 0,
    }
