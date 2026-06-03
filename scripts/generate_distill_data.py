#!/usr/bin/env python3
"""Generate distillation training data using solver-computed probabilities.

For each no-guess board, we play through the game using the solver and record
(channels, solver_probs, mask) at every step. These serve as supervised
training samples.

Usage:
    python scripts/generate_distill_data.py --width 6 --height 6 --mines 18 --n_games 50000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data.generator import record_game_trajectory


def generate(args):
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    samples_per_file = 10000
    buffer: list = []
    file_idx = 0
    total_games = 0
    total_samples = 0
    fail_count = 0
    t0 = time.time()

    print(f"Target: {args.n_games} games for {args.width}×{args.height}/{args.mines}")
    print(f"Output: {data_dir}/")
    print(f"Samples per file: {samples_per_file}")

    while total_games < args.n_games:
        result = record_game_trajectory(
            width=args.width,
            height=args.height,
            total_mines=args.mines,
            rng=rng,
            min_steps=args.min_steps,
        )
        if result is None:
            fail_count += 1
            if fail_count <= 10:
                print(f"  FAIL #{fail_count} — skipping")
            continue

        total_games += 1
        for step_data in result["trajectory"]:
            buffer.append((
                step_data["channels"],
                step_data["probs"],
                step_data["mask"],
            ))
            total_samples += 1

        # Flush buffer periodically
        if len(buffer) >= samples_per_file:
            _save_buffer(buffer, data_dir, file_idx)
            file_idx += 1
            buffer.clear()

        # Progress
        if total_games % 500 == 0:
            elapsed = time.time() - t0
            rate = total_games / max(elapsed, 1)
            print(f"  {total_games}/{args.n_games} games, {total_samples} samples, "
                  f"{rate:.1f} games/s, {fail_count} failures", flush=True)

    # Save remaining
    if buffer:
        _save_buffer(buffer, data_dir, file_idx)
        file_idx += 1

    elapsed = time.time() - t0
    print(f"\nDone: {total_games} games, {total_samples} samples, "
          f"{fail_count} failures in {elapsed:.1f}s")
    print(f"Files: {file_idx}")


def _save_buffer(buffer, data_dir, file_idx):
    """Save accumulated samples to a .npz file."""
    channels = np.stack([s[0] for s in buffer])
    probs = np.stack([s[1] for s in buffer])
    masks = np.stack([s[2] for s in buffer])
    out_path = data_dir / f"trajectories_{file_idx:04d}.npz"
    np.savez_compressed(out_path, channels=channels, probs=probs, masks=masks)


def main():
    p = argparse.ArgumentParser(
        description="Generate solver-distillation training data")
    p.add_argument("--width", type=int, default=6)
    p.add_argument("--height", type=int, default=6)
    p.add_argument("--mines", type=int, default=18)
    p.add_argument("--n_games", type=int, default=10000,
                   help="Number of complete games to record")
    p.add_argument("--data_dir", default="data/distill/6x6_18")
    p.add_argument("--min_steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
