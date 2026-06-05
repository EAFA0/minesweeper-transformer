"""Mixed-size training data generation.

Generates a single unified dataset with variable board sizes and mine densities.
All samples are padded to a uniform max_size (default 8×8) so the model can
batch them together. The mask excludes padded cells from loss.

Usage:
    python scripts/generate_data.py --mixed \
        --min_size 4 --max_size 8 \
        --min_density 0.1 --max_density 0.5 \
        --n_samples 12000 --output data/mixed
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from game.constants import MoveType, GameStatus
from game.probability_solver import ProbabilitySolver
from data.writer import TrajectoryWriter


def generate_mixed_data(
    output_dir: Path,
    n_samples: int = 12000,
    min_size: int = 4,
    max_size: int = 8,
    min_density: float = 0.1,
    max_density: float = 0.5,
    seed: int = 42,
    samples_per_file: int = 2000,
    show_progress: bool = True,
    start_file_idx: int = 0,
    existing_stats: Optional[dict] = None,
) -> dict:
    """Generate mixed training data with variable boards and densities.

    Each sample: random size (w,h) ∈ [min_size, max_size],
    random density ∈ [min_density, max_density], padded to max_size × max_size.
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "generated": 0,
        "attempts": 0,
        "total_steps": 0,
        "total_ambiguous_cells": 0,
        "start_time": time.time(),
        "config": {
            "n_samples": n_samples,
            "min_size": min_size, "max_size": max_size,
            "min_density": min_density, "max_density": max_density,
            "seed": seed,
        },
    }

    writer = TrajectoryWriter(
        output_dir=output_dir,
        prefix=f"mixed_{max_size}x{max_size}",
        samples_per_file=samples_per_file,
        start_file_idx=start_file_idx
    )

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_samples, desc="Generating mixed data")
        except ImportError:
            pass

    while stats["generated"] < n_samples:
        stats["attempts"] += 1

        # Random board config
        w = rng.integers(min_size, max_size + 1)
        h = rng.integers(min_size, max_size + 1)
        density = rng.uniform(min_density, max_density)
        mines = max(1, int(w * h * density))

        # We need to adapt the returned trajectory to fit TrajectoryWriter.
        # Wait, TrajectoryWriter expects 'mines', 'actions', 'masks', 'probs'
        # The mixed generator creates padded states. It is a bit different.
        # Let's fix this cleanly.

        trajectory = _record_padded_trajectory(
            w=w, h=h, mines=mines, pad_to=max_size, rng=rng,
        )

        if trajectory is None:
            continue

        stats["generated"] += 1
        stats["total_steps"] += trajectory["n_steps"]

        for step in trajectory["trajectory"]:
            stats["total_ambiguous_cells"] += step["n_ambiguous"]

        # Convert to TrajectoryWriter format
        # padded mines, actions, masks, probs
        actions = trajectory["actions"]
        writer.append({
            "mines": trajectory["mines_pad"],
            "actions": actions,
            "masks": trajectory["masks_pad"],
            "probs": trajectory["probs_pad"],
        })

        if pbar:
            pbar.update(1)
            pbar.set_postfix({
                "size": f"{w}×{h}", "mines": mines,
                "ok": stats["generated"],
            })

    writer.flush()

    stats["end_time"] = time.time()
    stats["elapsed_seconds"] = stats["end_time"] - stats["start_time"]

    if existing_stats:
        stats["attempts"] += existing_stats.get("attempts", 0)
        stats["generated"] += existing_stats.get("generated", 0)
        stats["total_steps"] += existing_stats.get("total_steps", 0)
        stats["total_ambiguous_cells"] += existing_stats.get("total_ambiguous_cells", 0)
        stats["elapsed_seconds"] += existing_stats.get("elapsed_seconds", 0.0)

    stats["avg_steps_per_game"] = stats["total_steps"] / max(1, stats["generated"])
    stats["output_files"] = writer.file_idx

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    if pbar:
        pbar.close()

    return stats


def _record_padded_trajectory(
    w: int, h: int, mines: int, pad_to: int,
    rng: np.random.Generator,
) -> Optional[dict]:
    """Play through a no-guess board, recording padded states."""
    from data.no_guess import generate_no_guess_board

    # Generate no-guess board at actual size
    game = generate_no_guess_board(
        width=w, height=h, total_mines=mines,
        rng=rng, max_attempts=200,
    )
    if game is None or game.status != GameStatus.PLAYING:
        return None

    mine_mask = game.get_mine_mask()
    mines_pad = np.zeros((pad_to, pad_to), dtype=bool)
    mines_pad[:h, :w] = mine_mask

    steps = []
    actions = []
    masks_pad = []
    probs_pad_list = []

    while game.status == GameStatus.PLAYING and len(steps) < 300:
        solver = ProbabilitySolver(game)
        probs = solver.compute_probabilities()
        if probs is None:
            break

        mask = game.covered_cells
        
        # Padded mask: True means covered. Pad area is covered.
        mask_pad = np.ones((pad_to, pad_to), dtype=bool)
        mask_pad[:h, :w] = mask
        masks_pad.append(mask_pad)

        prob_pad = np.zeros((pad_to, pad_to), dtype=np.float32)
        prob_pad[:h, :w] = probs
        probs_pad_list.append(prob_pad)

        n_ambig = int((probs[mask] > 0.01).sum())
        steps.append({
            "n_ambiguous": n_ambig,
        })

        covered = game.covered_cells
        masked_probs = np.where(covered, probs, 2.0)
        best_idx = np.argmin(masked_probs)
        best_r, best_c = divmod(int(best_idx), w)
        
        # Action index in padded coordinate space
        actions.append(best_r * pad_to + best_c)
        
        game.make_move(best_r, best_c, MoveType.REVEAL)

    return {
        "mines_pad": mines_pad,
        "actions": actions,
        "masks_pad": masks_pad,
        "probs_pad": probs_pad_list,
        "trajectory": steps,
        "n_steps": len(steps),
        "board_size": (h, w),
        "mines": mines,
    }



