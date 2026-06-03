"""Training data generation pipeline.

Generates supervised training data for probability distillation:
- No-guess minesweeper boards (8×8, 10 mines) via ms-toollib
- At each step, ProbabilitySolver computes exact P(mine) per covered cell
- Records (board_state → probability_matrix) pairs
- Model learns to estimate the solver's probability distribution (MSE loss)
"""

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from minesweeper_transformer.minesweeper.game import MinesweeperGame
from minesweeper_transformer.minesweeper.constants import CellState, MoveType, GameStatus
from minesweeper_transformer.minesweeper.probability_solver import ProbabilitySolver


def save_trajectory_buffer(
    buffer: List[dict],
    output_dir: Path,
    file_idx: int,
    *,
    include_counts: bool = True,
) -> Path:
    """Save trajectory steps to a compressed training data file."""
    all_channels = []
    all_probs = []
    all_masks = []

    for traj in buffer:
        for step in traj["trajectory"]:
            all_channels.append(step["channels"])
            all_probs.append(step["probs"])
            all_masks.append(step["mask"])

    save_dict = {
        "channels": np.stack(all_channels),
        "probs": np.stack(all_probs),
        "masks": np.stack(all_masks),
    }
    if include_counts:
        save_dict["n_games"] = np.array(len(buffer))
        save_dict["n_samples"] = np.array(len(all_channels))

    filepath = output_dir / f"data_{file_idx:04d}.npz"
    np.savez_compressed(filepath, **save_dict)
    return filepath


def record_game_trajectory(
    width: int = 8,
    height: int = 8,
    total_mines: int = 10,
    rng: Optional[np.random.Generator] = None,
    min_steps: int = 2,
) -> Optional[dict]:
    """Play through a no-guess board, recording (state, probability_matrix) at each step.

    At each step:
    1. ProbabilitySolver computes exact P(mine) for every covered cell
    2. Record (channels, probs, mask)
    3. Reveal the covered cell with lowest P(mine) to advance the game

    On no-guess boards, at least one cell always has P(mine)=0,
    so the game never gets stuck.

    Returns None if no steps recorded or generation fails.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate no-guess board
    from minesweeper_transformer.data.no_guess import generate_no_guess_board
    game = generate_no_guess_board(
        width=width, height=height, total_mines=total_mines,
        rng=rng, max_attempts=100,
    )
    if game is None:
        return None

    if game.status != GameStatus.PLAYING:
        return None

    mine_mask = game.get_mine_mask()
    steps = []
    step_idx = 0

    while game.status == GameStatus.PLAYING:
        # Compute probability matrix for current state
        prob_solver = ProbabilitySolver(game)
        probs = prob_solver.compute_probabilities()

        # Record state
        channels = game.board_to_channels().copy()
        mask = game.get_label_mask().copy()

        # Count stats for logging
        n_safe = int(np.sum((probs == 0.0) & mask))
        n_mine = int(np.sum((probs == 1.0) & mask))
        n_ambig = int(np.sum((probs > 0.0) & (probs < 1.0) & mask))

        steps.append({
            "step": step_idx,
            "channels": channels,
            "probs": probs.copy(),
            "mask": mask,
            "n_deduced_safe": n_safe,
            "n_deduced_mine": n_mine,
            "n_ambiguous": n_ambig,
        })

        # Find and reveal the cell with lowest P(mine) among covered cells
        covered = game.covered_cells
        if not covered.any():
            break

        masked_probs = np.where(covered, probs, 2.0)
        best_idx = np.argmin(masked_probs)
        best_r, best_c = divmod(int(best_idx), game.width)

        if probs[best_r, best_c] > 0.0:
            # Shouldn't happen on no-guess boards, but handle gracefully
            # If min prob > 0, we'd be guessing — skip this step
            break

        game.make_move(best_r, best_c, MoveType.REVEAL)
        step_idx += 1

    if len(steps) < min_steps:
        return None

    return {
        "width": width,
        "height": height,
        "total_mines": total_mines,
        "mine_mask": mine_mask,
        "n_steps": len(steps),
        "trajectory": steps,
    }


def generate_training_data(
    output_dir: Path,
    n_samples: int = 1000,
    width: int = 8,
    height: int = 8,
    total_mines: int = 10,
    seed: int = 42,
    samples_per_file: int = 100,
    show_progress: bool = True,
    start_file_idx: int = 0,
    existing_stats: Optional[dict] = None,
) -> dict:
    """Generate training data and save to disk.

    Returns summary dict with generation statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    stats = {
        "params": {
            "width": width, "height": height, "total_mines": total_mines,
            "n_samples_target": n_samples,
            "label_type": "probability_distillation",
        },
        "attempts": 0,
        "generated": 0,
        "total_steps": 0,
        "total_ambiguous_cells": 0,
        "start_time": time.time(),
    }

    buffer = []
    file_idx = start_file_idx

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_samples, desc="Generating training data (prob distillation)")
        except ImportError:
            pass

    while stats["generated"] < n_samples:
        stats["attempts"] += 1
        trajectory = record_game_trajectory(
            width=width, height=height, total_mines=total_mines, rng=rng,
        )

        if trajectory is None:
            continue

        stats["generated"] += 1
        stats["total_steps"] += trajectory["n_steps"]

        # Count ambiguous cells across all steps
        for step in trajectory["trajectory"]:
            stats["total_ambiguous_cells"] += step["n_ambiguous"]

        buffer.append(trajectory)

        if pbar:
            pbar.update(1)

        # Flush buffer to disk
        if len(buffer) >= samples_per_file:
            _save_buffer(buffer, output_dir, file_idx)
            buffer = []
            file_idx += 1

    # Flush remaining
    if buffer:
        _save_buffer(buffer, output_dir, file_idx)
        file_idx += 1

    stats["end_time"] = time.time()
    stats["elapsed_seconds"] = stats["end_time"] - stats["start_time"]
    
    if existing_stats:
        stats["attempts"] += existing_stats.get("attempts", 0)
        stats["generated"] += existing_stats.get("generated", 0)
        stats["total_steps"] += existing_stats.get("total_steps", 0)
        stats["total_ambiguous_cells"] += existing_stats.get("total_ambiguous_cells", 0)
        stats["elapsed_seconds"] += existing_stats.get("elapsed_seconds", 0.0)

    stats["avg_steps_per_game"] = stats["total_steps"] / max(1, stats["generated"])
    stats["avg_ambig_per_game"] = stats["total_ambiguous_cells"] / max(1, stats["total_steps"])
    stats["output_files"] = file_idx

    # Save stats
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))

    if pbar:
        pbar.close()

    return stats


def _save_buffer(buffer: List[dict], output_dir: Path, file_idx: int) -> None:
    """Save a batch of trajectories to a compressed .npz file."""
    save_trajectory_buffer(buffer, output_dir, file_idx, include_counts=True)

    # Also save mine masks for reference
    mine_masks = np.stack([t["mine_mask"] for t in buffer])
    meta_path = output_dir / f"meta_{file_idx:04d}.npz"
    np.savez_compressed(meta_path, mine_masks=mine_masks)
