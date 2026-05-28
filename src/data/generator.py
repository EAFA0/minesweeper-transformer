"""Training data generation pipeline.

Generates supervised training data for Phase 1:
- Random minesweeper boards (8×8, 10 mines)
- Constraint-propagation solver identifies provably safe/mine cells
- Records (board_state → labels) pairs at each step
- Only keeps fully solvable (no-guess) boards
"""

import json
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from minesweeper.game import MinesweeperGame
from minesweeper.constants import CellState, MoveType, GameStatus
from minesweeper.solver import ConstraintSolver


def generate_solvable_board(
    width: int = 8,
    height: int = 8,
    total_mines: int = 10,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 100,
) -> Optional[MinesweeperGame]:
    """Generate a board that's fully solvable by constraint propagation.

    Strategy:
    1. Generate random board with random first click
    2. Run constraint solver step-by-step
    3. If solver gets stuck (no deductive moves), discard
    4. If solver clears all safe cells, accept

    Returns the solved game, or None if max_attempts exceeded.
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(max_attempts):
        game = MinesweeperGame(width, height, total_mines)

        # Random first click — avoid edges for better openings
        r = rng.integers(1, height - 1)
        c = rng.integers(1, width - 1)
        game.make_move(r, c, MoveType.REVEAL)

        if game.status != GameStatus.PLAYING:
            continue  # shouldn't happen with first-click safety

        # Step-by-step constraint solving
        solver = ConstraintSolver(game)
        stuck = False

        while game.status == GameStatus.PLAYING:
            safe, mines = solver.find_safe_and_mines()

            if not safe and not mines:
                # No deductive moves available — game is not solvable
                stuck = True
                break

            # Reveal all safe cells
            for sr, sc in safe:
                game.make_move(sr, sc, MoveType.REVEAL)
                if game.status != GameStatus.PLAYING:
                    break

            # Flag all mine cells
            for mr, mc in mines:
                if game.visible[mr, mc] == CellState.COVERED:
                    game.make_move(mr, mc, MoveType.FLAG)

        if not stuck and game.status == GameStatus.WON:
            return game

    return None


def record_game_trajectory(
    width: int = 8,
    height: int = 8,
    total_mines: int = 10,
    rng: Optional[np.random.Generator] = None,
    require_win: bool = False,
    min_steps: int = 1,
    use_no_guess: bool = True,
) -> Optional[dict]:
    """Play through a board, recording (state, labels) at each step with solver guidance.

    If use_no_guess=True: generates boards guaranteed to be solvable without guessing
    (via ms-toollib). This ensures every training step has a logically deducible answer,
    producing cleaner training data.

    If require_win=True: only returns fully solvable trajectories (Phase 1).
    If require_win=False: returns partial trajectories — stops recording when solver
    gets stuck but keeps all previously recorded steps (Phase 2+ warmup).

    Returns None if no steps recorded or generation fails.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate the board
    if use_no_guess:
        from data.no_guess import generate_no_guess_board
        game = generate_no_guess_board(
            width=width, height=height, total_mines=total_mines,
            rng=rng, max_attempts=100,
        )
        if game is None:
            return None
    else:
        game = MinesweeperGame(width, height, total_mines)
        r = rng.integers(0, height)
        c = rng.integers(0, width)
        game.make_move(r, c, MoveType.REVEAL)

    if game.status != GameStatus.PLAYING:
        return None

    mine_mask = game.get_mine_mask()
    steps = []

    solver = ConstraintSolver(game)
    step_idx = 0

    while game.status == GameStatus.PLAYING:
        safe, mines = solver.find_safe_and_mines()

        if not safe and not mines:
            # Stuck — stop recording but don't discard
            break

        # Record state before making moves
        channels = game.board_to_channels().copy()
        labels = game.get_labels()
        mask = game.get_label_mask()

        steps.append({
            "step": step_idx,
            "channels": channels,
            "labels": labels,
            "mask": mask,
            "n_safe": len(safe),
            "n_mines": len(mines),
        })

        # Execute deductions
        for sr, sc in safe:
            game.make_move(sr, sc, MoveType.REVEAL)
            if game.status != GameStatus.PLAYING:
                break

        for mr, mc in mines:
            if game.visible[mr, mc] == CellState.COVERED:
                game.make_move(mr, mc, MoveType.FLAG)

        step_idx += 1

    if require_win and game.status != GameStatus.WON:
        return None

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
    require_win: bool = False,
    use_no_guess: bool = True,
) -> dict:
    """Generate training data and save to disk.

    When require_win=True: only keeps fully solvable boards (Phase 1, low density).
    When require_win=False: accepts partial trajectories — stops recording when
    solver gets stuck but keeps all valid deduction steps (Phase 2+, high density).

    Returns summary dict with generation statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    stats = {
        "params": {
            "width": width, "height": height, "total_mines": total_mines,
            "n_samples_target": n_samples, "require_win": require_win,
        },
        "attempts": 0,
        "generated": 0,
        "total_steps": 0,
        "start_time": time.time(),
    }

    buffer = []
    file_idx = 0

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_samples, desc="Generating training data")
        except ImportError:
            pass

    while stats["generated"] < n_samples:
        stats["attempts"] += 1
        trajectory = record_game_trajectory(
            width=width, height=height, total_mines=total_mines,
            rng=rng, require_win=require_win, use_no_guess=use_no_guess,
        )

        if trajectory is None:
            continue

        stats["generated"] += 1
        stats["total_steps"] += trajectory["n_steps"]
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
    stats["avg_steps_per_game"] = stats["total_steps"] / max(1, stats["generated"])
    stats["output_files"] = file_idx

    # Save stats
    with open(output_dir / "stats.json", "w") as f:
        # Convert numpy types for JSON serialization
        json.dump(stats, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))

    if pbar:
        pbar.close()

    return stats


def _save_buffer(buffer: List[dict], output_dir: Path, file_idx: int) -> None:
    """Save a batch of trajectories to a compressed .npz file."""
    # Flatten: each step becomes one training sample
    all_channels = []
    all_labels = []
    all_masks = []

    for traj in buffer:
        for step in traj["trajectory"]:
            all_channels.append(step["channels"])
            all_labels.append(step["labels"])
            all_masks.append(step["mask"])

    filepath = output_dir / f"data_{file_idx:04d}.npz"
    np.savez_compressed(
        filepath,
        channels=np.stack(all_channels),
        labels=np.stack(all_labels),
        masks=np.stack(all_masks),
        n_games=len(buffer),
        n_samples=len(all_channels),
    )

    # Also save mine masks for reference
    mine_masks = np.stack([t["mine_mask"] for t in buffer])
    meta_path = output_dir / f"meta_{file_idx:04d}.npz"
    np.savez_compressed(meta_path, mine_masks=mine_masks)
