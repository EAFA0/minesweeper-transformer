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

from game.game import MinesweeperGame
from game.constants import CellState, MoveType, GameStatus
from game.probability_solver import ProbabilitySolver
from config import TrainingConfig
from data.writer import TrajectoryWriter

_DEFAULT_CFG = TrainingConfig()

def generate_trajectory(
    width: int = _DEFAULT_CFG.board_width,
    height: int = _DEFAULT_CFG.board_height,
    total_mines: int = _DEFAULT_CFG.board_mines,
    compute_probs: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Optional[dict]:
    """Play through a board, recording the full trajectory.
    
    Returns:
        Dict with keys:
        - "mines": (H, W) boolean array
        - "actions": list of (r, c) tuples
        - "masks": list of (H, W) boolean arrays (True = covered)
        - "probs": list of (H, W) float arrays (only if compute_probs=True)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate self-validated board
    from data.self_validated import generate_self_validated_board
    game = generate_self_validated_board(width, height, total_mines, rng=rng)
    if game is None:
        return None
        
    mine_mask = game.get_mine_mask()

    traj = {
        "mines": mine_mask.copy(),
        "actions": [],
        "masks": [],
    }
    if compute_probs:
        traj["probs"] = []

    while game.status == GameStatus.PLAYING:
        covered = game.covered_cells
        traj["masks"].append(covered.copy())
        
        solver = ProbabilitySolver(game)
        probs = solver.compute_probabilities()
        
        if compute_probs:
            traj["probs"].append(probs.astype(np.float32))
            
        masked_probs = np.where(covered, probs, 2.0)
        best_idx = int(np.argmin(masked_probs))
        r, c = divmod(best_idx, width)
        
        traj["actions"].append((r, c))
        game.make_move(r, c, MoveType.REVEAL)
        
    return traj


def generate_training_data(
    output_dir: Path,
    n_samples: int = 1000,
    width: int = 8,
    height: int = 8,
    total_mines: int = 10,
    seed: int = 42,
    samples_per_file: int = 2000,
    start_file_idx: int = 0,
    show_progress: bool = True,
    existing_stats: Optional[dict] = None,
) -> dict:
    """Generate and save trajectory dataset sequentially."""
    rng = np.random.default_rng(seed)
    
    writer = TrajectoryWriter(
        output_dir=output_dir,
        prefix=f"{width}x{height}_{total_mines}",
        samples_per_file=samples_per_file,
        start_file_idx=start_file_idx
    )
    
    total_saved = 0
    total_attempts = 0
    total_steps = 0
    start_time = time.time()
    
    while total_saved < n_samples:
        total_attempts += 1
        traj = generate_trajectory(width, height, total_mines, compute_probs=True, rng=rng)
        if traj is not None:
            writer.append(traj)
            total_saved += 1
            total_steps += len(traj["actions"])
            if total_saved % samples_per_file == 0:
                print(f"Generated {total_saved}/{n_samples} trajectories...")
                
    writer.flush()
        
    duration = time.time() - start_time
    print(f"Finished generating {total_saved} trajectories in {duration:.1f}s")
    
    if existing_stats:
        total_attempts += existing_stats.get("attempts", 0)
        total_saved += existing_stats.get("generated", 0)
        total_steps += existing_stats.get("total_steps", 0)
        duration += existing_stats.get("elapsed_seconds", 0.0)

    return {
        "generated": total_saved,
        "attempts": total_attempts,
        "total_steps": total_steps,
        "avg_steps_per_game": total_steps / max(1, total_saved),
        "elapsed_seconds": duration,
        "output_files": writer.file_idx,
    }



if __name__ == "__main__":
    # Simple CLI for testing/generation
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--output", type=str, default="data")
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--height", type=int, default=8)
    p.add_argument("--mines", type=int, default=10)
    args = p.parse_args()
    
    generate_training_data(
        Path(args.output), 
        n_samples=args.n_samples,
        width=args.width,
        height=args.height,
        total_mines=args.mines
    )
