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

    # Generate no-guess board (or use self_validated if no_guess is too slow)
    from data.self_validated import generate_self_validated_board
    mine_mask, visible, _ = generate_self_validated_board(width, height, total_mines, rng=rng)
    if mine_mask is None or visible is None:
        return None

    game = MinesweeperGame.from_mine_mask(
        width, height, mine_mask, first_done=True, visible=visible
    )

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
    samples_per_file: int = 100,
) -> dict:
    """Generate and save trajectory dataset sequentially."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    
    total_saved = 0
    file_idx = 0
    buffer = []
    
    start_time = time.time()
    
    while total_saved < n_samples:
        traj = generate_trajectory(width, height, total_mines, compute_probs=True, rng=rng)
        if traj is not None:
            buffer.append(traj)
            total_saved += 1
            
            if len(buffer) >= samples_per_file:
                save_trajectory_buffer(buffer, output_dir, file_idx)
                file_idx += 1
                buffer.clear()
                print(f"Generated {total_saved}/{n_samples} trajectories...")
                
    if buffer:
        save_trajectory_buffer(buffer, output_dir, file_idx)
        
    duration = time.time() - start_time
    print(f"Finished generating {total_saved} trajectories in {duration:.1f}s")
    
    return {
        "n_trajectories": total_saved,
        "width": width,
        "height": height,
        "mines": total_mines,
        "duration": duration,
    }

def save_trajectory_buffer(
    buffer: List[dict],
    output_dir: Path,
    file_idx: int,
) -> Path:
    """Save a batch of full trajectories to a compressed .npz file."""
    data = {}
    for i, traj in enumerate(buffer):
        data[f"mines_{i}"] = traj["mines"]
        data[f"actions_{i}"] = np.array(traj["actions"], dtype=np.int32)
        data[f"masks_{i}"] = np.array(traj["masks"], dtype=bool)
        if "probs" in traj:
            data[f"probs_{i}"] = np.array(traj["probs"], dtype=np.float32)
            
    out_path = output_dir / f"data_{file_idx:04d}.npz"
    np.savez_compressed(out_path, **data)
    return out_path

if __name__ == "__main__":
    # Simple CLI for testing/generation
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--output", type=str, default="data/training")
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
