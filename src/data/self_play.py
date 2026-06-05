"""Self-play data collection for DAgger-style iterative training.

The model plays through boards, and at each step the ProbabilitySolver
provides ground-truth labels. This generates on-policy training data
that covers the states the model actually encounters, reducing
distribution shift compared to pure solver-guided trajectories.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

from game.game import MinesweeperGame
from game.constants import MoveType, GameStatus, CellState
from game.probability_solver import ProbabilitySolver
from data.no_guess import generate_no_guess_board
from data.writer import StateWriter
from config import TrainingConfig

_DEFAULT_CFG = TrainingConfig()


def collect_self_play_trajectory(
    game: MinesweeperGame,
    probs_2d: np.ndarray,
    max_steps: int = 200,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Play through a game using model-predicted probabilities.

    At each step:
    1. Record current (channels, solver_probs, mask)
    2. Pick cell with lowest model-predicted P(mine)
    3. Reveal and continue

    Returns list of (channels, probs, mask) — same format as supervised data.
    """
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    solver = ProbabilitySolver(game)

    for _ in range(max_steps):
        if game.status != GameStatus.PLAYING:
            break

        # Record current state with solver labels
        channels = game.board_to_channels()
        covered = game.covered_cells

        if not covered.any():
            break

        solver.reset()
        solver.set_board(game.board)
        solver_probs = solver.solve()

        mask = covered
        samples.append((channels, solver_probs, mask))

        # Pick model's best move
        masked_probs = np.where(mask, probs_2d, 2.0)
        best_idx = int(np.argmin(masked_probs))
        r, c = divmod(best_idx, game.width)

        game.make_move(r, c, MoveType.REVEAL)

    return samples


def generate_self_play_data(
    output_dir: Path,
    model,                    # MinesweeperTransformer
    device: str,
    width: int = _DEFAULT_CFG.board_width,
    height: int = _DEFAULT_CFG.board_height,
    total_mines: int = _DEFAULT_CFG.board_mines,
    n_games: int = 100,
    max_steps: int = 200,
    max_attempts: int = 100,
    refine_steps: int = 4,
    start_file_idx: int = 0,
    seed: int | None = None,
) -> dict:
    """Generate self-play training data.

    The model plays n_games, recording solver-labeled states at each step.
    Data is saved in the same .npz format as supervised training data.

    Returns stats dict with total_games, total_samples, files_written.
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    writer = StateWriter(
        output_dir=output_dir,
        prefix=f"{width}x{height}_{total_mines}",
        samples_per_file=2000,
        start_file_idx=start_file_idx
    )

    total_samples = 0
    total_games = 0

    model.eval()

    for game_idx in range(n_games):
        game = generate_no_guess_board(width, height, total_mines, max_attempts=max_attempts)
        if game is None:
            continue

        total_games += 1

        # Model plays the game
        while game.status == GameStatus.PLAYING:
            covered = game.covered_cells
            if not covered.any():
                break

            # Record state with solver labels
            channels_np = game.board_to_channels()
            solver = ProbabilitySolver(game)
            solver_probs = solver.compute_probabilities()

            sample = {
                "channels": channels_np.astype(np.float32),
                "probs": solver_probs.astype(np.float32),
                "mask": covered.copy(),
            }
            writer.append(sample)
            total_samples += 1

            # Model prediction
            ch_t = torch.from_numpy(channels_np).float().unsqueeze(0).to(device)
            with torch.no_grad():
                B, C, H, W = ch_t.shape
                mem = torch.zeros(B, 64, H, W, device=device)
                pv = torch.full((B, 1, H, W), 0.5, device=device)

                for step in range(refine_steps):
                    pv_old = pv.clone() if step > 0 else None
                    pv, mem = model._single_pass(ch_t, pv, mem)
                    if step > 0 and pv_old is not None:
                        max_change = (pv - pv_old).abs().max().item()
                        if max_change < 0.01:
                            break

            model_probs = pv.squeeze().cpu().numpy()

            # Pick safest cell and move
            masked = np.where(covered, model_probs, 2.0)
            best_idx = int(np.argmin(masked))
            r, c = divmod(best_idx, game.width)

            game.make_move(r, c, MoveType.REVEAL)

        # (Flushing is handled automatically by writer)

    # Flush remaining
    writer.flush()

    return {
        "total_games": total_games,
        "total_samples": total_samples,
        "files_written": writer.file_idx - start_file_idx,
    }
