"""Self-validated board generator — fast alternative to ms-toollib.

Uses our own ProbabilitySolver to verify board solvability.
Much faster than ms-toollib's SAT solver (~0.1-1s vs 10-50s per board).

Algorithm:
  1. Randomly place mines
  2. Zero-cell first click to open a safe region
  3. N warmup clicks: reveal random safe cells (opens the board for the model)
  4. ProbabilitySolver tries to finish the board
  5. If solved → return game with warmup clicks applied
  6. If stuck → retry with new random board

The warmup clicks ensure the model starts with enough visible information.
Without them, dense boards have too few visible cells for any model to reason.
"""

import time
from typing import Optional

import numpy as np

from game.game import MinesweeperGame
from game.constants import CellState, MoveType, GameStatus
from game.probability_solver import ProbabilitySolver
from config import TrainingConfig

_DEFAULT_CFG = TrainingConfig()

def generate_self_validated_board(
    width: int = _DEFAULT_CFG.board_width,
    height: int = _DEFAULT_CFG.board_height,
    total_mines: int = _DEFAULT_CFG.board_mines,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 50,
    max_steps: int = 300,
    warmup_clicks: int = 0,
    verbose: bool = False,
) -> Optional[MinesweeperGame]:
    """Generate a board that ProbabilitySolver can solve.

    Returns a game with first click + warmup clicks already applied,
    or None if max_attempts exceeded.

    warmup_clicks: number of random safe reveals to open the board.
      Higher values give the model more initial information.
      Default 0 (just the first click) — works well for sparse boards.
      Recommend 3-5 for dense boards (8×8/20+ mines).
    """
    if rng is None:
        rng = np.random.default_rng()

    for attempt in range(max_attempts):
        t0 = time.time()

        # Create game with random mines
        game = MinesweeperGame(width, height, total_mines)

        # Find a zero cell for first click
        zero_cells = []
        for rr in range(height):
            for cc in range(width):
                if game.board[rr, cc] != -1:
                    if game._count_adjacent_mines(rr, cc) == 0:
                        zero_cells.append((rr, cc))

        if not zero_cells:
            continue

        r, c = zero_cells[rng.integers(0, len(zero_cells))]
        game.make_move(r, c, MoveType.REVEAL)
        if game.status != GameStatus.PLAYING:
            continue

        # Apply warmup clicks: reveal random safe cells to open more of the board
        for _ in range(warmup_clicks):
            covered = game.covered_cells
            safe = []
            for rr in range(height):
                for cc in range(width):
                    if covered[rr, cc] and game.board[rr, cc] != -1:
                        safe.append((rr, cc))
            if not safe:
                break
            wr, wc = safe[rng.integers(0, len(safe))]
            game.make_move(wr, wc, MoveType.REVEAL)

        # Save state at this point (after warmup, before solver)
        mine_mask = game.get_mine_mask()
        visible_snapshot = game.visible.copy()

        # Try to solve from here
        solved = _try_solve_from(game, max_steps=max_steps)

        dt = time.time() - t0
        if solved:
            if verbose:
                visible = int((visible_snapshot >= 0).sum())
                print(f"  Attempt {attempt+1}: ✓ {dt:.1f}s  revealed={visible}/{width*height}")
            return MinesweeperGame.from_mine_mask(
                width, height, mine_mask,
                first_done=True, visible=visible_snapshot,
            )

        if verbose:
            print(f"  Attempt {attempt+1}: ✗ {dt:.1f}s")

    return None


def _try_solve_from(game: MinesweeperGame, max_steps: int = 300,
                    max_hints: int = 5) -> bool:
    """Try to solve from current game state using ProbabilitySolver.

    When stuck (no P=0 cell), uses a 'hint' — reveals a random
    safe cell. Gives up after max_hints hints.

    Returns True if the solver completes the board.
    """
    hints_used = 0

    for _ in range(max_steps):
        if game.status != GameStatus.PLAYING:
            return game.status == GameStatus.WON

        solver = ProbabilitySolver(game)
        probs = solver.compute_probabilities()

        if probs is None:
            return False

        covered = game.covered_cells
        safe_mask = covered & (probs == 0.0)

        if safe_mask.any():
            r, c = np.argwhere(safe_mask)[0]
            game.make_move(r, c, MoveType.REVEAL)
            if game.status == GameStatus.LOST:
                return False
        else:
            if hints_used >= max_hints:
                return False
            hints_used += 1
            if not _reveal_random_safe(game):
                return False

    return game.status == GameStatus.WON


def _reveal_random_safe(game: MinesweeperGame) -> bool:
    """Reveal a random covered safe cell. Returns False if none exists."""
    covered = game.covered_cells
    safe = []
    for rr in range(game.height):
        for cc in range(game.width):
            if covered[rr, cc] and game.board[rr, cc] != -1:
                safe.append((rr, cc))
    if not safe:
        return False
    r, c = safe[np.random.default_rng().integers(0, len(safe))]
    game.make_move(r, c, MoveType.REVEAL)
    return True
