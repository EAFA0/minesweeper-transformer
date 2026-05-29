"""Self-validated board generator — fast alternative to ms-toollib.

Uses our own ProbabilitySolver to verify board solvability.
Much faster than ms-toollib's SAT solver (~1-10s vs 10-50s per board).

Algorithm:
  1. Randomly place mines
  2. Random first click on a safe cell
  3. Use ProbabilitySolver to play through the board
  4. At each step, reveal any cell with P(mine) == 0
  5. If stuck (all covered cells have P > 0), board is too ambiguous → discard
  6. Success = solver completes the board (all safe cells revealed)

The returned game has its first click applied and is ready for model inference.
"""

import time
from typing import Optional

import numpy as np

from minesweeper.game import MinesweeperGame
from minesweeper.constants import CellState, MoveType, GameStatus
from minesweeper.probability_solver import ProbabilitySolver


def generate_self_validated_board(
    width: int = 8,
    height: int = 8,
    total_mines: int = 10,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 50,
    max_steps: int = 300,
    verbose: bool = False,
) -> Optional[MinesweeperGame]:
    """Generate a board that ProbabilitySolver can solve.

    Returns a game with first click already applied, or None if
    max_attempts exceeded.

    Args:
        width, height: board dimensions
        total_mines: number of mines
        rng: random generator (seeded or default)
        max_attempts: max board generation attempts
        max_steps: max solver steps before giving up
        verbose: print generation progress

    Returns:
        MinesweeperGame with first click applied, or None
    """
    if rng is None:
        rng = np.random.default_rng()

    for attempt in range(max_attempts):
        t0 = time.time()

        # Create game with random mines
        game = MinesweeperGame(width, height, total_mines)

        # Find a zero cell for first click (guaranteed to trigger flood fill)
        # A zero cell has no adjacent mines
        zero_cells = []
        for rr in range(height):
            for cc in range(width):
                if game.board[rr, cc] != -1:
                    adj = game._count_adjacent_mines(rr, cc)
                    if adj == 0:
                        zero_cells.append((rr, cc))

        if not zero_cells:
            if verbose:
                print(f"  Attempt {attempt+1}: no zero cell found, retrying")
            continue

        r, c = zero_cells[rng.integers(0, len(zero_cells))]
        game.make_move(r, c, MoveType.REVEAL)
        if game.status != GameStatus.PLAYING:
            if verbose:
                print(f"  Attempt {attempt+1}: first click hit mine, retrying")
            continue

        # Try to solve with ProbabilitySolver
        solved = _try_solve(game, max_steps=max_steps)

        dt = time.time() - t0
        if solved:
            if verbose:
                print(f"  Attempt {attempt+1}: ✓ solved in {dt:.1f}s")
            # Return fresh game with same mine layout
            return _recreate_game(width, height, game.get_mine_mask(), r, c)

        if verbose:
            print(f"  Attempt {attempt+1}: ✗ too hard ({dt:.1f}s), retrying")

    return None


def _try_solve(game: MinesweeperGame, max_steps: int = 300,
               warmup_clicks: int = 3) -> bool:
    """Try to solve the board using ProbabilitySolver.

    First does up to warmup_clicks random safe reveals to open the board,
    then switches to solver-driven selection.

    Returns True if the solver completes the board.
    """
    width, height = game.width, game.height

    # Warmup: reveal random safe cells to give the solver information
    for _ in range(warmup_clicks):
        if game.status != GameStatus.PLAYING:
            return game.status == GameStatus.WON
        covered = game.covered_cells
        if not covered.any():
            return True
        # Pick a random safe cell
        safe_indices = []
        for rr in range(height):
            for cc in range(width):
                if covered[rr, cc] and game.board[rr, cc] != -1:
                    safe_indices.append((rr, cc))
        if not safe_indices:
            return False
        r, c = safe_indices[np.random.default_rng().integers(0, len(safe_indices))]
        game.make_move(r, c, MoveType.REVEAL)

    # Solver-driven phase
    for _ in range(max_steps):
        if game.status != GameStatus.PLAYING:
            return game.status == GameStatus.WON

        solver = ProbabilitySolver(game)
        probs = solver.compute_probabilities()

        if probs is None:
            return False

        covered = game.covered_cells
        safe_mask = covered & (probs == 0.0)

        if not safe_mask.any():
            return False

        safe_indices = np.argwhere(safe_mask)
        r, c = safe_indices[0]
        game.make_move(r, c, MoveType.REVEAL)

        if game.status == GameStatus.LOST:
            return False

    return game.status == GameStatus.WON


def _recreate_game(
    width: int, height: int, mine_mask: np.ndarray,
    first_r: int, first_c: int,
) -> MinesweeperGame:
    """Recreate a game from mine mask, applying the same first click."""
    game = MinesweeperGame.__new__(MinesweeperGame)
    game.width = width
    game.height = height
    game.total_mines = int(mine_mask.sum())
    game.board = np.where(mine_mask, -1, 0).astype(np.int8)
    game.visible = np.full((height, width), CellState.COVERED, dtype=np.int8)
    game.status = GameStatus.PLAYING
    game.first_move = False
    game._mine_positions = np.argwhere(mine_mask)
    game._safe_covered = width * height - game.total_mines

    # Apply first click
    game._reveal(first_r, first_c)
    return game
