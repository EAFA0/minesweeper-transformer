"""No-guess board generation.

The project-level no-guess contract is stricter than "solvable by some
external solver": a generated board must be solvable by this repository's
ProbabilitySolver without ever selecting a non-zero mine-probability cell.
"""

from typing import Optional

import numpy as np

from game.constants import GameStatus, MoveType, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_MINES
from game.game import MinesweeperGame


NO_GUESS_EPS = 1e-6


def generate_no_guess_board(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    total_mines: int = DEFAULT_MINES,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 100,
) -> Optional[MinesweeperGame]:
    """Generate a game with a guaranteed no-guess board.

    Uses ms-toollib's laymine_solvable (SAT-based no-guess generator).
    Falls back to constraint-solver filtering when ms-toollib fails.

    Returns a MinesweeperGame with first move already made at a safe position,
    or None if generation fails after max_attempts.
    """
    if rng is None:
        rng = np.random.default_rng()

    try:
        import ms_toollib as mt
        _has_mstoolib = True
    except ImportError:
        _has_mstoolib = False
        print("⚠ ms-toollib not installed. Install: pip install ms-toollib")
        print("  Falling back to constraint-solver filtering...")
        return _generate_fallback(width, height, total_mines, rng, max_attempts)

    for attempt in range(max_attempts):
        # Random safe first-click position
        r = rng.integers(0, height)
        c = rng.integers(0, width)

        # Note: ms-toollib uses (row, col) = (x0, y0)
        # The function signature is laymine_solvable(row, column, mine_num, x0, y0)
        try:
            board_2d, success = mt.laymine_solvable(
                height, width, total_mines, r, c, max_times=10000
            )
        except Exception:
            # ms-toollib may fail at extreme densities
            success = False

        if not success:
            continue

        # Convert ms-toollib board to mine mask
        # ms-toollib: -1=mine, 0-8=number (pre-computed)
        # Our game:  -1=mine, 0=safe
        board_np = np.array(board_2d, dtype=np.int8)
        mine_mask = np.where(board_np == -1, -1, 0).astype(np.int8)

        # Create game and inject mine layout
        game = MinesweeperGame(width, height, total_mines)
        game.board = mine_mask
        game.first_move = False  # bypass random mine generation

        # Make first click — guaranteed safe by ms-toollib
        game.make_move(r, c, MoveType.REVEAL)

        if is_solver_no_guess(game):
            return game

    # All attempts exhausted
    if max_attempts > 0:
        print(f"⚠ ms-toollib failed after {max_attempts} attempts "
              f"({width}×{height}, {total_mines} mines)")
        print("  Falling back to constraint-solver filtering...")
        return _generate_fallback(width, height, total_mines, rng, max_attempts)

    return None


def _generate_fallback(
    width: int, height: int, total_mines: int,
    rng: np.random.Generator, max_attempts: int,
) -> Optional[MinesweeperGame]:
    """Fallback: use constraint-solver filtering (our existing approach).

    Only returns boards that our solver can fully solve.
    Much lower success rate than ms-toollib above ~15% mine density.
    """
    from game.solver import ConstraintSolver
    from game.constants import CellState, GameStatus

    for _ in range(max_attempts * 3):  # more attempts for low-success-rate fallback
        game = MinesweeperGame(width, height, total_mines)
        r = rng.integers(1, height - 1)
        c = rng.integers(1, width - 1)
        game.make_move(r, c, MoveType.REVEAL)

        if game.status != GameStatus.PLAYING:
            continue

        # Try to fully solve with constraint propagation
        solver = ConstraintSolver(game)
        step = 0
        while game.status == GameStatus.PLAYING and step < 200:
            safe, mines = solver.find_safe_and_mines()
            if not safe and not mines:
                break
            for sr, sc in safe:
                game.make_move(sr, sc, MoveType.REVEAL)
            for mr, mc in mines:
                if game.visible[mr, mc] == CellState.COVERED:
                    game.make_move(mr, mc, MoveType.FLAG)
            step += 1

        if game.status == GameStatus.WON:
            return game

    return None


def is_solver_no_guess(game: MinesweeperGame, max_steps: int = 300) -> bool:
    """Return True if ProbabilitySolver can finish the board without guessing."""
    from game.probability_solver import ProbabilitySolver

    probe = MinesweeperGame.from_mine_mask(
        game.width,
        game.height,
        game.get_mine_mask(),
        first_done=True,
        visible=game.visible.copy(),
    )

    steps = 0
    while probe.status == GameStatus.PLAYING and steps < max_steps:
        probs = ProbabilitySolver(probe).compute_probabilities()
        covered = probe.covered_cells
        if not covered.any():
            return True

        safe_mask = covered & (probs <= NO_GUESS_EPS)
        if not safe_mask.any():
            return False

        for r, c in zip(*np.where(safe_mask)):
            if probe.status != GameStatus.PLAYING:
                break
            probe.make_move(int(r), int(c), MoveType.REVEAL)
        steps += 1

    return probe.status == GameStatus.WON
