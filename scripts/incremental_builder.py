"""Incremental no-guess board builder.

Strategy: "Complete a constraint" — for each step, pick a revealed numbered cell,
assign mines to its covered neighbors to make remaining=0, then ConstraintSolver
can deduce the remaining neighbors are safe.

Key invariant: after each step, ConstraintSolver can find at least one safe cell.
We reveal ONE safe cell at a time and immediately "complete" any numbered cell
by placing mines, keeping the frontier alive.

Usage:
    python scripts/incremental_builder.py --width 8 --height 8 --mines 10 --n_boards 20
"""

import argparse
import sys
import time
from typing import List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, sys.path[0])

from game.constants import CellState, GameStatus
from game.game import MinesweeperGame
from game.solver import ConstraintSolver

_MINE = -1


def _place_mine(game: MinesweeperGame, r: int, c: int) -> None:
    """Place a mine and flag it so ConstraintSolver can see it."""
    game.board[r, c] = _MINE
    game.visible[r, c] = CellState.FLAGGED
    _update_visible(game, r, c)


def build_incremental(
    width: int,
    height: int,
    total_mines: int,
    rng: np.random.Generator,
    max_attempts: int = 500,
) -> Optional[MinesweeperGame]:
    """Build a no-guess board incrementally."""
    from data.no_guess import is_solver_no_guess

    for _ in range(max_attempts):
        result = _try_build(width, height, total_mines, rng)
        if result is not None and is_solver_no_guess(result):
            return result
    return None


def _try_build(
    width: int, height: int, total_mines: int, rng: np.random.Generator
) -> Optional[MinesweeperGame]:
    """Single incremental build attempt."""
    game = MinesweeperGame(width, height, total_mines)
    game.board = np.zeros((height, width), dtype=np.int8)
    game.first_move = False

    # Step 1: Pick first click location, place bootstrap mines around it
    r = rng.integers(0, height)
    c = rng.integers(0, width)

    assigned_mines: Set[Tuple[int, int]] = set()

    neighbors = list(game._neighbors(r, c))
    n_bootstrap = min(rng.integers(1, 4), len(neighbors), total_mines)
    if n_bootstrap > 0:
        bootstrap_idxs = rng.choice(len(neighbors), n_bootstrap, replace=False)
        for idx in bootstrap_idxs:
            pos = neighbors[idx]
            assigned_mines.add((int(pos[0]), int(pos[1])))
            _place_mine(game, pos[0], pos[1])

    # Step 2: First click (with flood-fill, but mines already placed)
    _reveal_cell(game, r, c)

    # Step 3: Main loop — reveal ONE safe cell at a time.
    # When a numbered cell is revealed, immediately "complete" its constraint
    # by placing mines, keeping the frontier alive for the next iteration.
    for _ in range(500):
        if game.status != GameStatus.PLAYING:
            break

        # Find safe cells via constraint propagation
        solver = ConstraintSolver(game)
        safe, deduced = solver.find_safe_and_mines()

        # Place any deduced mines
        for mr, mc in deduced:
            if (int(mr), int(mc)) not in assigned_mines:
                assigned_mines.add((int(mr), int(mc)))
                _place_mine(game, mr, mc)

        # Filter safe cells that are still covered
        safe_covered = [
            (sr, sc) for sr, sc in safe
            if game.visible[sr, sc] == CellState.COVERED
        ]

        if safe_covered:
            # Reveal ONE safe cell
            sr, sc = safe_covered[0]
            _reveal_cell(game, sr, sc)

            # If the revealed cell is a number (1-8), complete its constraint
            v = game.visible[sr, sc]
            if isinstance(v, (int, np.integer)) and 1 <= v <= 8:
                covered = []
                for nr, nc in game._neighbors(sr, sc):
                    if game.visible[nr, nc] == CellState.COVERED:
                        if (nr, nc) not in assigned_mines:
                            covered.append((nr, nc))

                remaining = int(v)
                if covered and remaining > 0 and remaining <= len(covered):
                    need = min(remaining, total_mines - len(assigned_mines))
                    if need > 0:
                        idxs = rng.choice(len(covered), need, replace=False)
                        for idx in idxs:
                            pos = covered[idx]
                            assigned_mines.add((int(pos[0]), int(pos[1])))
                            _place_mine(game, pos[0], pos[1])
            continue

        # No safe cells from constraint propagation
        if len(assigned_mines) >= total_mines:
            if _finish_reveal(game, assigned_mines):
                break
            return None

        # Fallback: complete a constraint to create safe cells
        if not _complete_constraint(game, assigned_mines, total_mines, rng):
            return None

    # Fill remaining mines in isolated cells
    remaining = total_mines - len(assigned_mines)
    if remaining > 0:
        isolated = _get_isolated(game, assigned_mines)
        if len(isolated) < remaining:
            return None
        idxs = rng.choice(len(isolated), remaining, replace=False)
        for idx in idxs:
            pos = isolated[idx]
            assigned_mines.add((int(pos[0]), int(pos[1])))
            _place_mine(game, pos[0], pos[1])

    _finalize(game, total_mines)
    return game


# ── Constraint Completion ───────────────────────────────────────────────────

def _complete_constraint(
    game: MinesweeperGame,
    assigned_mines: Set[Tuple[int, int]],
    total_mines: int,
    rng: np.random.Generator,
    max_trials: int = 500,
) -> bool:
    """Complete a single constraint to create provably safe cells."""
    need_mines = total_mines - len(assigned_mines)
    if need_mines <= 0:
        return False

    candidates = _get_numbered_with_covered(game, assigned_mines)
    if not candidates:
        return False

    saved_board = game.board.copy()
    saved_visible = game.visible.copy()

    for _ in range(max_trials):
        game.board = saved_board.copy()
        game.visible = saved_visible.copy()

        _, _, covered_neighbors, remaining = candidates[
            rng.integers(0, len(candidates))
        ]

        if remaining <= 0 or remaining > len(covered_neighbors):
            continue
        if remaining > need_mines:
            continue

        idxs = rng.choice(len(covered_neighbors), remaining, replace=False)
        for idx in idxs:
            pos = covered_neighbors[idx]
            _place_mine(game, pos[0], pos[1])

        solver = ConstraintSolver(game)
        safe, _ = solver.find_safe_and_mines()
        safe_covered = [
            (sr, sc) for sr, sc in safe
            if game.visible[sr, sc] == CellState.COVERED
        ]

        if safe_covered:
            for idx in idxs:
                pos = covered_neighbors[idx]
                assigned_mines.add((int(pos[0]), int(pos[1])))
            for sr, sc in safe_covered:
                if game.status != GameStatus.PLAYING:
                    break
                _reveal_cell(game, sr, sc)
            return True

    game.board = saved_board
    game.visible = saved_visible
    return False


def _get_numbered_with_covered(
    game: MinesweeperGame, assigned_mines: Set[Tuple[int, int]]
) -> List[Tuple[int, int, List[Tuple[int, int]], int]]:
    """Return list of (r, c, covered_neighbors, remaining_mines) for numbered cells."""
    result = []
    for r in range(game.height):
        for c in range(game.width):
            v = game.visible[r, c]
            if not (isinstance(v, (int, np.integer)) and 1 <= v <= 8):
                continue

            covered = []
            placed = 0
            for nr, nc in game._neighbors(r, c):
                if game.visible[nr, nc] == CellState.COVERED:
                    if (nr, nc) not in assigned_mines:
                        covered.append((nr, nc))
                elif game.visible[nr, nc] == CellState.FLAGGED:
                    placed += 1

            remaining = int(v) - placed
            if covered and remaining >= 0:
                result.append((r, c, covered, remaining))

    return result


# ── Core Operations ─────────────────────────────────────────────────────────

def _reveal_cell(game: MinesweeperGame, r: int, c: int) -> int:
    """Reveal a cell. Returns number of cells revealed (1 for numbers, N for 0-flood-fill).
    
    Flood-fills on 0s because 0-neighbors are trivially safe (no mines nearby).
    Single reveal for numbered cells to keep the frontier alive.
    """
    if game.visible[r, c] != CellState.COVERED:
        return 0
    if game.board[r, c] == _MINE:
        game.visible[r, c] = CellState.EXPLODED
        game.status = GameStatus.LOST
        return 0

    count = 0
    for nr, nc in game._neighbors(r, c):
        if game.board[nr, nc] == _MINE:
            count += 1
    game.visible[r, c] = count
    game._safe_covered -= 1
    revealed = 1

    if count == 0:
        # Flood-fill: all 0-neighbors are trivially safe
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            for nr, nc in game._neighbors(cr, cc):
                if game.visible[nr, nc] != CellState.COVERED:
                    continue
                ncount = 0
                for nnr, nnc in game._neighbors(nr, nc):
                    if game.board[nnr, nnc] == _MINE:
                        ncount += 1
                game.visible[nr, nc] = ncount
                game._safe_covered -= 1
                revealed += 1
                if ncount == 0:
                    stack.append((nr, nc))

    game._check_win()
    return revealed


def _update_visible(game: MinesweeperGame, mr: int, mc: int) -> None:
    """Update visible numbers on all revealed neighbors after placing a mine."""
    for nr, nc in game._neighbors(mr, mc):
        if game.visible[nr, nc] >= 0:
            count = 0
            for nnr, nnc in game._neighbors(nr, nc):
                if game.board[nnr, nnc] == _MINE:
                    count += 1
            game.visible[nr, nc] = count


def _finalize(game: MinesweeperGame, total_mines: int) -> None:
    """Sync all game state after construction."""
    game._mine_positions = np.argwhere(game.board == _MINE)
    revealed_count = int((game.visible >= 0).sum())
    game._safe_covered = game.width * game.height - total_mines - revealed_count
    game._check_win()


def _finish_reveal(
    game: MinesweeperGame,
    assigned_mines: Set[Tuple[int, int]],
) -> bool:
    """After all mines placed, reveal remaining safe cells via constraints."""
    for _ in range(500):
        if game.status != GameStatus.PLAYING:
            break

        solver = ConstraintSolver(game)
        safe, _ = solver.find_safe_and_mines()

        safe_covered = [
            (sr, sc) for sr, sc in safe
            if game.visible[sr, sc] == CellState.COVERED
        ]
        if not safe_covered:
            covered = set((int(r), int(c)) for r, c in zip(*np.where(game.covered_cells)))
            if covered == assigned_mines:
                return True
            return False

        for sr, sc in safe_covered:
            if game.status != GameStatus.PLAYING:
                break
            _reveal_cell(game, sr, sc)

    return game.status == GameStatus.WON


# ── Board Helpers ───────────────────────────────────────────────────────────

def _get_isolated(
    game: MinesweeperGame, assigned_mines: Set[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Return covered cells NOT adjacent to any revealed cell and not assigned."""
    frontier: Set[Tuple[int, int]] = set()
    for r in range(game.height):
        for c in range(game.width):
            if game.visible[r, c] >= 0:
                for nr, nc in game._neighbors(r, c):
                    if game.visible[nr, nc] == CellState.COVERED:
                        if (nr, nc) not in assigned_mines:
                            frontier.add((nr, nc))

    isolated: List[Tuple[int, int]] = []
    for r in range(game.height):
        for c in range(game.width):
            if game.visible[r, c] == CellState.COVERED:
                if (r, c) not in assigned_mines and (r, c) not in frontier:
                    isolated.append((r, c))
    return isolated


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Incremental no-guess board builder"
    )
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--n_boards", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Building {args.n_boards} boards ({args.width}x{args.height}, "
          f"{args.mines} mines)...", file=sys.stderr)

    t_start = time.perf_counter()
    success = 0
    times: List[float] = []

    for i in range(args.n_boards):
        t0 = time.perf_counter()
        game = build_incremental(args.width, args.height, args.mines, rng)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if game is not None:
            success += 1
            print(f"  [{i+1}/{args.n_boards}] OK  {elapsed:.4f}s", file=sys.stderr)
        else:
            print(f"  [{i+1}/{args.n_boards}] FAIL  {elapsed:.4f}s", file=sys.stderr)

    total = time.perf_counter() - t_start

    print()
    print(f"Results: {success}/{args.n_boards} boards generated "
          f"({success/args.n_boards*100:.1f}%)")
    print(f"Total time: {total:.2f}s")
    if times:
        sorted_times = sorted(times)
        n = len(sorted_times)
        print(f"Time per board: mean={np.mean(times):.4f}s  "
              f"median={np.median(times):.4f}s  "
              f"p95={sorted_times[int(n*0.95)]:.4f}s")


if __name__ == "__main__":
    main()
