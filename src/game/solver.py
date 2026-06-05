"""Constraint-based solver for generating training data.

Adapted from gamescomputersplay/minesweeper-solver.
Simplified to pure constraint propagation — no brute force, no guessing.
Used to filter boards that are "no-guess" solvable.
"""

from typing import List, Set, Tuple

import numpy as np

from .game import MinesweeperGame
from .constants import CellState


# ── Shared constraint utilities (used by ConstraintSolver & ProbabilitySolver) ──

def build_constraints(game: MinesweeperGame) -> List[Tuple[Set[Tuple[int, int]], int]]:
    """Build constraint set from the current visible board."""
    constraints: List[Tuple[Set[Tuple[int, int]], int]] = []
    for r in range(game.height):
        for c in range(game.width):
            v = game.visible[r, c]
            if not (isinstance(v, (int, np.integer)) and 1 <= v <= 8):
                continue
            covered = set()
            flagged = 0
            for nr, nc in game._neighbors(r, c):
                sv = game.visible[nr, nc]
                if sv == CellState.COVERED:
                    covered.add((nr, nc))
                elif sv == CellState.FLAGGED:
                    flagged += 1
            remaining = int(v) - flagged
            if covered:
                constraints.append((covered, remaining))
    return constraints


def normalize_constraints(
    constraints: List[Tuple[Set[Tuple[int, int]], int]],
    safe: Set[Tuple[int, int]],
    mines: Set[Tuple[int, int]],
) -> List[Tuple[Set[Tuple[int, int]], int]]:
    """Remove deduced cells from constraints, adjusting remaining counts."""
    result = []
    for cells, remaining in constraints:
        known_mines = len(cells & mines)
        cells = cells - mines - safe
        remaining -= known_mines
        if cells and remaining >= 0:
            result.append((cells, remaining))
    return result


class ConstraintSolver:
    """Deterministic constraint propagation solver.

    Uses two strategies:
    1. Trivial: if remaining_mines == 0 → all safe; if remaining_mines == len(cells) → all mines.
    2. Subset: if cells_A ⊆ cells_B, then (cells_B - cells_A) has exactly (rem_B - rem_A) mines.

    These two are sufficient for most no-guess boards.
    """

    def __init__(self, game: MinesweeperGame):
        self.game = game
        self.height = game.height
        self.width = game.width

    def find_safe_and_mines(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Analyze current board state. Returns (safe_cells, mine_cells)."""
        safe: Set[Tuple[int, int]] = set()
        mines: Set[Tuple[int, int]] = set()

        constraints = build_constraints(self.game)
        if not constraints:
            return [], []

        # Iterate: trivial deductions → subset → trivial → ... until fixed point
        changed = True
        while changed:
            changed = False

            # Phase 1: Trivial deductions + adjust remaining for known cells
            constraints, ch = self._apply_trivial(constraints, safe, mines)
            changed |= ch

            if not constraints:
                break

            # Phase 2: Subset deductions
            constraints, safe, mines, ch = self._apply_subset(constraints, safe, mines)
            changed |= ch

        return sorted(safe), sorted(mines)

    # ─── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _apply_trivial(
        constraints: List[Tuple[Set[Tuple[int, int]], int]],
        safe: Set[Tuple[int, int]],
        mines: Set[Tuple[int, int]],
    ) -> Tuple[List[Tuple[Set[Tuple[int, int]], int]], bool]:
        """Apply trivial deductions: all-safe or all-mine constraints.

        Returns (remaining_constraints, changed).
        """
        changed = False

        # First normalize to remove already-known cells
        constraints = normalize_constraints(constraints, safe, mines)

        new_constraints = []
        for cells, remaining in constraints:
            if remaining == 0:
                # All covered neighbors are safe
                safe.update(cells)
                changed = True
            elif remaining == len(cells):
                # All covered neighbors are mines
                mines.update(cells)
                changed = True
            else:
                new_constraints.append((cells, remaining))

        if changed:
            # Re-normalize after deductions
            new_constraints = normalize_constraints(
                new_constraints, safe, mines
            )

        return new_constraints, changed

    @staticmethod
    def _apply_subset(
        constraints: List[Tuple[Set[Tuple[int, int]], int]],
        safe: Set[Tuple[int, int]],
        mines: Set[Tuple[int, int]],
    ) -> Tuple[List[Tuple[Set[Tuple[int, int]], int]], Set, Set, bool]:
        """Apply subset deduction.

        Returns (remaining_constraints, safe, mines, changed).
        """
        changed = False
        n = len(constraints)

        for i in range(n):
            cells_a, rem_a = constraints[i]
            if not cells_a:
                continue
            for j in range(n):
                if i == j:
                    continue
                cells_b, rem_b = constraints[j]
                if not cells_b:
                    continue

                if cells_a.issubset(cells_b) and cells_a != cells_b:
                    diff = cells_b - cells_a
                    diff_mines = rem_b - rem_a

                    if diff_mines == 0:
                        safe.update(diff)
                        changed = True
                    elif diff_mines == len(diff):
                        mines.update(diff)
                        changed = True

        if changed:
            constraints = normalize_constraints(constraints, safe, mines)

        return constraints, safe, mines, changed

    def _get_frontier_cells(self) -> List[Tuple[int, int]]:
        """Return revealed number cells (1-8) that have at least one covered neighbor."""
        frontier = []
        for r in range(self.height):
            for c in range(self.width):
                v = self.game.visible[r, c]
                if isinstance(v, (int, np.integer)) and 1 <= v <= 8:
                    for nr, nc in self.game._neighbors(r, c):
                        if self.game.visible[nr, nc] == CellState.COVERED:
                            frontier.append((r, c))
                            break
        return frontier
