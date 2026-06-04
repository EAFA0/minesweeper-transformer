"""Exact probability solver via enumeration of consistent mine configurations.

Given a MinesweeperGame state, computes P(mine) for each covered cell by:
1. Running constraint propagation to find deduced cells (P=0 or P=1)
2. For remaining ambiguous cells, finding connected components
3. Enumerating all consistent mine assignments within each component (≤20 cells)
4. For larger components or isolated cells, using uniform probability fallback
"""

from typing import Dict, List, Set, Tuple

import numpy as np

from .constants import CellState
from .game import MinesweeperGame
from .solver import ConstraintSolver

# Maximum component size for exact enumeration (2^20 = 1M assignments max)
MAX_ENUM_CELLS = 20


class ProbabilitySolver:
    """Exact marginal probability computer for minesweeper states.

    Computes P(mine) for every covered cell. Uses exact enumeration
    for small components and uniform fallback for large ones.
    """

    def __init__(self, game: MinesweeperGame):
        self.game = game
        self.height = game.height
        self.width = game.width

    def compute_probabilities(self) -> np.ndarray:
        """Return (H, W) float32 array: P(mine) for each cell.

        Already-revealed cells get 0.0 (ignored in training).
        Deduced safe cells get 0.0, deduced mines get 1.0.
        Ambiguous cells get exact marginal probability ∈ [0, 1].
        """
        probs = np.full((self.height, self.width), -1.0, dtype=np.float32)

        # ── Step 1: Constraint propagation ────────────────────────────
        cs = ConstraintSolver(self.game)
        safe_cells, mine_cells = cs.find_safe_and_mines()
        safe_set: Set[Tuple[int, int]] = set(safe_cells)  # type: ignore[arg-type]
        mine_set: Set[Tuple[int, int]] = set(mine_cells)  # type: ignore[arg-type]

        for r, c in mine_set:
            probs[r, c] = 1.0
        for r, c in safe_set:
            probs[r, c] = 0.0

        # Revealed cells → 0.0 (not relevant for training)
        revealed_mask = ~self.game.covered_cells
        probs[revealed_mask] = 0.0

        # ── Step 2: Build constraints ─────────────────────────────────
        constraints = self._build_constraints()
        if not constraints:
            # No constraints: all covered cells are isolated
            self._fill_isolated_uniform(probs)
            return probs

        constraints = self._normalize_constraints(constraints, safe_set, mine_set)
        if not constraints:
            # All resolved by deduction — remaining cells are isolated
            self._fill_isolated_uniform(probs)
            return probs

        # ── Step 3: Collect ambiguous cells ───────────────────────────
        all_ambiguous: Set[Tuple[int, int]] = set()
        for cells, _ in constraints:
            all_ambiguous.update(cells)

        # ── Step 4: Find connected components ─────────────────────────
        components = self._find_components(all_ambiguous, constraints)

        # ── Step 5: Enumerate or fallback per component ───────────────
        for comp_cells, comp_constraints in components:
            if len(comp_cells) <= MAX_ENUM_CELLS:
                comp_probs = self._enumerate_exact(comp_cells, comp_constraints)
            else:
                comp_probs = self._uniform_fallback(comp_cells)
            for (r, c), p in comp_probs.items():
                probs[r, c] = p

        # Fill any remaining covered cells with uniform probability
        self._fill_isolated_uniform(probs)

        return probs

    # ─── Isolated / Uniform ───────────────────────────────────────────────

    def _fill_isolated_uniform(self, probs: np.ndarray) -> None:
        """Assign uniform P(mine) to covered cells not yet assigned (probs == -1)."""
        remaining = self.game.mine_count
        covered_mask = self.game.covered_cells
        # Count already-assigned mines in covered cells
        assigned_mines = np.sum(probs[covered_mask] == 1.0)
        # Unassigned covered cells (still at -1)
        unassigned_mask = covered_mask & (probs == -1.0)
        n_unassigned = int(np.sum(unassigned_mask))
        if n_unassigned > 0:
            remaining_unassigned = max(0, remaining - assigned_mines)
            p = min(1.0, remaining_unassigned / n_unassigned)
            probs[unassigned_mask] = p
        # Any remaining -1 cells (shouldn't exist) → 0.5
        still_unassigned = probs == -1.0
        probs[still_unassigned] = 0.5

    @staticmethod
    def _uniform_fallback(cells: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """Fallback: assign 0.5 to all cells in a large component."""
        return {cell: 0.5 for cell in cells}

    # ─── Constraint Building ──────────────────────────────────────────────

    def _build_constraints(self) -> List[Tuple[Set[Tuple[int, int]], int]]:
        """Build constraints from revealed number cells."""
        constraints: List[Tuple[Set[Tuple[int, int]], int]] = []
        for r in range(self.height):
            for c in range(self.width):
                v = self.game.visible[r, c]
                if not (isinstance(v, (int, np.integer)) and 1 <= v <= 8):
                    continue
                covered = set()
                flagged = 0
                for nr, nc in self.game._neighbors(r, c):
                    sv = self.game.visible[nr, nc]
                    if sv == CellState.COVERED:
                        covered.add((nr, nc))
                    elif sv == CellState.FLAGGED:
                        flagged += 1
                remaining = int(v) - flagged
                if covered:
                    constraints.append((covered, remaining))
        return constraints

    @staticmethod
    def _normalize_constraints(
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

    # ─── Component Detection ──────────────────────────────────────────────

    def _find_components(
        self,
        cells: Set[Tuple[int, int]],
        constraints: List[Tuple[Set[Tuple[int, int]], int]],
    ) -> List[Tuple[List[Tuple[int, int]], List[Tuple[Set[Tuple[int, int]], int]]]]:
        """Group ambiguous cells into independent connected components.

        Two cells are connected if they appear in the same constraint.
        """
        adj: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {cell: set() for cell in cells}
        for constraint_cells, _ in constraints:
            cell_list = list(constraint_cells)
            for i in range(len(cell_list)):
                for j in range(i + 1, len(cell_list)):
                    a, b = cell_list[i], cell_list[j]
                    if a in adj and b in adj:
                        adj[a].add(b)
                        adj[b].add(a)

        visited: Set[Tuple[int, int]] = set()
        components = []

        for cell in cells:
            if cell in visited:
                continue
            comp_cells: List[Tuple[int, int]] = []
            queue = [cell]
            visited.add(cell)
            while queue:
                current = queue.pop(0)
                comp_cells.append(current)
                for neighbor in adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            comp_set = set(comp_cells)
            comp_constraints = [
                (cs & comp_set, rm)
                for cs, rm in constraints
                if cs & comp_set
            ]
            components.append((comp_cells, comp_constraints))

        return components

    # ─── Exact Enumeration ────────────────────────────────────────────────

    def _enumerate_exact(
        self,
        cells: List[Tuple[int, int]],
        constraints: List[Tuple[Set[Tuple[int, int]], int]],
    ) -> Dict[Tuple[int, int], float]:
        """Enumerate all consistent mine assignments for a component (≤20 cells).

        Uses backtracking with constraint-based pruning.
        """
        n = len(cells)
        assert n <= MAX_ENUM_CELLS, f"Component too large: {n} > {MAX_ENUM_CELLS}"

        # 1. Setup indices and optimize ordering
        idx_constraints, order = self._prepare_enumeration(cells, constraints)
        if not idx_constraints:
            return {cell: 0.5 for cell in cells}

        # 2. Run backtracking search
        total_assignments, mine_counts = self._run_backtracking(n, idx_constraints)

        if total_assignments == 0:
            return {cell: 0.5 for cell in cells}

        # 3. Map results back to original cells
        result = {}
        for new_pos in range(n):
            orig_pos = order[new_pos]
            cell = cells[orig_pos]
            result[cell] = mine_counts[new_pos] / total_assignments

        return result

    def _prepare_enumeration(
        self, cells: List[Tuple[int, int]], constraints: List[Tuple[Set[Tuple[int, int]], int]]
    ) -> Tuple[List[Tuple[Set[int], int]], List[int]]:
        """Convert constraints to index-based, sort cells by degree for efficient pruning."""
        n = len(cells)
        cell_to_idx = {cell: i for i, cell in enumerate(cells)}

        idx_constraints: List[Tuple[Set[int], int]] = []
        for cs, remaining in constraints:
            indices = set(cell_to_idx[c] for c in cs)
            if indices:
                idx_constraints.append((indices, remaining))

        if not idx_constraints:
            return [], []

        degree = [0] * n
        for indices, _ in idx_constraints:
            for i in indices:
                degree[i] += 1
        order = sorted(range(n), key=lambda i: -degree[i])

        old_to_new = {old: new for new, old in enumerate(order)}
        ordered_constraints = []
        for indices, target in idx_constraints:
            new_indices = set(old_to_new[i] for i in indices)
            ordered_constraints.append((new_indices, target))

        return ordered_constraints, order

    def _run_backtracking(
        self, n: int, ordered_constraints: List[Tuple[Set[int], int]]
    ) -> Tuple[int, List[float]]:
        """Execute backtracking to count valid mine configurations."""
        mine_counts = [0.0] * n
        total = [0]  # mutable counter
        assignment = [-1] * n  # -1=unassigned, 0=safe, 1=mine

        def is_viable() -> bool:
            for indices, target in ordered_constraints:
                assigned = sum(1 for i in indices if assignment[i] == 1)
                unassigned = sum(1 for i in indices if assignment[i] == -1)
                if assigned > target or assigned + unassigned < target:
                    return False
            return True

        def backtrack(pos: int):
            if total[0] >= 500_000:
                return  # cap at 500K valid assignments
            
            if pos == n:
                for indices, target in ordered_constraints:
                    if sum(assignment[i] for i in indices) != target:
                        return
                total[0] += 1
                for i in range(n):
                    mine_counts[i] += assignment[i]
                return

            for val in (0, 1):
                assignment[pos] = val
                if is_viable():
                    backtrack(pos + 1)
            assignment[pos] = -1

        backtrack(0)
        return total[0], mine_counts
