"""Compare board distributions between incremental builder and ms-toollib.

Phase 4 of plan-fast-noguess-builder.md — distribution consistency check.

Usage:
    python scripts/compare_board_distribution.py --width 8 --height 8 --mines 10 --n_boards 200
"""

import argparse
import sys
import time
from collections import Counter
from typing import List, Set, Tuple

import numpy as np

sys.path.insert(0, sys.path[0])

from game.constants import GameStatus, MoveType
from game.game import MinesweeperGame
from game.probability_solver import ProbabilitySolver

_MINE = -1
NO_GUESS_EPS = 1e-6


# ── Metrics ─────────────────────────────────────────────────────────────────

def _normalize_to_first_move(game: MinesweeperGame) -> MinesweeperGame:
    """Reset board to first-move-only state (all covered except first click)."""
    # Find revealed cells
    revealed = np.where(game.visible >= 0)
    if len(revealed[0]) == 0:
        return game

    # Create fresh game with same mine layout
    fresh = MinesweeperGame(game.width, game.height, game.total_mines)
    fresh.board = game.board.copy()
    fresh.first_move = False

    # Make first click at the first revealed cell
    r, c = int(revealed[0][0]), int(revealed[1][0])
    fresh.make_move(r, c, MoveType.REVEAL)
    return fresh


def mine_clusters(game: MinesweeperGame) -> List[int]:
    """Return sizes of connected mine components (8-directional)."""
    mine_mask = game.board == _MINE
    visited = np.zeros_like(mine_mask, dtype=bool)
    clusters = []

    for r in range(game.height):
        for c in range(game.width):
            if mine_mask[r, c] and not visited[r, c]:
                size = 0
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    size += 1
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < game.height and 0 <= nc < game.width:
                                if mine_mask[nr, nc] and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    stack.append((nr, nc))
                clusters.append(size)
    return clusters


def frontier_complexity(game: MinesweeperGame) -> Tuple[float, int]:
    """Return (avg_constraints, max_connected_component_size) for frontier."""
    from game.solver import build_constraints

    constraints = build_constraints(game)
    n_constraints = len(constraints)

    # Build frontier cell graph (connected via shared constraints)
    frontier_cells: Set[Tuple[int, int]] = set()
    for cells, _ in constraints:
        frontier_cells.update(cells)

    if not frontier_cells:
        return 0.0, 0

    # Union-find on frontier cells
    parent = {cell: cell for cell in frontier_cells}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for cells, _ in constraints:
        cell_list = list(cells)
        for i in range(1, len(cell_list)):
            union(cell_list[0], cell_list[i])

    comp_sizes = Counter(find(c) for c in frontier_cells)
    max_comp = max(comp_sizes.values()) if comp_sizes else 0

    return float(n_constraints), max_comp


def solve_steps(game: MinesweeperGame, max_steps: int = 500) -> int:
    """Return number of steps ProbabilitySolver needs to solve the board."""
    probe = MinesweeperGame.from_mine_mask(
        game.width, game.height, game.get_mine_mask(),
        first_done=True, visible=game.visible.copy(),
    )

    steps = 0
    while probe.status == GameStatus.PLAYING and steps < max_steps:
        probs = ProbabilitySolver(probe).compute_probabilities()
        covered = probe.covered_cells
        if not covered.any():
            break

        safe_mask = covered & (probs <= NO_GUESS_EPS)
        if not safe_mask.any():
            break

        for r, c in zip(*np.where(safe_mask)):
            if probe.status != GameStatus.PLAYING:
                break
            probe.make_move(int(r), int(c), MoveType.REVEAL)
        steps += 1

    return steps


def ambiguous_cells(game: MinesweeperGame, max_steps: int = 500) -> List[int]:
    """Return per-step count of cells with P(mine) > 0 (ambiguous)."""
    probe = MinesweeperGame.from_mine_mask(
        game.width, game.height, game.get_mine_mask(),
        first_done=True, visible=game.visible.copy(),
    )

    counts = []
    steps = 0
    while probe.status == GameStatus.PLAYING and steps < max_steps:
        probs = ProbabilitySolver(probe).compute_probabilities()
        covered = probe.covered_cells
        if not covered.any():
            break

        safe_mask = covered & (probs <= NO_GUESS_EPS)
        if not safe_mask.any():
            break

        # Count ambiguous cells (covered, not safe)
        ambiguous = int((covered & (probs > NO_GUESS_EPS)).sum())
        counts.append(ambiguous)

        for r, c in zip(*np.where(safe_mask)):
            if probe.status != GameStatus.PLAYING:
                break
            probe.make_move(int(r), int(c), MoveType.REVEAL)
        steps += 1

    return counts


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare board distributions: incremental vs ms-toollib"
    )
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--n_boards", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Generate boards from both methods ──

    print(f"Generating {args.n_boards} boards ({args.width}x{args.height}, "
          f"{args.mines} mines) from each method...", file=sys.stderr)

    # Method A: Incremental builder
    from incremental_builder import build_incremental

    inc_boards: List[MinesweeperGame] = []
    t0 = time.perf_counter()
    for i in range(args.n_boards):
        g = build_incremental(args.width, args.height, args.mines, rng)
        if g is not None:
            inc_boards.append(g)
        if (i + 1) % 50 == 0:
            print(f"  Incremental: {i+1}/{args.n_boards} ({len(inc_boards)} ok)", file=sys.stderr)
    t_inc = time.perf_counter() - t0
    print(f"  Incremental: {len(inc_boards)}/{args.n_boards} boards in {t_inc:.1f}s", file=sys.stderr)

    # Method B: ms-toollib
    from data.no_guess import generate_no_guess_board

    ms_boards: List[MinesweeperGame] = []
    t0 = time.perf_counter()
    for i in range(args.n_boards):
        g = generate_no_guess_board(args.width, args.height, args.mines, rng, max_attempts=200)
        if g is not None:
            ms_boards.append(g)
        if (i + 1) % 50 == 0:
            print(f"  ms-toollib: {i+1}/{args.n_boards} ({len(ms_boards)} ok)", file=sys.stderr)
    t_ms = time.perf_counter() - t0
    print(f"  ms-toollib: {len(ms_boards)}/{args.n_boards} boards in {t_ms:.1f}s", file=sys.stderr)

    # ── Compute metrics ──

    print("\nComputing metrics...", file=sys.stderr)

    # Normalize all boards to first-move-only state
    inc_normalized = [_normalize_to_first_move(g) for g in inc_boards]
    ms_normalized = [_normalize_to_first_move(g) for g in ms_boards]

    # Mine clusters (board-level, independent of state)
    inc_clusters = []
    for g in inc_boards:
        inc_clusters.extend(mine_clusters(g))
    ms_clusters = []
    for g in ms_boards:
        ms_clusters.extend(mine_clusters(g))

    # Frontier complexity (computed on first-move state)
    inc_frontier = [frontier_complexity(g) for g in inc_normalized]
    ms_frontier = [frontier_complexity(g) for g in ms_normalized]

    # Solve steps (from first-move state)
    inc_steps = [solve_steps(g) for g in inc_normalized]
    ms_steps = [solve_steps(g) for g in ms_normalized]

    # Ambiguous cells (from first-move state)
    inc_ambig = []
    for g in inc_normalized:
        inc_ambig.extend(ambiguous_cells(g))
    ms_ambig = []
    for g in ms_normalized:
        ms_ambig.extend(ambiguous_cells(g))

    # ── Output ──

    def stats(arr, label=""):
        if not arr:
            return f"  {label}: N/A"
        a = np.array(arr)
        return (f"  {label}: mean={a.mean():.2f}  median={np.median(a):.1f}  "
                f"std={a.std():.2f}  min={a.min()}  max={a.max()}")

    print()
    print("=" * 70)
    print(f"Distribution Comparison: {args.width}x{args.height}, {args.mines} mines")
    print(f"  Incremental: {len(inc_boards)} boards  |  ms-toollib: {len(ms_boards)} boards")
    print("=" * 70)

    print("\n## Mine Cluster Sizes")
    print(stats(inc_clusters, "Incremental"))
    print(stats(ms_clusters, "ms-toollib "))

    print("\n## Frontier Complexity (avg constraints per state)")
    inc_constraints = [f[0] for f in inc_frontier]
    ms_constraints = [f[0] for f in ms_frontier]
    print(stats(inc_constraints, "Incremental"))
    print(stats(ms_constraints, "ms-toollib "))

    print("\n## Max Connected Frontier Component Size")
    inc_maxcomp = [f[1] for f in inc_frontier]
    ms_maxcomp = [f[1] for f in ms_frontier]
    print(stats(inc_maxcomp, "Incremental"))
    print(stats(ms_maxcomp, "ms-toollib "))

    print("\n## Solve Steps (ProbabilitySolver)")
    print(stats(inc_steps, "Incremental"))
    print(stats(ms_steps, "ms-toollib "))

    print("\n## Ambiguous Cells per Step")
    print(stats(inc_ambig, "Incremental"))
    print(stats(ms_ambig, "ms-toollib "))

    # ── Summary assessment ──
    print("\n## Assessment")
    issues = []

    if inc_clusters and ms_clusters:
        inc_mean = np.mean(inc_clusters)
        ms_mean = np.mean(ms_clusters)
        ratio = inc_mean / ms_mean if ms_mean > 0 else float('inf')
        if ratio < 0.5 or ratio > 2.0:
            issues.append(f"Mine cluster mean differs by {ratio:.1f}x (inc={inc_mean:.2f}, ms={ms_mean:.2f})")

    if inc_steps and ms_steps:
        inc_mean = np.mean(inc_steps)
        ms_mean = np.mean(ms_steps)
        ratio = inc_mean / ms_mean if ms_mean > 0 else float('inf')
        if ratio < 0.5 or ratio > 2.0:
            issues.append(f"Solve steps mean differs by {ratio:.1f}x (inc={inc_mean:.2f}, ms={ms_mean:.2f})")

    if inc_ambig and ms_ambig:
        inc_mean = np.mean(inc_ambig)
        ms_mean = np.mean(ms_ambig)
        ratio = inc_mean / ms_mean if ms_mean > 0 else float('inf')
        if ratio < 0.5 or ratio > 2.0:
            issues.append(f"Ambiguous cells mean differs by {ratio:.1f}x (inc={inc_mean:.2f}, ms={ms_mean:.2f})")

    if issues:
        print("⚠ Potential distribution shift detected:")
        for issue in issues:
            print(f"  - {issue}")
        print("  Evaluate impact on benchmark before switching generators.")
    else:
        print("✓ No systematic distribution shift detected.")


if __name__ == "__main__":
    main()
