"""Profile no-guess board generation to identify bottleneck.

Measures t_laymine (ms_toollib.laymine_solvable) vs t_verify (is_solver_no_guess)
to determine whether the bottleneck is in SAT-based generation or in the
ProbabilitySolver re-verification step.

Usage:
    python scripts/profile_noguess.py --width 10 --height 10 --mines 40 --n_samples 50
    python scripts/profile_noguess.py  # runs all three density levels
"""

import argparse
import statistics
import sys
import time
from typing import Optional, Tuple

import numpy as np

# Ensure project root is on path
sys.path.insert(0, sys.path[0])  # scripts/ dir is one level below project root


def _profile_single(
    width: int,
    height: int,
    total_mines: int,
    rng: np.random.Generator,
) -> Tuple[Optional[float], Optional[float], bool, bool, bool]:
    """Profile a single no-guess board generation attempt.

    Returns:
        (t_laymine, t_verify, laymine_success, verify_pass, board_obtained)
    """
    import ms_toollib as mt
    from game.constants import MoveType
    from game.game import MinesweeperGame
    from data.no_guess import is_solver_no_guess

    # Random first-click position
    r = rng.integers(0, height)
    c = rng.integers(0, width)

    # ── Phase 1: ms_toollib generation ──
    t0 = time.perf_counter()
    try:
        board_2d, success = mt.laymine_solvable(
            height, width, total_mines, r, c, max_times=10000
        )
    except Exception:
        success = False
    t_laymine = time.perf_counter() - t0

    if not success:
        return t_laymine, None, False, False, False

    # Convert ms-toollib board to mine mask
    board_np = np.array(board_2d, dtype=np.int8)
    mine_mask = np.where(board_np == -1, -1, 0).astype(np.int8)

    # Create game and inject mine layout
    game = MinesweeperGame(width, height, total_mines)
    game.board = mine_mask
    game.first_move = False
    game.make_move(r, c, MoveType.REVEAL)

    # ── Phase 2: ProbabilitySolver re-verification ──
    t0 = time.perf_counter()
    verify_pass = is_solver_no_guess(game)
    t_verify = time.perf_counter() - t0

    return t_laymine, t_verify, True, verify_pass, verify_pass


def _compute_stats(values: list[float]) -> dict:
    """Compute mean, median, p95 for a list of values."""
    if not values:
        return {"mean": None, "median": None, "p95": None, "n": 0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "mean": statistics.mean(sorted_vals),
        "median": statistics.median(sorted_vals),
        "p95": sorted_vals[int(n * 0.95)] if n >= 20 else sorted_vals[-1],
        "n": n,
    }


def profile_config(
    width: int,
    height: int,
    mines: int,
    n_samples: int,
    seed: int,
) -> dict:
    """Profile a single board configuration. Returns a dict of results."""
    rng = np.random.default_rng(seed)

    t_laymine_list: list[float] = []
    t_verify_list: list[float] = []
    laymine_success_count = 0
    verify_pass_count = 0
    board_count = 0
    total_wall_time = 0.0

    t_start = time.perf_counter()

    for _ in range(n_samples):
        iter_start = time.perf_counter()
        t_lay, t_ver, lay_ok, ver_ok, board_ok = _profile_single(
            width, height, mines, rng
        )
        iter_elapsed = time.perf_counter() - iter_start
        total_wall_time += iter_elapsed

        t_laymine_list.append(t_lay)
        if lay_ok:
            laymine_success_count += 1
        if t_ver is not None:
            t_verify_list.append(t_ver)
        if ver_ok:
            verify_pass_count += 1
        if board_ok:
            board_count += 1

    total_wall = time.perf_counter() - t_start

    laymine_stats = _compute_stats(t_laymine_list)
    verify_stats = _compute_stats(t_verify_list)

    laymine_success_rate = laymine_success_count / n_samples if n_samples > 0 else 0
    verify_pass_rate = (
        verify_pass_count / laymine_success_count if laymine_success_count > 0 else 0
    )
    avg_per_board = total_wall / board_count if board_count > 0 else float("inf")

    return {
        "config": f"{width}×{height}/{mines}",
        "n_samples": n_samples,
        "t_laymine": laymine_stats,
        "t_verify": verify_stats,
        "laymine_success_rate": laymine_success_rate,
        "verify_pass_rate": verify_pass_rate,
        "boards_generated": board_count,
        "total_wall_time": total_wall,
        "avg_per_board": avg_per_board,
    }


def print_markdown_table(results: list[dict]) -> None:
    """Print profiling results as a markdown table."""
    print()
    print("## Profiling Results: No-Guess Board Generation")
    print()
    print(
        "| Config | N | t_laymine mean | t_laymine median | t_laymine p95 | "
        "t_verify mean | t_verify median | t_verify p95 | "
        "laymine succ% | verify pass% | boards | avg/board |"
    )
    print(
        "|--------|---|---------------:|-----------------:|-------------:|"
        "--------------:|----------------:|-------------:|"
        "--------------:|-------------:|-------:|----------:|"
    )

    for r in results:
        lm = r["t_laymine"]
        tv = r["t_verify"]

        def fmt(v):
            if v is None:
                return "N/A"
            return f"{v:.4f}s"

        print(
            f"| {r['config']} | {r['n_samples']} | "
            f"{fmt(lm['mean'])} | {fmt(lm['median'])} | {fmt(lm['p95'])} | "
            f"{fmt(tv['mean'])} | {fmt(tv['median'])} | {fmt(tv['p95'])} | "
            f"{r['laymine_success_rate']:.1%} | {r['verify_pass_rate']:.1%} | "
            f"{r['boards_generated']} | {fmt(r['avg_per_board'])} |"
        )

    print()

    # Bottleneck analysis
    print("## Bottleneck Analysis")
    print()
    for r in results:
        lm = r["t_laymine"]
        tv = r["t_verify"]
        if lm["mean"] is None or tv["mean"] is None:
            print(f"- **{r['config']}**: insufficient data for comparison")
            continue
        ratio = lm["mean"] / tv["mean"] if tv["mean"] > 0 else float("inf")
        if ratio > 2:
            verdict = "Bottleneck is **t_laymine** (generation)"
        elif ratio < 0.5:
            verdict = "Bottleneck is **t_verify** (re-verification)"
        else:
            verdict = "Both phases comparable"
        print(
            f"- **{r['config']}**: t_laymine/t_verify = {ratio:.2f}x → {verdict}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Profile no-guess board generation"
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--mines", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # If specific config given, run only that
    if args.width is not None and args.height is not None and args.mines is not None:
        configs = [(args.width, args.height, args.mines)]
    else:
        # Default: three density levels
        configs = [
            (8, 8, 10),   # low density
            (8, 8, 32),   # medium density
            (10, 10, 40), # high density (target)
        ]

    results = []
    for width, height, mines in configs:
        print(f"Profiling {width}×{height}/{mines} ({args.n_samples} samples)...",
              file=sys.stderr)
        r = profile_config(width, height, mines, args.n_samples, args.seed)
        results.append(r)

    print_markdown_table(results)


if __name__ == "__main__":
    main()
