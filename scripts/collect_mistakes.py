import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from data.no_guess import generate_no_guess_board
from game.constants import GameStatus, MoveType
from game.game import MinesweeperGame
from game.probability_solver import ProbabilitySolver
from game.solver import ConstraintSolver
from training.evaluate import load_model
from training.trajectory_pool import TrajectoryPool
from utils.device import get_device


def _model_probs(model, game: MinesweeperGame, device: torch.device, refine_steps: int):
    channels = game.board_to_channels()
    with torch.no_grad():
        x = torch.from_numpy(channels).unsqueeze(0).to(device)
        if refine_steps <= 1:
            probs = model.predict(x, max_refine_steps=1)
        else:
            probs = model.refine(x, num_steps=refine_steps)[-1]
        probs = probs.squeeze(0)
        if probs.dim() == 3:
            probs = probs.squeeze(0)
    return probs.cpu().numpy()


def _pick_model_action(probs: np.ndarray, covered: np.ndarray, width: int):
    masked_probs = np.where(covered, probs, 2.0)
    best_idx = int(np.argmin(masked_probs))
    return divmod(best_idx, width)


def _safe_mask(game: MinesweeperGame, safe_cells) -> np.ndarray:
    mask = np.zeros_like(game.covered_cells, dtype=bool)
    for r, c in safe_cells:
        mask[r, c] = game.covered_cells[r, c]
    return mask


def _classify_state(
    game: MinesweeperGame,
    model_probs: np.ndarray,
    action: tuple[int, int],
    calibration_margin: float,
) -> tuple[str, dict, Optional[np.ndarray]]:
    covered = game.covered_cells
    mines = game.get_mine_mask()
    r, c = action

    safe_cells, _mine_cells = ConstraintSolver(game).find_safe_and_mines()
    safe_set = set(safe_cells)
    has_solver_safe = len(safe_set) > 0
    selected_is_mine = bool(mines[r, c])

    solver_probs = None
    category = "clean"

    if has_solver_safe and (r, c) not in safe_set:
        category = "rule_guard_avoidable"
        solver_probs = ProbabilitySolver(game).compute_probabilities()
    elif selected_is_mine:
        category = "hard_sorting"
        solver_probs = ProbabilitySolver(game).compute_probabilities()
    else:
        solver_probs = ProbabilitySolver(game).compute_probabilities()
        masked_solver = np.where(covered, solver_probs, 2.0)
        best_solver_prob = float(masked_solver.min())
        selected_solver_prob = float(solver_probs[r, c])
        if selected_solver_prob > best_solver_prob + calibration_margin:
            category = "calibration_drift"

    safe_model_prob = None
    if has_solver_safe:
        safe_candidates = _safe_mask(game, safe_cells)
        safe_model_prob = float(np.where(safe_candidates, model_probs, 2.0).min())

    masked_model = np.where(covered, model_probs, 2.0)
    masked_solver = None
    if solver_probs is not None:
        masked_solver = np.where(covered, solver_probs, 2.0)

    detail = {
        "category": category,
        "action": [int(r), int(c)],
        "selected_is_mine": selected_is_mine,
        "has_solver_safe": has_solver_safe,
        "solver_safe_count": int(len(safe_set)),
        "selected_model_prob": float(model_probs[r, c]),
        "best_model_prob": float(masked_model.min()),
        "best_solver_safe_model_prob": safe_model_prob,
        "selected_solver_prob": (
            float(solver_probs[r, c]) if solver_probs is not None else None
        ),
        "best_solver_prob": (
            float(masked_solver.min()) if masked_solver is not None else None
        ),
    }
    return category, detail, solver_probs


def _sample_from_state(
    game: MinesweeperGame,
    action: tuple[int, int],
    solver_probs: np.ndarray,
) -> dict:
    return {
        "mines": game.get_mine_mask().copy(),
        "actions": [action],
        "masks": [game.covered_cells.copy()],
        "probs": [solver_probs.astype(np.float32)],
    }


def _write_npz(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for i, sample in enumerate(samples):
        data[f"mines_{i}"] = sample["mines"]
        data[f"actions_{i}"] = np.array(sample["actions"], dtype=np.int32)
        data[f"masks_{i}"] = np.array(sample["masks"], dtype=bool)
        data[f"probs_{i}"] = np.array(sample["probs"], dtype=np.float32)
    np.savez_compressed(path, **data)


def _setup_pool(path: Optional[Path], width: int, height: int, mines: int):
    if path is None:
        return None
    return TrajectoryPool(
        board_width=width,
        board_height=height,
        board_mines=mines,
        data_dir=str(path),
        eval_mode=True,
    )


def _get_game(pool, idx: int, rng, width: int, height: int, mines: int):
    if pool is not None:
        return pool.get_eval_game(idx, rng)
    return generate_no_guess_board(
        width=width,
        height=height,
        total_mines=mines,
        rng=rng,
        max_attempts=200,
    )


def collect_mistakes(args) -> dict:
    device = get_device(args.device)
    model = load_model(args.checkpoint, device)
    rng = np.random.default_rng(args.seed)
    pool = _setup_pool(args.board_pool, args.width, args.height, args.mines)
    save_categories = {
        category.strip()
        for category in args.save_categories.split(",")
        if category.strip()
    }

    samples = []
    records = []
    counts = {
        "clean": 0,
        "rule_guard_avoidable": 0,
        "hard_sorting": 0,
        "calibration_drift": 0,
        "gen_failed": 0,
        "won": 0,
        "lost": 0,
        "stuck": 0,
        "early_stopped": 0,
        "steps": 0,
        "saved": 0,
    }

    t0 = time.time()
    for game_idx in range(args.n_games):
        game = _get_game(pool, game_idx, rng, args.width, args.height, args.mines)
        if game is None:
            counts["gen_failed"] += 1
            continue

        step = 0
        stopped_for_samples = False
        while game.status == GameStatus.PLAYING and step < args.max_steps:
            covered = game.covered_cells
            if not covered.any():
                break

            probs = _model_probs(model, game, device, args.refine_steps)
            action = _pick_model_action(probs, covered, game.width)
            category, detail, solver_probs = _classify_state(
                game, probs, action, args.calibration_margin
            )
            counts[category] += 1
            counts["steps"] += 1

            if category in save_categories and solver_probs is not None:
                samples.append(_sample_from_state(game, action, solver_probs))
                counts["saved"] += 1
                detail.update({"game": game_idx, "step": step})
                records.append(detail)
                if args.max_samples and len(samples) >= args.max_samples:
                    stopped_for_samples = True
                    break

            r, c = action
            game.make_move(r, c, MoveType.REVEAL)
            step += 1

        if stopped_for_samples:
            counts["early_stopped"] += 1
        elif game.status == GameStatus.WON:
            counts["won"] += 1
        elif game.status == GameStatus.LOST:
            counts["lost"] += 1
        else:
            counts["stuck"] += 1

        if args.max_samples and len(samples) >= args.max_samples:
            break

        if not args.quiet and (game_idx + 1) % max(1, min(50, args.n_games // 5)) == 0:
            print(
                f"[{game_idx + 1}/{args.n_games}] "
                f"saved={counts['saved']} "
                f"avoidable={counts['rule_guard_avoidable']} "
                f"hard={counts['hard_sorting']} "
                f"drift={counts['calibration_drift']}"
            )

    if pool is not None:
        pool.save_eval_cache()

    elapsed = time.time() - t0
    played = args.n_games - counts["gen_failed"]
    summary = {
        "checkpoint": args.checkpoint,
        "board": {
            "width": args.width,
            "height": args.height,
            "mines": args.mines,
        },
        "n_games": args.n_games,
        "played": played,
        "seed": args.seed,
        "refine_steps": args.refine_steps,
        "save_categories": sorted(save_categories),
        "counts": counts,
        "rates": {
            "win_rate": counts["won"] / max(1, played),
            "rule_guard_avoidable_per_step": counts["rule_guard_avoidable"] / max(1, counts["steps"]),
            "hard_sorting_per_step": counts["hard_sorting"] / max(1, counts["steps"]),
            "calibration_drift_per_step": counts["calibration_drift"] / max(1, counts["steps"]),
        },
        "elapsed_seconds": elapsed,
        "records": records,
    }

    if args.output:
        output = Path(args.output)
        _write_npz(output, samples)
        summary_path = output.with_suffix(".json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        summary["output"] = str(output)
        summary["summary"] = str(summary_path)

    return summary


def main():
    p = argparse.ArgumentParser(
        description="Collect model mistake states for failure mining diagnostics"
    )
    p.add_argument("checkpoint", help="Path to model checkpoint (.pt)")
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--height", type=int, default=8)
    p.add_argument("--mines", type=int, default=32)
    p.add_argument("--n_games", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--refine_steps", type=int, default=4)
    p.add_argument("--board_pool", type=Path, default=None)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--calibration_margin", type=float, default=0.05)
    p.add_argument(
        "--save_categories",
        default="rule_guard_avoidable,hard_sorting",
        help="Comma-separated categories to save into the training-compatible NPZ",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    summary = collect_mistakes(args)
    printable = dict(summary)
    printable.pop("records", None)
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
