"""Shared evaluation utilities for training and standalone eval.

Provides:
- evaluate_model: play N games and report win rate / action accuracy
- pick_action / play_one_game: low-level game interaction

Used by both scripts/evaluate.py and the training module's validation.
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from config import POLICY, TrainingConfig
from data.self_validated import generate_self_validated_board
from game.constants import GameStatus, MoveType
from game.game import MinesweeperGame
from model.architecture import MinesweeperTransformer, ModelConfig
from training.trajectory_pool import TrajectoryPool

_DEFAULT_CFG = TrainingConfig()

# ── Inference ───────────────────────────────────────────────────────────────

def pick_action(
    model: MinesweeperTransformer,
    game: MinesweeperGame,
    device: torch.device,
    refine_steps: int = None,
) -> Optional[Tuple[MoveType, int, int, int]]:
    """Choose the next move + return refinement steps actually used.

    Returns (MoveType, row, col, n_refine_steps) or None if no moves.
    """
    if refine_steps is None:
        refine_steps = POLICY.refinement.eval_max_steps

    channels = game.board_to_channels()
    with torch.no_grad():
        x = torch.from_numpy(channels).unsqueeze(0).to(device)
        probs = model.predict(x, max_refine_steps=refine_steps)
        
        probs_2d = probs.squeeze(0)  # (H, W) or (1, H, W)
        if probs_2d.dim() == 3:
            probs_2d = probs_2d.squeeze(0)
        probs = probs_2d.cpu().numpy()

    covered = game.covered_cells
    if not covered.any():
        return None

    masked_probs = np.where(covered, probs, 2.0)
    best_idx = np.argmin(masked_probs)
    best_r, best_c = divmod(int(best_idx), game.width)
    return MoveType.REVEAL, best_r, best_c, refine_steps


def play_one_game(
    model: MinesweeperTransformer,
    device: torch.device,
    game: MinesweeperGame,
    max_steps: int = 200,
    refine_steps: int = None,
) -> dict:
    """Play one game to completion. Returns detailed stats."""
    steps = 0
    safe_reveals = 0
    mine_hits = 0
    refine_steps_used: list = []

    while game.status == GameStatus.PLAYING and steps < max_steps:
        action = pick_action(model, game, device, refine_steps=refine_steps)
        if action is None:
            break

        move_type, mr, mc, n_refine = action
        refine_steps_used.append(n_refine)
        is_safe = not game.get_mine_mask()[mr, mc]
        game.make_move(mr, mc, move_type)
        steps += 1

        if is_safe:
            safe_reveals += 1
        else:
            mine_hits += 1

    return {
        "status": game.status,
        "steps": steps,
        "safe_reveals": safe_reveals,
        "mine_hits": mine_hits,
        "refine_steps": refine_steps_used
    }


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(
    model: MinesweeperTransformer,
    device: torch.device,
    n_games: int = 1000,
    width: int = _DEFAULT_CFG.board_width,
    height: int = _DEFAULT_CFG.board_height,
    total_mines: int = _DEFAULT_CFG.board_mines,
    seed: int = 42,
    board_pool_path: Optional[Path] = None,
    refine_steps: int = None,
    quiet: bool = False,
) -> dict:
    """Evaluate a model by playing N games. Returns aggregate stats.

    Args:
        model: already loaded and in eval mode
        n_games: number of games to play
        board_pool_path: .npz path for board caching (auto-generated if None)
        quiet: suppress per-game progress
    """
    device_t = device if isinstance(device, torch.device) else torch.device(device)
    rng = np.random.default_rng(seed)
    metrics = _EvalMetrics(n_games, quiet)

    pool = _setup_board_pool(board_pool_path, width, height, total_mines)
    if pool and pool.eval_size > 0 and not quiet:
        print(f"Board pool: {pool.eval_size} boards cached in {pool.data_dir}")

    t0 = time.time()

    for i in range(n_games):
        game = _get_game(pool, i, rng, width, height, total_mines)
        if game is None:
            metrics.add_gen_failure()
            continue

        stats = play_one_game(model, device_t, game, refine_steps=refine_steps)
        if not isinstance(stats.get("refine_steps"), list):
            print(f"DEBUG stats: {stats}")
        metrics.add_result(stats)
        metrics.maybe_print(i + 1, n_games, t0)

    if pool:
        pool.save_eval_cache()

    return metrics.summary(n_games, time.time() - t0)


class _EvalMetrics:
    """Internal accumulator for evaluation results."""

    def __init__(self, n_games: int, quiet: bool = False):
        self.won = 0
        self.lost = 0
        self.stuck = 0
        self.gen_failed = 0
        self.total_safe = 0
        self.total_mine = 0
        self.steps_list: list = []
        self.refine_steps_list: list = []  # all refine_step counts across all games
        self.n_games = n_games
        self.quiet = quiet
        self.progress_interval = max(1, min(50, n_games // 5))

    def add_gen_failure(self):
        self.gen_failed += 1

    def add_result(self, stats: dict):
        if stats["status"] == GameStatus.WON:
            self.won += 1
        elif stats["status"] == GameStatus.LOST:
            self.lost += 1
        else:
            self.stuck += 1
        self.steps_list.append(stats["steps"])
        self.total_safe += stats["safe_reveals"]
        self.total_mine += stats["mine_hits"]
        if stats.get("refine_steps"):
            self.refine_steps_list.extend(stats["refine_steps"])

    def maybe_print(self, i: int, n: int, t0: float):
        if self.quiet or i % self.progress_interval != 0:
            return
        played = i - self.gen_failed
        wr = self.won / max(1, played)
        acc = self.total_safe / max(1, self.total_safe + self.total_mine)
        elapsed = time.time() - t0
        print(
            f"  [{i:5d}/{n}] "
            f"win={self.won:4d} ({wr:.1%})  "
            f"loss={self.lost:4d}  "
            f"stuck={self.stuck:3d}  "
            f"act_acc={acc:.3f}  "
            f"({elapsed:.0f}s)"
        )

    def summary(self, n_games: int, elapsed: float) -> dict:
        played = n_games - self.gen_failed
        win_rate = self.won / max(1, played)
        total_reveals = self.total_safe + self.total_mine
        action_acc = self.total_safe / max(1, total_reveals)
        avg_steps = float(np.mean(self.steps_list)) if self.steps_list else 0.0
        avg_refine = float(np.mean(self.refine_steps_list)) if self.refine_steps_list else 0.0

        return {
            "n_games": n_games,
            "gen_failed": self.gen_failed,
            "won": self.won,
            "lost": self.lost,
            "stuck": self.stuck,
            "played": played,
            "win_rate": win_rate,
            "action_accuracy": action_acc,
            "avg_steps": avg_steps,
            "avg_refine_steps": avg_refine,
            "elapsed": elapsed,
        }


def _setup_board_pool(
    path: str, width: int, height: int, mines: int
) -> Optional[TrajectoryPool]:
    if not path:  # None or empty string
        path = Path(f"eval_boards_{width}x{height}_{mines}.npz")
    return TrajectoryPool(
        board_width=width,
        board_height=height,
        board_mines=mines,
        data_dir=str(path),
        eval_mode=True
    )


def _get_game(
    pool: Optional[TrajectoryPool],
    idx: int,
    rng: np.random.Generator,
    width: int,
    height: int,
    mines: int,
) -> Optional[MinesweeperGame]:
    if pool:
        return pool.get_eval_game(idx, rng)
    game = generate_self_validated_board(
        width=width, height=height, total_mines=mines,
        rng=rng, max_attempts=200,
    )
    if game is not None and game.status != GameStatus.PLAYING:
        return None
    return game


def load_model(checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch_version = ckpt.get("arch_version", "V4")
    
    if arch_version == "V1":
        from model.architecture_v1 import MinesweeperTransformerV1
        model = MinesweeperTransformerV1(ModelConfig()).to(device)
    elif arch_version == "V1_5":
        from model.architecture_v1_5 import MinesweeperTransformerV1_5
        model = MinesweeperTransformerV1_5(ModelConfig()).to(device)
    else:
        model = MinesweeperTransformer(ModelConfig()).to(device)
        
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
