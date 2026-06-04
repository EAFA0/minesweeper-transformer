"""Shared evaluation utilities for training and standalone eval.

Provides:
- BoardPool: cache and reuse self-validated no-guess boards
- evaluate_model: play N games and report win rate / action accuracy
- pick_action / play_one_game: low-level game interaction

Used by both scripts/evaluate.py and the training module's validation.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from config import POLICY, TrainingConfig
from data.self_validated import generate_self_validated_board
from game.constants import GameStatus, MoveType
from game.game import MinesweeperGame
from model.architecture import MinesweeperTransformer, ModelConfig

_DEFAULT_CFG = TrainingConfig()

# ── Board Pool ─────────────────────────────────────────────────────────────

class BoardPool:
    """Cache self-validated boards for fast reuse across eval runs."""

    def __init__(self, path: Path, width: int, height: int, mines: int):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.mines = mines
        self._cache_mines: Optional[List[np.ndarray]] = None
        self._cache_visible: Optional[List[np.ndarray]] = None
        self._unsaved = 0

    def _load(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if self._cache_mines is not None:
            return self._cache_mines, self._cache_visible
        if self.path.exists():
            data = np.load(self.path, allow_pickle=True)
            n = len(data.files) // 2
            self._cache_mines = [data[f"mines_{i}"] for i in range(n)]
            self._cache_visible = [data[f"visible_{i}"] for i in range(n)]
            return self._cache_mines, self._cache_visible
        self._cache_mines = []
        self._cache_visible = []
        return self._cache_mines, self._cache_visible

    def _save(self, mines: List[np.ndarray], visibles: List[np.ndarray]) -> None:
        save_dict = {}
        for i, (m, v) in enumerate(zip(mines, visibles)):
            save_dict[f"mines_{i}"] = m
            save_dict[f"visible_{i}"] = v
        np.savez_compressed(self.path, **save_dict)
        self._cache_mines = mines
        self._cache_visible = visibles
        self._unsaved = 0

    def get(self, idx: int, rng: np.random.Generator) -> Optional[MinesweeperGame]:
        """Get board #idx from pool, generating and caching if needed."""
        mines_list, visibles_list = self._load()
        if idx < len(mines_list):
            return MinesweeperGame.from_mine_mask(
                self.width, self.height, mines_list[idx],
                first_done=True, visible=visibles_list[idx],
            )

        # Generate with self-validated solver
        game = generate_self_validated_board(
            width=self.width, height=self.height, total_mines=self.mines,
            rng=rng, max_attempts=200,
        )

        if game is None or game.status != GameStatus.PLAYING:
            return None

        mine_mask = game.get_mine_mask()
        mines_list.append(mine_mask)
        visibles_list.append(game.visible.copy())

        self._unsaved += 1
        if self._unsaved >= 50:
            self._save(mines_list, visibles_list)

        return game

    def save_pending(self) -> None:
        if self._unsaved > 0 and self._cache_mines is not None:
            self._save(self._cache_mines, self._cache_visible)

    @property
    def size(self) -> int:
        mines, _ = self._load()
        return len(mines)



# ── Training Board Pool ────────────────────────────────────────────────────

def _mp_generate_board(width: int, height: int, mines: int, seed: int):
    """Multiprocessing worker: generate one board, return (mine_mask, visible)."""
    rng = np.random.default_rng(seed)
    game = generate_self_validated_board(
        width=width, height=height,
        total_mines=mines,
        rng=rng, max_attempts=200,
    )
    if game is not None and game.status == GameStatus.PLAYING:
        return (game.get_mine_mask(), game.visible.copy())
    return None


class TrainBoardPool:
    """Disk-backed pool of self-validated boards for online BCE training.

    Caches pre-generated boards as .npz (mine masks + visible state),
    same format as BoardPool.  On restart, loads existing cache instantly.
    Boards are consumed (pop'd) during training; pool refills from
    solver when low, optionally using multiple worker processes.
    """

    def __init__(self,
                 width: int = _DEFAULT_CFG.board_width,
                 height: int = _DEFAULT_CFG.board_height,
                 mines: int = _DEFAULT_CFG.board_mines,
                 pool_size: int = _DEFAULT_CFG.board_pool_size,
                 seed: int = 42,
                 num_workers: int = _DEFAULT_CFG.pool_workers,
                 cache_path: Optional[str] = None):
        self.width = width
        self.height = height
        self.mines = mines
        self.pool_size = pool_size
        self.num_workers = num_workers
        self.rng = np.random.default_rng(seed)

        if cache_path is None:
            cache_path = f"train_boards_{width}x{height}_{mines}.npz"
        self.path = Path(cache_path)

        self._mines_list: List[np.ndarray] = []
        self._visible_list: List[np.ndarray] = []
        self._unsaved = 0
        self._load_disk()
        self._fill()
        self._save_now()

    def _load_disk(self):
        if not self.path.exists():
            return
        data = np.load(self.path, allow_pickle=True)
        n = len(data.files) // 2
        self._mines_list = [data[f"mines_{i}"] for i in range(n)]
        self._visible_list = [data[f"visible_{i}"] for i in range(n)]
        print(f"  Loaded {n} boards from {self.path}")

    def _save_now(self):
        if not self._mines_list:
            return
        save_dict = {}
        for i, (m, v) in enumerate(zip(self._mines_list, self._visible_list)):
            save_dict[f"mines_{i}"] = m
            save_dict[f"visible_{i}"] = v
        np.savez_compressed(self.path, **save_dict)
        self._unsaved = 0

    def _generate_one(self) -> Optional[MinesweeperGame]:
        game = generate_self_validated_board(
            width=self.width, height=self.height,
            total_mines=self.mines,
            rng=self.rng, max_attempts=200,
        )
        if game is not None and game.status == GameStatus.PLAYING:
            return game
        return None

    def _fill(self):
        needed = self.pool_size - len(self._mines_list)
        if needed <= 0:
            return

        if self.num_workers > 1 and needed >= self.num_workers:
            self._fill_parallel(needed)
        else:
            self._fill_serial(needed)

        if self._unsaved >= 10:
            self._save_now()

    def _fill_serial(self, needed: int):
        for _ in range(needed):
            g = self._generate_one()
            if g is not None:
                self._mines_list.append(g.get_mine_mask())
                self._visible_list.append(g.visible.copy())
                self._unsaved += 1

    def _fill_parallel(self, needed: int):
        """Generate boards using multiprocessing."""
        from concurrent.futures import ProcessPoolExecutor
        import os

        seeds = [self.rng.integers(0, 2**31) for _ in range(needed)]
        chunksize = max(1, needed // (self.num_workers * 2))

        with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
            futures = [
                ex.submit(
                    _mp_generate_board,
                    self.width, self.height, self.mines, seed,
                )
                for seed in seeds
            ]
            for fut in futures:
                try:
                    result = fut.result(timeout=120)
                    if result is not None:
                        self._mines_list.append(result[0])
                        self._visible_list.append(result[1])
                        self._unsaved += 1
                except Exception:
                    pass  # skip failed generations

    def get(self) -> Optional[MinesweeperGame]:
        """Pop one fresh board. Auto-refills and saves to disk."""
        if not self._mines_list:
            self._fill()
        if not self._mines_list:
            return None

        mine = self._mines_list.pop()
        vis = self._visible_list.pop()

        # Save remaining boards to disk
        if self._mines_list:
            self._save_now()

        # Refill in background
        if len(self._mines_list) < self.pool_size // 2:
            self._fill()

        return MinesweeperGame.from_mine_mask(
            self.width, self.height, mine,
            first_done=True, visible=vis,
        )

    @property
    def available(self) -> int:
        return len(self._mines_list)



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
        results = model.refine(x, num_steps=refine_steps)
        probs = results[-1].squeeze(0).squeeze(0).cpu().numpy()
        n_refine = len(results)

    covered = game.covered_cells
    if not covered.any():
        return None

    masked_probs = np.where(covered, probs, 2.0)
    best_idx = np.argmin(masked_probs)
    best_r, best_c = divmod(int(best_idx), game.width)
    return MoveType.REVEAL, best_r, best_c, n_refine


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
        "refine_steps": refine_steps_used,
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
    if pool and pool.size > 0 and not quiet:
        print(f"Board pool: {pool.size} boards cached in {pool.path}")

    t0 = time.time()

    for i in range(n_games):
        game = _get_game(pool, i, rng, width, height, total_mines)
        if game is None:
            metrics.add_gen_failure()
            continue

        stats = play_one_game(model, device_t, game, refine_steps=refine_steps)
        metrics.add_result(stats)
        metrics.maybe_print(i + 1, n_games, t0)

    if pool:
        pool.save_pending()

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
    path: Optional[Path], width: int, height: int, mines: int
) -> Optional[BoardPool]:
    if not path:  # None or empty string
        path = Path(f"eval_boards_{width}x{height}_{mines}.npz")
    return BoardPool(path, width, height, mines)


def _get_game(
    pool: Optional[BoardPool],
    idx: int,
    rng: np.random.Generator,
    width: int,
    height: int,
    mines: int,
) -> Optional[MinesweeperGame]:
    if pool:
        return pool.get(idx, rng)
    game = generate_self_validated_board(
        width=width, height=height, total_mines=mines,
        rng=rng, max_attempts=200,
    )
    if game is not None and game.status != GameStatus.PLAYING:
        return None
    return game


def load_model(checkpoint_path: str, device: torch.device) -> MinesweeperTransformer:
    """Load a trained model from checkpoint."""
    model = MinesweeperTransformer(ModelConfig()).to(device)
    model.load_pretrained(checkpoint_path, device)
    model.eval()
    return model
