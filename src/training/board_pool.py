"""Board pools for training and evaluation.

Caches generated self-validated boards to disk to avoid repeated solver calls.
Provides multiprocessing generation for training.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import TrainingConfig
from data.self_validated import generate_self_validated_board
from game.constants import GameStatus
from game.game import MinesweeperGame

_DEFAULT_CFG = TrainingConfig()

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
    """Disk-backed pool of self-validated boards for online training.

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

        seeds = [self.rng.integers(0, 2**31) for _ in range(needed)]

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
