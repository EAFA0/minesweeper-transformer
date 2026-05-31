"""Board pool for RL training — pre-generate boards to eliminate per-episode overhead.

Supports both mixed-size and fixed-size pools.

Fixed mode:
  RLBoardPool("pool.npz", width=10, height=10, mines=40, target_size=200)

Mixed mode:
  RLBoardPool("pool.npz", min_size=6, max_size=10, min_density=0.1, max_density=0.4)
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from minesweeper.game import MinesweeperGame
from minesweeper.constants import GameStatus
from data.self_validated import generate_self_validated_board


class RLBoardPool:
    """Pre-generated board cache for RL training.

    Usage:
        pool = RLBoardPool("rl_boards.npz", width=10, height=10, mines=40, target_size=200)
        game, w, h = pool.sample(rng)  # returns (game, width, height)
    """

    def __init__(
        self,
        path: Path,
        min_size: int = 6,
        max_size: int = 10,
        min_density: float = 0.10,
        max_density: float = 0.40,
        target_size: int = 5000,
        width: Optional[int] = None,
        height: Optional[int] = None,
        mines: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.path = Path(path)
        self.min_size = min_size
        self.max_size = max_size
        self.min_density = min_density
        self.max_density = max_density
        self.target_size = target_size
        self.fixed_size = width is not None and height is not None and mines is not None
        self.width = width
        self.height = height
        self.mines = mines
        self.rng = rng or np.random.default_rng()

        # (mine_mask, visible, width, height) per board
        self._boards: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = np.load(self.path, allow_pickle=True)
        n = len(data.files) // 4  # mask_i, vis_i, w_i, h_i
        for i in range(n):
            self._boards.append((
                data[f"mask_{i}"],
                data[f"vis_{i}"],
                int(data[f"w_{i}"]),
                int(data[f"h_{i}"]),
            ))

    def _save(self) -> None:
        save_dict = {}
        for i, (mask, vis, w, h) in enumerate(self._boards):
            save_dict[f"mask_{i}"] = mask
            save_dict[f"vis_{i}"] = vis
            save_dict[f"w_{i}"] = np.array(w)
            save_dict[f"h_{i}"] = np.array(h)
        np.savez_compressed(self.path, **save_dict)

    def sample(self, rng: np.random.Generator) -> Optional[Tuple[MinesweeperGame, int, int]]:
        """Sample a random board from the pool, generating new ones if needed."""
        if len(self._boards) < 100:
            needed = self.target_size - len(self._boards)
            self._generate_batch(max(needed, 50))

        if not self._boards:
            return None

        idx = rng.integers(0, len(self._boards))
        mask, vis, w, h = self._boards[idx]
        return MinesweeperGame.from_mine_mask(w, h, mask, first_done=True, visible=vis), w, h

    def _generate_batch(self, n: int) -> None:
        """Generate n new boards and add to pool."""
        for _ in range(n):
            if self.fixed_size:
                w, h, mines = self.width, self.height, self.mines
            else:
                w = self.rng.integers(self.min_size, self.max_size + 1)
                h = self.rng.integers(self.min_size, self.max_size + 1)
                density = self.rng.uniform(self.min_density, self.max_density)
                mines = max(1, int(w * h * density))

            game = generate_self_validated_board(
                width=w, height=h, total_mines=mines,
                rng=self.rng,
            )
            if game is None or game.status != GameStatus.PLAYING:
                continue

            self._boards.append((
                game.get_mine_mask(),
                game.visible.copy(),
                w, h,
            ))

            if len(self._boards) >= self.target_size:
                break

        self._save()

    def fill(self) -> None:
        """Ensure pool has target_size boards."""
        needed = self.target_size - len(self._boards)
        if needed > 0:
            self._generate_batch(needed)

    @property
    def size(self) -> int:
        return len(self._boards)
