"""Read-only board pool for RL training.

Build pools with ``scripts/generate_rl_pool.py`` before training.

Fixed mode:
  rl_boards_10x10_40.npz

Mixed mode:
  rl_boards_mixed.npz
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from minesweeper.game import MinesweeperGame


def default_pool_path(
    width: Optional[int],
    height: Optional[int],
    mines: Optional[int],
    mixed: bool = False,
) -> str:
    """Return the canonical RL board pool path for fixed or mixed mode."""
    if mixed:
        return "rl_boards_mixed.npz"
    if width is None or height is None or mines is None:
        raise ValueError("width, height, and mines are required for fixed RL pool paths.")
    return f"rl_boards_{width}x{height}_{mines}.npz"


class RLBoardPool:
    """Pre-generated board cache for RL training, read-only during training.

    Usage:
        pool = RLBoardPool("rl_boards_10x10_40.npz")
        game, w, h = pool.sample(rng)  # returns (game, width, height)
    """

    def __init__(self, path: Path):
        self.path = Path(path)

        # (mine_mask, visible, width, height) per board
        self._boards: list[Tuple[np.ndarray, np.ndarray, int, int]] = []
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

    def sample(self, rng: np.random.Generator) -> Optional[Tuple[MinesweeperGame, int, int]]:
        """Sample a random board from the loaded pool."""
        if not self._boards:
            return None

        idx = rng.integers(0, len(self._boards))
        mask, vis, w, h = self._boards[idx]
        return MinesweeperGame.from_mine_mask(w, h, mask, first_done=True, visible=vis), w, h

    @property
    def size(self) -> int:
        return len(self._boards)
