"""EvalBoardPool for evaluation.

Caches generated self-validated boards to disk to avoid repeated solver calls
during evaluation. This ensures deterministic evaluation across epochs.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from data.self_validated import generate_self_validated_board
from game.constants import GameStatus
from game.game import MinesweeperGame

class EvalBoardPool:
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
