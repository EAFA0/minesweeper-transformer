"""RL environment — wraps MinesweeperGame for policy gradient training.

All boards are self-validated: generate → verify with ProbabilitySolver →
retry if unsolvable. Every game CAN be won, so win rate measures
model improvement cleanly.

Supports "mine continue" mode: when model hits a mine, game continues
with a penalty instead of ending. This provides denser training signal.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import numpy as np

from minesweeper.game import MinesweeperGame
from minesweeper.constants import CellState, MoveType, GameStatus
from data.self_validated import generate_self_validated_board


# ─── Reward Config ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rewards:
    reveal_safe: float = 1.0
    hit_mine: float = -5.0
    win: float = 100.0
    step_penalty: float = 0.0


# ─── Environment ────────────────────────────────────────────────────────────

class RLEnv:
    """RL environment for minesweeper — self-validated boards only.

    State: (10, H, W) board channels
    Action: (r, c) — reveal a covered cell
    Reward: immediate reward for the action

    Two modes:
      mine_continue=False: game ends on mine hit
      mine_continue=True: game continues after mine hit, only ends on win or max steps
    """

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        total_mines: int = 10,
        mine_continue: bool = False,
        max_steps: int = 200,
        warmup_clicks: int = 0,
        mixed: bool = False,                # random size + density each episode
        mixed_min_size: int = 6,
        mixed_max_size: int = 10,
        mixed_min_density: float = 0.10,
        mixed_max_density: float = 0.40,
        rewards: Optional[Rewards] = None,
        rng: Optional[np.random.Generator] = None,
        board_pool: Optional[Any] = None,  # RLBoardPool for pre-generated boards
    ):
        self.width = width
        self.height = height
        self.total_mines = total_mines
        self.mine_continue = mine_continue
        self.max_steps = max_steps
        self.warmup_clicks = warmup_clicks
        self.mixed = mixed
        self.mixed_min_size = mixed_min_size
        self.mixed_max_size = mixed_max_size
        self.mixed_min_density = mixed_min_density
        self.mixed_max_density = mixed_max_density
        self.rewards = rewards or Rewards()
        self.rng = rng or np.random.default_rng()
        self.board_pool = board_pool

        self.game: Optional[MinesweeperGame] = None
        self._steps: int = 0
        self._hits: int = 0

    def reset(self) -> np.ndarray:
        """Start a new game. Returns initial state channels (10, H, W)."""
        self._steps = 0
        self._hits = 0

        # Board pool: sample pre-generated board
        if self.board_pool is not None:
            result = self.board_pool.sample(self.rng)
            if result is not None:
                self.game, w, h = result
                self.width = w
                self.height = h
                self.total_mines = int(self.game.get_mine_mask().sum())
                return self.state

        # Mixed mode: random size + density each episode
        if self.mixed:
            w = self.rng.integers(self.mixed_min_size, self.mixed_max_size + 1)
            h = self.rng.integers(self.mixed_min_size, self.mixed_max_size + 1)
            density = self.rng.uniform(self.mixed_min_density, self.mixed_max_density)
            mines = max(1, int(w * h * density))
            self.width = w
            self.height = h
            self.total_mines = mines
        else:
            w, h, mines = self.width, self.height, self.total_mines

        # Try to generate board, retry with easier config if failed
        self.game = None
        for _ in range(5):  # up to 5 attempts with different seeds
            self.game = generate_self_validated_board(
                width=w, height=h, total_mines=mines,
                rng=self.rng, warmup_clicks=self.warmup_clicks,
            )
            if self.game is not None:
                break
            # Reduce mines slightly for next attempt if density is high
            if mines > w * h * 0.3:
                mines = max(1, mines - 1)

        if self.game is None:
            raise RuntimeError(
                f"Failed to generate self-validated board "
                f"{w}×{h}/{mines} (after retries)"
            )

        return self.state

    def step(self, r: int, c: int) -> Tuple[np.ndarray, float, bool]:
        """Execute a reveal action. Returns (next_state, reward, done)."""
        if self.game is None:
            raise RuntimeError("Call reset() first.")

        is_mine = (self.game.board[r, c] == -1)
        covered_before = self.game._safe_covered

        self.game.make_move(r, c, MoveType.REVEAL)
        self._steps += 1

        covered_after = self.game._safe_covered
        cells_revealed = covered_before - covered_after

        if is_mine:
            reward = self.rewards.hit_mine
        else:
            reward = cells_revealed * self.rewards.reveal_safe + self.rewards.step_penalty

        if self.game.status == GameStatus.WON:
            # Include reward for all unrevealed safe cells.
            # Without this, flagging all mines early yields less reward
            # than clicking every cell one-by-one — perverse incentive.
            unrevealed = self.game._safe_covered
            reward += self.rewards.win + unrevealed * self.rewards.reveal_safe
            return self.state, reward, True
        elif self.game.status == GameStatus.LOST:
            self._hits += 1
            if self.mine_continue:
                # Convert exploded mine to flag — "you should have flagged this"
                self.game.visible[r, c] = CellState.FLAGGED
                self.game.status = GameStatus.PLAYING
                return self.state, reward, False
            return self.state, reward, True
        elif self._steps >= self.max_steps:
            return self.state, reward, True

        return self.state, reward, False

    @property
    def state(self) -> np.ndarray:
        """Current board as (10, pad_H, pad_W) channels — padded for batching."""
        if self.game is None:
            pad = self.mixed_max_size if self.mixed else max(self.width, self.height)
            return np.zeros((10, pad, pad), dtype=np.float32)
        channels = self.game.board_to_channels().astype(np.float32)
        if self.mixed:
            _, H, W = channels.shape
            pad = self.mixed_max_size
            if H != pad or W != pad:
                padded = np.zeros((10, pad, pad), dtype=np.float32)
                padded[:, :H, :W] = channels
                padded[0, H:, :] = 1.0   # pad: covered
                padded[0, :, W:] = 1.0
                return padded
        return channels

    @property
    def covered_mask(self) -> np.ndarray:
        """Boolean array of covered cells: (pad_H, pad_W) — with padding masked out."""
        if self.game is None:
            return np.zeros((self.height, self.width), dtype=bool)
        actual = self.game.covered_cells
        if self.mixed:
            pad = self.mixed_max_size
            H, W = actual.shape
            padded = np.zeros((pad, pad), dtype=bool)
            padded[:H, :W] = actual
            return padded
        return actual

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def mine_hits(self) -> int:
        return self._hits
