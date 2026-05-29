"""RL environment — wraps MinesweeperGame for policy gradient training.

Supports two board modes:
  self_validated — boards solvable by ProbabilitySolver (fast, RL-safe)
  random        — completely random boards (includes guessing)

Supports "mine continue" mode: when model hits a mine, game continues
with a penalty instead of ending. This provides denser training signal.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from minesweeper.game import MinesweeperGame
from minesweeper.constants import CellState, MoveType, GameStatus
from data.self_validated import generate_self_validated_board


# ─── Reward Config ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rewards:
    reveal_safe: float = 1.0
    hit_mine: float = -10.0
    win: float = 20.0
    step_penalty: float = 0.0     # optional per-step penalty


# ─── Environment ────────────────────────────────────────────────────────────

class RLEnv:
    """RL environment for minesweeper.

    State: (10, H, W) board channels
    Action: (r, c) — reveal a covered cell
    Reward: immediate reward for the action

    Two modes:
      mine_continue=False (default): game ends on mine hit
      mine_continue=True: game continues after mine hit, only ends on win or max steps
    """

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        total_mines: int = 10,
        board_mode: str = "self_validated",  # "self_validated" | "random"
        mine_continue: bool = False,
        max_steps: int = 200,
        rewards: Optional[Rewards] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.width = width
        self.height = height
        self.total_mines = total_mines
        self.board_mode = board_mode
        self.mine_continue = mine_continue
        self.max_steps = max_steps
        self.rewards = rewards or Rewards()
        self.rng = rng or np.random.default_rng()

        self.game: Optional[MinesweeperGame] = None
        self._steps: int = 0
        self._hits: int = 0

    def reset(self) -> np.ndarray:
        """Start a new game. Returns initial state channels (10, H, W)."""
        self._steps = 0
        self._hits = 0

        if self.board_mode == "self_validated":
            self.game = generate_self_validated_board(
                width=self.width, height=self.height,
                total_mines=self.total_mines, rng=self.rng,
            )
            if self.game is None:
                # Fallback to random
                self.game = self._random_board()
        else:
            self.game = self._random_board()

        return self.state

    def step(self, r: int, c: int) -> Tuple[np.ndarray, float, bool]:
        """Execute a reveal action. Returns (next_state, reward, done)."""
        if self.game is None:
            raise RuntimeError("Call reset() first.")

        reward = self._compute_reward(r, c)
        self.game.make_move(r, c, MoveType.REVEAL)
        self._steps += 1

        # Game status
        if self.game.status == GameStatus.WON:
            reward += self.rewards.win
            return self.state, reward, True
        elif self.game.status == GameStatus.LOST:
            self._hits += 1
            if self.mine_continue:
                # Reset game status so we can continue
                self.game.status = GameStatus.PLAYING
                return self.state, reward, False
            return self.state, reward, True
        elif self._steps >= self.max_steps:
            return self.state, reward, True

        return self.state, reward, False

    @property
    def state(self) -> np.ndarray:
        """Current board as (10, H, W) channels."""
        if self.game is None:
            return np.zeros((10, self.height, self.width), dtype=np.float32)
        return self.game.board_to_channels().astype(np.float32)

    @property
    def covered_mask(self) -> np.ndarray:
        """Boolean array of covered cells: (H, W)."""
        if self.game is None:
            return np.zeros((self.height, self.width), dtype=bool)
        return self.game.covered_cells

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def mine_hits(self) -> int:
        return self._hits

    def _random_board(self) -> MinesweeperGame:
        game = MinesweeperGame(self.width, self.height, self.total_mines)
        # Random first click on safe cell
        r = self.rng.integers(0, self.height)
        c = self.rng.integers(0, self.width)
        game.make_move(r, c, MoveType.REVEAL)
        return game

    def _compute_reward(self, r: int, c: int) -> float:
        """Compute reward before executing the move."""
        if self.game is None:
            return 0.0
        if self.game.board[r, c] == -1:
            return self.rewards.hit_mine
        return self.rewards.reveal_safe + self.rewards.step_penalty
