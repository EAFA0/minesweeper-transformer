"""RL environment for Phase 2: wraps MinesweeperGame with solver-guided rewards.

Provides step-by-step interaction for policy gradient training.
Each step returns (next_state, reward, done) — standard RL interface.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from minesweeper.game import MinesweeperGame
from minesweeper.constants import CellState, MoveType, GameStatus
from minesweeper.solver import ConstraintSolver


# ─── Reward Constants ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rewards:
    """Immutable reward configuration."""
    reveal_safe: float = 1.0        # 翻开安全格
    reveal_zero: float = 3.0        # 翻开数字 0（触发 flood fill）
    flag_correct: float = 2.0       # 正确插旗
    flag_wrong: float = -5.0        # 错误插旗（把安全格当雷）
    hit_mine: float = -10.0         # 踩雷
    win: float = 20.0               # 胜利
    solver_approve_reveal: float = 0.5   # solver 判定安全，模型翻开
    solver_approve_flag: float = 1.0     # solver 判定是雷，模型插旗


# ─── Environment ────────────────────────────────────────────────────────────

class MinesweeperEnv:
    """RL environment wrapping MinesweeperGame.

    State: (10, H, W) channels (same as Phase 1 input)
    Action: (move_type, row, col) — reveal or flag a specific cell
    Reward: solver-guided dense reward (see Rewards)
    """

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        total_mines: int = 10,
        rewards: Optional[Rewards] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.width = width
        self.height = height
        self.total_mines = total_mines
        self.rewards = rewards or Rewards()
        self.rng = rng or np.random.default_rng()

        self.game: Optional[MinesweeperGame] = None
        self.solver: Optional[ConstraintSolver] = None
        self._prev_covered: int = 0
        self._steps: int = 0

    def reset(self) -> np.ndarray:
        """Start a new game. Returns initial state channels."""
        self.game = MinesweeperGame(self.width, self.height, self.total_mines)

        # Random first click
        r = self.rng.integers(0, self.height)
        c = self.rng.integers(0, self.width)
        self.game.make_move(r, c, MoveType.REVEAL)

        self.solver = ConstraintSolver(self.game)
        self._prev_covered = int(self.game.covered_cells.sum())
        self._steps = 0

        return self._get_state()

    def step(self, move_type: MoveType, r: int, c: int) -> Tuple[np.ndarray, float, bool]:
        """Execute an action. Returns (next_state, reward, done)."""
        if self.game is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        reward = self._compute_reward(move_type, r, c)

        self.game.make_move(r, c, move_type)
        self._steps += 1

        done = self.game.status != GameStatus.PLAYING
        if done:
            if self.game.status == GameStatus.WON:
                reward += self.rewards.win
            # hit_mine is already added in _compute_reward

        self._prev_covered = int(self.game.covered_cells.sum())
        return self._get_state(), reward, done

    def _get_state(self) -> np.ndarray:
        """Return current board state as (10, H, W) channels."""
        if self.game is None:
            raise RuntimeError("Game not initialized.")
        return self.game.board_to_channels()

    def _compute_reward(self, move_type: MoveType, r: int, c: int) -> float:
        """Compute reward for the intended action BEFORE executing it."""
        game = self.game
        if game is None:
            return 0.0

        # Get solver advice (what does logic say?)
        solver_safe, solver_mines = set(), set()
        if self.solver:
            safe, mines = self.solver.find_safe_and_mines()
            solver_safe = set(safe)
            solver_mines = set(mines)

        rwd = 0.0
        cell = (r, c)

        if move_type == MoveType.REVEAL:
            if game.board[r, c] == -1:  # it's a mine
                rwd += self.rewards.hit_mine
            else:
                rwd += self.rewards.reveal_safe
                # Count adjacent mines — if 0, it'll trigger flood fill
                adj_mines = game._count_adjacent_mines(r, c)
                if adj_mines == 0:
                    rwd += self.rewards.reveal_zero

                # Solver bonus
                if cell in solver_safe:
                    rwd += self.rewards.solver_approve_reveal
                elif cell in solver_mines:
                    # Model is revealing a cell solver thinks is a mine
                    # — risky, small penalty
                    rwd -= 1.0

        elif move_type == MoveType.FLAG:
            if game.visible[r, c] == CellState.COVERED:
                if game.board[r, c] == -1:  # correctly flagging a mine
                    rwd += self.rewards.flag_correct
                    if cell in solver_mines:
                        rwd += self.rewards.solver_approve_flag
                else:
                    rwd += self.rewards.flag_wrong

        return rwd

    @property
    def covered_cells(self) -> np.ndarray:
        """Boolean mask of covered cells."""
        if self.game is None:
            return np.zeros((self.height, self.width), dtype=bool)
        return self.game.covered_cells

    @property
    def steps(self) -> int:
        return self._steps
