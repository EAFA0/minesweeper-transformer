"""Core Minesweeper game logic — board generation, move execution, state management.

Adapted from gamescomputersplay/minesweeper-solver, stripped down to pure game logic
without GUI/screenshot dependencies.
"""

from typing import Optional, Tuple

import numpy as np

from .constants import (
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_MINES,
    CellState,
    GameStatus,
    MoveType,
)


# Internal board markers
_MINE = -1
_COVERED = 0
_REVEALED_BASE = 1  # revealed cells store number 1-8 as _REVEALED_BASE + count


class MinesweeperGame:
    """Pure Minesweeper game logic — no GUI, no I/O.

    Attributes:
        width, height: Board dimensions.
        total_mines: Total number of mines on the board.
        board: Internal (height, width) array:
               -1 = mine, 0 = covered empty, 1-8 = revealed number.
        visible: (height, width) array — what the player sees:
               CellState.COVERED, CellState.FLAGGED, CellState.EXPLODED, or a number 0-8.
        status: Current GameStatus.
        first_move: Whether the next move is the first move (triggers board generation).
    """

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        total_mines: int = DEFAULT_MINES,
    ):
        self.width = width
        self.height = height
        self.total_mines = total_mines
        self.board: np.ndarray = np.zeros((height, width), dtype=np.int8)
        self.visible: np.ndarray = np.full(
            (height, width), CellState.COVERED, dtype=np.int8
        )
        self.status: GameStatus = GameStatus.PLAYING
        self.first_move: bool = True
        self._mine_positions: Optional[np.ndarray] = None
        # Track remaining covered non-mine cells for win detection
        self._safe_covered: int = width * height - total_mines

    @classmethod
    def from_mine_mask(
        cls, width: int, height: int, mine_mask: np.ndarray,
        first_r: int = 0, first_c: int = 0, first_done: bool = False,
        visible: Optional[np.ndarray] = None,
    ) -> "MinesweeperGame":
        """Create a game with pre-placed mines (for board pool reuse).

        Args:
            width, height: board dimensions
            mine_mask: (height, width) bool array, True=mine
            first_r, first_c: position for first click (ignored if first_done=True)
            first_done: if True, first click already applied
            visible: if provided, restore visible state from saved board
        """
        game = cls.__new__(cls)
        game.width = width
        game.height = height
        mine_count = int(mine_mask.sum())
        game.total_mines = mine_count
        game.board = np.where(mine_mask, _MINE, 0).astype(np.int8)
        game.first_move = False  # mines already placed, skip _generate_board
        game._mine_positions = np.argwhere(mine_mask)
        game._safe_covered = width * height - mine_count
        game.status = GameStatus.PLAYING

        if visible is not None:
            game.visible = visible.copy()
            # Adjust _safe_covered: subtract already-revealed cells
            revealed = (visible >= 0).sum()
            game._safe_covered = width * height - mine_count - revealed
        elif first_done:
            game.visible = np.full((height, width), CellState.COVERED, dtype=np.int8)
        else:
            game.visible = np.full((height, width), CellState.COVERED, dtype=np.int8)
            # Apply first click (manually, since first_move=False skips _generate_board)
            game._reveal(first_r, first_c)

        return game

    # ─── Board Generation ────────────────────────────────────────────────

    def _generate_board(self, safe_r: int, safe_c: int) -> None:
        """Generate mine positions, guaranteeing (safe_r, safe_c) and its neighbors
        are mine-free (standard first-click safety rule).
        """
        safe_zone = set(self._neighbors(safe_r, safe_c))
        safe_zone.add((safe_r, safe_c))

        all_positions = [(r, c) for r in range(self.height) for c in range(self.width)]
        allowed = [p for p in all_positions if p not in safe_zone]

        if len(allowed) < self.total_mines:
            raise ValueError(
                f"Not enough safe positions for {self.total_mines} mines "
                f"on {self.width}x{self.height} board"
            )

        chosen = np.random.choice(len(allowed), self.total_mines, replace=False)
        for idx in chosen:
            r, c = allowed[idx]
            self.board[r, c] = _MINE

        self._mine_positions = np.argwhere(self.board == _MINE)

    # ─── Neighbor Iteration ───────────────────────────────────────────────

      
    def _neighbors(self, r: int, c: int) -> list[Tuple[int, int]]:
        """Return valid 8-directional neighbor coordinates."""
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    out.append((nr, nc))
        return out

    # ─── Move Execution ────────────────────────────────────────────────────

    def make_move(self, r: int, c: int, move_type: MoveType = MoveType.REVEAL) -> bool:
        """Execute a move. Returns True if the move was valid/legal."""
        if self.status != GameStatus.PLAYING:
            return False
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False

        if move_type == MoveType.FLAG:
            return self._flag(r, c)

        # Reveal move
        if self.visible[r, c] >= 0:  # already revealed
            return False
        if self.visible[r, c] == CellState.FLAGGED:
            return False

        # First move: generate board with safety guarantee
        if self.first_move:
            self._generate_board(r, c)
            self.first_move = False

        return self._reveal(r, c)

    def _flag(self, r: int, c: int) -> bool:
        """Toggle flag on a covered cell."""
        if self.visible[r, c] == CellState.COVERED:
            self.visible[r, c] = CellState.FLAGGED
            return True
        elif self.visible[r, c] == CellState.FLAGGED:
            self.visible[r, c] = CellState.COVERED
            return True
        return False

    def _reveal(self, r: int, c: int) -> bool:
        """Reveal cell (r, c). Handles mine hit, flood fill, and win detection."""
        if self.board[r, c] == _MINE:
            # Hit a mine — game over
            self.visible[r, c] = CellState.EXPLODED
            self.status = GameStatus.LOST
            return True

        # Flood fill from empty cells
        self._flood_fill(r, c)
        self._check_win()
        return True

    def _flood_fill(self, r: int, c: int) -> None:
        """Iterative flood fill: reveal empty cells and their numbered borders."""
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if self.visible[cr, cc] >= 0:
                continue  # already revealed

            mine_count = self._count_adjacent_mines(cr, cc)
            self.visible[cr, cc] = mine_count
            self._safe_covered -= 1

            if mine_count == 0:
                for nr, nc in self._neighbors(cr, cc):
                    if self.visible[nr, nc] < 0:  # covered or flagged
                        stack.append((nr, nc))

    def _count_adjacent_mines(self, r: int, c: int) -> int:
        """Count mines in the 8 cells around (r, c)."""
        count = 0
        for nr, nc in self._neighbors(r, c):
            if self.board[nr, nc] == _MINE:
                count += 1
        return count

    def _check_win(self) -> None:
        """Check if all safe cells have been revealed."""
        if self._safe_covered <= 0:
            self.status = GameStatus.WON

    # ─── Query Methods ─────────────────────────────────────────────────────

    @property
    def mine_count(self) -> int:
        """Number of mines remaining (visible indicator: total - flags)."""
        flagged = int(np.sum(self.visible == CellState.FLAGGED))
        return max(0, self.total_mines - flagged)

    @property
    def covered_cells(self) -> np.ndarray:
        """Boolean mask of covered (unrevealed, unflagged) cells."""
        return self.visible == CellState.COVERED

    def get_mine_mask(self) -> np.ndarray:
        """Boolean mask of mine positions (ground truth)."""
        return self.board == _MINE

    # ─── Serialization (for training data) ─────────────────────────────────

    def board_to_channels(self) -> np.ndarray:
        """Convert visible state to model input channels.

        Returns (NUM_CHANNELS, H, W) float32 array:
          [0]: covered mask (1=covered)
          [1]: flagged mask (1=flagged)
          [2:10]: number one-hot (e.g., a cell showing '3' has 1 at channel 4)
        """
        from .constants import CH_COVERED, CH_FLAGGED, CH_NUMBER_BASE, NUM_CHANNELS

        channels = np.zeros((NUM_CHANNELS, self.height, self.width), dtype=np.float32)

        # Covered mask
        channels[CH_COVERED] = (self.visible == CellState.COVERED).astype(np.float32)

        # Flagged mask
        channels[CH_FLAGGED] = (self.visible == CellState.FLAGGED).astype(np.float32)

        # Number one-hot
        revealed_mask = self.visible >= 0
        numbers = self.visible[revealed_mask].astype(np.int64)
        for n in range(1, 9):
            channels[CH_NUMBER_BASE + n - 1][revealed_mask] = (
                numbers == n
            ).astype(np.float32)

        return channels

    def get_labels(self) -> np.ndarray:
        """Return binary labels for covered cells: 1=mine, 0=safe.

        Returns (H, W) float32 array. Already-revealed cells are 0.
        """
        labels = (self.board == _MINE).astype(np.float32)
        # Mask out already revealed cells — they shouldn't contribute to loss
        return labels

    def get_label_mask(self) -> np.ndarray:
        """Return boolean mask: True for cells that should contribute to loss.
        Only covered (unrevealed) cells count.
        """
        return self.visible == CellState.COVERED
