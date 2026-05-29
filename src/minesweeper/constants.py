"""Minesweeper game constants and enums."""

from enum import IntEnum, auto


class CellState(IntEnum):
    """Cell state in the visible board (what the player sees)."""
    COVERED = -1      # 未翻开
    FLAGGED = -2      # 已插旗
    EXPLODED = -3     # 踩雷爆炸


class GameStatus(IntEnum):
    """Overall game status."""
    PLAYING = auto()
    WON = auto()
    LOST = auto()


class MoveType(IntEnum):
    """Type of move the player makes."""
    REVEAL = auto()   # 翻开
    FLAG = auto()     # 插旗 / 取消插旗


# Default board settings
DEFAULT_WIDTH = 8
DEFAULT_HEIGHT = 8
DEFAULT_MINES = 10

# Channel indices for model input
# 未翻开: 1 if cell is covered, 0 otherwise
CH_COVERED = 0
# 已插旗: 1 if cell is flagged, 0 otherwise
CH_FLAGGED = 1
# 数字 one-hot: 1-8 → channels 2-9
CH_NUMBER_BASE = 2
# Total channels: 2 (covered/flagged) + 8 (numbers) = 10
NUM_CHANNELS = 10
