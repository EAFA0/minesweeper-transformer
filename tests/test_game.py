"""Tests for core game logic and channel serialization."""

import unittest

import numpy as np

from game.constants import CellState, GameStatus, NUM_CHANNELS
from game.game import MinesweeperGame, board_state_to_channels


class BoardToChannelsTest(unittest.TestCase):
    def _make_game(self):
        # 3x3 board, single mine at (0, 0). First click at (2, 2) is safe.
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        return MinesweeperGame.from_mine_mask(3, 3, mask, first_r=2, first_c=2)

    def test_shape_and_dtype(self):
        game = self._make_game()
        ch = game.board_to_channels()
        self.assertEqual(ch.shape, (NUM_CHANNELS, 3, 3))
        self.assertEqual(ch.dtype, np.float32)

    def test_covered_channel_matches_visible(self):
        game = self._make_game()
        ch = game.board_to_channels()
        expected_covered = (game.visible == CellState.COVERED).astype(np.float32)
        np.testing.assert_array_equal(ch[0], expected_covered)

    def test_number_one_hot_is_exclusive(self):
        game = self._make_game()
        ch = game.board_to_channels()
        # Each revealed numbered cell sets at most one of channels 2..9.
        onehot_sum = ch[2:10].sum(axis=0)
        self.assertTrue(np.all(onehot_sum <= 1.0))

    def test_revealed_cell_number_channel(self):
        game = self._make_game()
        ch = game.board_to_channels()
        revealed = game.visible >= 0
        for r in range(3):
            for c in range(3):
                if revealed[r, c]:
                    n = int(game.visible[r, c])
                    if 1 <= n <= 8:
                        self.assertEqual(ch[2 + n - 1, r, c], 1.0)


class BoardStateToChannelsEquivalenceTest(unittest.TestCase):
    """The shared builder must reproduce the old per-cell loop byte-for-byte."""

    @staticmethod
    def _legacy_data_path(mask: np.ndarray, mines: np.ndarray) -> np.ndarray:
        H, W = mask.shape
        channels = np.zeros((10, H, W), dtype=np.float32)
        channels[0] = mask
        for r in range(H):
            for c in range(W):
                if not mask[r, c]:
                    rmin, rmax = max(0, r - 1), min(H, r + 2)
                    cmin, cmax = max(0, c - 1), min(W, c + 2)
                    adj = np.sum(mines[rmin:rmax, cmin:cmax])
                    if adj > 0:
                        channels[1 + int(adj), r, c] = 1.0
        return channels

    def test_matches_legacy_loop_on_random_boards(self):
        from training.trajectory_pool import _adjacent_mine_counts

        rng = np.random.default_rng(0)
        for _ in range(50):
            H, W = 8, 8
            mines = (rng.random((H, W)) < 0.2)
            mask = (rng.random((H, W)) < 0.5)  # covered mask
            # Valid trajectories never reveal a mine: mines are always covered.
            mask = mask | mines
            legacy = self._legacy_data_path(mask, mines.astype(np.int64))
            shared = board_state_to_channels(
                covered=mask,
                revealed=~mask,
                numbers=_adjacent_mine_counts(mines),
            )
            np.testing.assert_array_equal(shared, legacy)


class GameLogicTest(unittest.TestCase):
    def test_flood_fill_reveals_zero_region(self):
        # Two opposite-corner mines on a 5x5; revealing the center floods the
        # interior "0" region (more than just the clicked cell) without winning.
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 0] = True
        mask[4, 4] = True
        game = MinesweeperGame.from_mine_mask(5, 5, mask, first_done=True)
        game.make_move(2, 2)
        # Flood fill from the zero region reveals several cells at once.
        self.assertGreater(int((game.visible >= 0).sum()), 1)
        self.assertNotEqual(game.status, GameStatus.LOST)

    def test_hitting_mine_loses(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        game = MinesweeperGame.from_mine_mask(3, 3, mask, first_done=True)
        game.make_move(0, 0)
        self.assertEqual(game.status, GameStatus.LOST)

    def test_revealing_all_safe_cells_wins(self):
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        game = MinesweeperGame.from_mine_mask(2, 2, mask, first_done=True)
        for r, c in [(0, 1), (1, 0), (1, 1)]:
            if game.status == GameStatus.PLAYING:
                game.make_move(r, c)
        self.assertEqual(game.status, GameStatus.WON)

    def test_get_label_mask_is_covered(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        game = MinesweeperGame.from_mine_mask(3, 3, mask, first_r=2, first_c=2)
        np.testing.assert_array_equal(game.get_label_mask(), game.covered_cells)


if __name__ == "__main__":
    unittest.main()
