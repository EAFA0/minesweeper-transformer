"""Tests for the frontier helper used by online BCE training."""

import unittest

import numpy as np

from game.constants import CellState
from training.train import _compute_frontier


class FrontierTest(unittest.TestCase):
    def test_all_covered_has_no_frontier(self):
        visible = np.full((4, 4), CellState.COVERED, dtype=np.int8)
        frontier = _compute_frontier(visible)
        self.assertFalse(frontier.any())

    def test_frontier_is_covered_adjacent_to_revealed(self):
        # Reveal the center cell of a 3x3; all 8 neighbors become frontier.
        visible = np.full((3, 3), CellState.COVERED, dtype=np.int8)
        visible[1, 1] = 0  # revealed "0"
        frontier = _compute_frontier(visible)
        expected = np.ones((3, 3), dtype=bool)
        expected[1, 1] = False  # the revealed cell itself is not frontier
        np.testing.assert_array_equal(frontier, expected)

    def test_revealed_cells_are_never_frontier(self):
        visible = np.full((3, 3), CellState.COVERED, dtype=np.int8)
        visible[0, 0] = 1
        frontier = _compute_frontier(visible)
        self.assertFalse(frontier[0, 0])


if __name__ == "__main__":
    unittest.main()
