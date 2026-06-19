"""Tests for the model: ConstraintFeatureBuilder and MinesweeperTransformer."""

import unittest

import torch

from config import ModelConfig
from model.architecture import ConstraintFeatureBuilder, MinesweeperTransformer


def _empty_board(b=1, h=5, w=5):
    """All-covered 10-channel board (covered=1 everywhere, no numbers)."""
    board = torch.zeros(b, 10, h, w)
    board[:, 0] = 1.0  # covered channel
    return board


class ConstraintFeatureBuilderTest(unittest.TestCase):
    def setUp(self):
        self.builder = ConstraintFeatureBuilder()

    def test_output_shape(self):
        board = _empty_board()
        prev = torch.full((1, 1, 5, 5), 0.5)
        out = self.builder(board, prev)
        self.assertEqual(out.shape, (1, 8, 5, 5))

    def test_forced_safe_signal_on_satisfied_constraint(self):
        # 3x3: reveal center as "0" (no adjacent mines) -> neighbors forced safe.
        h = w = 3
        board = torch.zeros(1, 10, h, w)
        board[:, 0] = 1.0
        # center revealed: clear covered, it shows number 0 (no number one-hot set)
        board[0, 0, 1, 1] = 0.0
        prev = torch.zeros(1, 1, h, w)
        out = self.builder(board, prev)
        forced_safe = out[0, 5]  # channel 5 = forced_safe_signal
        # All 8 covered neighbors of the satisfied "0" constraint are forced safe.
        self.assertGreater(forced_safe[0, 0].item(), 0.0)
        self.assertGreater(forced_safe[1, 0].item(), 0.0)


class MinesweeperTransformerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = MinesweeperTransformer(ModelConfig())
        self.model.eval()

    def test_forward_output_shape(self):
        board = _empty_board(b=2, h=8, w=8)
        out = self.model(board)
        self.assertEqual(out.shape, (2, 1, 8, 8))

    def test_refine_returns_requested_steps(self):
        board = _empty_board(b=1, h=8, w=8)
        # In training mode, no early-stop, so we get exactly num_steps.
        self.model.train()
        results = self.model.refine(board, num_steps=4, return_logits=True)
        self.assertEqual(len(results), 4)
        self.assertEqual(results[-1].shape, (1, 1, 8, 8))

    def test_refine_variable_board_size(self):
        # Positional encoding interpolates, so non-16 grids must still work.
        board = _empty_board(b=1, h=10, w=10)
        out = self.model.refine(board, num_steps=2)
        self.assertEqual(out[-1].shape, (1, 1, 10, 10))

    def test_predict_returns_probabilities(self):
        board = _empty_board(b=1, h=8, w=8)
        probs = self.model.predict(board, max_refine_steps=3)
        self.assertEqual(probs.shape, (1, 1, 8, 8))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

    def test_group_norm_variant_trains_at_batch_one(self):
        # GroupNorm is batch-independent: a single-sample train step must work
        # (BatchNorm-with-batch=1 is the motivation for this option).
        cfg = ModelConfig()
        cfg.norm_type = "group"
        model = MinesweeperTransformer(cfg)
        self.assertEqual(model.num_parameters, MinesweeperTransformer(ModelConfig()).num_parameters)
        model.train()
        out = model(_empty_board(b=1, h=8, w=8))
        out.sum().backward()
        self.assertEqual(out.shape, (1, 1, 8, 8))


if __name__ == "__main__":
    unittest.main()
