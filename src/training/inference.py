"""Shared model inference helpers."""

from typing import Tuple

import numpy as np
import torch

from config import POLICY
from game.game import MinesweeperGame
from model.architecture import MinesweeperTransformer


@torch.no_grad()
def predict_mine_probs(
    model: MinesweeperTransformer,
    game: MinesweeperGame,
    device: torch.device,
    refine_steps: int | None = None,
) -> Tuple[np.ndarray, int]:
    """Return mine probabilities as (H, W) numpy array and steps used."""
    if refine_steps is None:
        refine_steps = POLICY.refinement.eval_max_steps

    channels = game.board_to_channels()
    x = torch.from_numpy(channels).unsqueeze(0).to(device)

    if refine_steps <= 1:
        probs = model.predict(x, max_refine_steps=1)
        n_refine_steps = 1
    else:
        refine_results = model.refine(
            x,
            num_steps=refine_steps,
            convergence_epsilon=POLICY.refinement.convergence_eps,
        )
        probs = refine_results[-1]
        n_refine_steps = len(refine_results)

    probs_2d = probs.squeeze(0)
    if probs_2d.dim() == 3:
        probs_2d = probs_2d.squeeze(0)
    return probs_2d.cpu().numpy(), n_refine_steps
