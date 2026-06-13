"""Shared training utilities — model construction and forward pass helpers.

Used by both online (train.py) and offline (train_supervised.py) training loops.
"""

import torch
import torch.nn as nn

from config import ModelConfig
from training.checkpoints import save_checkpoint
from training.losses import (
    build_denoising_initial_probs,
    compute_best_safe_ranking_loss,
    compute_loss,
    compute_solver_safe_set_ranking_loss,
)


# ── Model Construction ──────────────────────────────────────────────────────

def build_model(arch: str, model_config: ModelConfig, device: torch.device) -> nn.Module:  # noqa: ARG001
    """Instantiate the model for a given architecture version."""
    from model.architecture import MinesweeperTransformer
    return MinesweeperTransformer(model_config).to(device)


# ── Forward Pass ────────────────────────────────────────────────────────────

def model_forward(
    arch: str,  # noqa: ARG001
    model: nn.Module,
    x: torch.Tensor,
    refine_steps: int,
) -> torch.Tensor:
    """Unified forward pass.

    Returns (B, 1, H, W) sigmoid'd mine probabilities.
    """
    results = model.refine(x, num_steps=refine_steps, return_logits=True)
    raw = results[-1]                     # (B, 1, H, W) raw mine logits
    return torch.sigmoid(raw[:, 0:1])     # (B, 1, H, W) mine probs


def model_forward_logits(
    arch: str,  # noqa: ARG001
    model: nn.Module,
    x: torch.Tensor,
    refine_steps: int,
) -> torch.Tensor:
    """Unified forward pass for BCE loss.

    Returns raw mine logits with shape (B, 1, H, W).
    """
    results = model.refine(x, num_steps=refine_steps, return_logits=True)
    return results[-1][:, 0:1]

__all__ = [
    "build_model",
    "model_forward",
    "model_forward_logits",
    "compute_loss",
    "compute_best_safe_ranking_loss",
    "compute_solver_safe_set_ranking_loss",
    "build_denoising_initial_probs",
    "save_checkpoint",
]
