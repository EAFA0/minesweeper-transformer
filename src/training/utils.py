"""Shared training utilities — model construction and forward pass helpers.

Used by both online (train.py) and offline (train_supervised.py) training loops.
"""

import torch
import torch.nn as nn

from config import ModelConfig
from training.checkpoints import save_checkpoint
from training.losses import (
    compute_best_safe_ranking_loss,
    compute_loss,
)


# ── Model Construction ──────────────────────────────────────────────────────

def build_model(arch: str, model_config: ModelConfig, device: torch.device) -> nn.Module:  # noqa: ARG001
    """Instantiate the model for a given architecture version."""
    from model.architecture import MinesweeperTransformer
    return MinesweeperTransformer(model_config).to(device)

__all__ = [
    "build_model",
    "compute_loss",
    "compute_best_safe_ranking_loss",
    "save_checkpoint",
]
