"""Shared training utilities — model construction, forward pass, loss, checkpointing.

Used by both online (train.py) and offline (train_supervised.py) training loops
to eliminate duplicated arch/loss dispatch logic.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ── Model Construction ──────────────────────────────────────────────────────

def build_model(arch: str, model_config: ModelConfig, device: torch.device) -> nn.Module:
    """Instantiate the model for a given architecture version."""
    if arch == "V1":
        from model.architecture_v1 import MinesweeperTransformerV1
        return MinesweeperTransformerV1(model_config).to(device)
    elif arch == "V1_5":
        from model.architecture_v1_5 import MinesweeperTransformerV1_5
        return MinesweeperTransformerV1_5(model_config).to(device)
    else:
        from model.architecture import MinesweeperTransformer
        return MinesweeperTransformer(model_config).to(device)


# ── Forward Pass ────────────────────────────────────────────────────────────

def model_forward(
    arch: str,
    model: nn.Module,
    x: torch.Tensor,
    refine_steps: int,
) -> torch.Tensor:
    """Unified forward pass across all architectures.

    Returns (B, 1, H, W) sigmoid'd mine probabilities.
    """
    if arch == "V1":
        logits = model.forward(x)            # (B, 1, H, W) raw logits
        return torch.sigmoid(logits)

    elif arch == "V1_5":
        results = model.refine(x, num_steps=refine_steps, return_logits=True)
        raw = results[-1]                     # (B, 2, H, W) raw logits
        return torch.sigmoid(raw[:, 0:1])     # (B, 1, H, W) mine probs

    else:  # V4
        results = model.refine(x, num_steps=refine_steps)
        return results[-1][:, 0:1]            # (B, 1, H, W) sigmoid'd mine probs


# ── Loss Computation ────────────────────────────────────────────────────────

def compute_loss(
    loss_type: str,
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    pos_weight: float | None,
    device: torch.device,
) -> torch.Tensor:
    """Unified loss: BCE (with optional pos_weight) or MSE, masked to covered cells."""
    if loss_type == "bce":
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, device=device)
            weight = torch.where(targets[masks] > 0.5, pw, 1.0)
            return F.binary_cross_entropy(preds[masks], targets[masks], weight=weight)
        return F.binary_cross_entropy(preds[masks], targets[masks])
    else:
        return F.mse_loss(preds[masks], targets[masks])


# ── Checkpointing ───────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path | str,
    fname: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    loss_type: str,
    arch: str,
    epoch: int,
    win_rate: float,
    best_win_rate: float = 0.0,
    best_epoch: int = 0,
    train_loss: list | None = None,
    val_action_accuracy: list | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    data: dict = {
        "epoch": epoch,
        "arch_version": arch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "loss_type": loss_type,
        "best_win_rate": best_win_rate,
        "best_epoch": best_epoch,
        "win_rate": win_rate,
    }
    if train_loss is not None:
        data["train_loss"] = train_loss
    if val_action_accuracy is not None:
        data["val_action_accuracy"] = val_action_accuracy
    if scheduler is not None:
        data["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(data, Path(path) / fname)
