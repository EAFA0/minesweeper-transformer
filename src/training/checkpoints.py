"""Checkpoint loading and saving utilities."""

from pathlib import Path

import torch
import torch.nn as nn

from config import ModelConfig
from model.architecture import MinesweeperTransformer


def checkpoint_state_dict(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    """Load a checkpoint and return its model state dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return ckpt.get("model_state_dict", ckpt)


def load_model(checkpoint_path: str | Path, device: torch.device) -> MinesweeperTransformer:
    """Load a trained model from checkpoint and switch it to eval mode."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt.get("model_config", ModelConfig())

    model = MinesweeperTransformer(model_config).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


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
    """Save model, optimizer, scheduler, and training metadata."""
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
