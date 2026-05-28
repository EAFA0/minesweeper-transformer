"""Training loop for Phase 1 supervised learning.

Trains the MinesweeperTransformer to predict P(mine) for covered cells.
Uses BCEWithLogitsLoss with mask to ignore already-revealed cells.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.architecture import MinesweeperTransformer, ModelConfig
from training.dataset import MinesweeperDataset


@dataclass
class TrainingConfig:
    """Hyperparameters for training."""
    # Data
    data_dir: str = "data/training"
    val_ratio: float = 0.1
    augment: bool = True         # D4 data augmentation (rotation + flip)

    # Optimization
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 3e-4   # L2 regularization (was 1e-4)
    epochs: int = 50
    lr_scheduler: str = "cosine"
    grad_clip_norm: float = 1.0  # gradient clipping

    # Loss
    pos_weight: Optional[float] = None

    # Logging
    log_interval: int = 50
    save_dir: str = "checkpoints"
    device: str = "cpu"

    # Curriculum
    pretrained: str = ""          # path to pretrained checkpoint for curriculum transfer

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


@dataclass
class TrainingMetrics:
    """Accumulated training metrics."""
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_accuracy: list = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0


def compute_masked_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCE loss computed only on masked cells.

    Args:
        logits: (B, 1, H, W) raw logits
        labels: (B, H, W) ground truth (0/1)
        mask:   (B, H, W) boolean — True for cells to include
        pos_weight: scalar tensor for mine class weight
    """
    # Squeeze channel dim and select masked cells
    logits_masked = logits.squeeze(1)[mask]   # (N,)
    labels_masked = labels[mask]               # (N,)

    if logits_masked.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits_masked, labels_masked, pos_weight=pos_weight
    )
    return loss


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Accuracy on masked cells.

    Predictions: sigmoid(logits) > 0.5 → mine.
    """
    preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / max(1, total)


def train_epoch(
    model: MinesweeperTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    pos_weight: Optional[torch.Tensor],
    log_interval: int,
    grad_clip_norm: float = 1.0,
) -> float:
    """Train one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)

    for batch_idx, (channels, labels, mask) in enumerate(dataloader):
        channels = channels.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        logits = model(channels)
        loss = compute_masked_bce(logits, labels, mask, pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            acc = compute_accuracy(logits, labels, mask)
            print(
                f"  Batch {batch_idx + 1:4d}/{n_batches} | "
                f"loss: {loss.item():.4f} | acc: {acc:.3f}"
            )

    return total_loss / n_batches


@torch.no_grad()
def validate(
    model: MinesweeperTransformer,
    dataloader: DataLoader,
    device: str,
    pos_weight: Optional[torch.Tensor],
) -> tuple[float, float]:
    """Validate. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_cells = 0

    for channels, labels, mask in dataloader:
        channels = channels.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        logits = model(channels)
        loss = compute_masked_bce(logits, labels, mask, pos_weight)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
        total_correct += (preds[mask] == labels[mask]).sum().item()
        total_cells += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / max(1, total_cells)
    return avg_loss, accuracy


def train(config: TrainingConfig) -> TrainingMetrics:
    """Full training loop. Returns metrics for analysis."""
    device = torch.device(config.device)
    print(f"Training on: {device}")

    # Data
    train_dataset = MinesweeperDataset(
        Path(config.data_dir), split="train", val_ratio=config.val_ratio,
        augment=config.augment,
    )
    val_dataset = MinesweeperDataset(
        Path(config.data_dir), split="val", val_ratio=config.val_ratio,
        augment=False,  # no augmentation on validation
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    print(
        f"Train: {len(train_dataset)} samples (augment={config.augment}) | "
        f"Val: {len(val_dataset)} samples | "
        f"Mine ratio: {train_dataset.mine_ratio:.1%}"
    )

    # Model
    model_config = ModelConfig()
    model = MinesweeperTransformer(model_config).to(device)

    if config.pretrained:
        print(f"Loading pretrained weights from: {config.pretrained}")
        model.load_pretrained(config.pretrained, device=str(device))

    print(f"Model: {model.num_parameters:,} parameters")

    # Loss weight
    pos_weight = config.pos_weight
    if pos_weight is None:
        # Balance mine/safe classes: weight = n_safe / n_mines
        mine_ratio = train_dataset.mine_ratio
        if mine_ratio > 0:
            pos_weight_val = (1 - mine_ratio) / mine_ratio
            print(f"Auto pos_weight: {pos_weight_val:.2f} (safe {1-mine_ratio:.1%} : mine {mine_ratio:.1%})")
        else:
            pos_weight_val = 1.0
        pos_weight = torch.tensor([pos_weight_val], device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    if config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    else:
        scheduler = None

    # Checkpoint dir
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = TrainingMetrics()
    t0 = time.time()

    for epoch in range(1, config.epochs + 1):
        print(f"\n═══ Epoch {epoch}/{config.epochs} ═══")
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, pos_weight,
            config.log_interval, config.grad_clip_norm,
        )
        metrics.train_loss.append(train_loss)

        # Validate
        val_loss, val_acc = validate(model, val_loader, device, pos_weight)
        metrics.val_loss.append(val_loss)
        metrics.val_accuracy.append(val_acc)

        # Scheduler
        if config.lr_scheduler == "plateau" and scheduler:
            scheduler.step(val_loss)
        elif config.lr_scheduler == "cosine" and scheduler:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        status = ""
        if val_loss < metrics.best_val_loss:
            metrics.best_val_loss = val_loss
            metrics.best_epoch = epoch
            status = " ★ BEST"
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "model_config": model_config,
                },
                save_dir / "best_model.pt",
            )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f} | "
            f"LR: {lr:.2e} | "
            f"Time: {epoch_time:.1f}s{status}"
        )

    total_time = time.time() - t0
    print(f"\n═══ Training complete in {total_time:.0f}s ═══")
    print(f"Best val loss: {metrics.best_val_loss:.4f} at epoch {metrics.best_epoch}")

    # Save final model
    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
        },
        save_dir / "final_model.pt",
    )

    # Save metrics
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "train_loss": metrics.train_loss,
                "val_loss": metrics.val_loss,
                "val_accuracy": metrics.val_accuracy,
                "best_val_loss": metrics.best_val_loss,
                "best_epoch": metrics.best_epoch,
            },
            f,
            indent=2,
        )

    return metrics
