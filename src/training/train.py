"""Training loop for probability distillation.

Trains the MinesweeperTransformer to predict P(mine) for covered cells.
Uses MSE loss against solver-computed probability labels.

Supports:
- Fresh training from scratch
- Curriculum transfer (--pretrained: weights only)
- Training resume (--resume: weights + optimizer state + metrics)
- Iterative refinement training (refinement_steps > 1)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import POLICY
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
    weight_decay: float = 3e-4
    epochs: int = 50
    lr_scheduler: str = "cosine"
    grad_clip_norm: float = 1.0

    # Iterative refinement
    refinement_steps: int = POLICY.refinement.train_max_steps
    # Training samples k in [1, refinement_steps] for adaptive refinement.

    # Logging
    log_interval: int = 50
    save_dir: str = "checkpoints"
    device: str = "cpu"

    # Checkpoint
    pretrained: str = ""          # curriculum transfer: load weights only, fresh optimizer
    resume_from: str = ""         # resume training: load weights + optimizer + epoch + metrics

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
    val_action_accuracy: list = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0


def compute_masked_mse(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss computed only on masked (covered) cells."""
    pred_masked = pred_probs.squeeze(1)[mask]
    target_masked = target_probs[mask]

    if pred_masked.numel() == 0:
        return torch.tensor(0.0, device=pred_probs.device, requires_grad=True)

    return nn.functional.mse_loss(pred_masked, target_masked)


def compute_accuracy(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Binary accuracy on masked cells (thresholded at 0.5)."""
    preds = (pred_probs.squeeze(1) > threshold).float()
    targets = (target_probs > threshold).float()
    correct = (preds[mask] == targets[mask]).sum().item()
    total = mask.sum().item()
    return correct / max(1, total)


def compute_action_accuracy(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Action accuracy: would clicking the lowest-P(mine) cell be safe?

    For each sample, finds the covered cell with lowest predicted P(mine),
    then checks if the target P(mine) for that cell is 0.
    """
    B = pred_probs.shape[0]
    correct = 0
    total = 0

    pred_2d = pred_probs.squeeze(1)  # (B, H, W)

    for b in range(B):
        m = mask[b]
        if not m.any():
            continue
        total += 1

        pred_masked = pred_2d[b].clone()
        pred_masked[~m] = float('inf')
        best_idx = torch.argmin(pred_masked.view(-1))

        if target_probs[b].view(-1)[best_idx] == 0.0:
            correct += 1

    return correct / max(1, total)


def train_epoch(
    model: MinesweeperTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    log_interval: int,
    grad_clip_norm: float = 1.0,
    refinement_steps: int = POLICY.refinement.train_max_steps,
    tau_stop: float = 0.95,
) -> float:
    """Train one epoch. Returns average loss.

    When refinement_steps > 1: adaptive training with random step sampling.
    Each batch randomly chooses k ∈ [1, refinement_steps], runs k iterations,
    then computes loss ONLY on step k. A ponder penalty punishes the model
    for running many steps without reaching high confidence.
    """
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)
    import random

    for batch_idx, (channels, probs, mask) in enumerate(dataloader):
        channels = channels.to(device)
        probs = probs.to(device)
        mask = mask.to(device)
        B = channels.shape[0]

        optimizer.zero_grad()

        if refinement_steps > 1:
            # Random step count for this batch
            k = random.randint(1, refinement_steps)

            # Initial prior
            prev = torch.full((B, 1, channels.shape[2], channels.shape[3]), 0.5, device=device)

            # Run k iterations with detach — no BPTT through refinement chain
            for _ in range(k):
                raw = model._single_pass(channels, prev)
                pred = torch.sigmoid(raw[:, 0:1])
                conf = torch.sigmoid(raw[:, 1:2])
                prev = pred.detach()  # cut gradient, save memory

            # Loss on step k only
            prob_loss = compute_masked_mse(pred, probs, mask)
            conf_target = 1.0 - 2.0 * torch.abs(probs.unsqueeze(1) - 0.5)
            conf_loss = compute_masked_mse(conf, conf_target.squeeze(1), mask)

            # Ponder penalty: punish running deep without confidence
            confidence_gap = torch.clamp(conf.new_tensor(tau_stop) - conf, min=0.0)
            ponder_penalty = (k - 1) * (confidence_gap * conf_target).mean()

            loss = prob_loss + 0.3 * conf_loss + 0.1 * ponder_penalty
            pred_probs = pred

        else:
            raw = model(channels)
            pred_probs = torch.sigmoid(raw[:, 0:1])
            loss = compute_masked_mse(pred_probs, probs, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            acc = compute_accuracy(pred_probs, probs, mask)
            act_acc = compute_action_accuracy(pred_probs, probs, mask)
            print(
                f"  Batch {batch_idx + 1:4d}/{n_batches} | "
                f"loss: {loss.item():.4f} | acc: {acc:.3f} | act_acc: {act_acc:.3f}"
            )

    return total_loss / n_batches


@torch.no_grad()
def validate(
    model: MinesweeperTransformer,
    dataloader: DataLoader,
    device: str | torch.device,
    refinement_steps: int = POLICY.refinement.train_max_steps,
) -> tuple[float, float, float]:
    """Validate. Returns (avg_loss, accuracy, action_accuracy).

    When refinement_steps > 1, uses iterative refinement for evaluation.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_action_correct = 0
    total_cells = 0
    total_action_samples = 0

    for channels, probs, mask in dataloader:
        channels = channels.to(device)
        probs = probs.to(device)
        mask = mask.to(device)

        if refinement_steps > 1:
            all_outputs = model.refine(channels, num_steps=refinement_steps)
            pred_probs = all_outputs[-1][:, 0:1]
            loss = compute_masked_mse(pred_probs, probs, mask)
        else:
            raw = model(channels)
            pred_probs = torch.sigmoid(raw[:, 0:1])
            loss = compute_masked_mse(pred_probs, probs, mask)

        total_loss += loss.item()

        preds = (pred_probs.squeeze(1) > 0.5).float()
        targets = (probs > 0.5).float()
        total_correct += (preds[mask] == targets[mask]).sum().item()
        total_cells += mask.sum().item()

        act_acc = compute_action_accuracy(pred_probs, probs, mask)
        total_action_correct += act_acc * mask.shape[0]
        total_action_samples += mask.shape[0]

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / max(1, total_cells)
    action_accuracy = total_action_correct / max(1, total_action_samples)
    return avg_loss, accuracy, action_accuracy


def train(config: TrainingConfig) -> TrainingMetrics:
    """Full training loop. Returns metrics for analysis.

    Supports:
    - Fresh training
    - --pretrained: curriculum transfer (weights only)
    - --resume: continue from checkpoint (weights + optimizer + metrics)
    """
    device = torch.device(config.device)
    print(f"Training on: {device}")
    print(f"Refinement policy: train k=1-{config.refinement_steps}")

    # ── Data ──────────────────────────────────────────────────────────
    train_dataset = MinesweeperDataset(
        Path(config.data_dir), split="train", val_ratio=config.val_ratio,
        augment=config.augment,
    )
    val_dataset = MinesweeperDataset(
        Path(config.data_dir), split="val", val_ratio=config.val_ratio,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    dist = train_dataset.prob_distribution
    print(
        f"Train: {len(train_dataset)} samples (augment={config.augment}) | "
        f"Val: {len(val_dataset)} samples"
    )
    print(
        f"Prob distribution: {dist['frac_deduced_safe']:.1%} safe | "
        f"{dist['frac_deduced_mine']:.1%} mine | "
        f"{dist['frac_ambiguous']:.1%} ambiguous"
    )

    # ── Model ─────────────────────────────────────────────────────────
    model_config = ModelConfig()
    model = MinesweeperTransformer(model_config).to(device)
    start_epoch = 0
    metrics = TrainingMetrics()

    if config.resume_from:
        print(f"Resuming from: {config.resume_from}")
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        # Restore previous metrics to continue the curves
        if "train_loss" in ckpt:
            metrics.train_loss = ckpt["train_loss"]
            metrics.val_loss = ckpt["val_loss"]
            metrics.val_accuracy = ckpt.get("val_accuracy", [])
            metrics.val_action_accuracy = ckpt.get("val_action_accuracy", [])
            metrics.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            metrics.best_epoch = ckpt.get("best_epoch", 0)
        print(
            f"  Resumed from epoch {start_epoch}, "
            f"best val_loss={metrics.best_val_loss:.4f} at epoch {metrics.best_epoch}"
        )
        # Auto-extend: if target epochs <= current, add configured epochs on top
        if config.epochs <= start_epoch:
            add = config.epochs
            config.epochs = start_epoch + add
            print(f"  Auto-extending: +{add} epochs → total {config.epochs}")
    elif config.pretrained:
        print(f"Loading pretrained weights from: {config.pretrained}")
        model.load_pretrained(config.pretrained, device=str(device))

    print(f"Model: {model.num_parameters:,} parameters")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if config.resume_from:
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # Reset LR — old scheduler may have decayed it to near-zero
            for pg in optimizer.param_groups:
                pg["lr"] = config.learning_rate
            print(f"  Optimizer state restored, LR reset to {config.learning_rate:.1e}")

    # ── Scheduler ─────────────────────────────────────────────────────
    remaining_epochs = config.epochs - start_epoch
    if config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, remaining_epochs)
        )
    elif config.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    else:
        scheduler = None

    # ── Save dir ──────────────────────────────────────────────────────
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, config.epochs + 1):
        print(f"\n═══ Epoch {epoch}/{config.epochs} (lr={optimizer.param_groups[0]['lr']:.2e}) ═══")
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            config.log_interval, config.grad_clip_norm,
            refinement_steps=config.refinement_steps,
        )
        metrics.train_loss.append(train_loss)

        # Validate
        val_loss, val_acc, val_act_acc = validate(
            model, val_loader, device,
            refinement_steps=config.refinement_steps,
        )
        metrics.val_loss.append(val_loss)
        metrics.val_accuracy.append(val_acc)
        metrics.val_action_accuracy.append(val_act_acc)

        # Scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        status = ""
        if val_loss < metrics.best_val_loss:
            metrics.best_val_loss = val_loss
            metrics.best_epoch = epoch
            status = " ★ BEST"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_action_accuracy": val_act_acc,
                    "model_config": model_config,
                    "loss_type": "mse",
                    # Save full metric history for potential resume
                    "train_loss": metrics.train_loss,
                    "val_loss_curve": metrics.val_loss,
                    "val_accuracy_curve": metrics.val_accuracy,
                    "val_action_accuracy_curve": metrics.val_action_accuracy,
                    "best_val_loss": metrics.best_val_loss,
                    "best_epoch": metrics.best_epoch,
                },
                save_dir / "best_model.pt",
            )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f} | "
            f"Act Acc: {val_act_acc:.3f} | "
            f"LR: {lr:.2e} | "
            f"Time: {epoch_time:.1f}s{status}"
        )

    total_time = time.time() - t0
    print(f"\n═══ Training complete in {total_time:.0f}s ═══")
    print(f"Best val loss: {metrics.best_val_loss:.4f} at epoch {metrics.best_epoch}")

    # Save final checkpoint (full state for potential resume)
    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "loss_type": "mse",
            "train_loss": metrics.train_loss,
            "val_loss": metrics.val_loss,
            "val_accuracy": metrics.val_accuracy,
            "val_action_accuracy": metrics.val_action_accuracy,
            "best_val_loss": metrics.best_val_loss,
            "best_epoch": metrics.best_epoch,
        },
        save_dir / "final_model.pt",
    )

    # Standalone metrics.json for quick inspection
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "train_loss": metrics.train_loss,
                "val_loss": metrics.val_loss,
                "val_accuracy": metrics.val_accuracy,
                "val_action_accuracy": metrics.val_action_accuracy,
                "best_val_loss": metrics.best_val_loss,
                "best_epoch": metrics.best_epoch,
            },
            f,
            indent=2,
        )

    return metrics
