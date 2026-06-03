"""Supervised distillation training: model learns solver-computed probabilities.

The solver provides soft labels (P(mine) ∈ [0,1]) for every covered cell,
encoding global constraint reasoning the model should internalize.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.architecture import MinesweeperTransformer, ModelConfig
from game.constants import NUM_CHANNELS


@dataclass
class DistillConfig:
    data_dir: str = "data/distill/6x6_18"
    board_width: int = 6
    board_height: int = 6
    board_mines: int = 18
    n_epochs: int = 20
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 3e-4
    grad_clip_norm: float = 1.0
    refinement_steps: int = 4
    hidden_channels: int = 64
    d_model: int = 64
    num_transformer_layers: int = 3
    num_attention_heads: int = 4
    d_ff: int = 256
    eval_interval_epochs: int = 1
    eval_games: int = 200
    save_dir: str = "checkpoints/distill_6x6_18"
    device: str = "auto"
    seed: int = 42


# ── Dataset ─────────────────────────────────────────────────────────────────


class DistillDataset(Dataset):
    """Pre-loads all (channels, probs, masks) into memory."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        files = sorted(self.data_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        all_channels = []
        all_probs = []
        all_masks = []
        for f in files:
            with np.load(f) as data:
                all_channels.append(data["channels"])
                all_probs.append(data["probs"])
                all_masks.append(data["masks"])

        self.channels = torch.from_numpy(np.concatenate(all_channels)).float()
        self.probs = torch.from_numpy(np.concatenate(all_probs)).float()
        self.masks = torch.from_numpy(np.concatenate(all_masks)).float()
        # Collapse 2D probs/masks: (N, H, W)
        if self.probs.ndim == 3:
            pass  # already (N, H, W)
        print(f"DistillDataset: {len(self)} samples, "
              f"ch={list(self.channels.shape)} pr={list(self.probs.shape)}")

    def __len__(self) -> int:
        return len(self.channels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.channels[idx], self.probs[idx], self.masks[idx]


# ── Training ────────────────────────────────────────────────────────────────


def train_distill(config: DistillConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)
    print(f"Device: {device}")

    # Dataset
    dataset = DistillDataset(config.data_dir)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            drop_last=False, num_workers=0,
                            pin_memory=(device.type == "cuda"))
    print(f"Batches/epoch: {len(dataloader)}")

    # Model
    model_config = ModelConfig(
        in_channels=NUM_CHANNELS,
        hidden_channels=config.hidden_channels,
        d_model=config.d_model,
        num_transformer_layers=config.num_transformer_layers,
        num_attention_heads=config.num_attention_heads,
        d_ff=config.d_ff,
        board_height=config.board_height,
        board_width=config.board_width,
        refinement_steps=config.refinement_steps,
    )
    model = MinesweeperTransformer(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.n_epochs * len(dataloader), eta_min=0,
    )

    train_losses: List[float] = []
    val_accs: List[float] = []
    val_wrs: List[float] = []
    best_wr = 0.0
    best_epoch = 0

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Board pool for eval
    board_pool_path = save_dir / "board_pool.npz"

    t0 = time.time()
    global_step = 0

    for epoch in range(1, config.n_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for channels, probs, masks in dataloader:
            channels = channels.to(device)
            probs = probs.to(device)
            masks = masks.to(device)

            model_probs = model(channels)  # (B, H, W)

            mask_bool = masks > 0.5
            if mask_bool.any():
                pred = model_probs[mask_bool]
                target = probs[mask_bool]
                loss = F.binary_cross_entropy(pred, target)
            else:
                loss = torch.tensor(0.0, device=device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Eval
        if epoch % config.eval_interval_epochs == 0:
            model.eval()
            with torch.no_grad():
                result = evaluate_model(
                    model, device,
                    n_games=config.eval_games,
                    width=config.board_width, height=config.board_height,
                    total_mines=config.board_mines,
                    board_pool_path=board_pool_path,
                    refine_steps=config.refinement_steps,
                    quiet=True,
                )
            model.train()

            wr = result["win_rate"]
            acc = result["action_accuracy"]
            val_wrs.append(wr)
            val_accs.append(acc)

            elapsed = time.time() - t0
            best_mark = ""
            if wr > best_wr:
                best_wr = wr
                best_epoch = epoch
                best_mark = " ★BEST"
            print(f"Epoch {epoch:3d}/{config.n_epochs} | loss={avg_loss:.4f} | "
                  f"wr={wr:.2%} | acc={acc:.2%} | best={best_wr:.2%}@e{best_epoch} | "
                  f"{elapsed:.0f}s{best_mark}", flush=True)

            _save_checkpoint(
                save_dir, model, optimizer, scheduler, model_config, config,
                epoch, best_wr, wr, train_losses, val_wrs, val_accs,
            )

    print(f"\nDone. Best WR: {best_wr:.2%} at epoch {best_epoch}")
    return model


def _save_checkpoint(save_dir, model, optimizer, scheduler, model_config,
                     train_config, epoch, best_wr, wr,
                     train_losses, val_wrs, val_accs):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model_config.__dict__,
        "epoch": epoch,
        "best_win_rate": best_wr,
        "win_rate": wr,
        "train_losses": train_losses,
        "val_win_rates": val_wrs,
        "val_action_accuracies": val_accs,
        "loss_type": "distill_bce",
        "arch_version": "V4",
    }
    torch.save(checkpoint, save_dir / "latest.pt")
    if wr >= best_wr:
        torch.save(checkpoint, save_dir / "best.pt")
