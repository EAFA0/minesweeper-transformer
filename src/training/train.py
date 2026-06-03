"""Training loop for Minesweeper Transformer.

Supports two modes:
- Supervised (MSE): train from pre-generated .npz probability distillation data
- Online BCE: generate self-validated boards on-the-fly, compute BCE loss on frontier

Shared:
- Full BPTT refinement (no detach between steps)
- Common evaluation via training.evaluate
- Checkpoint save/load for both modes
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import POLICY
from game.constants import CellState, GameStatus, MoveType
from game.game import MinesweeperGame
from model.architecture import MinesweeperTransformer, ModelConfig
from training.dataset import MinesweeperDataset
from training.evaluate import evaluate_model as evaluate_game_model
from training.evaluate import load_model, TrainBoardPool


@dataclass
class TrainingConfig:
    """Hyperparameters for training."""
    # Mode
    mode: str = "supervised"      # "supervised" or "online"

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

    # Online BCE settings
    n_games: int = 0              # total games for online BCE (0 = use epochs)
    eval_interval_games: int = 50 # run evaluation every N games
    eval_games: int = 100         # number of eval games per evaluation
    board_pool_path: str = ""     # path to eval board pool .npz
    board_pool_size: int = 64     # training board pool size (disk-cached)
    board_width: int = 8          # board width for online BCE
    board_height: int = 8         # board height for online BCE
    board_mines: int = 10         # board mines for online BCE
    max_game_steps: int = 200     # max steps per game

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
) -> float:
    """Train one epoch. Returns average loss.

    When refinement_steps > 1: Fixed unrolling for `refinement_steps`.
    Only the final step's prediction is used for MSE loss calculation.
    """
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)

    for batch_idx, (channels, probs, mask) in enumerate(dataloader):
        channels = channels.to(device)
        probs = probs.to(device)
        mask = mask.to(device)
        
        B = channels.shape[0]

        optimizer.zero_grad()
        
        # Initial prior (Hidden State Memory + Probs)
        mem_state = torch.zeros((B, model.config.hidden_channels, channels.shape[2], channels.shape[3]), device=device)
        prev_probs = torch.full((B, 1, channels.shape[2], channels.shape[3]), 0.5, device=device)

        # Full BPTT through refinement loop — no detach between steps.
        # Gradient flows through all steps, teaching the model to produce
        # useful intermediate prev_probs and mem_state.
        for step in range(refinement_steps):
            prev_probs, mem_state = model._single_pass(channels, prev_probs, mem_state)

        loss = compute_masked_mse(prev_probs, probs, mask)
        pred_probs = prev_probs

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

        all_outputs = model.refine(channels, num_steps=refinement_steps)
        pred_probs = all_outputs[-1]
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


# ═══════════════════════════════════════════════════════════════════════════
# Online BCE Training (self-validated boards, frontier BCE loss)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_frontier(visible: np.ndarray) -> np.ndarray:
    """Return bool mask: covered cells adjacent to at least one revealed cell.

    Only frontier cells have enough local information for meaningful inference.
    Cells far from the revealed region have no nearby clues to work with.
    """
    H, W = visible.shape
    revealed = visible >= 0  # 0-8 number cells are "revealed"
    frontier = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if not revealed[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and visible[nr, nc] == CellState.COVERED:
                        frontier[nr, nc] = True
    return frontier


def train_online(config: TrainingConfig) -> TrainingMetrics:
    """Online BCE training using self-validated no-guess boards.

    Generates boards on-the-fly. For each game step:
    1. Full BPTT refinement (train mode, no detach)
    2. Action selection (no_grad, lowest P(mine) among covered)
    3. BCE loss on frontier cells with ground-truth mine mask labels
    4. Backprop + optimize

    Periodic evaluation via shared evaluate module.
    """
    device = torch.device(config.device)
    print(f"Training on: {device}")
    print(f"Online BCE — {config.n_games} games, refine={config.refinement_steps}")

    # Model
    model_config = ModelConfig()
    model = MinesweeperTransformer(model_config).to(device)
    start_game = 0
    metrics = TrainingMetrics()

    if config.resume_from:
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_game = ckpt.get("epoch", 0) * config.eval_interval_games
        if "train_loss" in ckpt:
            metrics.train_loss = ckpt["train_loss"]
            metrics.val_loss = ckpt["val_loss"]
            metrics.val_accuracy = ckpt.get("val_accuracy", [])
            metrics.val_action_accuracy = ckpt.get("val_action_accuracy", [])
            metrics.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            metrics.best_epoch = ckpt.get("best_epoch", 0)
        print(f"  Resumed from game ~{start_game}")
    elif config.pretrained:
        print(f"Loading pretrained weights from: {config.pretrained}")
        model.load_pretrained(config.pretrained, device=str(device))

    print(f"Model: {model.num_parameters:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if config.resume_from:
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for pg in optimizer.param_groups:
                pg["lr"] = config.learning_rate

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Board pool for training (disk-backed, survives restarts)
    pool = TrainBoardPool(
        width=config.board_width,
        height=config.board_height,
        mines=config.board_mines,
        pool_size=config.board_pool_size,
    )
    print(f"Training board pool: {pool.available} boards ready ({pool.path})")

    rng = np.random.default_rng(42)
    t0 = time.time()
    best_win_rate = 0.0

    n_games = config.n_games or (config.epochs * config.eval_interval_games)

    # Use eval mode during forward to prevent BatchNorm corruption from
    # batch_size=1 train-mode statistics. Gradients still flow through
    # learnable gamma/beta even in eval mode.
    model.eval()

    for game_idx in range(n_games):
        # Get fresh board from pool
        game = pool.get()
        if game is None or game.status != GameStatus.PLAYING:
            continue

        game_loss = 0.0
        game_steps = 0

        while game.status == GameStatus.PLAYING and game_steps < config.max_game_steps:
            channels = game.board_to_channels()
            ch_t = torch.from_numpy(channels).unsqueeze(0).float().to(device)
            B, _, H, W = ch_t.shape

            # Initial prior
            mem_state = torch.zeros((B, model.config.hidden_channels, H, W), device=device)
            prev_probs = torch.full((B, 1, H, W), 0.5, device=device)

            # Full BPTT refinement
            for step in range(config.refinement_steps):
                prev_probs, mem_state = model._single_pass(ch_t, prev_probs, mem_state)

            pv = prev_probs  # (B, 1, H, W), in computation graph

            # Action selection (no_grad — don't backprop through action logic)
            with torch.no_grad():
                covered = game.covered_cells
                if not covered.any():
                    break
                probs_np = pv[0, 0].cpu().numpy()
                masked = np.where(covered, probs_np, 2.0)
                best_idx = int(np.argmin(masked))
                r, c = divmod(best_idx, W)

            # Compute frontier BCE loss
            frontier = _compute_frontier(game.visible)
            if frontier.any():
                mine_mask = torch.from_numpy(game.get_mine_mask()).float().to(device)
                frontier_t = torch.from_numpy(frontier).bool().to(device)

                probs_frontier = pv[0, 0][frontier_t]
                labels_frontier = mine_mask[frontier_t]

                # All frontier cells contribute (both safe=0 and mine=1)
                bce_loss = nn.functional.binary_cross_entropy(
                    probs_frontier, labels_frontier,
                )
                bce_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()

                game_loss += bce_loss.item()
            else:
                # No frontier — just move
                optimizer.zero_grad()

            # Step game
            is_safe = not game.get_mine_mask()[r, c]
            game.make_move(r, c, MoveType.REVEAL)
            game_steps += 1

        avg_loss = game_loss / max(1, game_steps)
        metrics.train_loss.append(avg_loss)

        # Periodic evaluation
        if (game_idx + 1) % config.eval_interval_games == 0:
            wr, act_acc = _run_eval(
                model, device, config, game_idx, n_games, t0,
            )
            metrics.val_action_accuracy.append(act_acc)

            # Always save latest checkpoint (for partial eval)
            torch.save(
                {
                    "epoch": game_idx + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "win_rate": wr,
                    "action_accuracy": act_acc,
                    "model_config": model_config,
                    "loss_type": "online_bce",
                    "train_loss": metrics.train_loss,
                    "val_action_accuracy": metrics.val_action_accuracy,
                    "best_win_rate": best_win_rate,
                    "best_epoch": metrics.best_epoch,
                },
                save_dir / "latest.pt",
            )

            # Save best separately
            if wr > best_win_rate:
                best_win_rate = wr
                metrics.best_epoch = game_idx + 1
                # Copy latest → best_model.pt
                import shutil
                shutil.copy2(save_dir / "latest.pt", save_dir / "best_model.pt")

        # Print progress
        if (game_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            games_done = game_idx + 1
            print(
                f"  Game {games_done:5d}/{n_games} | "
                f"loss={avg_loss:.4f} | "
                f"best_wr={best_win_rate:.1%} | "
                f"({elapsed:.0f}s)"
            )

    total_time = time.time() - t0
    print(f"\n═══ Training complete in {total_time:.0f}s ═══")
    print(f"Best win rate: {best_win_rate:.2%} at game {metrics.best_epoch}")

    torch.save(
        {
            "epoch": n_games,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "loss_type": "online_bce",
            "train_loss": metrics.train_loss,
            "val_action_accuracy": metrics.val_action_accuracy,
            "best_win_rate": best_win_rate,
            "best_epoch": metrics.best_epoch,
        },
        save_dir / "final_model.pt",
    )

    return metrics


def _run_eval(model, device, config, game_idx, n_games, t0):
    """Run evaluation using shared evaluate module. Returns (win_rate, action_accuracy)."""
    print(f"\n  ── Eval at game {game_idx+1}/{n_games} ──")
    result = evaluate_game_model(
        model,
        device,
        n_games=config.eval_games,
        width=config.board_width,
        height=config.board_height,
        total_mines=config.board_mines,
        seed=42 + game_idx,
        board_pool_path=Path(config.board_pool_path) if config.board_pool_path else None,
        refine_steps=config.refinement_steps,
        quiet=False,
    )
    wr = result["win_rate"]
    acc = result["action_accuracy"]
    elapsed = time.time() - t0
    print(
        f"  Eval: wr={wr:.1%} ({result['won']}/{result['played']}) "
        f"act_acc={acc:.3f} "
        f"stuck={result['stuck']} "
        f"({elapsed:.0f}s total)"
    )
    return wr, acc
