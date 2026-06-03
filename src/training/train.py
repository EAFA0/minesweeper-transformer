"""Online BCE training for Minesweeper Transformer.

Generates self-validated boards from a disk-backed pool, computes BCE loss
on frontier (determined) cells with full BPTT refinement.
"""

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from config import POLICY
from game.constants import CellState, GameStatus, MoveType
from game.game import MinesweeperGame
from model.architecture import MinesweeperTransformer, ModelConfig
from training.evaluate import evaluate_model as evaluate_game_model
from training.evaluate import load_model, TrainBoardPool


@dataclass
class TrainingConfig:
    # Board
    board_width: int = 8
    board_height: int = 8
    board_mines: int = 10
    max_game_steps: int = 200

    # Pool
    board_pool_size: int = 64
    pool_workers: int = 0         # 0 = serial, >=2 = multiprocessing

    # Training
    n_games: int = 5000
    eval_interval_games: int = 200
    eval_games: int = 100
    board_pool_path: str = ""

    # Optimizer
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 3e-4
    grad_clip_norm: float = 1.0

    # Refinement (from POLICY)
    @property
    def refinement_steps(self) -> int:
        return POLICY.refinement.train_max_steps

    # Logging
    log_interval: int = 50
    save_dir: str = "checkpoints"
    device: str = "cpu"

    # Checkpoint
    pretrained: str = ""
    resume_from: str = ""


@dataclass
class TrainingMetrics:
    train_loss: List[float] = field(default_factory=list)
    val_action_accuracy: List[float] = field(default_factory=list)
    best_win_rate: float = 0.0
    best_epoch: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Frontier helper
# ═══════════════════════════════════════════════════════════════════════════

def _compute_frontier(visible: np.ndarray) -> np.ndarray:
    """Return bool mask: covered cells adjacent to at least one revealed cell."""
    H, W = visible.shape
    revealed = visible >= 0
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


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train(config: TrainingConfig) -> TrainingMetrics:
    """Online BCE training: self-validated boards + frontier BCE loss + full BPTT.

    Uses a disk-backed board pool to avoid repeated solver calls.
    Periodic evaluation via shared evaluate module.
    """
    device = torch.device(config.device)
    print(f"Device: {device}")
    print(f"Online BCE — {config.n_games} games, "
          f"{config.board_width}×{config.board_height}/{config.board_mines} mines, "
          f"refine={config.refinement_steps}")

    model_config = ModelConfig()
    model = MinesweeperTransformer(model_config).to(device)
    start_game = 0
    metrics = TrainingMetrics()

    if config.resume_from:
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_game = ckpt.get("epoch", 0)
        metrics.best_win_rate = ckpt.get("best_win_rate", 0.0)
        metrics.best_epoch = ckpt.get("best_epoch", 0)
        if "train_loss" in ckpt:
            metrics.train_loss = ckpt["train_loss"]
            metrics.val_action_accuracy = ckpt.get("val_action_accuracy", [])
        print(f"  Resumed from game {start_game}")
    elif config.pretrained:
        print(f"Loading pretrained: {config.pretrained}")
        model.load_pretrained(config.pretrained, device=str(device))
    else:
        print("Training from scratch (cold start)")

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

    # Cosine LR schedule: decay from initial_lr → min_lr over n_games
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.n_games, eta_min=config.min_lr,
    )
    # Step scheduler forward if resuming mid-training
    for _ in range(start_game):
        scheduler.step()
    print(f"LR schedule: cosine {config.learning_rate:.0e} → {config.min_lr:.0e} over {config.n_games} games")

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Board pool (disk-backed, optional multiprocessing)
    pool = TrainBoardPool(
        width=config.board_width,
        height=config.board_height,
        mines=config.board_mines,
        pool_size=config.board_pool_size,
        num_workers=config.pool_workers,
    )
    print(f"Board pool: {pool.available} boards ({pool.path})")

    # Use eval mode to prevent BatchNorm corruption from batch_size=1
    model.eval()

    t0 = time.time()
    best_win_rate = metrics.best_win_rate

    for game_idx in range(start_game, start_game + config.n_games):
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
            for _step in range(config.refinement_steps):
                prev_probs, mem_state = model._single_pass(ch_t, prev_probs, mem_state)

            pv = prev_probs  # (B, 1, H, W) in computation graph

            # Action selection (no_grad)
            with torch.no_grad():
                covered = game.covered_cells
                if not covered.any():
                    break
                probs_np = pv[0, 0].cpu().numpy()
                masked = np.where(covered, probs_np, 2.0)
                best_idx = int(np.argmin(masked))
                r, c = divmod(best_idx, W)

            # BCE loss on frontier cells
            frontier = _compute_frontier(game.visible)
            if frontier.any():
                mine_mask = torch.from_numpy(game.get_mine_mask()).float().to(device)
                frontier_t = torch.from_numpy(frontier).bool().to(device)

                probs_frontier = pv[0, 0][frontier_t]
                labels_frontier = mine_mask[frontier_t]

                bce_loss = nn.functional.binary_cross_entropy(probs_frontier, labels_frontier)
                bce_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                game_loss += bce_loss.item()
            else:
                optimizer.zero_grad()

            game.make_move(r, c, MoveType.REVEAL)
            game_steps += 1

        avg_loss = game_loss / max(1, game_steps)
        metrics.train_loss.append(avg_loss)
        scheduler.step()  # cosine decay each game

        # Periodic eval + checkpoint
        if (game_idx + 1) % config.eval_interval_games == 0:
            wr, _ = _run_eval(model, device, config, game_idx + 1, config.n_games, t0)

            _save_checkpoint(
                save_dir, "latest.pt",
                model, optimizer, model_config, metrics,
                game_idx + 1, best_win_rate, wr, scheduler,
            )

            if wr > best_win_rate:
                best_win_rate = wr
                metrics.best_epoch = game_idx + 1
                metrics.best_win_rate = best_win_rate
                shutil.copy2(save_dir / "latest.pt", save_dir / "best_model.pt")
                print(f"  🏆 New best: {best_win_rate:.1%}")

        if (game_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Game {game_idx+1:5d} | loss={avg_loss:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.1e} | {elapsed:.0f}s")

    total_time = time.time() - t0
    print(f"\n═══ Done in {total_time:.0f}s ═══")
    print(f"Best win rate: {best_win_rate:.2%} at game {metrics.best_epoch}")

    _save_checkpoint(
        save_dir, "final_model.pt",
        model, optimizer, model_config, metrics,
        config.n_games, best_win_rate, best_win_rate, scheduler,
    )

    return metrics


def _save_checkpoint(path, fname, model, optimizer, model_config, metrics, epoch, best_wr, wr, scheduler=None):
    data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "loss_type": "online_bce",
        "train_loss": metrics.train_loss,
        "val_action_accuracy": metrics.val_action_accuracy,
        "best_win_rate": best_wr,
        "best_epoch": metrics.best_epoch,
        "win_rate": wr,
    }
    if scheduler is not None:
        data["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(data, Path(path) / fname)


def _run_eval(model, device, config, game_idx, n_games, t0):
    print(f"\n  ── Eval at game {game_idx}/{n_games} ──")
    result = evaluate_game_model(
        model, device,
        n_games=config.eval_games,
        width=config.board_width, height=config.board_height,
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
        f"act_acc={acc:.3f} stuck={result['stuck']} "
        f"({elapsed:.0f}s total)"
    )
    return wr, acc
