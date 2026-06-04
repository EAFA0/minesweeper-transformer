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
import torch.nn.functional as F

from config import POLICY, TrainingConfig, TrainingMetrics
from game.constants import CellState, GameStatus, MoveType
from game.probability_solver import ProbabilitySolver
from model.architecture import MinesweeperTransformer, ModelConfig
from training.evaluate import evaluate_model as evaluate_game_model
from training.evaluate import load_model, TrainBoardPool


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
    """Online training: self-validated boards + chosen loss (BCE/MSE) + full BPTT.

    Uses a disk-backed board pool to avoid repeated solver calls.
    Periodic evaluation via shared evaluate module.
    """
    device = torch.device(config.device)
    print(f"Device: {device}")
    print(f"Online {config.loss_type.upper()} — {config.n_games} games, "
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

    # Use train mode: BN statistics adapt to data distribution over time.
    # V4 CNN runs once per forward call, so single-sample BN noise is
    # acceptable and far better than frozen statistics.
    model.train()

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

            # Full BPTT: CNN once → Transformer self-loop → Decoder
            pv, _ = model.forward(ch_t)  # (B, 1, H, W) in computation graph

            # Action selection (no_grad)
            with torch.no_grad():
                covered = game.covered_cells
                if not covered.any():
                    break
                probs_np = pv[0, 0].cpu().numpy()
                masked = np.where(covered, probs_np, 2.0)
                best_idx = int(np.argmin(masked))
                r, c = divmod(best_idx, config.board_width)

            # Loss computation
            if config.loss_type == "bce":
                # BCE loss on frontier cells
                frontier = _compute_frontier(game.visible)
                if frontier.any():
                    mine_mask = torch.from_numpy(game.get_mine_mask()).float().to(device)
                    frontier_t = torch.from_numpy(frontier).bool().to(device)

                    probs_frontier = pv[0, 0][frontier_t]
                    labels_frontier = mine_mask[frontier_t]

                    loss = F.binary_cross_entropy(probs_frontier, labels_frontier)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    game_loss += loss.item()
                else:
                    optimizer.zero_grad()
            elif config.loss_type == "mse":
                # MSE loss on all covered cells using ProbabilitySolver
                covered_t = torch.from_numpy(covered).bool().to(device)
                if covered_t.any():
                    solver = ProbabilitySolver(game)
                    solver_probs = solver.compute_probabilities()
                    solver_t = torch.from_numpy(solver_probs).float().to(device)

                    probs_covered = pv[0, 0][covered_t]
                    targets_covered = solver_t[covered_t]

                    loss = F.mse_loss(probs_covered, targets_covered)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    game_loss += loss.item()
                else:
                    optimizer.zero_grad()
            else:
                raise ValueError(f"Unknown loss_type: {config.loss_type}")

            game.make_move(r, c, MoveType.REVEAL)
            game_steps += 1

        avg_loss = game_loss / max(1, game_steps)
        metrics.train_loss.append(avg_loss)
        scheduler.step()  # cosine decay each game

        # Periodic eval + checkpoint
        if (game_idx + 1) % config.eval_interval_games == 0:
            wr, acc = _run_eval(model, device, config, game_idx + 1, config.n_games, t0)
            metrics.val_action_accuracy.append(acc)

            _save_checkpoint(
                save_dir, "latest.pt",
                model, optimizer, model_config, config, metrics,
                game_idx + 1, best_win_rate, wr, scheduler,
            )

            if wr > best_win_rate:
                best_win_rate = wr
                metrics.best_epoch = game_idx + 1
                metrics.best_win_rate = best_win_rate
                shutil.copy2(save_dir / "latest.pt", save_dir / "best_model.pt")
                print(f"  🏆 New best: {best_win_rate:.1%}")

        if (game_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Game {game_idx+1:5d} | loss={avg_loss:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.1e} | {elapsed:.0f}s")

    total_time = time.time() - t0
    print(f"\n═══ Done in {total_time:.0f}s ═══")
    print(f"Best win rate: {best_win_rate:.2%} at game {metrics.best_epoch}")

    _save_checkpoint(
        save_dir, "final_model.pt",
        model, optimizer, model_config, config, metrics,
        config.n_games, best_win_rate, best_win_rate, scheduler,
    )

    return metrics


def _save_checkpoint(path, fname, model, optimizer, model_config, config, metrics, epoch, best_wr, wr, scheduler=None):
    data = {
        "epoch": epoch,
        "arch_version": "V4",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "loss_type": config.loss_type,
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
        f"avg_steps={result['avg_steps']:.1f} refine={result['avg_refine_steps']:.1f} "
        f"({elapsed:.0f}s)"
    )
    return wr, acc
