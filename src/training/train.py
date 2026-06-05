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
from training.trajectory_pool import TrajectoryPool
from training.evaluate import evaluate_model as evaluate_game_model


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

@dataclass
class TrainingContext:
    """Bundles training state to avoid long parameter lists."""
    model: MinesweeperTransformer
    model_config: ModelConfig
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    metrics: TrainingMetrics
    device: torch.device
    save_dir: Path
    arch: str = "V4"
    start_game: int = 0
    best_win_rate: float = 0.0
    t0: float = 0.0

def _setup_training_state(config: TrainingConfig, device: torch.device, arch: str) -> TrainingContext:
    """Initialize model, optimizer, scheduler, and load checkpoints if needed."""
    model_config = ModelConfig()
    
    if arch == "V1":
        from model.architecture_v1 import MinesweeperTransformerV1
        model = MinesweeperTransformerV1(model_config).to(device)
    elif arch == "V1_5":
        from model.architecture_v1_5 import MinesweeperTransformerV1_5
        model = MinesweeperTransformerV1_5(model_config).to(device)
    else:
        model = MinesweeperTransformer(model_config).to(device)
        
    metrics = TrainingMetrics()
    start_game = 0

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
        model.load_pretrained(config.pretrained, device=device)
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

    return TrainingContext(
        model=model,
        model_config=model_config,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        device=device,
        save_dir=Path(config.save_dir),
        arch=arch,
        start_game=start_game,
        best_win_rate=metrics.best_win_rate,
        t0=time.time()
    )


def _compute_loss_and_step(
    config: TrainingConfig, 
    ctx: TrainingContext, 
    game: 'MinesweeperGame', 
    pv: torch.Tensor, 
    covered: np.ndarray
) -> float:
    """Compute the specified loss (BCE/MSE) and perform an optimization step."""
    loss_val = 0.0
    if config.loss_type == "bce":
        # BCE loss on frontier cells
        frontier = _compute_frontier(game.visible)
        if frontier.any():
            mine_mask = torch.from_numpy(game.get_mine_mask()).float().to(ctx.device)
            frontier_t = torch.from_numpy(frontier).bool().to(ctx.device)

            probs_frontier = pv[0, 0][frontier_t]
            labels_frontier = mine_mask[frontier_t]

            loss = F.binary_cross_entropy(probs_frontier, labels_frontier)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), config.grad_clip_norm)
            ctx.optimizer.step()
            ctx.optimizer.zero_grad()
            loss_val = loss.item()
        else:
            ctx.optimizer.zero_grad()
    elif config.loss_type == "mse":
        # MSE loss on FRONTIER cells only
        frontier = _compute_frontier(game.visible)
        if frontier.any():
            frontier_t = torch.from_numpy(frontier).bool().to(ctx.device)
            solver = ProbabilitySolver(game)
            solver_probs = solver.compute_probabilities()
            solver_t = torch.from_numpy(solver_probs).float().to(ctx.device)

            probs_frontier = pv[0, 0][frontier_t]
            targets_frontier = solver_t[frontier_t]

            loss = F.mse_loss(probs_frontier, targets_frontier)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), config.grad_clip_norm)
            ctx.optimizer.step()
            ctx.optimizer.zero_grad()
            loss_val = loss.item()
        else:
            ctx.optimizer.zero_grad()
    else:
        raise ValueError(f"Unknown loss_type: {config.loss_type}")
        
    return loss_val


def _evaluate_and_checkpoint(
    config: TrainingConfig, 
    ctx: TrainingContext, 
    game_idx: int,
    arch: str
) -> float:
    """Run evaluation and save checkpoint if appropriate."""
    wr, acc = _run_eval(ctx.model, ctx.device, config, game_idx + 1, config.n_games, ctx.t0)
    ctx.metrics.val_action_accuracy.append(acc)

    _save_checkpoint(
        ctx.save_dir, "latest.pt",
        ctx.model, ctx.optimizer, ctx.model_config, config, ctx.metrics,
        game_idx + 1, ctx.best_win_rate, wr, arch, ctx.scheduler,
    )

    if wr > ctx.best_win_rate:
        ctx.best_win_rate = wr
        ctx.metrics.best_epoch = game_idx + 1
        ctx.metrics.best_win_rate = ctx.best_win_rate
        shutil.copy2(ctx.save_dir / "latest.pt", ctx.save_dir / "best_model.pt")
        print(f"  🏆 New best: {ctx.best_win_rate:.1%}")

    return ctx.best_win_rate


def _play_training_game(
    config: TrainingConfig, 
    ctx: TrainingContext, 
    game: 'MinesweeperGame'
) -> float:
    """Play one game for training, computing loss and updating weights."""
    game_loss = 0.0
    game_steps = 0

    while game.status == GameStatus.PLAYING and game_steps < config.max_game_steps:
        channels = game.board_to_channels()
        ch_t = torch.from_numpy(channels).unsqueeze(0).float().to(ctx.device)

        # Full BPTT: CNN once → Transformer self-loop → Decoder (or V3 logic)
        if ctx.arch == "V1":
            pv_raw = ctx.model.forward(ch_t)    # V1: (B,1,H,W) logits
            pv = torch.sigmoid(pv_raw)           # (B,1,H,W) probabilities

        elif ctx.arch == "V1_5":
            # V1_5: iterative refinement with full BPTT
            refine_results = ctx.model.refine(
                ch_t, num_steps=config.refinement_steps,
                return_logits=True
            )
            raw = refine_results[-1]             # (B, 2, H, W) raw logits
            probs = torch.sigmoid(raw[:, 0:1])    # (B, 1, H, W) mine probs
            conf = raw[:, 1:2]                    # (B, 1, H, W) conf logit
            pv = torch.cat([probs, conf], dim=1)  # (B, 2, H, W)

        else:
            # V4: iterative refinement with full BPTT (grounding + residual)
            refine_results = ctx.model.refine(ch_t, num_steps=config.refinement_steps)
            # refine() returns sigmoid'd probs for both channels
            probs = refine_results[-1][:, 0:1]  # (B, 1, H, W) mine probs
            conf = refine_results[-1][:, 1:2]   # (B, 1, H, W) conf probs
            pv = torch.cat([probs, conf], dim=1)  # (B, 2, H, W)

        # Action selection (no_grad)
        with torch.no_grad():
            covered = game.covered_cells
            if not covered.any():
                break
            
            if config.loss_type == "mse":
                solver = ProbabilitySolver(game)
                solver_probs = solver.compute_probabilities()
                masked = np.where(covered, solver_probs, 2.0)
            else:
                probs_np = pv[0, 0].cpu().numpy()
                masked = np.where(covered, probs_np, 2.0)
                
            best_idx = int(np.argmin(masked))
            r, c = divmod(best_idx, config.board_width)

        # Loss computation
        loss_val = _compute_loss_and_step(config, ctx, game, pv, covered)
        game_loss += loss_val

        game.make_move(r, c, MoveType.REVEAL)
        game_steps += 1

    return game_loss / max(1, game_steps)


def train(config: TrainingConfig, arch: str = "V4") -> TrainingMetrics:
    """Online training: self-validated boards + chosen loss (BCE/MSE) + full BPTT.

    Uses a disk-backed board pool to avoid repeated solver calls.
    Periodic evaluation via shared evaluate module.
    """
    from utils.device import get_device
    device = get_device(config.device)
    print(f"Device: {device} | Arch: {arch}")
    print(f"Online {config.loss_type.upper()} — {config.n_games} games, "
          f"{config.board_width}×{config.board_height}/{config.board_mines} mines, "
          f"refine={config.refinement_steps}")

    ctx = _setup_training_state(config, device, arch)
    ctx.save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the unified TrajectoryPool for online mode (compute_probs=False for speed)
    pool = TrajectoryPool(
        board_width=config.board_width,
        board_height=config.board_height,
        board_mines=config.board_mines,
        pool_size=config.pool_size,
        pool_workers=config.pool_workers,
        mixed_mode=config.mixed_mode,
        compute_probs=False,
    )
    print("Board pool: TrajectoryPool initialized")

    # Use train mode: BN statistics adapt to data distribution over time.
    # V4 CNN runs once per forward call, so single-sample BN noise is
    # acceptable and far better than frozen statistics.
    ctx.model.train()

    from game.game import MinesweeperGame

    for game_idx in range(ctx.start_game, ctx.start_game + config.n_games):
        # 1. Fetch initial board from pool
        mine_mask, visible = pool.pop()
        
        # 2. Setup game
        game = MinesweeperGame.from_mine_mask(
            config.board_width, config.board_height, mine_mask, first_done=True, visible=visible
        )
        if game is None or game.status != GameStatus.PLAYING:
            continue

        avg_loss = _play_training_game(config, ctx, game)
        
        ctx.metrics.train_loss.append(avg_loss)
        ctx.scheduler.step()  # cosine decay each game

        # Periodic eval + checkpoint
        if (game_idx + 1) % config.eval_interval_games == 0:
            _evaluate_and_checkpoint(config, ctx, game_idx, arch)

        if (game_idx + 1) % 100 == 0:
            elapsed = time.time() - ctx.t0
            print(f"  Game {game_idx+1:5d} | loss={avg_loss:.4f} | "
                  f"lr={ctx.scheduler.get_last_lr()[0]:.1e} | {elapsed:.0f}s")

    total_time = time.time() - ctx.t0
    print(f"\n═══ Done in {total_time:.0f}s ═══")
    print(f"Best win rate: {ctx.best_win_rate:.2%} at game {ctx.metrics.best_epoch}")

    _save_checkpoint(
        ctx.save_dir, "final_model.pt",
        ctx.model, ctx.optimizer, ctx.model_config, config, ctx.metrics,
        config.n_games, ctx.best_win_rate, ctx.best_win_rate, arch, ctx.scheduler,
    )

    return ctx.metrics


def _save_checkpoint(path, fname, model, optimizer, model_config, config, metrics, epoch, best_wr, wr, arch, scheduler=None):
    data = {
        "epoch": epoch,
        "arch_version": arch,
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
