"""Online BCE training — learn from game feedback + solver proof.

Replaces REINFORCE with direct per-cell supervision:
  - BCE on chosen cell (game reveals mine/safe ground truth)
  - MSE on solver-determined cells (P=0 or P=1, mathematically proven)
  - Skip ambiguous cells (P∈(0,1) — solver itself is uncertain)

Key difference from supervised: labels come from the game itself (online),
so training covers states the model actually encounters (on-policy).
"""

from __future__ import annotations

import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

from model.architecture import MinesweeperTransformer, ModelConfig
from minesweeper.probability_solver import ProbabilitySolver
from minesweeper.constants import MoveType, GameStatus, CellState
from data.no_guess import generate_no_guess_board


def train_online(
    model: MinesweeperTransformer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    width: int = 8, height: int = 8, mines: int = 10,
    n_games: int = 5000,
    refine_steps: int = 4,
    lr_scheduler=None,
    save_dir: str = "checkpoints/online",
    eval_every: int = 200,
    eval_games: int = 50,
    temperature: float = 0.0,
    bce_weight: float = 1.0,
    mse_weight: float = 1.0,
    max_attempts: int = 100,
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_games = 0
    best_wr = 0.0

    for game_idx in range(n_games):
        game = generate_no_guess_board(width, height, mines, max_attempts=max_attempts)
        if game is None:
            continue
        total_games += 1

        # ── Play one game, computing loss at each step ──
        game_bce = 0.0
        game_mse = 0.0
        game_det = 0
        game_steps = 0

        optimizer.zero_grad()

        while game.status == GameStatus.PLAYING:
            covered = game.covered_cells
            if not covered.any():
                break

            # Forward pass in eval mode (stable BatchNorm stats for B=1)
            model.eval()
            channels = game.board_to_channels()
            ch_t = torch.from_numpy(channels).float().unsqueeze(0).to(device)

            B, C, H, W = ch_t.shape
            mem = torch.zeros(B, model.config.hidden_channels, H, W, device=device)
            pv = torch.full((B, 1, H, W), 0.5, device=device)

            for step in range(refine_steps):
                pv_old = pv.clone() if step > 0 else None
                pv, mem = model._single_pass(ch_t, pv, mem)
                if step > 0 and pv_old is not None:
                    if (pv - pv_old).abs().max().item() < 0.01:
                        break

            # Switch to train mode for loss and backward
            model.train()

            # ── Pick action (detach from graph) ──
            with torch.no_grad():
                probs_np = pv.squeeze().cpu().numpy()
                if temperature <= 0:
                    masked = np.where(covered, probs_np, 2.0)
                    best = int(np.argmin(masked))
                else:
                    masked_logits = np.where(covered, -probs_np / temperature, -1e10)
                    exp_vals = np.exp(masked_logits - masked_logits.max())
                    probs_flat = exp_vals.flatten() / exp_vals.sum()
                    best = int(np.random.choice(len(probs_flat), p=probs_flat))
            r, c = divmod(best, game.width)

            # ── Solver on CURRENT state (before the move) ──
            solver = ProbabilitySolver(game)
            solver_probs = solver.compute_probabilities()
            determined = (solver_probs == 0.0) | (solver_probs == 1.0)
            n_det = int(determined.sum())

            # ── Reveal ──
            is_mine = (game.board[r, c] == -1)
            game.make_move(r, c, MoveType.REVEAL)

            # ── BCE loss on chosen cell (ground truth from game) ──
            p_chosen = pv[0, 0, r, c]
            label = 1.0 if is_mine else 0.0
            bce = F.binary_cross_entropy(
                p_chosen, torch.tensor(label, device=device)
            )

            # ── MSE loss on solver-determined cells ──
            mse = torch.tensor(0.0, device=device)
            if n_det > 0:
                solver_t = torch.from_numpy(solver_probs.astype(np.float32)).to(device)
                det_t = torch.from_numpy(determined).to(device)
                pred_det = pv[0, 0][det_t]
                target_det = solver_t[det_t]
                mse = F.mse_loss(pred_det, target_det)

            loss = bce_weight * bce + mse_weight * mse
            loss.backward()

            game_bce += bce.item()
            game_mse += mse.item()
            game_det += n_det
            game_steps += 1

            # Handle mine_continue
            if is_mine and game.status == GameStatus.LOST:
                game.visible[r, c] = CellState.FLAGGED
                game.status = GameStatus.PLAYING

        # ── Update after each game ──
        if game_steps > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

        # ── Logging ──
        if total_games % 10 == 0 or total_games == 1:
            avg_bce = game_bce / max(game_steps, 1)
            avg_mse = game_mse / max(game_steps, 1)
            avg_det = game_det / max(game_steps, 1)
            elapsed = time.time() - t0
            print(
                f"  Game {total_games:5d}/{n_games} | "
                f"bce={avg_bce:.4f} mse={avg_mse:.4f} det={avg_det:.0f} "
                f"steps={game_steps:2d} | {elapsed:.0f}s"
            )

        # ── Evaluation ──
        if total_games % eval_every == 0:
            eval_wr = evaluate_model(
                model, width, height, mines, device,
                n_games=eval_games, refine_steps=refine_steps,
            )
            print(f"  ══ Eval @ game {total_games}: wr={eval_wr:.1%} ══")

            if eval_wr >= best_wr:
                best_wr = eval_wr
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": ModelConfig(),
                        "total_games": total_games,
                        "eval_wr": eval_wr,
                    },
                    save_path / "best_model.pt",
                )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "total_games": total_games,
                    "eval_wr": eval_wr,
                },
                save_path / "latest.pt",
            )

    # ── Final ──
    final_wr = evaluate_model(
        model, width, height, mines, device,
        n_games=100, refine_steps=refine_steps,
    )
    print(f"\n╔══════════════════════════════════════════╗")
    print(f"║  Final: wr={final_wr:.1%}  best={best_wr:.1%}  games={total_games}")
    print(f"╚══════════════════════════════════════════╝")

    return final_wr


@torch.no_grad()
def evaluate_model(
    model: MinesweeperTransformer,
    width: int, height: int, mines: int,
    device: torch.device,
    n_games: int = 50,
    refine_steps: int = 4,
) -> float:
    """Evaluate win rate."""
    model.eval()
    wins = 0
    total = 0

    for _ in range(n_games):
        game = generate_no_guess_board(width, height, mines, max_attempts=100)
        if game is None:
            continue
        total += 1

        while game.status == GameStatus.PLAYING:
            covered = game.covered_cells
            if not covered.any():
                break

            channels = game.board_to_channels()
            ch_t = torch.from_numpy(channels).float().unsqueeze(0).to(device)
            B, C, H, W = ch_t.shape
            mem = torch.zeros(B, 64, H, W, device=device)
            pv = torch.full((B, 1, H, W), 0.5, device=device)

            for step in range(refine_steps):
                pv_old = pv.clone() if step > 0 else None
                pv, mem = model._single_pass(ch_t, pv, mem)
                if step > 0 and pv_old is not None:
                    if (pv - pv_old).abs().max().item() < 0.01:
                        break

            probs = pv.squeeze().cpu().numpy()
            masked = np.where(covered, probs, 2.0)
            best = int(np.argmin(masked))
            r, c = divmod(best, game.width)

            game.make_move(r, c, MoveType.REVEAL)

            if game.status == GameStatus.WON:
                wins += 1
                break
            if game.status == GameStatus.LOST:
                break

    return wins / total if total > 0 else 0.0


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online BCE training")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--n_games", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--save_dir", default="checkpoints/online")
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)

    print(f"=== Online BCE @ {args.width}×{args.height}/{args.mines} ===")
    print(f"Device: {dev} | Games: {args.n_games} | LR: {args.lr}")
    print(f"BCE×{args.bce_weight} MSE×{args.mse_weight} τ={args.temperature}")

    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    model = MinesweeperTransformer().to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_games,
    )

    train_online(
        model, optimizer, dev,
        width=args.width, height=args.height, mines=args.mines,
        n_games=args.n_games,
        lr_scheduler=scheduler,
        save_dir=args.save_dir,
        eval_every=args.eval_every,
        temperature=args.temperature,
        bce_weight=args.bce_weight,
        mse_weight=args.mse_weight,
    )
