"""REINFORCE policy gradient fine-tuning for Minesweeper Transformer.

Trains the model to improve win rate through self-play experience.
Warm-starts from supervised probability distillation checkpoint.

Key differences from Phase 1 (supervised):
  - No solver labels — model learns from rewards
  - Stochastic policy (temperature-scaled softmax over P(mine))
  - Can use self-validated boards (solvable) or random boards

Policy:
  π(reveal cell i) ∝ softmax(-P(mine)_i / τ)
  Lower P(mine) → higher probability of being chosen.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model.architecture import MinesweeperTransformer, ModelConfig
from training.rl_env import RLEnv, Rewards


# ─── Config ─────────────────────────────────────────────────────────────────

@dataclass
class RLConfig:
    # Environment
    width: int = 8
    height: int = 8
    total_mines: int = 10
    mine_continue: bool = False
    warmup_clicks: int = 0
    mixed_env: bool = True           # random size + density each episode
    mixed_min_size: int = 6
    mixed_max_size: int = 10
    mixed_min_density: float = 0.10
    mixed_max_density: float = 0.40
    board_pool_path: str = ""  # pre-generate boards for faster RL (recommended)

    # RL hyperparameters
    temperature: float = 1.0
    gamma: float = 0.95
    baseline_ema: float = 0.1

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-4
    games_per_batch: int = 16
    total_games: int = 5000
    grad_clip_norm: float = 1.0

    # Iterative refinement (uses refine() during inference)
    refine_steps: int = 8  # iterative refinement for action selection + gradient

    # Checkpoint
    pretrained_path: str = ""
    save_dir: str = "checkpoints/rl"
    save_every: int = 500
    log_every: int = 100
    eval_every: int = 500
    eval_games: int = 50

    # Device
    device: str = "cpu"


# ─── Policy ─────────────────────────────────────────────────────────────────

def action_log_probs(
    logits_2d: torch.Tensor,      # (H, W) raw logits from model
    covered: torch.Tensor,        # (H, W) bool
    temperature: float,
) -> torch.Tensor:
    """Compute log π(i) for all covered cells.

    π(i) ∝ exp(-logit_i / τ), masked to covered cells only.
    Returns (H, W) log-probabilities.
    """
    H, W = logits_2d.shape
    flat_logits = logits_2d.flatten()
    flat_covered = covered.flatten()

    # Prefer low logits (safe cells)
    policy_logits = torch.where(
        flat_covered,
        -flat_logits / temperature,
        torch.tensor(-float('inf'), device=logits_2d.device),
    )
    log_probs_flat = policy_logits - torch.logsumexp(policy_logits, dim=0)
    return log_probs_flat.reshape(H, W)


# ─── Model output helper ────────────────────────────────────────────────────

def get_logits(
    model: MinesweeperTransformer,
    state: np.ndarray,
    device: str,
) -> torch.Tensor:
    """Get per-cell mine logits for action selection (single pass, fast).

    Trajectory collection uses single-pass inference for speed.
    Refinement is applied during gradient computation only.
    """
    x = torch.from_numpy(state).unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model(x)
    return raw.squeeze(0)[0]  # (H, W) — channel 0 raw logits


# ─── Game Simulation ────────────────────────────────────────────────────────

@torch.no_grad()
def play_game(
    env: RLEnv,
    model: MinesweeperTransformer,
    temperature: float,
    device: str,
    deterministic: bool = False,
    refine_steps: int = 1,
    max_steps: int = 200,
) -> Tuple[float, int, int, int]:
    """Play one game. Returns (total_return, n_steps, win_flag, mine_hits)."""
    state = env.reset()
    total_return = 0.0
    steps = 0

    for _ in range(max_steps):
        covered = env.covered_mask
        if not covered.any():
            break

        logits = get_logits(model, state, device)
        covered_t = torch.from_numpy(covered).to(device)

        if deterministic:
            masked = torch.where(covered_t, logits, torch.tensor(float('inf'), device=device))
            idx = torch.argmin(masked).item()
        else:
            log_probs = action_log_probs(logits, covered_t, temperature)
            probs = torch.exp(log_probs.flatten())
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1).item()

        r, c = divmod(idx, covered.shape[1])  # padded width
        state, reward, done = env.step(r, c)
        total_return += reward
        steps += 1

        if done:
            break

    won = 1 if env.game is not None and env.game.status.value >= 3 else 0
    # GameStatus.WON likely has value 3 (after playing)
    # Actually check GameStatus directly
    from minesweeper.constants import GameStatus
    won = 1 if env.game is not None and env.game.status == GameStatus.WON else 0
    return total_return, steps, won, env.mine_hits


def collect_eval(
    env: RLEnv,
    model: MinesweeperTransformer,
    device: str,
    n_games: int,
    refine_steps: int = 1,
) -> Tuple[float, float, float]:
    """Evaluation: deterministic play. Returns (win_rate, avg_return, avg_steps)."""
    wins = 0
    total_return = 0.0
    total_steps = 0

    for _ in range(n_games):
        r, steps, won, _ = play_game(
            env, model, 0.3, device,
            deterministic=True, refine_steps=refine_steps,
        )
        wins += won
        total_return += r
        total_steps += steps

    return wins / n_games, total_return / n_games, total_steps / n_games


# ─── REINFORCE Update ──────────────────────────────────────────────────────

def reinforce_step(
    model: MinesweeperTransformer,
    optimizer: torch.optim.Optimizer,
    env: RLEnv,
    temperature: float,
    gamma: float,
    baseline: float,
    device: str,
    n_games: int = 8,
    refine_steps: int = 5,
) -> Tuple[float, float, float]:
    """One REINFORCE update. Returns (loss, avg_return, new_baseline).

    Collects n_games of trajectories, computes policy gradient,
    updates model parameters.
    """
    model.train()

    states: List[torch.Tensor] = []
    action_coords: List[Tuple[int, int]] = []
    advantages: List[float] = []
    total_return = 0.0
    n_steps_total = 0

    for _ in range(n_games):
        state = env.reset()
        traj_rew: List[float] = []

        for _ in range(env.max_steps):
            covered = env.covered_mask
            if not covered.any():
                break

            logits = get_logits(model, state, device)
            covered_t = torch.from_numpy(covered).to(device)

            log_probs = action_log_probs(logits, covered_t, temperature)
            probs = torch.exp(log_probs.flatten())
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1).item()
            r, c = divmod(idx, covered.shape[1])  # padded width

            next_state, reward, done = env.step(r, c)

            traj_rew.append(reward)
            states.append(torch.from_numpy(state))
            action_coords.append((r, c))
            n_steps_total += 1

            state = next_state
            if done:
                break

        # Compute returns (G_t = r_t + γ·r_{t+1} + ...)
        G = 0.0
        for reward in reversed(traj_rew):
            G = reward + gamma * G
            advantages.append(G - baseline)
        total_return += sum(traj_rew)

    if n_steps_total == 0:
        return 0.0, 0.0, baseline

    # Advantage normalization (stabilizes REINFORCE)
    adv_tensor = torch.tensor(advantages, dtype=torch.float32)
    adv_mean = adv_tensor.mean()
    adv_std = adv_tensor.std() + 1e-8
    advantages = ((adv_tensor - adv_mean) / adv_std).tolist()

    # Policy gradient: compute logits through refinement (with detach — no BPTT chain)
    optimizer.zero_grad()
    batch_states = torch.stack(states).to(device)

    if refine_steps > 1:
        import random
        B, _, H, W = batch_states.shape
        k = random.randint(1, refine_steps)
        prev = torch.full((B, 1, H, W), 0.5, device=device)
        for _ in range(k):
            raw = model._single_pass(batch_states, prev)
            pred = torch.sigmoid(raw[:, 0:1])
            conf = torch.sigmoid(raw[:, 1:2])
            prev = pred.detach()  # cut BPTT chain — no 8-step graph explosion
        probs = pred  # (N, 1, H, W) — step k P(mine)
        eps = 1e-7
        probs = probs.clamp(eps, 1 - eps)
        logits = torch.log(probs / (1 - probs)).squeeze(1)
    else:
        raw = model(batch_states)  # (N, 2, H, W)
        logits = raw[:, 0]  # (N, H, W) — channel 0 = P(mine) logits

    log_probs_sum = 0.0
    for i, (r, c) in enumerate(action_coords):
        covered_mask = (batch_states[i, 0] == 1).to(device)
        lp = action_log_probs(logits[i], covered_mask, temperature)
        log_probs_sum += lp[r, c] * advantages[i]

    loss = -log_probs_sum / n_steps_total
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    avg_return = total_return / n_games
    new_baseline = baseline * (1 - 0.1) + avg_return * 0.1
    return loss.item(), avg_return, new_baseline


# ─── Full Training Loop ────────────────────────────────────────────────────

def train_rl(config: RLConfig) -> dict:
    """Run REINFORCE training. Returns metrics dict."""
    device = torch.device(config.device)
    print(f"=== RL Fine-tuning (REINFORCE) ===")
    print(f"Board: {config.width}×{config.height}, {config.total_mines} mines")
    print(f"Mine-continue: {config.mine_continue} | Refine: {config.refine_steps} steps")
    print(f"Device: {device}")

    # Load pretrained model
    model = MinesweeperTransformer(ModelConfig()).to(device)
    if config.pretrained_path:
        model.load_pretrained(config.pretrained_path, device=str(device))
        print(f"Loaded pretrained: {model.num_parameters:,} params")
    else:
        print(f"Fresh model: {model.num_parameters:,} params")

    # Board pool (optional, pre-generate for faster RL)
    board_rng = np.random.default_rng(42)
    train_pool = None
    eval_pool = None
    if config.board_pool_path:
        from training.rl_board_pool import RLBoardPool
        p = Path(config.board_pool_path)
        train_pool = RLBoardPool(
            p, min_size=config.mixed_min_size, max_size=config.mixed_max_size,
            min_density=config.mixed_min_density, max_density=config.mixed_max_density,
            target_size=config.total_games, rng=board_rng,
        )
        print(f"Board pool: {train_pool.size}/{config.total_games} boards in {p}")
        if train_pool.size < config.total_games:
            print(f"  Filling pool (this may take a while)...")
            train_pool.fill()
            print(f"  Done — {train_pool.size} boards ready")
        eval_pool = train_pool  # share pool for eval

    # Environments (separate for train/eval to avoid state leaks)
    rng = np.random.default_rng(42)
    train_env = RLEnv(
        width=config.width, height=config.height,
        total_mines=config.total_mines,
        mine_continue=config.mine_continue,
        warmup_clicks=config.warmup_clicks,
        mixed=config.mixed_env,
        mixed_min_size=config.mixed_min_size,
        mixed_max_size=config.mixed_max_size,
        mixed_min_density=config.mixed_min_density,
        mixed_max_density=config.mixed_max_density,
        rng=rng,
        board_pool=train_pool,
    )
    eval_env = RLEnv(
        width=config.width, height=config.height,
        total_mines=config.total_mines,
        mine_continue=False,
        warmup_clicks=config.warmup_clicks,
        mixed=config.mixed_env,
        mixed_min_size=config.mixed_min_size,
        mixed_max_size=config.mixed_max_size,
        mixed_min_density=config.mixed_min_density,
        mixed_max_density=config.mixed_max_density,
        rng=np.random.default_rng(99),
        board_pool=eval_pool,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    # Metrics
    baseline = 0.0
    metrics = {
        "game": [], "loss": [], "avg_return": [],
        "win_rate": [], "eval_win_rate": [],
    }

    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_played = 0

    for batch_start in range(1, config.total_games + 1, config.games_per_batch):
        loss, avg_ret, baseline = reinforce_step(
            model, optimizer, train_env,
            temperature=config.temperature,
            gamma=config.gamma, baseline=baseline,
            device=device, n_games=config.games_per_batch,
            refine_steps=config.refine_steps,
        )

        total_played += config.games_per_batch
        metrics["game"].append(total_played)
        metrics["loss"].append(loss)
        metrics["avg_return"].append(avg_ret)

        if total_played % config.log_every == 0 or total_played <= config.games_per_batch:
            eval_wr, _, _ = collect_eval(
                eval_env, model, device,
                n_games=min(20, config.eval_games),
                refine_steps=config.refine_steps,
            )
            metrics["eval_win_rate"].append(eval_wr)
            elapsed = time.time() - t0
            print(
                f"  Game {total_played:5d}/{config.total_games} | "
                f"loss={loss:.4f} | ret={avg_ret:.1f} | "
                f"eval_wr={eval_wr:.1%} | "
                f"baseline={baseline:.1f} | "
                f"{elapsed:.0f}s"
            )

        if total_played % config.save_every == 0:
            ckpt_path = Path(config.save_dir) / f"rl_model_{total_played}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "game": total_played,
                "baseline": baseline,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Final save
    final_path = Path(config.save_dir) / "rl_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "game": config.total_games,
        "baseline": baseline,
    }, final_path)

    # Final eval
    final_wr, final_ret, final_steps = collect_eval(
        eval_env, model, device,
        n_games=config.eval_games,
        refine_steps=config.refine_steps,
    )
    print(f"\n╔{'═'*58}╗")
    print(f"║  Final: wr={final_wr:.1%}  ret={final_ret:.1f}  steps={final_steps:.0f}")
    print(f"╚{'═'*58}╝")

    metrics["final_win_rate"] = final_wr
    metrics["final_avg_return"] = final_ret
    return metrics
