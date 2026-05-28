"""Phase 2: REINFORCE policy gradient fine-tuning.

=== What is REINFORCE? ===

REINFORCE is the simplest policy gradient algorithm. Instead of learning from
labeled data (Phase 1), the model learns from its own experience:

  1. Play a game using current policy π(a|s)
  2. If the game went well (won / high score), increase probability of the
     actions that led there. If it went badly, decrease them.
  3. Repeat. The model naturally discovers better strategies.

Key formula:
  loss = -mean( log π(a|s) × (G - baseline) )

Where:
  - π(a|s): probability of taking action a in state s
  - G: actual return (sum of rewards) from that step onward
  - baseline: average return (reduces variance, speeds up learning)

=== Our Policy ===

The model outputs P(mine) for each cell. Our policy:
  π(reveal cell i) ∝ exp(-P(mine)_i / τ)
  
We prefer cells the model thinks are safe (low P(mine)).
τ (temperature) controls exploration: hotter → more random, colder → greedier.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from minesweeper.constants import MoveType, GameStatus
from model.architecture import MinesweeperTransformer, ModelConfig
from training.rl_env import MinesweeperEnv, Rewards


# ─── Config ────────────────────────────────────────────────────────────────

@dataclass
class RLConfig:
    # Environment
    width: int = 8
    height: int = 8
    total_mines: int = 10

    # RL hyperparameters
    temperature: float = 0.3       # exploration (0.3 = mostly greedy)
    gamma: float = 0.95            # discount future rewards slightly
    baseline_ema: float = 0.1      # how fast baseline adapts (0.1 = smooth)

    # Training
    lr: float = 1e-4               # fine-tuning: low LR to preserve Phase 1 knowledge
    weight_decay: float = 1e-4
    games_per_batch: int = 16      # collect N games → one gradient step
    total_games: int = 5000

    # Checkpoint
    pretrained_path: str = "checkpoints/best_model.pt"
    save_dir: str = "checkpoints/rl"
    save_every: int = 500
    log_every: int = 100

    # Device
    device: str = "cpu"


# ─── Policy: P(mine) → action probabilities ────────────────────────────────

def action_log_probs(
    logits: torch.Tensor,          # (H, W) raw logits from model
    covered: torch.Tensor,         # (H, W) bool
    temperature: float,
) -> torch.Tensor:
    """Compute log π(reveal cell i) for all covered cells.

    π(i) = softmax(-logits / τ) over covered cells only.
    Returns (H, W) tensor with log probabilities (0 for non-covered).
    """
    H, W = logits.shape
    flat_logits = logits.flatten()  # (H*W,)
    flat_covered = covered.flatten()

    # Policy: prefer cells with LOW logit (model thinks safe)
    # log π ∝ -logit / τ
    policy_logits = torch.where(
        flat_covered,
        -flat_logits / temperature,
        torch.tensor(-float('inf'), device=logits.device),
    )
    log_probs_flat = policy_logits - torch.logsumexp(policy_logits, dim=0)
    return log_probs_flat.reshape(H, W)


# ─── Game Simulation ────────────────────────────────────────────────────────

@torch.no_grad()
def play_game(
    env: MinesweeperEnv,
    model: MinesweeperTransformer,
    temperature: float,
    device: str,
    deterministic: bool = False,
    max_steps: int = 80,
) -> Tuple[float, int, int]:
    """Play one game. Returns (total_return, n_steps, win_flag).

    If deterministic=True, always pick the cell with lowest P(mine).
    Otherwise, sample from the policy (for exploration during training).
    """
    state = env.reset()
    total_return = 0.0
    steps = 0

    for _ in range(max_steps):
        covered = env.covered_cells
        if not covered.any():
            break

        # Get model predictions
        x = torch.from_numpy(state).unsqueeze(0).to(device)
        logits = model(x).squeeze(0).squeeze(0)  # (H, W)

        covered_t = torch.from_numpy(covered).to(device)

        if deterministic:
            # Greedy: pick lowest P(mine)
            masked = torch.where(covered_t, logits, torch.tensor(float('inf'), device=device))
            idx = torch.argmin(masked).item()
        else:
            # Sample from policy
            log_probs = action_log_probs(logits, covered_t, temperature)
            probs = torch.exp(log_probs.flatten())
            # Renormalize (some numerical error from float32)
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1).item()

        r, c = divmod(idx, env.width)
        state, reward, done = env.step(MoveType.REVEAL, r, c)
        total_return += reward
        steps += 1

        if done:
            break

    won = 1 if env.game is not None and env.game.status == GameStatus.WON else 0
    return total_return, steps, won


def collect_batch(
    env: MinesweeperEnv,
    model: MinesweeperTransformer,
    temperature: float,
    device: str,
    n_games: int,
) -> Tuple[List[float], List[int]]:
    """Play n_games with exploration. Returns (returns, win_flags)."""
    returns = []
    wins = []
    for _ in range(n_games):
        r, _, w = play_game(env, model, temperature, device, deterministic=False)
        returns.append(r)
        wins.append(w)
    return returns, wins


# ─── REINFORCE Update ──────────────────────────────────────────────────────

def reinforce_step(
    model: MinesweeperTransformer,
    optimizer: torch.optim.Optimizer,
    env: MinesweeperEnv,
    temperature: float,
    gamma: float,
    baseline: float,
    device: str,
    n_games: int = 8,
) -> Tuple[float, float]:
    """One REINFORCE update: collect trajectories, compute policy gradient.

    Returns (avg_loss, avg_return).
    """
    model.train()

    all_states: List[torch.Tensor] = []      # (C, H, W) tensors
    all_action_coords: List[Tuple[int, int]] = []
    all_advantages: List[float] = []

    total_return = 0.0
    n_steps = 0

    for _ in range(n_games):
        state = env.reset()
        trajectory_returns: List[float] = []

        # Play one game, recording (state, action, reward)
        for step_i in range(80):
            covered = env.covered_cells
            if not covered.any():
                break

            x = torch.from_numpy(state).unsqueeze(0).to(device)
            logits = model(x).squeeze(0).squeeze(0)  # (H, W)
            covered_t = torch.from_numpy(covered).to(device)

            # Sample action
            log_probs = action_log_probs(logits, covered_t, temperature)
            probs = torch.exp(log_probs.flatten())
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1).item()
            r, c = divmod(idx, env.width)

            state_next, reward, done = env.step(MoveType.REVEAL, r, c)

            trajectory_returns.append(reward)
            all_states.append(torch.from_numpy(state))
            all_action_coords.append((r, c))
            n_steps += 1

            state = state_next
            if done:
                break

        # Compute returns for this trajectory (G_t = r_t + γ·r_{t+1} + ...)
        G = 0.0
        advantages = []
        for reward in reversed(trajectory_returns):
            G = reward + gamma * G
            advantages.append(G - baseline)
        advantages.reverse()

        all_advantages.extend(advantages)
        total_return += sum(trajectory_returns)

    if n_steps == 0:
        return 0.0, 0.0

    # Policy gradient: loss = -mean(log_prob * advantage)
    # We need to compute log π for each (state, action) with gradients
    optimizer.zero_grad()

    batch_states = torch.stack(all_states).to(device)  # (N, C, H, W)
    logits = model(batch_states).squeeze(1)  # (N, H, W)

    log_probs_sum = 0.0
    for i, (r, c) in enumerate(all_action_coords):
        covered_mask = (batch_states[i, 0] == 1).to(device)  # channel 0 = covered
        lp = action_log_probs(logits[i], covered_mask, temperature)
        log_probs_sum += lp[r, c] * all_advantages[i]

    loss = -log_probs_sum / n_steps
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    avg_return = total_return / n_games
    return loss.item(), avg_return


# ─── Full Training Loop ────────────────────────────────────────────────────

def train_rl(config: RLConfig) -> dict:
    """Run REINFORCE training. Returns metrics dict."""
    device = torch.device(config.device)
    print(f"=== Phase 2: REINFORCE Fine-tuning ===")
    print(f"Device: {device}")

    # Load pretrained model
    ckpt = torch.load(config.pretrained_path, map_location=device, weights_only=False)
    model_config = ckpt.get("model_config", ModelConfig())
    model = MinesweeperTransformer(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded pretrained model: {model.num_parameters:,} params")

    # Environment
    rewards = Rewards()
    env = MinesweeperEnv(
        width=config.width, height=config.height,
        total_mines=config.total_mines, rewards=rewards,
        rng=np.random.default_rng(42),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Metrics
    metrics = {
        "loss": [], "avg_return": [], "win_rate": [],
        "baseline": [],
    }

    # Running baseline (EMA of returns)
    baseline = 0.0

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    games_played = 0

    while games_played < config.total_games:
        # REINFORCE update
        loss, avg_return = reinforce_step(
            model, optimizer, env,
            temperature=config.temperature,
            gamma=config.gamma,
            baseline=baseline,
            device=device,
            n_games=config.games_per_batch,
        )

        # Update baseline
        baseline = (1 - config.baseline_ema) * baseline + config.baseline_ema * avg_return

        games_played += config.games_per_batch
        metrics["loss"].append(loss)
        metrics["avg_return"].append(avg_return)
        metrics["baseline"].append(baseline)

        # Log
        if games_played % config.log_every == 0:
            # Evaluate win rate (deterministic play)
            eval_returns = []
            eval_wins = 0
            env_eval = MinesweeperEnv(
                width=config.width, height=config.height,
                total_mines=config.total_mines, rewards=rewards,
                rng=np.random.default_rng(999),
            )
            for _ in range(50):
                _, _, won = play_game(
                    env_eval, model, config.temperature,
                    device, deterministic=True,
                )
                eval_wins += won
            wr = eval_wins / 50

            metrics["win_rate"].append((games_played, wr))

            elapsed = time.time() - t0
            print(
                f"[{games_played:5d}/{config.total_games}] "
                f"loss: {loss:.4f} | "
                f"avg_return: {avg_return:.1f} | "
                f"baseline: {baseline:.1f} | "
                f"win_rate: {wr:.1%} | "
                f"{elapsed:.0f}s"
            )

        # Save checkpoint
        if games_played % config.save_every == 0:
            torch.save({
                "games_played": games_played,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "model_config": model_config,
            }, save_dir / f"rl_checkpoint_{games_played}.pt")

    # Final save
    torch.save({
        "games_played": games_played,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "model_config": model_config,
    }, save_dir / "rl_final.pt")

    # Save metrics
    with open(save_dir / "rl_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete in {time.time() - t0:.0f}s")
    return metrics
