"""PPO (Proximal Policy Optimization) for Phase 2 RL fine-tuning.

=== What is PPO? ===

PPO improves on REINFORCE with two key mechanisms:

1. Clip: limits how much the policy can change per update. If the new policy
   probability is too different from the old one, the gradient is clipped.
   This prevents one bad game from destroying all learned knowledge.

2. Critic (value network): learns to predict "how good is this state?".
   The advantage A = actual_return - predicted_value tells us whether an
   action was better or worse than expected. This reduces variance drastically.

=== Policy ===

π(reveal cell i) = softmax(-P(mine)_i / τ), over covered cells only.
τ anneals from 1.0 → 0.3 during training (explore → exploit).

=== Reward ===

  +1   safe reveal       +3   flood fill trigger
  +2   correct flag      -10  hit mine (game ends)
  +100 win
"""

import json
import time
from dataclasses import dataclass, field
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
class PPOConfig:
    width: int = 8
    height: int = 8
    total_mines: int = 10

    # PPO hyperparams
    lr: float = 3e-4
    gamma: float = 0.99          # discount
    gae_lambda: float = 0.95     # GAE smoothing
    clip_epsilon: float = 0.2    # PPO clip range
    value_coef: float = 0.5      # value loss weight
    entropy_coef: float = 0.01   # entropy bonus (encourage exploration)
    max_grad_norm: float = 0.5

    # Training schedule
    temperature_start: float = 1.0
    temperature_end: float = 0.3
    games_per_update: int = 16    # collect N games per PPO step
    ppo_epochs: int = 4           # PPO update epochs per batch
    total_games: int = 20000
    eval_every: int = 500
    eval_games: int = 100

    # Checkpoint
    pretrained: str = ""
    save_dir: str = "checkpoints/S2"
    device: str = "cpu"


# ─── Value Network (Critic) ────────────────────────────────────────────────

class ValueHead(nn.Module):
    """Critic head: predicts state value from transformer features."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, d_model, H, W) → values: (B,)"""
        # Global average pool over spatial dims
        pooled = features.mean(dim=[2, 3])  # (B, d_model)
        return self.net(pooled).squeeze(-1)  # (B,)


class ActorCritic(nn.Module):
    """Actor (transformer) + Critic (value head) for PPO."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.actor = MinesweeperTransformer(config)
        self.critic = ValueHead(config.d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, value)."""
        B, C, H, W = x.shape
        features = self.actor.cnn(x)
        features = self.actor.pos_encoding(features)
        seq = features.flatten(2).transpose(1, 2)
        seq = self.actor.transformer(seq)
        features = seq.transpose(1, 2).reshape(B, self.actor.config.d_model, H, W)

        logits = self.actor.output_head(features)  # (B, 1, H, W)
        value = self.critic(features)               # (B,)
        return logits, value

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_actor_pretrained(self, path: str, device: str = "cpu") -> None:
        """Load Phase 1 weights into actor only."""
        self.actor.load_pretrained(path, device)

    @torch.no_grad()
    def act(self, x: torch.Tensor, covered: torch.Tensor, temperature: float,
            deterministic: bool = False) -> Tuple[int, float, float]:
        """Choose action given state. Returns (flat_idx, log_prob, value)."""
        logits, value = self.forward(x.unsqueeze(0))  # (1,1,H,W), (1,)
        logits = logits.squeeze(0).squeeze(0)  # (H, W)

        # Policy: prefer cells with low P(mine)
        flat = logits.flatten()
        cov = covered.flatten().to(x.device)

        policy_logits = torch.where(
            cov,
            -flat / temperature,
            torch.tensor(-float('inf'), device=x.device),
        )
        log_probs_flat = policy_logits - torch.logsumexp(policy_logits, dim=0)
        probs = torch.exp(log_probs_flat)

        if deterministic:
            idx = torch.argmax(probs).item()
        else:
            idx = torch.multinomial(probs, 1).item()

        log_prob = log_probs_flat[idx].item()
        return idx, log_prob, value.item()


# ─── PPO Update ─────────────────────────────────────────────────────────────

@dataclass
class RolloutStep:
    state: np.ndarray
    action_idx: int
    log_prob: float
    reward: float
    done: bool
    value: float


def compute_gae(
    rewards: List[float], values: List[float], dones: List[bool],
    gamma: float, gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    """Compute Generalized Advantage Estimation (GAE) and returns.

    GAE smooths the advantage estimation, reducing variance.
    Returns (advantages, returns).
    """
    T = len(rewards)
    advantages = [0.0] * T
    returns = [0.0] * T

    gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            next_value = 0.0
            gae = 0.0

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
        next_value = values[t]

    return advantages, returns


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,            # (N, C, H, W)
    covered_masks: torch.Tensor,     # (N, H, W) bool
    actions: torch.Tensor,           # (N,) flat indices
    old_log_probs: torch.Tensor,     # (N,)
    advantages: torch.Tensor,        # (N,)
    returns: torch.Tensor,           # (N,)
    temperature: float,
    config: PPOConfig,
) -> dict:
    """One PPO update epoch. Returns loss components for logging."""
    model.train()

    logits, values = model(states)  # (N,1,H,W), (N,)

    # Recompute log probs for the taken actions
    H, W = states.shape[2], states.shape[3]
    logits_flat = logits.squeeze(1).reshape(states.shape[0], -1)  # (N, H*W)
    cov_flat = covered_masks.reshape(states.shape[0], -1)         # (N, H*W)

    policy_logits = torch.where(
        cov_flat,
        -logits_flat / temperature,
        torch.tensor(-float('inf'), device=states.device),
    )
    log_probs_all = policy_logits - torch.logsumexp(policy_logits, dim=1, keepdim=True)
    action_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    # PPO ratio: π_new / π_old
    ratio = torch.exp(action_log_probs - old_log_probs)

    # Clipped surrogate loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss (MSE)
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus (encourage exploration)
    probs = torch.exp(log_probs_all)
    entropy = -(probs * log_probs_all).sum(dim=1).mean()

    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_clipped": (ratio.abs() > 1 + config.clip_epsilon).float().mean().item(),
    }


# ─── Game Loop ─────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_rollout(
    model: ActorCritic, env: MinesweeperEnv,
    temperature: float, device: str, max_steps: int = 80,
) -> List[RolloutStep]:
    """Play one game, collecting (s, a, logπ, r, done, v) at each step."""
    state = env.reset()
    steps: List[RolloutStep] = []

    for _ in range(max_steps):
        covered = env.covered_cells
        if not covered.any():
            break

        x = torch.from_numpy(state).to(device)
        cov = torch.from_numpy(covered).to(device)

        idx, log_prob, value = model.act(x, cov, temperature)
        r, c = divmod(idx, env.width)

        next_state, reward, done = env.step(MoveType.REVEAL, r, c)

        steps.append(RolloutStep(
            state=state.copy(), action_idx=idx,
            log_prob=log_prob, reward=reward,
            done=done, value=value,
        ))

        state = next_state
        if done:
            break

    return steps


@torch.no_grad()
def evaluate(model: ActorCritic, env: MinesweeperEnv,
             n_games: int, device: str) -> float:
    """Evaluate win rate with deterministic play."""
    model.eval()
    wins = 0
    for _ in range(n_games):
        state = env.reset()
        for _ in range(80):
            covered = env.covered_cells
            if not covered.any():
                wins += 1
                break
            x = torch.from_numpy(state).to(device)
            cov = torch.from_numpy(covered).to(device)
            idx, _, _ = model.act(x, cov, temperature=0.1, deterministic=True)
            r, c = divmod(idx, env.width)
            state, _, done = env.step(MoveType.REVEAL, r, c)
            if done:
                if env.game and env.game.status == GameStatus.WON:
                    wins += 1
                break
    model.train()
    return wins / n_games


# ─── Full Training Loop ────────────────────────────────────────────────────

def train_ppo(config: PPOConfig) -> dict:
    device = torch.device(config.device)
    print(f"=== PPO Training: {config.width}×{config.height} / {config.total_mines} mines ===")
    print(f"Device: {device}")

    # Model
    model_cfg = ModelConfig()
    model = ActorCritic(model_cfg).to(device)

    if config.pretrained:
        print(f"Loading pretrained actor from: {config.pretrained}")
        model.load_actor_pretrained(config.pretrained, str(device))

    print(f"Params: {model.num_parameters:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Environment
    env = MinesweeperEnv(
        width=config.width, height=config.height,
        total_mines=config.total_mines,
        rewards=Rewards(),
        rng=np.random.default_rng(42),
    )

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    history = {"win_rate": [], "policy_loss": [], "value_loss": [],
               "entropy": [], "temperature": []}

    games_played = 0
    t0 = time.time()
    best_wr = 0.0

    while games_played < config.total_games:
        # Temperature annealing
        progress = min(games_played / config.total_games, 1.0)
        temp = config.temperature_start + (config.temperature_end - config.temperature_start) * progress

        # Collect rollouts
        all_steps: List[RolloutStep] = []
        total_reward = 0.0
        for _ in range(config.games_per_update):
            steps = collect_rollout(model, env, temp, device)
            all_steps.extend(steps)
            total_reward += sum(s.reward for s in steps)

        games_played += config.games_per_update

        if not all_steps:
            continue

        # Prepare batch
        N = len(all_steps)
        states = torch.stack([torch.from_numpy(s.state) for s in all_steps]).to(device)
        covered = torch.stack([
            torch.from_numpy((s.state[0] == 1)) for s in all_steps  # ch0 = covered
        ]).to(device)
        actions = torch.tensor([s.action_idx for s in all_steps], dtype=torch.long, device=device)
        old_lps = torch.tensor([s.log_prob for s in all_steps], device=device)
        rewards_list = [s.reward for s in all_steps]
        values_list = [s.value for s in all_steps]
        dones_list = [s.done for s in all_steps]

        # GAE
        advantages, returns = compute_gae(
            rewards_list, values_list, dones_list,
            config.gamma, config.gae_lambda,
        )
        adv_tensor = torch.tensor(advantages, device=device)
        ret_tensor = torch.tensor(returns, device=device)

        # Normalize advantages
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # PPO update (multiple epochs on same data)
        epoch_metrics = []
        for _ in range(config.ppo_epochs):
            m = ppo_update(
                model, optimizer, states, covered, actions,
                old_lps, adv_tensor, ret_tensor, temp, config,
            )
            epoch_metrics.append(m)

        # Log
        avg_m = {k: np.mean([e[k] for e in epoch_metrics]) for k in epoch_metrics[0]}
        avg_return = total_reward / config.games_per_update

        if games_played % config.eval_every == 0:
            env_eval = MinesweeperEnv(
                width=config.width, height=config.height,
                total_mines=config.total_mines,
                rewards=Rewards(),
                rng=np.random.default_rng(999),
            )
            wr = evaluate(model, env_eval, config.eval_games, device)
            history["win_rate"].append((games_played, wr))
            elapsed = time.time() - t0

            # Save best
            if wr > best_wr:
                best_wr = wr
                torch.save({
                    "games_played": games_played,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                }, save_dir / "best_model.pt")

            print(
                f"[{games_played:6d}/{config.total_games}] "
                f"temp: {temp:.2f} | "
                f"ret: {avg_return:.1f} | "
                f"p_loss: {avg_m['policy_loss']:.3f} | "
                f"v_loss: {avg_m['value_loss']:.3f} | "
                f"entropy: {avg_m['entropy']:.3f} | "
                f"win: {wr:.1%} | "
                f"{elapsed:.0f}s"
            )

            history["policy_loss"].append((games_played, avg_m["policy_loss"]))
            history["value_loss"].append((games_played, avg_m["value_loss"]))
            history["entropy"].append((games_played, avg_m["entropy"]))
            history["temperature"].append((games_played, temp))

    # Final save
    torch.save({
        "games_played": config.total_games,
        "model_state_dict": model.state_dict(),
        "config": config,
        "best_win_rate": best_wr,
    }, save_dir / "final_model.pt")

    with open(save_dir / "ppo_metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete in {time.time() - t0:.0f}s | Best WR: {best_wr:.1%}")
    return history
