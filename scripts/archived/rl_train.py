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

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch

from config import POLICY
from model.architecture import MinesweeperTransformer, ModelConfig
from training.rl_env import RLEnv, Rewards
from training.rl_board_pool import RLBoardPool
from minesweeper.constants import GameStatus
from minesweeper.probability_solver import ProbabilitySolver


# ─── Config ─────────────────────────────────────────────────────────────────

@dataclass
class RLConfig:
    # Environment
    width: int = 8
    height: int = 8
    total_mines: int = 10
    mine_continue: bool = True   # True = learn from mine patterns via flag channel
    warmup_clicks: int = 0
    mixed_env: bool = False          # random size + density each episode
    mixed_min_size: int = 6
    mixed_max_size: int = 10
    mixed_min_density: float = 0.10
    mixed_max_density: float = 0.40
    board_pool_path: str = ""  # pre-generate boards for faster RL (recommended)

    # Reward shaping
    reward_reveal_safe: float = POLICY.rl_rewards.reveal_safe
    reward_floodfill_bonus: float = POLICY.rl_rewards.floodfill_bonus
    reward_hit_mine: float = POLICY.rl_rewards.hit_mine
    reward_step_penalty: float = POLICY.rl_rewards.step_penalty

    # RL hyperparameters
    temperature: float = 1.0
    entropy_coef: float = 0.0  # pretrained RL fine-tuning does not need forced exploration

    # Conservative RL: MSE anchoring prevents catastrophic forgetting
    conservative_alpha: float = 0.9   # initial weight for MSE loss (1.0 = pure supervised)
    alpha_decay: float = 0.9999       # per-batch decay: α ← α × decay

    # Temperature annealing: start high (explore) → low (exploit)
    temperature_min: float = 0.05
    temperature_decay: float = 0.999  # per-batch decay

    # Architecture overrides (RL-specific)
    dropout: float = 0.0  # RL doesn't need dropout — pretrained weights are already converged

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-4
    games_per_batch: int = 16
    total_games: int = 5000
    grad_clip_norm: float = 1.0

    # Iterative refinement (uses refine() during inference)
    refine_steps: int = POLICY.refinement.rl_steps
    # Rollout and gradient recomputation must use the same refinement count.

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
    device: str | torch.device,
    refine_steps: int = POLICY.refinement.rl_steps,
) -> torch.Tensor:
    """Get per-cell mine logits for action selection.

    When refine_steps > 1, runs the refinement loop and converts
    probabilities back to logits for softmax action selection.
    """
    x = torch.from_numpy(state).unsqueeze(0).to(device)
    model.eval()  # CRITICAL: eval mode so BatchNorm uses running stats (not per-sample B=1 stats)
    with torch.no_grad():
        if refine_steps > 1:
            results = model.refine(x, num_steps=refine_steps)
            probs = results[-1][:, 0:1]
            eps = 1e-7
            probs = probs.clamp(eps, 1 - eps)
            logits = torch.log(probs / (1 - probs))
            return logits.squeeze(0)[0]
        else:
            probs, _ = model(x)                # V3 returns (probs, mem_state)
            eps = 1e-7
            probs = probs.clamp(eps, 1 - eps)
            logits = torch.log(probs / (1 - probs))
            return logits.squeeze(0)[0]         # (H, W) logits


# ─── Game Simulation ────────────────────────────────────────────────────────

@torch.no_grad()
def play_game(
    env: RLEnv,
    model: MinesweeperTransformer,
    temperature: float,
    device: str | torch.device,
    deterministic: bool = False,
    refine_steps: int = POLICY.refinement.rl_steps,
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

        logits = get_logits(model, state, device, refine_steps=refine_steps)
        covered_t = torch.from_numpy(covered).to(device)

        if deterministic:
            masked = torch.where(covered_t, logits, torch.tensor(float('inf'), device=device))
            idx = int(torch.argmin(masked).item())
        else:
            log_probs = action_log_probs(logits, covered_t, temperature)
            probs = torch.exp(log_probs.flatten())
            probs = probs / probs.sum()
            idx = int(torch.multinomial(probs, 1).item())

        r, c = divmod(idx, covered.shape[1])  # padded width
        state, reward, done = env.step(r, c)
        total_return += reward
        steps += 1

        if done:
            break

    won = 1 if env.game is not None and env.game.status == GameStatus.WON else 0
    return total_return, steps, won, env.mine_hits


def collect_eval(
    env: RLEnv,
    model: MinesweeperTransformer,
    device: str | torch.device,
    n_games: int,
    refine_steps: int = POLICY.refinement.rl_steps,
) -> Tuple[float, float, float]:
    """Evaluation: deterministic play. Returns (win_rate, avg_return, avg_steps)."""
    model.eval()  # defense-in-depth: ensure BatchNorm uses running stats
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


# ─── MSE Anchoring (Conservative RL) ──────────────────────────────────────

@torch.no_grad()
def compute_solver_targets(
    states: torch.Tensor,  # (N, C, H, W) channel tensors
    game: object,           # MinesweeperGame (current state)
) -> torch.Tensor:
    """Run ProbabilitySolver to get ground-truth P(mine) for each state.

    Works with self-validated boards where solver provides accurate labels.
    Returns (N, H, W) target probability tensor.
    """
    import numpy as np
    N, _, H, W = states.shape
    targets = torch.zeros(N, H, W)

    for i in range(N):
        solver = ProbabilitySolver(game)
        try:
            probs = solver.compute_probabilities()
            targets[i] = torch.from_numpy(probs.astype(np.float32))
        except Exception:
            targets[i] = 0.5  # fallback for solver errors

    return targets


# ─── REINFORCE Update ──────────────────────────────────────────────────────

def reinforce_step(
    model: MinesweeperTransformer,
    optimizer: torch.optim.Optimizer,
    env: RLEnv,
    temperature: float,
    entropy_coef: float,
    baseline: float,
    device: str | torch.device,
    n_games: int = 8,
    refine_steps: int = POLICY.refinement.rl_steps,
    conservative_alpha: float = 0.0,  # 0=pure RL, 1=pure supervised
) -> Tuple[float, float, float, float, float]:
    """One REINFORCE update with optional MSE anchoring.

    Returns (total_loss, rl_loss, mse_loss, avg_return, new_baseline).

    When conservative_alpha > 0, adds MSE loss against solver labels
    to prevent catastrophic forgetting.
    """
    model.eval()

    actual_refine = refine_steps if refine_steps > 1 else 1

    states: List[torch.Tensor] = []
    covered_masks: List[torch.Tensor] = []
    action_coords: List[Tuple[int, int]] = []
    advantages: List[float] = []
    solver_targets: List[torch.Tensor] = []  # for MSE anchoring
    total_return = 0.0
    n_steps_total = 0

    for _ in range(n_games):
        state = env.reset()
        game_return = 0.0

        for _ in range(env.max_steps):
            covered = env.covered_mask
            if not covered.any():
                break

            logits = get_logits(model, state, device, refine_steps=actual_refine)
            covered_t = torch.from_numpy(covered).to(device)

            log_probs = action_log_probs(logits, covered_t, temperature)
            probs = torch.exp(log_probs.flatten())
            probs = probs / probs.sum()
            idx = int(torch.multinomial(probs, 1).item())
            r, c = divmod(idx, covered.shape[1])

            # Collect solver labels BEFORE the move (current state)
            if conservative_alpha > 0 and env.game is not None:
                target = env.solver_probs()
                solver_targets.append(torch.from_numpy(target))

            next_state, reward, done = env.step(r, c)
            game_return += reward

            states.append(torch.from_numpy(state))
            covered_masks.append(covered_t)
            action_coords.append((r, c))
            advantages.append(reward)
            n_steps_total += 1

            state = next_state
            if done:
                break

        total_return += game_return

    if n_steps_total == 0:
        return 0.0, 0.0, 0.0, 0.0, baseline

    # Advantage Normalization
    adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    if len(adv_tensor) > 1:
        adv_mean = adv_tensor.mean()
        adv_std = adv_tensor.std()
        adv_t = (adv_tensor - adv_mean) / (adv_std + 1e-8)
    else:
        adv_t = adv_tensor - adv_tensor.mean()

    optimizer.zero_grad()
    batch_states = torch.stack(states).to(device)
    covered_masks_t = torch.stack(covered_masks).to(device)
    r_coords = torch.tensor([r for r, c in action_coords], device=device)
    c_coords = torch.tensor([c for r, c in action_coords], device=device)

    # ── REINFORCE loss ──
    chunk_size = 64
    total_rl_loss = 0.0

    for i in range(0, n_steps_total, chunk_size):
        chunk_states = batch_states[i:i+chunk_size]
        chunk_covered = covered_masks_t[i:i+chunk_size]
        chunk_adv = adv_t[i:i+chunk_size]
        chunk_r = r_coords[i:i+chunk_size]
        chunk_c = c_coords[i:i+chunk_size]

        B, _, H, W = chunk_states.shape

        if actual_refine > 1:
            prev = torch.full((B, 1, H, W), 0.5, device=device)
            mem = torch.zeros(B, model.config.hidden_channels, H, W, device=device)
            for _ in range(actual_refine):
                prev, mem = model._single_pass(chunk_states, prev, mem)
            probs = prev
            eps = 1e-7
            probs = probs.clamp(eps, 1 - eps)
            chunk_logits = torch.log(probs / (1 - probs)).squeeze(1)
        else:
            probs, _ = model(chunk_states)
            eps = 1e-7
            probs = probs.clamp(eps, 1 - eps)
            chunk_logits = torch.log(probs / (1 - probs)).squeeze(1)

        flat_logits = chunk_logits.view(B, -1)
        flat_covered = chunk_covered.view(B, -1)

        policy_logits = torch.where(
            flat_covered,
            -flat_logits / temperature,
            torch.tensor(-float('inf'), device=device)
        )
        log_probs_flat = policy_logits - torch.logsumexp(policy_logits, dim=1, keepdim=True)
        log_probs = log_probs_flat.view(B, H, W)

        chosen_log_probs = log_probs[torch.arange(B, device=device), chunk_r, chunk_c]
        policy_loss = -(chosen_log_probs * chunk_adv).sum() / n_steps_total
        policy_loss.backward()
        total_rl_loss += policy_loss.item()

    # ── MSE anchoring loss ──
    total_mse_loss = 0.0
    if conservative_alpha > 0 and len(solver_targets) > 0:
        # Recompute model outputs with fresh forward pass for MSE
        # (detached from RL graph to keep gradients separate)
        solver_t = torch.stack(solver_targets).to(device)
        N_mse = solver_t.shape[0]
        # Process in chunks to save memory
        mse_chunk = 64
        for i in range(0, N_mse, mse_chunk):
            end = min(i + mse_chunk, N_mse)
            chunk_s = batch_states[i:end]
            chunk_mask = covered_masks_t[i:end]
            chunk_target = solver_t[i:end]

            Bc, _, Hc, Wc = chunk_s.shape
            if actual_refine > 1:
                pv = torch.full((Bc, 1, Hc, Wc), 0.5, device=device)
                mem = torch.zeros(Bc, model.config.hidden_channels, Hc, Wc, device=device)
                for _ in range(actual_refine):
                    pv, mem = model._single_pass(chunk_s, pv, mem)
                pred_probs = pv
            else:
                raw = model(chunk_s)
                pred_probs = torch.sigmoid(raw[:, 0:1])

            chunk_target = chunk_target.unsqueeze(1)
            # Squeeze mask to match pred_probs shape for indexing
            mask_bool = chunk_mask.bool().unsqueeze(1)  # (B, 1, H, W)
            mse = torch.nn.functional.mse_loss(
                pred_probs[mask_bool],
                chunk_target[mask_bool],
            )
            (conservative_alpha * mse).backward()
            total_mse_loss += mse.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    total_loss = total_rl_loss + total_mse_loss
    avg_return = total_return / n_games
    new_baseline = baseline * (1 - 0.1) + avg_return * 0.1
    return total_loss, total_rl_loss, total_mse_loss, avg_return, new_baseline


# ─── Full Training Loop ────────────────────────────────────────────────────

def train_rl(config: RLConfig) -> dict:
    """Run REINFORCE training. Returns metrics dict."""
    device = torch.device(config.device)
    print("=== RL Fine-tuning (REINFORCE) ===")
    print(f"Board: {config.width}×{config.height}, {config.total_mines} mines")
    print(f"Mine-continue: {config.mine_continue} | Refine: {config.refine_steps} steps")
    print(
        f"Rewards: safe={config.reward_reveal_safe}, "
        f"floodfill_bonus={config.reward_floodfill_bonus}, "
        f"mine={config.reward_hit_mine}, step={config.reward_step_penalty}"
    )
    if config.conservative_alpha > 0:
        print(f"Conservative RL: α={config.conservative_alpha}, decay={config.alpha_decay}")
    print(f"Temperature: {config.temperature} → min {config.temperature_min} (decay {config.temperature_decay})")
    print(f"Device: {device}")

    # Load pretrained model
    model = MinesweeperTransformer(ModelConfig()).to(device)
    if config.pretrained_path:
        model.load_pretrained(config.pretrained_path, device=str(device))
        print(f"Loaded pretrained: {model.num_parameters:,} params")
    else:
        print(f"Fresh model: {model.num_parameters:,} params")

    # Disable dropout for RL — pretrained weights are already converged
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = config.dropout
    print(f"Dropout set to {config.dropout}")

    # Board pool (optional, read-only by design; build it via scripts/generate_rl_pool.py)
    train_pool = None
    eval_pool = None
    if config.board_pool_path:
        p = Path(config.board_pool_path)
        train_pool = RLBoardPool(p)
        print(f"Board pool: {train_pool.size} boards loaded from {p}")
        if train_pool.size == 0:
            raise RuntimeError(
                f"Board pool is empty: {p}. "
                f"Build it first with scripts/generate_rl_pool.py or pass --no_board_pool."
            )
        elif train_pool.size < config.total_games:
            print(f"  Pool ({train_pool.size}) < total_games ({config.total_games}) — boards will be reused.")
        eval_pool = train_pool  # share pool for eval

    # Environments (separate for train/eval to avoid state leaks)
    rng = np.random.default_rng(42)
    rewards = Rewards(
        reveal_safe=config.reward_reveal_safe,
        floodfill_bonus=config.reward_floodfill_bonus,
        hit_mine=config.reward_hit_mine,
        step_penalty=config.reward_step_penalty,
    )
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
        rewards=rewards,
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
        rewards=rewards,
        rng=np.random.default_rng(99),
        board_pool=eval_pool,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    # Metrics
    baseline = 0.0
    alpha = config.conservative_alpha
    temperature = config.temperature  # current temperature for annealing
    metrics: dict[str, Any] = {
        "game": [], "loss": [], "rl_loss": [], "mse_loss": [],
        "avg_return": [], "win_rate": [], "eval_win_rate": [],
        "alpha": [], "temperature": [],
    }

    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_played = 0

    for batch_start in range(1, config.total_games + 1, config.games_per_batch):
        loss, rl_loss, mse_loss, avg_ret, baseline = reinforce_step(
            model, optimizer, train_env,
            temperature=temperature,
            entropy_coef=config.entropy_coef,
            baseline=baseline,
            device=device, n_games=config.games_per_batch,
            refine_steps=config.refine_steps,
            conservative_alpha=alpha,
        )

        # Anneal alpha and temperature
        alpha *= config.alpha_decay
        temperature = max(temperature * config.temperature_decay, config.temperature_min)

        total_played += config.games_per_batch
        metrics["game"].append(total_played)
        metrics["loss"].append(loss)
        metrics["rl_loss"].append(rl_loss)
        metrics["mse_loss"].append(mse_loss)
        metrics["avg_return"].append(avg_ret)
        metrics["alpha"].append(alpha)
        metrics["temperature"].append(temperature)

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
                f"loss={loss:.4f} rl={rl_loss:.4f} mse={mse_loss:.4f} | "
                f"ret={avg_ret:.1f} | eval_wr={eval_wr:.1%} | "
                f"α={alpha:.3f} τ={temperature:.3f} | {elapsed:.0f}s"
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

    # Final eval (collect_eval internally calls model.eval(), but be explicit)
    model.eval()
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
