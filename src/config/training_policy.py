"""Project-wide training/evaluation policy.

This file is the single source of truth for cross-stage behavior that must stay
consistent between supervised training, RL fine-tuning, and evaluation.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RefinementPolicy:
    """Global policy for refinement iterations.
    
    Training: Fixed unfolding (BPTT) for a small number of steps to learn core logic 
              while preventing gradient explosion and saving compute.
    Inference: Can run much longer to solve hard edge cases using early stop.
    """

    train_max_steps: int = 4     # Reduced to 4 to save compute during BPTT training
    eval_max_steps: int = 4      # Match train_max_steps; prevents train/eval mismatch
    rl_steps: int = 4
    convergence_eps: float = 0.01  # stop when max|ΔP| < 1% (tightened from 5%)


@dataclass(frozen=True)
class RLRewardPolicy:
    """Default reward shaping for conservative RL fine-tuning."""

    reveal_safe: float = 1.0
    floodfill_bonus: float = 0.05
    hit_mine: float = -1.0   # less punishing with mine_continue
    win_bonus: float = 10.0   # make winning clearly valuable
    step_penalty: float = 0.0


@dataclass(frozen=True)
class ProjectPolicy:
    """Top-level policy snapshot for project-wide defaults."""

    refinement: RefinementPolicy = field(default_factory=RefinementPolicy)
    rl_rewards: RLRewardPolicy = field(default_factory=RLRewardPolicy)


POLICY = ProjectPolicy()
