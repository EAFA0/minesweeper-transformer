"""Project-wide training/evaluation policy.

This file is the single source of truth for cross-stage behavior that must stay
consistent between supervised training, RL fine-tuning, and evaluation.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RefinementPolicy:
    """Iterative refinement policy shared by all training/evaluation loops."""

    train_max_steps: int = 16
    eval_max_steps: int = 16
    rl_steps: int = 16
    convergence_eps: float = 1e-3


@dataclass(frozen=True)
class RLRewardPolicy:
    """Default reward shaping for conservative RL fine-tuning."""

    reveal_safe: float = 1.0
    floodfill_bonus: float = 0.05
    hit_mine: float = -20.0
    step_penalty: float = 0.0


@dataclass(frozen=True)
class ProjectPolicy:
    """Top-level policy snapshot for project-wide defaults."""

    refinement: RefinementPolicy = field(default_factory=RefinementPolicy)
    rl_rewards: RLRewardPolicy = field(default_factory=RLRewardPolicy)


POLICY = ProjectPolicy()
