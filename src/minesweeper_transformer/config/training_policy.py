"""Project-wide training/evaluation policy.

This file is the single source of truth for cross-stage behavior that must stay
consistent between supervised training and evaluation.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RefinementPolicy:
    """Global policy for refinement iterations.

    Training: Fixed BPTT unfolding (no detach between steps).
    Inference: Same steps as training; early stop on convergence.
    """

    train_max_steps: int = 4
    eval_max_steps: int = 4      # must match train_max_steps
    convergence_eps: float = 0.01  # stop when max|ΔP| < 1%


@dataclass(frozen=True)
class ProjectPolicy:
    """Top-level policy snapshot for project-wide defaults."""

    refinement: RefinementPolicy = field(default_factory=RefinementPolicy)


POLICY = ProjectPolicy()
