"""Training recipe: named multi-phase training strategies.

A recipe replaces the manual combination of --mode/--loss_type/--stage by
encoding a sequence of phases (e.g. MSE warmup -> online BCE finetune).
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RecipePhase:
    """A single phase within a training recipe."""

    mode: str = "online"          # "online" | "supervised"
    loss_type: str = "bce"        # "bce" | "mse"
    n_games: int = 5000
    lr: float = 3e-4
    board_width: int = 8
    board_height: int = 8
    board_mines: int = 10
    refinement_steps: int = 4
    pretrained: str = ""
    save_dir: str = ""
    desc: str = ""


@dataclass
class TrainingRecipe:
    """Named sequence of training phases."""

    name: str
    phases: List[RecipePhase] = field(default_factory=list)


# ── Predefined recipes ──────────────────────────────────────────────────────

RECIPES: Dict[str, TrainingRecipe] = {
    "v5_s1": TrainingRecipe(
        name="v5_s1",
        phases=[
            RecipePhase(
                mode="supervised", loss_type="mse", n_games=5000,
                board_width=8, board_height=8, board_mines=10,
                lr=3e-4, save_dir="checkpoints/v5_s1_mse",
                desc="S1 MSE warmup (probability calibration)",
            ),
            RecipePhase(
                mode="online", loss_type="bce", n_games=3000,
                board_width=8, board_height=8, board_mines=10,
                lr=1e-4, pretrained="checkpoints/v5_s1_mse/best_model.pt",
                save_dir="checkpoints/v5_s1_bce",
                desc="S1 online BCE finetune (on-policy alignment)",
            ),
        ],
    ),
}


def apply_recipe_phase(phase: RecipePhase, config) -> None:
    """Apply a single RecipePhase to a TrainingConfig object in-place."""
    config.mode = phase.mode
    config.loss_type = phase.loss_type
    config.n_games = phase.n_games
    config.learning_rate = phase.lr
    config.board_width = phase.board_width
    config.board_height = phase.board_height
    config.board_mines = phase.board_mines
    config.refinement_steps = phase.refinement_steps
    if phase.pretrained:
        config.pretrained = phase.pretrained
    if phase.save_dir:
        config.save_dir = phase.save_dir
