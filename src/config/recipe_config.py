"""Training recipe: named training strategies.

A recipe replaces the manual combination of --mode/--loss_type/--stage by
encoding one or more reproducible training phases.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RecipePhase:
    """A single phase within a training recipe."""

    mode: str = "online"          # "online" | "supervised"
    # "bce" | "mse" | "deep_mse" | "deep_mse_rank"
    loss_type: str = "bce"
    n_games: int = 5000
    lr: float = 3e-4
    board_width: int = 8
    board_height: int = 8
    board_mines: int = 10
    refinement_steps: int = 4
    pretrained: str = ""
    save_dir: str = ""
    data_dir: str = "data"
    desc: str = ""


@dataclass
class TrainingRecipe:
    """Named sequence of training phases."""

    name: str
    phases: List[RecipePhase] = field(default_factory=list)


# ── Predefined recipes ──────────────────────────────────────────────────────

RECIPES: Dict[str, TrainingRecipe] = {
    "v5_s1_rank": TrainingRecipe(
        name="v5_s1_rank",
        phases=[
            RecipePhase(
                mode="supervised", loss_type="deep_mse_rank", n_games=5000,
                board_width=8, board_height=8, board_mines=10,
                lr=3e-4, save_dir="checkpoints/v5_s1_rank",
                data_dir="data",
                desc="S1 supervised Deep-MSE + best-safe ranking — 8x8/10",
            ),
        ],
    ),
    "v5_curriculum_replay": TrainingRecipe(
        name="v5_curriculum_replay",
        phases=[
            RecipePhase(
                mode="supervised", loss_type="deep_mse_rank", n_games=10000,
                board_width=8, board_height=8, board_mines=10,
                lr=3e-4, save_dir="checkpoints/v5_replay_S1",
                data_dir="data/S1",
                desc="S1 supervised Deep-MSE + ranking — 8x8/10",
            ),
            RecipePhase(
                mode="supervised", loss_type="deep_mse_rank", n_games=10000,
                board_width=8, board_height=8, board_mines=15,
                lr=3e-4, save_dir="checkpoints/v5_replay_S2",
                data_dir="data/S2:0.7,data/S1:0.3",
                desc="S2 replay mix — 70% 8x8/15 + 30% S1",
            ),
            RecipePhase(
                mode="supervised", loss_type="deep_mse_rank", n_games=10000,
                board_width=8, board_height=8, board_mines=20,
                lr=3e-4, save_dir="checkpoints/v5_replay_S3",
                data_dir="data/S3:0.7,data/S1:0.15,data/S2:0.15",
                desc="S3 replay mix — 70% 8x8/20 + 15% S1 + 15% S2",
            ),
            RecipePhase(
                mode="supervised", loss_type="deep_mse_rank", n_games=10000,
                board_width=8, board_height=8, board_mines=25,
                lr=3e-4, save_dir="checkpoints/v5_replay_S4",
                data_dir="data/S4:0.7,data/S1:0.1,data/S2:0.1,data/S3:0.1",
                desc="S4 replay mix — 70% 8x8/25 + 10% S1 + 10% S2 + 10% S3",
            ),
            RecipePhase(
                mode="supervised", loss_type="deep_mse_rank", n_games=10000,
                board_width=8, board_height=8, board_mines=32,
                lr=3e-4, save_dir="checkpoints/v5_replay_S5",
                data_dir="data/S5:0.6,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1",
                desc="S5 max-density replay — 60% 8x8/32 + 10% S1/S2/S3/S4",
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
    if phase.data_dir:
        config.data_dir = phase.data_dir
