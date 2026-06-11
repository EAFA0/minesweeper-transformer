"""Training configuration and metrics."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    # Board
    board_width: int = 8
    board_height: int = 8
    board_mines: int = 10
    max_game_steps: int = 200

    # Pool
    pool_size: int = 100
    pool_workers: int = 2         # 0 = serial, >=1 = multiprocessing background workers
    data_dir: str = "data"  # strict no-guess offline npz dir
    mixed_mode: bool = False      # if True, randomizes board size and mine density

    # Training
    mode: str = "online"  # "online" or "supervised"
    n_games: int = 5000
    eval_interval_games: int = 200
    eval_games: int = 100
    board_pool_path: str = ""
    # "bce", "mse", "deep_mse", "deep_mse_rank", "deep_mse_denoise_rank",
    # or "deep_mse_solver_safe_rank"
    loss_type: str = "bce"
    rank_loss_weight: float = 0.1
    rank_loss_margin: float = 0.5
    rank_safe_threshold: float = 1e-6

    # Optimizer
    learning_rate: float = 3e-4
    min_lr: float = 0.0
    weight_decay: float = 3e-4
    grad_clip_norm: float = 1.0

    # Refinement — training steps override, default from POLICY
    refinement_steps: int = 4  # default matches POLICY.refinement.train_max_steps

    # Logging
    save_dir: str = "checkpoints"
    device: str = "auto"

    # Supervised Mode
    epochs: int = 5

    # Checkpoint
    pretrained: str = ""
    resume_from: str = ""


@dataclass
class TrainingMetrics:
    train_loss: List[float] = field(default_factory=list)
    val_action_accuracy: List[float] = field(default_factory=list)
    best_win_rate: float = 0.0
    best_epoch: int = 0
