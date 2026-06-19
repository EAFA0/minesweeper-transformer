"""Model configuration and hyperparameters."""

from dataclasses import dataclass
from .training_policy import POLICY


@dataclass
class ModelConfig:
    """Configuration for MinesweeperTransformer."""
    # Input
    in_channels: int = 10       # covered + flagged + 8 number channels

    # Output
    num_classes: int = 1        # 1 for BCE loss (P(mine))

    # Channels
    d_model: int = 64           # CNN output = Transformer dim
    cnn_layers: int = 3         # number of Conv layers
    norm_type: str = "batch"    # "batch" or "group" (batch-independent, for online batch=1)
    group_norm_groups: int = 8  # groups for GroupNorm when norm_type="group"

    # Transformer
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.2

    # Positional encoding
    pe_grid_size: int = 16

    # Refinement
    @property
    def refinement_steps(self) -> int:
        return POLICY.refinement.train_max_steps
