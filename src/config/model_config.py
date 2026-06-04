"""Model configuration and hyperparameters."""

from dataclasses import dataclass
from .training_policy import POLICY


@dataclass
class ModelConfig:
    """Configuration for MinesweeperTransformer."""
    # Input
    in_channels: int = 10       # covered + flagged + 8 number channels

    # Channels
    d_model: int = 64           # CNN output = Transformer dim
    cnn_layers: int = 3         # number of Conv layers

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
