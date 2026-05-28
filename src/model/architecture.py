"""CNN + Transformer hybrid architecture for Minesweeper.

Architecture:
    Input: (B, 10, H, W) — covered, flagged, numbers 1-8 one-hot
    → CNN frontend (3× Conv3×3, BN, ReLU) → (B, d_model, H, W)
    → 2D learnable positional encoding
    → Flatten to sequence → (B, H*W, d_model)
    → Transformer encoder (N layers, multi-head self-attention)
    → Reshape to spatial → (B, d_model, H, W)
    → Output head (Conv1×1) → (B, 1, H, W)
    → Sigmoid → P(mine) per cell
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for MinesweeperTransformer."""
    # Input
    in_channels: int = 10       # covered + flagged + 8 number channels
    board_size: int = 8         # assumes square board

    # CNN frontend
    cnn_channels: int = 64      # output channels of CNN
    cnn_layers: int = 3         # number of Conv layers

    # Transformer
    d_model: int = 64           # must match cnn_channels
    nhead: int = 4              # attention heads
    num_layers: int = 3         # transformer encoder layers
    dim_feedforward: int = 256  # FFN hidden dim
    dropout: float = 0.2   # increased from 0.1 for regularization

    # Output
    num_classes: int = 1        # binary classification → 1 channel + sigmoid

    def __post_init__(self):
        if self.cnn_channels != self.d_model:
            raise ValueError(
                f"cnn_channels ({self.cnn_channels}) must match d_model ({self.d_model})"
            )


class CNNEncoder(nn.Module):
    """Convolutional frontend that preserves spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3):
        super().__init__()
        layers = []
        current = in_channels
        for i in range(num_layers):
            layers.append(
                nn.Conv2d(current, out_channels, kernel_size=3, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            current = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Learnable2DPositionalEncoding(nn.Module):
    """2D learnable positional encoding for grid-structured inputs."""

    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()
        self.row_embed = nn.Parameter(torch.randn(1, d_model, height, 1) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, d_model, 1, width) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W) with positional encoding added."""
        return x + self.row_embed + self.col_embed


class TransformerEncoder(nn.Module):
    """Stack of TransformerEncoderLayers with optional LayerNorm at the end."""

    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,  # input shape: (B, S, C)
            norm_first=True,   # pre-LN (better training stability)
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, C) → (B, S, C)"""
        x = self.encoder(x)
        return self.norm(x)


class MinesweeperTransformer(nn.Module):
    """Full CNN + Transformer model for minesweeper.

    Input:  (B, 10, H, W)
    Output: (B, 1, H, W) — logits for P(mine), apply sigmoid for probabilities.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        H = W = config.board_size

        # CNN frontend
        self.cnn = CNNEncoder(
            in_channels=config.in_channels,
            out_channels=config.d_model,
            num_layers=config.cnn_layers,
        )

        # Positional encoding
        self.pos_encoding = Learnable2DPositionalEncoding(
            d_model=config.d_model, height=H, width=W
        )

        # Transformer
        self.transformer = TransformerEncoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )

        # Output head
        self.output_head = nn.Conv2d(config.d_model, config.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W) input channels

        Returns:
            (B, 1, H, W) logits — sigmoid for P(mine)
        """
        B, C, H, W = x.shape

        # CNN: (B, C, H, W) → (B, d_model, H, W)
        features = self.cnn(x)

        # Positional encoding
        features = self.pos_encoding(features)

        # Reshape to sequence: (B, d_model, H, W) → (B, H*W, d_model)
        seq = features.flatten(2).transpose(1, 2)  # (B, H*W, d_model)

        # Transformer
        seq = self.transformer(seq)  # (B, H*W, d_model)

        # Reshape back: (B, H*W, d_model) → (B, d_model, H, W)
        features = seq.transpose(1, 2).reshape(B, self.config.d_model, H, W)

        # Output head
        logits = self.output_head(features)  # (B, 1, H, W)

        return logits

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(mine) probabilities (sigmoid-applied)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
