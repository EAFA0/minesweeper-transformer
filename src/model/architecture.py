"""CNN + Transformer hybrid architecture for Minesweeper.

Architecture:
    Input: (B, 10, H, W) — covered, flagged, numbers 1-8 one-hot
    → CNN frontend (3× Conv3×3, BN, ReLU) → (B, d_model, H, W)
    → 2D learnable positional encoding (interpolatable)
    → Flatten to sequence → (B, H*W, d_model)
    → Transformer encoder (N layers, multi-head self-attention)
    → Reshape to spatial → (B, d_model, H, W)
    → Output head (Conv1×1) → (B, 1, H, W)
    → Sigmoid → P(mine) per cell

Supports variable board sizes via bilinear PE interpolation (ViT-style).
CNN and transformer layers are size-agnostic — only PE adapts to input size.
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

    # CNN frontend
    cnn_channels: int = 64      # output channels of CNN
    cnn_layers: int = 3         # number of Conv layers

    # Transformer
    d_model: int = 64           # must match cnn_channels
    nhead: int = 4              # attention heads
    num_layers: int = 3         # transformer encoder layers
    dim_feedforward: int = 256  # FFN hidden dim
    dropout: float = 0.2

    # Positional encoding — reference grid size for interpolation
    pe_grid_size: int = 16      # PE is learned at 16×16, bilinear-interpolated to any H×W

    # Output
    num_classes: int = 1

    def __post_init__(self):
        if self.cnn_channels != self.d_model:
            raise ValueError(
                f"cnn_channels ({self.cnn_channels}) must match d_model ({self.d_model})"
            )


# ─── Building Blocks ───────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """Convolutional frontend that preserves spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3):
        super().__init__()
        layers = []
        current = in_channels
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(current, out_channels, kernel_size=3, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            current = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InterpolatablePositionalEncoding(nn.Module):
    """2D learnable PE with bilinear interpolation for variable input sizes.

    PE is stored at a reference grid size (default 16×16). At forward time,
    it's bilinear-interpolated to match the input's H×W.
    This allows a model trained on one board size to transfer to another.
    """

    def __init__(self, d_model: int, ref_grid: int = 16):
        super().__init__()
        # Learned at reference size: (1, d_model, ref_grid, ref_grid)
        self.pe = nn.Parameter(torch.randn(1, d_model, ref_grid, ref_grid) * 0.02)
        self.ref_grid = ref_grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W) with PE added."""
        _, _, H, W = x.shape
        if H == self.ref_grid and W == self.ref_grid:
            pe = self.pe
        else:
            pe = F.interpolate(
                self.pe, size=(H, W), mode='bilinear', align_corners=False
            )
        return x + pe


class TransformerEncoder(nn.Module):
    """Stack of TransformerEncoderLayers."""

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, C) → (B, S, C)"""
        x = self.encoder(x)
        return self.norm(x)


# ─── Full Model ────────────────────────────────────────────────────────────

class MinesweeperTransformer(nn.Module):
    """CNN + Transformer model for minesweeper. Supports variable board sizes.

    Input:  (B, 10, H, W)
    Output: (B, 1, H, W) — logits for P(mine). Apply sigmoid for probabilities.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # CNN frontend (size-agnostic)
        self.cnn = CNNEncoder(
            in_channels=config.in_channels,
            out_channels=config.d_model,
            num_layers=config.cnn_layers,
        )

        # Positional encoding (interpolatable to any H,W)
        self.pos_encoding = InterpolatablePositionalEncoding(
            d_model=config.d_model, ref_grid=config.pe_grid_size,
        )

        # Transformer (sequence-length agnostic)
        self.transformer = TransformerEncoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )

        # Output head (size-agnostic Conv1×1)
        self.output_head = nn.Conv2d(config.d_model, config.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. H,W inferred from input tensor.

        Args:
            x: (B, C, H, W) input channels

        Returns:
            (B, 1, H, W) logits
        """
        B, C, H, W = x.shape

        # CNN: (B, C, H, W) → (B, d_model, H, W)
        features = self.cnn(x)

        # PE (interpolated to H,W)
        features = self.pos_encoding(features)

        # Reshape to sequence: (B, d_model, H, W) → (B, H*W, d_model)
        seq = features.flatten(2).transpose(1, 2)

        # Transformer
        seq = self.transformer(seq)

        # Reshape back: (B, H*W, d_model) → (B, d_model, H, W)
        features = seq.transpose(1, 2).reshape(B, self.config.d_model, H, W)

        # Output head
        logits = self.output_head(features)  # (B, 1, H, W)

        return logits

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(mine) probabilities (sigmoid-applied)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def load_pretrained(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load weights from a checkpoint, with automatic format migration.

        Handles:
        - Old format (row_embed + col_embed) → new format (interpolatable pe)
        - Missing/additional keys are ignored
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Migrate old positional encoding format
        migrated = {}
        for key, value in state_dict.items():
            if key.startswith("pos_encoding.row_embed") or key.startswith("pos_encoding.col_embed"):
                # Old format: separate row/col embeddings
                # New format: single pe grid (1, d_model, ref, ref)
                # We can't perfectly migrate, so skip and keep new PE initialized
                continue
            migrated[key] = value

        # Load with strict=False to allow PE mismatch
        missing, unexpected = self.load_state_dict(migrated, strict=False)
        if missing:
            print(f"  (PE reinitialized for new grid size — {len(missing)} keys)")
        if unexpected:
            print(f"  (ignored {len(unexpected)} old-format keys)")
