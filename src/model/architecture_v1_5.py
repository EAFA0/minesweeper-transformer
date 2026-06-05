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

Iterative Refinement mode:
    The model can be called multiple times, feeding its own output back
    as an additional input channel (prev_probs). This allows the model
    to "rethink" its predictions — detecting contradictions and refining.
    
    Step 0: prev_probs = 0.5 (uniform)
    Step 1: probs = model(board, prev_probs)
    Step 2: probs = model(board, probs)     ← sees previous guess
    Step 3: probs = model(board, probs)     ← further refinement
    
    Shared weights across all steps — the model learns a "refinement operator".

Supports variable board sizes via bilinear PE interpolation (ViT-style).
"""

from dataclasses import dataclass
from typing import List, Optional
from config import ModelConfig

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """2D learnable PE with bilinear interpolation for variable input sizes."""

    def __init__(self, d_model: int, ref_grid: int = 16):
        super().__init__()
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

class MinesweeperTransformerV1_5(nn.Module):
    """CNN + Transformer model for minesweeper.

    Single-pass (refinement_steps=1):
        Input:  (B, 10, H, W)
        Output: (B, 1, H, W) — logits for P(mine)

    Iterative refinement (refinement_steps=N):
        Internally feeds own output back as an 11th channel,
        running N passes with shared weights. Returns list of
        N probability tensors for progressive-loss training.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # CNN: 11 input channels = 10 board + 1 prev_probs
        self.cnn = CNNEncoder(
            in_channels=config.in_channels + 1,
            out_channels=config.d_model,
            num_layers=config.cnn_layers,
        )

        # Positional encoding (interpolatable)
        self.pos_encoding = InterpolatablePositionalEncoding(
            d_model=config.d_model, ref_grid=config.pe_grid_size,
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
        self.output_head = nn.Conv2d(config.d_model, 2, kernel_size=1)

    def _single_pass(self, board: torch.Tensor, prev_probs: torch.Tensor) -> torch.Tensor:
        """Internal: one forward pass with board + prev_probs concatenated.

        Args:
            board:      (B, 10, H, W) board channels
            prev_probs: (B, 1, H, W) previous probability estimate

        Returns:
            (B, 2, H, W) raw outputs — [0]=P(mine) logit, [1]=confidence logit
        """
        B, C, H, W = board.shape
        x = torch.cat([board, prev_probs], dim=1)  # (B, 11, H, W)

        # CNN
        features = self.cnn(x)

        # PE
        features = self.pos_encoding(features)

        # Sequence → Transformer
        seq = features.flatten(2).transpose(1, 2)
        seq = self.transformer(seq)

        # Back to spatial
        features = seq.transpose(1, 2).reshape(B, self.config.d_model, H, W)

        # Output
        return self.output_head(features)  # (B, 2, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-pass forward (standard mode).

        Args:
            x: (B, 10, H, W) board channels

        Returns:
            (B, 2, H, W) raw outputs — [0]=P(mine) logit, [1]=confidence logit
        """
        B, _, H, W = x.shape
        prev = torch.zeros(B, 1, H, W, device=x.device)
        return self._single_pass(x, prev)

    def refine(self, board: torch.Tensor, num_steps: int = 5,
               confidence_threshold: float = 0.95,
               return_logits: bool = False) -> List[torch.Tensor]:
        """Iterative refinement: run model N times with self-feedback.

        Step 0: prev = uniform(0.5)
        Step k: probs_k, conf_k = model(board, probs_{k-1})

        Full BPTT: gradients flow through all refinement steps (no detach).

        Args:
            board:                (B, 10, H, W) board channels
            num_steps:            max refinement iterations
            confidence_threshold: early-stop when mean(conf) > this
            return_logits:        if True, save raw logits (for BCEWithLogitsLoss);
                                  if False, save sigmoid'd probs/conf (for inference)

        Returns:
            List of (B, 2, H, W) tensors per step.
            Last element is the final output.
        """
        B, _, H, W = board.shape
        probs = torch.full((B, 1, H, W), 0.5, device=board.device)
        results = []

        for _ in range(num_steps):
            raw = self._single_pass(board, probs)  # (B, 2, H, W) — full BPTT

            if return_logits:
                results.append(raw)  # raw logits: channel 0 = mine, channel 1 = conf
            else:
                mine_prob = torch.sigmoid(raw[:, 0:1])
                conf_prob = torch.sigmoid(raw[:, 1:2])
                results.append(torch.cat([mine_prob, conf_prob], dim=1))

            # Next step input: sigmoid'd mine probability (in comp graph → BPTT)
            probs = torch.sigmoid(raw[:, 0:1])

            # Early stop only during inference with non-logits mode
            if not self.training and not return_logits:
                if conf_prob.mean() > confidence_threshold:
                    break

        return results

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, max_refine_steps: int = 1) -> torch.Tensor:
        """Helper for evaluation. Returns probability matrix (1, H, W)."""
        if max_refine_steps <= 1:
            raw = self.forward(x)
            return torch.sigmoid(raw[:, 0:1])
        results = self.refine(x, num_steps=max_refine_steps)
        return results[-1][:, 0:1]

    def load_pretrained(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load weights from a checkpoint, with automatic format migration.

        Handles:
        - Old 10-channel CNN → new 11-channel: extra channel zero-padded
        - Old positional encoding formats
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Migrate old 10-channel CNN to 11-channel
        cnn_key = "cnn.net.0.weight"
        if cnn_key in state_dict:
            old_w = state_dict[cnn_key]  # (64, 10, 3, 3)
            new_w = self.cnn.net[0].weight.data  # (64, 11, 3, 3)
            if old_w.shape[1] == 10 and new_w.shape[1] == 11:
                # Pad: copy first 10 channels, zero for 11th
                padded = torch.zeros_like(new_w)
                padded[:, :10] = old_w
                state_dict[cnn_key] = padded
                print("  (Migrated CNN: 10→11 channels, extra channel zero-padded)")

        # Filter old positional encoding keys
        migrated = {}
        for key, value in state_dict.items():
            if key.startswith("pos_encoding.row_embed") or key.startswith("pos_encoding.col_embed"):
                continue
            migrated[key] = value

        missing, unexpected = self.load_state_dict(migrated, strict=False)
        if missing:
            print(f"  (PE reinitialized — {len(missing)} new keys)")
        if unexpected:
            print(f"  (ignored {len(unexpected)} old-format keys)")
