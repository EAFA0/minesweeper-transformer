"""CNN + Transformer architecture with explicit constraint residual channels.

The architecture uses an explicit feedback loop: previous mine probability
estimates and rule-derived constraint channels are fed back as input, making
local number-constraint imbalance directly visible to the network.

Input: 10 board + 1 prev_probs + 4 constraints = 15 channels
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


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
        _, _, H, W = x.shape
        if H == self.ref_grid and W == self.ref_grid:
            pe = self.pe
        else:
            pe = F.interpolate(
                self.pe, size=(H, W), mode="bilinear", align_corners=False
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
        x = self.encoder(x)
        return self.norm(x)


class ConstraintFeatureBuilder(nn.Module):
    """Build board-sized constraint channels from board state and prev_probs."""

    def __init__(self):
        super().__init__()
        kernel = torch.ones(1, 1, 3, 3)
        kernel[0, 0, 1, 1] = 0.0
        self.register_buffer("neighbor_kernel", kernel)

    def _neighbor_sum(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.neighbor_kernel.to(dtype=x.dtype), padding=1)

    def forward(self, board: torch.Tensor, prev_probs: torch.Tensor) -> torch.Tensor:
        """Return 4 constraint channels aligned to covered candidate cells.

        Channels:
            0: mean residual from adjacent revealed constraints
            1: mean absolute residual magnitude
            2: adjacent constraint count normalized by 8
            3: mean target remaining mines from adjacent revealed constraints
        """
        covered = board[:, 0:1]
        flagged = board[:, 1:2]
        number_channels = board[:, 2:10]

        weights = torch.arange(
            1, 9, device=board.device, dtype=board.dtype
        ).view(1, 8, 1, 1)
        number_value = (number_channels * weights).sum(dim=1, keepdim=True)

        known_cell = (1.0 - covered - flagged).clamp(0.0, 1.0)
        constraint_mask = known_cell

        flagged_neighbors = self._neighbor_sum(flagged)
        predicted_neighbors = self._neighbor_sum(prev_probs * covered)

        target_remaining = number_value - flagged_neighbors
        residual = (target_remaining - predicted_neighbors) * constraint_mask
        target_remaining = target_remaining * constraint_mask

        constraint_count = self._neighbor_sum(constraint_mask) * covered
        denom = constraint_count.clamp_min(1.0)

        residual_sum = self._neighbor_sum(residual) * covered
        abs_residual_sum = self._neighbor_sum(residual.abs()) * covered
        target_sum = self._neighbor_sum(target_remaining) * covered

        mean_residual = residual_sum / denom
        mean_abs_residual = abs_residual_sum / denom
        constraint_count_norm = (constraint_count / 8.0).clamp(0.0, 1.0)
        mean_target_remaining = target_sum / denom

        return torch.cat(
            [
                mean_residual,
                mean_abs_residual,
                constraint_count_norm,
                mean_target_remaining,
            ],
            dim=1,
        )


class MinesweeperTransformer(nn.Module):
    """CNN + Transformer model with explicit constraint feedback refinement.

    Single-pass (refinement_steps=1):
        Input:  (B, 10, H, W)
        Output: (B, 1, H, W) — raw mine logits

    Iterative refinement (refinement_steps=N):
        Internally feeds own output back as prev_probs + constraint channels,
        running N passes with shared weights (full BPTT).
    """

    constraint_channels: int = 4

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        self.constraint_features = ConstraintFeatureBuilder()

        self.cnn = CNNEncoder(
            in_channels=config.in_channels + 1 + self.constraint_channels,
            out_channels=config.d_model,
            num_layers=config.cnn_layers,
        )

        self.pos_encoding = InterpolatablePositionalEncoding(
            d_model=config.d_model, ref_grid=config.pe_grid_size,
        )

        self.transformer = TransformerEncoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )

        self.output_head = nn.Conv2d(config.d_model, 1, kernel_size=1)

    def _single_pass(
        self,
        board: torch.Tensor,
        prev_probs: torch.Tensor,
    ) -> torch.Tensor:
        B, _, H, W = board.shape
        constraint_channels = self.constraint_features(board, prev_probs)
        x = torch.cat([board, prev_probs, constraint_channels], dim=1)

        features = self.cnn(x)
        features = self.pos_encoding(features)

        seq = features.flatten(2).transpose(1, 2)
        seq = self.transformer(seq)
        features = seq.transpose(1, 2).reshape(B, self.config.d_model, H, W)

        return self.output_head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        prev = torch.full((B, 1, H, W), 0.5, device=x.device, dtype=x.dtype)
        return self._single_pass(x, prev)

    def refine(self, board: torch.Tensor, num_steps: int = 5,
               convergence_epsilon: float = 0.01,
               return_logits: bool = False) -> List[torch.Tensor]:
        B, _, H, W = board.shape
        probs = torch.full((B, 1, H, W), 0.5, device=board.device, dtype=board.dtype)
        results = []

        for _ in range(num_steps):
            raw = self._single_pass(board, probs)
            mine_logits = raw[:, 0:1]
            mine_prob = torch.sigmoid(mine_logits)

            if return_logits:
                results.append(mine_logits)
            else:
                results.append(mine_prob)

            if not self.training and not return_logits:
                delta = (mine_prob - probs).abs().amax()
                if delta < convergence_epsilon:
                    break

            probs = mine_prob

        return results

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, max_refine_steps: int = 1) -> torch.Tensor:
        if max_refine_steps <= 1:
            raw = self.forward(x)
            return torch.sigmoid(raw[:, 0:1])
        results = self.refine(x, num_steps=max_refine_steps)
        return results[-1][:, 0:1]

    def load_pretrained(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load weights from a checkpoint with automatic channel migration."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        cnn_key = "cnn.net.0.weight"
        if cnn_key in state_dict:
            old_w = state_dict[cnn_key]
            new_w = self.cnn.net[0].weight.data
            if old_w.shape[1] != new_w.shape[1]:
                padded = torch.zeros_like(new_w)
                keep = min(old_w.shape[1], new_w.shape[1])
                padded[:, :keep] = old_w[:, :keep]
                state_dict[cnn_key] = padded
                print(
                    f"  (Migrated CNN input channels: {old_w.shape[1]} -> {new_w.shape[1]})"
                )

        head_w_key = "output_head.weight"
        head_b_key = "output_head.bias"
        if head_w_key in state_dict:
            old_w = state_dict[head_w_key]
            new_w = self.output_head.weight.data
            if old_w.shape != new_w.shape:
                padded = torch.zeros_like(new_w)
                keep_out = min(old_w.shape[0], new_w.shape[0])
                keep_in = min(old_w.shape[1], new_w.shape[1])
                padded[:keep_out, :keep_in] = old_w[:keep_out, :keep_in]
                state_dict[head_w_key] = padded
                print(
                    f"  (Migrated output head weight: {tuple(old_w.shape)} -> {tuple(new_w.shape)})"
                )
        if head_b_key in state_dict:
            old_b = state_dict[head_b_key]
            new_b = self.output_head.bias.data
            if old_b.shape != new_b.shape:
                padded = torch.zeros_like(new_b)
                keep = min(old_b.shape[0], new_b.shape[0])
                padded[:keep] = old_b[:keep]
                state_dict[head_b_key] = padded
                print(
                    f"  (Migrated output head bias: {tuple(old_b.shape)} -> {tuple(new_b.shape)})"
                )

        migrated = {}
        for key, value in state_dict.items():
            if key.startswith("pos_encoding.row_embed") or key.startswith(
                "pos_encoding.col_embed"
            ):
                continue
            migrated[key] = value

        missing, unexpected = self.load_state_dict(migrated, strict=False)
        if missing:
            print(f"  (initialized {len(missing)} new keys)")
        if unexpected:
            print(f"  (ignored {len(unexpected)} incompatible keys)")
