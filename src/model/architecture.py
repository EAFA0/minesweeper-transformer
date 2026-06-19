"""CNN + Transformer architecture with explicit constraint residual channels.

The architecture uses an explicit feedback loop: previous mine probability
estimates and rule-derived constraint channels are fed back as input, making
local number-constraint imbalance directly visible to the network.

Input: 10 board + 1 prev_probs + 8 constraints = 19 channels
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class CNNEncoder(nn.Module):
    """Convolutional frontend that preserves spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3,
                 norm_type: str = "batch", group_norm_groups: int = 8):
        super().__init__()
        layers = []
        current = in_channels
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(current, out_channels, kernel_size=3, padding=1, bias=False)
            )
            if norm_type == "group":
                layers.append(nn.GroupNorm(group_norm_groups, out_channels))
            elif norm_type == "batch":
                layers.append(nn.BatchNorm2d(out_channels))
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")
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

    def _neighbor_max(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        default: float = 0.0,
    ) -> torch.Tensor:
        fill = torch.full_like(values, -1e4)
        masked = torch.where(mask > 0, values, fill)
        pooled = F.max_pool2d(masked, kernel_size=3, stride=1, padding=1)
        has_value = self._neighbor_sum(mask).clamp(0.0, 1.0)
        return torch.where(has_value > 0, pooled, torch.full_like(values, default))

    def _neighbor_min(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        default: float = 0.0,
    ) -> torch.Tensor:
        return -self._neighbor_max(-values, mask, default=-default)

    def forward(self, board: torch.Tensor, prev_probs: torch.Tensor) -> torch.Tensor:
        """Return 8 constraint channels aligned to covered candidate cells.

        Channels:
            0: mean residual from adjacent revealed constraints
            1: max residual from adjacent revealed constraints
            2: min residual from adjacent revealed constraints
            3: mean absolute residual magnitude
            4: adjacent constraint count normalized by 8
            5: forced-safe signal from adjacent hard constraints
            6: forced-mine signal from adjacent hard constraints
            7: minimum hard-rule slack normalized by 8
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
        covered_neighbors = self._neighbor_sum(covered)
        predicted_neighbors = self._neighbor_sum(prev_probs * covered)

        target_remaining = number_value - flagged_neighbors
        residual = (target_remaining - predicted_neighbors) * constraint_mask

        constraint_count = self._neighbor_sum(constraint_mask) * covered
        denom = constraint_count.clamp_min(1.0)

        residual_sum = self._neighbor_sum(residual) * covered
        abs_residual_sum = self._neighbor_sum(residual.abs()) * covered

        mean_residual = residual_sum / denom
        max_residual = self._neighbor_max(residual, constraint_mask) * covered
        min_residual = self._neighbor_min(residual, constraint_mask) * covered
        mean_abs_residual = abs_residual_sum / denom
        constraint_count_norm = (constraint_count / 8.0).clamp(0.0, 1.0)

        hard_eps = 1e-4
        forced_safe_constraint = (
            (target_remaining <= hard_eps) & (constraint_mask > 0)
        ).to(dtype=board.dtype)
        forced_mine_constraint = (
            ((covered_neighbors - target_remaining).abs() <= hard_eps)
            & (covered_neighbors > 0)
            & (constraint_mask > 0)
        ).to(dtype=board.dtype)
        forced_safe_signal = (
            self._neighbor_sum(forced_safe_constraint).clamp(0.0, 1.0) * covered
        )
        forced_mine_signal = (
            self._neighbor_sum(forced_mine_constraint).clamp(0.0, 1.0) * covered
        )

        safe_slack = target_remaining.clamp_min(0.0)
        mine_slack = (covered_neighbors - target_remaining).clamp_min(0.0)
        hard_rule_slack = torch.minimum(safe_slack, mine_slack) * constraint_mask
        min_slack = (
            self._neighbor_min(hard_rule_slack, constraint_mask, default=8.0) * covered
        )
        min_slack_norm = (min_slack / 8.0).clamp(0.0, 1.0)

        return torch.cat(
            [
                mean_residual,
                max_residual,
                min_residual,
                mean_abs_residual,
                constraint_count_norm,
                forced_safe_signal,
                forced_mine_signal,
                min_slack_norm,
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

    constraint_channels: int = 8

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
            norm_type=getattr(config, "norm_type", "batch"),
            group_norm_groups=getattr(config, "group_norm_groups", 8),
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
               return_logits: bool = False,
               initial_probs: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        B, _, H, W = board.shape
        if initial_probs is None:
            probs = torch.full((B, 1, H, W), 0.5, device=board.device, dtype=board.dtype)
        else:
            probs = initial_probs.to(device=board.device, dtype=board.dtype)
            if probs.shape != (B, 1, H, W):
                expected = (B, 1, H, W)
                raise ValueError(
                    f"initial_probs must have shape {expected}, got {tuple(probs.shape)}"
                )
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
        """Load weights from a checkpoint.

        Architecture shape changes intentionally fail fast. Historical
        checkpoints should be retrained instead of migrated.
        """
        from training.checkpoints import checkpoint_state_dict

        self.load_state_dict(checkpoint_state_dict(checkpoint_path, device=device))
