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

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for MinesweeperTransformer."""
    # Input
    in_channels: int = 10       # covered + flagged + 8 number channels
    # Internal: +1 channel for prev_probs during iterative refinement
    # (the extra channel is handled automatically)

    # CNN frontend
    cnn_channels: int = 64      # output channels of CNN
    cnn_layers: int = 3         # number of Conv layers

    # Transformer
    d_model: int = 64           # must match cnn_channels
    nhead: int = 4              # attention heads
    num_layers: int = 4         # transformer encoder layers
    dim_feedforward: int = 256  # FFN hidden dim
    dropout: float = 0.2

    # Positional encoding — reference grid size for interpolation
    pe_grid_size: int = 16      # PE is learned at 16×16, bilinear-interpolated to any H×W

    # Iterative refinement
    refinement_steps: int = 8   # max refinement iterations

    # Output
    num_classes: int = 2        # [0]=P(mine) logit, [1]=confidence logit

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

class MinesweeperTransformer(nn.Module):
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
        self.output_head = nn.Conv2d(config.d_model, config.num_classes, kernel_size=1)

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
        return self.output_head(features)  # (B, 1, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-pass forward (matches refinement step 1).

        Uses prev_probs=0.5 so the model sees the same initial state
        as the first step of iterative refinement.

        Args:
            x: (B, 10, H, W) board channels

        Returns:
            (B, 2, H, W) raw outputs — [0]=P(mine) logit, [1]=confidence logit
        """
        B, _, H, W = x.shape
        prev = torch.full((B, 1, H, W), 0.5, device=x.device)
        return self._single_pass(x, prev)

    def refine(self, board: torch.Tensor, num_steps: int = 5,
               confidence_threshold: float = 0.95,
               convergence_eps: float = 1e-3) -> List[torch.Tensor]:
        """Iterative refinement: run model N times with self-feedback.

        Step 0: prev = uniform(0.5)
        Step k: probs_k, conf_k = model(board, probs_{k-1})

        During inference, stops early when P(mine) predictions have converged
        (max absolute change between steps < convergence_eps).  This is more
        reliable than confidence-based stopping because the confidence head
        has no explicit supervision.

        During training, always runs num_steps for consistent loss.

        Args:
            board:                (B, 10, H, W) board channels
            num_steps:            max refinement iterations
            confidence_threshold: (deprecated, kept for API compat)
            convergence_eps:      stop when max|P_t - P_{t-1}| < this

        Returns:
            List of (B, 2, H, W) tensors — [probs_sigmoid, conf_sigmoid] per step
            Last element is the final output.
        """
        B, _, H, W = board.shape
        probs = torch.full((B, 1, H, W), 0.5, device=board.device)
        prev_probs = probs.clone()
        results = []

        for step in range(num_steps):
            raw = self._single_pass(board, probs)  # (B, 2, H, W)
            probs = torch.sigmoid(raw[:, 0:1])      # P(mine)
            conf = torch.sigmoid(raw[:, 1:2])       # confidence
            results.append(torch.cat([probs, conf], dim=1))

            # Early stop (inference only): has P(mine) converged?
            if not self.training and step > 0:
                max_change = (probs - prev_probs).abs().max().item()
                if max_change < convergence_eps:
                    break

            # Detach and save for convergence check
            prev_probs = probs.detach()
            probs = prev_probs.clone()

        return results

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, max_refine_steps: int = 16) -> torch.Tensor:
        """Return P(mine) probabilities with adaptive refinement.

        Inference uses more steps than training (default 12 vs train's random 1-8).
        Model stops early when confident, so 12 is just an upper bound.

        For models trained without refinement (confidence head zero):
        skips iteration — single pass only.
        """
        # Quick check: if confidence head was never trained, skip refinement
        if self.output_head.bias[1].abs().max() < 1e-8:
            return torch.sigmoid(self.forward(x)[:, 0:1])
        results = self.refine(x, num_steps=max_refine_steps)
        return results[-1][:, 0:1]

    @torch.no_grad()
    def diagnose_refinement(self, x: torch.Tensor, max_refine_steps: int = 8,
                            convergence_eps: float = 1e-3) -> dict:
        """Run refinement and report per-sample step statistics.

        Uses convergence-based early stop (same as refine()):
        stops when max |P_t - P_{t-1}| < convergence_eps.

        Returns dict with:
            probs:            (B, 1, H, W) final P(mine) probs
            n_steps:          tensor of shape (B,) — steps taken per sample
            mean_steps:       average steps across batch
            max_steps:        max steps in batch
            min_steps:        min steps in batch
            early_stop_rate:  fraction that stopped early (< max_steps)
            step_distribution: count per step bucket
        """
        B = x.shape[0]
        probs = torch.full((B, 1, x.shape[2], x.shape[3]), 0.5, device=x.device)
        prev_probs = probs.clone()
        n_steps = torch.zeros(B, dtype=torch.long, device=x.device)
        done = torch.zeros(B, dtype=torch.bool, device=x.device)

        for step in range(max_refine_steps):
            raw = self._single_pass(x, probs)
            probs_new = torch.sigmoid(raw[:, 0:1])

            # Per-sample convergence: max |P_t - P_{t-1}| < eps
            if step > 0:
                max_change = (probs_new - prev_probs).abs().view(B, -1).max(dim=1).values
                converged_now = ~done & (max_change < convergence_eps)
                done = done | converged_now
                n_steps = torch.where(converged_now, step + 1, n_steps)

                if done.all():
                    probs = probs_new
                    break

            prev_probs = probs_new.clone()
            probs = probs_new

        n_steps = torch.where(~done, max_refine_steps, n_steps)

        return {
            "probs": probs,
            "n_steps": n_steps,
            "mean_steps": n_steps.float().mean().item(),
            "max_steps": n_steps.max().item(),
            "min_steps": n_steps.min().item(),
            "early_stop_rate": (n_steps < max_refine_steps).float().mean().item(),
            "step_distribution": torch.bincount(n_steps, minlength=max_refine_steps + 1).tolist(),
        }

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
            old_w = state_dict[cnn_key]
            new_w = self.cnn.net[0].weight.data
            if old_w.shape[1] == 10 and new_w.shape[1] == 11:
                padded = torch.zeros_like(new_w)
                padded[:, :10] = old_w
                state_dict[cnn_key] = padded
                print("  (Migrated CNN: 10→11 channels, extra channel zero-padded)")

        # Migrate old 1-channel output to 2-channel (prob + confidence)
        out_w_key = "output_head.weight"
        out_b_key = "output_head.bias"
        if out_w_key in state_dict:
            old_w = state_dict[out_w_key]
            new_w = self.output_head.weight.data
            if old_w.shape[0] == 1 and new_w.shape[0] == 2:
                padded_w = torch.zeros_like(new_w)
                padded_w[0:1] = old_w  # copy prob channel, confidence stays zero
                state_dict[out_w_key] = padded_w
            if out_b_key in state_dict:
                old_b = state_dict[out_b_key]
                new_b = self.output_head.bias.data
                if old_b.shape[0] == 1 and new_b.shape[0] == 2:
                    padded_b = torch.zeros_like(new_b)
                    padded_b[0:1] = old_b
                    state_dict[out_b_key] = padded_b
                    print("  (Migrated output head: 1→2 channels, confidence initialized to zero)")

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
