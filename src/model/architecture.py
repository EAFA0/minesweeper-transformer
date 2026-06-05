"""CNN + Transformer Refinement Architecture (V4).

Breaking change from V3: CNN runs ONCE, Transformer self-loops N times,
decode happens only at the end.  V3 checkpoints are NOT compatible.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import POLICY, ModelConfig


class CNNEncoder(nn.Module):
    """Convolutional frontend (10ch board → d_model feature map)."""

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
    """2D learnable PE with bilinear interpolation."""

    def __init__(self, d_model: int, ref_grid: int = 16):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, d_model, ref_grid, ref_grid) * 0.02)
        self.ref_grid = ref_grid

    def get_spatial(self, H: int, W: int) -> torch.Tensor:
        if H == self.ref_grid and W == self.ref_grid:
            return self.pe
        return F.interpolate(self.pe, size=(H, W), mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add PE to spatial features: (B, C, H, W) → (B, C, H, W)."""
        _, _, H, W = x.shape
        return x + self.get_spatial(H, W)

    def get_seq(self, H: int, W: int) -> torch.Tensor:
        """Return PE as sequence: (1, H*W, d_model)."""
        return self.get_spatial(H, W).flatten(2).transpose(1, 2)  # (1, H*W, d_model)


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
        # Removed final LayerNorm to avoid squashing confidence growth during loop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, C) → (B, S, C)"""
        x = self.encoder(x)
        return x


class DecoderHead(nn.Module):
    """1×1 Conv: d_model channels → 1 logit channel."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MinesweeperTransformer(nn.Module):
    """CNN (once) → Transformer self-loop (N×) → Decoder → P(mine).

    The CNN encodes the board into a feature map.  The Transformer then
    refines its internal high-dimensional memory state over multiple steps.
    Only the final memory state is decoded into probabilities.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        self.cnn = CNNEncoder(
            in_channels=config.in_channels,
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

        self.decoder = DecoderHead(config.d_model)

    def load_pretrained(self, checkpoint_path: str, device: torch.device):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict, strict=True)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ─── Core ops ────────────────────────────────────────────────────────

    def _extract_features(self, board: torch.Tensor) -> torch.Tensor:
        """CNN: board (B, 10, H, W) → features (B, d_model, H, W)."""
        return self.cnn(board)

    def _to_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial → sequence: (B, C, H, W) → (B, H*W, C)."""
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).transpose(1, 2)

    def _to_spatial(self, seq: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Sequence → spatial: (B, H*W, C) → (B, C, H, W)."""
        return seq.transpose(1, 2).reshape(seq.shape[0], seq.shape[2], H, W)

    def _transformer_step(self, mem_seq: torch.Tensor, features_seq: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """One Transformer self-loop step with external residual and feature injection.

        Args:
            mem_seq: (B, H*W, d_model) current memory sequence
            features_seq: (B, H*W, d_model) original CNN features for grounding
        Returns:
            new_mem_seq: (B, H*W, d_model)
        """
        pe_seq = self.pos_encoding.get_seq(H, W).to(mem_seq.device)
        # Inject original features and positional encoding into the current memory state
        x = mem_seq + pe_seq + features_seq
        
        # External residual connection to prevent state drift
        return mem_seq + self.transformer(x)

    def _init_memory(self, board: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Run CNN and first Transformer step. Returns (mem_seq, features_seq, H, W)."""
        _, _, H, W = board.shape
        features = self._extract_features(board)
        features_seq = self._to_seq(features)
        
        mem_seq = features_seq.clone()
        mem_seq = self._transformer_step(mem_seq, features_seq, H, W)
        
        return mem_seq, features_seq, H, W

    # ─── Public API ──────────────────────────────────────────────────────

    def forward(self, board: torch.Tensor,
                num_refine_steps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward: CNN → Transformer loop → Decoder.

        Returns (probs, mem_spatial) where probs is sigmoid(P(mine)).
        """
        if num_refine_steps is None:
            num_refine_steps = self.config.refinement_steps

        mem_seq, features_seq, H, W = self._init_memory(board)

        # 3. Refinement: Transformer self-loop
        for _step in range(num_refine_steps - 1):
            mem_seq = self._transformer_step(mem_seq, features_seq, H, W)

        # 4. Decode final memory to probabilities
        mem_spatial = self._to_spatial(mem_seq, H, W)        # (B, d_model, H, W)
        probs = torch.sigmoid(self.decoder(mem_spatial))     # (B, 1, H, W)

        return probs, mem_spatial

    def refine(self, board: torch.Tensor,
               num_steps: int = POLICY.refinement.eval_max_steps,
               convergence_eps: float = POLICY.refinement.convergence_eps,
               ) -> List[torch.Tensor]:
        """Iterative refinement with early-stopping for inference.

        Returns list of probs at each step (for convergence tracking).
        """
        mem_seq, features_seq, H, W = self._init_memory(board)

        mem_spatial = self._to_spatial(mem_seq, H, W)
        probs = torch.sigmoid(self.decoder(mem_spatial))
        results = [probs]
        prev_probs = probs

        # 3. Refinement
        for step in range(1, num_steps):
            mem_seq = self._transformer_step(mem_seq, features_seq, H, W)
            mem_spatial = self._to_spatial(mem_seq, H, W)
            probs = torch.sigmoid(self.decoder(mem_spatial))
            results.append(probs)

            if not self.training and step > 0:
                max_change = (probs - prev_probs).abs().max().item()
                if max_change < convergence_eps:
                    break
            if not self.training:
                prev_probs = probs.detach()
            else:
                prev_probs = probs

        return results

    @torch.no_grad()
    def predict(self, x: torch.Tensor,
                max_refine_steps: int = POLICY.refinement.eval_max_steps) -> torch.Tensor:
        """Return final P(mine) with adaptive refinement — mine channel only (B, 1, H, W)."""
        results = self.refine(x, num_steps=max_refine_steps)
        return results[-1][:, 0:1]  # (B, 1, H, W) — mine probs only

    @torch.no_grad()
    def diagnose_refinement(self, x: torch.Tensor,
                            max_refine_steps: int = POLICY.refinement.eval_max_steps,
                            convergence_eps: float = POLICY.refinement.convergence_eps,
                            ) -> dict:
        """Run refinement and report per-sample step statistics."""
        B = x.shape[0]

        mem_seq, features_seq, H, W = self._init_memory(x)

        mem_spatial = self._to_spatial(mem_seq, H, W)
        probs = torch.sigmoid(self.decoder(mem_spatial))

        n_steps = torch.ones(B, dtype=torch.long, device=x.device)
        done = torch.zeros(B, dtype=torch.bool, device=x.device)
        final_probs = probs.clone()
        prev_probs = probs

        for step in range(1, max_refine_steps):
            mem_seq = self._transformer_step(mem_seq, features_seq, H, W)
            mem_spatial = self._to_spatial(mem_seq, H, W)
            probs = torch.sigmoid(self.decoder(mem_spatial))

            max_change = (probs - prev_probs).abs().view(B, -1).max(dim=1).values
            converged_now = ~done & (max_change < convergence_eps)
            done = done | converged_now
            n_steps = torch.where(converged_now, step + 1, n_steps)
            final_probs = torch.where(converged_now.view(B, 1, 1, 1), probs, final_probs)

            if done.all():
                break
            prev_probs = probs.clone()

        n_steps = torch.where(~done, max_refine_steps, n_steps)
        final_probs = torch.where(~done.view(B, 1, 1, 1), probs, final_probs)

        return {
            "probs": final_probs,
            "n_steps": n_steps,
            "mean_steps": n_steps.float().mean().item(),
            "max_steps": n_steps.max().item(),
            "min_steps": n_steps.min().item(),
            "early_stop_rate": (n_steps < max_refine_steps).float().mean().item(),
            "step_distribution": torch.bincount(n_steps, minlength=max_refine_steps + 1).tolist(),
        }
