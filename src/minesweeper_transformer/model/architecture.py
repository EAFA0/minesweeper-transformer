"""Hidden State Refinement Architecture (V2) for Minesweeper.

Architecture:
    Input: (B, 10, H, W) — covered, flagged, numbers 1-8 one-hot
    
    Iterative Refinement (Hidden State Loop):
        Step 0: mem_state = zeros(B, d_model, H, W)
        Step k: 
            concat(board, mem_state) -> CNN -> PE -> Transformer -> mem_state_{k}
            
    Decoder (Translation):
        At any step k, the hidden mem_state can be translated to probabilities:
        mem_state_{k} -> 1x1 Conv (Decoder) -> (B, 1, H, W) -> Sigmoid -> P(mine)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from minesweeper_transformer.config import POLICY

@dataclass
class ModelConfig:
    """Configuration for MinesweeperTransformer."""
    # Input
    in_channels: int = 10       # covered + flagged + 8 number channels
    
    # Hidden State Memory
    hidden_channels: int = 64   # Channels for the recurrent mem_state

    # CNN frontend
    cnn_channels: int = 64      # output channels of CNN (should match hidden_channels)
    cnn_layers: int = 3         # number of Conv layers

    # Transformer
    d_model: int = 64           # must match cnn_channels
    nhead: int = 4              # attention heads
    num_layers: int = 4         # transformer encoder layers
    dim_feedforward: int = 256  # FFN hidden dim
    dropout: float = 0.2

    # Positional encoding
    pe_grid_size: int = 16      # PE is learned at 16×16, bilinear-interpolated to any H×W

    # Iterative refinement
    refinement_steps: int = POLICY.refinement.train_max_steps

    # Output
    num_classes: int = 1        # P(mine) logit

    def __post_init__(self):
        if self.cnn_channels != self.d_model:
            raise ValueError(
                f"cnn_channels ({self.cnn_channels}) must match d_model ({self.d_model})"
            )
        if self.hidden_channels != self.d_model:
            raise ValueError(
                f"hidden_channels ({self.hidden_channels}) must match d_model ({self.d_model})"
            )


class CNNEncoder(nn.Module):
    """Convolutional frontend that preserves spatial resolution.
    Takes concatenated [board, prev_probs] as input.
    """

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


class DecoderHead(nn.Module):
    """Translates high-dimensional memory state to probability map using 1x1 Conv."""
    
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # bias=True is critical here. It provides a global prior (e.g. board mine density).
        # When mem_state is 0, sigmoid(bias) gives the base probability of a cell being a mine.
        self.net = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MinesweeperTransformer(nn.Module):
    """CNN + Transformer model with Hidden State Refinement.
    
    Maintains a high-dimensional memory state across refinement steps,
    translating it to probability distributions only at the output via a Decoder.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # CNN: board input (10) + prev_probs (1)
        self.cnn = CNNEncoder(
            in_channels=config.in_channels + config.num_classes,
            out_channels=config.d_model,
            num_layers=config.cnn_layers,
        )

        # Positional encoding
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

        # Decoder: translates hidden memory state to probability map
        self.decoder = DecoderHead(config.d_model, num_classes=1)

    def load_pretrained(self, checkpoint_path: str, device: str | torch.device):
        """Load pretrained weights from a checkpoint.
        Strict matching is enforced to ensure architecture consistency.
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict, strict=True)

    def _single_pass(self, board: torch.Tensor, prev_probs: torch.Tensor, mem_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal: one forward pass updating BOTH the high-dimensional memory AND the probabilities.

        Args:
            board:      (B, 10, H, W) static board channels
            prev_probs: (B, 1, H, W)  probabilities from previous step
            mem_state:  (B, d_model, H, W) high-dimensional hidden memory from previous step

        Returns:
            new_probs:     (B, 1, H, W) updated probabilities
            new_mem_state: (B, d_model, H, W) updated high-dimensional memory
        """
        B, C, H, W = board.shape
        
        # 1. Local Interaction (CNN): Combine board with explicit 1D probabilities
        x = torch.cat([board, prev_probs], dim=1)  # (B, 11, H, W)
        local_features = self.cnn(x)

        # 2. Memory Injection: Combine CNN's new insights with Transformer's old high-dimensional memory
        combined_features = local_features + mem_state

        # 3. Global Reasoning (Transformer)
        #    Add positional encoding at token level to prevent PE accumulation
        #    across refinement steps (PE was previously on CNN output, which got
        #    baked into mem_state and doubled each iteration).
        seq = combined_features.flatten(2).transpose(1, 2)  # (B, H*W, d_model)
        _, _, H_feat, W_feat = combined_features.shape
        if H_feat == self.pos_encoding.ref_grid and W_feat == self.pos_encoding.ref_grid:
            pe = self.pos_encoding.pe
        else:
            pe = F.interpolate(
                self.pos_encoding.pe, size=(H_feat, W_feat),
                mode='bilinear', align_corners=False,
            )
        pe_seq = pe.flatten(2).transpose(1, 2)  # (1, H*W, d_model)
        seq = seq + pe_seq
        seq = self.transformer(seq)

        # 4. Extract new states
        new_mem_state = seq.transpose(1, 2).reshape(B, self.config.d_model, H, W)
        
        # 5. Translate memory to probabilities
        raw_logits = self.decoder(new_mem_state)
        new_probs = torch.sigmoid(raw_logits)
        
        return new_probs, new_mem_state

    def forward(self, board: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-pass forward (matches refinement step 1)."""
        B, _, H, W = board.shape
        
        mem_state = torch.zeros((B, self.config.hidden_channels, H, W), device=board.device)
        prev_probs = torch.full((B, 1, H, W), 0.5, device=board.device)
        
        # Single forward pass generating initial probabilities and initial memory
        probs, mem_state = self._single_pass(board, prev_probs, mem_state)
        
        return probs, mem_state

    def refine(self, board: torch.Tensor, num_steps: int = POLICY.refinement.eval_max_steps,
               convergence_eps: float = POLICY.refinement.convergence_eps) -> List[torch.Tensor]:
        """Iterative refinement maintaining high-dimensional hidden state.

        Args:
            board:           (B, 10, H, W) board channels
            num_steps:       max refinement iterations
            convergence_eps: stop when max|P_t - P_{t-1}| < this

        Returns:
            List of (B, 1, H, W) tensors — sigmoid(P(mine)) translated at each step.
        """
        B, _, H, W = board.shape
        
        # Initialize empty memory state
        mem_state = torch.zeros((B, self.config.hidden_channels, H, W), device=board.device)
        
        # We need to track the previous probability map to check for convergence
        prev_probs = torch.full((B, 1, H, W), 0.5, device=board.device)
        results = []

        for step in range(num_steps):
            # Run one step of combined memory + probability inference
            probs, mem_state = self._single_pass(board, prev_probs, mem_state)
            results.append(probs)

            # Early stop based on probability convergence
            if not self.training and step > 0:
                # Compare max absolute difference in probabilities
                max_change = (probs - prev_probs).abs().max().item()
                if max_change < convergence_eps:
                    break

            # Detach only in eval mode; training should not use refine()
            # (train_epoch has its own BPTT loop without detach).
            if not self.training:
                prev_probs = probs.detach()
            else:
                raise RuntimeError(
                    "refine() should not be called during training. "
                    "Use the BPTT loop in train_epoch() instead."
                )

        return results

    @torch.no_grad()
    def predict(self, x: torch.Tensor, max_refine_steps: int = POLICY.refinement.eval_max_steps) -> torch.Tensor:
        """Return P(mine) probabilities with adaptive refinement."""
        results = self.refine(x, num_steps=max_refine_steps)
        return results[-1]

    @torch.no_grad()
    def diagnose_refinement(self, x: torch.Tensor, max_refine_steps: int = POLICY.refinement.eval_max_steps,
                            convergence_eps: float = POLICY.refinement.convergence_eps) -> dict:
        """Run refinement and report per-sample step statistics."""
        B, _, H, W = x.shape
        
        mem_state = torch.zeros((B, self.config.hidden_channels, H, W), device=x.device)
        prev_probs = torch.full((B, 1, H, W), 0.5, device=x.device)
        
        n_steps = torch.zeros(B, dtype=torch.long, device=x.device)
        done = torch.zeros(B, dtype=torch.bool, device=x.device)
        
        final_probs = prev_probs.clone()

        for step in range(max_refine_steps):
            probs_new, mem_state = self._single_pass(x, prev_probs, mem_state)

            if step > 0:
                max_change = (probs_new - prev_probs).abs().view(B, -1).max(dim=1).values
                converged_now = ~done & (max_change < convergence_eps)
                done = done | converged_now
                n_steps = torch.where(converged_now, step + 1, n_steps)
                
                # Capture final probs for newly converged samples
                final_probs = torch.where(converged_now.view(B, 1, 1, 1), probs_new, final_probs)

                if done.all():
                    break
                    
            prev_probs = probs_new.clone()

        # Handle samples that hit max_refine_steps without converging
        n_steps = torch.where(~done, max_refine_steps, n_steps)
        final_probs = torch.where(~done.view(B, 1, 1, 1), probs_new, final_probs)

        return {
            "probs": final_probs,
            "n_steps": n_steps,
            "mean_steps": n_steps.float().mean().item(),
            "max_steps": n_steps.max().item(),
            "min_steps": n_steps.min().item(),
            "early_stop_rate": (n_steps < max_refine_steps).float().mean().item(),
            "step_distribution": torch.bincount(n_steps, minlength=max_refine_steps + 1).tolist(),
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
