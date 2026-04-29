"""
Latent stretch-artifact cleaner: model.

Same architectural family as the boundary-repair LatentEditor, but:
  - no boundary mask (artifacts are everywhere → cleaning is global),
  - conditioning on the stretch factor r (so the model can specialize
    its denoising to the artifact regime, e.g. r ≈ 1 needs ~no edit while
    r far from 1 needs aggressive cleanup),
  - residual prediction with zero-init output → starts as identity, which
    is the right prior since round-trip with r=1 is exactly the input.

Larger than the editor (8 layers, d_model=384) because the cleanup task is
sequence-global, not local.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .model import sinusoidal_phase_embed  # reuse


class LatentStretchCleaner(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 8,
        max_len: int = 256,
    ):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.r_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # identity at init

    def forward(
        self,
        L_input: torch.Tensor,    # [B, T, 64]  artifact-laden
        stretch_r: torch.Tensor,  # [B]         stretch factor that produced L_input
    ) -> torch.Tensor:
        B, T, _ = L_input.shape
        h = self.in_proj(L_input) + self.pos[:, :T]
        # encode r as a normalized scalar in roughly [-1, 1] for stable embedding
        r_norm = torch.log(stretch_r.clamp_min(1e-3))  # log keeps r=1 → 0
        r_emb = self.r_mlp(sinusoidal_phase_embed(r_norm, h.shape[-1]))
        h = h + r_emb[:, None, :]
        h = self.tr(h)
        residual = self.out(h)
        return L_input + residual
