"""
Boundary-repair latent editor.

Small 1D transformer over [T, 64] Oobleck latents that predicts a *residual*
on top of L_naive. Conditioned on:
  - a boundary mask (which frames are within +/-R of the cut),
  - the sub-frame phase offset of the cut, in [0, 1).

Residual prediction means interior frames are pass-through by default,
which is the right inductive bias: only the few frames straddling the cut
need to change.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


def sinusoidal_phase_embed(phase: torch.Tensor, dim: int) -> torch.Tensor:
    # phase: [B] in [0,1)  ->  [B, dim]
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=phase.device) / max(half - 1, 1)
    )
    args = phase[:, None] * freqs[None] * 2 * math.pi
    return torch.cat([args.sin(), args.cos()], dim=-1)


class LatentEditor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        max_len: int = 256,
    ):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim + 1, d_model)  # +1 = mask channel
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.phase_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # start as identity (residual = 0)

    def forward(
        self,
        L_naive: torch.Tensor,  # [B, T, 64]
        mask: torch.Tensor,     # [B, T]
        phase: torch.Tensor,    # [B]
    ) -> torch.Tensor:
        B, T, _ = L_naive.shape
        x = torch.cat([L_naive, mask.unsqueeze(-1)], dim=-1)  # [B,T,65]
        h = self.in_proj(x) + self.pos[:, :T]
        phase_emb = self.phase_mlp(sinusoidal_phase_embed(phase, h.shape[-1]))
        h = h + phase_emb[:, None, :]
        h = self.tr(h)
        residual = self.out(h)            # [B, T, 64]
        # only allow edits inside the boundary region
        residual = residual * mask.unsqueeze(-1)
        return L_naive + residual
