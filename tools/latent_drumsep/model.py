"""Latent drum sub-separator model.

Input :  L_drum   [B, T, 64]   single-source drum latent
Output:  L_stems  [B, n_stems, T, 64]   sub-stem latents

Architecture is a small transformer over the time axis, identical
shape-wise to latent_editor.stretch_model.LatentStretchCleaner. The
output head is initialized to zero so that at step 0 the network
returns n_stems IDENTICAL copies of the input — a sane starting point
to break from. Same trick as latent_demucs/distill_model.py.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from . import STEMS


class LatentDrumSubsep(nn.Module):
    def __init__(
        self,
        n_stems: int = len(STEMS),
        latent_dim: int = 64,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 8,
        max_len: int = 256,
    ):
        super().__init__()
        self.n_stems = n_stems
        self.latent_dim = latent_dim

        self.in_proj = nn.Linear(latent_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        # output the per-stem RESIDUAL away from "n copies of input".
        self.out = nn.Linear(d_model, n_stems * latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, L_drum: torch.Tensor) -> torch.Tensor:
        """L_drum: [B, T, 64]  →  [B, n_stems, T, 64]"""
        B, T, D = L_drum.shape
        h = self.in_proj(L_drum) + self.pos[:, :T]
        h = self.tr(h)
        residual = self.out(h).view(B, T, self.n_stems, D)  # [B, T, S, 64]
        # broadcast: at init, residual=0 → output = n copies of L_drum
        out = L_drum.unsqueeze(2) + residual                # [B, T, S, 64]
        return out.permute(0, 2, 1, 3).contiguous()         # [B, S, T, 64]


if __name__ == "__main__":
    m = LatentDrumSubsep()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"params: {n:.1f}M")
    x = torch.randn(2, 64, 64)
    y = m(x)
    print(f"in {tuple(x.shape)} → out {tuple(y.shape)}")
    # identity-init sanity: every stem should equal the input
    assert torch.allclose(y[:, 0], x), "stem 0 != input at init"
    assert torch.allclose(y[:, -1], x), "last stem != input at init"
    print("identity-init OK")
