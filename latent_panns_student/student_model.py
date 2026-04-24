"""Latent → PANNs student.

Input:  Oobleck VAE latents [B, T, 64] @ 25 Hz  (or [B, 64, T] — accepted both ways)
Output: [B, n_classes] sigmoid-ready logits matching PANNs CNN14 clipwise output.

Architecture:
  Conv stem      (64 → d_model) — one 1×1 proj + GELU
  ResBlocks      (GroupNorm + GELU) ×2   — local timbral context
  Transformer    encoder layers ×4       — global aggregate
  Global avg pool over time
  Linear head    → n_classes             — matches PANNs AudioSet (527)

Distilled from PANNs CNN14 (panns_inference package).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c: int, k: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, k, padding=k // 2)
        self.conv2 = nn.Conv1d(c, c, k, padding=k // 2)
        self.norm1 = nn.GroupNorm(8, c)
        self.norm2 = nn.GroupNorm(8, c)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class LatentPANNsStudent(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        d_model: int = 256,
        n_conv_blocks: int = 2,
        n_tr_layers: int = 4,
        n_heads: int = 8,
        n_classes: int = 527,
        max_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes  = n_classes

        self.in_proj = nn.Conv1d(latent_dim, d_model, 1)
        self.act = nn.GELU()
        self.conv_stack = nn.Sequential(
            *[ResBlock(d_model) for _ in range(n_conv_blocks)]
        )

        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_tr_layers)
        self.ln_out = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, n_classes)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [B, T, 64] or [B, 64, T] → logits [B, n_classes]."""
        if latent.dim() != 3:
            raise ValueError(f"expected 3D latent, got {latent.shape}")
        if latent.shape[1] == self.latent_dim:
            x = latent                                    # [B, 64, T]
        else:
            x = latent.transpose(1, 2)                    # [B, 64, T]

        h = self.act(self.in_proj(x))                     # [B, D, T]
        h = self.conv_stack(h)                            # [B, D, T]
        h = h.transpose(1, 2)                             # [B, T, D]
        T = h.shape[1]
        h = h + self.pos[:, :T]
        h = self.tr(h)
        h = self.ln_out(h)
        h = h.mean(dim=1)                                 # [B, D] global avg pool
        return self.head(h)                               # [B, n_classes]

    @torch.no_grad()
    def predict(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(latent))


if __name__ == "__main__":
    m = LatentPANNsStudent()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    for shape in ((2, 750, 64), (2, 64, 750), (1, 300, 64)):
        x = torch.randn(*shape)
        y = m(x)
        print(f"{n:.1f}M  in {tuple(x.shape)} → out {tuple(y.shape)}")
