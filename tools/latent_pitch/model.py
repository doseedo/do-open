"""
Latent → MIDI student model.

Input:  latent sequence [B, T, 64] @ 25 Hz
Output: per-frame predictions
        onset_logits  [B, T, 128]
        frame_logits  [B, T, 128]
        velocity      [B, T, 128]   (sigmoid in [0,1])

Architecture: small 1D residual conv stack + a few transformer layers, then
three parallel heads. The conv stack handles the local timbral context the
VAE latents already encode; the transformer layers give global musical
context (chord/melody continuity).

Operates entirely on latents -- no decoding to audio. ~5-10× faster than
running BasicPitch on the decoded waveform.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c, k=5):
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


class LatentBasicPitchStudent(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        d_model: int = 256,
        n_conv_blocks: int = 4,
        n_tr_layers: int = 4,
        n_heads: int = 8,
        n_pitch: int = 128,
        max_len: int = 2048,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(latent_dim, d_model, 1)
        self.conv_stack = nn.Sequential(*[ResBlock(d_model) for _ in range(n_conv_blocks)])

        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_tr_layers)

        self.head_onset = nn.Linear(d_model, n_pitch)
        self.head_frame = nn.Linear(d_model, n_pitch)
        self.head_vel   = nn.Linear(d_model, n_pitch)
        # v2: sub-frame onset offset in [0,1) -- only meaningful where the
        # onset head fires. Predicts WHERE inside the 40 ms frame the
        # attack actually lands, so we get ~1 ms timing resolution despite
        # the 25 Hz frame grid.
        self.head_onset_offset = nn.Linear(d_model, n_pitch)

    def forward(self, latent: torch.Tensor):
        # latent: [B, T, 64]
        x = latent.transpose(1, 2)              # [B, 64, T]
        h = self.in_proj(x)
        h = self.conv_stack(h)
        h = h.transpose(1, 2)                   # [B, T, d]
        T = h.shape[1]
        h = h + self.pos[:, :T]
        h = self.tr(h)
        return {
            "onset_logits":  self.head_onset(h),
            "frame_logits":  self.head_frame(h),
            "velocity":      torch.sigmoid(self.head_vel(h)),
            "onset_offset":  torch.sigmoid(self.head_onset_offset(h)),  # [0,1) within frame
        }
