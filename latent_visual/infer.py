"""Production runtime for the latent → peak-envelope model.

Lazy singleton, mirrors latent_demucs.runtime.LatentDemucsRuntime /
latent_drumsep.infer.LatentDrumsepRuntime.

The model maps a [T, 64] VAE latent to a [2, T] (min, max) amplitude
envelope — effectively a per-frame RMS/loudness curve at the native
latent frame rate (25 fps = 40 ms/frame). Used as an onset signal for
drum stem latents: peak-to-peak = (max - min), then peak-pick its
first difference to find transients.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn


_DEFAULT_CKPT = "/scratch/latent_visual_ckpts/latent_visual_final.pt"


class LatentToPeakEnvelope(nn.Module):
    """[B, 64, T] → [B, 2, T] (min, max) amplitude envelope. ~62K params."""

    def __init__(self, in_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, 2, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class LatentVisualRuntime:
    _instance: "Optional[LatentVisualRuntime]" = None

    @classmethod
    def get(cls, ckpt_path: str = _DEFAULT_CKPT) -> "LatentVisualRuntime":
        if cls._instance is None:
            cls._instance = cls(ckpt_path)
        return cls._instance

    def __init__(self, ckpt_path: str = _DEFAULT_CKPT, device: str = "cuda"):
        self.device = device
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = LatentToPeakEnvelope().to(device)
        self.model.load_state_dict(sd["model"])
        self.model.eval()

    @torch.no_grad()
    def envelope(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [T, 64] or [B, T, 64] or [B, 64, T] → [2, T] or [B, 2, T]
        of (min, max) amplitude bounds per frame."""
        L = latent.to(self.device)
        if L.dim() == 2:
            # [T, 64] → [1, 64, T]
            L = L.transpose(0, 1).unsqueeze(0)
            out = self.model(L.float())  # [1, 2, T]
            return out.squeeze(0).cpu()
        if L.dim() == 3 and L.shape[-1] == 64:
            # [B, T, 64] → [B, 64, T]
            L = L.transpose(1, 2)
        return self.model(L.float()).cpu()

    @torch.no_grad()
    def onsets(
        self,
        latent: torch.Tensor,
        delta: float = 0.03,
        pre_max: int = 1,
        post_max: int = 1,
        wait: int = 2,
    ) -> np.ndarray:
        """Peak-pick onsets from the envelope's first difference.

        latent: [T, 64] drum-stem latent.
        delta:  minimum positive spike magnitude to count as an onset.
        pre_max/post_max: local-max window (frames) around candidate.
        wait:   minimum frames between consecutive onsets.

        Returns: 1-D int array of onset frame indices (in latent frames,
        at 25 fps).
        """
        env = self.envelope(latent)                 # [2, T]
        ptp = (env[1] - env[0]).float().numpy()     # [T]
        # First difference: positive = rising edge = onset candidate
        d = np.diff(ptp, prepend=ptp[0])            # [T]
        d = np.clip(d, 0.0, None)

        onsets = []
        T = d.shape[0]
        last = -(wait + 1)
        for i in range(T):
            if d[i] < delta:
                continue
            lo = max(0, i - pre_max)
            hi = min(T, i + post_max + 1)
            if d[i] < d[lo:hi].max():
                continue
            if i - last < wait:
                continue
            onsets.append(i)
            last = i
        return np.asarray(onsets, dtype=np.int64)
