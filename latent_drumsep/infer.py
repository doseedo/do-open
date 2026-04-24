"""Production runtime for the latent drum sub-separator.

Lazy singleton, mirrors latent_demucs.runtime.LatentDemucsRuntime.
"""
from __future__ import annotations
from typing import Dict, Optional
import torch

from .model import LatentDrumSubsep
from . import STEMS

_DEFAULT_CKPT = "/scratch/latent_drumsep_ckpts/drumsep_final.pt"


class LatentDrumsepRuntime:
    _instance: "Optional[LatentDrumsepRuntime]" = None

    @classmethod
    def get(cls, ckpt_path: str = _DEFAULT_CKPT) -> "LatentDrumsepRuntime":
        if cls._instance is None:
            cls._instance = cls(ckpt_path)
        return cls._instance

    def __init__(self, ckpt_path: str = _DEFAULT_CKPT, device: str = "cuda"):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        pos = ckpt["model"].get("pos")
        max_len = pos.shape[1] if pos is not None else 256
        self.max_len = max_len
        self.stems = tuple(ckpt.get("stems", STEMS))
        self.model = LatentDrumSubsep(n_stems=len(self.stems),
                                      max_len=max_len).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def split(self, L_drum: torch.Tensor) -> Dict[str, torch.Tensor]:
        """L_drum: [T, 64]  ->  {stem_name: [T, 64]}.

        Supports inputs longer than the trained max_len by processing
        non-overlapping chunks and concatenating stem outputs along the
        time axis.
        """
        x = L_drum.to(self.device).unsqueeze(0)        # [1, T, 64]
        T = x.shape[1]
        if T <= self.max_len:
            y = self.model(x)[0]                        # [S, T, 64]
        else:
            parts = []
            for s in range(0, T, self.max_len):
                e = min(s + self.max_len, T)
                chunk = x[:, s:e, :]
                yc = self.model(chunk)[0]               # [S, chunk_T, 64]
                parts.append(yc)
            y = torch.cat(parts, dim=1)                  # [S, T, 64]
        return {name: y[i].cpu() for i, name in enumerate(self.stems)}
