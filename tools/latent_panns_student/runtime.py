"""Inference wrapper for the latent-PANNs student.

Loads a trained checkpoint and returns AudioSet label + instrument-type
predictions from an Oobleck VAE latent.
"""
import os
import sys
from pathlib import Path
from typing import List

import torch

import importlib.util as _ilu
_here = os.path.dirname(os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location(
    "latent_panns_student._student_model",
    os.path.join(_here, "student_model.py"),
)
_sm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sm)
LatentPANNsStudent = _sm.LatentPANNsStudent


LATENT_FPS = 25
CHUNK_FRAMES = LATENT_FPS * 30                # 750


def _load_panns_labels():
    from panns_inference import labels
    return list(labels)


class LatentPANNsRuntime:
    _inst = None

    @classmethod
    def get(cls, ckpt_path: str, device: str = "cuda"):
        if cls._inst is None or cls._inst.ckpt_path != ckpt_path:
            cls._inst = cls(ckpt_path, device=device)
        return cls._inst

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.ckpt_path = ckpt_path
        self.device = device

        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = LatentPANNsStudent()
        self.model.load_state_dict(sd["model"])
        self.model.eval().to(device)

        self.labels = _load_panns_labels()

    @torch.no_grad()
    def scores(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [T,64] or [64,T] or [B,T,64] → [B, 527] probabilities."""
        if latent.dim() == 2:
            if latent.shape[0] == 64 and latent.shape[1] != 64:
                lat = latent.t()
            else:
                lat = latent
            lat = lat.unsqueeze(0)                      # [1, T, 64]
        elif latent.dim() == 3:
            lat = latent
            if lat.shape[1] == 64 and lat.shape[2] != 64:
                lat = lat.transpose(1, 2)
        else:
            raise ValueError(f"bad latent dim {latent.shape}")

        # pad / crop to one CHUNK_FRAMES window for now (server gives ≤30 s)
        T = lat.shape[1]
        if T < CHUNK_FRAMES:
            lat = torch.nn.functional.pad(
                lat, (0, 0, 0, CHUNK_FRAMES - T))
        elif T > CHUNK_FRAMES:
            lat = lat[:, :CHUNK_FRAMES]

        lat = lat.to(self.device).float()
        logits = self.model(lat)                        # [B, 527]
        return torch.sigmoid(logits)

    def top_k(self, latent: torch.Tensor, k: int = 5) -> List[dict]:
        probs = self.scores(latent)[0].cpu().tolist()
        idxs = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
        return [{"label": self.labels[i], "score": float(probs[i])}
                for i in idxs]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latent", required=True)
    args = ap.parse_args()
    rt = LatentPANNsRuntime(args.ckpt)
    raw = torch.load(args.latent, map_location="cpu", weights_only=False)
    lat = raw["latents"] if isinstance(raw, dict) else raw
    print("latent:", tuple(lat.shape))
    for r in rt.top_k(lat, k=10):
        print(f"  {r['score']:.3f}  {r['label']}")
