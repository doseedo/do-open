"""LatentDemucs student model runtime.

Loads `WaveformToFourStemLatents` from /scratch/latent_demucs/code/
and exposes a single function that takes a 48k stereo wav path and
returns a dict of {stem_name -> [T, 64] latent} ready to feed into
the rest of the latent-domain pipeline."""
from __future__ import annotations
import os, sys, time
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import soundfile as sf
import librosa

# The student's source code lives next to its checkpoint on this VM.
_DISTILL_CODE = "/scratch/latent_demucs/code"
_DISTILL_CKPT = "/scratch/latent_demucs/distill_final.pt"

if _DISTILL_CODE not in sys.path:
    sys.path.insert(0, _DISTILL_CODE)

STEM_NAMES = ("drums", "bass", "vocals", "other")
SR = 48000
SAMPLES_PER_FRAME = 1920


class LatentDemucsRuntime:
    """Lazy-loaded student that holds the model on GPU after first call."""

    _instance: "Optional[LatentDemucsRuntime]" = None

    @classmethod
    def get(cls) -> "LatentDemucsRuntime":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, ckpt_path: str = _DISTILL_CKPT, device: str = "cuda"):
        from distill_model import WaveformToFourStemLatents  # type: ignore
        self.device = device
        self.model = WaveformToFourStemLatents().to(device).eval()
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Checkpoint stores under "model"; some older runs use the
        # raw state dict at the top level.
        state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        self.model.load_state_dict(state)
        self.model = self.model.to(torch.bfloat16)
        print(f"[latent_demucs] loaded {ckpt_path} on {device}")

    @torch.no_grad()
    def separate(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Dict[str, torch.Tensor]:
        """audio: [N, 2] stereo float32 (or [N] mono — auto-duplicated).
        Returns {stem_name: [T, 64] cpu float32 latent}."""
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)
        if audio.shape[1] == 1:
            audio = np.concatenate([audio, audio], axis=1)
        if sr != SR:
            audio = np.stack([
                librosa.resample(audio[:, c].astype(np.float32), orig_sr=sr, target_sr=SR)
                for c in range(audio.shape[1])
            ], axis=1)
        # Pad to a multiple of SAMPLES_PER_FRAME so the encoder backbone
        # produces a clean integer number of latent frames.
        n = audio.shape[0]
        padded = ((n + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
        if padded != n:
            pad = np.zeros((padded - n, 2), dtype=np.float32)
            audio = np.concatenate([audio, pad], axis=0)
        x = torch.from_numpy(audio.T).float().unsqueeze(0).to(self.device).to(torch.bfloat16)
        # x: [1, 2, samples]
        out = self.model(x)   # [1, 4, 64, T]
        out = out.squeeze(0).float().cpu()  # [4, 64, T]
        return {
            name: out[i].transpose(0, 1).contiguous()  # [T, 64]
            for i, name in enumerate(STEM_NAMES)
        }


def latent_demucs_separate(audio_path: str) -> Dict[str, torch.Tensor]:
    """One-shot helper: load the wav, run the student, return per-stem
    latents in [T, 64] cpu float32 form."""
    audio, sr = sf.read(audio_path, always_2d=True, dtype="float32")
    rt = LatentDemucsRuntime.get()
    return rt.separate(audio, sr)
