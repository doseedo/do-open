"""
Latent stretch-artifact cleaner: dataset.

Core insight (from arlo): the training pair is INVERSE of the obvious one.
We never train with an "original waveform input → stretched audio output"
pair, because that puts artifacts in the target and the model learns to
*synthesize* artifacts. Instead:

    INPUT  = round-trip-stretched latent  (artifact-laden, length L)
    TARGET = original latent              (clean, length L)

Round-trip = stretch by r, then stretch by 1/r → length is preserved, but
the waveform now carries two passes of phase-vocoder artifacts. The model
learns to *remove* stretch artifacts. At inference time you do a single
stretch (cleaner than what the model trained on) and the model removes its
artifacts → clean stretched output.

This guarantees the target is always dry, never WSOLA-tainted.
"""
from __future__ import annotations
import glob, os, random
from typing import List

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset

from .dataset import VAE_HZ, SR, SAMPLES_PER_FRAME, _list_latent_files, _load_latent


class LatentStretchDataset(Dataset):
    """
    Each item:
        L_input  [T, 64]   round-trip-stretched, artifact-laden latents
        L_target [T, 64]   clean original latents (same T)
        stretch_r scalar   stretch factor that was applied (∈ [r_min, r_max])

    We crop random windows from a random latent file, decode → waveform,
    apply librosa.effects.time_stretch by factor r, then by 1/r (to restore
    length), re-encode → L_input. The same waveform window encoded directly
    is L_target (we re-encode rather than reuse the source latents to keep
    the encoder distribution identical between input and target).
    """

    def __init__(
        self,
        roots: List[str],
        vae,
        win_frames: int = 64,
        r_min: float = 0.6,
        r_max: float = 1.7,
        device: str = "cuda",
        seed: int = 0,
    ):
        self.files = _list_latent_files(roots)
        if len(self.files) < 1:
            raise RuntimeError(f"No latent files in {roots}")
        self.vae = vae
        self.win = win_frames
        self.r_min, self.r_max = r_min, r_max
        self.device = device
        self.rng = random.Random(seed)

    def __len__(self):
        return 10_000_000  # virtual

    @torch.no_grad()
    def _decode(self, lat_TC: torch.Tensor) -> torch.Tensor:
        x = lat_TC.transpose(0, 1).unsqueeze(0).to(self.device, torch.bfloat16)
        return self.vae.decode(x).sample[0].float().cpu()  # [2, S]

    @torch.no_grad()
    def _encode(self, wav_2S: torch.Tensor) -> torch.Tensor:
        x = wav_2S.unsqueeze(0).to(self.device, torch.bfloat16)
        out = self.vae.encode(x)
        lat = out.latent_dist.sample() if hasattr(out, "latent_dist") else out.sample
        return lat[0].transpose(0, 1).float().cpu()  # [T, 64]

    def _crop(self, lat: torch.Tensor, n: int) -> torch.Tensor:
        T = lat.shape[0]
        if T <= n:
            return torch.cat([lat, torch.zeros(n - T, 64)], 0)
        s = self.rng.randint(0, T - n)
        return lat[s : s + n].clone()

    def _stretch_round_trip(self, wav_2S: np.ndarray, r: float) -> np.ndarray:
        """Phase-vocoder round-trip; preserves length, doubles the artifacts."""
        out = np.empty_like(wav_2S)
        for c in range(wav_2S.shape[0]):
            y = librosa.effects.time_stretch(wav_2S[c], rate=r)        # length L/r
            y = librosa.effects.time_stretch(y, rate=1.0 / r)          # back to ~L
            # length-correct (phase vocoder rounds frames; trim/pad to L)
            if y.shape[0] < wav_2S.shape[1]:
                y = np.pad(y, (0, wav_2S.shape[1] - y.shape[0]))
            else:
                y = y[: wav_2S.shape[1]]
            out[c] = y
        return out

    def __getitem__(self, idx):
        for _ in range(8):
            try:
                L = _load_latent(self.rng.choice(self.files))
                if L.shape[0] >= 2 and L.shape[1] == 64:
                    break
            except Exception:
                continue
        else:
            raise RuntimeError("could not load any stretch latent after retries")
        L = self._crop(L, self.win)

        # decode original window once
        wav = self._decode(L).numpy()  # [2, win*1920]
        S = wav.shape[1]

        # round-trip stretch
        r = self.rng.uniform(self.r_min, self.r_max)
        wav_rt = self._stretch_round_trip(wav, r)

        # encode both through the SAME encoder so target distribution matches
        L_target = self._encode(torch.from_numpy(wav).float())
        L_input  = self._encode(torch.from_numpy(wav_rt).float())

        # align lengths to self.win
        def fix(lat):
            if lat.shape[0] >= self.win:
                return lat[: self.win]
            return torch.cat([lat, torch.zeros(self.win - lat.shape[0], 64)], 0)
        L_target = fix(L_target)
        L_input = fix(L_input)

        return {
            "L_input": L_input,
            "L_target": L_target,
            "stretch_r": torch.tensor(r, dtype=torch.float32),
        }


def stretch_collate(batch):
    return {
        "L_input":   torch.stack([b["L_input"]   for b in batch]),
        "L_target":  torch.stack([b["L_target"]  for b in batch]),
        "stretch_r": torch.stack([b["stretch_r"] for b in batch]),
    }


class CachedStretchDataset(Dataset):
    def __init__(self, cache_dir: str, lru_shards: int = 4):
        self.shard_paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.pt")))
        if not self.shard_paths:
            raise RuntimeError(f"No shards in {cache_dir}")
        self.index = []
        for sidx, p in enumerate(self.shard_paths):
            shard = torch.load(p, map_location="cpu", weights_only=False)
            for j in range(len(shard)):
                self.index.append((sidx, j))
        self._lru: dict = {}
        self._lru_cap = lru_shards

    def _shard(self, sidx: int):
        if sidx in self._lru:
            return self._lru[sidx]
        if len(self._lru) >= self._lru_cap:
            self._lru.pop(next(iter(self._lru)))
        s = torch.load(self.shard_paths[sidx], map_location="cpu", weights_only=False)
        self._lru[sidx] = s
        return s

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sidx, j = self.index[i]
        return self._shard(sidx)[j]
