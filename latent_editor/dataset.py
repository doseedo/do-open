"""
Boundary-repair dataset for Oobleck latent editor.

Each item simulates a sample-accurate splice in waveform space and produces:
    L_naive   [T, 64]  : naively-spliced latents (nearest frame, no fade)
    L_target  [T, 64]  : ground-truth latents = encode(splice in waveform)
    mask      [T]      : 1 where the boundary repair window lives
    phase     scalar   : sub-frame offset of the cut, in [0, 1)
    cut_frame int      : index of the boundary frame in L_naive

Latents are stored as [T, 64] @ VAE_HZ=25, audio @ 48 kHz, so
    SAMPLES_PER_FRAME = 48000 / 25 = 1920.

We never need raw audio: we decode existing latent files with the frozen
Oobleck VAE, perform the splice in waveform, and re-encode for the target.
"""
from __future__ import annotations
import glob, os, random
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset

VAE_HZ = 25
SR = 48000
SAMPLES_PER_FRAME = SR // VAE_HZ  # 1920


@dataclass
class EditorBatch:
    L_naive: torch.Tensor   # [B, T, 64]
    L_target: torch.Tensor  # [B, T, 64]
    mask: torch.Tensor      # [B, T] float
    phase: torch.Tensor     # [B] float in [0,1)
    cut_frame: torch.Tensor # [B] long
    # optional, for waveform-loss eval:
    wav_target: Optional[torch.Tensor] = None  # [B, 2, S_window]
    wav_window: Optional[tuple] = None         # (start_frame, end_frame)


def _list_latent_files(roots: List[str]) -> List[str]:
    files = []
    for r in roots:
        if os.path.isdir(r):
            files.extend(sorted(glob.glob(os.path.join(r, "**", "*.vae.pt"), recursive=True)))
        elif r.endswith(".pt"):
            files.append(r)
    return files


def _load_latent(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    lat = obj["latents"] if isinstance(obj, dict) else obj
    if lat.dim() == 3:  # [1, T, 64] or [1, 64, T]
        lat = lat.squeeze(0)
    if lat.shape[0] == 64 and lat.shape[1] != 64:
        lat = lat.transpose(0, 1)  # -> [T, 64]
    return lat.float()  # [T, 64]


class LatentSpliceDataset(Dataset):
    """
    Synthesizes one boundary-repair example per __getitem__ by:
      1. picking two latent files (a, b),
      2. picking a random window length (in frames) and a random sub-frame
         cut offset,
      3. decoding a + b → waveform with the *frozen* VAE,
      4. cutting at the exact sample, concatenating, re-encoding → L_target,
      5. building L_naive by nearest-frame splice of the originals.
    """

    def __init__(
        self,
        roots: List[str],
        vae,                       # frozen Oobleck VAE (handler.vae)
        win_frames: int = 64,      # crop length, ~2.56 s
        boundary_radius: int = 4,  # frames flagged in mask on each side of cut
        device: str = "cuda",
        seed: int = 0,
    ):
        self.files = _list_latent_files(roots)
        if len(self.files) < 2:
            raise RuntimeError(f"Need >=2 latent files; found {len(self.files)} in {roots}")
        self.vae = vae
        self.win = win_frames
        self.r = boundary_radius
        self.device = device
        self.rng = random.Random(seed)

    def __len__(self):
        return 10_000_000  # virtual; training loop controls steps

    @torch.no_grad()
    def _decode(self, lat_TC: torch.Tensor) -> torch.Tensor:
        # vae expects [B, 64, T] -> returns .sample [B, 2, S]
        x = lat_TC.transpose(0, 1).unsqueeze(0).to(self.device, torch.bfloat16)
        return self.vae.decode(x).sample[0].float().cpu()  # [2, S]

    @torch.no_grad()
    def _encode(self, wav_2S: torch.Tensor) -> torch.Tensor:
        x = wav_2S.unsqueeze(0).to(self.device, torch.bfloat16)
        out = self.vae.encode(x)
        lat = out.latent_dist.sample() if hasattr(out, "latent_dist") else out.sample
        return lat[0].transpose(0, 1).float().cpu()  # [T, 64]

    def _pick_crop(self, lat: torch.Tensor, n: int) -> torch.Tensor:
        T = lat.shape[0]
        if T <= n:
            pad = torch.zeros(n - T, 64)
            return torch.cat([lat, pad], 0)
        s = self.rng.randint(0, T - n)
        return lat[s : s + n].clone()

    def _safe_load(self, max_tries: int = 8) -> torch.Tensor:
        for _ in range(max_tries):
            f = self.rng.choice(self.files)
            try:
                lat = _load_latent(f)
                if lat.shape[0] >= 2 and lat.shape[1] == 64:
                    return lat
            except Exception:
                continue
        raise RuntimeError("could not load any latent after retries")

    def __getitem__(self, idx):
        # 1. pick two clips (robust to corrupt files in /Latents2)
        La = self._safe_load()
        Lb = self._safe_load()

        half = self.win // 2
        La = self._pick_crop(La, self.win)
        Lb = self._pick_crop(Lb, self.win)

        # 2. cut frame and sub-frame phase
        cut_frame = half
        phase = self.rng.random()  # [0, 1)
        sub = int(round(phase * SAMPLES_PER_FRAME))

        # 3. decode both
        Wa = self._decode(La)  # [2, win*1920]
        Wb = self._decode(Lb)
        S = self.win * SAMPLES_PER_FRAME
        cut_sample = cut_frame * SAMPLES_PER_FRAME + sub

        # 4. waveform-domain splice -> ground-truth latents
        Wt = torch.cat([Wa[:, :cut_sample], Wb[:, cut_sample:S]], dim=1)
        # pad/truncate to S to keep encoder input length consistent
        if Wt.shape[1] < S:
            Wt = torch.nn.functional.pad(Wt, (0, S - Wt.shape[1]))
        else:
            Wt = Wt[:, :S]
        L_target = self._encode(Wt)  # [T_enc, 64]
        # encoder may return win or win+/-1; align to self.win
        if L_target.shape[0] >= self.win:
            L_target = L_target[: self.win]
        else:
            L_target = torch.cat(
                [L_target, torch.zeros(self.win - L_target.shape[0], 64)], 0
            )

        # 5. naive frame-aligned splice
        L_naive = torch.cat([La[:cut_frame], Lb[cut_frame:]], dim=0)

        # boundary mask
        mask = torch.zeros(self.win)
        lo = max(0, cut_frame - self.r)
        hi = min(self.win, cut_frame + self.r + 1)
        mask[lo:hi] = 1.0

        return {
            "L_naive": L_naive,
            "L_target": L_target,
            "mask": mask,
            "phase": torch.tensor(phase, dtype=torch.float32),
            "cut_frame": torch.tensor(cut_frame, dtype=torch.long),
            "wav_target": Wt,  # for STFT loss around boundary
        }


class CachedSpliceDataset(Dataset):
    """Loads all shards into memory once, then serves from a flat list.

    50k examples × ~80KB each ≈ 4 GB — fits comfortably. Loading once in
    the main process and using num_workers=0 avoids the multi-worker shard
    re-init storm we saw earlier.
    """

    def __init__(self, cache_dir: str):
        self.shard_paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.pt")))
        if not self.shard_paths:
            raise RuntimeError(f"No shards in {cache_dir}")
        self.items = []
        for p in self.shard_paths:
            shard = torch.load(p, map_location="cpu", weights_only=False)
            self.items.extend(shard)
        print(f"  loaded {len(self.items)} examples from {len(self.shard_paths)} shards")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    has_wav = "wav_target" in batch[0]
    return EditorBatch(
        L_naive=torch.stack([b["L_naive"] for b in batch]),
        L_target=torch.stack([b["L_target"] for b in batch]),
        mask=torch.stack([b["mask"] for b in batch]),
        phase=torch.stack([b["phase"] for b in batch]),
        cut_frame=torch.stack([b["cut_frame"] for b in batch]),
        wav_target=torch.stack([b["wav_target"] for b in batch]) if has_wav else None,
    )
