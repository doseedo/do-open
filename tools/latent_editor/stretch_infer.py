"""
Production inference for the latent stretch cleaner.

Pipeline:
    1. waveform = vae.decode(L_in)
    2. wav_stretched = librosa.effects.time_stretch(waveform, rate=1/r)
       (rate<1 → longer; rate>1 → shorter; we follow librosa's convention)
    3. L_stretched_dirty = vae.encode(wav_stretched)
    4. L_clean = StretchCleaner(L_stretched_dirty, r)
    5. return L_clean

The cleaner is conditioned on r so it knows how much artifact to expect.
"""
from __future__ import annotations
import numpy as np
import torch
import librosa

from .stretch_model import LatentStretchCleaner
from .dataset import SAMPLES_PER_FRAME


class LatentStretchRuntime:
    def __init__(self, ckpt_path: str, vae, device: str = "cuda"):
        self.device = device
        self.vae = vae
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        pos = ckpt["model"].get("pos")
        max_len = pos.shape[1] if pos is not None else 256
        self.model = LatentStretchCleaner(max_len=max_len).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def stretch(
        self,
        L_in: torch.Tensor,    # [T, 64]
        target_frames: int,    # desired output length in latent frames
    ) -> torch.Tensor:
        T = L_in.shape[0]
        if target_frames == T:
            return L_in
        # decode
        x = L_in.transpose(0, 1).unsqueeze(0).to(self.device, torch.bfloat16)
        wav = self.vae.decode(x).sample[0].float().cpu().numpy()  # [2, S]

        # stretch in waveform with phase vocoder
        target_samples = target_frames * SAMPLES_PER_FRAME
        # librosa rate = original_len / target_len
        rate = wav.shape[1] / target_samples
        out = np.empty((wav.shape[0], target_samples), dtype=np.float32)
        for c in range(wav.shape[0]):
            y = librosa.effects.time_stretch(wav[c], rate=rate)
            if y.shape[0] < target_samples:
                y = np.pad(y, (0, target_samples - y.shape[0]))
            else:
                y = y[:target_samples]
            out[c] = y

        # re-encode → dirty latents
        wav_t = torch.from_numpy(out).unsqueeze(0).to(self.device, torch.bfloat16)
        enc = self.vae.encode(wav_t)
        L_dirty = (enc.latent_dist.sample() if hasattr(enc, "latent_dist") else enc.sample)
        L_dirty = L_dirty[0].transpose(0, 1).float()  # [T', 64]
        # length sanity
        if L_dirty.shape[0] != target_frames:
            if L_dirty.shape[0] > target_frames:
                L_dirty = L_dirty[:target_frames]
            else:
                pad = torch.zeros(target_frames - L_dirty.shape[0], 64, device=L_dirty.device)
                L_dirty = torch.cat([L_dirty, pad], 0)

        # clean with model
        r = torch.tensor([1.0 / rate], device=self.device, dtype=torch.float32)
        L_clean = self.model(L_dirty.unsqueeze(0).to(self.device), r)[0]
        return L_clean.cpu()
