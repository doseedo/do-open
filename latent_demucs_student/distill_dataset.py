"""Dataset for distillation: (mix waveform, 4 teacher stem latents).

Walks /scratch/mixesV7_latents for sessions with full teacher data, looks
for the matching flac at /scratch/mixesV7/<session>/full_mix.flac (a
parallel local mirror that must be synced separately), and yields:

  audio:  [2, N_samples] @ 48 kHz   (stereo, random crop)
  stems:  [4, 64, T]                 (drums, bass, vocals, other)

Crop is on a per-frame basis to keep audio and latent aligned: T frames
of latent → T*1920 audio samples (since latent fps = 25 → 1920 samples
per frame at 48 kHz).
"""
import os, random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset

LATENT_ROOT = Path("/scratch/mixesV7_latents")
WAV_ROOT    = Path("/scratch/mixesV7")
MUSDB_LATENT_ROOT = Path("/scratch/musdb18_latents")
MUSDB_WAV_ROOT    = Path("/scratch/musdb18_wavs")
STEMS = ["drums", "bass", "vocals", "other"]
SAMPLE_RATE = 48000
LATENT_FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // LATENT_FPS  # 1920


def _load_latent(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    z = raw["latents"] if isinstance(raw, dict) else raw
    if z.dim() == 2 and z.shape[0] == 64:
        z = z.t()           # [T, 64]
    return z.float()


def _load_audio_48k_stereo(flac_path):
    audio_np, sr = sf.read(str(flac_path), dtype="float32")
    audio = torch.from_numpy(audio_np.T if audio_np.ndim > 1 else audio_np[None])
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio                # [2, N]


class DistillDataset(Dataset):
    def __init__(self, crop_frames=200, seed=0):
        self.crop_frames = crop_frames
        self.crop_samples = crop_frames * SAMPLES_PER_FRAME
        self.rng = random.Random(seed)

        self.items = []
        # Source 1: mixesV7 (demucs-teacher latents next to full_mix.flac)
        n_v7 = 0
        for full in LATENT_ROOT.rglob("full_mix.vae.pt"):
            if not all((full.parent / f"teacher_{s}.vae.pt").exists() for s in STEMS):
                continue
            rel = full.parent.relative_to(LATENT_ROOT)
            flac = WAV_ROOT / rel / "full_mix.flac"
            if not flac.exists():
                continue
            self.items.append({
                "src": "v7",
                "audio": flac,
                "stem_paths": [full.parent / f"teacher_{s}.vae.pt" for s in STEMS],
            })
            n_v7 += 1
        # Source 2: MUSDB18-HQ (real GT stems, mixture.wav extracted from zip)
        n_musdb = 0
        for mixlat in MUSDB_LATENT_ROOT.rglob("mixture.vae.pt"):
            track_dir = mixlat.parent
            rel = track_dir.relative_to(MUSDB_LATENT_ROOT)
            wav = MUSDB_WAV_ROOT / rel / "mixture.wav"
            if not wav.exists():
                continue
            stem_paths = [track_dir / f"{s}.vae.pt" for s in STEMS]
            if not all(p.exists() for p in stem_paths):
                continue
            self.items.append({
                "src": "musdb",
                "audio": wav,
                "stem_paths": stem_paths,
            })
            n_musdb += 1
        print(f"[distill dataset] v7={n_v7} musdb={n_musdb} total={len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        try:
            audio = _load_audio_48k_stereo(it["audio"])            # [2, N]
            stems = [_load_latent(p) for p in it["stem_paths"]]
        except Exception:
            return self.__getitem__((idx + 1) % len(self.items))

        # Frames available
        T_lat = min([s.shape[0] for s in stems])
        T_aud = audio.shape[1] // SAMPLES_PER_FRAME
        T_avail = min(T_lat, T_aud)
        if T_avail <= self.crop_frames:
            # Pad
            pad_frames = self.crop_frames - T_avail
            pad_samples = pad_frames * SAMPLES_PER_FRAME
            audio_w = F.pad(audio[:, :T_avail * SAMPLES_PER_FRAME], (0, pad_samples))
            stems_w = torch.stack([
                F.pad(s[:T_avail], (0, 0, 0, pad_frames)) for s in stems
            ])
        else:
            start_f = self.rng.randint(0, T_avail - self.crop_frames)
            end_f = start_f + self.crop_frames
            audio_w = audio[:, start_f * SAMPLES_PER_FRAME : end_f * SAMPLES_PER_FRAME]
            stems_w = torch.stack([s[start_f:end_f] for s in stems])  # [4, T, 64]

        # → audio [2, N_samples], stems [4, 64, T]
        return audio_w.contiguous(), stems_w.transpose(1, 2).contiguous()


def collate(batch):
    audio = torch.stack([b[0] for b in batch])    # [B, 2, N]
    stems = torch.stack([b[1] for b in batch])    # [B, 4, 64, T]
    return audio, stems
