"""Dataset for 6-stem distillation: (mix waveform, 6 stem latents, mask).

Sources:
  (a) mixesV7 with `stem6_*.vae.pt` files (GT stems from Latents2, grouped
      into 6 categories by build_stem6_targets.py)
  (b) MUSDB18-HQ with split 'other' stem (real GT for drums/bass/vocals,
      htdemucs_6s split for guitar/piano/other_split)

Stem order (6): drums, bass, other, vocals, guitar, piano
              (matches htdemucs_6s.sources output order)

Each item yields:
  audio:  [2, N_samples] @ 48 kHz (stereo waveform crop)
  stems:  [6, 64, T]               (target latents per stem)
  mask:   [6]                       (1.0 if stem present, 0.0 if absent)
"""
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset

# ── Paths ────────────────────────────────────────────────────────────
V7_LATENT_ROOT = Path("/scratch/mixesV7_latents")
V7_WAV_ROOT = Path("/scratch/mixesV7")
MUSDB_LATENT_ROOT = Path("/scratch/musdb18_latents")
MUSDB_WAV_ROOT = Path("/scratch/musdb18_wavs")
STEM_INDEX_PATH = Path("/scratch/latent_demucs_student/stem_group_index.pt")
STEM_POOL_DIR = Path("/scratch/latent_demucs_student/stem_pool")

STEMS_6 = ["drums", "bass", "other", "vocals", "guitar", "piano"]
SAMPLE_RATE = 48000
LATENT_FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // LATENT_FPS  # 1920


def _load_latent(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    z = raw["latents"] if isinstance(raw, dict) else raw
    if z.dim() == 2 and z.shape[0] == 64:
        z = z.t()
    return z.float()


def _load_audio_48k_stereo(p):
    audio_np, sr = sf.read(str(p), dtype="float32")
    audio = torch.from_numpy(audio_np.T if audio_np.ndim > 1 else audio_np[None])
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio


class DistillDataset6(Dataset):
    def __init__(self, crop_frames=200, seed=0, use_musdb=True, use_v7=True):
        self.crop_frames = crop_frames
        self.crop_samples = crop_frames * SAMPLES_PER_FRAME
        self.rng = random.Random(seed)
        self.items = []

        # (a) mixesV7 with stem6_*.vae.pt (GT stems from build_stem6_targets)
        n_v7 = 0
        if use_v7:
            for full in V7_LATENT_ROOT.rglob("full_mix.vae.pt"):
                sess = full.parent
                # Need at least 2 stem6 files
                stem_paths = {}
                mask = []
                for s in STEMS_6:
                    p = sess / f"stem6_{s}.vae.pt"
                    if p.exists():
                        stem_paths[s] = p
                        mask.append(1)
                    else:
                        stem_paths[s] = None
                        mask.append(0)
                if sum(mask) < 2:
                    continue
                # Need matching flac
                rel = sess.relative_to(V7_LATENT_ROOT)
                flac = V7_WAV_ROOT / rel / "full_mix.flac"
                if not flac.exists():
                    continue
                self.items.append({
                    "src": "v7",
                    "audio": flac,
                    "stem_paths": [stem_paths[s] for s in STEMS_6],
                    "mask": mask,
                })
                n_v7 += 1

        # (b) MUSDB18-HQ with split 'other' → guitar/piano/other_split
        # Real GT: drums, bass, vocals. htdemucs_6s split: guitar, piano, other_split.
        # Stem mapping: drums=drums, bass=bass, other=other_split, vocals=vocals,
        #               guitar=guitar, piano=piano
        MUSDB_STEM_FILES = {
            "drums": "drums.vae.pt",
            "bass": "bass.vae.pt",
            "other": "other_split.vae.pt",
            "vocals": "vocals.vae.pt",
            "guitar": "guitar.vae.pt",
            "piano": "piano.vae.pt",
        }
        n_musdb = 0
        if use_musdb and MUSDB_LATENT_ROOT.exists():
            for mix_lat in MUSDB_LATENT_ROOT.rglob("mixture.vae.pt"):
                track_dir = mix_lat.parent
                rel = track_dir.relative_to(MUSDB_LATENT_ROOT)
                wav = MUSDB_WAV_ROOT / rel / "mixture.wav"
                if not wav.exists():
                    continue
                # Check which stems exist
                stem_paths = {}
                mask = []
                for s in STEMS_6:
                    p = track_dir / MUSDB_STEM_FILES[s]
                    if p.exists():
                        stem_paths[s] = p
                        mask.append(1)
                    else:
                        stem_paths[s] = None
                        mask.append(0)
                if sum(mask) < 2:
                    continue
                self.items.append({
                    "src": "musdb",
                    "audio": wav,
                    "stem_paths": [stem_paths[s] for s in STEMS_6],
                    "mask": mask,
                })
                n_musdb += 1

        # (c) MUSDB augmented: drums/bass/vocals from MUSDB GT wavs,
        #     guitar/piano/other from pre-decoded Latents2 stem pool.
        #     Mixture is rebuilt by summing all 6 stem wavs — no mismatch.
        n_musdb_aug = 0
        self.stem_pool = {}  # category → [(wav_path, latent_path), ...]
        if use_musdb and STEM_POOL_DIR.exists():
            for cat in ["guitar", "piano", "other"]:
                cat_dir = STEM_POOL_DIR / cat
                if not cat_dir.exists():
                    continue
                pairs = []
                for wp in sorted(cat_dir.glob("*.wav")):
                    lp = wp.parent / (wp.stem + ".latent.pt")
                    if lp.exists():
                        pairs.append((str(wp), str(lp)))
                self.stem_pool[cat] = pairs

            pool_counts = {c: len(v) for c, v in self.stem_pool.items()}
            has_pool = all(len(v) > 0 for v in self.stem_pool.values())

            if has_pool:
                # For each MUSDB track with individual stem wavs, create augmented items
                for split in ["train", "test"]:
                    split_dir = MUSDB_WAV_ROOT / split
                    if not split_dir.exists():
                        continue
                    for track_dir in sorted(split_dir.iterdir()):
                        if not track_dir.is_dir():
                            continue
                        # Need drums.wav, bass.wav, vocals.wav
                        stem_wavs = {}
                        for s in ["drums", "bass", "vocals"]:
                            p = track_dir / f"{s}.wav"
                            if p.exists():
                                stem_wavs[s] = str(p)
                        if len(stem_wavs) < 2:
                            continue
                        # Also need the GT latents for drums/bass/vocals
                        lat_dir = MUSDB_LATENT_ROOT / split / track_dir.name
                        stem_lats = {}
                        for s in ["drums", "bass", "vocals"]:
                            p = lat_dir / f"{s}.vae.pt"
                            if p.exists():
                                stem_lats[s] = str(p)
                        if len(stem_lats) < 2:
                            continue
                        # Create 2 augmented variants per track
                        for _ in range(2):
                            self.items.append({
                                "src": "musdb_aug",
                                "stem_wavs": stem_wavs,    # real stem audio
                                "stem_lats": stem_lats,     # real stem latents
                            })
                            n_musdb_aug += 1
                print(f"  stem pool: {pool_counts}")

        print(f"[distill6 dataset] v7={n_v7} musdb={n_musdb} "
              f"musdb_aug={n_musdb_aug} total={len(self.items)}")

    def _resolve_latent_path(self, candidates):
        """Try each candidate path, return first that exists."""
        if isinstance(candidates, str):
            return candidates if os.path.exists(candidates) else None
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _sample_random_stem(self, category, T_frames):
        """Sample a random stem latent from the index, trimmed to T_frames."""
        if self.stem_index is None or category not in self.stem_index:
            return None
        candidates_list = self.stem_index[category]
        # Try up to 5 times to find a loadable stem
        for _ in range(5):
            candidates = self.rng.choice(candidates_list)
            path = self._resolve_latent_path(candidates)
            if path is None:
                continue
            try:
                z = _load_latent(path)  # [T, 64]
                T = z.shape[0]
                if T <= 0:
                    continue
                if T >= T_frames:
                    start = self.rng.randint(0, T - T_frames)
                    return z[start:start + T_frames]
                else:
                    return F.pad(z, (0, 0, 0, T_frames - T))
            except Exception:
                continue
        return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]

        # Handle augmented MUSDB items specially
        if it["src"] == "musdb_aug":
            return self._getitem_musdb_aug(it)

        try:
            audio = _load_audio_48k_stereo(it["audio"])  # [2, N]
            stems = []
            for p in it["stem_paths"]:
                if p is None:
                    stems.append(None)
                else:
                    stems.append(_load_latent(p))
        except Exception:
            return self.__getitem__((idx + 1) % len(self.items))

        # Frames available
        T_present = [s.shape[0] for s in stems if s is not None]
        T_lat = min(T_present) if T_present else 0
        T_aud = audio.shape[1] // SAMPLES_PER_FRAME
        T_avail = min(T_lat, T_aud)
        if T_avail <= self.crop_frames:
            pad_frames = self.crop_frames - T_avail
            pad_samples = pad_frames * SAMPLES_PER_FRAME
            audio_w = F.pad(audio[:, :T_avail * SAMPLES_PER_FRAME], (0, pad_samples))
            stems_w = torch.zeros(6, self.crop_frames, 64)
            for i, s in enumerate(stems):
                if s is None:
                    continue
                stems_w[i, :T_avail] = s[:T_avail]
        else:
            start_f = self.rng.randint(0, T_avail - self.crop_frames)
            end_f = start_f + self.crop_frames
            audio_w = audio[:, start_f * SAMPLES_PER_FRAME:end_f * SAMPLES_PER_FRAME]
            stems_w = torch.zeros(6, self.crop_frames, 64)
            for i, s in enumerate(stems):
                if s is None:
                    continue
                stems_w[i] = s[start_f:end_f]

        mask = torch.tensor(it["mask"], dtype=torch.float32)
        # → audio [2, N], stems [6, 64, T], mask [6]
        return audio_w.contiguous(), stems_w.transpose(1, 2).contiguous(), mask

    def _getitem_musdb_aug(self, it):
        """Augmented MUSDB: rebuild mixture from real stems + pool stems.

        Mixture = drums.wav + bass.wav + vocals.wav + pool_guitar.wav
                  + pool_piano.wav + pool_other.wav
        All 6 stems have matching latent targets. No input/target mismatch.
        """
        # Load real MUSDB stem wavs
        stem_audio = {}  # stem_name → [2, N]
        for s, p in it["stem_wavs"].items():
            try:
                stem_audio[s] = _load_audio_48k_stereo(p)
            except Exception:
                pass
        if len(stem_audio) < 2:
            return self.__getitem__(self.rng.randint(0, len(self.items) - 1))

        # Load real MUSDB stem latents
        stem_latents = {}  # stem_name → [T, 64]
        for s, p in it["stem_lats"].items():
            try:
                stem_latents[s] = _load_latent(p)
            except Exception:
                pass

        # Get the reference length from stems
        N_audio = min(a.shape[1] for a in stem_audio.values())
        T_lat = min(z.shape[0] for z in stem_latents.values()) if stem_latents else N_audio // SAMPLES_PER_FRAME

        # Sample random pool stems for guitar/piano/other
        pool_audio = {}   # category → [2, N]
        pool_latents = {} # category → [T, 64]
        for cat in ["guitar", "piano", "other"]:
            if cat not in self.stem_pool or not self.stem_pool[cat]:
                continue
            wav_path, lat_path = self.rng.choice(self.stem_pool[cat])
            try:
                a = _load_audio_48k_stereo(wav_path)
                z = _load_latent(lat_path)
                pool_audio[cat] = a
                pool_latents[cat] = z
            except Exception:
                continue

        # Determine crop window
        T_avail = min(T_lat, N_audio // SAMPLES_PER_FRAME)
        for z in pool_latents.values():
            T_avail = min(T_avail, z.shape[0])
        if T_avail < 10:
            return self.__getitem__(self.rng.randint(0, len(self.items) - 1))

        if T_avail <= self.crop_frames:
            start_f = 0
            crop_f = T_avail
        else:
            start_f = self.rng.randint(0, T_avail - self.crop_frames)
            crop_f = self.crop_frames

        start_s = start_f * SAMPLES_PER_FRAME
        end_s = start_s + crop_f * SAMPLES_PER_FRAME
        end_f = start_f + crop_f

        # Build the synthetic mixture by summing all stem audio
        mix = torch.zeros(2, crop_f * SAMPLES_PER_FRAME)
        for a in stem_audio.values():
            seg = a[:, start_s:end_s]
            mix[:, :seg.shape[1]] += seg
        for a in pool_audio.values():
            seg = a[:, start_s:end_s]
            mix[:, :seg.shape[1]] += seg

        # Pad if needed
        if crop_f < self.crop_frames:
            pad_samples = (self.crop_frames - crop_f) * SAMPLES_PER_FRAME
            mix = F.pad(mix, (0, pad_samples))

        # Build 6-stem latent targets
        # Order: drums(0), bass(1), other(2), vocals(3), guitar(4), piano(5)
        stems_w = torch.zeros(6, self.crop_frames, 64)
        mask = [0.0] * 6

        stem_to_idx = {"drums": 0, "bass": 1, "vocals": 3}
        for s, si in stem_to_idx.items():
            if s in stem_latents:
                z = stem_latents[s]
                usable = min(z.shape[0] - start_f, self.crop_frames)
                if usable > 0:
                    stems_w[si, :usable] = z[start_f:start_f + usable]
                    mask[si] = 1.0

        pool_to_idx = {"guitar": 4, "piano": 5, "other": 2}
        for cat, si in pool_to_idx.items():
            if cat in pool_latents:
                z = pool_latents[cat]
                usable = min(z.shape[0] - start_f, self.crop_frames)
                if usable > 0:
                    stems_w[si, :usable] = z[start_f:start_f + usable]
                    mask[si] = 1.0

        mask = torch.tensor(mask, dtype=torch.float32)
        return mix.contiguous(), stems_w.transpose(1, 2).contiguous(), mask


def collate(batch):
    audio = torch.stack([b[0] for b in batch])  # [B, 2, N]
    stems = torch.stack([b[1] for b in batch])  # [B, 6, 64, T]
    mask = torch.stack([b[2] for b in batch])   # [B, 6]
    return audio, stems, mask
