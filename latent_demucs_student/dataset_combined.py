"""Combined dataset for latent-demucs student.

Two sources:
  (a) Demucs teacher pairs (full 4-stem supervision):
      mix = full_mix.vae.pt, targets = teacher_{drums,bass,vocals,other}.vae.pt
      mask = [1,1,1,1]
  (b) GT single-stem pairs (per-stem supervision, masked loss):
      mix = full_mix.vae.pt, target_one = stem.vae.pt for class c
      targets[c] = stem, mask = one-hot at c
      timeline-aligned via first_start_samples → frame offset
"""
import json, os, random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CLASSES = ["drums", "bass", "vocals", "other"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
LATENT_FPS = 25
SAMPLE_RATE = 48000
SAMPLES_PER_FRAME = SAMPLE_RATE / LATENT_FPS  # 1920

LATENT_ROOT = Path("/scratch/mixesV7_latents")
GT_PAIRS_JSON = "/scratch/cache/gt_pairs.json"


def _load(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    z = raw["latents"] if isinstance(raw, dict) else raw
    if z.dim() == 2 and z.shape[0] == 64:
        z = z.t()
    return z.float()                       # [T, 64]


class CombinedSeparationDataset(Dataset):
    def __init__(self, crop_frames=400, seed=0,
                 use_teacher=True, use_gt=True):
        self.crop = crop_frames
        self.rng = random.Random(seed)
        self.items = []                    # list of dicts {kind, ...}

        if use_teacher:
            for full in LATENT_ROOT.rglob("full_mix.vae.pt"):
                if all((full.parent / f"teacher_{s}.vae.pt").exists() for s in CLASSES):
                    self.items.append({"kind": "teacher", "dir": full.parent})

        n_teach = len(self.items)

        if use_gt and os.path.exists(GT_PAIRS_JSON):
            with open(GT_PAIRS_JSON) as f:
                gt = json.load(f)
            for p in gt:
                if not (os.path.exists(p["mix_path"]) and os.path.exists(p["stem_path"])):
                    continue
                self.items.append({
                    "kind": "gt",
                    "mix": p["mix_path"],
                    "stem": p["stem_path"],
                    "class": p["class"],
                    "offset_frames": int(round(p["first_start_samples"] / SAMPLES_PER_FRAME)),
                })

        n_gt = len(self.items) - n_teach
        print(f"[combined dataset] teacher={n_teach}  gt={n_gt}  total={len(self.items)}")

    def __len__(self):
        return len(self.items)

    def _sample_window(self, T_avail):
        if T_avail <= self.crop:
            return 0
        return self.rng.randint(0, T_avail - self.crop)

    def __getitem__(self, idx):
        it = self.items[idx]
        try:
            if it["kind"] == "teacher":
                d = it["dir"]
                mix = _load(d / "full_mix.vae.pt")           # [T, 64]
                stems = [_load(d / f"teacher_{c}.vae.pt") for c in CLASSES]
                T = min([mix.shape[0]] + [s.shape[0] for s in stems])
                start = self._sample_window(T)
                end = start + self.crop
                mix_w = mix[start:end]
                stems_w = torch.stack([s[start:end] for s in stems])  # [4, T, 64]
                if mix_w.shape[0] < self.crop:
                    pad = self.crop - mix_w.shape[0]
                    mix_w = F.pad(mix_w, (0, 0, 0, pad))
                    stems_w = F.pad(stems_w, (0, 0, 0, pad))
                mask = torch.ones(4)
                return (mix_w.transpose(0, 1).contiguous(),       # [64, T]
                        stems_w.transpose(1, 2).contiguous(),     # [4, 64, T]
                        mask)
            else:  # gt
                mix = _load(it["mix"])
                stem = _load(it["stem"])
                off = max(0, it["offset_frames"])
                # mix is the session timeline; stem starts at frame `off` in that timeline.
                # So stem[0] aligns with mix[off]. We pick a window in stem-local coordinates.
                T_stem = stem.shape[0]
                T_mix_aligned = mix.shape[0] - off
                T_overlap = min(T_stem, max(0, T_mix_aligned))
                if T_overlap < 16:
                    return self.__getitem__((idx + 1) % len(self.items))
                start_local = self._sample_window(T_overlap)
                end_local = min(start_local + self.crop, T_overlap)
                mix_w  = mix[off + start_local: off + end_local]
                stem_w = stem[start_local:end_local]
                actual = mix_w.shape[0]
                # build 4-channel target with stem placed at its class slot, others zero
                stems_w = torch.zeros(4, actual, 64)
                ci = CLASS_IDX[it["class"]]
                stems_w[ci] = stem_w
                if actual < self.crop:
                    pad = self.crop - actual
                    mix_w = F.pad(mix_w, (0, 0, 0, pad))
                    stems_w = F.pad(stems_w, (0, 0, 0, pad))
                mask = torch.zeros(4); mask[ci] = 1
                return (mix_w.transpose(0, 1).contiguous(),
                        stems_w.transpose(1, 2).contiguous(),
                        mask)
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.items))


def collate(batch):
    mix = torch.stack([b[0] for b in batch])   # [B, 64, T]
    stems = torch.stack([b[1] for b in batch]) # [B, 4, 64, T]
    mask = torch.stack([b[2] for b in batch])  # [B, 4]
    return mix, stems, mask
