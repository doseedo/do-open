"""Dataset: cached (latent, PANNs scores) pairs."""
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


LATENT_DIM = 64
LATENT_CHUNK_FRAMES = 750                # 30 s @ 25 Hz


class LatentPANNsDataset(Dataset):
    def __init__(self,
                 index_json: str = "/scratch/latent_panns_student/cache_index.json",
                 crop_frames: int = LATENT_CHUNK_FRAMES,
                 seed: int = 0):
        self.entries = json.load(open(index_json))
        self.crop = crop_frames
        self.rng = random.Random(seed)
        print(f"[panns-ds] {len(self.entries)} cached entries from {index_json}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        try:
            d = torch.load(e["path"], map_location="cpu", weights_only=False)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.entries))

        lat = d["latent"].float()                       # [T, 64]
        if lat.shape[0] < self.crop:
            lat = torch.nn.functional.pad(
                lat, (0, 0, 0, self.crop - lat.shape[0]))
        elif lat.shape[0] > self.crop:
            start = self.rng.randint(0, lat.shape[0] - self.crop)
            lat = lat[start:start + self.crop]

        scores = d["scores"].float()                    # [527]
        return {"latent": lat, "scores": scores}


def collate(batch):
    lat = torch.stack([b["latent"] for b in batch])     # [B, T, 64]
    sc  = torch.stack([b["scores"] for b in batch])     # [B, 527]
    return {"latent": lat, "scores": sc}
