"""Cached dataset for the latent drum sub-separator."""
from __future__ import annotations
import glob, os
from typing import List
import torch
from torch.utils.data import Dataset


class CachedDrumsepDataset(Dataset):
    def __init__(self, cache_dir: str, lru_shards: int = 4):
        self.shard_paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.pt")))
        if not self.shard_paths:
            raise RuntimeError(f"no shards in {cache_dir}")
        self.index: List[tuple] = []
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
        s = torch.load(self.shard_paths[sidx],
                       map_location="cpu", weights_only=False)
        self._lru[sidx] = s
        return s

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sidx, j = self.index[i]
        return self._shard(sidx)[j]


def collate(batch):
    return {
        "L_mix":   torch.stack([b["L_mix"]   for b in batch]),  # [B, T, 64]
        "L_stems": torch.stack([b["L_stems"] for b in batch]),  # [B, S, T, 64]
    }
