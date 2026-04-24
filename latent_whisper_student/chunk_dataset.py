"""Per-chunk dataset for the vocal latent-whisper student.

Walks /scratch/latent_whisper_student/chunks/*.pt — each file is one 30 s
training example produced by gen_vocal_teacher.py.
"""
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


class VocalChunkDataset(Dataset):
    def __init__(self, chunks_dir: str, max_items: int | None = None):
        self.root = Path(chunks_dir)
        self.files = sorted(self.root.glob("*.pt"))
        if max_items:
            self.files = self.files[:max_items]
        print(f"[vocal-chunk-ds] {len(self.files)} chunks from {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.files))
        return {
            "mix_lat":        d["mix_lat"].float(),           # [64, 750]
            "target_hidden":  d["encoder_hidden"].float(),    # [1500, D]
            "tokens":         d["tokens"].long() if torch.is_tensor(d["tokens"])
                              else torch.as_tensor(d["tokens"], dtype=torch.long),
        }


def collate(batch, max_tokens: int = 224, pad_id: int = 50257):
    mix = torch.stack([b["mix_lat"] for b in batch])
    tgt = torch.stack([b["target_hidden"] for b in batch])

    lens = [min(len(b["tokens"]), max_tokens) for b in batch]
    T = max(lens) if lens else 1
    tok = torch.full((len(batch), T), pad_id, dtype=torch.long)
    mask = torch.zeros(len(batch), T, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = lens[i]
        if L > 0:
            tok[i, :L] = b["tokens"][:L]
            mask[i, :L] = True

    return {
        "mix_lat":       mix,
        "target_hidden": tgt,
        "tokens":        tok,
        "tok_mask":      mask,
    }
