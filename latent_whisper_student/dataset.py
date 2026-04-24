"""Dataset: pair Oobleck latent chunks with Whisper teacher outputs.

Each item = one 30 s chunk, yielding:
    mix_lat       : [64, 750] float32       (student input)
    target_hidden : [1500, d_model] float16  (student regression target)
    tokens        : [T_tok] int64            (CE target through frozen decoder)
    tok_mask      : [T_tok] bool             (1 = real token)

The chunk is chosen at random per __getitem__, and tokens are padded to
``max_tokens`` in the collate fn.
"""
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CHUNK_LATENT_FRAMES = 750
LATENT_DIM = 64


def _load_latent(path: Path) -> torch.Tensor:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    z = raw["latents"] if isinstance(raw, dict) else raw
    if z.dim() == 2 and z.shape[0] == 64 and z.shape[1] != 64:
        z = z.t()                                    # → [T, 64]
    return z.float()


class LatentWhisperDataset(Dataset):
    """Walks a latent root for paired (latent, whisper-teacher) files."""

    def __init__(self,
                 latent_root: str,
                 model_name: str = "base",
                 latent_glob: str = "*.vae.pt",
                 seed: int = 0,
                 min_tokens: int = 2):
        self.latent_root = Path(latent_root)
        self.model_tag = model_name.replace("-", "_")
        self.rng = random.Random(seed)

        suffix = f"teacher_whisper_{self.model_tag}.pt"
        self.items = []   # list of (latent_path, teacher_path, n_chunks, d_model)

        for lp in sorted(self.latent_root.rglob(latent_glob)):
            tp = lp.parent / f"{lp.name.replace('.vae.pt','')}.{suffix}"
            if not tp.exists():
                continue
            # lightweight peek (avoid loading whole hidden state)
            try:
                head = torch.load(tp, map_location="cpu", weights_only=False)
            except Exception:
                continue
            n_chunks = int(head.get("n_chunks", 0))
            if n_chunks == 0:
                continue
            d_model = int(head.get("d_model", head["encoder_hidden"].shape[-1]))
            self.items.append({
                "lat": lp,
                "teach": tp,
                "n_chunks": n_chunks,
                "d_model": d_model,
            })

        print(f"[latent-whisper-ds] {len(self.items)} sessions "
              f"({sum(i['n_chunks'] for i in self.items)} chunks) "
              f"from {self.latent_root}")

    def __len__(self):
        # one "item" per chunk, but because chunks-per-session varies we use
        # session count × average and sample randomly inside __getitem__.
        return sum(i["n_chunks"] for i in self.items)

    def _pick(self, idx):
        # deterministic mapping idx → (session, chunk) so dataloader shuffle
        # still provides randomness but every chunk is visited once per epoch.
        k = idx
        for it in self.items:
            if k < it["n_chunks"]:
                return it, k
            k -= it["n_chunks"]
        # fallback
        it = self.items[idx % len(self.items)]
        return it, idx % it["n_chunks"]

    def __getitem__(self, idx):
        it, ci = self._pick(idx)
        lat_full = _load_latent(it["lat"])                # [T, 64]
        s = ci * CHUNK_LATENT_FRAMES
        e = s + CHUNK_LATENT_FRAMES
        lat = lat_full[s:e]
        if lat.shape[0] < CHUNK_LATENT_FRAMES:
            lat = F.pad(lat, (0, 0, 0, CHUNK_LATENT_FRAMES - lat.shape[0]))
        lat = lat.transpose(0, 1).contiguous()            # [64, 750]

        teach = torch.load(it["teach"], map_location="cpu", weights_only=False)
        enc_hidden = teach["encoder_hidden"][ci]          # [1500, d_model] fp16
        tokens = teach["tokens"][ci]
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.as_tensor(tokens, dtype=torch.long)
        else:
            tokens = tokens.long()

        return {
            "mix_lat": lat,                               # [64, 750] fp32
            "target_hidden": enc_hidden.float(),          # [1500, d_model] fp32
            "tokens": tokens,                             # [T_tok]
        }


def collate(batch, max_tokens: int = 224, pad_id: int = 50257):
    """Stack latents + hidden targets; pad tokens to the longest in batch."""
    mix = torch.stack([b["mix_lat"] for b in batch])                 # [B, 64, 750]
    tgt = torch.stack([b["target_hidden"] for b in batch])           # [B, 1500, D]

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
        "mix_lat": mix,
        "target_hidden": tgt,
        "tokens": tok,
        "tok_mask": mask,
    }
