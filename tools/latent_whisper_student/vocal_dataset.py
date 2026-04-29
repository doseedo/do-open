"""Dataset for the latent-lyric student.

Reads precached per-pair .pt files written by
`scripts/precache_pairs.py`. Each cache file holds:
    latent       : [T, 64] fp16
    token_ids    : [N] int32
    word_timings : list of (start, end, word)   — possibly empty
    has_timing   : bool
    duration_s   : float

Returns one item per clip:
    mix_lat : [64, CHUNK_FRAMES] fp32   (zero-padded / cropped 30 s window)
    tokens  : [T_tok + 2] int64         (ACE-Step tokens w/ SOS / EOS)

For clips longer than 30 s we pick a random 30 s window and slice the token
sequence using `word_timings` so the tokens line up (roughly) with the window.
"""
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from student_model import PAD_ID, SOS_ID, EOS_ID, ACE_VOCAB


LATENT_FPS = 25
CHUNK_SECONDS = 30
CHUNK_FRAMES = LATENT_FPS * CHUNK_SECONDS          # 750
LATENT_DIM = 64


def _load_cache(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _slice_tokens_by_time(token_ids: list[int],
                          word_timings: list,
                          win_start: float, win_end: float) -> list[int]:
    """Proportionally slice a token list to a [win_start, win_end) window
    using word timings.

    word_timings is a list of [start, end, word] triples covering the whole
    clip; its length gives us the word count. We can't BPE-split each word
    individually here, so we approximate: take tokens in proportion to the
    fraction of words that fall inside the window.
    """
    if not word_timings:
        return list(token_ids)
    n_words = len(word_timings)
    inside = [i for i, (s, e, _) in enumerate(word_timings)
              if e > win_start and s < win_end]
    if not inside:
        return []
    i0, i1 = inside[0], inside[-1] + 1
    tok_start = int(round(len(token_ids) * i0 / n_words))
    tok_end   = int(round(len(token_ids) * i1 / n_words))
    tok_start = max(0, min(len(token_ids), tok_start))
    tok_end   = max(tok_start, min(len(token_ids), tok_end))
    return token_ids[tok_start:tok_end]


class VocalLyricDataset(Dataset):
    def __init__(self,
                 index_json: str = "/scratch/latent_whisper_student/cache_index.json",
                 max_tokens: int = 448,
                 seed: int = 0):
        self.entries = json.load(open(index_json))
        self.max_tokens = max_tokens
        self.rng = random.Random(seed)
        print(f"[vocal-lyric-ds] {len(self.entries)} cached pairs "
              f"from {index_json}")

    def __len__(self):
        return len(self.entries)

    def _bad(self, idx):
        return self.__getitem__((idx + 1) % len(self.entries))

    def __getitem__(self, idx):
        e = self.entries[idx]
        try:
            d = _load_cache(e["path"])
        except Exception:
            return self._bad(idx)

        lat = d["latent"].float()                        # [T, 64]
        T = lat.shape[0]
        tokens_all = d["token_ids"].tolist()
        word_timings = d.get("word_timings") or []
        has_timing = bool(d.get("has_timing"))

        if len(tokens_all) == 0 or T < 10:
            return self._bad(idx)

        # Pick a 30 s window
        if T <= CHUNK_FRAMES:
            start_f = 0
            end_f = T
            tokens = tokens_all
        else:
            start_f = self.rng.randint(0, T - CHUNK_FRAMES)
            end_f = start_f + CHUNK_FRAMES
            if has_timing:
                tokens = _slice_tokens_by_time(
                    tokens_all, word_timings,
                    win_start=start_f / LATENT_FPS,
                    win_end=end_f / LATENT_FPS)
            else:
                # No timing → approximate by proportion of time window
                frac_s = start_f / T
                frac_e = end_f / T
                ts = int(round(len(tokens_all) * frac_s))
                te = int(round(len(tokens_all) * frac_e))
                tokens = tokens_all[ts:te]

        if len(tokens) == 0:
            return self._bad(idx)

        # Clip the latent to CHUNK_FRAMES (pad if shorter)
        lat_w = lat[start_f:end_f]
        if lat_w.shape[0] < CHUNK_FRAMES:
            pad = CHUNK_FRAMES - lat_w.shape[0]
            lat_w = F.pad(lat_w, (0, 0, 0, pad))
        mix_lat = lat_w.transpose(0, 1).contiguous()     # [64, 750]

        # Sanity: tokens must be within ACE_VOCAB range (we shift nothing)
        tokens = [t for t in tokens if 0 <= t < ACE_VOCAB]
        if not tokens:
            return self._bad(idx)
        # Cap length (room for SOS / EOS)
        tokens = tokens[: self.max_tokens - 2]
        tok = torch.tensor([SOS_ID] + tokens + [EOS_ID], dtype=torch.long)

        return {"mix_lat": mix_lat, "tokens": tok}


def collate(batch):
    B = len(batch)
    mix = torch.stack([b["mix_lat"] for b in batch])     # [B, 64, 750]

    lens = [b["tokens"].shape[0] for b in batch]
    T_max = max(lens)
    tok = torch.full((B, T_max), PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        tok[i, :lens[i]] = b["tokens"]

    return {
        "mix_lat": mix,
        "tokens":  tok,
        "lengths": torch.tensor(lens, dtype=torch.long),
    }
