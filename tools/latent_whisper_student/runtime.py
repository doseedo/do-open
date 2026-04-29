"""Inference wrapper for the latent-lyric student.

Takes an Oobleck VAE latent (or a [T, 64]/ [64, T] tensor) and produces
lyric text via autoregressive greedy decoding over ACE-Step tokens, then
detokenizes with VoiceBpeTokenizer.
"""
import os
import sys
from pathlib import Path
from typing import List

import torch

import importlib.util as _ilu
_here = os.path.dirname(os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location(
    "latent_whisper_student._student_model",
    os.path.join(_here, "student_model.py"),
)
_sm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sm)
LatentLyricStudent = _sm.LatentLyricStudent
configure = _sm.configure
PAD_ID = _sm.PAD_ID
SOS_ID = _sm.SOS_ID
EOS_ID = _sm.EOS_ID
ACE_VOCAB = _sm.ACE_VOCAB


LATENT_FPS = 25
CHUNK_FRAMES = LATENT_FPS * 30               # 750


def _load_tokenizer():
    """Load ACE-Step's VoiceBpeTokenizer from whichever path is available."""
    for p in (
        "/scratch/doseedo-Do/home/arlo/Data/dø/do/models",
        "/scratch/Do/home/arlo/Data/dø/do/models",
    ):
        if os.path.isdir(p):
            sys.path.insert(0, p)
            break
    from lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
    return VoiceBpeTokenizer()


class LatentLyricRuntime:
    _inst = None

    @classmethod
    def get(cls, ckpt_path: str, device: str = "cuda"):
        if cls._inst is None or cls._inst.ckpt_path != ckpt_path:
            cls._inst = cls(ckpt_path, device=device)
        return cls._inst

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.ckpt_path = ckpt_path
        self.device = device

        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = sd.get("cfg") or configure(sd.get("size", "base"))
        self.cfg = cfg
        self.model = LatentLyricStudent(**cfg)
        self.model.load_state_dict(sd["model"])
        self.model.eval().to(device)

        self.tok = _load_tokenizer()

    @torch.no_grad()
    def _latent_to_chunks(self, latent_tc: torch.Tensor) -> torch.Tensor:
        if latent_tc.dim() != 2:
            raise ValueError(f"expected 2D latent, got {latent_tc.shape}")
        if latent_tc.shape[0] == 64 and latent_tc.shape[1] != 64:
            lat = latent_tc                                  # [64, T]
        else:
            lat = latent_tc.t()                              # [64, T]
        T = lat.shape[1]
        pad = (-T) % CHUNK_FRAMES
        if pad:
            lat = torch.nn.functional.pad(lat, (0, pad))
        N = lat.shape[1] // CHUNK_FRAMES
        chunks = lat.reshape(64, N, CHUNK_FRAMES).permute(1, 0, 2).contiguous()
        return chunks.to(self.device).float()                 # [N, 64, 750]

    @torch.no_grad()
    def transcribe(self, latent_tc: torch.Tensor,
                   language: str = "en",
                   max_len: int = 512,
                   temperature: float = 1.0) -> List[str]:
        chunks = self._latent_to_chunks(latent_tc)            # [N, 64, 750]
        out = []
        for i in range(chunks.shape[0]):
            ids = self.model.generate(chunks[i:i+1],
                                      max_len=max_len,
                                      sos_id=SOS_ID,
                                      eos_id=EOS_ID,
                                      temperature=temperature)
            seq = ids[0].tolist()
            # Strip SOS at start, stop at first EOS
            if seq and seq[0] == SOS_ID:
                seq = seq[1:]
            if EOS_ID in seq:
                seq = seq[:seq.index(EOS_ID)]
            # Drop any out-of-range / special tokens before detokenizing
            seq = [t for t in seq if 0 <= t < ACE_VOCAB]
            try:
                text = self.tok.decode(seq, language=language)
            except TypeError:
                text = self.tok.decode(seq)
            out.append(text)
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latent", required=True)
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    rt = LatentLyricRuntime(args.ckpt)
    raw = torch.load(args.latent, map_location="cpu", weights_only=False)
    lat = raw["latents"] if isinstance(raw, dict) else raw
    print(f"latent shape: {tuple(lat.shape)}")
    out = rt.transcribe(lat, language=args.language)
    for i, t in enumerate(out):
        print(f"[chunk {i}] {t}")
