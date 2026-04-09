"""
doae.format — read/write .doae (Doseedo Audio Engine) files

File layout:
  DoaeHeader      (28 bytes)
  DoaeStemInfo[]  (96 bytes × stem_count)
  [padding to 8-byte align if needed]
  latent data blocks (float32, channel-first [64][T])
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

DOAE_MAGIC        = 0x45414F44   # b"DOAE"
DOAE_VERSION      = 1
DOAE_SAMPLE_RATE  = 48000
DOAE_LATENT_DIM   = 64
DOAE_DOWNSAMPLE   = 1920         # audio samples per latent frame

HEADER_FMT   = "<IHH I 16x"     # magic, version, stem_count, sample_rate, 16 reserved bytes
HEADER_SIZE  = struct.calcsize(HEADER_FMT)   # = 28

STEM_FMT     = "<64s f f B 3x Q I I"   # name, gain, pan, muted, _pad, offset, frames, dim
STEM_SIZE    = struct.calcsize(STEM_FMT)     # = 96


@dataclass
class Stem:
    name:          str
    latent:        np.ndarray       # shape [64, T], float32
    gain:          float = 1.0
    pan:           float = 0.0     # -1.0 (L) to +1.0 (R)
    muted:         bool  = False


@dataclass
class DoaeFile:
    stems:       List[Stem] = field(default_factory=list)
    sample_rate: int = DOAE_SAMPLE_RATE

    # ------------------------------------------------------------------ write

    def save(self, path: str | Path) -> None:
        path = Path(path)
        stems = self.stems

        # compute latent offsets
        header_end = HEADER_SIZE + STEM_SIZE * len(stems)
        # align to 64 bytes
        data_start = (header_end + 63) & ~63

        offsets = []
        cursor = data_start
        for s in stems:
            lat = np.asarray(s.latent, dtype=np.float32)
            if lat.ndim != 2 or lat.shape[0] != DOAE_LATENT_DIM:
                raise ValueError(f"Stem '{s.name}' latent must be [64, T], got {lat.shape}")
            offsets.append(cursor)
            cursor += lat.nbytes

        # --- header ---
        hdr = struct.pack(HEADER_FMT,
                          DOAE_MAGIC, DOAE_VERSION, len(stems), self.sample_rate)

        # --- stem info blocks ---
        stem_blocks = b""
        for i, s in enumerate(stems):
            lat = np.asarray(s.latent, dtype=np.float32)
            T = lat.shape[1]
            name_bytes = s.name.encode("utf-8")[:63].ljust(64, b"\x00")
            stem_blocks += struct.pack(STEM_FMT,
                                       name_bytes,
                                       float(s.gain),
                                       float(s.pan),
                                       int(s.muted),
                                       offsets[i],
                                       T,
                                       DOAE_LATENT_DIM)

        with open(path, "wb") as f:
            f.write(hdr)
            f.write(stem_blocks)
            # pad to data_start
            pad = data_start - (HEADER_SIZE + STEM_SIZE * len(stems))
            f.write(b"\x00" * pad)
            # latent data
            for s in stems:
                lat = np.asarray(s.latent, dtype=np.float32)
                f.write(lat.tobytes())

    # ------------------------------------------------------------------ read

    @classmethod
    def load(cls, path: str | Path) -> "DoaeFile":
        path = Path(path)
        with open(path, "rb") as f:
            raw = f.read()

        magic, version, stem_count, sample_rate = struct.unpack_from(HEADER_FMT, raw, 0)
        if magic != DOAE_MAGIC:
            raise ValueError(f"Not a .doae file (magic={magic:#010x})")
        if version != DOAE_VERSION:
            raise ValueError(f"Unsupported .doae version {version}")

        stems = []
        for i in range(stem_count):
            off = HEADER_SIZE + i * STEM_SIZE
            (name_b, gain, pan, muted,
             latent_offset, latent_frames, latent_dim) = struct.unpack_from(STEM_FMT, raw, off)
            name = name_b.rstrip(b"\x00").decode("utf-8", errors="replace")
            nbytes = latent_frames * latent_dim * 4
            latent_raw = raw[latent_offset : latent_offset + nbytes]
            latent = np.frombuffer(latent_raw, dtype=np.float32).reshape(latent_dim, latent_frames)
            stems.append(Stem(name=name, latent=latent,
                              gain=gain, pan=pan, muted=bool(muted)))

        return cls(stems=stems, sample_rate=sample_rate)

    # ------------------------------------------------------------------ helpers

    @property
    def duration_seconds(self) -> float:
        if not self.stems:
            return 0.0
        T = self.stems[0].latent.shape[1]
        return T * DOAE_DOWNSAMPLE / self.sample_rate

    def info(self) -> str:
        lines = [f".doae v{DOAE_VERSION}  {self.duration_seconds:.1f}s @ {self.sample_rate}Hz"]
        for i, s in enumerate(self.stems):
            T = s.latent.shape[1]
            lines.append(f"  stem {i}: '{s.name}'  T={T}  gain={s.gain:.2f}  pan={s.pan:.2f}"
                         f"  {'MUTED' if s.muted else 'active'}")
        return "\n".join(lines)


# ------------------------------------------------------------------ CLI

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python format.py <file.doae>")
        sys.exit(1)
    df = DoaeFile.load(sys.argv[1])
    print(df.info())
