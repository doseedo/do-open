"""
Latent → BasicPitch student dataset.

Pairs Oobleck VAE latents with BasicPitch MIDI transcriptions. The two
trees mirror each other on the bucket; we never index, we never walk
deeply. We just compute the BasicPitch twin path from a given latent path
by string substitution.

    /…/Latents2/protoolsA/<rel>.vae.pt
        ↔
    /…/BasicPitch/<rel>.mid

(rel = date/session/Audio Files/name)

Each __getitem__ picks a random date, then a random session under it, then
a random latent file in that session's "Audio Files" dir. If the BasicPitch
twin doesn't exist for that file, we retry. All lookups are O(small dir
listing) on the FUSE mount -- no full traversal.
"""
from __future__ import annotations
import os, random
from typing import Optional

import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset

LATENT_DIM = 64
VAE_HZ = 25
N_PITCH = 128
DEFAULT_LATENTS_ROOT    = "/mnt/data/system_home/arlo/gcs-bucket/Latents2/protoolsA"
DEFAULT_BASICPITCH_ROOT = "/mnt/data/system_home/arlo/gcs-bucket/BasicPitch"
DEFAULT_MULTIF0_ROOT    = "/mnt/data/system_home/arlo/gcs-bucket/MultiF0/protoolsA"


def latent_to_mid_path(
    latent_path: str,
    latents_root: str = DEFAULT_LATENTS_ROOT,
    basicpitch_root: str = DEFAULT_BASICPITCH_ROOT,
) -> str:
    """Pure-string mirror lookup. Does not touch the filesystem."""
    rel = os.path.relpath(latent_path, latents_root)
    if rel.endswith(".vae.pt"):
        rel = rel[: -len(".vae.pt")]
    return os.path.join(basicpitch_root, rel + ".mid")


def latent_to_multif0_path(
    latent_path: str,
    latents_root: str = DEFAULT_LATENTS_ROOT,
    multif0_root: str = DEFAULT_MULTIF0_ROOT,
) -> str:
    """Mirror lookup for the BasicPitch multi-F0 contour npy."""
    rel = os.path.relpath(latent_path, latents_root)
    if rel.endswith(".vae.pt"):
        rel = rel[: -len(".vae.pt")]
    return os.path.join(multif0_root, rel + ".npy")


def multif0_to_frame_target(
    mf0: np.ndarray, n_frames: int, n_pitch: int = 128,
) -> np.ndarray:
    """Convert a [T_mf0, 264] BasicPitch multi-F0 contour to a [n_frames, 128]
    continuous frame target aligned to the latent grid.

    264 = 88 semitones × 3 cent-bins. We aggregate the 3 cent-bins per
    semitone with a max (semitone is 'active' if any cent-bin is active),
    then place the 88 semitones into the [21, 109) MIDI range, then
    average-pool the time axis from BasicPitch's ~86 Hz to the latent's
    25 Hz.
    """
    T_src = mf0.shape[0]
    # 1. semitone aggregation: max over each group of 3 cent-bins
    semis = mf0.reshape(T_src, 88, 3).max(axis=2)  # [T_src, 88]
    # 2. place into [128] MIDI grid (BasicPitch covers MIDI 21..108)
    out_pitch = np.zeros((T_src, n_pitch), dtype=np.float32)
    out_pitch[:, 21:21 + 88] = semis
    # 3. time downsample with average pooling
    out = np.zeros((n_frames, n_pitch), dtype=np.float32)
    for i in range(n_frames):
        s_lo = int(round(i * T_src / max(1, n_frames)))
        s_hi = int(round((i + 1) * T_src / max(1, n_frames)))
        s_hi = max(s_hi, s_lo + 1)
        out[i] = out_pitch[s_lo:s_hi].mean(axis=0)
    return out


def _safe_listdir(path: str):
    try:
        return os.listdir(path)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []


def _load_latent(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    lat = obj["latents"] if isinstance(obj, dict) else obj
    if lat.dim() == 3:
        lat = lat.squeeze(0)
    if lat.shape[0] == 64 and lat.shape[1] != 64:
        lat = lat.transpose(0, 1)
    return lat.float()  # [T, 64]


def midi_to_rolls(mid_path: str, n_frames: int, hz: float = VAE_HZ):
    """Rasterize a BasicPitch .mid into onset/frame/velocity/offset rolls
    aligned to the latent frame grid (1/hz seconds per frame).

    Returns four [n_frames, 128] float tensors:
        onset    binary, 1 where a note attack lands
        frame    binary, 1 where a note is currently sounding
        velocity normalized [0,1], replicated across active frames
        offset   sub-frame onset position in [0,1), 0 elsewhere
    """
    pm = pretty_midi.PrettyMIDI(mid_path)
    onset  = np.zeros((n_frames, N_PITCH), dtype=np.float32)
    frame  = np.zeros((n_frames, N_PITCH), dtype=np.float32)
    vel    = np.zeros((n_frames, N_PITCH), dtype=np.float32)
    offset = np.zeros((n_frames, N_PITCH), dtype=np.float32)
    for inst in pm.instruments:
        for note in inst.notes:
            p = int(note.pitch)
            if p < 0 or p >= N_PITCH:
                continue
            t_frac = note.start * hz
            f0 = int(np.floor(t_frac))
            sub = float(t_frac - f0)
            f1 = int(round(note.end * hz))
            f0c = max(0, min(n_frames - 1, f0))
            f1c = max(f0c + 1, min(n_frames, f1))
            onset[f0c, p]  = 1.0
            offset[f0c, p] = sub
            frame[f0c:f1c, p] = 1.0
            v = note.velocity / 127.0
            vel[f0c:f1c, p] = np.maximum(vel[f0c:f1c, p], v)
    return (
        torch.from_numpy(onset),
        torch.from_numpy(frame),
        torch.from_numpy(vel),
        torch.from_numpy(offset),
    )


class LatentMidiPairDataset(Dataset):
    """No indexing. Random walk over the protoolsA tree per item."""

    def __init__(
        self,
        latents_root: str = DEFAULT_LATENTS_ROOT,
        basicpitch_root: str = DEFAULT_BASICPITCH_ROOT,
        multif0_root: str = DEFAULT_MULTIF0_ROOT,
        win_frames: int = 256,
        seed: int = 0,
        virtual_size: int = 1_000_000,
        require_multif0: bool = False,
    ):
        self.latents_root = latents_root
        self.basicpitch_root = basicpitch_root
        self.multif0_root = multif0_root
        self.win = win_frames
        self.rng = random.Random(seed)
        self.virtual_size = virtual_size
        self.require_multif0 = require_multif0
        # one shallow listing on init: the date dirs (cheap, ~hundreds)
        self.dates = sorted(_safe_listdir(latents_root))
        if not self.dates:
            raise RuntimeError(f"No date dirs under {latents_root}")

    def __len__(self):
        return self.virtual_size

    def _random_latent(self, max_tries: int = 32) -> Optional[str]:
        """Pick a random (date, session, audio_files_dir, latent_file) tuple."""
        for _ in range(max_tries):
            d = self.rng.choice(self.dates)
            d_path = os.path.join(self.latents_root, d)
            # second level: usually "New" or "Prev"
            level2 = _safe_listdir(d_path)
            if not level2:
                continue
            l2 = self.rng.choice(level2)
            l2_path = os.path.join(d_path, l2)
            sessions = _safe_listdir(l2_path)
            if not sessions:
                continue
            sess = self.rng.choice(sessions)
            sess_path = os.path.join(l2_path, sess)
            # try common subdirs first; fall back to listing
            for sub in ("Audio Files", "Audio_Files", "Audio"):
                sub_path = os.path.join(sess_path, sub)
                if os.path.isdir(sub_path):
                    files = [f for f in _safe_listdir(sub_path) if f.endswith(".vae.pt")]
                    if files:
                        return os.path.join(sub_path, self.rng.choice(files))
            files = [f for f in _safe_listdir(sess_path) if f.endswith(".vae.pt")]
            if files:
                return os.path.join(sess_path, self.rng.choice(files))
        return None

    def _safe_pair(self):
        # Retry forever; the success rate is high but FUSE can hiccup.
        # Returns (L, mid_path, mf0_path_or_None).
        attempts = 0
        while True:
            attempts += 1
            lat_p = self._random_latent()
            if lat_p:
                mid_p = latent_to_mid_path(lat_p, self.latents_root, self.basicpitch_root)
                mf0_p = latent_to_multif0_path(lat_p, self.latents_root, self.multif0_root)
                mf0_ok = os.path.exists(mf0_p)
                if self.require_multif0 and not mf0_ok:
                    continue
                if os.path.exists(mid_p):
                    try:
                        L = _load_latent(lat_p)
                        if L.shape[0] >= 4 and L.shape[1] == 64:
                            return L, mid_p, (mf0_p if mf0_ok else None)
                    except Exception:
                        pass
            if attempts % 200 == 0:
                import warnings
                warnings.warn(f"latent_pitch: {attempts} consecutive misses")

    def __getitem__(self, idx):
        L, mid_p, mf0_p = self._safe_pair()
        T_full = L.shape[0]
        try:
            onset_full, frame_full, vel_full, offset_full = midi_to_rolls(mid_p, T_full)
        except Exception:
            onset_full  = torch.zeros(T_full, N_PITCH)
            frame_full  = torch.zeros(T_full, N_PITCH)
            vel_full    = torch.zeros(T_full, N_PITCH)
            offset_full = torch.zeros(T_full, N_PITCH)
        # If multi-F0 contours are available, REPLACE the binary frame
        # target with the continuous BasicPitch posterior. Onset/vel/offset
        # stay rasterized from .mid because multi-F0 doesn't provide them.
        if mf0_p is not None:
            try:
                mf0 = np.load(mf0_p)
                frame_full_np = multif0_to_frame_target(mf0, T_full, N_PITCH)
                frame_full = torch.from_numpy(frame_full_np)
            except Exception:
                pass  # fall back to binary frame target

        n = min(self.win, T_full)
        s = self.rng.randint(0, T_full - n) if T_full > n else 0
        L_w    = L[s : s + n]
        onset  = onset_full[s : s + n]
        frame  = frame_full[s : s + n]
        vel    = vel_full[s : s + n]
        offset = offset_full[s : s + n]

        if n < self.win:
            pad = self.win - n
            L_w    = torch.cat([L_w,    torch.zeros(pad, LATENT_DIM)], 0)
            onset  = torch.cat([onset,  torch.zeros(pad, N_PITCH)], 0)
            frame  = torch.cat([frame,  torch.zeros(pad, N_PITCH)], 0)
            vel    = torch.cat([vel,    torch.zeros(pad, N_PITCH)], 0)
            offset = torch.cat([offset, torch.zeros(pad, N_PITCH)], 0)
            mask   = torch.cat([torch.ones(n), torch.zeros(pad)])
        else:
            mask = torch.ones(self.win)

        return {
            "latent": L_w,
            "onset": onset,
            "frame": frame,
            "velocity": vel,
            "offset": offset,
            "mask": mask,
        }


def collate_pitch(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
