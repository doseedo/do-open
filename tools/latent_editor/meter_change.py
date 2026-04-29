"""Latent-domain meter change for the studio.

Drop-in replacement for time-sig-editor/server.py::run_meter_change's
waveform path. Operates entirely on latents (1 frame = 1920 samples
@ 48 kHz from the Oobleck VAE).

Two main entry points:

  latent_meter_change_substem(
      L_src,           # [T_src, 64] latent
      audio_48,        # [S, C] @48k — needed for source-onset detection on hh
      bs_48,           # bar_starts in 48kHz samples
      stem_name,       # 'kick','snare','tom','hh','ride','crash','perc'
      src_n, src_den,
      tgt_n, tgt_den,
      stretch_fn=None, # optional override (linear interp by default)
  ) -> torch.Tensor [T_tgt, 64]

  latent_meter_change_drum_kit(
      substem_latents, # dict[stem_name -> L_src]
      substem_audio,   # dict[stem_name -> [S, C] @48k]
      bs_48,
      src_n, src_den,
      tgt_n, tgt_den,
      stretch_fn=None,
  ) -> dict[stem_name -> [T_tgt, 64]]

Routing matches production after the 2026-04 revert:
  - kick / snare / tom / ride / crash / perc → splice path (linear interp
    stretch on the compressed regions, no editor)
  - hh / hat / hihat → per-slot pluck-and-place via editor.edit, with
    each slot a contiguous source slice and the editor running once per
    slot boundary in its training distribution (two distinct streams).

When the trained LatentStretchRuntime cleaner is available, pass it as
`stretch_fn=lambda L, target_frames: stretch_rt.stretch(L, target_frames)`
to swap the splice path to the cleaner-backed model. Until then, linear
interp is used.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .dataset import SAMPLES_PER_FRAME    # 1920
from .infer import LatentEditorRuntime

SR = 48000

# Stem routing — matches production after the 2026-04 revert
ACCENT_SUBSTEMS  = {"kick", "snare", "tom", "toms", "bass"}
FORCE_QUANTIZE   = {"hh", "hat", "hihat", "ride", "crash", "cymbal"}
MIXED_SUBSTEMS   = {"perc"}


# ── Stretch helper (swap-in point for trained LatentStretchRuntime) ──

def _interp_time(L_seg: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Linear interpolation along the time axis. Drop-in replacement for
    a real latent stretcher. When LatentStretchRuntime is trained, pass
    it as `stretch_fn` to override this default."""
    if L_seg.shape[0] == target_frames or L_seg.shape[0] == 0:
        return L_seg
    x = L_seg.transpose(0, 1).unsqueeze(0)        # [1, 64, T]
    y = torch.nn.functional.interpolate(
        x, size=target_frames, mode="linear", align_corners=False
    )
    return y.squeeze(0).transpose(0, 1)            # [target_frames, 64]


# ── Helpers shared by both paths ─────────────────────────────────────

def canonical_tgt_bar_samples(src_bar_samples, src_n, tgt_n, tgt_den):
    if tgt_den == 8:
        return (src_bar_samples * tgt_n) // (src_n * 2)
    return (src_bar_samples * tgt_n) // src_n


def _compute_src_positions_for_bar(audio_48, bar_lo, bar_hi, src_n):
    """Mirror of time-sig-editor/server.py::_process_per_bar's
    uniform-grid + snap-to-loudest-peak logic for hh quantization."""
    src_bar = bar_hi - bar_lo
    n_src_cells = src_n * 4   # 16ths-per-bar for x/4 src
    bar_audio = audio_48[bar_lo:bar_hi]
    if bar_audio.ndim == 2:
        mono = bar_audio.mean(axis=1).astype(np.float32)
    else:
        mono = bar_audio.astype(np.float32)
    mabs = np.abs(mono)
    sm = max(1, int(0.002 * SR))
    if sm > 1:
        mabs = np.convolve(mabs, np.ones(sm) / sm, mode="same")
    if float(mabs.max()) < 1e-4:
        return [bar_lo + int(round(s * src_bar / n_src_cells)) for s in range(n_src_cells)], n_src_cells
    half_cell = max(1, src_bar // (n_src_cells * 2))
    snapped = []
    for s in range(n_src_cells):
        g = int(round(s * src_bar / n_src_cells))
        lo = max(0, g - half_cell)
        hi = min(src_bar, g + half_cell)
        if hi <= lo:
            snapped.append(bar_lo + g)
        else:
            local_pk = lo + int(np.argmax(mabs[lo:hi]))
            snapped.append(bar_lo + local_pk)
    return snapped, n_src_cells


# ── Drum-remap (NO STRETCH) ──────────────────────────────────────────

def _detect_subdiv_from_audio(audio_48, bar_lo, bar_hi):
    """Coarse 8th vs 16th detection for one bar by counting onsets."""
    bar = audio_48[bar_lo:bar_hi]
    if bar.ndim == 2:
        mono = bar.mean(axis=1).astype(np.float32)
    else:
        mono = bar.astype(np.float32)
    env = np.abs(mono)
    sm = max(1, int(0.005 * SR))
    env = np.convolve(env, np.ones(sm)/sm, mode='same')
    pk = float(env.max())
    if pk < 1e-4:
        return None
    thr = pk * 0.25
    min_gap = max(1, len(env) // 32)
    n = 0; i = 0
    while i < len(env):
        if env[i] >= thr:
            n += 1
            i += min_gap
        else:
            i += 1
    # >12 onsets per 4/4 bar → 16th, else 8th
    return '16th' if n >= 12 else 'eighth'


def _latent_drum_remap_bar_accent(L_bar, src_n, src_den, tgt_n, tgt_den):
    """Latent equivalent of waveform _drum_remap_bar (accent stems): 4/4
    → 7/8 musical grouping cut (4 + 1.5 + 1.5) without stretch. Frame-grid
    operations. Falls back to literal cell cut for other meter combos."""
    src_sig = f"{src_n}/{src_den}"
    tgt_sig = f"{tgt_n}/{tgt_den}"
    T = L_bar.shape[0]
    if T < 8:
        return L_bar
    e = T / 8.0  # frames per eighth in source
    if src_sig == "4/4" and tgt_sig == "7/8":
        f4 = int(round(4 * e))
        f55 = int(round(5.5 * e))
        f6 = int(round(6 * e))
        f75 = int(round(7.5 * e))
        first    = L_bar[:f4]
        mid      = L_bar[f4:f55]              # first 1.5 eighths of beat 3
        last     = L_bar[f6:f75]              # first 1.5 eighths of beat 4 (snare)
        return torch.cat([first, mid, last], dim=0)
    if src_sig == "4/4" and tgt_sig == "6/8":
        return L_bar[:int(round(6 * e))]
    if src_sig == "4/4" and tgt_sig == "3/4":
        return L_bar[:int(round(6 * e))]
    if src_sig == "4/4" and tgt_sig == "5/4":
        last_beat = L_bar[int(round(6*e)):]
        return torch.cat([L_bar, last_beat], dim=0)
    # Generic literal-eighth cell cut/repeat
    src_eighths = src_n * (2 if src_den == 4 else 1)
    tgt_eighths = tgt_n * (2 if tgt_den == 4 else 1)
    if tgt_eighths <= src_eighths:
        return L_bar[:int(round(tgt_eighths * e))]
    reps = (tgt_eighths + src_eighths - 1) // src_eighths
    return L_bar.repeat(reps, 1)[:int(round(tgt_eighths * e))]


def _latent_drum_remap_bar_cymbal(L_bar, src_n, src_den, tgt_n, tgt_den, subdiv):
    """Cymbal-like (hh/ride/crash): literal cell cut on the locked
    eighth or 16th grid — drop the trailing cells that don't fit."""
    T = L_bar.shape[0]
    cells_per_eighth = 2 if subdiv == '16th' else 1
    n_src_cells = (src_n * 2 if src_den == 4 else src_n) * cells_per_eighth
    n_tgt_cells = tgt_n * cells_per_eighth if tgt_den == 8 else tgt_n * 2 * cells_per_eighth
    if n_src_cells == 0:
        return L_bar
    cell = T / n_src_cells
    if n_tgt_cells <= n_src_cells:
        return L_bar[:int(round(n_tgt_cells * cell))]
    # target longer → repeat last cell as filler
    base = L_bar[:int(round(n_src_cells * cell))]
    extra_cells = n_tgt_cells - n_src_cells
    filler = base[-int(round(cell)):] if base.shape[0] >= int(round(cell)) else base[-1:]
    return torch.cat([base] + [filler] * extra_cells, dim=0)


def _latent_cymbal_per_cell_editor(L, audio_48, bs_48, editor,
                                    src_n, src_den, tgt_n, tgt_den, subdiv):
    """Per-cell editor.edit cymbal path (mirrors the working tears
    test_meter_compare.py latent_pluck_and_place with the cell count
    derived from the LOCKED subdivision instead of always 14). For hh on
    16th lock and 4/4→7/8 this is exactly the original 14-slot path; for
    8th lock it becomes 7-slot. Keeps the editor's per-boundary
    smoothing pass that gave the original tears version its quality."""
    if tgt_den == 8:
        cells_per_eighth = 2 if subdiv == '16th' else 1
        n_tgt_slots = tgt_n * cells_per_eighth
    else:
        n_tgt_slots = tgt_n * (2 if subdiv == '16th' else 1) * 2
    out_chunks = []
    for k in range(len(bs_48) - 1):
        bar_lo = bs_48[k]
        bar_hi = min(bs_48[k + 1], audio_48.shape[0])
        if bar_hi - bar_lo < SAMPLES_PER_FRAME * 2:
            continue
        src_bar_samples = bar_hi - bar_lo
        tgt_bar_samples = canonical_tgt_bar_samples(src_bar_samples, src_n, tgt_n, tgt_den)
        target_bar_frames = max(1, int(round(tgt_bar_samples / SAMPLES_PER_FRAME)))
        snapped, n_src_cells = _compute_src_positions_for_bar(audio_48, bar_lo, bar_hi, src_n)
        out_slot_starts = [(s * target_bar_frames) // n_tgt_slots for s in range(n_tgt_slots + 1)]
        slot_frame_counts = [max(1, out_slot_starts[s + 1] - out_slot_starts[s]) for s in range(n_tgt_slots)]
        L_out = None
        for s_idx in range(n_tgt_slots):
            j = int(round(s_idx * n_src_cells / n_tgt_slots))
            j = min(j, n_src_cells - 1)
            src_pos = snapped[j]
            src_frame_start = max(0, src_pos // SAMPLES_PER_FRAME)
            src_frame_end = min(L.shape[0], src_frame_start + slot_frame_counts[s_idx])
            src_slice = L[src_frame_start:src_frame_end]
            if src_slice.shape[0] == 0:
                continue
            if L_out is None:
                L_out = src_slice.clone()
                continue
            cut_sample = L_out.shape[0] * SAMPLES_PER_FRAME
            L_a = torch.cat([L_out, torch.zeros(src_slice.shape[0], 64)], dim=0)
            L_b = torch.cat([torch.zeros(L_out.shape[0], 64), src_slice], dim=0)
            L_out = editor.edit(L_a, L_b, cut_sample)
        if L_out is None:
            continue
        if L_out.shape[0] > target_bar_frames:
            L_out = L_out[:target_bar_frames]
        elif L_out.shape[0] < target_bar_frames:
            pad = target_bar_frames - L_out.shape[0]
            L_out = torch.cat([L_out, L_out[-1:].expand(pad, -1)], dim=0)
        out_chunks.append(L_out)
    return torch.cat(out_chunks, dim=0) if out_chunks else L


def _latent_drum_kit_path(
    L, audio_48, bs_48, editor, stem_name,
    src_n, src_den, tgt_n, tgt_den,
):
    """Unified drum-stem latent meter change. NO time-stretch.
    cymbal-like → literal subdivision-grid cut.
    accent → musical grouping cut (4+1.5+1.5 for 4/4→7/8).
    Editor.edit smooths boundaries between bars (frame-aligned cut)."""
    name = stem_name.split('/')[-1].lower()
    cymbal_like = name in {'hh', 'hat', 'hihat', 'ride', 'crash', 'cymbal'}

    # Stem-level subdivision lock for cymbal-ish stems (vote across bars).
    subdiv_lock = None
    if cymbal_like and audio_48 is not None:
        votes = {}
        for k in range(len(bs_48) - 1):
            sd = _detect_subdiv_from_audio(audio_48, bs_48[k], min(bs_48[k+1], audio_48.shape[0]))
            if sd:
                votes[sd] = votes.get(sd, 0) + 1
        if votes:
            subdiv_lock = max(votes.items(), key=lambda kv: kv[1])[0]

    # Cymbal-like stems → per-cell editor.edit path (mirrors the working
    # tears compare-2667846b version: every cell boundary smoothed by the
    # latent editor in its training distribution).
    if cymbal_like and audio_48 is not None and subdiv_lock is not None:
        return _latent_cymbal_per_cell_editor(
            L, audio_48, bs_48, editor,
            src_n, src_den, tgt_n, tgt_den, subdiv_lock,
        )

    out_chunks = []
    for k in range(len(bs_48) - 1):
        f_lo = int(round(bs_48[k]     / SAMPLES_PER_FRAME))
        f_hi = int(round(bs_48[k + 1] / SAMPLES_PER_FRAME))
        f_hi = min(f_hi, L.shape[0])
        if f_hi - f_lo < 4:
            continue
        L_bar = L[f_lo:f_hi]
        if False:
            L_out_bar = None
        else:
            L_out_bar = _latent_drum_remap_bar_accent(L_bar, src_n, src_den, tgt_n, tgt_den)
        # Canonical target length
        src_bar_samples = bs_48[k+1] - bs_48[k]
        tgt_bar_samples = canonical_tgt_bar_samples(src_bar_samples, src_n, tgt_n, tgt_den)
        target_bar_frames = max(1, int(round(tgt_bar_samples / SAMPLES_PER_FRAME)))
        if L_out_bar.shape[0] > target_bar_frames:
            L_out_bar = L_out_bar[:target_bar_frames]
        elif L_out_bar.shape[0] < target_bar_frames:
            pad = target_bar_frames - L_out_bar.shape[0]
            L_out_bar = torch.cat([L_out_bar, L_out_bar[-1:].expand(pad, -1)], dim=0)
        out_chunks.append(L_out_bar)

    if not out_chunks:
        return L

    # Concat with editor smoothing at bar boundaries (frame-aligned cut).
    L_out = out_chunks[0]
    for nxt in out_chunks[1:]:
        cut_sample = L_out.shape[0] * SAMPLES_PER_FRAME
        L_a = torch.cat([L_out, torch.zeros(nxt.shape[0], 64)], dim=0)
        L_b = torch.cat([torch.zeros(L_out.shape[0], 64), nxt], dim=0)
        L_out = editor.edit(L_a, L_b, cut_sample)
    return L_out


# ── Splice path (kick/snare/tom/ride/crash) ──────────────────────────

def latent_remap_bar_4_4_to_7_8(L_bar, target_bar_frames, stretch_fn):
    """Latent equivalent of _remap_bar for 4/4 → 7/8: 4 + 1.5 + 1.5
    grouping with the compressed regions stretched via `stretch_fn`."""
    T = L_bar.shape[0]
    if T < 8:
        return stretch_fn(L_bar, target_bar_frames)
    eighth_frames = T / 8.0
    f_4 = int(round(4 * eighth_frames))
    f_6 = int(round(6 * eighth_frames))
    first    = L_bar[:f_4]
    mid_src  = L_bar[f_4:f_6]
    last_src = L_bar[f_6:]
    f_first_t = int(round(target_bar_frames * 4.0 / 7.0))
    f_mid_t   = int(round(target_bar_frames * (4.0 + 1.5) / 7.0)) - f_first_t
    f_last_t  = target_bar_frames - f_first_t - f_mid_t
    first_resized   = stretch_fn(first,    max(1, f_first_t))
    mid_compressed  = stretch_fn(mid_src,  max(1, f_mid_t))
    last_compressed = stretch_fn(last_src, max(1, f_last_t))
    out = torch.cat([first_resized, mid_compressed, last_compressed], dim=0)
    if out.shape[0] != target_bar_frames:
        out = stretch_fn(out, target_bar_frames)
    return out


def _latent_splice_path(L, bs_48, src_n, src_den, tgt_n, tgt_den, stretch_fn):
    out_chunks = []
    for k in range(len(bs_48) - 1):
        bar_lo_f = int(round(bs_48[k]     / SAMPLES_PER_FRAME))
        bar_hi_f = int(round(bs_48[k + 1] / SAMPLES_PER_FRAME))
        bar_hi_f = min(bar_hi_f, L.shape[0])
        if bar_hi_f - bar_lo_f < 4:
            continue
        bar_lat = L[bar_lo_f:bar_hi_f]
        src_bar_samples = bs_48[k + 1] - bs_48[k]
        tgt_bar_samples = canonical_tgt_bar_samples(src_bar_samples, src_n, tgt_n, tgt_den)
        target_bar_frames = max(1, int(round(tgt_bar_samples / SAMPLES_PER_FRAME)))
        # Currently only the 4/4→7/8 case is implemented; everything else
        # falls back to a single proportional stretch of the whole bar.
        if (src_n, src_den, tgt_n, tgt_den) == (4, 4, 7, 8):
            out_chunks.append(latent_remap_bar_4_4_to_7_8(bar_lat, target_bar_frames, stretch_fn))
        else:
            out_chunks.append(stretch_fn(bar_lat, target_bar_frames))
    return torch.cat(out_chunks, dim=0) if out_chunks else L


# ── Pluck-and-place path (hh/hat/hihat) ──────────────────────────────

def _latent_pluck_and_place(L, audio_48, bs_48, editor: LatentEditorRuntime,
                             src_n, src_den, tgt_n, tgt_den):
    n_tgt_slots = tgt_n * 2 if tgt_den == 8 else tgt_n * 4
    out_chunks = []
    for k in range(len(bs_48) - 1):
        bar_lo = bs_48[k]
        bar_hi = min(bs_48[k + 1], audio_48.shape[0])
        if bar_hi - bar_lo < SAMPLES_PER_FRAME * 2:
            continue
        src_bar_samples = bar_hi - bar_lo
        tgt_bar_samples = canonical_tgt_bar_samples(src_bar_samples, src_n, tgt_n, tgt_den)
        target_bar_frames = max(1, int(round(tgt_bar_samples / SAMPLES_PER_FRAME)))
        slot_samples = tgt_bar_samples // n_tgt_slots

        snapped, n_src_cells = _compute_src_positions_for_bar(audio_48, bar_lo, bar_hi, src_n)

        # FLOOR cumulative slot start frames so L_out never overshoots
        # the target slot start. Slot frame counts sum to target_bar_frames.
        out_slot_starts = [
            (s * target_bar_frames) // n_tgt_slots
            for s in range(n_tgt_slots + 1)
        ]
        slot_frame_counts = [
            max(1, out_slot_starts[s + 1] - out_slot_starts[s])
            for s in range(n_tgt_slots)
        ]

        L_out = None
        for s_idx in range(n_tgt_slots):
            j = int(round(s_idx * n_src_cells / n_tgt_slots))
            j = min(j, n_src_cells - 1)
            src_pos = snapped[j]
            src_frame_start = max(0, src_pos // SAMPLES_PER_FRAME)
            src_frame_end = min(L.shape[0], src_frame_start + slot_frame_counts[s_idx])
            src_slice = L[src_frame_start:src_frame_end]
            if src_slice.shape[0] == 0:
                continue

            if L_out is None:
                L_out = src_slice.clone()
                continue

            # Frame-aligned cut so hh's slot grid stays in sync with the
            # splice path's frame-aligned segment lengths. (Sample-accurate
            # sub-frame phase made hh drift relative to kick/snare in
            # latent space, audible as "hh ahead of the beat".)
            cut_sample = L_out.shape[0] * SAMPLES_PER_FRAME
            L_a = torch.cat([L_out, torch.zeros(src_slice.shape[0], 64)], dim=0)
            L_b = torch.cat([torch.zeros(L_out.shape[0], 64), src_slice], dim=0)
            L_out = editor.edit(L_a, L_b, cut_sample)

        if L_out is None:
            continue
        if L_out.shape[0] > target_bar_frames:
            L_out = L_out[:target_bar_frames]
        elif L_out.shape[0] < target_bar_frames:
            pad = target_bar_frames - L_out.shape[0]
            L_out = torch.cat([L_out, L_out[-1:].expand(pad, -1)], dim=0)
        out_chunks.append(L_out)
    return torch.cat(out_chunks, dim=0) if out_chunks else L


# ── Public entry points ──────────────────────────────────────────────

def latent_meter_change_substem(
    L_src: torch.Tensor,
    audio_48: np.ndarray,
    bs_48: List[int],
    stem_name: str,
    src_n: int, src_den: int,
    tgt_n: int, tgt_den: int,
    editor: LatentEditorRuntime,
    stretch_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """Process one drum substem in latent space. Returns a [T_tgt, 64]
    latent ready to decode."""
    name = stem_name.split("/")[-1].lower()
    drum_names = {'kick','snare','tom','toms','hh','hat','hihat','ride','crash','cymbal','bass'}
    if name in drum_names:
        return _latent_drum_kit_path(
            L_src, audio_48, bs_48, editor, stem_name,
            src_n, src_den, tgt_n, tgt_den,
        )
    # Non-drum stems still use the splice path with optional stretch_fn.
    if stretch_fn is None:
        stretch_fn = _interp_time
    return _latent_splice_path(L_src, bs_48, src_n, src_den, tgt_n, tgt_den, stretch_fn)


def latent_meter_change_drum_kit(
    substem_latents: Dict[str, torch.Tensor],
    substem_audio: Dict[str, np.ndarray],
    bs_48: List[int],
    src_n: int, src_den: int,
    tgt_n: int, tgt_den: int,
    editor: LatentEditorRuntime,
    stretch_fn: Optional[Callable] = None,
) -> Dict[str, torch.Tensor]:
    """Process all drum substems through the latent meter-change pipeline.
    Returns dict[stem_name -> processed latent]. Same routing as
    production: hh through pluck-and-place, others through splice."""
    out = {}
    for name, L_src in substem_latents.items():
        audio_48 = substem_audio.get(name)
        if audio_48 is None:
            # hh path needs audio for source-position snap; without it
            # we can only do the splice path.
            stem_name = "splice"
        else:
            stem_name = name
        out[name] = latent_meter_change_substem(
            L_src=L_src, audio_48=audio_48, bs_48=bs_48, stem_name=stem_name,
            src_n=src_n, src_den=src_den, tgt_n=tgt_n, tgt_den=tgt_den,
            editor=editor, stretch_fn=stretch_fn,
        )
    return out
