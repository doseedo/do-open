"""
Latent-domain time-signature primitives.

These replace the waveform-domain crop/concat operations in
time-sig-editor/server.py::_remap_bar for the cases that are pure
concat/drop/duplicate (no time-stretch). For meter changes that require
time-stretching (e.g. 4/4<->7/8 with dotted-eighths), latent-space stretching
needs a separate model -- those cases return None and the caller falls back
to the existing waveform path.

All inputs/outputs are [T, 64] latents at VAE_HZ=25 (1 frame = 1920 samples
@ 48 kHz).
"""
from __future__ import annotations
from typing import Optional
import torch

from .dataset import SAMPLES_PER_FRAME
from .infer import LatentEditorRuntime
from .stretch_infer import LatentStretchRuntime  # noqa: F401  (used by callers)


# ---- low-level primitives ----------------------------------------------------

def latent_concat_at_sample(
    rt: LatentEditorRuntime,
    L_a: torch.Tensor,    # [Ta, 64]
    L_b: torch.Tensor,    # [Tb, 64]
    cut_sample: int,      # sample-accurate cut into L_a (samples @ 48 kHz)
) -> torch.Tensor:
    """Sample-accurate splice: keep L_a[:cut_sample] then L_b[cut_sample:]."""
    return rt.edit(L_a, L_b, cut_sample)


def latent_drop_segment(
    rt: LatentEditorRuntime,
    L: torch.Tensor,         # [T, 64]
    start_sample: int,       # delete samples [start_sample, end_sample)
    end_sample: int,
) -> torch.Tensor:
    """Sample-accurate deletion via splice."""
    # Build a "right side" sequence whose contents from end_sample onward
    # become the post-cut tail when concatenated at start_sample.
    # We construct L_b such that L_b[start_sample:] == L[end_sample:].
    # In frame terms: shift L by (end-start) samples leftwards.
    delta_samples = end_sample - start_sample
    delta_frames = delta_samples // SAMPLES_PER_FRAME
    # Round-trip: drop delta_frames at the (frame-aligned) join, then let the
    # sub-frame remainder be repaired by the editor at the cut.
    L_a = L
    # Build L_b by removing delta_frames around the cut region.
    L_b_full = torch.cat([L[:start_sample // SAMPLES_PER_FRAME],
                          L[start_sample // SAMPLES_PER_FRAME + delta_frames:]], dim=0)
    # Pad to match L_a length so indexing works
    if L_b_full.shape[0] < L_a.shape[0]:
        L_b_full = torch.cat(
            [L_b_full, torch.zeros(L_a.shape[0] - L_b_full.shape[0], 64)], 0
        )
    return rt.edit(L_a, L_b_full, start_sample)


def latent_duplicate_segment(
    rt: LatentEditorRuntime,
    L: torch.Tensor,
    src_start_sample: int,
    src_end_sample: int,
) -> torch.Tensor:
    """Append a duplicate of L[src_start:src_end] to the end of L."""
    # Frame-aligned duplication is exact; the only boundary needing repair
    # is the join between original tail and the duplicated copy.
    sf = src_start_sample // SAMPLES_PER_FRAME
    ef = src_end_sample // SAMPLES_PER_FRAME
    dup = L[sf:ef]
    naive = torch.cat([L, dup], dim=0)  # [T+dup, 64]
    # Build a side B that, when used as right-half of edit() at sample T*1920,
    # gives us the repaired join.
    join_sample = L.shape[0] * SAMPLES_PER_FRAME
    L_b = torch.cat([torch.zeros(L.shape[0], 64), dup], dim=0)
    return rt.edit(L, L_b, join_sample)


# ---- bar-level remap (mirrors _remap_bar in server.py) -----------------------

def latent_remap_bar(
    rt: LatentEditorRuntime,
    bar_lat: torch.Tensor,         # [T_bar, 64]
    src_n: int, src_den: int,
    tgt_n: int, tgt_den: int,
    beat_samples: int,
    eighth_samples: int,
    stretch_rt: Optional["LatentStretchRuntime"] = None,
) -> Optional[torch.Tensor]:
    """Latent analog of server.py::_remap_bar.

    Returns None for meter changes that require time-stretching (those still
    need the waveform path or a future latent-stretch model).
    """
    src = f"{src_n}/{src_den}"
    tgt = f"{tgt_n}/{tgt_den}"

    # ---- pure-drop cases ----
    if (src, tgt) == ("4/4", "6/8"):
        # keep first 6 eighths, drop last 2
        return latent_drop_segment(
            rt, bar_lat,
            start_sample=6 * eighth_samples,
            end_sample=8 * eighth_samples,
        )
    if (src, tgt) == ("4/4", "3/4"):
        return latent_drop_segment(
            rt, bar_lat,
            start_sample=3 * beat_samples,
            end_sample=4 * beat_samples,
        )
    if (src, tgt) == ("5/4", "4/4"):
        return latent_drop_segment(
            rt, bar_lat,
            start_sample=4 * beat_samples,
            end_sample=5 * beat_samples,
        )

    # ---- pure-duplicate cases ----
    if (src, tgt) == ("3/4", "4/4"):
        return latent_duplicate_segment(
            rt, bar_lat,
            src_start_sample=2 * beat_samples,
            src_end_sample=3 * beat_samples,
        )
    if (src, tgt) == ("4/4", "5/4"):
        return latent_duplicate_segment(
            rt, bar_lat,
            src_start_sample=3 * beat_samples,
            src_end_sample=4 * beat_samples,
        )

    # ---- stretch-required cases ----
    # These need the LatentStretchRuntime; if not provided, return None and
    # let the caller fall back to waveform _remap_bar.
    if stretch_rt is None:
        return None

    def _stretch_segment(seg, target_samples):
        target_frames = max(1, target_samples // SAMPLES_PER_FRAME)
        return stretch_rt.stretch(seg, target_frames)

    if (src, tgt) == ("4/4", "7/8"):
        # 4 eighths intact, beats 3..3.5 → dotted-8 (1.5 eighths),
        # beats 4..4.5 → dotted-8 (1.5 eighths). Total = 7 eighths.
        e = eighth_samples
        first = bar_lat[: 4 * e // SAMPLES_PER_FRAME]
        mid_src = bar_lat[4 * e // SAMPLES_PER_FRAME : 6 * e // SAMPLES_PER_FRAME]
        mid = _stretch_segment(mid_src, int(round(1.5 * e)))
        last_src = bar_lat[6 * e // SAMPLES_PER_FRAME : 8 * e // SAMPLES_PER_FRAME]
        last = _stretch_segment(last_src, int(round(1.5 * e)))
        return torch.cat([first, mid, last], dim=0)

    if (src, tgt) == ("7/8", "4/4"):
        e = eighth_samples
        split = 4 * e // SAMPLES_PER_FRAME
        first = bar_lat[:split]
        last = _stretch_segment(bar_lat[split:], 4 * e)
        return torch.cat([first, last], dim=0)

    if (src, tgt) == ("6/8", "4/4"):
        e = eighth_samples
        split = 4 * e // SAMPLES_PER_FRAME
        first = bar_lat[:split]
        last = _stretch_segment(bar_lat[split:], 4 * e)
        return torch.cat([first, last], dim=0)

    if (src, tgt) == ("3/4", "7/8"):
        e = eighth_samples
        split = 4 * e // SAMPLES_PER_FRAME
        first = bar_lat[:split]
        last = _stretch_segment(bar_lat[split:], 3 * e)
        return torch.cat([first, last], dim=0)

    # Generic fallback: proportional latent stretch over the whole bar.
    src_eighths = src_n * (2 if src_den == 4 else 1)
    tgt_eighths = tgt_n * (2 if tgt_den == 4 else 1)
    target_samples = eighth_samples * tgt_eighths
    return _stretch_segment(bar_lat, target_samples)
