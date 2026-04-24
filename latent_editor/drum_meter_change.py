"""Generalized drum-stem meter-change (latent-domain).

Same core algorithm as working_reference_universal.py, but parameterized
so (src_n, src_den, tgt_n, tgt_den) are arguments — no module globals.
Produces the same paths/outputs as the reference when invoked with
(4, 4, 7, 8) on the same inputs.

Pipeline per substem (all in VAE latent space):
  1. Detect hh subdivision for hh/hat/hihat stems (vote src_eighths vs
     src_eighths * 2 onsets per bar → "eighth" or "16th").
  2. Route by stem class:
       accent (kick/snare/tom/toms/bass)
         → region-based cut-and-paste with editor.edit smoothing.
           Region layout comes from REGION_LAYOUTS[(src, tgt)].
       force_quantize (hh/hat/hihat)
         → if detected subdivision == source-native eighth grid and the
           meter change is an "N cells drop" or "N cells add" on that
           grid: literal cell cut (0 edits), phase-locked to
           canonical_tgt_bar_samples.
         → else: per-cell pluck-and-place via editor.edit.
       mixed (ride/crash/cymbal)
         → same region-based path as accent (they contain sparse
           accents; no need to quantize their decay tails).
  3. Preallocate per-bar L_out from silence_latent(n), NEVER torch.zeros.
  4. Paste via editor.edit(L_out, L_b_clone_with_paste, sample_offset)
     at the trained "two-stream, one cut" distribution.
  5. Canonical per-bar target length from canonical_tgt_bar_samples so
     stems stay phase-locked across the whole song.

This module exposes a single entry point:

  drum_meter_change_substems(
      substems,                                     # {name: (wav, sr)}
      bs_48,                                        # bar starts @ 48k samples
      src_meter, tgt_meter,                         # (n, den) tuples
      vae, editor, silence_frame,                   # preloaded models
  ) -> {name: processed_wav_48k_np_array}

So it can be driven from a CLI test script OR wired into the server.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Callable, Optional
import glob, os, time
import numpy as np
import torch
import librosa

SR = 48000
SAMPLES_PER_FRAME = 1920           # Oobleck VAE frame = 1920 samples @ 48k (25 fps)

# Stem-class name sets. Drumsep outputs these; other separators can
# remap into the same buckets by re-keying the input dict.
ACCENT_SUBSTEMS  = {"kick", "snare", "tom", "toms", "bass"}
FORCE_QUANTIZE   = {"hh", "hat", "hihat"}
MIXED_SUBSTEMS   = {"ride", "crash", "cymbal"}

# ── Region layouts (eighth units) for accent/mixed meter conversions ─
# Each entry maps (src_n, src_den, tgt_n, tgt_den) →
#   list of (src_start_eighth, src_end_eighth, tgt_start_eighth)
# where the region is taken from source eighths [src_start..src_end) and
# pasted at target eighth tgt_start. Fractional eighths are allowed.
#
# Design rules:
#   • Source attacks retain their original micro-timing within each region
#   • Region placement preserves where the attack "feels" in the target meter
#     (e.g. snare on beat 4 of a 4/4 → tgt eighth 5.5 of 7/8 so it's the
#     "and" of the last 3-eighth grouping, matching the 4+1.5+1.5 feel)
REGION_LAYOUTS: Dict[Tuple[int, int, int, int], List[Tuple[float, float, float]]] = {
    (4, 4, 7, 8): [(0.0, 4.0, 0.0), (4.0, 5.5, 4.0), (6.0, 7.5, 5.5)],
    (4, 4, 6, 8): [(0.0, 6.0, 0.0)],
    (4, 4, 3, 4): [(0.0, 6.0, 0.0)],
    (3, 4, 4, 4): [(0.0, 6.0, 0.0), (4.0, 6.0, 6.0)],    # duplicate last beat
    (4, 4, 5, 4): [(0.0, 8.0, 0.0), (6.0, 8.0, 8.0)],    # duplicate last beat
    (7, 8, 4, 4): [(0.0, 7.0, 0.0), (5.0, 7.0, 7.0)],    # duplicate last beat
    (6, 8, 4, 4): [(0.0, 6.0, 0.0), (4.0, 6.0, 6.0)],    # duplicate last beat
    (3, 4, 7, 8): [(0.0, 4.0, 0.0), (4.0, 6.0, 4.0), (5.0, 6.0, 6.0)],
}


# ── Meter helpers ────────────────────────────────────────────────────

def meter_to_eighths(n: int, den: int) -> int:
    """Number of eighth-note cells in one bar of the given meter."""
    if den == 4: return n * 2
    if den == 8: return n
    if den == 2: return n * 4
    raise ValueError(f"unsupported denominator {den}")


def canonical_tgt_bar_samples(src_bar_samples: int, src_n: int, src_den: int,
                              tgt_n: int, tgt_den: int) -> int:
    """Target bar length in samples, derived purely from source bar length
    and meter ratio. Matches the waveform-domain helper of the same name."""
    src_e = meter_to_eighths(src_n, src_den)
    tgt_e = meter_to_eighths(tgt_n, tgt_den)
    return (src_bar_samples * tgt_e) // src_e


# ── Silence latent ───────────────────────────────────────────────────

def make_silence_latent(silence_frame: torch.Tensor):
    """Return a closure silence_latent(n) → [n, 64] from a single [1, 64]
    silence frame loaded from the model checkpoint."""
    def silence_latent(n: int) -> torch.Tensor:
        return silence_frame.expand(n, -1).clone()
    return silence_latent


# ── Onset detection (generic, not meter-specific) ────────────────────

def _smoothed_envelope(audio_seg: np.ndarray) -> np.ndarray:
    mono = audio_seg.mean(axis=1).astype(np.float32) if audio_seg.ndim == 2 else audio_seg.astype(np.float32)
    env = np.abs(mono)
    sm = max(1, int(0.005 * SR))
    return np.convolve(env, np.ones(sm) / sm, mode="same")


def _region_has_onset(audio_48: np.ndarray, lo: int, hi: int, bar_peak: float,
                      thresh_frac: float = 0.25) -> bool:
    if hi <= lo or bar_peak < 1e-6:
        return False
    env = _smoothed_envelope(audio_48[lo:hi])
    return float(env.max()) >= bar_peak * thresh_frac


def detect_hh_subdivision(audio_48: np.ndarray, bs_48: List[int],
                          src_n: int, src_den: int) -> str:
    """Vote 'eighth' vs '16th' by onset count per bar, snapped to the
    expected counts for the source meter:
        eighth → src_eighths        (e.g. 8 for 4/4, 6 for 3/4)
        16th   → src_eighths * 2    (e.g. 16 for 4/4, 12 for 3/4)
    Ties go to 'eighth' (over-detection from bleed is the common failure)."""
    src_e = meter_to_eighths(src_n, src_den)
    expect_8 = src_e
    expect_16 = src_e * 2
    votes = {"eighth": 0, "16th": 0}
    for k in range(len(bs_48) - 1):
        lo = bs_48[k]
        hi = min(bs_48[k + 1], audio_48.shape[0])
        if hi - lo < SAMPLES_PER_FRAME * 2:
            continue
        env = _smoothed_envelope(audio_48[lo:hi])
        pk = float(env.max())
        if pk < 1e-4:
            continue
        thr = pk * 0.15
        # min_gap sized so we don't double-count any onset narrower than
        # half an expected-16th cell.
        min_gap = max(1, len(env) // (expect_16 + 4))
        n, i = 0, 0
        while i < len(env):
            if env[i] >= thr:
                n += 1
                i += min_gap
            else:
                i += 1
        votes["16th" if (n - expect_8) > (expect_16 - n) else "eighth"] += 1
    if votes["eighth"] == 0 and votes["16th"] == 0:
        return "eighth"
    return "16th" if votes["16th"] > votes["eighth"] else "eighth"


# ── Per-bar source-cell snap grid (for pluck-and-place) ──────────────

def compute_src_positions_for_bar(audio_48: np.ndarray, bar_lo: int, bar_hi: int,
                                  n_cells: int) -> Tuple[List[int], int]:
    src_bar = bar_hi - bar_lo
    bar_audio = audio_48[bar_lo:bar_hi]
    env = _smoothed_envelope(bar_audio)
    if float(env.max()) < 1e-4:
        return [bar_lo + int(round(s * src_bar / n_cells)) for s in range(n_cells)], n_cells
    half = max(1, src_bar // (n_cells * 2))
    out = []
    for s in range(n_cells):
        g = int(round(s * src_bar / n_cells))
        a = max(0, g - half)
        b = min(src_bar, g + half)
        if b <= a:
            out.append(bar_lo + g)
        else:
            out.append(bar_lo + a + int(np.argmax(env[a:b])))
    return out, n_cells


# ── Accent/Mixed path: region cut + editor.edit smoothing ────────────

def latent_drum_accent_cut(
    L: torch.Tensor,
    audio_48: np.ndarray,
    bs_48: List[int],
    src_n: int, src_den: int,
    tgt_n: int, tgt_den: int,
    editor,
    silence_latent: Callable[[int], torch.Tensor],
) -> Tuple[torch.Tensor, int]:
    """Region-based cut-and-paste for accent/mixed drum stems (no time
    stretch). Regions are defined in eighth units by REGION_LAYOUTS; if
    the requested meter pair isn't in the table we fall back to a
    literal eighth-cell cut/repeat."""
    key = (src_n, src_den, tgt_n, tgt_den)
    layout = REGION_LAYOUTS.get(key)
    src_e = meter_to_eighths(src_n, src_den)
    tgt_e = meter_to_eighths(tgt_n, tgt_den)
    if layout is None:
        # Fallback: literal eighth cell cut (drop) or repeat (add)
        if tgt_e <= src_e:
            layout = [(0.0, float(tgt_e), 0.0)]
        else:
            # Append the last (tgt_e - src_e) eighths of the source
            extra = tgt_e - src_e
            layout = [(0.0, float(src_e), 0.0),
                      (float(src_e - extra), float(src_e), float(src_e))]

    out_chunks = []
    n_edits = 0
    n_bars = len(bs_48) - 1
    for k in range(n_bars):
        bar_lo_s = bs_48[k]
        bar_hi_s = min(bs_48[k + 1], audio_48.shape[0])
        if bar_hi_s - bar_lo_s < SAMPLES_PER_FRAME * max(4, src_e // 2):
            continue
        f_lo = int(round(bar_lo_s / SAMPLES_PER_FRAME))
        f_hi = min(int(round(bar_hi_s / SAMPLES_PER_FRAME)), L.shape[0])
        if f_hi - f_lo < src_e:
            continue
        L_bar = L[f_lo:f_hi]
        T = L_bar.shape[0]
        src_frames_per_eighth = T / src_e

        src_bar_samples = bs_48[k + 1] - bs_48[k]
        tgt_bar_samples = canonical_tgt_bar_samples(src_bar_samples, src_n, src_den, tgt_n, tgt_den)
        target_bar_frames = max(1, int(round(tgt_bar_samples / SAMPLES_PER_FRAME)))
        tgt_samples_per_eighth = tgt_bar_samples / tgt_e

        # Bar peak on real audio for onset gating
        bar_audio = audio_48[bar_lo_s:bar_hi_s]
        mono_bar = bar_audio.mean(axis=1).astype(np.float32) if bar_audio.ndim == 2 else bar_audio.astype(np.float32)
        bar_peak = float(np.abs(mono_bar).max())
        src_samples_per_eighth = src_bar_samples / src_e

        L_out = silence_latent(target_bar_frames)

        for region_idx, (src_s, src_e_end, tgt_s) in enumerate(layout):
            # Gate: does this source region actually have an onset? If
            # not, skip the paste so the silence-latent stays underneath.
            seg_lo = bar_lo_s + int(round(src_s * src_samples_per_eighth))
            seg_hi = bar_lo_s + int(round(src_e_end * src_samples_per_eighth))
            if not _region_has_onset(audio_48, seg_lo, seg_hi, bar_peak):
                continue

            src_f_s = int(round(src_s * src_frames_per_eighth))
            src_f_e = int(round(src_e_end * src_frames_per_eighth))
            region = L_bar[src_f_s:src_f_e]
            if region.shape[0] == 0:
                continue

            tgt_sample_offset = int(round(tgt_s * tgt_samples_per_eighth))
            cut_frame = tgt_sample_offset // SAMPLES_PER_FRAME
            if cut_frame >= L_out.shape[0]:
                continue

            if region_idx == 0 and tgt_sample_offset == 0:
                # Nothing to join yet — direct paste (no editor needed)
                paste_n = min(region.shape[0], target_bar_frames)
                L_out = L_out.clone()
                L_out[:paste_n] = region[:paste_n]
                continue

            L_b = L_out.clone()
            paste_end = min(L_b.shape[0], cut_frame + region.shape[0])
            L_b[cut_frame:paste_end] = region[:paste_end - cut_frame]
            L_out = editor.edit(L_out, L_b, tgt_sample_offset)
            n_edits += 1

        out_chunks.append(L_out)
    return (torch.cat(out_chunks, dim=0) if out_chunks else L), n_edits


# ── HH literal cut (when source subdivision already fits target grid) ─

def latent_hh_literal_cut(
    L: torch.Tensor,
    bs_48: List[int],
    src_n: int, src_den: int,
    tgt_n: int, tgt_den: int,
    silence_latent: Callable[[int], torch.Tensor],
) -> Tuple[torch.Tensor, int]:
    """When hh is on the source eighth grid AND the meter change only
    changes the eighth count, a literal per-bar cut (or repeat) is all
    that's needed. Per-bar output length is LOCKED to canonical_tgt_bar
    frames so hh stays phase-aligned with other stems."""
    out_chunks = []
    n_bars = len(bs_48) - 1
    src_e = meter_to_eighths(src_n, src_den)
    tgt_e = meter_to_eighths(tgt_n, tgt_den)
    for k in range(n_bars):
        f_lo = int(round(bs_48[k] / SAMPLES_PER_FRAME))
        f_hi = min(int(round(bs_48[k + 1] / SAMPLES_PER_FRAME)), L.shape[0])
        if f_hi - f_lo < src_e:
            continue
        L_bar = L[f_lo:f_hi]
        src_bar_samples = bs_48[k + 1] - bs_48[k]
        tgt_bar_samples = canonical_tgt_bar_samples(src_bar_samples, src_n, src_den, tgt_n, tgt_den)
        target_bar_frames = max(1, int(round(tgt_bar_samples / SAMPLES_PER_FRAME)))

        if tgt_e <= src_e:
            # Cut: keep the first target_bar_frames of the source bar.
            if L_bar.shape[0] >= target_bar_frames:
                out_chunks.append(L_bar[:target_bar_frames])
            else:
                pad = target_bar_frames - L_bar.shape[0]
                out_chunks.append(torch.cat([L_bar, silence_latent(pad)], dim=0))
        else:
            # Add: source + tail-repeat to reach target frame count.
            extra = target_bar_frames - L_bar.shape[0]
            if extra <= 0:
                out_chunks.append(L_bar[:target_bar_frames])
            else:
                tail = L_bar[-extra:] if extra <= L_bar.shape[0] else L_bar
                out_chunks.append(torch.cat([L_bar, tail[:extra]], dim=0))
    return (torch.cat(out_chunks, dim=0) if out_chunks else L), 0


# ── HH pluck-and-place (when source needs quantization to target grid) ─

def latent_pluck_and_place(
    L: torch.Tensor,
    audio_48: np.ndarray,
    bs_48: List[int],
    src_n: int, src_den: int,
    tgt_n: int, tgt_den: int,
    subdiv: str,
    editor,
    silence_latent: Callable[[int], torch.Tensor],
) -> Tuple[torch.Tensor, int]:
    """compare-2667846b version, generalized: preallocated L_bar =
    silence, per-slot chunk from snapped src cell, L_b = L_bar.clone()
    with paste, cut_sample = s_idx * slot_samples (sample-accurate).

    Slot count and source cell grid both scale with the detected
    subdivision: eighth → src_e cells → tgt_e slots; 16th → src_e*2
    cells → tgt_e*2 slots."""
    src_e = meter_to_eighths(src_n, src_den)
    tgt_e = meter_to_eighths(tgt_n, tgt_den)
    cells_per_eighth = 2 if subdiv == "16th" else 1
    n_tgt_slots = tgt_e * cells_per_eighth
    n_src_cells_grid = src_e * cells_per_eighth

    chunks = []
    n_edits = 0
    n_bars = len(bs_48) - 1
    for k in range(n_bars):
        bar_lo = bs_48[k]
        bar_hi = min(bs_48[k + 1], audio_48.shape[0])
        if bar_hi - bar_lo < SAMPLES_PER_FRAME * 2:
            continue
        src_bar_samples = bar_hi - bar_lo
        tgt_bar_samples = canonical_tgt_bar_samples(src_bar_samples, src_n, src_den, tgt_n, tgt_den)
        target_bar_frames = max(1, int(round(tgt_bar_samples / SAMPLES_PER_FRAME)))
        slot_samples = tgt_bar_samples // n_tgt_slots
        slot_frames = max(1, int(round(slot_samples / SAMPLES_PER_FRAME)))
        snapped, n_src_cells = compute_src_positions_for_bar(audio_48, bar_lo, bar_hi, n_src_cells_grid)

        L_bar = silence_latent(target_bar_frames)
        for s_idx in range(n_tgt_slots):
            j = min(int(round(s_idx * n_src_cells / n_tgt_slots)), n_src_cells - 1)
            src_pos = snapped[j]
            src_f_start = max(0, src_pos // SAMPLES_PER_FRAME)
            src_f_end = min(L.shape[0], src_f_start + slot_frames + 4)
            chunk = L[src_f_start:src_f_end]
            if chunk.shape[0] == 0:
                continue
            cut_sample = s_idx * slot_samples
            cut_frame = cut_sample // SAMPLES_PER_FRAME
            if cut_frame >= L_bar.shape[0]:
                continue
            L_b = L_bar.clone()
            paste_end = min(L_b.shape[0], cut_frame + chunk.shape[0])
            L_b[cut_frame:paste_end] = chunk[:paste_end - cut_frame]
            L_bar = editor.edit(L_bar, L_b, cut_sample)
            n_edits += 1
        chunks.append(L_bar)
    return (torch.cat(chunks, dim=0) if chunks else L), n_edits


# ── VAE encode/decode helpers (chunked to fit small GPU headroom) ────

def vae_encode_chunked(vae_gpu, audio_2d_48: np.ndarray,
                       chunk_frames: int = 128) -> torch.Tensor:
    """[samples, 2] @ 48k → [T, 64] latent. Chunked to `chunk_frames`
    latent-frame windows so any single encode fits the GPU headroom."""
    chunk_samples = chunk_frames * SAMPLES_PER_FRAME
    if audio_2d_48.shape[1] == 1:
        audio_2d_48 = np.concatenate([audio_2d_48, audio_2d_48], axis=1)
    pieces = []
    total = audio_2d_48.shape[0]
    i = 0
    while i < total:
        end = min(total, i + chunk_samples)
        seg = audio_2d_48[i:end]
        y_t = torch.from_numpy(seg.T).float().unsqueeze(0).cuda().bfloat16()
        with torch.no_grad():
            L = vae_gpu.encode(y_t).latent_dist.sample().squeeze(0).transpose(0, 1).float().cpu()
        pieces.append(L)
        torch.cuda.empty_cache()
        i = end
    return torch.cat(pieces, dim=0)


def vae_decode_chunked(vae_gpu, L: torch.Tensor, chunk_frames: int = 64) -> np.ndarray:
    """[T, 64] → [samples, 2] @ 48k. Chunked by latent-frame window."""
    pieces = []
    T = L.shape[0]
    i = 0
    while i < T:
        end = min(T, i + chunk_frames)
        seg = L[i:end]
        with torch.no_grad():
            z = seg.transpose(0, 1).unsqueeze(0).cuda().bfloat16()
            a = vae_gpu.decode(z).sample.squeeze(0).cpu().float().numpy()
        pieces.append(a)
        torch.cuda.empty_cache()
        i = end
    return np.concatenate(pieces, axis=1).T


# ── Stem classifier (maps arbitrary substem names → class) ───────────

def classify_stem(name: str) -> str:
    n = name.split("/")[-1].lower()
    if n in FORCE_QUANTIZE: return "force_quantize"
    if n in ACCENT_SUBSTEMS: return "accent"
    if n in MIXED_SUBSTEMS:  return "mixed"
    return "mixed"   # safest default — same path as accent/mixed


# ── Public entry point ───────────────────────────────────────────────

def drum_meter_change_substems_to_latents(
    substems: Dict[str, Tuple[np.ndarray, int]],
    bs_48: List[int],
    src_meter: Tuple[int, int],
    tgt_meter: Tuple[int, int],
    vae_gpu,
    editor,
    silence_frame: torch.Tensor,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Same routing as drum_meter_change_substems, but returns the
    processed LATENTS (not decoded wavs). Used by the latent-only path
    so the server never decodes — the browser does that locally via
    WebGPU. ~6× faster on the backend (no per-substem vae.decode)."""
    src_n, src_den = src_meter
    tgt_n, tgt_den = tgt_meter
    silence_latent = make_silence_latent(silence_frame)

    encoded: Dict[str, Tuple[torch.Tensor, np.ndarray]] = {}
    for name, (wav, sr) in substems.items():
        if wav.ndim == 1:
            wav = np.stack([wav, wav], axis=-1)
        a48 = np.stack([
            librosa.resample(wav[:, c].astype(np.float32), orig_sr=sr, target_sr=SR)
            for c in range(wav.shape[1])
        ], axis=1)
        L = vae_encode_chunked(vae_gpu, a48)
        encoded[name] = (L, a48)

    hh_subdiv = "eighth"
    for name, (_, a48) in encoded.items():
        if classify_stem(name) == "force_quantize":
            hh_subdiv = detect_hh_subdivision(a48, bs_48, src_n, src_den)
            if verbose:
                print(f"  hh_subdivision_lock = {hh_subdiv}")
            break

    out_lats: Dict[str, torch.Tensor] = {}
    for name, (L, a48) in encoded.items():
        cls = classify_stem(name)
        if cls == "force_quantize":
            if hh_subdiv == "eighth":
                L_out, n_edits = latent_hh_literal_cut(
                    L, bs_48, src_n, src_den, tgt_n, tgt_den, silence_latent,
                )
                route = "cut/hh-eighth"
            else:
                L_out, n_edits = latent_pluck_and_place(
                    L, a48, bs_48, src_n, src_den, tgt_n, tgt_den,
                    hh_subdiv, editor, silence_latent,
                )
                route = f"pluck/hh-{hh_subdiv}"
        else:
            L_out, n_edits = latent_drum_accent_cut(
                L, a48, bs_48, src_n, src_den, tgt_n, tgt_den,
                editor, silence_latent,
            )
            route = "cut/accent" if cls == "accent" else "cut/mixed"
        out_lats[name] = L_out
        if verbose:
            print(f"    {name:7s}: route={route:18s} L_out={tuple(L_out.shape)} edits={n_edits}")
    return out_lats


def drum_meter_change_substems(
    substems: Dict[str, Tuple[np.ndarray, int]],
    bs_48: List[int],
    src_meter: Tuple[int, int],
    tgt_meter: Tuple[int, int],
    vae_gpu,
    editor,
    silence_frame: torch.Tensor,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Process a dict of drum substems through the generalized latent
    meter-change pipeline. Returns processed substems as 48k wav arrays
    keyed by the same names (caller can mix / save)."""
    src_n, src_den = src_meter
    tgt_n, tgt_den = tgt_meter
    silence_latent = make_silence_latent(silence_frame)

    # First pass: encode everything + find hh subdivision lock once.
    encoded: Dict[str, Tuple[torch.Tensor, np.ndarray]] = {}
    for name, (wav, sr) in substems.items():
        if wav.ndim == 1:
            wav = np.stack([wav, wav], axis=-1)
        a48 = np.stack([
            librosa.resample(wav[:, c].astype(np.float32), orig_sr=sr, target_sr=SR)
            for c in range(wav.shape[1])
        ], axis=1)
        L = vae_encode_chunked(vae_gpu, a48)
        encoded[name] = (L, a48)

    hh_subdiv = "eighth"
    for name, (_, a48) in encoded.items():
        if classify_stem(name) == "force_quantize":
            hh_subdiv = detect_hh_subdivision(a48, bs_48, src_n, src_den)
            if verbose:
                print(f"  hh_subdivision_lock = {hh_subdiv}")
            break

    # Second pass: route each substem.
    processed: Dict[str, np.ndarray] = {}
    for name, (L, a48) in encoded.items():
        cls = classify_stem(name)
        t0 = time.perf_counter()
        if cls == "accent" or cls == "mixed":
            L_out, n_edits = latent_drum_accent_cut(
                L, a48, bs_48, src_n, src_den, tgt_n, tgt_den,
                editor, silence_latent,
            )
            route = "cut/accent" if cls == "accent" else "cut/mixed"
        elif cls == "force_quantize":
            # If hh pattern already sits on the source eighth grid AND
            # we only need to drop/add whole eighth cells, literal cut
            # is sufficient (0 edits) and faster.
            if hh_subdiv == "eighth":
                L_out, n_edits = latent_hh_literal_cut(
                    L, bs_48, src_n, src_den, tgt_n, tgt_den, silence_latent,
                )
                route = "cut/hh-eighth"
            else:
                L_out, n_edits = latent_pluck_and_place(
                    L, a48, bs_48, src_n, src_den, tgt_n, tgt_den,
                    hh_subdiv, editor, silence_latent,
                )
                route = f"pluck/hh-{hh_subdiv}"
        else:
            L_out, n_edits = latent_drum_accent_cut(
                L, a48, bs_48, src_n, src_den, tgt_n, tgt_den,
                editor, silence_latent,
            )
            route = "cut/default"
        audio_out = vae_decode_chunked(vae_gpu, L_out)
        processed[name] = audio_out
        if verbose:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"    {name:7s}: route={route:18s} L_in={tuple(L.shape)} L_out={tuple(L_out.shape)} "
                  f"edits={n_edits:4d} time={elapsed:6.0f}ms")

    return processed


# ── Standalone driver (mirrors working_reference_universal.py) ───────
if __name__ == "__main__":
    import sys, soundfile as sf, hashlib
    sys.path.insert(0, "/home/arlo/do2")
    from latent_editor.infer import LatentEditorRuntime
    from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck
    import beat_this.inference as bti

    print("[init] loading vae + editor + silence frame…")
    vae_gpu = (AutoencoderOobleck
               .from_pretrained("/scratch/ACE-Step-1.5/checkpoints/vae")
               .to("cuda").to(torch.bfloat16).eval())
    editor_rt = LatentEditorRuntime("/scratch/latent_editor_ckpts/editor_final.pt", device="cuda")
    _sl_raw = torch.load(
        "/scratch/ACE-Step-1.5/checkpoints/acestep-v15-sft/silence_latent.pt",
        weights_only=True,
    ).transpose(1, 2).float()
    SILENCE_FRAME = _sl_raw[0, 0:1, :].clone()

    f2b = bti.File2Beats(checkpoint_path="final0", dbn=True)

    def detect_bs_48(src_path):
        beats, downbeats = f2b(src_path)
        bpm = 60.0 / float(np.median(np.diff(beats)))
        y_full, sr_in = sf.read(src_path, always_2d=True)
        offset = int(round(float(downbeats[0]) * sr_in))
        bs_src = [int(round(float(t) * sr_in)) - offset for t in downbeats]
        bs_src.append(len(y_full) - offset)
        return [int(round(b * SR / sr_in)) for b in bs_src], bpm

    def md5(path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            while True:
                b = f.read(1 << 16)
                if not b: break
                h.update(b)
        return h.hexdigest()[:12]

    SONGS = [
        {
            "label":     "tears",
            "src":       "/home/arlo/do2/time-sig-editor/tearsforfearseverybodywantstoruletheworldofficia.mp3",
            "sub_dir":   "/scratch/stemphonic_outputs/compare-2667846b/drum_stems",
            "out_dir":   "/scratch/stemphonic_outputs/generalized_tears",
            "reference": "/scratch/stemphonic_outputs/universal_tears/drums_meter.wav",
        },
        {
            "label":     "fortunate",
            "src":       "/home/arlo/do2/time-sig-editor/fortunate.wav",
            "sub_dir":   "/scratch/stemphonic_outputs/fortunate_v2_drums/drum_stems",
            "out_dir":   "/scratch/stemphonic_outputs/generalized_fortunate",
            "reference": "/scratch/stemphonic_outputs/universal_fortunate/drums_meter.wav",
        },
    ]

    for cfg in SONGS:
        label = cfg["label"]
        print(f"\n========== {label} ==========")
        bs_48, bpm = detect_bs_48(cfg["src"])
        print(f"  bpm={bpm:.2f}  bars={len(bs_48)-1}")

        substems = {}
        for p in sorted(glob.glob(os.path.join(cfg["sub_dir"], "*.wav"))):
            name = os.path.splitext(os.path.basename(p))[0]
            wav, sr = sf.read(p, always_2d=True)
            substems[name] = (wav, sr)

        processed = drum_meter_change_substems(
            substems=substems,
            bs_48=bs_48,
            src_meter=(4, 4),
            tgt_meter=(7, 8),
            vae_gpu=vae_gpu,
            editor=editor_rt,
            silence_frame=SILENCE_FRAME,
        )

        os.makedirs(cfg["out_dir"], exist_ok=True)
        ml = max(s.shape[0] for s in processed.values())
        mix = np.zeros((ml, 2), dtype=np.float32)
        for name, s in processed.items():
            n = s.shape[0]
            if s.shape[1] == 1:
                s = np.concatenate([s, s], axis=1)
            mix[:n] += s[:n]
        peak = np.abs(mix).max()
        if peak > 0.95:
            mix *= 0.95 / peak
        out_path = os.path.join(cfg["out_dir"], "drums_meter.wav")
        sf.write(out_path, mix, SR)
        print(f"  → {out_path}  md5={md5(out_path)}  shape={mix.shape}")

        ref = cfg.get("reference")
        if ref and os.path.exists(ref):
            ref_arr, _ = sf.read(ref, always_2d=True)
            n = min(ref_arr.shape[0], mix.shape[0])
            diff = float(np.sqrt(np.mean((ref_arr[:n] - mix[:n]) ** 2)))
            ref_rms = float(np.sqrt(np.mean(ref_arr[:n] ** 2)))
            print(f"  ref_md5={md5(ref)}  diff_rms={diff:.5f}  rel_err={diff/max(ref_rms,1e-9):.3f}")
