"""ACE-Step bridge for video_grid_music scene.

Handles:
- ACE-Step model loading (singleton)
- Grid timeseries → conditioning signal mapping
- Generation via genfromweb5.generate()
- Post-generation z-space sync enforcement using inverse_patch axes
- DCAE decode from edited z-latents
"""

import sys
import os
import random
import tempfile
import numpy as np
import torch
import torch.nn.functional as F

# Add Data dir for genfromweb5 imports
sys.path.insert(0, "/home/arlo/Data")
sys.path.insert(0, "/home/arlo/Data/ACE-Step")

from .. import config

# Lazy imports — heavy modules loaded only when needed
_MODEL = None
_AXES_CACHE = None

# ACE-Step constants
COND_FPS = 43.066
DCAE_SR = 44100
DCAE_HOP = 4096
DCAE_LATENT_FPS = DCAE_SR / DCAE_HOP  # ~10.77 fps

# Musical scales for piano roll construction
PENTATONIC = [0, 2, 4, 7, 9]  # C D E G A
MAJOR = [0, 2, 4, 5, 7, 9, 11]

# Instrument groups/subgroups for random selection
INSTRUMENT_POOL = [
    ("piano", "acoustic_piano"),
    ("piano", "keys"),
    ("guitar", "acoustic_guitar"),
    ("guitar", "electric_guitar"),
    ("bass", "electric_bass"),
    ("strings", "violin"),
    ("strings", "cello"),
    ("brass", "trumpet"),
    ("brass", "trombone"),
    ("winds", "flute"),
    ("winds", "clarinet"),
    ("winds", "sax"),
]

# Register-appropriate instruments for multi-stem band splitting
STEM_INSTRUMENT_POOLS = {
    "bass": [
        ("bass", "electric_bass"),
        ("strings", "cello"),
        ("brass", "trombone"),
    ],
    "mid": [
        ("piano", "acoustic_piano"),
        ("guitar", "acoustic_guitar"),
        ("guitar", "electric_guitar"),
        ("strings", "violin"),
        ("brass", "trumpet"),
        ("winds", "clarinet"),
        ("winds", "sax"),
    ],
    "high": [
        ("piano", "keys"),
        ("winds", "flute"),
        ("piano", "acoustic_piano"),
        ("strings", "violin"),
        ("winds", "clarinet"),
    ],
}

# Stem band colors for visualization (RGB)
STEM_BAND_COLORS = [
    (80, 140, 255),   # bass — blue
    (100, 220, 120),  # mid — green
    (255, 180, 80),   # high — orange
    (220, 100, 220),  # extra — purple
]


def is_model_available():
    """Check if ACE-Step checkpoint exists."""
    return os.path.isfile(config.ACESTEP_CKPT)


def load_model():
    """Load ACE-Step Pipeline (singleton, cached after first call)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from genfromweb5 import load_model_any_ckpt
    print("[acestep_bridge] Loading ACE-Step model...")
    _MODEL = load_model_any_ckpt(
        config.ACESTEP_CKPT,
        config.ACESTEP_CKPT_DIR,
        config.ACESTEP_MANIFEST,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _MODEL.to(device).eval()
    print(f"[acestep_bridge] Model loaded on {device}")
    return _MODEL


def pick_random_instrument(seed):
    """Pick a random (group, subgroup) pair, seeded for reproducibility."""
    rng = random.Random(seed)
    return rng.choice(INSTRUMENT_POOL)


def _resample_timeseries(arr, target_len):
    """Resample last axis of arr from current length to target_len via linear interp."""
    src_len = arr.shape[-1]
    if src_len == target_len:
        return arr
    x_src = np.linspace(0, 1, src_len)
    x_tgt = np.linspace(0, 1, target_len)
    # Handle multi-dim arrays
    orig_shape = arr.shape
    flat = arr.reshape(-1, src_len)
    result = np.zeros((flat.shape[0], target_len), dtype=np.float32)
    for i in range(flat.shape[0]):
        result[i] = np.interp(x_tgt, x_src, flat[i])
    return result.reshape(*orig_shape[:-1], target_len)


def build_conditioning_from_grid(
    motion_ts, hue_ts, sat_ts, bright_ts,
    grid_rows, grid_cols, motion_threshold=6.0,
    duration_sec=30.0,
    retrigger_mode="reactive",
):
    """
    Map grid timeseries → 5 ACE-Step conditioning arrays.

    Args:
        motion_ts: [rows, cols, T_video] smoothed motion per cell
        hue_ts: [rows, cols, T_video] mean hue per cell (0-179)
        sat_ts: [rows, cols, T_video] mean saturation per cell (0-255)
        bright_ts: [rows, cols, T_video] mean brightness per cell (0-255)
        grid_rows, grid_cols: grid dimensions
        motion_threshold: activation threshold
        duration_sec: video duration in seconds

    Returns:
        (pr, amp, rframe, rbend, encodec_tokens) numpy/torch arrays
        ready for genfromweb5.generate()
    """
    n_cond = int(duration_sec * COND_FPS)

    # Resample all from video fps to conditioning fps
    motion = _resample_timeseries(motion_ts.astype(np.float32), n_cond)
    hue = _resample_timeseries(hue_ts.astype(np.float32), n_cond)
    sat = _resample_timeseries(sat_ts.astype(np.float32), n_cond)
    bright = _resample_timeseries(bright_ts.astype(np.float32), n_cond)

    # 1. Piano Roll [128, T] — two modes:
    #    "reactive" — global event detection: note bursts only at macro motion changes
    #    "steady"   — per-cell retrigger at ~300ms while motion stays high (metronomic)
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks as _find_peaks
    pr = np.zeros((128, n_cond), dtype=np.float32)
    rframe = np.zeros(n_cond, dtype=np.float32)
    scale = PENTATONIC
    MAX_SIMULTANEOUS = 6

    if retrigger_mode == "reactive":
        # --- GLOBAL EVENT-DRIVEN approach ---
        # Detect macro-level motion events across the whole grid,
        # create note bursts at those moments, silence between them.
        # Target ~30-40% temporal coverage for the model to differentiate.

        # Global motion energy per frame
        global_energy = motion.sum(axis=(0, 1))  # [n_cond]
        smoothed = gaussian_filter1d(global_energy, sigma=3)

        # Detect motion peaks = "musical events"
        min_event_dist = int(COND_FPS * 0.5)  # min 500ms between events
        event_peaks, event_props = _find_peaks(
            smoothed, distance=min_event_dist,
            height=np.percentile(smoothed, 40),  # top 60% of motion moments
            prominence=smoothed.std() * 0.3,
        )

        # Also detect sharp rises (derivative peaks = scene cuts, fast motion onset)
        deriv = np.diff(smoothed, prepend=smoothed[0])
        deriv_peaks, _ = _find_peaks(
            deriv, distance=min_event_dist,
            height=np.percentile(deriv[deriv > 0], 70) if (deriv > 0).any() else 0,
        )
        all_events = sorted(set(event_peaks.tolist() + deriv_peaks.tolist()))

        # For each event, create a short note burst from the most active cells
        note_sustain = int(COND_FPS * 0.3)  # ~300ms per burst

        for evt_t in all_events:
            # Find top cells by motion at this moment
            cell_motions = []
            for r in range(grid_rows):
                for c in range(grid_cols):
                    cell_motions.append((motion[r, c, evt_t], r, c))
            cell_motions.sort(reverse=True)

            # Pick top N cells (up to MAX_SIMULTANEOUS)
            n_notes = min(MAX_SIMULTANEOUS, max(1, int(
                np.clip(smoothed[evt_t] / (smoothed.max() + 1e-8), 0.2, 1.0) * MAX_SIMULTANEOUS
            )))

            for m_val, r, c in cell_motions[:n_notes]:
                if m_val < motion_threshold * 0.3:
                    continue
                scale_idx = c % len(scale)
                octave_from_col = c // len(scale)
                row_octave = (grid_rows - 1 - r) // max(1, grid_rows // 3)
                pitch = (4 + octave_from_col + row_octave) * 12 + scale[scale_idx]
                pitch = max(36, min(96, pitch))

                velocity = np.clip(m_val / (motion_threshold * 2), 0.3, 1.0)
                velocity *= (0.5 + 0.5 * bright[r, c, evt_t] / 255.0)
                end = min(evt_t + note_sustain, n_cond)
                decay = np.exp(-np.arange(end - evt_t) / (note_sustain * 0.4))
                pr[pitch, evt_t:end] = np.maximum(pr[pitch, evt_t:end], velocity * decay)

            rframe[evt_t] = 1.0

        print(f"[cond] Reactive: {len(all_events)} global events detected")

    else:
        # --- STEADY per-cell retrigger approach (metronomic) ---
        retrigger_cooldown = int(COND_FPS * 0.3)
        note_sustain = int(COND_FPS * 0.12)

        for r in range(grid_rows):
            for c in range(grid_cols):
                scale_idx = c % len(scale)
                octave_from_col = c // len(scale)
                row_octave = (grid_rows - 1 - r) // max(1, grid_rows // 3)
                pitch = (4 + octave_from_col + row_octave) * 12 + scale[scale_idx]
                pitch = max(36, min(96, pitch))

                m = motion[r, c]
                above = m > motion_threshold
                last_trig = -retrigger_cooldown
                for t in range(n_cond):
                    if above[t] and (t - last_trig) >= retrigger_cooldown:
                        velocity = np.clip(m[t] / (motion_threshold * 2), 0.3, 1.0)
                        velocity *= (0.5 + 0.5 * bright[r, c, t] / 255.0)
                        end = min(t + note_sustain, n_cond)
                        decay = np.exp(-np.arange(end - t) / (note_sustain * 0.4))
                        pr[pitch, t:end] = np.maximum(pr[pitch, t:end], velocity * decay)
                        rframe[t] = 1.0
                        last_trig = t

        # Sparsify to max simultaneous notes
        for t in range(n_cond):
            active_idx = np.where(pr[:, t] > 0.05)[0]
            if len(active_idx) > MAX_SIMULTANEOUS:
                vals = pr[active_idx, t]
                keep = active_idx[np.argsort(vals)[-MAX_SIMULTANEOUS:]]
                mask = np.zeros(128, dtype=np.float32)
                mask[keep] = 1.0
                pr[:, t] *= mask

    pr_active = (pr > 0.1).sum()
    pr_active_frames = (pr.max(axis=0) > 0.1).sum()
    print(f"[cond] Retrigger mode: {retrigger_mode}")
    print(f"[cond] Piano roll: {pr_active} active bins, "
          f"{pr_active_frames}/{n_cond} frames with notes ({pr_active_frames/n_cond*100:.1f}%)")
    print(f"[cond] Rframe onsets: {int(rframe.sum())}")

    # 2. Amplitude [T] — motion energy, sustained
    global_motion = motion.sum(axis=(0, 1))  # [n_cond]
    amp_max = global_motion.max() + 1e-8
    amp = global_motion / amp_max
    amp = gaussian_filter1d(amp, sigma=2).astype(np.float32)
    amp = np.clip(amp, 0.1, 1.0)
    print(f"[cond] Amplitude: mean={amp.mean():.3f}, min={amp.min():.3f}, max={amp.max():.3f}")

    # 4. Rbend [T] — hue velocity across active cells
    hue_deriv = np.diff(hue, axis=-1, prepend=hue[:, :, :1])
    hue_deriv = np.where(hue_deriv > 90, hue_deriv - 179, hue_deriv)
    hue_deriv = np.where(hue_deriv < -90, hue_deriv + 179, hue_deriv)
    active_mask = motion > (motion_threshold * 0.3)
    active_count = active_mask.sum(axis=(0, 1)) + 1e-8
    weighted_deriv = (hue_deriv * active_mask).sum(axis=(0, 1)) / active_count
    rbend_max = np.abs(weighted_deriv).max() + 1e-8
    rbend = np.clip(weighted_deriv / rbend_max, -1, 1).astype(np.float32)

    # 5. EnCodec tokens [1, 8, T_enc] — zeroed (not needed, model ignores this stream)
    T_enc = n_cond // 4
    encodec_tokens = torch.zeros((1, 8, T_enc), dtype=torch.long)

    return pr, amp, rframe, rbend, encodec_tokens


def generate_from_grid_analysis(
    motion_ts, hue_ts, sat_ts, bright_ts,
    grid_rows, grid_cols,
    motion_threshold=6.0,
    duration_sec=30.0,
    seed=42,
    noise_level=1.0,
    steps=30,
    cfg_weight=3.0,
    adapter_scale=1.0,
    temp_dir=None,
    retrigger_mode="reactive",
):
    """
    Full pipeline: grid analysis → conditioning → ACE-Step generation.

    Returns dict with:
        audio_path: str — path to generated WAV
        z_latents: torch.Tensor [1, 8, 16, T_lat] — raw z-space latents
        z_flat: torch.Tensor [T_lat, 128] — flattened for visualization
        group: str — instrument group used
        subgroup: str — instrument subgroup used
    """
    model = load_model()

    # Pick random instrument
    group, subgroup = pick_random_instrument(seed)
    print(f"[acestep_bridge] Instrument: {group}/{subgroup}")

    # Build conditioning
    pr, amp, rframe, rbend, encodec_tokens = build_conditioning_from_grid(
        motion_ts, hue_ts, sat_ts, bright_ts,
        grid_rows, grid_cols, motion_threshold, duration_sec,
        retrigger_mode=retrigger_mode,
    )
    print(f"[acestep_bridge] Conditioning: pr={pr.shape}, amp={amp.shape}, "
          f"rframe={rframe.shape}, rbend={rbend.shape}, enc={encodec_tokens.shape}")

    # Generate via genfromweb5
    from genfromweb5 import generate as ace_generate

    # Save CWD, change to temp dir for output
    orig_cwd = os.getcwd()
    out_dir = temp_dir or tempfile.mkdtemp(prefix="acestep_grid_")
    os.makedirs(os.path.join(out_dir, "generated_ui"), exist_ok=True)
    os.chdir(out_dir)

    try:
        result = ace_generate(
            model=model,
            piano_roll=pr,
            amp=amp,
            rframe=rframe,
            rbend=rbend,
            encodec_tokens=encodec_tokens,
            group=group,
            subgroup=subgroup,
            steps=steps,
            seed=seed,
            adapter_scale=adapter_scale,
            cfg_weight=cfg_weight,
            t0=1.0,
            sr_out=DCAE_SR,
            instrument_strength=1.5,
            inst_boost=2.5,
            piano_roll_gain=1.5,
            amp_gain=1.5,
            rframe_gain=1.0,
            rbend_gain=0.5,
            encodec_gain=0.3,      # low but >0 to avoid zeroing film/channel pathways
            use_overlap_decoder=True,
            noise_level=noise_level,
            return_latents=True,
        )
    finally:
        os.chdir(orig_cwd)

    audio_path, z_latents = result  # z_latents: [1, 8, 16, T_lat]
    # Make absolute — ace_generate returns path relative to out_dir
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(out_dir, audio_path)
    T_lat = z_latents.shape[-1]
    z_flat = z_latents.reshape(1, 128, T_lat).permute(0, 2, 1).squeeze(0)  # [T_lat, 128]

    print(f"[acestep_bridge] Generated: {audio_path}, z_latents={z_latents.shape}")

    return {
        "audio_path": audio_path,
        "z_latents": z_latents,  # [1, 8, 16, T_lat]
        "z_flat": z_flat,        # [T_lat, 128]
        "group": group,
        "subgroup": subgroup,
    }


def _split_grid_into_bands(grid_rows, num_stems):
    """Split grid rows into horizontal bands for multi-stem generation.

    Returns list of (row_start, row_end, band_name) tuples.
    Bottom rows = bass, top rows = high.
    """
    if num_stems == 1:
        return [(0, grid_rows, "mid")]

    band_names = ["bass", "mid", "high", "high"]  # up to 4 stems
    rows_per_band = grid_rows / num_stems
    bands = []
    for i in range(num_stems):
        row_start = int(round(i * rows_per_band))
        row_end = int(round((i + 1) * rows_per_band))
        row_end = min(row_end, grid_rows)
        # Bottom rows (high row indices) = bass, top rows (low indices) = high
        band_idx = num_stems - 1 - i
        name = band_names[min(band_idx, len(band_names) - 1)]
        bands.append((row_start, row_end, name))
    return bands


def build_conditioning_from_grid_subset(
    motion_ts, hue_ts, sat_ts, bright_ts,
    grid_rows, grid_cols,
    row_start, row_end,
    band_name="mid",
    motion_threshold=6.0,
    duration_sec=30.0,
    retrigger_mode="reactive",
):
    """Build conditioning signals from a subset of grid rows (one horizontal band).

    Same output format as build_conditioning_from_grid but only analyzes
    rows [row_start, row_end). Pitch range is adjusted by band register.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks as _find_peaks

    n_cond = int(duration_sec * COND_FPS)
    band_rows = row_end - row_start

    # Extract and resample just this band's timeseries
    motion_band = motion_ts[row_start:row_end, :, :]
    hue_band = hue_ts[row_start:row_end, :, :]
    sat_band = sat_ts[row_start:row_end, :, :]
    bright_band = bright_ts[row_start:row_end, :, :]

    motion = _resample_timeseries(motion_band.astype(np.float32), n_cond)
    hue = _resample_timeseries(hue_band.astype(np.float32), n_cond)
    sat = _resample_timeseries(sat_band.astype(np.float32), n_cond)
    bright = _resample_timeseries(bright_band.astype(np.float32), n_cond)

    # Register-appropriate pitch/scale settings
    if band_name == "bass":
        scale = [0, 3, 5, 7, 10]  # minor pentatonic — fits bass
        pitch_lo, pitch_hi = 28, 55
        MAX_SIMULTANEOUS = 3
    elif band_name == "high":
        scale = PENTATONIC
        pitch_lo, pitch_hi = 65, 96
        MAX_SIMULTANEOUS = 5
    else:  # mid
        scale = MAJOR
        pitch_lo, pitch_hi = 48, 78
        MAX_SIMULTANEOUS = 6

    pr = np.zeros((128, n_cond), dtype=np.float32)
    rframe = np.zeros(n_cond, dtype=np.float32)

    if retrigger_mode == "reactive":
        global_energy = motion.sum(axis=(0, 1))
        smoothed = gaussian_filter1d(global_energy, sigma=3)

        min_event_dist = int(COND_FPS * 0.5)
        pct_thresh = np.percentile(smoothed, 40) if smoothed.max() > 0 else 0
        event_peaks, _ = _find_peaks(
            smoothed, distance=min_event_dist,
            height=pct_thresh,
            prominence=smoothed.std() * 0.3 if smoothed.std() > 0 else 0,
        )

        deriv = np.diff(smoothed, prepend=smoothed[0])
        pos_deriv = deriv[deriv > 0]
        deriv_thresh = np.percentile(pos_deriv, 70) if len(pos_deriv) > 0 else 0
        deriv_peaks, _ = _find_peaks(
            deriv, distance=min_event_dist, height=deriv_thresh,
        )
        all_events = sorted(set(event_peaks.tolist() + deriv_peaks.tolist()))

        note_sustain = int(COND_FPS * 0.3)
        for evt_t in all_events:
            cell_motions = []
            for r in range(band_rows):
                for c in range(grid_cols):
                    cell_motions.append((motion[r, c, evt_t], r, c))
            cell_motions.sort(reverse=True)

            energy_ratio = smoothed[evt_t] / (smoothed.max() + 1e-8)
            n_notes = min(MAX_SIMULTANEOUS, max(1, int(
                np.clip(energy_ratio, 0.2, 1.0) * MAX_SIMULTANEOUS
            )))

            for m_val, r, c in cell_motions[:n_notes]:
                if m_val < motion_threshold * 0.3:
                    continue
                # Map column to pitch within band's register
                col_norm = c / max(1, grid_cols - 1)
                pitch = int(pitch_lo + col_norm * (pitch_hi - pitch_lo))
                # Snap to scale
                pitch_pc = pitch % 12
                best = min(scale, key=lambda s: min(abs(pitch_pc - s), 12 - abs(pitch_pc - s)))
                pitch = (pitch // 12) * 12 + best
                pitch = max(pitch_lo, min(pitch_hi, pitch))

                velocity = np.clip(m_val / (motion_threshold * 2), 0.3, 1.0)
                velocity *= (0.5 + 0.5 * bright[r, c, evt_t] / 255.0)
                end = min(evt_t + note_sustain, n_cond)
                decay = np.exp(-np.arange(end - evt_t) / (note_sustain * 0.4))
                pr[pitch, evt_t:end] = np.maximum(pr[pitch, evt_t:end], velocity * decay)

            rframe[evt_t] = 1.0

        print(f"[cond:{band_name}] Reactive: {len(all_events)} events, "
              f"rows {row_start}-{row_end}, pitches {pitch_lo}-{pitch_hi}")
    else:
        retrigger_cooldown = int(COND_FPS * 0.3)
        note_sustain = int(COND_FPS * 0.12)

        for r in range(band_rows):
            for c in range(grid_cols):
                col_norm = c / max(1, grid_cols - 1)
                pitch = int(pitch_lo + col_norm * (pitch_hi - pitch_lo))
                pitch_pc = pitch % 12
                best = min(scale, key=lambda s: min(abs(pitch_pc - s), 12 - abs(pitch_pc - s)))
                pitch = (pitch // 12) * 12 + best
                pitch = max(pitch_lo, min(pitch_hi, pitch))

                m = motion[r, c]
                above = m > motion_threshold
                last_trig = -retrigger_cooldown
                for t in range(n_cond):
                    if above[t] and (t - last_trig) >= retrigger_cooldown:
                        velocity = np.clip(m[t] / (motion_threshold * 2), 0.3, 1.0)
                        velocity *= (0.5 + 0.5 * bright[r, c, t] / 255.0)
                        end = min(t + note_sustain, n_cond)
                        decay = np.exp(-np.arange(end - t) / (note_sustain * 0.4))
                        pr[pitch, t:end] = np.maximum(pr[pitch, t:end], velocity * decay)
                        rframe[t] = 1.0
                        last_trig = t

        # Sparsify
        for t in range(n_cond):
            active_idx = np.where(pr[:, t] > 0.05)[0]
            if len(active_idx) > MAX_SIMULTANEOUS:
                vals = pr[active_idx, t]
                keep = active_idx[np.argsort(vals)[-MAX_SIMULTANEOUS:]]
                mask = np.zeros(128, dtype=np.float32)
                mask[keep] = 1.0
                pr[:, t] *= mask

    # Amplitude
    global_motion = motion.sum(axis=(0, 1))
    amp_max = global_motion.max() + 1e-8
    amp = global_motion / amp_max
    amp = gaussian_filter1d(amp, sigma=2).astype(np.float32)
    amp = np.clip(amp, 0.1, 1.0)

    # Rbend
    hue_deriv = np.diff(hue, axis=-1, prepend=hue[:, :, :1])
    hue_deriv = np.where(hue_deriv > 90, hue_deriv - 179, hue_deriv)
    hue_deriv = np.where(hue_deriv < -90, hue_deriv + 179, hue_deriv)
    active_mask = motion > (motion_threshold * 0.3)
    active_count = active_mask.sum(axis=(0, 1)) + 1e-8
    weighted_deriv = (hue_deriv * active_mask).sum(axis=(0, 1)) / active_count
    rbend_max = np.abs(weighted_deriv).max() + 1e-8
    rbend = np.clip(weighted_deriv / rbend_max, -1, 1).astype(np.float32)

    # EnCodec tokens (zeroed)
    T_enc = n_cond // 4
    encodec_tokens = torch.zeros((1, 8, T_enc), dtype=torch.long)

    return pr, amp, rframe, rbend, encodec_tokens


def generate_multi_stem_from_grid(
    motion_ts, hue_ts, sat_ts, bright_ts,
    grid_rows, grid_cols,
    num_stems=3,
    motion_threshold=6.0,
    duration_sec=30.0,
    seed=42,
    noise_level=1.0,
    steps=30,
    cfg_weight=3.0,
    adapter_scale=1.0,
    temp_dir=None,
    retrigger_mode="reactive",
):
    """
    Multi-stem pipeline: split grid into horizontal bands, generate one ACE-Step
    stem per band with register-appropriate instruments.

    Returns list of dicts, each with:
        audio_path, z_latents, z_flat, group, subgroup, band_name, row_start, row_end
    """
    model = load_model()
    bands = _split_grid_into_bands(grid_rows, num_stems)
    rng = random.Random(seed)

    results = []
    for band_idx, (row_start, row_end, band_name) in enumerate(bands):
        band_seed = seed + band_idx * 1000

        # Pick register-appropriate instrument
        pool = STEM_INSTRUMENT_POOLS.get(band_name, STEM_INSTRUMENT_POOLS["mid"])
        group, subgroup = rng.choice(pool)
        print(f"[multi-stem] Band {band_idx} ({band_name}): rows {row_start}-{row_end}, "
              f"instrument {group}/{subgroup}")

        # Build band-specific conditioning
        pr, amp, rframe, rbend, encodec_tokens = build_conditioning_from_grid_subset(
            motion_ts, hue_ts, sat_ts, bright_ts,
            grid_rows, grid_cols,
            row_start, row_end,
            band_name=band_name,
            motion_threshold=motion_threshold,
            duration_sec=duration_sec,
            retrigger_mode=retrigger_mode,
        )

        # Generate via ACE-Step
        from genfromweb5 import generate as ace_generate

        orig_cwd = os.getcwd()
        out_dir = temp_dir or tempfile.mkdtemp(prefix=f"acestep_stem{band_idx}_")
        stem_out_dir = os.path.join(out_dir, f"stem_{band_idx}")
        os.makedirs(os.path.join(stem_out_dir, "generated_ui"), exist_ok=True)
        os.chdir(stem_out_dir)

        try:
            result = ace_generate(
                model=model,
                piano_roll=pr,
                amp=amp,
                rframe=rframe,
                rbend=rbend,
                encodec_tokens=encodec_tokens,
                group=group,
                subgroup=subgroup,
                steps=steps,
                seed=band_seed,
                adapter_scale=adapter_scale,
                cfg_weight=cfg_weight,
                t0=1.0,
                sr_out=DCAE_SR,
                instrument_strength=1.5,
                inst_boost=2.5,
                piano_roll_gain=1.5,
                amp_gain=1.5,
                rframe_gain=1.0,
                rbend_gain=0.5,
                encodec_gain=0.3,
                use_overlap_decoder=True,
                noise_level=noise_level,
                return_latents=True,
            )
        finally:
            os.chdir(orig_cwd)

        audio_path, z_latents = result
        # Make absolute — ace_generate returns path relative to stem_out_dir
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(stem_out_dir, audio_path)
        T_lat = z_latents.shape[-1]
        z_flat = z_latents.reshape(1, 128, T_lat).permute(0, 2, 1).squeeze(0)

        print(f"[multi-stem] Band {band_idx} ({band_name}): generated {audio_path}")

        results.append({
            "audio_path": audio_path,
            "z_latents": z_latents,
            "z_flat": z_flat,
            "group": group,
            "subgroup": subgroup,
            "band_name": band_name,
            "band_idx": band_idx,
            "row_start": row_start,
            "row_end": row_end,
        })

    return results


# ============================================================
# Inverse Patch Axes Loading
# ============================================================

def load_inverse_patch_axes():
    """Load validated z-space axes and operations (cached)."""
    global _AXES_CACHE
    if _AXES_CACHE is not None:
        return _AXES_CACHE

    axes = {}

    # Contrastive PCA/ICA axes
    if os.path.isfile(config.CONTRASTIVE_AXES):
        data = torch.load(config.CONTRASTIVE_AXES, weights_only=False, map_location="cpu")
        # within_pca: list of {'direction': array[128], 'variance_explained': float}
        pca_dirs = []
        for ax in data.get("within_pca", [])[:8]:
            d = ax["direction"]
            if isinstance(d, list):
                d = np.array(d, dtype=np.float32)
            pca_dirs.append(torch.from_numpy(d).float())
        axes["contrastive_pca"] = pca_dirs

        ica_dirs = []
        for d in data.get("within_ica", [])[:6]:
            if isinstance(d, list):
                d = np.array(d, dtype=np.float32)
            ica_dirs.append(torch.from_numpy(np.array(d)).float())
        axes["contrastive_ica"] = ica_dirs
        print(f"[axes] Loaded {len(pca_dirs)} PCA + {len(ica_dirs)} ICA contrastive axes")

    # Pitchbin axes
    if os.path.isfile(config.PITCHBIN_AXES):
        data = torch.load(config.PITCHBIN_AXES, weights_only=False, map_location="cpu")
        pb_dirs = []
        for ax in data.get("pitchbin_pca", [])[:6]:
            d = ax["direction"]
            if isinstance(d, list):
                d = np.array(d, dtype=np.float32)
            pb_dirs.append(torch.from_numpy(d).float())
        axes["pitchbin_pca"] = pb_dirs
        print(f"[axes] Loaded {len(pb_dirs)} pitchbin axes")

    # Operation tree (optional, heavier)
    if os.path.isfile(config.OPERATION_TREE):
        try:
            # Add scripts dir so OperationTreeCodec can be imported
            scripts_dir = os.path.join(
                os.path.dirname(config.INVERSE_PATCH_DIR), "scripts"
            )
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from phase2_operation_tree import OperationTreeCodec

            ckpt = torch.load(config.OPERATION_TREE, weights_only=False, map_location="cpu")
            tree = OperationTreeCodec(
                z_dim=128,
                n_ops=ckpt["n_ops"],
                param_dim=ckpt["param_dim"],
                top_k=ckpt["top_k"],
            )
            tree.load_state_dict(ckpt["model"])
            tree.eval()
            axes["operation_tree"] = tree
            print(f"[axes] Loaded operation tree ({ckpt['n_ops']} ops, top_k={ckpt['top_k']})")
        except Exception as e:
            print(f"[axes] Could not load operation tree: {e}")

    _AXES_CACHE = axes
    return axes


# ============================================================
# Post-Generation Z-Space Sync Enforcement
# ============================================================

def post_process_z_for_sync(z_latents, motion_ts, grid_rows, grid_cols, duration_sec=30.0):
    """
    Detect sync mismatches between video motion and z-space energy,
    then edit z along validated axes to force alignment.

    Args:
        z_latents: [1, 8, 16, T_lat] from generation
        motion_ts: [rows, cols, T_video] motion timeseries at video fps
        grid_rows, grid_cols: grid dimensions
        duration_sec: video duration

    Returns:
        z_edited: [1, 8, 16, T_lat] — possibly edited z-latents
        n_edits: int — number of sync corrections applied
    """
    from scipy.signal import find_peaks

    T_lat = z_latents.shape[-1]
    z_flat = z_latents.reshape(1, 128, T_lat).permute(0, 2, 1).squeeze(0)  # [T_lat, 128]

    # Video motion envelope (sum across all cells) at video fps
    motion_envelope = motion_ts.sum(axis=(0, 1))  # [T_video]
    T_video = len(motion_envelope)
    video_fps = T_video / duration_sec

    # Z-space energy envelope
    z_energy = (z_flat ** 2).sum(dim=-1).numpy()  # [T_lat]

    # Find peaks
    motion_peak_height = np.percentile(motion_envelope, 70) if len(motion_envelope) > 10 else 1.0
    z_peak_height = np.percentile(z_energy, 70) if len(z_energy) > 10 else 0.1

    video_peaks, _ = find_peaks(motion_envelope, height=motion_peak_height, distance=int(video_fps * 0.3))
    z_peaks, _ = find_peaks(z_energy, height=z_peak_height, distance=max(1, int(DCAE_LATENT_FPS * 0.3)))

    video_peak_times = video_peaks / video_fps
    z_peak_times = z_peaks / DCAE_LATENT_FPS

    # Match video peaks to z peaks, find mismatches > 150ms
    mismatches = []
    for vt in video_peak_times:
        if len(z_peak_times) == 0:
            mismatches.append((vt, None, float("inf")))
            continue
        dists = np.abs(z_peak_times - vt)
        nearest_idx = np.argmin(dists)
        offset = z_peak_times[nearest_idx] - vt
        if abs(offset) > 0.15:
            mismatches.append((vt, z_peak_times[nearest_idx], offset))

    if not mismatches:
        print(f"[z-sync] Audio already synced ({len(video_peaks)} motion peaks matched)")
        return z_latents, 0

    print(f"[z-sync] Found {len(mismatches)} sync mismatches (>150ms), applying z-space corrections...")

    # Load axes for editing
    axes = load_inverse_patch_axes()
    pca_axes = axes.get("contrastive_pca", [])
    if not pca_axes:
        print("[z-sync] No contrastive axes available, skipping sync correction")
        return z_latents, 0

    # Use first PCA axis as energy/loudness axis
    energy_axis = pca_axes[0].clone()
    energy_axis = energy_axis / (energy_axis.norm() + 1e-8)

    z_edited = z_flat.clone()  # [T_lat, 128]

    for video_time, z_time, offset in mismatches:
        target_frame = int(video_time * DCAE_LATENT_FPS)
        target_frame = min(target_frame, T_lat - 1)

        # Motion magnitude at this video time (for boost scaling)
        vid_frame = int(video_time * video_fps)
        vid_frame = min(vid_frame, T_video - 1)
        motion_mag = motion_envelope[vid_frame] / (motion_envelope.max() + 1e-8)

        # Boost z along energy axis at video peak, gaussian window
        window_radius = 4
        for df in range(-window_radius, window_radius + 1):
            f = target_frame + df
            if 0 <= f < T_lat:
                w = np.exp(-0.5 * (df / 2.0) ** 2)
                delta = motion_mag * 1.5 * w
                z_edited[f] = z_edited[f] + delta * energy_axis

        # Attenuate the misplaced z-peak if it was far off
        if z_time is not None and abs(offset) > 0.2:
            misplaced_frame = int(z_time * DCAE_LATENT_FPS)
            for df in range(-2, 3):
                f = misplaced_frame + df
                if 0 <= f < T_lat:
                    w = np.exp(-0.5 * (df / 1.5) ** 2) * 0.4
                    proj = (z_edited[f] * energy_axis).sum()
                    z_edited[f] = z_edited[f] - w * proj * energy_axis

    # Reshape back to [1, 8, 16, T_lat]
    z_result = z_edited.unsqueeze(0).permute(0, 2, 1).reshape(1, 8, 16, T_lat)

    print(f"[z-sync] Applied {len(mismatches)} sync corrections")
    return z_result, len(mismatches)


def decode_z_to_audio(z_latents, duration_sec=30.0, temp_dir=None):
    """
    Decode z-latents to audio via DCAE.

    Args:
        z_latents: [1, 8, 16, T_lat]
        duration_sec: target audio duration
        temp_dir: directory for output WAV

    Returns:
        audio_path: str — path to decoded WAV
    """
    import torchaudio
    from genfromweb5 import apply_final_audio_processing

    model = load_model()

    # Ensure DCAE is on GPU (it may have been left on CPU after transformer offload)
    cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.dcae.to(cuda_device)

    p = next(model.dcae.parameters(), None)
    dcae_device = p.device if p is not None else cuda_device
    dcae_dtype = p.dtype if p is not None else torch.float32

    x = z_latents.to(device=dcae_device, dtype=dcae_dtype)
    audio_len = int(duration_sec * DCAE_SR)
    audio_lengths = torch.tensor([audio_len], dtype=torch.long, device=dcae_device)

    # Free GPU memory before decode
    if dcae_device.type == "cuda":
        torch.cuda.empty_cache()

    print("[acestep_bridge] Decoding z-latents to audio...")
    if hasattr(model.dcae, "decode_overlap"):
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=(dcae_device.type == "cuda")
        ):
            sr_pred, wav_pred = model.dcae.decode_overlap(
                x, audio_lengths=audio_lengths, sr=DCAE_SR
            )
    else:
        sr_pred, wav_pred = model.dcae.decode(x, audio_lengths=audio_lengths, sr=DCAE_SR)

    wav = wav_pred[0].float().cpu()
    wav = apply_final_audio_processing(wav, sample_rate=sr_pred)

    out_dir = temp_dir or tempfile.mkdtemp(prefix="acestep_decode_")
    # Use unique filename to avoid overwrites during multi-stem decodes
    import time as _time
    out_path = os.path.join(out_dir, f"z_synced_audio_{int(_time.time()*1000) % 100000}.wav")
    torchaudio.save(out_path, wav, sr_pred)
    print(f"[acestep_bridge] Decoded audio: {out_path}")
    return out_path


# ============================================================
# Z-Space Visualization Helpers
# ============================================================

def z_frame_to_grid_colors(z_frame, grid_rows, grid_cols, pca_axes=None):
    """
    Map a single z-space frame [128] to per-cell RGB colors.

    Uses 3 contrastive PCA axes as RGB channels for a subtle color tint.

    Returns:
        grid_colors: numpy array [grid_rows, grid_cols, 3] uint8
    """
    if pca_axes is None or len(pca_axes) < 3:
        return np.zeros((grid_rows, grid_cols, 3), dtype=np.uint8)

    z = z_frame.float()

    # Project onto first 3 PCA axes → 3 scalars
    projections = torch.stack([
        (z * pca_axes[0]).sum(),
        (z * pca_axes[1]).sum(),
        (z * pca_axes[2]).sum(),
    ])

    # Normalize to [0, 1] centered at 0.5 (sigma ~3)
    normalized = (projections / 3.0 * 0.5 + 0.5).clamp(0, 1).numpy()

    # Base RGB from projections
    base_r = normalized[0]
    base_g = normalized[1]
    base_b = normalized[2]

    # Per-cell variation using z-space dims
    grid_colors = np.zeros((grid_rows, grid_cols, 3), dtype=np.uint8)
    z_np = z.numpy()
    cells = grid_rows * grid_cols

    for r in range(grid_rows):
        for c in range(grid_cols):
            cell_idx = r * grid_cols + c
            dim_idx = cell_idx % 128
            cell_energy = min(1.0, abs(z_np[dim_idx]) / 2.0)
            mod = 0.3 + 0.7 * cell_energy

            grid_colors[r, c, 0] = int(base_r * mod * 255)
            grid_colors[r, c, 1] = int(base_g * mod * 255)
            grid_colors[r, c, 2] = int(base_b * mod * 255)

    return grid_colors
