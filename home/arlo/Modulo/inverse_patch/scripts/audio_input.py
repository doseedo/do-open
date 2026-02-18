#!/usr/bin/env python3
"""Real-audio input pipeline for inverse synthesis.

Handles: loading, resampling, mono conversion, normalization,
onset detection, note segmentation, and variable-length audio.

Usage:
    from audio_input import load_and_segment

    segments = load_and_segment("recording.wav")
    for seg in segments:
        result = optimize_patch_full(seg['audio'], pitch=seg['pitch'])
"""

import numpy as np
from pathlib import Path

SAMPLE_RATE = 44100
DEFAULT_DURATION = 2.0
DEFAULT_N_SAMPLES = int(SAMPLE_RATE * DEFAULT_DURATION)


def load_audio(path, sr=SAMPLE_RATE):
    """Load audio file, convert to mono float32, resample to target sr.

    Returns (audio_np, sample_rate).
    """
    import torchaudio
    import torch

    path = str(path)
    waveform, orig_sr = torchaudio.load(path)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = resampler(waveform)

    audio = waveform.squeeze(0).numpy().astype(np.float32)
    return audio, sr


def normalize(audio, target_peak=0.8):
    """Peak-normalize audio."""
    peak = np.abs(audio).max()
    if peak > 1e-6:
        return (audio / peak * target_peak).astype(np.float32)
    return audio.astype(np.float32)


def trim_silence(audio, sr=SAMPLE_RATE, threshold_db=-40, min_silence_ms=50):
    """Trim leading and trailing silence.

    Returns (trimmed_audio, start_sample, end_sample).
    """
    threshold = 10 ** (threshold_db / 20.0)
    frame_size = max(int(sr * min_silence_ms / 1000), 64)
    n_frames = len(audio) // frame_size

    if n_frames < 1:
        return audio, 0, len(audio)

    rms = np.array([
        np.sqrt(np.mean(audio[i * frame_size:(i + 1) * frame_size] ** 2))
        for i in range(n_frames)
    ])

    active = np.where(rms > threshold)[0]
    if len(active) == 0:
        return audio, 0, len(audio)

    start = max(0, active[0] * frame_size - frame_size)  # small pre-roll
    end = min(len(audio), (active[-1] + 2) * frame_size)  # small post-roll
    return audio[start:end], start, end


def detect_onsets(audio, sr=SAMPLE_RATE, threshold_db=-20, min_gap_ms=200):
    """Detect note onsets using energy-based onset detection.

    Returns list of onset sample positions.
    """
    frame_ms = 10
    frame_size = sr * frame_ms // 1000
    n_frames = len(audio) // frame_size
    min_gap_frames = max(int(min_gap_ms / frame_ms), 1)

    if n_frames < 3:
        return [0]

    # RMS per frame
    rms = np.array([
        np.sqrt(np.mean(audio[i * frame_size:(i + 1) * frame_size] ** 2))
        for i in range(n_frames)
    ])

    # Onset detection function: positive first derivative of RMS (energy rise)
    odf = np.diff(rms)
    odf = np.maximum(odf, 0)  # only rising edges

    # Adaptive threshold: mean + 2*std of the ODF
    threshold_abs = 10 ** (threshold_db / 20.0)
    odf_threshold = max(np.mean(odf) + 2.0 * np.std(odf), threshold_abs * 0.1)

    # Find peaks in ODF above threshold with minimum gap
    onsets = []
    last_onset = -min_gap_frames
    for i in range(1, len(odf) - 1):
        if odf[i] > odf_threshold and odf[i] >= odf[i - 1] and odf[i] >= odf[i + 1]:
            if i - last_onset >= min_gap_frames:
                # Refine: look back for the actual energy start (where RMS first rises)
                start_frame = i
                while start_frame > 0 and rms[start_frame] > rms[start_frame - 1] * 0.8:
                    start_frame -= 1
                onsets.append(start_frame * frame_size)
                last_onset = i

    if not onsets:
        # No clear onsets — treat whole signal as one note
        # Find the first frame above threshold
        active = np.where(rms > threshold_abs)[0]
        if len(active) > 0:
            return [max(0, active[0] * frame_size - frame_size)]
        return [0]

    return onsets


def segment_notes(audio, onsets, sr=SAMPLE_RATE, max_duration_s=3.0,
                  min_duration_s=0.1, target_duration_s=DEFAULT_DURATION):
    """Segment audio at onset points into individual note regions.

    Each segment is padded/trimmed to target_duration_s for the optimizer.
    Returns list of dicts with 'audio', 'start_s', 'duration_s', 'original_audio'.
    """
    max_samples = int(max_duration_s * sr)
    min_samples = int(min_duration_s * sr)
    target_samples = int(target_duration_s * sr)

    segments = []
    for i, onset in enumerate(onsets):
        # End is either next onset or end of audio
        if i + 1 < len(onsets):
            end = onsets[i + 1]
        else:
            end = len(audio)

        # Clamp duration
        duration = min(end - onset, max_samples)
        if duration < min_samples:
            continue

        seg = audio[onset:onset + duration].copy()

        # Store original length before padding
        original_len = len(seg)

        # Pad or trim to target length for optimizer
        if len(seg) < target_samples:
            # Zero-pad (natural fade to silence)
            padded = np.zeros(target_samples, dtype=np.float32)
            padded[:len(seg)] = seg
            seg_padded = padded
        elif len(seg) > target_samples:
            # Apply fade-out at target boundary
            seg_padded = seg[:target_samples].copy()
            fade_len = min(int(0.05 * sr), target_samples // 4)  # 50ms fade
            if fade_len > 0:
                fade = np.linspace(1, 0, fade_len, dtype=np.float32)
                seg_padded[-fade_len:] *= fade
        else:
            seg_padded = seg

        segments.append({
            'audio': normalize(seg_padded),
            'original_audio': normalize(seg[:original_len]),
            'start_s': onset / sr,
            'duration_s': original_len / sr,
            'onset_sample': onset,
        })

    return segments


def detect_pitch_yin(audio, sr=SAMPLE_RATE, min_hz=40, max_hz=2000):
    """YIN pitch detection with multi-strategy robustness.

    Uses multiple analysis windows and strategies to handle:
    - Filtered synths (where fundamental may be attenuated)
    - Plucks with fast attack (use attack portion where spectrum is brightest)
    - Octave errors (verify with autocorrelation)

    Returns (pitch_hz, confidence) or (None, 0.0).
    """
    x = audio.astype(np.float64)

    # Find active portion
    frame_sz = sr // 100
    n_frames = len(x) // frame_sz
    if n_frames < 2:
        return None, 0.0

    rms = np.array([np.sqrt(np.mean(x[i * frame_sz:(i + 1) * frame_sz] ** 2))
                     for i in range(n_frames)])
    rms_thresh = np.max(rms) * 0.1
    active_frames = np.where(rms > rms_thresh)[0]
    if len(active_frames) < 2:
        return None, 0.0

    candidates = []

    # Strategy 1: Analyze around peak energy (attack phase for plucks)
    peak_frame = np.argmax(rms)
    min_win = max(int(5 * sr / min_hz), sr // 10)  # at least 100ms or 5 periods
    center = peak_frame * frame_sz
    start = max(0, center - min_win // 8)
    end = min(len(x), start + min_win)
    seg = x[start:end] - np.mean(x[start:end])
    if len(seg) >= sr // 20:
        p, c = _yin_core(seg, sr, min_hz, max_hz, 0.15)
        if p:
            candidates.append(('peak', p, c))

    # Strategy 2: Longer window from start of activity (more periods = more accurate)
    act_start = active_frames[0] * frame_sz
    act_end = min(len(x), act_start + sr // 2)  # up to 0.5s
    seg2 = x[act_start:act_end] - np.mean(x[act_start:act_end])
    if len(seg2) >= sr // 10:
        p, c = _yin_core(seg2, sr, min_hz, max_hz, 0.12)
        if p:
            candidates.append(('long', p, c))

    # Strategy 3: Post-attack (skip first 50ms to avoid transients)
    skip = sr // 20
    post_start = min(center + skip, len(x) - sr // 10)
    if post_start > 0 and post_start + sr // 10 < len(x):
        seg3 = x[post_start:post_start + sr // 5]
        seg3 = seg3 - seg3.mean()
        if len(seg3) >= sr // 20:
            p, c = _yin_core(seg3, sr, min_hz, max_hz, 0.15)
            if p:
                candidates.append(('post', p, c))

    if not candidates:
        return None, 0.0

    # Get best YIN candidate
    candidates.sort(key=lambda c: c[2], reverse=True)
    best_pitch, best_conf = candidates[0][1], candidates[0][2]

    # Spectral verification: check if an octave above has significant energy
    # This catches the common case where YIN finds a subharmonic
    best_pitch = _verify_octave_spectral(x, sr, best_pitch, active_frames, frame_sz)

    return best_pitch, best_conf


def _verify_octave_spectral(x, sr, yin_pitch, active_frames, frame_sz):
    """Verify YIN pitch against spectral peaks. Fix octave errors.

    If FFT shows a clear peak at 2*yin_pitch that is stronger than
    the peak at yin_pitch, the true pitch is likely the octave above.
    """
    if yin_pitch is None or yin_pitch <= 0:
        return yin_pitch

    # Use the active portion for FFT
    start = active_frames[0] * frame_sz
    end = min(len(x), (active_frames[-1] + 1) * frame_sz)
    seg = x[start:end]
    if len(seg) < 2048:
        return yin_pitch

    # Use a reasonable FFT size
    n_fft = min(len(seg), 8192)
    spec = np.abs(np.fft.rfft(seg[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    freq_res = freqs[1] if len(freqs) > 1 else 1.0

    def peak_energy(f0):
        """Get peak energy around a frequency."""
        bin_f = int(round(f0 / freq_res))
        win = max(2, int(f0 * 0.05 / freq_res))  # ±5%
        lo = max(0, bin_f - win)
        hi = min(len(spec), bin_f + win + 1)
        if lo >= hi:
            return 0.0
        return float(np.max(spec[lo:hi]))

    # Compare energy at yin_pitch vs octave above
    e_base = peak_energy(yin_pitch)
    e_octave = peak_energy(yin_pitch * 2)

    # If the octave above has more energy, it's likely the true fundamental
    # (common with filtered saws where YIN finds the subharmonic)
    if e_octave > e_base * 0.5 and yin_pitch * 2 < sr / 2:
        # Also verify: does yin_pitch * 2 have harmonics?
        e_h2 = peak_energy(yin_pitch * 4)  # 2nd harmonic of octave
        e_h3 = peak_energy(yin_pitch * 6)  # 3rd harmonic of octave
        harmonic_support = (e_h2 > e_base * 0.1) or (e_h3 > e_base * 0.1)
        if e_octave > e_base * 1.5 or (e_octave > e_base * 0.8 and harmonic_support):
            return yin_pitch * 2

    return yin_pitch


def _yin_core(x, sr, min_hz, max_hz, threshold=0.15):
    """Core YIN algorithm."""
    if len(x) < 100 or np.abs(x).max() < 1e-8:
        return None, 0.0

    min_lag = max(2, sr // max_hz)
    max_lag = min(len(x) // 2, sr // min_hz)
    if min_lag >= max_lag:
        return None, 0.0

    W = max_lag

    # Difference function
    d = np.zeros(max_lag + 1)
    for tau in range(1, max_lag + 1):
        diff = x[:W] - x[tau:tau + W]
        d[tau] = np.sum(diff * diff)

    # Cumulative mean normalized difference
    d_prime = np.ones(max_lag + 1)
    running_sum = 0.0
    for tau in range(1, max_lag + 1):
        running_sum += d[tau]
        if running_sum > 0:
            d_prime[tau] = d[tau] * tau / running_sum

    # Threshold search
    best_tau = None
    for tau in range(min_lag, max_lag + 1):
        if d_prime[tau] < threshold:
            while tau + 1 <= max_lag and d_prime[tau + 1] < d_prime[tau]:
                tau += 1
            best_tau = tau
            break

    if best_tau is None:
        search = d_prime[min_lag:max_lag + 1]
        if len(search) == 0:
            return None, 0.0
        min_idx = np.argmin(search)
        if search[min_idx] > 0.5:
            return None, 0.0
        best_tau = min_idx + min_lag

    # Parabolic interpolation
    tau = best_tau
    if 1 <= tau < max_lag:
        s0 = d_prime[tau - 1]
        s1 = d_prime[tau]
        s2 = d_prime[tau + 1]
        denom = 2 * s1 - s2 - s0
        if abs(denom) > 1e-10:
            tau = tau + (s0 - s2) / (2 * denom)

    return sr / tau, 1.0 - d_prime[best_tau]


def estimate_noise_floor(audio, sr=SAMPLE_RATE):
    """Estimate background noise floor level (RMS of quietest frames).

    Returns noise_floor_rms.
    """
    frame_sz = sr // 100  # 10ms
    n_frames = len(audio) // frame_sz
    if n_frames < 5:
        return 0.0

    rms = np.array([
        np.sqrt(np.mean(audio[i * frame_sz:(i + 1) * frame_sz] ** 2))
        for i in range(n_frames)
    ])

    # Noise floor is the 10th percentile of frame RMS
    return float(np.percentile(rms, 10))


def load_and_segment(path, sr=SAMPLE_RATE, max_notes=16, auto_pitch=True):
    """Full pipeline: load file → segment notes → detect pitch.

    Returns list of dicts:
        {
            'audio': np.array (padded to 2s for optimizer),
            'original_audio': np.array (original length, normalized),
            'pitch': float or None,
            'pitch_confidence': float,
            'start_s': float,
            'duration_s': float,
        }
    """
    audio, sr = load_audio(path, sr)
    audio = normalize(audio)

    # Trim silence
    audio, trim_start, trim_end = trim_silence(audio, sr)
    if len(audio) < int(0.05 * sr):
        return []

    # Detect onsets
    onsets = detect_onsets(audio, sr)

    # Segment
    segments = segment_notes(audio, onsets, sr)

    if not segments:
        # Fallback: treat whole audio as one segment
        target_samples = int(DEFAULT_DURATION * sr)
        if len(audio) < target_samples:
            padded = np.zeros(target_samples, dtype=np.float32)
            padded[:len(audio)] = audio
        else:
            padded = audio[:target_samples].copy()
        segments = [{
            'audio': normalize(padded),
            'original_audio': normalize(audio),
            'start_s': 0.0,
            'duration_s': len(audio) / sr,
            'onset_sample': 0,
        }]

    # Limit number of notes
    segments = segments[:max_notes]

    # Detect pitch for each segment
    if auto_pitch:
        for seg in segments:
            pitch, conf = detect_pitch_yin(seg['audio'], sr)
            seg['pitch'] = pitch
            seg['pitch_confidence'] = conf

    return segments


def prepare_single(audio_or_path, sr=SAMPLE_RATE, pitch=None):
    """Quick helper: prepare a single audio for optimization.

    Accepts either a file path or a numpy array.
    Returns (audio_2s, detected_pitch).
    """
    if isinstance(audio_or_path, (str, Path)):
        audio, sr = load_audio(audio_or_path, sr)
    else:
        audio = np.asarray(audio_or_path, dtype=np.float32)

    audio = normalize(audio)
    audio, _, _ = trim_silence(audio, sr)

    # Pad/trim to 2s
    target = int(DEFAULT_DURATION * sr)
    if len(audio) < target:
        padded = np.zeros(target, dtype=np.float32)
        padded[:len(audio)] = audio
        audio = padded
    elif len(audio) > target:
        audio = audio[:target].copy()
        fade_len = min(int(0.05 * sr), target // 4)
        if fade_len > 0:
            audio[-fade_len:] *= np.linspace(1, 0, fade_len, dtype=np.float32)

    audio = normalize(audio)

    if pitch is None:
        pitch, _ = detect_pitch_yin(audio, sr)

    return audio, pitch
