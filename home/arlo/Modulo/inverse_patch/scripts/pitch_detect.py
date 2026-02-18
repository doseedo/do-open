#!/usr/bin/env python3
"""Pitch detection for inverse synthesis.

Uses torchcrepe (GPU) with librosa.pyin fallback (CPU).
Returns median pitch over active frames with octave error correction.
"""

import numpy as np

SAMPLE_RATE = 44100


def detect_pitch(audio, sr=SAMPLE_RATE, use_gpu=True):
    """Detect fundamental frequency from audio.

    Strategy:
    1. Autocorrelation (fast, reliable for synth audio)
    2. If autocorr fails & use_gpu, try torchcrepe (better for natural audio)

    Args:
        audio: numpy array, mono audio
        sr: sample rate
        use_gpu: try torchcrepe as fallback

    Returns:
        (pitch_hz: float or None, confidence: float)
        pitch_hz is None if no pitch detected.
    """
    # Autocorrelation-based pitch detection (robust for synth audio)
    pitch, conf = _detect_pitch_autocorr(audio, sr)
    if pitch is not None and conf > 0.5:
        return pitch, conf

    # CREPE fallback for natural/recorded audio
    if use_gpu:
        try:
            crepe_pitch, crepe_conf = _detect_pitch_crepe(audio, sr)
            if crepe_pitch is not None and crepe_conf > 0.3:
                return crepe_pitch, crepe_conf
        except Exception:
            pass

    # Return whatever autocorrelation found (even low confidence)
    return pitch, conf


def _detect_pitch_crepe(audio, sr):
    """Pitch detection using torchcrepe (GPU-accelerated)."""
    import torch
    import torchcrepe
    import torchaudio

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Resample to 16kHz for CREPE
    crepe_sr = 16000
    audio_t = torch.from_numpy(audio.copy()).float()
    if audio_t.ndim == 1:
        audio_t = audio_t.unsqueeze(0)  # [1, N]
    if sr != crepe_sr:
        audio_t = torchaudio.functional.resample(audio_t, sr, crepe_sr)

    # torchcrepe expects [B, N] — already [1, N]
    audio_t = audio_t.to(device)

    with torch.inference_mode():
        f0, periodicity = torchcrepe.predict(
            audio_t,
            sample_rate=crepe_sr,
            hop_length=int(crepe_sr / 100),  # 10ms frames
            fmin=30,
            fmax=4000,
            pad=True,
            model="tiny",
            batch_size=256,
            device=device,
            return_periodicity=True,
        )
        f0 = f0[0].cpu().numpy()
        periodicity = periodicity[0].cpu().numpy()

    return _process_pitch_frames(f0, periodicity)


def _detect_pitch_autocorr(audio, sr, fmin=30, fmax=4000):
    """Pitch detection using normalized autocorrelation.

    Robust, pure-numpy implementation. No external dependencies.
    Works well for monophonic synthesizer audio.
    """
    # Use a segment from the sustain portion (skip attack/release)
    n_samples = len(audio)
    # Skip first 10% (attack) and last 20% (release)
    start = max(int(n_samples * 0.1), sr // 10)
    end = max(int(n_samples * 0.8), start + sr // 2)
    segment = audio[start:end].astype(np.float64)

    if len(segment) < sr // 10:
        return None, 0.0

    # Check if there's meaningful signal
    rms = np.sqrt(np.mean(segment ** 2))
    if rms < 0.01:
        return None, 0.0

    # Window the segment
    frame_size = min(len(segment), int(sr * 0.05))  # 50ms frame
    n_frames = max(1, len(segment) // frame_size - 1)

    pitch_estimates = []
    confidences = []

    lag_min = max(1, int(sr / fmax))
    lag_max = min(frame_size - 1, int(sr / fmin))

    for fi in range(n_frames):
        frame_start = fi * frame_size
        frame = segment[frame_start:frame_start + frame_size]

        if len(frame) < lag_max + 1:
            continue

        # Normalized autocorrelation using cumulative mean normalized difference
        # (YIN algorithm simplified)
        N = len(frame)
        diff = np.zeros(lag_max + 1)

        for tau in range(1, lag_max + 1):
            diff[tau] = np.sum((frame[:N - tau] - frame[tau:N]) ** 2)

        # Cumulative mean normalized difference
        if diff[1:].sum() < 1e-10:
            continue

        cmndf = np.ones(lag_max + 1)
        running_sum = 0.0
        for tau in range(1, lag_max + 1):
            running_sum += diff[tau]
            cmndf[tau] = diff[tau] * tau / (running_sum + 1e-10)

        # Find first dip below threshold in CMNDF
        threshold = 0.15
        best_tau = None
        for tau in range(lag_min, lag_max + 1):
            if cmndf[tau] < threshold:
                # Find local minimum
                if tau + 1 < len(cmndf) and cmndf[tau] <= cmndf[tau + 1]:
                    best_tau = tau
                    break

        # If no dip below threshold, use absolute minimum
        if best_tau is None:
            search = cmndf[lag_min:lag_max + 1]
            best_tau = np.argmin(search) + lag_min
            if cmndf[best_tau] > 0.5:
                continue  # Not confident enough

        # Parabolic interpolation for sub-sample accuracy
        if 1 <= best_tau < len(cmndf) - 1:
            a = cmndf[best_tau - 1]
            b = cmndf[best_tau]
            c = cmndf[best_tau + 1]
            denom = 2.0 * (2 * b - a - c)
            if abs(denom) > 1e-8:
                delta = (a - c) / denom
                best_tau_interp = best_tau + delta
            else:
                best_tau_interp = float(best_tau)
        else:
            best_tau_interp = float(best_tau)

        f0 = sr / best_tau_interp
        conf = 1.0 - cmndf[best_tau]

        if fmin <= f0 <= fmax and conf > 0.3:
            pitch_estimates.append(f0)
            confidences.append(conf)

    if len(pitch_estimates) < 2:
        return None, 0.0

    pitch_estimates = np.array(pitch_estimates)
    confidences = np.array(confidences)

    return _process_pitch_frames(pitch_estimates, confidences, conf_threshold=0.3)


def _detect_pitch_pyin(audio, sr):
    """Pitch detection using librosa.pyin (CPU fallback)."""
    import librosa
    import warnings

    # Use frame_length=4096 to support low fmin=30Hz (need >= 2 periods)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced, _ = librosa.pyin(
            audio.astype(np.float32),
            fmin=30,
            fmax=4000,
            sr=sr,
            frame_length=4096,
            hop_length=512,
        )

    if f0 is None or len(f0) == 0:
        return None, 0.0

    # Convert voiced flag to confidence
    periodicity = voiced.astype(np.float32)
    f0 = np.nan_to_num(f0, nan=0.0)

    return _process_pitch_frames(f0, periodicity)


def _process_pitch_frames(f0, periodicity, conf_threshold=0.5):
    """Process pitch frames: filter by confidence, compute median, correct octave errors.

    Args:
        f0: array of pitch estimates per frame
        periodicity: array of confidence per frame
        conf_threshold: minimum confidence to include a frame

    Returns:
        (pitch_hz: float or None, confidence: float)
    """
    # Filter by confidence
    mask = periodicity > conf_threshold
    if mask.sum() < 3:
        return None, 0.0

    voiced_f0 = f0[mask]
    voiced_conf = periodicity[mask]

    # Remove zeros
    nonzero = voiced_f0 > 10
    if nonzero.sum() < 3:
        return None, 0.0

    voiced_f0 = voiced_f0[nonzero]
    voiced_conf = voiced_conf[nonzero]

    # Compute weighted median pitch
    median_pitch = float(np.median(voiced_f0))

    # Harmonic correction: check if some frames detect a sub-multiple of median.
    # If frames cluster at BOTH f and f/2 (or f/3), the lower is the fundamental.
    # Key: only correct if there are actual frames near the sub-frequency.
    log_pitches = np.log2(voiced_f0 / median_pitch)

    # Count frames within ±0.15 octaves of various sub-harmonics
    near_median = np.sum(np.abs(log_pitches) < 0.15)

    for divisor in [2, 3]:
        sub_pitch = median_pitch / divisor
        if sub_pitch < 30:
            continue
        log_sub = np.log2(voiced_f0 / sub_pitch)
        # Count frames actually near the sub-pitch (not just harmonics of it)
        near_sub = np.sum(np.abs(log_sub) < 0.15)
        # Only correct if there are real frames at the sub-pitch
        # (not just that harmonics of the sub-pitch match existing detections)
        if near_sub >= 3 and near_sub > len(voiced_f0) * 0.1:
            median_pitch = sub_pitch
            break

    avg_conf = float(np.mean(voiced_conf))

    return median_pitch, avg_conf


def detect_pitch_multi(audio, sr=SAMPLE_RATE, max_notes=4, use_gpu=True):
    """Detect multiple pitches in a chord.

    Uses spectral peak analysis with harmonic grouping.
    Unlike detect_pitch (which uses autocorrelation), this method
    works directly on the spectrum to find multiple fundamentals.

    Args:
        audio: numpy array, mono audio
        sr: sample rate
        max_notes: maximum number of notes to detect
        use_gpu: try GPU methods first

    Returns:
        list of (pitch_hz, confidence) tuples, sorted by confidence desc.
    """
    # Use spectral peaks to find all notes
    n_fft = 8192
    # Use a windowed segment from the sustain portion
    start = min(len(audio) // 4, sr // 2)
    end = min(start + n_fft * 2, len(audio))
    segment = audio[start:end]

    if len(segment) < n_fft:
        # Fallback to monophonic detection
        pitch, conf = detect_pitch(audio, sr, use_gpu)
        return [(pitch, conf)] if pitch else []

    spec = np.abs(np.fft.rfft(segment[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    freq_res = sr / n_fft

    from scipy.signal import find_peaks
    spec_db = 20 * np.log10(spec + 1e-10)
    spec_db -= spec_db.max()
    peaks, props = find_peaks(spec_db, height=-30, distance=5, prominence=6)

    if len(peaks) < 2:
        pitch, conf = detect_pitch(audio, sr, use_gpu)
        return [(pitch, conf)] if pitch else []

    peak_freqs = freqs[peaks]
    peak_amps = spec[peaks]

    # Group peaks into harmonic series
    # For each candidate fundamental, count how many peaks are harmonics
    candidate_fundamentals = set()

    for pf in peak_freqs:
        if pf < 30 or pf > 4000:
            continue
        # This could be the 1st, 2nd, 3rd, or 4th harmonic
        for h in [1, 2, 3, 4]:
            f0_candidate = pf / h
            if 30 < f0_candidate < 4000:
                candidate_fundamentals.add(round(f0_candidate, 1))

    # Score each candidate fundamental by harmonic support
    scored = []
    for f0_cand in candidate_fundamentals:
        harmonic_count = 0
        harmonic_energy = 0.0
        for h in range(1, 8):
            fh = f0_cand * h
            if fh > sr / 2:
                break
            # Check if there's a peak near this harmonic
            bin_h = int(fh / freq_res)
            lo = max(0, bin_h - 3)
            hi = min(len(spec), bin_h + 4)
            region = spec[lo:hi]
            if len(region) > 0 and region.max() > spec.max() * 0.05:
                harmonic_count += 1
                harmonic_energy += region.max()

        if harmonic_count >= 2:
            score = harmonic_count * 0.3 + harmonic_energy / (spec.max() + 1e-8) * 0.7
            scored.append((f0_cand, score))

    if not scored:
        pitch, conf = detect_pitch(audio, sr, use_gpu)
        return [(pitch, conf)] if pitch else []

    # Sort by score desc
    scored.sort(key=lambda x: -x[1])

    # De-duplicate: merge candidates within 5% of each other (keep higher-scored)
    results = []
    for f0_cand, score in scored:
        is_dup = False
        for existing_f0, _ in results:
            if abs(f0_cand - existing_f0) / existing_f0 < 0.05:
                is_dup = True
                break
        if not is_dup:
            conf = min(score / 3.0, 1.0)
            if conf > 0.2:
                results.append((f0_cand, conf))
        if len(results) >= max_notes:
            break

    return results
