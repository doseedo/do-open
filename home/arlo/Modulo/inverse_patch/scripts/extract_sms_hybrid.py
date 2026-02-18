#!/usr/bin/env python3
"""
Hybrid SMS Extraction: Sines + Noise Bands

Extracts:
1. Sinusoidal tracks (deterministic - freq, amp, phase matter)
2. Noise band amplitudes (stochastic - only amp per band matters)

The residual (what's left after removing sines) is analyzed into
mel-spaced noise bands. This captures breath, consonants, transients.
"""

import os
import sys
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional, List
import orjson
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from scipy.signal import butter, sosfilt
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_SINES = 128  # Increased from 64 to capture more harmonics
N_NOISE_BANDS = 8  # Mel-spaced bands


@dataclass
class SineTrack:
    """A single sinusoidal track across time."""
    freqs: List[float]
    amps: List[float]
    phases: List[float]
    start_frame: int
    last_frame: int
    active: bool = True


def get_noise_band_edges(n_bands: int = N_NOISE_BANDS,
                         min_freq: float = 100.0,
                         max_freq: float = 16000.0) -> np.ndarray:
    """Get mel-spaced frequency band edges."""
    # Mel scale
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    mel_min = hz_to_mel(min_freq)
    mel_max = hz_to_mel(max_freq)
    mel_edges = np.linspace(mel_min, mel_max, n_bands + 1)
    hz_edges = mel_to_hz(mel_edges)
    return hz_edges


def parabolic_interp(mag: np.ndarray, peak_idx: int) -> Tuple[float, float]:
    """Parabolic interpolation for sub-bin frequency precision."""
    if peak_idx <= 0 or peak_idx >= len(mag) - 1:
        return 0.0, mag[peak_idx]

    alpha = mag[peak_idx - 1]
    beta = mag[peak_idx]
    gamma = mag[peak_idx + 1]

    denom = alpha - 2*beta + gamma
    if abs(denom) < 1e-10:
        return 0.0, beta

    p = 0.5 * (alpha - gamma) / denom
    interp_amp = beta - 0.25 * (alpha - gamma) * p

    return float(np.clip(p, -0.5, 0.5)), float(interp_amp)


def find_peaks_with_interp(mag: np.ndarray, phase: np.ndarray, freq_bins: np.ndarray,
                           min_bin: int, max_bin: int, min_amp: float,
                           max_peaks: int = 100) -> List[Tuple[float, float, float]]:
    """Find spectral peaks with parabolic interpolation (vectorized)."""
    mag_valid = mag[min_bin:max_bin]

    # Vectorized peak finding: local maxima above threshold
    is_peak = (mag_valid[1:-1] > mag_valid[:-2]) & (mag_valid[1:-1] > mag_valid[2:]) & (mag_valid[1:-1] > min_amp)
    peak_indices = np.where(is_peak)[0] + 1 + min_bin  # +1 for offset, +min_bin for global index

    if len(peak_indices) == 0:
        return []

    # Sort by amplitude and take top peaks
    peak_amps = mag[peak_indices]
    top_k = min(max_peaks, len(peak_indices))
    top_indices = np.argpartition(peak_amps, -top_k)[-top_k:]
    peaks = peak_indices[top_indices[np.argsort(peak_amps[top_indices])[::-1]]]

    # Vectorized parabolic interpolation
    alpha = mag[peaks - 1]
    beta = mag[peaks]
    gamma = mag[peaks + 1]
    denom = alpha - 2*beta + gamma
    valid = np.abs(denom) >= 1e-10

    offsets = np.zeros(len(peaks))
    interp_amps = beta.copy()
    offsets[valid] = np.clip(0.5 * (alpha[valid] - gamma[valid]) / denom[valid], -0.5, 0.5)
    interp_amps[valid] = beta[valid] - 0.25 * (alpha[valid] - gamma[valid]) * offsets[valid]

    bin_width = freq_bins[1] - freq_bins[0]
    freqs = freq_bins[peaks] + offsets * bin_width

    return list(zip(freqs.tolist(), interp_amps.tolist(), phase[peaks].tolist()))


def track_sinusoids(frames_data: List[List[Tuple[float, float, float]]],
                    max_tracks: int = N_SINES,
                    freq_tolerance_hz: float = 150.0,  # Wide for bass/vibrato/pitch shifts
                    max_gap_frames: int = 15) -> List[SineTrack]:  # Long gaps ok
    """Connect peaks across frames into continuous tracks (optimized)."""
    all_tracks = []
    # Use numpy arrays for active track state (much faster than list of objects)
    # Columns: freq, amp, phase, start_frame, last_frame, active
    max_active = max_tracks * 2
    track_freqs = np.zeros(max_active, dtype=np.float32)
    track_amps = np.zeros(max_active, dtype=np.float32)
    track_phases = np.zeros(max_active, dtype=np.float32)
    track_last_frame = np.zeros(max_active, dtype=np.int32)
    track_start_frame = np.zeros(max_active, dtype=np.int32)
    track_active = np.zeros(max_active, dtype=bool)
    track_histories = [None] * max_active  # Store full histories
    n_tracks = 0

    for frame_idx, peaks in enumerate(frames_data):
        active_mask = track_active[:n_tracks]
        active_indices = np.where(active_mask)[0]

        if not peaks:
            # Extend all active tracks with zeros
            for ti in active_indices:
                track_histories[ti][0].append(track_freqs[ti])
                track_histories[ti][1].append(0.0)
                track_histories[ti][2].append(track_phases[ti])
            continue

        # Sort peaks by amplitude (descending)
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
        peak_freqs = np.array([p[0] for p in peaks], dtype=np.float32)
        peak_amps = np.array([p[1] for p in peaks], dtype=np.float32)
        peak_phases = np.array([p[2] for p in peaks], dtype=np.float32)

        matched_peaks = np.zeros(len(peaks), dtype=bool)
        matched_tracks = np.zeros(n_tracks, dtype=bool)

        if len(active_indices) > 0:
            # Vectorized distance computation: [n_peaks] vs [n_active]
            active_freqs = track_freqs[active_indices]
            # Distance matrix: [n_peaks, n_active]
            dist_matrix = np.abs(peak_freqs[:, None] - active_freqs[None, :])

            # Greedy matching: for each peak (in amp order), find best unmatched track
            for peak_idx in range(len(peaks)):
                if matched_peaks[peak_idx]:
                    continue
                # Get distances to unmatched active tracks
                dists = dist_matrix[peak_idx].copy()
                dists[matched_tracks[active_indices]] = np.inf
                best_local = np.argmin(dists)
                if dists[best_local] < freq_tolerance_hz:
                    ti = active_indices[best_local]
                    # Update track
                    track_freqs[ti] = peak_freqs[peak_idx]
                    track_amps[ti] = peak_amps[peak_idx]
                    track_phases[ti] = peak_phases[peak_idx]
                    track_last_frame[ti] = frame_idx
                    track_histories[ti][0].append(float(peak_freqs[peak_idx]))
                    track_histories[ti][1].append(float(peak_amps[peak_idx]))
                    track_histories[ti][2].append(float(peak_phases[peak_idx]))
                    matched_peaks[peak_idx] = True
                    matched_tracks[ti] = True

        # Handle unmatched active tracks
        for ti in active_indices:
            if not matched_tracks[ti]:
                gap = frame_idx - track_last_frame[ti]
                if gap > max_gap_frames:
                    track_active[ti] = False
                    # Save completed track
                    hist = track_histories[ti]
                    if len(hist[0]) > 1:  # Reduced from 3 - keep more short tracks
                        all_tracks.append(SineTrack(
                            freqs=hist[0], amps=hist[1], phases=hist[2],
                            start_frame=int(track_start_frame[ti]),
                            last_frame=int(track_last_frame[ti]), active=False
                        ))
                else:
                    # Extend with decay
                    track_histories[ti][0].append(float(track_freqs[ti]))
                    track_histories[ti][1].append(float(track_amps[ti]) * 0.5)
                    track_histories[ti][2].append(float(track_phases[ti]))

        # Create new tracks for unmatched peaks
        n_active = track_active[:n_tracks].sum()
        for peak_idx in range(len(peaks)):
            if matched_peaks[peak_idx]:
                continue
            if n_active >= max_tracks:
                break
            if n_tracks >= max_active:
                break
            ti = n_tracks
            track_freqs[ti] = peak_freqs[peak_idx]
            track_amps[ti] = peak_amps[peak_idx]
            track_phases[ti] = peak_phases[peak_idx]
            track_start_frame[ti] = frame_idx
            track_last_frame[ti] = frame_idx
            track_active[ti] = True
            track_histories[ti] = [
                [float(peak_freqs[peak_idx])],
                [float(peak_amps[peak_idx])],
                [float(peak_phases[peak_idx])]
            ]
            n_tracks += 1
            n_active += 1

    # Collect remaining active tracks
    for ti in range(n_tracks):
        if track_active[ti]:
            hist = track_histories[ti]
            if len(hist[0]) > 1:  # Reduced from 3 - keep more short tracks
                all_tracks.append(SineTrack(
                    freqs=hist[0], amps=hist[1], phases=hist[2],
                    start_frame=int(track_start_frame[ti]),
                    last_frame=int(track_last_frame[ti]), active=False
                ))

    return all_tracks


def tracks_to_matrix(tracks: List[SineTrack], n_frames: int, n_sines: int,
                     global_max_amp: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert tracks to fixed-size matrices.

    Fixed: Use smarter slot assignment - find slot that ends EARLIEST before track start,
    or if no slot is free, find slot with LOWEST total amplitude to replace.
    """
    freqs = np.zeros((n_frames, n_sines), dtype=np.float32)
    amps = np.zeros((n_frames, n_sines), dtype=np.float32)
    phases = np.zeros((n_frames, n_sines), dtype=np.float32)

    # Sort tracks by total amplitude (loudest first)
    tracks = sorted(tracks, key=lambda t: sum(t.amps), reverse=True)
    slot_end_frame = np.full(n_sines, -1, dtype=np.int32)
    slot_total_amp = np.zeros(n_sines, dtype=np.float32)  # Track cumulative amplitude per slot

    for track in tracks:
        start = track.start_frame
        end = start + len(track.freqs)
        track_amp = sum(track.amps)

        # Find best available slot
        best_slot = None

        # First try: find any slot that's free (ended before this track starts)
        free_slots = np.where(slot_end_frame < start)[0]
        if len(free_slots) > 0:
            # Pick the slot with lowest total amplitude (least important content)
            best_slot = free_slots[slot_total_amp[free_slots].argmin()]

        # If no free slot, find slot we can overwrite (if this track is louder)
        if best_slot is None:
            # Find slot with lowest amplitude that this track could replace
            weakest_slot = slot_total_amp.argmin()
            if track_amp > slot_total_amp[weakest_slot] * 0.5:  # Only replace if significantly louder
                best_slot = weakest_slot
                # Clear the slot for reuse
                slot_end_frame[best_slot] = -1
                slot_total_amp[best_slot] = 0

        if best_slot is None:
            continue  # Track is weaker than all existing content

        for i, (f, a, p) in enumerate(zip(track.freqs, track.amps, track.phases)):
            frame = start + i
            if frame < n_frames:
                freqs[frame, best_slot] = f
                amps[frame, best_slot] = a / global_max_amp
                phases[frame, best_slot] = p

        slot_end_frame[best_slot] = end
        slot_total_amp[best_slot] += track_amp

    return freqs, amps, phases


def synthesize_sines_for_subtraction(freqs: np.ndarray, amps: np.ndarray,
                                      phases: np.ndarray, n_samples: int,
                                      sr: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesize sines for residual computation."""
    n_frames = freqs.shape[0]
    hop = n_samples // n_frames

    audio = np.zeros(n_samples, dtype=np.float32)

    for i in range(freqs.shape[1]):
        if amps[:, i].max() < 0.001:
            continue

        # Interpolate to audio rate
        t_frames = np.linspace(0, n_samples, n_frames)
        t_audio = np.arange(n_samples)

        freq_interp = np.interp(t_audio, t_frames, freqs[:, i])
        amp_interp = np.interp(t_audio, t_frames, amps[:, i])

        # Continuous phase
        phase = np.cumsum(2 * np.pi * freq_interp / sr)
        audio += amp_interp * np.sin(phase)

    return audio


def extract_noise_bands_from_residual(residual: np.ndarray, sr: int, n_frames: int,
                                       band_edges: np.ndarray) -> np.ndarray:
    """OLD: Extract noise from time-domain residual (has phase artifacts)."""
    n_bands = len(band_edges) - 1
    noise_amps = np.zeros((n_frames, n_bands), dtype=np.float32)

    hop = len(residual) // n_frames
    window = np.hanning(N_FFT)
    freq_bins = np.fft.rfftfreq(N_FFT, 1/sr)
    band_bins = np.searchsorted(freq_bins, band_edges)

    for t in range(n_frames):
        start = t * hop
        end = min(start + N_FFT, len(residual))
        if end - start < N_FFT // 2:
            continue

        frame = np.zeros(N_FFT)
        frame[:end-start] = residual[start:end]
        frame[:len(window)] *= window[:len(frame)]
        spectrum = np.abs(np.fft.rfft(frame))

        for b in range(n_bands):
            lo, hi = band_bins[b], band_bins[b + 1]
            if hi > lo:
                noise_amps[t, b] = np.sqrt(np.mean(spectrum[lo:hi] ** 2))

    max_amp = noise_amps.max()
    if max_amp > 1e-10:
        noise_amps /= max_amp
    return noise_amps


def extract_noise_bands_spectral(audio: np.ndarray, sr: int, n_frames: int,
                                  sine_freqs: np.ndarray, sine_amps: np.ndarray,
                                  band_edges: np.ndarray,
                                  sine_bandwidth_hz: float = 100.0) -> np.ndarray:
    """
    Extract noise by masking out sine frequencies from spectrum (vectorized).
    """
    n_bands = len(band_edges) - 1
    noise_amps = np.zeros((n_frames, n_bands), dtype=np.float32)

    audio_padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode='reflect')
    window = np.hanning(N_FFT)
    freq_bins = np.fft.rfftfreq(N_FFT, 1/sr)
    band_bins = np.searchsorted(freq_bins, band_edges)
    n_bins = len(freq_bins)

    bin_width = freq_bins[1] - freq_bins[0]
    sine_mask_bins = int(sine_bandwidth_hz / bin_width)

    # Pre-compute all spectra at once using stride tricks for speed
    # Process in chunks to avoid memory issues
    chunk_size = 256

    for chunk_start in range(0, n_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_frames)
        chunk_frames = chunk_end - chunk_start

        # Build frames matrix
        frames_matrix = np.zeros((chunk_frames, N_FFT), dtype=np.float32)
        for i, t in enumerate(range(chunk_start, chunk_end)):
            start = t * HOP_LENGTH
            frames_matrix[i] = audio_padded[start:start + N_FFT] * window

        # Batch FFT
        spectra = np.abs(np.fft.rfft(frames_matrix, axis=1))

        # Build masks for this chunk
        masks = np.ones((chunk_frames, n_bins), dtype=np.float32)
        for i, t in enumerate(range(chunk_start, chunk_end)):
            for j in range(sine_freqs.shape[1]):
                if sine_amps[t, j] > 0.01:
                    center_bin = int(sine_freqs[t, j] / bin_width)
                    lo = max(0, center_bin - sine_mask_bins)
                    hi = min(n_bins, center_bin + sine_mask_bins + 1)
                    masks[i, lo:hi] = 0.0

        # Apply masks
        masked_spectra = spectra * masks

        # Compute band energies
        for b in range(n_bands):
            lo, hi = band_bins[b], band_bins[b + 1]
            if hi > lo:
                noise_amps[chunk_start:chunk_end, b] = np.sqrt(
                    np.mean(masked_spectra[:, lo:hi] ** 2, axis=1)
                )

    # Normalize
    max_amp = noise_amps.max()
    if max_amp > 1e-10:
        noise_amps /= max_amp

    return noise_amps


def smooth_arrays(freqs: np.ndarray, amps: np.ndarray, noise_amps: np.ndarray,
                  freq_smooth: int = 3, amp_smooth: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply temporal smoothing."""
    from scipy.ndimage import uniform_filter1d

    freqs_smooth = np.zeros_like(freqs)
    amps_smooth = np.zeros_like(amps)

    for i in range(freqs.shape[1]):
        mask = freqs[:, i] > 0
        if mask.sum() > freq_smooth:
            freqs_smooth[mask, i] = uniform_filter1d(freqs[mask, i], freq_smooth)
            amps_smooth[mask, i] = uniform_filter1d(amps[mask, i], amp_smooth)
        else:
            freqs_smooth[:, i] = freqs[:, i]
            amps_smooth[:, i] = amps[:, i]

    # Smooth noise bands
    noise_smooth = uniform_filter1d(noise_amps, amp_smooth, axis=0)

    return freqs_smooth, amps_smooth, noise_smooth


def extract_sms_hybrid(audio: np.ndarray, sr: int = SAMPLE_RATE,
                       n_sines: int = N_SINES,
                       n_noise_bands: int = N_NOISE_BANDS,
                       min_freq: float = 50.0,
                       max_freq: float = 16000.0,
                       relative_thresh_db: float = -50.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract sinusoidal + noise band parameters.

    Returns:
        freqs: [n_frames, n_sines] - tracked frequencies in Hz
        amps: [n_frames, n_sines] - sine amplitudes (0-1)
        phases: [n_frames, n_sines] - phases in radians
        noise_amps: [n_frames, n_noise_bands] - noise band amplitudes (0-1)
    """
    window = np.hanning(N_FFT).astype(np.float32)
    audio_padded = np.pad(audio.astype(np.float32), (N_FFT // 2, N_FFT // 2), mode='reflect')
    n_frames = 1 + (len(audio_padded) - N_FFT) // HOP_LENGTH
    freq_bins = np.fft.rfftfreq(N_FFT, 1/sr)

    min_bin = max(1, int(min_freq * N_FFT / sr))
    max_bin = min(len(freq_bins) - 1, int(max_freq * N_FFT / sr))

    # Build all frames at once using stride tricks (vectorized)
    from numpy.lib.stride_tricks import as_strided
    shape = (n_frames, N_FFT)
    strides = (audio_padded.strides[0] * HOP_LENGTH, audio_padded.strides[0])
    frames_matrix = as_strided(audio_padded, shape=shape, strides=strides).copy()
    frames_matrix *= window  # Apply window to all frames

    # Single batched FFT for all frames
    spectra = np.fft.rfft(frames_matrix, axis=1)
    mags = np.abs(spectra)
    phases_all = np.angle(spectra)

    # Find global max from all magnitudes at once
    global_max = mags[:, min_bin:max_bin].max()

    if global_max < 1e-10:
        return (np.zeros((n_frames, n_sines), dtype=np.float32),
                np.zeros((n_frames, n_sines), dtype=np.float32),
                np.zeros((n_frames, n_sines), dtype=np.float32),
                np.zeros((n_frames, n_noise_bands), dtype=np.float32))

    min_amp = 10 ** (relative_thresh_db / 20) * global_max

    # Extract peaks per frame (uses pre-computed spectra)
    frames_data = []
    for t in range(n_frames):
        peaks = find_peaks_with_interp(mags[t], phases_all[t], freq_bins, min_bin, max_bin, min_amp)
        frames_data.append(peaks)

    # Track sinusoids
    tracks = track_sinusoids(frames_data, max_tracks=n_sines * 2)

    # Convert to matrices
    freqs, amps, phases = tracks_to_matrix(tracks, n_frames, n_sines, global_max)

    # Extract noise bands using spectral masking (not time-domain subtraction)
    # This avoids phase artifacts and preserves high-frequency noise
    band_edges = get_noise_band_edges(n_noise_bands, min_freq, max_freq)
    noise_amps = extract_noise_bands_spectral(
        audio, sr, n_frames, freqs, amps, band_edges,
        sine_bandwidth_hz=80.0  # Mask 80Hz around each sine
    )

    # Smooth everything
    freqs, amps, noise_amps = smooth_arrays(freqs, amps, noise_amps)

    return freqs, amps, phases, noise_amps


def resample_frames(data: np.ndarray, target_frames: int) -> np.ndarray:
    src_frames = data.shape[0]
    if src_frames == target_frames:
        return data
    src_times = np.linspace(0, 1, src_frames)
    tgt_times = np.linspace(0, 1, target_frames)
    result = np.zeros((target_frames, data.shape[1]), dtype=data.dtype)
    for i in range(data.shape[1]):
        result[:, i] = np.interp(tgt_times, src_times, data[:, i])
    return result


def estimate_rms_from_params(amps: np.ndarray, noise_amps: np.ndarray) -> float:
    """
    Fast RMS estimate from extracted parameters without full synthesis.

    Empirically calibrated: actual synthesis RMS ≈ 0.5 * theoretical estimate
    due to phase cancellation and interpolation effects.
    """
    # Sine RMS: sum of squared amplitudes / 2 (power of sine = A^2/2)
    sine_power_per_frame = (amps ** 2).sum(axis=1) / 2  # [n_frames]
    sine_rms = np.sqrt(sine_power_per_frame.mean())

    # Noise RMS
    noise_power_per_frame = (noise_amps ** 2).sum(axis=1)
    noise_rms = np.sqrt(noise_power_per_frame.mean())

    # Combined RMS (assuming uncorrelated)
    combined_rms = np.sqrt(sine_rms ** 2 + noise_rms ** 2)

    # Empirical correction factor (synthesis produces ~0.5x theoretical)
    CORRECTION_FACTOR = 0.5
    return combined_rms * CORRECTION_FACTOR


def calibrate_amplitudes(freqs: np.ndarray, amps: np.ndarray, noise_amps: np.ndarray,
                         original_audio: np.ndarray, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrate extracted amplitudes so that synthesis matches original RMS.
    Uses fast RMS estimation instead of full synthesis.
    Returns calibrated (amps, noise_amps).
    """
    orig_rms = np.sqrt(np.mean(original_audio ** 2))

    if orig_rms < 1e-10:
        return amps, noise_amps

    # Fast RMS estimate from params
    estimated_rms = estimate_rms_from_params(amps, noise_amps)

    if estimated_rms < 1e-10:
        return amps, noise_amps

    # Scale factor to match original RMS
    scale = orig_rms / estimated_rms

    # Apply scale to both sine and noise amplitudes
    amps_calibrated = amps * scale
    noise_amps_calibrated = noise_amps * scale

    return amps_calibrated.astype(np.float32), noise_amps_calibrated.astype(np.float32)


def get_conditioning_path(audio_path: str) -> Optional[str]:
    """Convert audio path to conditioning amp path."""
    # /home/arlo/gcs-bucket/protools/... -> /home/arlo/gcs-bucket/Conditioning/protools/...
    # /home/arlo/gcs-bucket/protoolsA/... -> /home/arlo/gcs-bucket/Conditioning/protoolsA/...
    base = '/home/arlo/gcs-bucket/'
    if audio_path.startswith(base + 'protools/'):
        rel = audio_path[len(base):]  # protools/...
        name = Path(audio_path).stem
        parent = str(Path(audio_path).parent)
        return f"{base}Conditioning/{rel[:-4]}.amp.npy".replace('.wav.amp.npy', '.amp.npy')
    elif audio_path.startswith(base + 'protoolsA/'):
        rel = audio_path[len(base):]
        return f"{base}Conditioning/{rel[:-4]}.amp.npy".replace('.wav.amp.npy', '.amp.npy')
    return None


def process_one(args) -> Optional[dict]:
    idx, item, output_dir, min_active_sines, min_amp_thresh = args
    try:
        # Fast filter using conditioning amp if available
        if item.get('has_conditioning'):
            cond_path = get_conditioning_path(item['audio_path'])
            if cond_path and Path(cond_path).exists():
                amp_cond = np.load(cond_path)
                if amp_cond.max() < min_amp_thresh:
                    return None  # Silent sample, skip fast

        audio, sr = torchaudio.load(item['audio_path'])
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        audio = audio.mean(dim=0).numpy() if audio.shape[0] > 1 else audio.squeeze(0).numpy()

        # Skip near-silent audio (fallback if no conditioning)
        if np.abs(audio).max() < 0.01:
            return None

        lat_data = torch.load(item['latent_path'], weights_only=True, map_location='cpu')
        latent = lat_data.get('latents', lat_data.get('latent'))
        if latent is None:
            return None
        dcae_frames = latent.shape[-1]

        # Extract hybrid (sines + noise)
        freqs, amps, phases, noise_amps = extract_sms_hybrid(audio, sr=SAMPLE_RATE)

        # Calibrate amplitudes so synthesis matches original RMS
        amps, noise_amps = calibrate_amplitudes(freqs, amps, noise_amps, audio, SAMPLE_RATE)

        # Filter: skip samples with insufficient activity
        active_per_frame = (amps > 0.05).sum(axis=1).mean()
        if active_per_frame < min_active_sines:
            return None

        # Resample to match DCAE frames
        freqs = resample_frames(freqs, dcae_frames)
        amps = resample_frames(amps, dcae_frames)
        phases = resample_frames(phases, dcae_frames)
        noise_amps = resample_frames(noise_amps, dcae_frames)

        out_path = output_dir / f"sms_{idx:06d}.pt"
        torch.save({
            'freqs': torch.from_numpy(freqs).float(),
            'amps': torch.from_numpy(amps).float(),
            'phases': torch.from_numpy(phases).float(),
            'noise_amps': torch.from_numpy(noise_amps).float(),
            'latent_path': item['latent_path'],
            'audio_path': item['audio_path'],
        }, str(out_path))

        return {'idx': idx, 'path': str(out_path), 'latent_path': item['latent_path']}
    except Exception as e:
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='/home/arlo/gcs-bucket/Manifests/unified_manifest.json')
    parser.add_argument('--output_dir', default='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_hybrid')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--min_active_sines', type=float, default=0.0,
                        help='Min avg active sines per frame to keep sample (0 = no filter)')
    parser.add_argument('--min_amp', type=float, default=0.1,
                        help='Min conditioning amp to keep sample (filters silent via amp.npy)')
    parser.add_argument('--conditioning_only', action='store_true',
                        help='Only process samples with has_conditioning=true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("SMS Hybrid Extraction: Sines + Noise Bands")
    print(f"  {N_SINES} sine tracks")
    print(f"  {N_NOISE_BANDS} mel-spaced noise bands")
    print(f"  Min active sines: {args.min_active_sines}")
    print(f"  Min amp (fast filter): {args.min_amp}")
    print(f"  Conditioning only: {args.conditioning_only}")
    print(f"  Threshold: -50 dB")
    print()

    print(f"Loading manifest...")
    with open(args.manifest, 'rb') as f:
        data = orjson.loads(f.read())

    entries = data.get('entries', data)
    if isinstance(entries, dict):
        entries = list(entries.values())

    # Filter entries
    items = [e for e in entries if e.get('has_latent') and e.get('latent_path') and e.get('audio_path')]
    if args.conditioning_only:
        items = [e for e in items if e.get('has_conditioning')]
        print(f"  Filtered to {len(items)} samples with conditioning")

    if args.max_samples:
        items = items[:args.max_samples]

    print(f"Processing {len(items)} samples...")

    results = []
    args_list = [(i, item, output_dir, args.min_active_sines, args.min_amp) for i, item in enumerate(items)]
    total = len(items)

    import time
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, a): a[0] for a in args_list}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
            done += 1

            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            pct = done * 100 // total
            bar = '█' * (done * 30 // total) + '░' * (30 - done * 30 // total)

            print(f"\r  [{bar}] {done}/{total} ({pct}%) | {len(results)} ok | {rate:.1f}/s | ETA {eta:.0f}s  ", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n  Done: {len(results)}/{total} in {elapsed:.1f}s")

    manifest_path = output_dir / 'sms_manifest.json'
    with open(manifest_path, 'wb') as f:
        f.write(orjson.dumps({
            'n_samples': len(results),
            'n_sines': N_SINES,
            'n_noise_bands': N_NOISE_BANDS,
            'entries': results
        }))
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
