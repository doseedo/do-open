#!/usr/bin/env python3
"""
SMS Extraction with Sinusoidal Tracking.

Improvements over basic SMS:
1. Sinusoidal tracking - connects peaks across frames into continuous tracks
2. Track birth/death handling - smooth onset/offset
3. Frequency interpolation - no more wobble
4. Phase continuity - synthesizes clean audio

Based on Serra & Smith's Spectral Modeling Synthesis approach.
"""

import os
import sys
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional, List
import orjson
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_SINES = 64


@dataclass
class SineTrack:
    """A single sinusoidal track across time."""
    freqs: List[float]      # Frequency per frame
    amps: List[float]       # Amplitude per frame
    phases: List[float]     # Phase per frame
    start_frame: int        # When track started
    last_frame: int         # Last frame with a match
    active: bool = True     # Is track still being extended?


def parabolic_interp(mag: np.ndarray, peak_idx: int) -> Tuple[float, float]:
    """
    Parabolic interpolation for sub-bin frequency precision.
    Returns (bin_offset, amplitude).
    """
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
    """
    Find spectral peaks with parabolic interpolation.
    Returns list of (freq_hz, amplitude, phase).
    """
    # Work in valid range
    mag_valid = mag[min_bin:max_bin]

    # Find local maxima
    peaks = []
    for i in range(1, len(mag_valid) - 1):
        if mag_valid[i] > mag_valid[i-1] and mag_valid[i] > mag_valid[i+1]:
            if mag_valid[i] > min_amp:
                peaks.append(i + min_bin)

    if not peaks:
        return []

    # Sort by magnitude
    peaks = sorted(peaks, key=lambda p: mag[p], reverse=True)[:max_peaks]

    # Interpolate each peak
    results = []
    bin_width = freq_bins[1] - freq_bins[0]

    for pk in peaks:
        offset, amp = parabolic_interp(mag, pk)
        freq = freq_bins[pk] + offset * bin_width
        ph = phase[pk]  # Could interpolate phase too
        results.append((freq, amp, ph))

    return results


def track_sinusoids(frames_data: List[List[Tuple[float, float, float]]],
                    max_tracks: int = N_SINES,
                    freq_tolerance_hz: float = 50.0,
                    max_gap_frames: int = 3) -> List[SineTrack]:
    """
    Connect peaks across frames into continuous tracks.

    Args:
        frames_data: List of frames, each frame is list of (freq, amp, phase)
        max_tracks: Maximum simultaneous tracks
        freq_tolerance_hz: Max freq deviation to match a peak to a track
        max_gap_frames: How many frames a track can survive without a match

    Returns:
        List of SineTrack objects
    """
    all_tracks = []
    active_tracks = []

    for frame_idx, peaks in enumerate(frames_data):
        if not peaks:
            # No peaks this frame - age all tracks
            for track in active_tracks:
                track.freqs.append(track.freqs[-1] if track.freqs else 0)
                track.amps.append(0)  # Fade out
                track.phases.append(track.phases[-1] if track.phases else 0)
            continue

        # Sort peaks by amplitude for priority matching
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)

        # Match peaks to existing tracks
        matched_peaks = set()
        matched_tracks = set()

        for peak_idx, (freq, amp, phase) in enumerate(peaks):
            best_track = None
            best_dist = freq_tolerance_hz

            for track_idx, track in enumerate(active_tracks):
                if track_idx in matched_tracks:
                    continue
                if not track.active:
                    continue

                # Predict frequency (simple: use last freq)
                pred_freq = track.freqs[-1] if track.freqs else freq
                dist = abs(freq - pred_freq)

                if dist < best_dist:
                    best_dist = dist
                    best_track = track_idx

            if best_track is not None:
                # Extend track
                track = active_tracks[best_track]
                track.freqs.append(freq)
                track.amps.append(amp)
                track.phases.append(phase)
                track.last_frame = frame_idx
                matched_peaks.add(peak_idx)
                matched_tracks.add(best_track)

        # Handle unmatched tracks (gap or death)
        for track_idx, track in enumerate(active_tracks):
            if track_idx not in matched_tracks and track.active:
                gap = frame_idx - track.last_frame
                if gap > max_gap_frames:
                    # Track dies
                    track.active = False
                else:
                    # Interpolate through gap
                    track.freqs.append(track.freqs[-1])
                    track.amps.append(track.amps[-1] * 0.5)  # Fade
                    track.phases.append(track.phases[-1])

        # Start new tracks for unmatched peaks
        for peak_idx, (freq, amp, phase) in enumerate(peaks):
            if peak_idx in matched_peaks:
                continue

            # Count active tracks
            n_active = sum(1 for t in active_tracks if t.active)
            if n_active >= max_tracks:
                continue

            # Create new track
            new_track = SineTrack(
                freqs=[freq],
                amps=[amp],
                phases=[phase],
                start_frame=frame_idx,
                last_frame=frame_idx,
                active=True,
            )
            active_tracks.append(new_track)

        # Move dead tracks to finished list
        still_active = []
        for track in active_tracks:
            if track.active:
                still_active.append(track)
            else:
                if len(track.freqs) > 3:  # Only keep tracks with some duration
                    all_tracks.append(track)
        active_tracks = still_active

    # Finalize remaining active tracks
    all_tracks.extend([t for t in active_tracks if len(t.freqs) > 3])

    return all_tracks


def tracks_to_matrix(tracks: List[SineTrack], n_frames: int, n_sines: int = N_SINES,
                     global_max_amp: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert tracks to fixed-size matrices [n_frames, n_sines].

    Assigns tracks to sine slots, handling overlap.
    """
    freqs = np.zeros((n_frames, n_sines), dtype=np.float32)
    amps = np.zeros((n_frames, n_sines), dtype=np.float32)
    phases = np.zeros((n_frames, n_sines), dtype=np.float32)

    # Sort tracks by total energy (loudest first)
    tracks = sorted(tracks, key=lambda t: sum(t.amps), reverse=True)

    # Assign each track to a slot
    slot_end_frame = [-1] * n_sines  # When each slot becomes free

    for track in tracks:
        start = track.start_frame
        end = start + len(track.freqs)

        # Find a free slot
        best_slot = None
        for slot in range(n_sines):
            if slot_end_frame[slot] < start:
                best_slot = slot
                break

        if best_slot is None:
            continue  # No free slot, skip track

        # Fill slot
        for i, (f, a, p) in enumerate(zip(track.freqs, track.amps, track.phases)):
            frame = start + i
            if frame < n_frames:
                freqs[frame, best_slot] = f
                amps[frame, best_slot] = a / global_max_amp
                phases[frame, best_slot] = p

        slot_end_frame[best_slot] = end

    return freqs, amps, phases


def smooth_tracks(freqs: np.ndarray, amps: np.ndarray,
                  freq_smooth: int = 3, amp_smooth: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Apply temporal smoothing to reduce artifacts."""
    from scipy.ndimage import uniform_filter1d

    freqs_smooth = np.zeros_like(freqs)
    amps_smooth = np.zeros_like(amps)

    for i in range(freqs.shape[1]):
        # Only smooth non-zero regions
        mask = freqs[:, i] > 0
        if mask.sum() > freq_smooth:
            # Smooth frequency
            freqs_smooth[mask, i] = uniform_filter1d(freqs[mask, i], freq_smooth)
            # Smooth amplitude
            amps_smooth[mask, i] = uniform_filter1d(amps[mask, i], amp_smooth)
        else:
            freqs_smooth[:, i] = freqs[:, i]
            amps_smooth[:, i] = amps[:, i]

    return freqs_smooth, amps_smooth


def extract_sms_tracked(audio: np.ndarray, sr: int = SAMPLE_RATE,
                        n_sines: int = N_SINES,
                        min_freq: float = 50.0,
                        max_freq: float = 16000.0,
                        relative_thresh_db: float = -50.0,
                        freq_tolerance_hz: float = 40.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract sinusoidal parameters with proper tracking.

    Returns:
        freqs: [n_frames, n_sines] - tracked frequencies in Hz
        amps: [n_frames, n_sines] - amplitudes (0-1)
        phases: [n_frames, n_sines] - phases in radians
    """
    window = np.hanning(N_FFT)
    audio_padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode='reflect')
    n_frames = 1 + (len(audio_padded) - N_FFT) // HOP_LENGTH
    freq_bins = np.fft.rfftfreq(N_FFT, 1/sr)

    min_bin = max(1, int(min_freq * N_FFT / sr))
    max_bin = min(len(freq_bins) - 1, int(max_freq * N_FFT / sr))

    # First pass: find global max
    global_max = 0.0
    for t in range(n_frames):
        start = t * HOP_LENGTH
        frame = audio_padded[start:start + N_FFT] * window
        mag = np.abs(np.fft.rfft(frame))
        global_max = max(global_max, mag[min_bin:max_bin].max())

    if global_max < 1e-10:
        return (np.zeros((n_frames, n_sines), dtype=np.float32),
                np.zeros((n_frames, n_sines), dtype=np.float32),
                np.zeros((n_frames, n_sines), dtype=np.float32))

    min_amp = 10 ** (relative_thresh_db / 20) * global_max

    # Second pass: extract peaks per frame
    frames_data = []
    for t in range(n_frames):
        start = t * HOP_LENGTH
        frame = audio_padded[start:start + N_FFT] * window
        spectrum = np.fft.rfft(frame)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        peaks = find_peaks_with_interp(mag, phase, freq_bins, min_bin, max_bin, min_amp)
        frames_data.append(peaks)

    # Track sinusoids across frames
    tracks = track_sinusoids(frames_data, max_tracks=n_sines * 2,  # Allow more, we'll prune
                             freq_tolerance_hz=freq_tolerance_hz)

    # Convert to matrices
    freqs, amps, phases = tracks_to_matrix(tracks, n_frames, n_sines, global_max)

    # Smooth to reduce artifacts
    freqs, amps = smooth_tracks(freqs, amps)

    return freqs, amps, phases


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


def process_one(args) -> Optional[dict]:
    idx, item, output_dir = args
    try:
        audio, sr = torchaudio.load(item['audio_path'])
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        audio = audio.mean(dim=0).numpy() if audio.shape[0] > 1 else audio.squeeze(0).numpy()

        lat_data = torch.load(item['latent_path'], weights_only=True, map_location='cpu')
        latent = lat_data.get('latents', lat_data.get('latent'))
        if latent is None:
            return None
        dcae_frames = latent.shape[-1]

        # Extract with tracking
        freqs, amps, phases = extract_sms_tracked(audio, sr=SAMPLE_RATE)

        # Resample to match DCAE frames
        freqs = resample_frames(freqs, dcae_frames)
        amps = resample_frames(amps, dcae_frames)
        phases = resample_frames(phases, dcae_frames)

        out_path = output_dir / f"sms_{idx:06d}.pt"
        torch.save({
            'freqs': torch.from_numpy(freqs).float(),
            'amps': torch.from_numpy(amps).float(),
            'phases': torch.from_numpy(phases).float(),
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
    parser.add_argument('--output_dir', default='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_tracked')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest...")
    with open(args.manifest, 'rb') as f:
        data = orjson.loads(f.read())

    entries = data.get('entries', data)
    if isinstance(entries, dict):
        entries = list(entries.values())

    items = [e for e in entries if e.get('has_latent') and e.get('latent_path') and e.get('audio_path')]
    if args.max_samples:
        items = items[:args.max_samples]

    print(f"Processing {len(items)} samples with sinusoidal tracking...")
    print(f"  Using {args.workers} threads")

    results = []
    args_list = [(i, item, output_dir) for i, item in enumerate(items)]
    total = len(items)

    import time
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
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
            bar_len = 30
            filled = done * bar_len // total
            bar = '█' * filled + '░' * (bar_len - filled)

            print(f"\r  [{bar}] {done}/{total} ({pct}%) | {len(results)} ok | {rate:.1f}/s | ETA {eta:.0f}s  ", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n  Done: {len(results)}/{total} in {elapsed:.1f}s ({len(results)/elapsed:.1f}/s)")

    manifest_path = output_dir / 'sms_manifest.json'
    with open(manifest_path, 'wb') as f:
        f.write(orjson.dumps({'n_samples': len(results), 'n_sines': N_SINES, 'entries': results}))
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
