#!/usr/bin/env python3
"""
Extract SMS (Spectral Modeling Synthesis) parameters from audio.

This creates ground truth sine parameters for training the SAMI mapper.
Run once as preprocessing, then training is fast (just MSE on params).

Output: For each audio file, saves (freqs, amps, phases) aligned to DCAE frames.
"""

import os
import sys
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional
import orjson
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter1d

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512  # ~86 fps, close to DCAE frame rate
N_SINES = 64  # Max sines to extract per frame


def parabolic_interpolation(mag: np.ndarray, phase: np.ndarray, peak_idx: int,
                            freq_bins: np.ndarray) -> Tuple[float, float, float]:
    """
    Parabolic interpolation around a peak for sub-bin frequency precision.

    Returns: (frequency_hz, amplitude, phase_radians)
    """
    if peak_idx <= 0 or peak_idx >= len(mag) - 1:
        return freq_bins[peak_idx], mag[peak_idx], phase[peak_idx]

    # Parabolic interpolation on log magnitude
    alpha = np.log(mag[peak_idx - 1] + 1e-10)
    beta = np.log(mag[peak_idx] + 1e-10)
    gamma = np.log(mag[peak_idx + 1] + 1e-10)

    # Peak offset from bin center
    p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma + 1e-10)
    p = np.clip(p, -0.5, 0.5)

    # Interpolated values
    freq_idx = peak_idx + p
    freq = freq_bins[0] + freq_idx * (freq_bins[1] - freq_bins[0])

    # Amplitude at interpolated position
    amp = mag[peak_idx] - 0.25 * (alpha - gamma) * p

    # Phase interpolation (linear)
    ph = phase[peak_idx] + p * (phase[peak_idx + 1] - phase[peak_idx - 1]) / 2

    return float(freq), float(amp), float(ph)


def extract_sms_params(audio: np.ndarray, sr: int = SAMPLE_RATE,
                       n_sines: int = N_SINES, n_fft: int = N_FFT,
                       hop_length: int = HOP_LENGTH,
                       min_amp_db: float = -60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract sinusoidal parameters using SMS-style peak picking.

    Args:
        audio: Mono audio signal
        sr: Sample rate
        n_sines: Max number of sines per frame
        n_fft: FFT size
        hop_length: Hop size for STFT
        min_amp_db: Minimum amplitude threshold in dB

    Returns:
        freqs: [n_frames, n_sines] frequencies in Hz (0 = inactive)
        amps: [n_frames, n_sines] amplitudes (linear, normalized)
        phases: [n_frames, n_sines] phases in radians
    """
    # Compute STFT
    window = np.hanning(n_fft)

    # Pad audio
    audio_padded = np.pad(audio, (n_fft // 2, n_fft // 2), mode='reflect')

    n_frames = 1 + (len(audio_padded) - n_fft) // hop_length

    # Frequency bins
    freq_bins = np.fft.rfftfreq(n_fft, 1/sr)

    # Minimum amplitude threshold
    min_amp = 10 ** (min_amp_db / 20)

    # Output arrays
    all_freqs = np.zeros((n_frames, n_sines), dtype=np.float32)
    all_amps = np.zeros((n_frames, n_sines), dtype=np.float32)
    all_phases = np.zeros((n_frames, n_sines), dtype=np.float32)

    for t in range(n_frames):
        start = t * hop_length
        frame = audio_padded[start:start + n_fft] * window

        # FFT
        spectrum = np.fft.rfft(frame)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Normalize magnitude
        mag_max = mag.max()
        if mag_max < 1e-10:
            continue
        mag_norm = mag / mag_max

        # Find peaks (local maxima above threshold)
        # Use scipy's find_peaks with prominence
        peaks, properties = find_peaks(
            mag_norm,
            height=min_amp / mag_max,
            distance=3,  # Minimum 3 bins apart (~65 Hz at 44100/2048)
            prominence=0.01,  # Require some prominence
        )

        if len(peaks) == 0:
            continue

        # Sort by amplitude (descending)
        peak_amps = mag[peaks]
        sort_idx = np.argsort(peak_amps)[::-1]
        peaks = peaks[sort_idx]

        # Extract top n_sines peaks with interpolation
        for i, peak in enumerate(peaks[:n_sines]):
            freq, amp, ph = parabolic_interpolation(mag, phase, peak, freq_bins)

            # Skip DC and very low frequencies
            if freq < 20:
                continue

            # Skip above Nyquist
            if freq > sr / 2 - 100:
                continue

            all_freqs[t, i] = freq
            all_amps[t, i] = amp / mag_max  # Normalize
            all_phases[t, i] = ph

    # Sort each frame by frequency for consistent ordering
    for t in range(n_frames):
        active = all_freqs[t] > 0
        if active.sum() > 0:
            order = np.argsort(all_freqs[t])[::-1]  # Sort by freq descending
            # Put zeros last
            nonzero = all_freqs[t] > 0
            order = np.concatenate([np.where(nonzero)[0][np.argsort(all_freqs[t, nonzero])],
                                    np.where(~nonzero)[0]])
            all_freqs[t] = all_freqs[t, order]
            all_amps[t] = all_amps[t, order]
            all_phases[t] = all_phases[t, order]

    return all_freqs, all_amps, all_phases


def resample_frames(data: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample SMS frames to match DCAE frame count."""
    src_frames = data.shape[0]

    if src_frames == target_frames:
        return data

    # Linear interpolation
    src_times = np.linspace(0, 1, src_frames)
    tgt_times = np.linspace(0, 1, target_frames)

    # Interpolate each sine track independently
    result = np.zeros((target_frames, data.shape[1]), dtype=data.dtype)
    for i in range(data.shape[1]):
        result[:, i] = np.interp(tgt_times, src_times, data[:, i])

    return result


def process_one(args) -> Optional[dict]:
    """Process a single audio file."""
    item_idx, item, target_frames, output_dir = args

    try:
        audio_path = item['audio_path']
        latent_path = item['latent_path']

        # Load audio
        audio, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

        # Convert to mono numpy
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio.squeeze(0)
        audio = audio.numpy()

        # Load latent to get frame count
        lat_data = torch.load(latent_path, weights_only=True, map_location='cpu')
        if 'latents' in lat_data:
            latent = lat_data['latents']
        elif 'latent' in lat_data:
            latent = lat_data['latent']
        else:
            return None

        dcae_frames = latent.shape[-1]

        # Extract SMS params
        freqs, amps, phases = extract_sms_params(audio)

        # Resample to match DCAE frames
        freqs = resample_frames(freqs, dcae_frames)
        amps = resample_frames(amps, dcae_frames)
        phases = resample_frames(phases, dcae_frames)

        # Save
        out_path = output_dir / f"sms_{item_idx:06d}.pt"
        torch.save({
            'freqs': torch.from_numpy(freqs).float(),
            'amps': torch.from_numpy(amps).float(),
            'phases': torch.from_numpy(phases).float(),
            'latent_path': latent_path,
            'audio_path': audio_path,
            'dcae_frames': dcae_frames,
        }, str(out_path))

        return {
            'idx': item_idx,
            'path': str(out_path),
            'latent_path': latent_path,
            'n_frames': dcae_frames,
            'mean_active': (freqs > 0).sum(axis=1).mean(),
        }

    except Exception as e:
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/gcs-bucket/Manifests/unified_manifest.json')
    parser.add_argument('--output_dir', type=str,
                        default='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_params')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print("SMS Parameter Extraction")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    print(f"\nLoading manifest from {args.manifest}...")
    with open(args.manifest, 'rb') as f:
        data = orjson.loads(f.read())

    entries = data.get('entries', data)
    if isinstance(entries, dict):
        entries = list(entries.values())

    # Filter valid entries
    items = []
    for entry in entries:
        if not entry.get('has_latent', False):
            continue
        if entry.get('latent_path') is None:
            continue
        if entry.get('audio_path') is None:
            continue
        items.append(entry)

    if args.max_samples:
        items = items[:args.max_samples]

    print(f"Processing {len(items)} samples...")

    # Process in parallel
    results = []
    args_list = [(i, item, 22, output_dir) for i, item in enumerate(items)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, a): a[0] for a in args_list}

        done = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            done += 1
            if done % 100 == 0:
                print(f"\r  Processed {done}/{len(items)}...", end="", flush=True)

    print(f"\r  Processed {len(results)}/{len(items)} successfully")

    # Save manifest
    manifest = {
        'n_samples': len(results),
        'n_sines': N_SINES,
        'sample_rate': SAMPLE_RATE,
        'entries': results,
    }

    manifest_path = output_dir / 'sms_manifest.json'
    with open(manifest_path, 'wb') as f:
        f.write(orjson.dumps(manifest))

    print(f"\nSaved manifest to {manifest_path}")

    # Stats
    if results:
        mean_active = np.mean([r['mean_active'] for r in results])
        print(f"\nStats:")
        print(f"  Mean active sines per frame: {mean_active:.1f}/{N_SINES}")


if __name__ == "__main__":
    main()
