#!/usr/bin/env python3
"""
Fast SMS extraction - optimized for GCS.

Uses threading for I/O parallelism and simplified peak picking.
"""

import os
import sys
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional
import orjson
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_SINES = 64


def extract_sms_simple(audio: np.ndarray, sr: int = SAMPLE_RATE,
                       n_sines: int = N_SINES,
                       min_freq: float = 50.0,
                       max_freq: float = 16000.0,
                       relative_thresh_db: float = -40.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast SMS extraction using numpy only."""
    window = np.hanning(N_FFT)
    audio_padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode='reflect')
    n_frames = 1 + (len(audio_padded) - N_FFT) // HOP_LENGTH
    freq_bins = np.fft.rfftfreq(N_FFT, 1/sr)

    # Frequency bin limits
    min_bin = int(min_freq * N_FFT / sr)
    max_bin = int(max_freq * N_FFT / sr)

    all_freqs = np.zeros((n_frames, n_sines), dtype=np.float32)
    all_amps = np.zeros((n_frames, n_sines), dtype=np.float32)
    all_phases = np.zeros((n_frames, n_sines), dtype=np.float32)

    # First pass: find global max for normalization
    global_max = 0.0
    for t in range(n_frames):
        start = t * HOP_LENGTH
        frame = audio_padded[start:start + N_FFT] * window
        spectrum = np.fft.rfft(frame)
        mag = np.abs(spectrum)
        global_max = max(global_max, mag[min_bin:max_bin].max())

    if global_max < 1e-10:
        return all_freqs, all_amps, all_phases

    # Relative threshold (e.g., -40dB below global max)
    min_amp = 10 ** (relative_thresh_db / 20) * global_max

    # Second pass: extract peaks
    for t in range(n_frames):
        start = t * HOP_LENGTH
        frame = audio_padded[start:start + N_FFT] * window
        spectrum = np.fft.rfft(frame)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Only look in valid frequency range
        mag_valid = mag[min_bin:max_bin]

        # Simple peak finding: local maxima
        peaks = np.where((mag_valid[1:-1] > mag_valid[:-2]) & (mag_valid[1:-1] > mag_valid[2:]))[0] + 1
        if len(peaks) == 0:
            continue

        # Shift back to original indices
        peaks = peaks + min_bin

        # Filter by absolute threshold
        peaks = peaks[mag[peaks] > min_amp]
        if len(peaks) == 0:
            continue

        # Sort by magnitude, take top n_sines
        peak_mags = mag[peaks]
        top_idx = np.argsort(peak_mags)[-n_sines:][::-1]
        top_peaks = peaks[top_idx]

        sine_idx = 0
        for pk in top_peaks:
            if sine_idx >= n_sines:
                break
            all_freqs[t, sine_idx] = freq_bins[pk]
            all_amps[t, sine_idx] = mag[pk] / global_max  # Global normalization
            all_phases[t, sine_idx] = phase[pk]
            sine_idx += 1

    # Sort each frame by frequency (non-zero first, then zeros)
    for t in range(n_frames):
        nonzero = all_freqs[t] > 0
        if nonzero.sum() > 0:
            # Sort non-zero by frequency, keep zeros at end
            nz_idx = np.where(nonzero)[0]
            z_idx = np.where(~nonzero)[0]
            nz_order = nz_idx[np.argsort(all_freqs[t, nz_idx])]
            order = np.concatenate([nz_order, z_idx])
            all_freqs[t] = all_freqs[t, order]
            all_amps[t] = all_amps[t, order]
            all_phases[t] = all_phases[t, order]

    return all_freqs, all_amps, all_phases


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

        freqs, amps, phases = extract_sms_simple(audio)
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
    parser.add_argument('--output_dir', default='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_params')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--workers', type=int, default=32)
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

    print(f"Processing {len(items)} samples with {args.workers} threads...")

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

            # Progress bar
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
        f.write(orjson.dumps({'n_samples': len(results), 'entries': results}))
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
