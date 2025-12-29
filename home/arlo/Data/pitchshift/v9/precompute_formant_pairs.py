#!/usr/bin/env python3
"""
v9 Preprocessing: Create PAIRED formant-shifted data.

Optimizations:
- Analyze once per file, warp for multiple shifts
- 12 CPU workers for pyworld
- Batch GPU encoding
- Disk cache (faster than pickling large arrays)
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/Data/dø')

# NOTE: torch imports moved to main() to avoid CUDA fork issues


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def process_audio_cpu(args: Tuple) -> Optional[Dict]:
    """CPU worker: Load, analyze once, warp multiple times, save to cache."""
    entry, shifts, cache_dir, max_duration = args

    try:
        import pyworld as pw
        import soundfile as sf
        from scipy.interpolate import interp1d

        audio_path = fix_path(entry.get('audio_path', ''))
        if not audio_path or not os.path.exists(audio_path):
            return None

        file_id = hashlib.md5(audio_path.encode()).hexdigest()[:12]

        # Load audio
        audio, sr = sf.read(audio_path)

        # Check duration
        duration = len(audio) / sr
        if duration > max_duration:
            print(f"Skipping long file: {file_id} ({duration:.0f}s > {max_duration}s)", flush=True)
            return None

        # Mix to mono for pyworld
        if len(audio.shape) > 1:
            audio_mono = audio.mean(axis=1)
        else:
            audio_mono = audio

        # Check if has content (very low threshold)
        rms = np.sqrt(np.mean(audio_mono**2))
        if rms < 0.0001:
            print(f"Skipping silent: {audio_path} (RMS={rms:.6f})", flush=True)
            return None

        print(f"Processing: {file_id} (RMS={rms:.4f})", flush=True)

        # Save original (stereo for DCAE)
        original_path = os.path.join(cache_dir, f"{file_id}_original.wav")
        if len(audio.shape) > 1:
            sf.write(original_path, audio, sr)
        else:
            sf.write(original_path, np.stack([audio, audio], axis=1), sr)

        # === WORLD analysis (ONCE) ===
        x = audio_mono.astype(np.float64)
        f0, t = pw.dio(x, sr, f0_floor=50, f0_ceil=1200)
        f0 = pw.stonemask(x, f0, t, sr)
        sp = pw.cheaptrick(x, f0, t, sr)
        ap = pw.d4c(x, f0, t, sr)

        n_freq = sp.shape[1]
        old_freqs = np.arange(n_freq)

        result = {
            'file_id': file_id,
            'audio_path': audio_path,
            'original_cache': original_path,
            'shifts': {},
            'group': entry.get('group', 'brass'),
            'sub_group': entry.get('sub_group', 'trumpet'),
        }

        # === Warp and synthesize for each shift (fast) ===
        for semitones in shifts:
            ratio = 2.0 ** (semitones / 12.0)
            new_freqs = old_freqs / ratio

            sp_warped = np.zeros_like(sp)
            ap_warped = np.zeros_like(ap)

            for i in range(sp.shape[0]):
                interp_sp = interp1d(old_freqs, sp[i], kind='linear',
                                    bounds_error=False, fill_value=(sp[i, 0], sp[i, -1]))
                sp_warped[i] = interp_sp(new_freqs)

                interp_ap = interp1d(old_freqs, ap[i], kind='linear',
                                    bounds_error=False, fill_value=(ap[i, 0], ap[i, -1]))
                ap_warped[i] = interp_ap(new_freqs)

            y = pw.synthesize(f0, sp_warped, ap_warped, sr)

            # Save shifted (stereo)
            shift_path = os.path.join(cache_dir, f"{file_id}_shift{semitones:+d}.wav")
            sf.write(shift_path, np.stack([y, y], axis=1).astype(np.float32), sr)
            result['shifts'][semitones] = shift_path

        return result

    except Exception as e:
        import traceback
        print(f"Worker error on {entry.get('audio_path', 'unknown')}: {e}")
        traceback.print_exc()
        return None


def main():
    # Force spawn to avoid fork issues with CUDA
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='/home/arlo/Data/mute_translator/mute_manifest_deduped.json')
    parser.add_argument('--output_dir', default='/mnt/msdd2/pitchshift_v9_formant_pairs')
    parser.add_argument('--cache_dir', default='/mnt/msdd2/pitchshift_v9_cache')
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--shifts', default='-12,+12')
    parser.add_argument('--max_entries', type=int, default=None)
    parser.add_argument('--max_duration', type=float, default=60.0,
                        help='Skip files longer than this (seconds)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    shifts = [int(s) for s in args.shifts.split(',')]
    print(f"Shifts: {shifts}")

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    entries = [e for e in manifest
               if e.get('sub_group') == 'trumpet' and e.get('is_muted') == False]
    print(f"Found {len(entries)} trumpet entries")

    if args.max_entries:
        entries = entries[:args.max_entries]

    # Skip existing
    def needs_processing(entry):
        file_id = hashlib.md5(fix_path(entry.get('audio_path', '')).encode()).hexdigest()[:12]
        for shift in shifts:
            if not (output_dir / f"{file_id}_shift{shift:+d}.pt").exists():
                return True
        return False

    before = len(entries)
    entries = [e for e in entries if needs_processing(e)]
    if before - len(entries) > 0:
        print(f"Skipping {before - len(entries)} existing, {len(entries)} remaining")

    if not entries:
        print("Nothing to process!")
        return

    # Phase 1: CPU pyworld
    print(f"\n=== Phase 1: Pyworld ({args.num_workers} workers, max_dur={args.max_duration}s) ===")
    cpu_args = [(e, shifts, str(cache_dir), args.max_duration) for e in entries]

    processed = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_audio_cpu, arg): i for i, arg in enumerate(cpu_args)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Pyworld"):
            result = future.result()
            if result is not None:
                processed.append(result)

    print(f"Processed: {len(processed)}")

    # Phase 2: GPU DCAE encoding
    print(f"\n=== Phase 2: DCAE encoding ===")

    # Import torch AFTER multiprocessing is done
    import torch
    import torchaudio
    from do.pipeline_do import DoTrainComponents
    components = DoTrainComponents(checkpoint_dir="/home/arlo/Data/ACE-Step/checkpoints", device_id=0)
    components.load_dcae()
    dcae = components.music_dcae.eval()

    @torch.no_grad()
    def encode(path):
        audio, sr = torchaudio.load(path)
        if sr != 44100:
            audio = torchaudio.functional.resample(audio, sr, 44100)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        result = dcae.encode(audio.unsqueeze(0).cuda())
        return result[0][0].cpu() if isinstance(result, tuple) else result[0].cpu()

    all_pairs = []
    for item in tqdm(processed, desc="Encoding"):
        try:
            natural = encode(item['original_cache'])

            for shift, shift_path in item['shifts'].items():
                shifted = encode(shift_path)

                min_t = min(natural.shape[-1], shifted.shape[-1])
                pair_id = f"{item['file_id']}_shift{shift:+d}"

                torch.save({
                    'natural': natural[..., :min_t],
                    'shifted': shifted[..., :min_t],
                    'shift': shift,
                    'direction': 1 if shift > 0 else 0,
                    'source_audio': item['audio_path'],
                }, output_dir / f"{pair_id}.pt")

                all_pairs.append({
                    'pair_id': pair_id,
                    'pair_path': str(output_dir / f"{pair_id}.pt"),
                    'shift': shift,
                    'direction': 1 if shift > 0 else 0,
                })

            # Cleanup cache
            os.unlink(item['original_cache'])
            for p in item['shifts'].values():
                os.unlink(p)
        except Exception as e:
            print(f"Error: {e}")

    # Save manifest
    with open(output_dir / "manifest.json", 'w') as f:
        json.dump({'pairs': all_pairs, 'total': len(all_pairs)}, f)

    print(f"\n=== DONE: {len(all_pairs)} pairs ===")


if __name__ == "__main__":
    main()
