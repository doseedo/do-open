#!/usr/bin/env python3
"""
Batch apply EQ and mix settings to all translated outputs.
Uses the same settings tested on sample_03.
"""

import numpy as np
import json
import os
import argparse
from scipy.signal import sosfilt, butter
from scipy.io import wavfile
from tqdm import tqdm

# =============================================================================
# EQ SETTINGS (same as tested on sample_03)
# =============================================================================

# EQ Settings for OLD model
eq_bands_old = [
    {"type": "highpass", "freq": 260.0, "gain": 0.0,  "Q": 0.71, "slope_dB_per_oct": 12},
    {"type": "peaking",  "freq": 139.0, "gain": +0.1, "Q": 1.00},
    {"type": "peaking",  "freq": 100.0, "gain":  0.0, "Q": 0.60},
    {"type": "peaking",  "freq": 4020.0,"gain": -4.2, "Q": 0.54},
    {"type": "peaking",  "freq": 2660.0,"gain": +3.2, "Q": 0.41},
    {"type": "peaking",  "freq": 4800.0,"gain": +1.2, "Q": 1.00},
    {"type": "peaking",  "freq": 7150.0,"gain": +4.6, "Q": 0.89},
]

# EQ Settings for NEW model
eq_bands_new = [
    {"type": "highpass", "freq": 262.0,  "gain": 0.0,  "Q": 1.10, "slope_dB_per_oct": 12},
    {"type": "peaking",  "freq": 75.0,   "gain": 0.0,  "Q": 1.00},
    {"type": "peaking",  "freq": 100.0,  "gain": 0.0,  "Q": 0.60},
    {"type": "peaking",  "freq": 7450.0, "gain": -4.8, "Q": 0.39},
    {"type": "peaking",  "freq": 8250.0, "gain": +5.5, "Q": 0.79},
    {"type": "peaking",  "freq": 8400.0, "gain": +6.8, "Q": 0.51},
    {"type": "peaking",  "freq": 240.0,  "gain": +0.4, "Q": 1.00}
]

# EQ for final mix (Pro-Q style)
eq_bands_proq = [
    {"type": "highpass", "freq": 151.0, "gain": 0.0, "Q": 0.54, "slope_dB_per_oct": 12},
    {"type": "peaking", "freq": 47.4,   "gain": -1.2,   "Q": 1.00},
    {"type": "peaking", "freq": 426.0,  "gain": -3.2,   "Q": 0.51},
    {"type": "peaking", "freq": 1176.6, "gain": -25.53, "Q": 3.773},
    {"type": "peaking", "freq": 1492.4, "gain": -22.11, "Q": 12.76},
    {"type": "peaking", "freq": 1862.5, "gain": -13.63, "Q": 4.667},
    {"type": "peaking", "freq": 1591.6, "gain": +1.66,  "Q": 1.022},
    {"type": "peaking", "freq": 3040.0, "gain": -2.7,   "Q": 0.98},
    {"type": "peaking", "freq": 3300.0, "gain": -7.0,   "Q": 0.51}
]

# New model gain reduction
NEW_MODEL_GAIN_DB = -9.5

# =============================================================================
# DSP FUNCTIONS
# =============================================================================

def db_to_lin(db):
    return 10 ** (db / 20)


def design_peaking_sos(freq, gain_db, Q, sr):
    """Design a peaking EQ filter using Audio EQ Cookbook formula."""
    if abs(gain_db) < 0.001:
        return np.array([[1, 0, 0, 1, 0, 0]])

    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1, a1/a0, a2/a0])

    return np.array([[b[0], b[1], b[2], 1, a[1], a[2]]])


def design_highpass_sos(freq, Q, sr, order=2):
    """Design a highpass filter as second-order sections."""
    nyq = sr / 2
    normalized_freq = freq / nyq
    if normalized_freq >= 1.0:
        normalized_freq = 0.99
    sos = butter(order, normalized_freq, btype='high', output='sos')
    return sos


def apply_eq_bands(audio, eq_bands, sr):
    """Apply a list of EQ bands to audio."""
    result = audio.copy().astype(np.float64)

    for band in eq_bands:
        band_type = band["type"]
        freq = band["freq"]
        Q = band.get("Q", 1.0)
        gain = band.get("gain", 0.0)

        if band_type == "highpass":
            slope = band.get("slope_dB_per_oct", 12)
            order = slope // 6
            sos = design_highpass_sos(freq, Q, sr, order=int(order))
        elif band_type == "peaking":
            sos = design_peaking_sos(freq, gain, Q, sr)
        else:
            continue

        if result.ndim == 1:
            result = sosfilt(sos, result)
        else:
            for ch in range(result.shape[0]):
                result[ch] = sosfilt(sos, result[ch])

    return result


def load_wav(path):
    """Load wav file, return (audio, sr) with audio as float [-1, 1]."""
    sr, audio = wavfile.read(path)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    if audio.ndim == 2:
        audio = audio.T

    return audio, sr


def save_wav(path, audio, sr):
    """Save audio as wav file."""
    if audio.ndim == 2:
        audio = audio.T
    audio = np.clip(audio, -1, 1)
    audio_int = (audio * 32767).astype(np.int16)
    wavfile.write(path, sr, audio_int)


def process_sample(old_path, new_path, output_path):
    """Process a single sample: EQ both, mix, apply final EQ, save."""
    try:
        # Load
        old_audio, sr = load_wav(old_path)
        new_audio, sr_new = load_wav(new_path)

        if sr != sr_new:
            return False, "Sample rate mismatch"

        # Apply EQ to old model
        old_eq = apply_eq_bands(old_audio, eq_bands_old, sr)

        # Apply EQ to new model
        new_eq = apply_eq_bands(new_audio, eq_bands_new, sr)

        # Apply gain reduction to new model
        new_eq = new_eq * db_to_lin(NEW_MODEL_GAIN_DB)

        # Mix together
        min_len = min(old_eq.shape[-1], new_eq.shape[-1])
        if old_eq.ndim == 1:
            mixed = old_eq[:min_len] + new_eq[:min_len]
        else:
            mixed = old_eq[:, :min_len] + new_eq[:, :min_len]

        # Apply final Pro-Q EQ
        final = apply_eq_bands(mixed, eq_bands_proq, sr)

        # Normalize to prevent clipping
        peak = np.max(np.abs(final))
        if peak > 0.99:
            final = final * 0.95 / peak

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_wav(output_path, final, sr)

        return True, None

    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Batch apply EQ and mix to all translations')
    parser.add_argument('--input_json', default='/mnt/msdd2/mute_translator_outputs/translation_results.json',
                        help='Input JSON with translation paths')
    parser.add_argument('--output_json', default='/mnt/msdd2/mute_translator_outputs/mixed_results.json',
                        help='Output JSON with mixed paths')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip samples that already have mixed output')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to process')
    args = parser.parse_args()

    # Load input JSON
    print(f"Loading: {args.input_json}")
    with open(args.input_json) as f:
        data = json.load(f)

    samples = data['samples']
    print(f"Found {len(samples)} samples")

    if args.limit:
        samples = samples[:args.limit]
        print(f"Limited to {args.limit} samples")

    results = []
    success_count = 0
    skip_count = 0
    error_count = 0

    for sample in tqdm(samples, desc="Mixing"):
        old_path = sample['old_model_translated']
        new_path = sample['new_model_translated']

        # Output path: same directory as inputs, named "mixed_final.wav"
        sample_dir = os.path.dirname(old_path)
        output_path = os.path.join(sample_dir, "mixed_final.wav")

        # Skip if exists
        if args.skip_existing and os.path.exists(output_path):
            skip_count += 1
            results.append({
                **sample,
                'mixed_path': output_path,
                'status': 'skipped'
            })
            continue

        # Check inputs exist
        if not os.path.exists(old_path):
            error_count += 1
            results.append({
                **sample,
                'mixed_path': None,
                'status': 'error',
                'error': f'Old model file not found: {old_path}'
            })
            continue

        if not os.path.exists(new_path):
            error_count += 1
            results.append({
                **sample,
                'mixed_path': None,
                'status': 'error',
                'error': f'New model file not found: {new_path}'
            })
            continue

        # Process
        success, error = process_sample(old_path, new_path, output_path)

        if success:
            success_count += 1
            results.append({
                **sample,
                'mixed_path': output_path,
                'status': 'success'
            })
        else:
            error_count += 1
            results.append({
                **sample,
                'mixed_path': None,
                'status': 'error',
                'error': error
            })

        # Save incremental results every 50 samples
        if len(results) % 50 == 0:
            with open(args.output_json, 'w') as f:
                json.dump({
                    'eq_settings': {
                        'eq_bands_old': eq_bands_old,
                        'eq_bands_new': eq_bands_new,
                        'eq_bands_proq': eq_bands_proq,
                        'new_model_gain_db': NEW_MODEL_GAIN_DB
                    },
                    'total_processed': len(results),
                    'success': success_count,
                    'skipped': skip_count,
                    'errors': error_count,
                    'samples': results
                }, f, indent=2)

    # Final save
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors:  {error_count}")

    with open(args.output_json, 'w') as f:
        json.dump({
            'eq_settings': {
                'eq_bands_old': eq_bands_old,
                'eq_bands_new': eq_bands_new,
                'eq_bands_proq': eq_bands_proq,
                'new_model_gain_db': NEW_MODEL_GAIN_DB
            },
            'total_processed': len(results),
            'success': success_count,
            'skipped': skip_count,
            'errors': error_count,
            'samples': results
        }, f, indent=2)

    print(f"\nResults saved to: {args.output_json}")


if __name__ == '__main__':
    main()
