#!/usr/bin/env python3
"""
Apply EQ settings to old/new model outputs and mix them together.
"""

import numpy as np
import json
from scipy.signal import sosfilt, butter, iirpeak, bilinear_zpk
from scipy.io import wavfile
import os

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


def db_to_lin(db):
    return 10 ** (db / 20)


def design_peaking_sos(freq, gain_db, Q, sr):
    """Design a peaking EQ filter as second-order sections.
    Uses Audio EQ Cookbook formula - same formula works for both boost and cut.
    """
    if abs(gain_db) < 0.001:
        # Neutral - return passthrough
        return np.array([[1, 0, 0, 1, 0, 0]])

    # A = sqrt(10^(dBgain/20)) = 10^(dBgain/40)
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * Q)

    # Same formula for both boost AND cut (A handles the difference)
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    # Normalize
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1, a1/a0, a2/a0])

    return np.array([[b[0], b[1], b[2], 1, a[1], a[2]]])


def design_highpass_sos(freq, Q, sr, order=2):
    """Design a highpass filter as second-order sections."""
    # For 12 dB/oct, use order=2 (one biquad)
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
            order = slope // 6  # 12 dB/oct = order 2, 24 dB/oct = order 4
            sos = design_highpass_sos(freq, Q, sr, order=int(order))
        elif band_type == "peaking":
            sos = design_peaking_sos(freq, gain, Q, sr)
        else:
            continue

        # Apply to each channel
        if result.ndim == 1:
            result = sosfilt(sos, result)
        else:
            for ch in range(result.shape[0]):
                result[ch] = sosfilt(sos, result[ch])

    return result


def load_wav(path):
    """Load wav file, return (audio, sr) with audio as float [-1, 1]."""
    sr, audio = wavfile.read(path)

    # Convert to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.float32 or audio.dtype == np.float64:
        pass

    # Transpose to (channels, samples) if stereo
    if audio.ndim == 2:
        audio = audio.T

    return audio, sr


def save_wav(path, audio, sr):
    """Save audio as wav file."""
    # Transpose back to (samples, channels)
    if audio.ndim == 2:
        audio = audio.T

    # Clip and convert to int16
    audio = np.clip(audio, -1, 1)
    audio_int = (audio * 32767).astype(np.int16)

    wavfile.write(path, sr, audio_int)


def main():
    sample_dir = "/home/arlo/Data/mute_translator/comparison_3way/sample_03_Trumpets 2.22_55"

    old_path = os.path.join(sample_dir, "2_old_model_translated.wav")
    new_path = os.path.join(sample_dir, "3_new_model_translated.wav")

    print(f"Loading old model output: {old_path}")
    old_audio, sr = load_wav(old_path)
    print(f"  Shape: {old_audio.shape}, SR: {sr}")

    print(f"Loading new model output: {new_path}")
    new_audio, sr_new = load_wav(new_path)
    print(f"  Shape: {new_audio.shape}, SR: {sr_new}")

    assert sr == sr_new, "Sample rates must match"

    # Apply EQ to old model
    print("\nApplying EQ to OLD model output...")
    old_eq = apply_eq_bands(old_audio, eq_bands_old, sr)

    # Apply EQ to new model
    print("Applying EQ to NEW model output...")
    new_eq = apply_eq_bands(new_audio, eq_bands_new, sr)

    # Apply gain reduction to new model
    print(f"Applying {NEW_MODEL_GAIN_DB} dB gain to NEW model...")
    new_eq = new_eq * db_to_lin(NEW_MODEL_GAIN_DB)

    # Mix together
    print("Mixing old + new...")
    # Ensure same length
    min_len = min(old_eq.shape[-1], new_eq.shape[-1])
    if old_eq.ndim == 1:
        mixed = old_eq[:min_len] + new_eq[:min_len]
    else:
        mixed = old_eq[:, :min_len] + new_eq[:, :min_len]

    # Apply final Pro-Q EQ
    print("Applying Pro-Q EQ to mix...")
    final = apply_eq_bands(mixed, eq_bands_proq, sr)

    # Normalize to prevent clipping
    peak = np.max(np.abs(final))
    if peak > 0.99:
        print(f"  Normalizing (peak was {peak:.2f})")
        final = final * 0.95 / peak

    # Save outputs
    old_eq_path = os.path.join(sample_dir, "4_old_model_eq.wav")
    new_eq_path = os.path.join(sample_dir, "5_new_model_eq_gained.wav")
    mixed_path = os.path.join(sample_dir, "6_mixed_raw.wav")
    final_path = os.path.join(sample_dir, "7_final_mixed_eq.wav")

    print(f"\nSaving outputs...")
    save_wav(old_eq_path, old_eq, sr)
    print(f"  {old_eq_path}")

    save_wav(new_eq_path, new_eq, sr)
    print(f"  {new_eq_path}")

    # Also save mixed before Pro-Q for comparison
    mixed_norm = mixed.copy()
    peak_m = np.max(np.abs(mixed_norm))
    if peak_m > 0.99:
        mixed_norm = mixed_norm * 0.95 / peak_m
    save_wav(mixed_path, mixed_norm, sr)
    print(f"  {mixed_path}")

    save_wav(final_path, final, sr)
    print(f"  {final_path}")

    # Save EQ settings to JSON
    eq_settings = {
        "eq_bands_old": eq_bands_old,
        "eq_bands_new": eq_bands_new,
        "eq_bands_proq_final": eq_bands_proq,
        "new_model_gain_db": NEW_MODEL_GAIN_DB
    }

    settings_path = os.path.join(sample_dir, "eq_settings.json")
    with open(settings_path, 'w') as f:
        json.dump(eq_settings, f, indent=2)
    print(f"  {settings_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
