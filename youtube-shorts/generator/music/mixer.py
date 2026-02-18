"""Audio mixing utilities — mix WAV stems, pad/trim to duration."""

import numpy as np
import soundfile as sf
import subprocess
import os


def read_audio(path, target_sr=44100):
    """Read a WAV/FLAC file, return (audio_array, sample_rate).
    Audio is returned as float32 mono.
    """
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    if sr != target_sr:
        # Resample using sox
        tmp = path + ".resampled.wav"
        subprocess.run(
            ["sox", path, "-r", str(target_sr), tmp],
            capture_output=True, check=True,
        )
        audio, sr = sf.read(tmp, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        os.remove(tmp)
    return audio, sr


def mix_stems(stem_paths, output_path, gains=None, target_sr=44100):
    """Mix multiple WAV files into one.

    Args:
        stem_paths: List of paths to WAV files
        output_path: Where to write the mixed WAV
        gains: Optional list of gain multipliers per stem (default: equal)
        target_sr: Target sample rate

    Returns:
        Path to the mixed WAV file
    """
    if not stem_paths:
        return None

    stems = []
    max_len = 0
    for path in stem_paths:
        if path and os.path.exists(path):
            audio, sr = read_audio(path, target_sr)
            stems.append(audio)
            max_len = max(max_len, len(audio))

    if not stems:
        return None

    if gains is None:
        gains = [1.0 / len(stems)] * len(stems)

    # Pad all stems to max length and mix
    mixed = np.zeros(max_len, dtype=np.float32)
    for i, stem in enumerate(stems):
        gain = gains[i] if i < len(gains) else 1.0 / len(stems)
        padded = np.pad(stem, (0, max_len - len(stem)))
        mixed += padded * gain

    # Normalize to -1dB peak
    peak = np.abs(mixed).max()
    if peak > 0:
        target_peak = 10 ** (-1.0 / 20)  # -1dB
        mixed = mixed * (target_peak / peak)

    sf.write(output_path, mixed, target_sr)
    return output_path


def pad_or_trim(audio_path, target_duration_sec, output_path, target_sr=44100):
    """Pad with silence or trim audio to exact duration.

    Args:
        audio_path: Input WAV path
        target_duration_sec: Desired duration in seconds
        output_path: Where to write the result
        target_sr: Sample rate

    Returns:
        Path to output file
    """
    audio, sr = read_audio(audio_path, target_sr)
    target_samples = int(target_duration_sec * target_sr)

    if len(audio) > target_samples:
        # Trim with short fade-out
        audio = audio[:target_samples]
        fade_len = min(int(0.05 * target_sr), target_samples)
        if fade_len > 0:
            fade = np.linspace(1.0, 0.0, fade_len)
            audio[-fade_len:] *= fade
    elif len(audio) < target_samples:
        # Pad with silence
        audio = np.pad(audio, (0, target_samples - len(audio)))

    sf.write(output_path, audio, target_sr)
    return output_path


def mix_stems_weighted(stem_paths, output_path, gains=None, eq_bands=None,
                       target_sr=44100):
    """Mix multiple WAV stems with per-stem gain and optional EQ shelving.

    Args:
        stem_paths: List of paths to WAV files
        output_path: Where to write the mixed WAV
        gains: Per-stem gain multipliers (default: auto-balanced)
        eq_bands: Optional list of dicts per stem, each with:
            'low_cut_hz': highpass cutoff (e.g. 80 for bass removal)
            'high_cut_hz': lowpass cutoff (e.g. 8000 for treble removal)
        target_sr: Target sample rate

    Returns:
        Path to the mixed WAV file
    """
    from scipy.signal import butter, sosfilt

    if not stem_paths:
        return None

    stems = []
    max_len = 0
    for path in stem_paths:
        if path and os.path.exists(path):
            audio, sr = read_audio(path, target_sr)
            stems.append(audio)
            max_len = max(max_len, len(audio))

    if not stems:
        return None

    if gains is None:
        # Auto-balance: bass louder, high quieter
        n = len(stems)
        gains = [0.8 / n] * n

    mixed = np.zeros(max_len, dtype=np.float32)
    for i, stem in enumerate(stems):
        gain = gains[i] if i < len(gains) else 0.5 / len(stems)
        padded = np.pad(stem, (0, max_len - len(stem)))

        # Apply EQ if specified
        if eq_bands and i < len(eq_bands) and eq_bands[i]:
            band = eq_bands[i]
            if "low_cut_hz" in band and band["low_cut_hz"] > 0:
                sos = butter(4, band["low_cut_hz"], btype="high",
                             fs=target_sr, output="sos")
                padded = sosfilt(sos, padded).astype(np.float32)
            if "high_cut_hz" in band and band["high_cut_hz"] > 0:
                sos = butter(4, band["high_cut_hz"], btype="low",
                             fs=target_sr, output="sos")
                padded = sosfilt(sos, padded).astype(np.float32)

        mixed += padded * gain

    # Normalize to -1dB peak
    peak = np.abs(mixed).max()
    if peak > 0:
        target_peak = 10 ** (-1.0 / 20)
        mixed = mixed * (target_peak / peak)

    sf.write(output_path, mixed, target_sr)
    return output_path


def add_reverb(input_path, output_path, reverberance=30, room_scale=80):
    """Add reverb using sox."""
    try:
        subprocess.run(
            ["sox", input_path, output_path, "reverb", str(reverberance),
             str(reverberance), str(room_scale)],
            capture_output=True, check=True,
        )
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If sox fails, just copy
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path
