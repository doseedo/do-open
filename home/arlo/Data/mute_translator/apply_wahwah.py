#!/usr/bin/env python3
"""
Apply wah-wah effect to muted trumpet audio.

Three modes available:
1. Reference mode: Extract wah pattern from a reference recording
2. LFO mode: Oscillating wah with rate/depth/waveform control
3. Precise mode: User-defined automation graph (time, value pairs)
4. Envelope mode: Amplitude-following wah with threshold and attack/decay

Usage:
    # Reference mode (most accurate)
    python apply_wahwah.py --mode reference --input sustained.wav --reference wahwah.mp3 --output out.wav

    # LFO mode
    python apply_wahwah.py --mode lfo --input in.wav --output out.wav --rate 2.5 --depth 0.8 --waveform sine

    # Precise mode (automation file: JSON with "points": [[time, value], ...])
    python apply_wahwah.py --mode precise --input in.wav --output out.wav --automation auto.json

    # Envelope mode (amplitude follower)
    python apply_wahwah.py --mode envelope --input in.wav --output out.wav --threshold 0.1 --attack 0.01 --decay 0.1
"""

import argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
from scipy.ndimage import gaussian_filter1d
import subprocess
import json
import os


def load_audio(path, target_sr=44100):
    """Load audio file, converting to consistent format."""
    tmp_path = '/tmp/audio_load_temp.wav'
    subprocess.run(['ffmpeg', '-y', '-i', path, '-ar', str(target_sr),
                    '-ac', '1', '-acodec', 'pcm_s16le', tmp_path],
                   capture_output=True)

    sr, audio = wavfile.read(tmp_path)
    audio = audio.astype(np.float64) / 32768.0
    return audio, sr


# =============================================================================
# MODULATION GENERATORS
# =============================================================================

def generate_lfo_modulation(duration, sr, rate=2.5, depth=0.8, waveform='sine',
                            phase=0.0, offset=0.5):
    """
    Generate LFO-based modulation.

    Args:
        duration: Duration in seconds
        sr: Sample rate for output resolution
        rate: Oscillation rate in Hz
        depth: Modulation depth (0-1), how far from center it swings
        waveform: 'sine', 'triangle', 'sawtooth', 'square'
        phase: Starting phase in radians
        offset: Center value (0.5 = centered between 0 and 1)

    Returns:
        modulation: Array of values (0=closed, 1=open)
        times: Corresponding time points
    """
    num_points = int(duration * 100)  # 100 points per second
    t = np.linspace(0, duration, num_points)

    if waveform == 'sine':
        wave = np.sin(2 * np.pi * rate * t + phase)
    elif waveform == 'triangle':
        wave = 2 * np.abs(2 * ((rate * t + phase/(2*np.pi)) % 1) - 1) - 1
    elif waveform == 'sawtooth':
        wave = 2 * ((rate * t + phase/(2*np.pi)) % 1) - 1
    elif waveform == 'square':
        wave = np.sign(np.sin(2 * np.pi * rate * t + phase))
    else:
        wave = np.sin(2 * np.pi * rate * t + phase)

    modulation = offset + depth * 0.5 * wave
    modulation = np.clip(modulation, 0, 1)

    return modulation, t


def generate_precise_modulation(automation_points, duration):
    """
    Generate modulation from precise automation points.

    Args:
        automation_points: List of [time, value] pairs
            - time: seconds
            - value: 0-1 (0=closed, 1=open)
        duration: Total duration in seconds

    Returns:
        modulation: Interpolated modulation curve
        times: Time points
    """
    points = np.array(automation_points)
    times_in = points[:, 0]
    values_in = points[:, 1]

    # Output at 100 points per second
    num_points = int(duration * 100)
    times_out = np.linspace(0, duration, num_points)

    # Linear interpolation
    modulation = np.interp(times_out, times_in, values_in)
    modulation = np.clip(modulation, 0, 1)

    return modulation, times_out


def generate_envelope_modulation(audio, sr, threshold=0.1, attack=0.01, decay=0.1,
                                  sensitivity=1.0, invert=False,
                                  gate_mode=False, min_val=0.0, max_val=1.0):
    """
    Generate modulation that follows the audio amplitude envelope.

    Args:
        audio: Input audio signal
        sr: Sample rate
        threshold: Amplitude threshold (0-1) below which wah stays closed
        attack: Attack time in seconds (how fast wah opens)
        decay: Decay/release time in seconds (how fast wah closes)
        sensitivity: How much amplitude affects wah position (multiplier)
        invert: If True, louder = more closed (instead of more open)
        gate_mode: If True, use hard threshold (on/off), else smooth follow
        min_val: Minimum modulation value
        max_val: Maximum modulation value

    Returns:
        modulation: Envelope-following modulation
        times: Time points
    """
    # Compute amplitude envelope
    frame_size = int(sr * 0.01)  # 10ms frames
    hop = frame_size // 2

    num_frames = len(audio) // hop
    envelope = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop
        end = min(start + frame_size, len(audio))
        envelope[i] = np.sqrt(np.mean(audio[start:end]**2))

    # Normalize envelope
    env_max = envelope.max()
    if env_max > 0:
        envelope = envelope / env_max

    # Apply threshold
    if gate_mode:
        # Hard gate: above threshold = max, below = min
        envelope = np.where(envelope > threshold, 1.0, 0.0)
    else:
        # Soft threshold with sensitivity
        envelope = np.clip((envelope - threshold) * sensitivity / (1 - threshold + 1e-6), 0, 1)

    # Apply attack/decay smoothing
    attack_coef = 1 - np.exp(-1 / (attack * sr / hop + 1e-6))
    decay_coef = 1 - np.exp(-1 / (decay * sr / hop + 1e-6))

    smoothed = np.zeros_like(envelope)
    smoothed[0] = envelope[0]

    for i in range(1, len(envelope)):
        if envelope[i] > smoothed[i-1]:
            # Attack: rising
            smoothed[i] = smoothed[i-1] + attack_coef * (envelope[i] - smoothed[i-1])
        else:
            # Decay: falling
            smoothed[i] = smoothed[i-1] + decay_coef * (envelope[i] - smoothed[i-1])

    # Scale to min/max range
    modulation = min_val + (max_val - min_val) * smoothed

    # Invert if requested
    if invert:
        modulation = max_val - (modulation - min_val)

    modulation = np.clip(modulation, 0, 1)

    # Time points
    times = np.arange(num_frames) * hop / sr

    return modulation, times


# =============================================================================
# WAH EFFECT APPLICATION
# =============================================================================

def apply_wah_from_reference(sustained, reference, sr,
                              smoothing_sigma=2.0,
                              max_gain_db=20, min_gain_db=-40):
    """
    Apply wah effect by computing spectral transfer function from reference.
    """
    nperseg = 2048
    hop = 256
    noverlap = nperseg - hop

    f, t, Zxx_ref = stft(reference, fs=sr, nperseg=nperseg, noverlap=noverlap)
    f, _, Zxx_sus = stft(sustained, fs=sr, nperseg=nperseg, noverlap=noverlap)

    S_ref = np.abs(Zxx_ref)
    S_sus = np.abs(Zxx_sus)

    min_frames = min(S_ref.shape[1], S_sus.shape[1])

    transfer = S_ref[:, :min_frames] / (S_sus[:, :min_frames] + 1e-10)

    transfer_smooth = np.zeros_like(transfer)
    for freq_bin in range(transfer.shape[0]):
        transfer_smooth[freq_bin, :] = gaussian_filter1d(
            transfer[freq_bin, :], sigma=smoothing_sigma
        )

    max_gain = 10 ** (max_gain_db / 20)
    min_gain = 10 ** (min_gain_db / 20)
    transfer_clipped = np.clip(transfer_smooth, min_gain, max_gain)

    Zxx_applied = Zxx_sus[:, :min_frames] * transfer_clipped

    _, applied = istft(Zxx_applied, fs=sr, nperseg=nperseg, noverlap=noverlap)

    ref_rms = np.sqrt(np.mean(reference**2))
    app_rms = np.sqrt(np.mean(applied**2))
    if app_rms > 0:
        applied = applied * (ref_rms / app_rms)

    max_val = np.abs(applied).max()
    if max_val > 0.95:
        applied = applied * 0.9 / max_val

    return applied


def apply_wah_from_modulation(audio, sr, modulation, mod_times,
                               closed_boost_freq=800, closed_boost_db=12,
                               closed_cut_freq=2500, closed_cut_db=-18,
                               open_boost_freq=3000, open_boost_db=6,
                               resonance_q=2.5):
    """
    Apply wah effect using a modulation curve.

    Args:
        audio: Input audio
        sr: Sample rate
        modulation: Time-varying modulation (0=closed, 1=open)
        mod_times: Time points for modulation values
        closed_boost_freq: Frequency to boost when closed
        closed_boost_db: dB boost at peak when fully closed
        closed_cut_freq: Frequency above which to cut when closed
        closed_cut_db: dB cut above cutoff when fully closed
        open_boost_freq: Frequency to boost when open
        open_boost_db: dB boost when fully open
        resonance_q: Q factor for resonant peak
    """
    nperseg = 2048
    hop = 256
    noverlap = nperseg - hop

    f, t, Zxx = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)

    # Interpolate modulation to STFT time frames
    mod_interp = np.interp(t, mod_times, modulation)

    Zxx_mod = Zxx.copy()

    for i, mod_val in enumerate(mod_interp):
        # mod_val: 0 = fully closed, 0.5 = neutral, 1 = fully open
        # closed_amount: 1 at mod=0, 0 at mod>=0.5
        # open_amount: 0 at mod<=0.5, 1 at mod=1

        closed_amount = np.clip(1.0 - 2.0 * mod_val, 0, 1)
        open_amount = np.clip(2.0 * mod_val - 1.0, 0, 1)

        gain = np.ones_like(f)

        if closed_amount > 0:
            # CLOSED WAH: boost low-mids, cut highs
            boost_bw = closed_boost_freq / resonance_q
            boost_curve = np.exp(-0.5 * ((f - closed_boost_freq) / boost_bw)**2)
            boost_gain = 10 ** (closed_boost_db * closed_amount * boost_curve / 20)

            cut_curve = 1.0 / (1.0 + np.exp((f - closed_cut_freq) / 300))
            cut_gain = 10 ** (closed_cut_db * closed_amount * (1 - cut_curve) / 20)

            gain = gain * boost_gain * cut_gain

        if open_amount > 0:
            # OPEN WAH: boost highs
            open_curve = 1.0 / (1.0 + np.exp(-(f - open_boost_freq) / 500))
            open_gain = 10 ** (open_boost_db * open_amount * open_curve / 20)

            gain = gain * open_gain

        Zxx_mod[:, i] = Zxx[:, i] * gain

    _, audio_out = istft(Zxx_mod, fs=sr, nperseg=nperseg, noverlap=noverlap)

    min_len = min(len(audio), len(audio_out))
    audio_out = audio_out[:min_len]

    max_val = np.abs(audio_out).max()
    if max_val > 0.95:
        audio_out = audio_out * 0.9 / max_val

    return audio_out


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Apply wah-wah effect with multiple modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reference mode
  python apply_wahwah.py --mode reference --input dry.wav --reference wah.mp3 --output out.wav

  # LFO mode
  python apply_wahwah.py --mode lfo --input dry.wav --output out.wav --rate 2.5 --depth 0.8

  # Precise mode (automation JSON: {"points": [[0, 0.5], [1.0, 0], [2.0, 1], ...]})
  python apply_wahwah.py --mode precise --input dry.wav --output out.wav --automation curve.json

  # Envelope mode
  python apply_wahwah.py --mode envelope --input dry.wav --output out.wav --threshold 0.1 --attack 0.02 --decay 0.2
        """
    )

    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Input audio file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output wav file')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['reference', 'lfo', 'precise', 'envelope'],
                        help='Modulation mode')

    # Reference mode
    parser.add_argument('--reference', type=str, default=None,
                        help='Reference wah-wah file (reference mode)')
    parser.add_argument('--smoothing', type=float, default=2.0,
                        help='Transfer function smoothing (reference mode)')

    # LFO mode
    parser.add_argument('--rate', type=float, default=2.5,
                        help='LFO rate in Hz (lfo mode)')
    parser.add_argument('--depth', type=float, default=0.8,
                        help='LFO depth 0-1 (lfo mode)')
    parser.add_argument('--waveform', type=str, default='sine',
                        choices=['sine', 'triangle', 'sawtooth', 'square'],
                        help='LFO waveform (lfo mode)')
    parser.add_argument('--phase', type=float, default=0.0,
                        help='LFO starting phase in radians (lfo mode)')
    parser.add_argument('--offset', type=float, default=0.5,
                        help='LFO center offset 0-1 (lfo mode)')

    # Precise mode
    parser.add_argument('--automation', type=str, default=None,
                        help='Automation JSON file with "points": [[time, value], ...] (precise mode)')
    parser.add_argument('--automation_points', type=str, default=None,
                        help='Inline automation as JSON string (precise mode)')

    # Envelope mode
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Amplitude threshold 0-1 (envelope mode)')
    parser.add_argument('--attack', type=float, default=0.01,
                        help='Attack time in seconds (envelope mode)')
    parser.add_argument('--decay', type=float, default=0.1,
                        help='Decay/release time in seconds (envelope mode)')
    parser.add_argument('--sensitivity', type=float, default=1.0,
                        help='Envelope sensitivity multiplier (envelope mode)')
    parser.add_argument('--invert', action='store_true',
                        help='Invert envelope: louder=closed (envelope mode)')
    parser.add_argument('--gate', action='store_true',
                        help='Use hard gate instead of smooth follow (envelope mode)')
    parser.add_argument('--env_min', type=float, default=0.0,
                        help='Minimum envelope value (envelope mode)')
    parser.add_argument('--env_max', type=float, default=1.0,
                        help='Maximum envelope value (envelope mode)')

    # Wah shaping parameters (for lfo, precise, envelope modes)
    parser.add_argument('--closed_boost_freq', type=float, default=800,
                        help='Resonant boost frequency when closed (Hz)')
    parser.add_argument('--closed_boost_db', type=float, default=12,
                        help='Boost amount when closed (dB)')
    parser.add_argument('--closed_cut_freq', type=float, default=2500,
                        help='High frequency cutoff when closed (Hz)')
    parser.add_argument('--closed_cut_db', type=float, default=-18,
                        help='High cut amount when closed (dB)')
    parser.add_argument('--open_boost_freq', type=float, default=3000,
                        help='High boost frequency when open (Hz)')
    parser.add_argument('--open_boost_db', type=float, default=6,
                        help='High boost amount when open (dB)')
    parser.add_argument('--resonance', type=float, default=2.5,
                        help='Resonance Q factor')

    args = parser.parse_args()

    # Load input
    print(f"Loading input: {args.input}")
    audio, sr = load_audio(args.input)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.2f}s, SR: {sr}")

    # Process based on mode
    if args.mode == 'reference':
        if not args.reference:
            parser.error("--reference required for reference mode")

        print(f"\nReference mode: {args.reference}")
        reference, _ = load_audio(args.reference, sr)
        print(f"  Reference duration: {len(reference)/sr:.2f}s")

        output = apply_wah_from_reference(
            audio, reference, sr,
            smoothing_sigma=args.smoothing
        )

    elif args.mode == 'lfo':
        print(f"\nLFO mode: {args.waveform} @ {args.rate} Hz, depth={args.depth}")

        modulation, mod_times = generate_lfo_modulation(
            duration, sr,
            rate=args.rate,
            depth=args.depth,
            waveform=args.waveform,
            phase=args.phase,
            offset=args.offset
        )

        output = apply_wah_from_modulation(
            audio, sr, modulation, mod_times,
            closed_boost_freq=args.closed_boost_freq,
            closed_boost_db=args.closed_boost_db,
            closed_cut_freq=args.closed_cut_freq,
            closed_cut_db=args.closed_cut_db,
            open_boost_freq=args.open_boost_freq,
            open_boost_db=args.open_boost_db,
            resonance_q=args.resonance
        )

    elif args.mode == 'precise':
        # Load automation points
        if args.automation:
            print(f"\nPrecise mode: loading {args.automation}")
            with open(args.automation, 'r') as f:
                auto_data = json.load(f)
            points = auto_data.get('points', auto_data)
        elif args.automation_points:
            print(f"\nPrecise mode: inline automation")
            points = json.loads(args.automation_points)
        else:
            parser.error("--automation or --automation_points required for precise mode")

        print(f"  {len(points)} automation points")

        modulation, mod_times = generate_precise_modulation(points, duration)

        output = apply_wah_from_modulation(
            audio, sr, modulation, mod_times,
            closed_boost_freq=args.closed_boost_freq,
            closed_boost_db=args.closed_boost_db,
            closed_cut_freq=args.closed_cut_freq,
            closed_cut_db=args.closed_cut_db,
            open_boost_freq=args.open_boost_freq,
            open_boost_db=args.open_boost_db,
            resonance_q=args.resonance
        )

    elif args.mode == 'envelope':
        print(f"\nEnvelope mode:")
        print(f"  Threshold: {args.threshold}, Attack: {args.attack}s, Decay: {args.decay}s")
        print(f"  Sensitivity: {args.sensitivity}, Invert: {args.invert}, Gate: {args.gate}")

        modulation, mod_times = generate_envelope_modulation(
            audio, sr,
            threshold=args.threshold,
            attack=args.attack,
            decay=args.decay,
            sensitivity=args.sensitivity,
            invert=args.invert,
            gate_mode=args.gate,
            min_val=args.env_min,
            max_val=args.env_max
        )

        output = apply_wah_from_modulation(
            audio, sr, modulation, mod_times,
            closed_boost_freq=args.closed_boost_freq,
            closed_boost_db=args.closed_boost_db,
            closed_cut_freq=args.closed_cut_freq,
            closed_cut_db=args.closed_cut_db,
            open_boost_freq=args.open_boost_freq,
            open_boost_db=args.open_boost_db,
            resonance_q=args.resonance
        )

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    output_int = (output * 32767).astype(np.int16)
    wavfile.write(args.output, sr, output_int)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
