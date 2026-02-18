#!/usr/bin/env python3
"""Chord/polyphonic inverse synthesis.

Given audio of a chord, separate individual notes and recover a single
synth patch that can reproduce all notes.

Strategy:
1. Separate harmonic from percussive (HPSS)
2. Detect all pitches in the chord
3. Isolate the strongest note via comb filtering
4. Recover patch from isolated note using optimize_patch_full
5. Verify patch sounds correct at other detected pitches
"""

import sys
sys.path.insert(0, 'scripts')

import numpy as np
from fast_dsp import (
    full_render, ALL_WF_FNS, SAMPLE_RATE, N_SAMPLES,
    spectral_similarity, fast_normalize,
)


def separate_harmonic_percussive(audio, sr=SAMPLE_RATE):
    """Separate audio into harmonic and percussive components using HPSS.

    Returns (harmonic, percussive) as numpy arrays.
    """
    import librosa

    # HPSS via librosa
    S = librosa.stft(audio.astype(np.float32), n_fft=2048, hop_length=512)
    H, P = librosa.decompose.hpss(S, margin=3.0)

    harmonic = librosa.istft(H, hop_length=512, length=len(audio))
    percussive = librosa.istft(P, hop_length=512, length=len(audio))

    return harmonic.astype(np.float32), percussive.astype(np.float32)


def isolate_note_comb(audio, pitch_hz, sr=SAMPLE_RATE, n_harmonics=12):
    """Isolate a single note from a chord using comb filtering.

    Extracts the harmonic series of the given pitch by bandpass-filtering
    around each harmonic.

    Args:
        audio: mono audio
        pitch_hz: fundamental frequency to isolate
        sr: sample rate
        n_harmonics: number of harmonics to extract

    Returns:
        isolated audio (numpy array)
    """
    from scipy.signal import butter, filtfilt

    n = len(audio)
    isolated = np.zeros(n, dtype=np.float64)

    for h in range(1, n_harmonics + 1):
        f_center = pitch_hz * h
        if f_center >= sr / 2 - 100:
            break

        # Bandwidth: proportional to harmonic number (wider for higher harmonics)
        bw = max(pitch_hz * 0.4, 30)  # at least 30Hz bandwidth

        f_low = max(20, f_center - bw / 2)
        f_high = min(sr / 2 - 1, f_center + bw / 2)

        if f_high <= f_low:
            continue

        # Butterworth bandpass
        try:
            b, a = butter(2, [f_low / (sr / 2), f_high / (sr / 2)], btype='band')
            harmonic = filtfilt(b, a, audio.astype(np.float64))
            isolated += harmonic
        except Exception:
            continue

    return fast_normalize(isolated.astype(np.float32))


def detect_chord_pitches(audio, sr=SAMPLE_RATE, max_notes=6):
    """Detect multiple pitches in a chord.

    Uses multi-pitch detection via spectral peak analysis
    with harmonic grouping.

    Args:
        audio: mono audio
        sr: sample rate
        max_notes: maximum number of notes

    Returns:
        list of (pitch_hz, confidence) tuples, sorted by confidence desc.
    """
    from pitch_detect import detect_pitch_multi
    return detect_pitch_multi(audio, sr, max_notes=max_notes)


def recover_chord_patch(audio, sr=SAMPLE_RATE, verbose=True):
    """Recover a single synth patch from a chord recording.

    Steps:
    1. Separate harmonic content
    2. Detect all pitches in the chord
    3. Isolate the strongest note
    4. Recover patch parameters from the isolated note
    5. Verify the patch works at other pitches

    Args:
        audio: mono audio of a chord
        sr: sample rate
        verbose: print progress

    Returns:
        dict with:
            'patch': recovered patch parameters
            'pitches': list of detected pitches
            'primary_pitch': pitch of the strongest note
            'spectral_sim': spectral similarity on primary note
            'cross_pitch_sim': avg spectral similarity across all notes
    """
    from test_audio_domain import optimize_patch_full

    if verbose:
        print("\n=== Chord Inverse Synthesis ===")

    # Step 1: Separate harmonic content
    if verbose:
        print("  Step 1: HPSS separation...")
    try:
        harmonic, percussive = separate_harmonic_percussive(audio, sr)
        harm_rms = np.sqrt(np.mean(harmonic ** 2))
        perc_rms = np.sqrt(np.mean(percussive ** 2))
        if verbose:
            print(f"    Harmonic RMS: {harm_rms:.4f}, Percussive RMS: {perc_rms:.4f}")
        # Use harmonic component if it has reasonable energy
        if harm_rms > 0.01:
            analysis_audio = harmonic
        else:
            analysis_audio = audio
    except Exception as e:
        if verbose:
            print(f"    HPSS failed ({e}), using original audio")
        analysis_audio = audio

    # Step 2: Detect pitches
    if verbose:
        print("  Step 2: Detecting pitches...")
    pitches = detect_chord_pitches(analysis_audio, sr)

    if not pitches:
        if verbose:
            print("    No pitches detected!")
        return {
            'patch': None,
            'pitches': [],
            'primary_pitch': None,
            'spectral_sim': 0.0,
            'cross_pitch_sim': 0.0,
        }

    if verbose:
        print(f"    Detected {len(pitches)} pitches:")
        for p, c in pitches:
            print(f"      {p:.1f}Hz (conf={c:.2f})")

    primary_pitch = pitches[0][0]

    # Step 3: Isolate the strongest note
    if verbose:
        print(f"  Step 3: Isolating primary note ({primary_pitch:.1f}Hz)...")
    if len(pitches) > 1:
        isolated = isolate_note_comb(analysis_audio, primary_pitch, sr)
    else:
        isolated = analysis_audio

    # Pad/trim to standard length
    if len(isolated) < N_SAMPLES:
        isolated = np.pad(isolated, (0, N_SAMPLES - len(isolated)))
    else:
        isolated = isolated[:N_SAMPLES]

    # Step 4: Recover patch from isolated note
    if verbose:
        print("  Step 4: Recovering patch parameters...")
    result = optimize_patch_full(isolated, primary_pitch, verbose=verbose)

    # Step 5: Cross-pitch verification
    cross_sims = [result['spectral_sim']]

    if len(pitches) > 1 and result.get('waveform'):
        wf_name = result.get('waveform', 'saw')
        filter_type = result.get('filter_type', 'lowpass')

        if verbose:
            print("  Step 5: Cross-pitch verification...")

        for pitch_hz, conf in pitches[1:4]:  # Check up to 3 other notes
            # Isolate this note
            note_audio = isolate_note_comb(analysis_audio, pitch_hz, sr)
            if len(note_audio) < N_SAMPLES:
                note_audio = np.pad(note_audio, (0, N_SAMPLES - len(note_audio)))
            else:
                note_audio = note_audio[:N_SAMPLES]

            # Render the recovered patch at this pitch
            wf_fn = ALL_WF_FNS.get(wf_name, ALL_WF_FNS['saw'])
            waveform = wf_fn(pitch_hz) if wf_name != 'noise' else wf_fn()

            params = [
                result.get('filter_base_hz', 500),
                result.get('filter_peak_hz', 3000),
                result.get('resonance', 0.3),
                *result.get('filter_adsr', (0.01, 0.15, 0.5, 0.1, 1.0)),
                *result.get('amp_adsr', (0.01, 0.15, 0.5, 0.1, 1.0)),
            ]

            rendered = full_render(params, waveform, filter_type=filter_type)
            sim = spectral_similarity(rendered, note_audio)
            cross_sims.append(sim)

            if verbose:
                print(f"    {pitch_hz:.1f}Hz: spec={sim:.4f}")

    cross_pitch_sim = float(np.mean(cross_sims))

    if verbose:
        print(f"\n  Result: primary_spec={result['spectral_sim']:.4f}, "
              f"cross_pitch_sim={cross_pitch_sim:.4f}")
        print(f"  Pitches: {[f'{p:.0f}Hz' for p, _ in pitches]}")
        wf = result.get('waveform', result.get('synth_type', '?'))
        print(f"  Waveform: {wf}, Filter: {result.get('filter_type', 'lowpass')}")

    return {
        'patch': result,
        'pitches': [(p, c) for p, c in pitches],
        'primary_pitch': primary_pitch,
        'spectral_sim': result['spectral_sim'],
        'cross_pitch_sim': cross_pitch_sim,
    }


def generate_test_chord(patch_def, pitches, sr=SAMPLE_RATE, n_samples=N_SAMPLES):
    """Generate a test chord from a patch definition at multiple pitches.

    Mixes individual notes at equal volume.

    Args:
        patch_def: dict with waveform, filter, amp params (same as TARGET_PATCHES)
        pitches: list of pitch frequencies in Hz
        sr: sample rate
        n_samples: number of samples

    Returns:
        mixed chord audio (numpy array)
    """
    from test_audio_domain import generate_target_audio

    mixed = np.zeros(n_samples, dtype=np.float64)

    for pitch in pitches:
        # Create a copy of the patch def with the new pitch
        pdef = dict(patch_def)
        pdef['pitch'] = pitch
        note_audio = generate_target_audio(pdef)
        mixed += note_audio[:n_samples].astype(np.float64)

    # Normalize
    return fast_normalize(mixed.astype(np.float32))
