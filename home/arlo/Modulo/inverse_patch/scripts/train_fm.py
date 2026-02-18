#!/usr/bin/env python3
"""
Train FM Synthesis Module — Frequency Modulation in z-space

FM synthesis: carrier frequency modulated by another oscillator.
  output(t) = sin(2π*f_c*t + I * sin(2π*f_m*t))

The model learns: (z_carrier, z_modulator, fm_index) → z_fm
Per-frame FM index for time-varying modulation depth (envelope/LFO control).

Training data:
  - Carrier/modulator frequency pairs at various ratios (harmonic + inharmonic)
  - FM indices from 0 (clean) to 8 (bright/metallic)
  - Static and time-varying (envelope, LFO) FM index curves
  - Multiple carrier waveforms (sine, saw, square)

Tests:
1. Static FM index — classic timbres at fixed modulation depth
2. FM index sweep — smooth transition from clean to bright
3. Harmonic vs inharmonic ratios — musical vs metallic/bell-like
4. Time-varying FM index — pluck envelope on FM depth (DX7 style)
5. Unseen combinations — generalize to untrained frequency pairs
"""

import sys
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

sys.path.insert(0, "/home/arlo/Data/ACE-Step")
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

sys.path.insert(0, str(Path(__file__).parent))
from latent_modules import LatentFM, Envelope, LFO

DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

SAMPLE_RATE = 44100
DURATION = 2.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
MAX_FM_INDEX = 8.0  # max modulation index for normalization

BASE_DIR = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch")
OUTPUT_DIR = BASE_DIR / "test_outputs" / "latent_fm"


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# DSP: FM Synthesis
# ============================================================

def _normalize(audio):
    peak = np.abs(audio).max()
    if peak > 1e-6:
        return (audio / peak * 0.8).astype(np.float32)
    return audio.astype(np.float32)


def _apply_envelope(audio):
    attack = int(0.01 * SAMPLE_RATE)
    release = int(0.05 * SAMPLE_RATE)
    env = np.ones_like(audio)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return _normalize(audio * env)


def generate_sine(pitch):
    t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float32)
    return _apply_envelope(np.sin(2 * np.pi * pitch * t))


def generate_saw(pitch):
    t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float32)
    audio = np.zeros_like(t)
    for h in range(1, 40):
        freq = pitch * h
        if freq > SAMPLE_RATE / 2: break
        audio += ((-1) ** (h + 1)) * np.sin(2 * np.pi * freq * t) / h
    return _apply_envelope(audio)


def generate_square(pitch):
    t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float32)
    audio = np.zeros_like(t)
    for h in range(1, 40, 2):
        freq = pitch * h
        if freq > SAMPLE_RATE / 2: break
        audio += np.sin(2 * np.pi * freq * t) / h
    return _apply_envelope(audio)


def generate_triangle(pitch):
    t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float32)
    audio = np.zeros_like(t)
    for i, h in enumerate(range(1, 40, 2)):
        freq = pitch * h
        if freq > SAMPLE_RATE / 2: break
        audio += ((-1) ** i) * np.sin(2 * np.pi * freq * t) / (h * h)
    return _apply_envelope(audio)


def dsp_fm_static(carrier_freq, mod_freq, fm_index, carrier_type='sine'):
    """FM synthesis with static modulation index.

    output(t) = waveform(2π*f_c*t + I * sin(2π*f_m*t))

    For sine carrier: classic Chowning FM.
    For other carriers: phase-modulated versions (richer spectra).
    """
    t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float64)
    modulator = np.sin(2 * np.pi * mod_freq * t)
    phase = 2 * np.pi * carrier_freq * t + fm_index * modulator

    if carrier_type == 'sine':
        output = np.sin(phase)
    elif carrier_type == 'saw':
        # Phase-wrapped sawtooth
        output = 2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
    elif carrier_type == 'square':
        output = np.sign(np.sin(phase))
    else:
        output = np.sin(phase)

    return _apply_envelope(output.astype(np.float32))


def dsp_fm_temporal(carrier_freq, mod_freq, fm_index_curve, carrier_type='sine'):
    """FM synthesis with time-varying modulation index.

    fm_index_curve: [N_SAMPLES] array of FM index values over time.
    """
    t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float64)
    modulator = np.sin(2 * np.pi * mod_freq * t)
    phase = 2 * np.pi * carrier_freq * t + fm_index_curve * modulator

    if carrier_type == 'sine':
        output = np.sin(phase)
    elif carrier_type == 'saw':
        output = 2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
    elif carrier_type == 'square':
        output = np.sign(np.sin(phase))
    else:
        output = np.sin(phase)

    return _apply_envelope(output.astype(np.float32))


def make_fm_envelope(peak_index, attack_s, decay_s, sustain_ratio, release_s, note_off_s):
    """ADSR envelope for FM index (audio-rate)."""
    sustain_val = peak_index * sustain_ratio
    env = np.zeros(N_SAMPLES, dtype=np.float64)

    attack_n = max(1, int(attack_s * SAMPLE_RATE))
    decay_n = max(1, int(decay_s * SAMPLE_RATE))
    release_n = max(1, int(release_s * SAMPLE_RATE))
    note_off_n = int(note_off_s * SAMPLE_RATE)

    pos = 0
    end = min(pos + attack_n, N_SAMPLES)
    if end > pos: env[pos:end] = np.linspace(0, peak_index, end - pos)
    pos = end

    end = min(pos + decay_n, N_SAMPLES)
    if end > pos: env[pos:end] = np.linspace(peak_index, sustain_val, end - pos)
    pos = end

    end = min(note_off_n, N_SAMPLES)
    if end > pos: env[pos:end] = sustain_val
    pos = end

    end = min(pos + release_n, N_SAMPLES)
    if end > pos: env[pos:end] = np.linspace(sustain_val, 0, end - pos)
    pos = end

    return env


def make_fm_lfo(center_index, depth, rate_hz):
    """Sinusoidal FM index modulation."""
    t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float64)
    return np.clip(center_index + depth * np.sin(2 * np.pi * rate_hz * t), 0, MAX_FM_INDEX)


def resample_to_z(curve, T):
    """Resample audio-rate curve to z-frame rate."""
    frame_size = len(curve) // T
    out = np.zeros(T, dtype=np.float32)
    for i in range(T):
        s = i * frame_size
        e = min(s + frame_size, len(curve))
        out[i] = curve[s:e].mean()
    return out


# ============================================================
# DCAE Helpers
# ============================================================

def encode_audio(dcae, audio, device='cuda'):
    audio_tensor = torch.from_numpy(audio).float().to(device)
    audio_stereo = audio_tensor.unsqueeze(0).unsqueeze(0).expand(-1, 2, -1)
    audio_lengths = torch.tensor([audio_stereo.shape[-1]], device=device)
    with torch.no_grad():
        z, _ = dcae.encode(audio_stereo, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
    return z.squeeze(0)


def decode_latent(dcae, z, device='cuda'):
    z_4d = z.unsqueeze(0) if z.dim() == 3 else z
    audio_lengths = torch.tensor([N_SAMPLES], device=device)
    with torch.no_grad():
        sr, wavs = dcae.decode(z_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
    return wavs[0].mean(dim=0).cpu().numpy()


def _save_wav(path, audio):
    torchaudio.save(str(path), torch.from_numpy(audio).unsqueeze(0).float(), SAMPLE_RATE)


# ============================================================
# Training
# ============================================================

def train_module(model, data_tensors, device, epochs=500, batch_size=16, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    inputs = data_tensors['inputs']
    target = data_tensors['target']
    n_samples = target.shape[0]

    print(f"  Training for {epochs} epochs, {n_samples} samples...")
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            batch_inputs = [inp[idx] for inp in inputs]

            z_pred = model(*batch_inputs)
            loss = F.mse_loss(z_pred, target[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:3d}: loss = {total_loss / n_batches:.6f}")

    return model


def generate_training_data(dcae, device):
    """Generate FM training pairs.

    Three categories:
    1. Static FM index — various carrier/mod frequency pairs and indices
    2. FM index envelope — DX7-style: index decays over time
    3. FM index LFO — vibrato-like modulation of FM depth
    """
    print("Generating FM training data...")

    # Carrier/modulator frequency pairs with ratio labels
    # Harmonic ratios: produce pitched, musical tones
    # Inharmonic ratios: produce bell-like, metallic tones
    freq_pairs = [
        # (carrier_hz, mod_hz, label)
        # Harmonic: c:m integer ratios
        (220, 220, 'h_1:1'),      # same frequency
        (220, 110, 'h_2:1'),      # octave below
        (220, 440, 'h_1:2'),      # octave above
        (330, 220, 'h_3:2'),      # fifth
        (440, 220, 'h_2:1_hi'),   # octave
        (110, 110, 'h_1:1_lo'),   # unison low
        (110, 220, 'h_1:2_lo'),   # octave mod
        (440, 440, 'h_1:1_hi'),   # unison high
        (330, 330, 'h_1:1_mid'),  # unison mid
        (220, 330, 'h_2:3'),      # inverted fifth
        # Inharmonic: non-integer ratios → bells, metallic
        (220, 277, 'ih_1:1.26'),  # minor third ratio
        (220, 311, 'ih_1:1.41'),  # sqrt(2) - very inharmonic
        (440, 553, 'ih_1:1.26h'), # minor third, higher
        (110, 155, 'ih_1:1.41l'), # sqrt(2), lower
        (330, 415, 'ih_1:1.26m'), # minor third, mid
    ]

    # FM indices (normalized to [0, 1] for the model, actual values for DSP)
    static_indices = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

    carrier_types = ['sine', 'saw']  # sine = classic FM, saw = aggressive FM

    # Pre-encode all carrier and modulator waveforms
    print("  Encoding source waveforms...")
    carrier_cache = {}
    mod_cache = {}

    carrier_fns = {'sine': generate_sine, 'saw': generate_saw}

    for f_c, f_m, label in freq_pairs:
        for ct in carrier_types:
            key_c = (ct, f_c)
            if key_c not in carrier_cache:
                audio_c = carrier_fns[ct](f_c)
                z_c = encode_audio(dcae, audio_c, device)
                carrier_cache[key_c] = {'audio': audio_c, 'z': z_c.cpu()}
                print(f"    carrier {ct}@{f_c}Hz")

        key_m = ('sine', f_m)
        if key_m not in mod_cache:
            audio_m = generate_sine(f_m)
            z_m = encode_audio(dcae, audio_m, device)
            mod_cache[key_m] = {'audio': audio_m, 'z': z_m.cpu()}
            print(f"    mod sine@{f_m}Hz")

    z_T = list(carrier_cache.values())[0]['z'].shape[-1]
    print(f"  z frames: T={z_T}")

    # --- Generate static FM pairs ---
    print("  Generating static FM pairs...")
    data_zc, data_zm, data_zw, data_idx = [], [], [], []
    count = 0

    for f_c, f_m, label in freq_pairs:
        for ct in carrier_types:
            for fm_idx in static_indices:
                audio_fm = dsp_fm_static(f_c, f_m, fm_idx, ct)
                z_fm = encode_audio(dcae, audio_fm, device)

                idx_norm = fm_idx / MAX_FM_INDEX  # normalize to [0, 1]

                data_zc.append(carrier_cache[(ct, f_c)]['z'])
                data_zm.append(mod_cache[('sine', f_m)]['z'])
                data_zw.append(z_fm.cpu())
                data_idx.append(torch.full((z_T,), idx_norm))

                count += 1
                if count % 30 == 0:
                    print(f"    {count} static pairs...")
                    clear_memory()

    print(f"  Static pairs: {count}")

    # --- Generate temporal FM pairs (envelope + LFO) ---
    print("  Generating temporal FM pairs...")

    # FM envelope configs: (peak_index, attack, decay, sustain_ratio, release, note_off)
    env_configs = [
        ('pluck_bright', 6.0, 0.005, 0.2, 0.1, 0.3, 1.2),
        ('pluck_dark', 3.0, 0.005, 0.15, 0.2, 0.2, 0.8),
        ('bell_decay', 8.0, 0.002, 0.8, 0.0, 0.1, 0.1),
        ('slow_swell', 4.0, 0.5, 0.1, 0.8, 0.3, 1.5),
        ('perc_snap', 6.0, 0.002, 0.05, 0.0, 0.1, 0.1),
    ]

    # LFO configs: (center_index, depth, rate_hz)
    lfo_configs = [
        ('slow_vibrato', 2.0, 1.5, 2.0),
        ('fast_shimmer', 3.0, 2.0, 6.0),
        ('deep_wobble', 4.0, 3.0, 1.5),
    ]

    # Use subset of freq pairs for temporal data (too many combos otherwise)
    temporal_pairs = freq_pairs[:8]

    for f_c, f_m, label in temporal_pairs:
        for ct in ['sine', 'saw']:
            # Envelopes
            for env_name, peak, a, d, s, r, noff in env_configs:
                fm_curve = make_fm_envelope(peak, a, d, s, r, noff)
                audio_fm = dsp_fm_temporal(f_c, f_m, fm_curve, ct)
                z_fm = encode_audio(dcae, audio_fm, device)

                fm_curve_z = resample_to_z(fm_curve, z_T) / MAX_FM_INDEX

                data_zc.append(carrier_cache[(ct, f_c)]['z'])
                data_zm.append(mod_cache[('sine', f_m)]['z'])
                data_zw.append(z_fm.cpu())
                data_idx.append(torch.from_numpy(fm_curve_z).float())

                count += 1
                if count % 30 == 0:
                    print(f"    {count} pairs (temporal)...")
                    clear_memory()

            # LFOs
            for lfo_name, center, depth, rate in lfo_configs:
                fm_curve = make_fm_lfo(center, depth, rate)
                audio_fm = dsp_fm_temporal(f_c, f_m, fm_curve, ct)
                z_fm = encode_audio(dcae, audio_fm, device)

                fm_curve_z = resample_to_z(fm_curve, z_T) / MAX_FM_INDEX

                data_zc.append(carrier_cache[(ct, f_c)]['z'])
                data_zm.append(mod_cache[('sine', f_m)]['z'])
                data_zw.append(z_fm.cpu())
                data_idx.append(torch.from_numpy(fm_curve_z).float())

                count += 1
                if count % 30 == 0:
                    print(f"    {count} pairs (LFO)...")
                    clear_memory()

    print(f"  Total training pairs: {count}")
    return data_zc, data_zm, data_zw, data_idx, z_T, carrier_cache, mod_cache


# ============================================================
# Tests
# ============================================================

def test_static_fm(model, dcae, carrier_cache, mod_cache, z_T, device, out_dir):
    """Test 1: Static FM index at various depths."""
    print("\n" + "=" * 60)
    print("Test 1: Static FM Index")
    print("=" * 60)

    d = out_dir / "static"
    d.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Classic sine FM at various indices
    for f_c, f_m, label in [(220, 220, '1to1'), (220, 110, '2to1'), (220, 277, 'bell')]:
        carrier = carrier_cache[('sine', f_c)]
        modulator = mod_cache[('sine', f_m)]
        zc = carrier['z'].unsqueeze(0).to(device)
        zm = modulator['z'].unsqueeze(0).to(device)

        _save_wav(d / f"{label}_carrier.wav", carrier['audio'])

        for fm_idx in [0.0, 1.0, 2.0, 4.0, 8.0]:
            idx_norm = fm_idx / MAX_FM_INDEX
            idx_t = torch.full((1, z_T), idx_norm, device=device)

            with torch.no_grad():
                z_pred = model(zc, zm, idx_t)

            audio_pred = decode_latent(dcae, z_pred.squeeze(0), device)
            audio_gt = dsp_fm_static(f_c, f_m, fm_idx, 'sine')

            _save_wav(d / f"{label}_idx{fm_idx:.0f}_pred.wav", audio_pred)
            _save_wav(d / f"{label}_idx{fm_idx:.0f}_gt.wav", audio_gt)

            # Measure
            z_gt = encode_audio(dcae, audio_gt, device)
            cos = F.cosine_similarity(
                z_pred.flatten().unsqueeze(0),
                z_gt.flatten().unsqueeze(0)
            ).item()
            mse = F.mse_loss(z_pred.squeeze(0), z_gt).item()
            print(f"  {label} idx={fm_idx:.0f}: cos={cos:.4f} mse={mse:.6f}")

        clear_memory()


def test_fm_sweep(model, dcae, carrier_cache, mod_cache, z_T, device, out_dir):
    """Test 2: Smooth FM index sweep from 0 to max."""
    print("\n" + "=" * 60)
    print("Test 2: FM Index Sweep")
    print("=" * 60)

    d = out_dir / "sweep"
    d.mkdir(parents=True, exist_ok=True)

    model.eval()

    for f_c, f_m, label in [(220, 220, '1to1'), (220, 277, 'bell')]:
        carrier = carrier_cache[('sine', f_c)]
        modulator = mod_cache[('sine', f_m)]
        zc = carrier['z'].unsqueeze(0).to(device)
        zm = modulator['z'].unsqueeze(0).to(device)

        # Sweep index from 0 to 8 over duration
        idx_curve = np.linspace(0, 1.0, z_T, dtype=np.float32)  # normalized
        idx_t = torch.from_numpy(idx_curve).float().unsqueeze(0).to(device)

        with torch.no_grad():
            z_pred = model(zc, zm, idx_t)

        audio_pred = decode_latent(dcae, z_pred.squeeze(0), device)

        # DSP ground truth
        fm_curve_audio = np.linspace(0, MAX_FM_INDEX, N_SAMPLES)
        audio_gt = dsp_fm_temporal(f_c, f_m, fm_curve_audio, 'sine')

        _save_wav(d / f"{label}_sweep_pred.wav", audio_pred)
        _save_wav(d / f"{label}_sweep_gt.wav", audio_gt)

        z_gt = encode_audio(dcae, audio_gt, device)
        cos = F.cosine_similarity(
            z_pred.flatten().unsqueeze(0), z_gt.flatten().unsqueeze(0)
        ).item()
        print(f"  {label} sweep: cos={cos:.4f}")

    clear_memory()


def test_harmonic_vs_inharmonic(model, dcae, carrier_cache, mod_cache, z_T, device, out_dir):
    """Test 3: Harmonic (musical) vs inharmonic (metallic/bell) ratios."""
    print("\n" + "=" * 60)
    print("Test 3: Harmonic vs Inharmonic Ratios")
    print("=" * 60)

    d = out_dir / "ratios"
    d.mkdir(parents=True, exist_ok=True)

    model.eval()

    test_cases = [
        (220, 220, 'harmonic_1to1', 4.0),    # unison → rich
        (220, 440, 'harmonic_1to2', 4.0),     # octave → bright
        (330, 220, 'harmonic_3to2', 4.0),     # fifth → organ-like
        (220, 277, 'inharmonic_bell', 4.0),   # minor 3rd → bell
        (220, 311, 'inharmonic_metal', 4.0),  # sqrt(2) → metallic
    ]

    for f_c, f_m, label, fm_idx in test_cases:
        ct = 'sine'
        if (ct, f_c) not in carrier_cache:
            # Encode on the fly
            audio_c = generate_sine(f_c)
            z_c = encode_audio(dcae, audio_c, device)
            carrier_cache[(ct, f_c)] = {'audio': audio_c, 'z': z_c.cpu()}
        if ('sine', f_m) not in mod_cache:
            audio_m = generate_sine(f_m)
            z_m = encode_audio(dcae, audio_m, device)
            mod_cache[('sine', f_m)] = {'audio': audio_m, 'z': z_m.cpu()}

        zc = carrier_cache[(ct, f_c)]['z'].unsqueeze(0).to(device)
        zm = mod_cache[('sine', f_m)]['z'].unsqueeze(0).to(device)

        idx_norm = fm_idx / MAX_FM_INDEX
        idx_t = torch.full((1, z_T), idx_norm, device=device)

        with torch.no_grad():
            z_pred = model(zc, zm, idx_t)

        audio_pred = decode_latent(dcae, z_pred.squeeze(0), device)
        audio_gt = dsp_fm_static(f_c, f_m, fm_idx, ct)

        _save_wav(d / f"{label}_pred.wav", audio_pred)
        _save_wav(d / f"{label}_gt.wav", audio_gt)

        z_gt = encode_audio(dcae, audio_gt, device)
        cos = F.cosine_similarity(
            z_pred.flatten().unsqueeze(0), z_gt.flatten().unsqueeze(0)
        ).item()
        print(f"  {label}: cos={cos:.4f}")

    clear_memory()


def test_temporal_fm(model, dcae, carrier_cache, mod_cache, z_T, device, out_dir):
    """Test 4: Time-varying FM index — DX7 style envelopes."""
    print("\n" + "=" * 60)
    print("Test 4: Time-Varying FM Index (DX7 Style)")
    print("=" * 60)

    d = out_dir / "temporal"
    d.mkdir(parents=True, exist_ok=True)

    model.eval()

    # DX7-style patches: FM index follows envelope
    patches = {
        'electric_piano': {
            'f_c': 220, 'f_m': 220, 'ct': 'sine',
            'env': make_fm_envelope(5.0, 0.005, 0.3, 0.1, 0.4, 1.2),
            'desc': 'Classic EP: bright attack decaying to mellow',
        },
        'bell_hit': {
            'f_c': 440, 'f_m': 553, 'ct': 'sine',
            'env': make_fm_envelope(8.0, 0.002, 1.5, 0.0, 0.1, 0.1),
            'desc': 'Inharmonic bell: high index, long decay',
        },
        'bass_pluck': {
            'f_c': 110, 'f_m': 110, 'ct': 'sine',
            'env': make_fm_envelope(4.0, 0.005, 0.1, 0.2, 0.2, 0.6),
            'desc': 'FM bass: quick bright pluck',
        },
        'metallic_pad': {
            'f_c': 220, 'f_m': 311, 'ct': 'sine',
            'env': make_fm_envelope(3.0, 0.3, 0.2, 0.6, 0.5, 1.5),
            'desc': 'Slow attack metallic pad',
        },
        'saw_fm_pluck': {
            'f_c': 220, 'f_m': 220, 'ct': 'saw',
            'env': make_fm_envelope(4.0, 0.005, 0.15, 0.1, 0.3, 1.0),
            'desc': 'Saw carrier FM: aggressive pluck',
        },
    }

    # Include unseen configs
    patches['unseen_bright_bell'] = {
        'f_c': 330, 'f_m': 415, 'ct': 'sine',
        'env': make_fm_envelope(7.0, 0.003, 1.0, 0.05, 0.2, 0.2),
        'desc': 'UNSEEN: bright inharmonic bell',
    }
    patches['unseen_lfo_shimmer'] = {
        'f_c': 440, 'f_m': 440, 'ct': 'sine',
        'env': make_fm_lfo(2.0, 1.5, 4.0),
        'desc': 'UNSEEN: LFO modulating FM depth',
    }

    for name, patch in patches.items():
        f_c = patch['f_c']
        f_m = patch['f_m']
        ct = patch['ct']

        # Ensure sources are cached
        if (ct, f_c) not in carrier_cache:
            fn = {'sine': generate_sine, 'saw': generate_saw}[ct]
            audio_c = fn(f_c)
            z_c = encode_audio(dcae, audio_c, device)
            carrier_cache[(ct, f_c)] = {'audio': audio_c, 'z': z_c.cpu()}
        if ('sine', f_m) not in mod_cache:
            audio_m = generate_sine(f_m)
            z_m = encode_audio(dcae, audio_m, device)
            mod_cache[('sine', f_m)] = {'audio': audio_m, 'z': z_m.cpu()}

        zc = carrier_cache[(ct, f_c)]['z'].unsqueeze(0).to(device)
        zm = mod_cache[('sine', f_m)]['z'].unsqueeze(0).to(device)

        fm_curve = patch['env']
        fm_curve_z = resample_to_z(fm_curve, z_T) / MAX_FM_INDEX
        idx_t = torch.from_numpy(fm_curve_z).float().unsqueeze(0).to(device)

        with torch.no_grad():
            z_pred = model(zc, zm, idx_t)

        audio_pred = decode_latent(dcae, z_pred.squeeze(0), device)
        audio_gt = dsp_fm_temporal(f_c, f_m, fm_curve, ct)

        _save_wav(d / f"{name}_pred.wav", audio_pred)
        _save_wav(d / f"{name}_gt.wav", audio_gt)

        z_gt = encode_audio(dcae, audio_gt, device)
        cos = F.cosine_similarity(
            z_pred.flatten().unsqueeze(0), z_gt.flatten().unsqueeze(0)
        ).item()
        print(f"  {name:20s}: cos={cos:.4f}  ({patch['desc']})")

    clear_memory()


def test_linearity(model, dcae, carrier_cache, mod_cache, z_T, device, out_dir):
    """Test 5: How bad is naive z interpolation vs learned FM?"""
    print("\n" + "=" * 60)
    print("Test 5: Linearity Test — Naive vs Learned")
    print("=" * 60)

    model.eval()

    # Compare: naive approach (lerp between carrier and some target)
    # vs learned FM model
    f_c, f_m = 220, 220
    carrier = carrier_cache[('sine', f_c)]
    modulator = mod_cache[('sine', f_m)]

    zc = carrier['z'].unsqueeze(0).to(device)
    zm = modulator['z'].unsqueeze(0).to(device)

    for fm_idx in [1.0, 2.0, 4.0, 8.0]:
        audio_gt = dsp_fm_static(f_c, f_m, fm_idx, 'sine')
        z_gt = encode_audio(dcae, audio_gt, device)

        idx_norm = fm_idx / MAX_FM_INDEX
        idx_t = torch.full((1, z_T), idx_norm, device=device)

        # Learned
        with torch.no_grad():
            z_model = model(zc, zm, idx_t)

        # Naive: linear interpolation carrier → ground truth (cheating baseline)
        # Actually: naive attempt = carrier_z * (1 - idx) + mod_z * idx
        z_naive = zc * (1 - idx_norm) + zm * idx_norm

        cos_model = F.cosine_similarity(
            z_model.flatten().unsqueeze(0), z_gt.flatten().unsqueeze(0)
        ).item()
        cos_naive = F.cosine_similarity(
            z_naive.flatten().unsqueeze(0), z_gt.flatten().unsqueeze(0)
        ).item()
        mse_model = F.mse_loss(z_model.squeeze(0), z_gt).item()
        mse_naive = F.mse_loss(z_naive.squeeze(0), z_gt).item()

        improvement = ((mse_naive - mse_model) / mse_naive * 100) if mse_naive > 0 else 0
        print(f"  idx={fm_idx:.0f}: model cos={cos_model:.4f} mse={mse_model:.6f} | "
              f"naive cos={cos_naive:.4f} mse={mse_naive:.6f} | "
              f"improvement={improvement:.1f}%")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("LATENT FM SYNTHESIS — Train + Test")
    print("=" * 60)

    device = 'cuda'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_PATH,
        vocoder_checkpoint_path=VOCODER_PATH,
    ).to(device)
    dcae.eval()
    print("DCAE loaded!\n")

    # Generate training data
    data_zc, data_zm, data_zw, data_idx, z_T, carrier_cache, mod_cache = \
        generate_training_data(dcae, device)
    clear_memory()

    # Build model
    model = LatentFM(n_channels=8, latent_dim=16, cond_dim=128, n_blocks=6).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nLatentFM params: {n_params:,}")

    # Train
    tensors = {
        'inputs': [
            torch.stack(data_zc).to(device),
            torch.stack(data_zm).to(device),
            torch.stack(data_idx).to(device),
        ],
        'target': torch.stack(data_zw).to(device),
    }

    # Free lists
    del data_zc, data_zm, data_zw, data_idx
    clear_memory()

    model = train_module(model, tensors, device, epochs=500, batch_size=16)

    # Save checkpoint
    torch.save({
        'model': model.state_dict(),
        'config': {'n_channels': 8, 'latent_dim': 16, 'cond_dim': 128, 'n_blocks': 6},
    }, str(OUTPUT_DIR / "latent_fm.pt"))
    print(f"\nCheckpoint saved to {OUTPUT_DIR / 'latent_fm.pt'}")

    # Free training data
    del tensors
    clear_memory()

    # Tests
    test_static_fm(model, dcae, carrier_cache, mod_cache, z_T, device, OUTPUT_DIR)
    test_fm_sweep(model, dcae, carrier_cache, mod_cache, z_T, device, OUTPUT_DIR)
    test_harmonic_vs_inharmonic(model, dcae, carrier_cache, mod_cache, z_T, device, OUTPUT_DIR)
    test_temporal_fm(model, dcae, carrier_cache, mod_cache, z_T, device, OUTPUT_DIR)
    test_linearity(model, dcae, carrier_cache, mod_cache, z_T, device, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("FM SYNTHESIS MODULE — COMPLETE")
    print("=" * 60)
    print(f"\nOutputs: {OUTPUT_DIR}")
    print("\nKey files to listen to:")
    print("  static/1to1_idx4_pred.wav — classic FM at moderate depth")
    print("  static/bell_idx8_pred.wav — metallic bell (inharmonic + high index)")
    print("  temporal/electric_piano_pred.wav — DX7-style EP")
    print("  temporal/bell_hit_pred.wav — FM bell with long decay")
    print("  ratios/inharmonic_metal_pred.wav — sqrt(2) ratio = pure metal")


if __name__ == "__main__":
    main()
