#!/usr/bin/env python3
"""
Train All Modules — Batch training for the Latent Modular Synth

Trains all remaining transform modules sequentially, sharing DCAE.
Each module follows the same pattern:
  1. Generate DSP ground truth (dry audio → effect → wet audio)
  2. Encode both through DCAE
  3. Train neural z→z transform
  4. Run tests

Modules trained:
  - Wavefolder (nonlinear distortion)
  - Distortion (overdrive + tone)
  - Ring Modulator (multiply two signals)
  - Delay (echo with feedback)
  - Reverb (room/space)
  - Chorus (modulated delay)
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
from scipy.signal import lfilter

sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

sys.path.insert(0, "/home/arlo/Data/ACE-Step")
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

sys.path.insert(0, str(Path(__file__).parent))
from latent_modules import (
    LatentWavefolder, LatentDistortion, LatentRingMod,
    LatentDelay, LatentReverb, LatentChorus,
    Envelope, LFO,
)

DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

SAMPLE_RATE = 44100
DURATION = 2.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)

OUTPUT_BASE = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/test_outputs")


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Shared DSP utilities
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


def generate_pulse(pitch, duty=0.25):
    t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float32)
    audio = np.zeros_like(t)
    for h in range(1, 40):
        freq = pitch * h
        if freq > SAMPLE_RATE / 2: break
        audio += np.sin(np.pi * h * duty) * np.sin(2 * np.pi * freq * t) / h
    return _apply_envelope(audio)


def generate_sine(pitch):
    t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float32)
    return _apply_envelope(np.sin(2 * np.pi * pitch * t))


# ============================================================
# DCAE helpers
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
# Shared training loop
# ============================================================

def encode_sources(dcae, device):
    """Encode standard source waveforms. Shared across all modules."""
    print("  Encoding sources...")
    waveform_fns = {
        'saw': generate_saw, 'square': generate_square, 'triangle': generate_triangle,
    }
    pitches = [110, 220, 330, 440]
    sources = {}
    z_T = None
    for wf, fn in waveform_fns.items():
        for p in pitches:
            audio = fn(p)
            z = encode_audio(dcae, audio, device)
            sources[(wf, p)] = {'audio': audio, 'z': z.cpu()}
            if z_T is None:
                z_T = z.shape[-1]
            print(f"    {wf}@{p}Hz: z {z.shape}")
    return sources, z_T


def train_module(model, data_tensors, device, epochs=400, batch_size=16, lr=1e-3):
    """Generic training loop for any module.

    data_tensors: dict with 'inputs' (list of tensors) and 'target' tensor
    The model's forward() is called with *inputs.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    inputs = data_tensors['inputs']  # list of tensors on device
    target = data_tensors['target']  # tensor on device
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


# ============================================================
# DSP Effects
# ============================================================

def dsp_wavefold(audio, fold_amount):
    """Wavefold: fold waveform peaks back. fold_amount in [1, 8]."""
    a = audio * fold_amount
    # Triangular folding
    folded = np.arcsin(np.clip(np.sin(a * np.pi / 2), -1, 1)) * 2 / np.pi
    return _normalize(folded)


def dsp_distortion(audio, drive, tone_hz):
    """Soft-clip distortion with tone filter.
    drive: gain before clipping (1-20)
    tone_hz: post-distortion lowpass cutoff
    """
    driven = np.tanh(audio * drive)
    # Simple 1-pole lowpass for tone
    w0 = 2 * np.pi * np.clip(tone_hz, 200, SAMPLE_RATE / 2 - 100) / SAMPLE_RATE
    alpha = np.sin(w0) / (2 * 0.707)
    b0 = (1 - np.cos(w0)) / 2
    b1 = 1 - np.cos(w0)
    b2 = (1 - np.cos(w0)) / 2
    a0 = 1 + alpha
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha
    b = [b0/a0, b1/a0, b2/a0]
    a = [1, a1/a0, a2/a0]
    filtered = lfilter(b, a, driven)
    return _normalize(filtered)


def dsp_ringmod(carrier, modulator, depth):
    """Ring modulation: multiply carrier with modulator.
    depth: 0=dry carrier, 1=full ring mod
    """
    modulated = carrier * (1 - depth + depth * modulator)
    return _normalize(modulated)


def dsp_delay(audio, time_ms, feedback, mix):
    """Simple feedback delay.
    time_ms: delay time in ms (50-500)
    feedback: feedback amount (0-0.8)
    mix: dry/wet (0-1)
    """
    n = len(audio)
    delay_samples = int(time_ms * SAMPLE_RATE / 1000)
    wet = np.zeros(n, dtype=np.float64)

    for tap in range(1, 15):
        offset = tap * delay_samples
        if offset >= n:
            break
        gain = feedback ** tap
        if gain < 0.001:
            break
        wet[offset:] += audio[:n - offset] * gain

    mixed = audio * (1 - mix) + wet.astype(np.float32) * mix
    return _normalize(mixed)


def dsp_reverb(audio, size, decay, mix):
    """Simple algorithmic reverb using multiple comb + allpass filters.
    size: room size (scales delay lengths) [0.2-1.0]
    decay: RT60-ish control [0.2-0.9]
    """
    # Schroeder reverb: 4 comb filters + 2 allpass
    base_delays_ms = [29.7, 37.1, 41.1, 43.7]  # Schroeder primes
    allpass_delays_ms = [5.0, 1.7]

    comb_out = np.zeros_like(audio, dtype=np.float64)
    for delay_ms in base_delays_ms:
        d = int(delay_ms * size * SAMPLE_RATE / 1000)
        if d < 1: d = 1
        g = decay ** (d / SAMPLE_RATE * 3)  # feedback gain
        buf = np.zeros(len(audio), dtype=np.float64)
        for n in range(len(audio)):
            if n >= d:
                buf[n] = audio[n] + g * buf[n - d]
            else:
                buf[n] = audio[n]
        comb_out += buf

    comb_out /= len(base_delays_ms)

    # Allpass filters
    out = comb_out
    for delay_ms in allpass_delays_ms:
        d = int(delay_ms * size * SAMPLE_RATE / 1000)
        if d < 1: d = 1
        g = 0.7
        buf = np.zeros(len(audio), dtype=np.float64)
        for n in range(len(audio)):
            if n >= d:
                buf[n] = -g * out[n] + out[n - d] + g * buf[n - d]
            else:
                buf[n] = -g * out[n]
        out = buf

    wet = out.astype(np.float32)
    mixed = audio * (1 - mix) + _normalize(wet) * mix
    return _normalize(mixed)


def dsp_chorus(audio, rate, depth_ms, mix):
    """Chorus via modulated delay.
    rate: LFO rate in Hz (0.5-5)
    depth_ms: modulation depth in ms (1-10)
    mix: dry/wet (0-1)
    """
    t = np.arange(N_SAMPLES, dtype=np.float32)
    # LFO modulates delay time
    base_delay_ms = 10.0
    mod = depth_ms * np.sin(2 * np.pi * rate * t / SAMPLE_RATE)
    delay_samples = ((base_delay_ms + mod) * SAMPLE_RATE / 1000).astype(np.float32)

    wet = np.zeros_like(audio)
    for n in range(N_SAMPLES):
        d = delay_samples[n]
        d_int = int(d)
        d_frac = d - d_int
        if n - d_int - 1 >= 0:
            wet[n] = audio[n - d_int] * (1 - d_frac) + audio[n - d_int - 1] * d_frac
        elif n - d_int >= 0:
            wet[n] = audio[n - d_int]

    mixed = audio * (1 - mix) + wet * mix
    return _normalize(mixed)


# ============================================================
# Module-specific data generation + testing
# ============================================================

def train_wavefolder(dcae, sources, z_T, device):
    """Train the Wavefolder module."""
    print("\n" + "=" * 60)
    print("TRAINING: Wavefolder")
    print("=" * 60)

    out_dir = OUTPUT_BASE / "latent_wavefolder"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_amounts = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    fold_norm = {f: (f - 1.0) / 7.0 for f in fold_amounts}  # normalize to ~[0, 1]

    data_z_dry, data_z_wet, data_fold = [], [], []
    count = 0

    for (wf, pitch), src in sources.items():
        for fold in fold_amounts:
            audio_wet = dsp_wavefold(src['audio'], fold)
            z_wet = encode_audio(dcae, audio_wet, device)
            fn = fold_norm[fold]

            # Static fold amount → constant per-frame
            data_z_dry.append(src['z'])
            data_z_wet.append(z_wet.cpu())
            data_fold.append(torch.full((z_T,), fn))

            count += 1
            if count % 28 == 0:
                print(f"    {count}/{len(sources) * len(fold_amounts)}")
                clear_memory()

    print(f"  Total: {count} pairs")

    model = LatentWavefolder(n_channels=8, latent_dim=16, cond_dim=64, n_blocks=4).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    tensors = {
        'inputs': [
            torch.stack(data_z_dry).to(device),
            torch.stack(data_fold).to(device),
        ],
        'target': torch.stack(data_z_wet).to(device),
    }
    model = train_module(model, tensors, device, epochs=400)

    torch.save({
        'model': model.state_dict(),
        'config': {'n_channels': 8, 'latent_dim': 16, 'cond_dim': 64, 'n_blocks': 4},
    }, str(out_dir / "latent_wavefolder.pt"))

    # Quick test: fold sweep on saw@220
    model.eval()
    src = sources[('saw', 220)]
    z_dry = src['z'].unsqueeze(0).to(device)
    _save_wav(out_dir / "dry.wav", src['audio'])

    for fold in [1.0, 2.0, 4.0, 8.0]:
        fn = (fold - 1.0) / 7.0
        fold_t = torch.full((1, z_T), fn, device=device)
        with torch.no_grad():
            z_wet = model(z_dry, fold_t)
        _save_wav(out_dir / f"fold_{fold:.0f}_pred.wav",
                  decode_latent(dcae, z_wet.squeeze(0), device))
        _save_wav(out_dir / f"fold_{fold:.0f}_gt.wav",
                  dsp_wavefold(src['audio'], fold))
        print(f"  fold={fold:.0f}: saved")

    print("  Wavefolder DONE")
    clear_memory()
    return model


def train_distortion(dcae, sources, z_T, device):
    """Train the Distortion module."""
    print("\n" + "=" * 60)
    print("TRAINING: Distortion")
    print("=" * 60)

    out_dir = OUTPUT_BASE / "latent_distortion"
    out_dir.mkdir(parents=True, exist_ok=True)

    drives = [1.0, 2.0, 4.0, 8.0, 15.0]
    tones = [1000, 3000, 8000]
    drive_norm = {d: (d - 1.0) / 14.0 for d in drives}
    tone_norm = {t: (np.log10(t) - np.log10(500)) / (np.log10(10000) - np.log10(500))
                 for t in tones}

    data_z_dry, data_z_wet, data_drive, data_tone = [], [], [], []
    count = 0

    for (wf, pitch), src in sources.items():
        for drive in drives:
            for tone in tones:
                audio_wet = dsp_distortion(src['audio'], drive, tone)
                z_wet = encode_audio(dcae, audio_wet, device)

                data_z_dry.append(src['z'])
                data_z_wet.append(z_wet.cpu())
                data_drive.append(torch.full((z_T,), drive_norm[drive]))
                data_tone.append(tone_norm[tone])

                count += 1
                if count % 30 == 0:
                    print(f"    {count}/{len(sources) * len(drives) * len(tones)}")
                    clear_memory()

    print(f"  Total: {count} pairs")

    model = LatentDistortion(n_channels=8, latent_dim=16, cond_dim=64, n_blocks=4).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    tensors = {
        'inputs': [
            torch.stack(data_z_dry).to(device),
            torch.stack(data_drive).to(device),
            torch.tensor(data_tone).float().to(device),
        ],
        'target': torch.stack(data_z_wet).to(device),
    }
    model = train_module(model, tensors, device, epochs=400)

    torch.save({
        'model': model.state_dict(),
        'config': {'n_channels': 8, 'latent_dim': 16, 'cond_dim': 64, 'n_blocks': 4},
    }, str(out_dir / "latent_distortion.pt"))

    # Test
    model.eval()
    src = sources[('saw', 220)]
    z_dry = src['z'].unsqueeze(0).to(device)
    _save_wav(out_dir / "dry.wav", src['audio'])

    for drive in [2.0, 8.0, 15.0]:
        dn = (drive - 1.0) / 14.0
        tn = tone_norm[3000]
        drive_t = torch.full((1, z_T), dn, device=device)
        tone_t = torch.tensor([tn]).float().to(device)
        with torch.no_grad():
            z_wet = model(z_dry, drive_t, tone_t)
        _save_wav(out_dir / f"drive_{drive:.0f}_pred.wav",
                  decode_latent(dcae, z_wet.squeeze(0), device))
        _save_wav(out_dir / f"drive_{drive:.0f}_gt.wav",
                  dsp_distortion(src['audio'], drive, 3000))
        print(f"  drive={drive:.0f}: saved")

    print("  Distortion DONE")
    clear_memory()
    return model


def train_ringmod(dcae, sources, z_T, device):
    """Train the Ring Modulator module."""
    print("\n" + "=" * 60)
    print("TRAINING: Ring Modulator")
    print("=" * 60)

    out_dir = OUTPUT_BASE / "latent_ringmod"
    out_dir.mkdir(parents=True, exist_ok=True)

    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    source_keys = list(sources.keys())

    # Pair each source with a few modulators at different pitches
    pairs = []
    for wf in ['saw', 'square', 'triangle']:
        pairs.append(((wf, 220), ('sine', 110)))
        pairs.append(((wf, 220), ('sine', 330)))
        pairs.append(((wf, 220), ('sine', 440)))
        pairs.append(((wf, 110), ('sine', 220)))

    # Pre-encode sine modulators
    sine_sources = {}
    for pitch in [110, 220, 330, 440]:
        audio = generate_sine(pitch)
        z = encode_audio(dcae, audio, device)
        sine_sources[('sine', pitch)] = {'audio': audio, 'z': z.cpu()}
        print(f"    sine@{pitch}Hz: encoded")

    data_zc, data_zm, data_zw, data_depth = [], [], [], []
    count = 0

    for carrier_key, mod_key in pairs:
        carrier = sources[carrier_key]
        modulator = sine_sources[mod_key]

        for depth in depths:
            audio_wet = dsp_ringmod(carrier['audio'], modulator['audio'], depth)
            z_wet = encode_audio(dcae, audio_wet, device)

            data_zc.append(carrier['z'])
            data_zm.append(modulator['z'])
            data_zw.append(z_wet.cpu())
            data_depth.append(torch.full((z_T,), depth))

            count += 1
            if count % 20 == 0:
                print(f"    {count}/{len(pairs) * len(depths)}")
                clear_memory()

    print(f"  Total: {count} pairs")

    model = LatentRingMod(n_channels=8, latent_dim=16, cond_dim=64, n_blocks=4).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    tensors = {
        'inputs': [
            torch.stack(data_zc).to(device),
            torch.stack(data_zm).to(device),
            torch.stack(data_depth).to(device),
        ],
        'target': torch.stack(data_zw).to(device),
    }
    model = train_module(model, tensors, device, epochs=400)

    torch.save({
        'model': model.state_dict(),
        'config': {'n_channels': 8, 'latent_dim': 16, 'cond_dim': 64, 'n_blocks': 4},
    }, str(out_dir / "latent_ringmod.pt"))

    # Test
    model.eval()
    carrier = sources[('saw', 220)]
    modulator = sine_sources[('sine', 330)]
    zc = carrier['z'].unsqueeze(0).to(device)
    zm = modulator['z'].unsqueeze(0).to(device)

    _save_wav(out_dir / "carrier.wav", carrier['audio'])
    _save_wav(out_dir / "modulator.wav", modulator['audio'])

    for depth in [0.25, 0.5, 1.0]:
        dt = torch.full((1, z_T), depth, device=device)
        with torch.no_grad():
            z_wet = model(zc, zm, dt)
        _save_wav(out_dir / f"depth_{depth:.2f}_pred.wav",
                  decode_latent(dcae, z_wet.squeeze(0), device))
        _save_wav(out_dir / f"depth_{depth:.2f}_gt.wav",
                  dsp_ringmod(carrier['audio'], modulator['audio'], depth))
        print(f"  depth={depth:.2f}: saved")

    print("  Ring Modulator DONE")
    clear_memory()
    return model


def train_delay(dcae, sources, z_T, device):
    """Train the Delay module."""
    print("\n" + "=" * 60)
    print("TRAINING: Delay")
    print("=" * 60)

    out_dir = OUTPUT_BASE / "latent_delay"
    out_dir.mkdir(parents=True, exist_ok=True)

    times_ms = [100, 200, 300, 400, 500]
    feedbacks = [0.0, 0.3, 0.5, 0.7]
    mixes = [0.3, 0.5, 0.7]

    time_norm = {t: (t - 50) / 450 for t in times_ms}
    # feedback and mix are already in [0,1]

    data_z_dry, data_z_wet = [], []
    data_time, data_fb, data_mix = [], [], []
    count = 0
    total = len(sources) * len(times_ms) * len(feedbacks) * len(mixes)

    for (wf, pitch), src in sources.items():
        for time_ms in times_ms:
            for fb in feedbacks:
                for mx in mixes:
                    audio_wet = dsp_delay(src['audio'], time_ms, fb, mx)
                    z_wet = encode_audio(dcae, audio_wet, device)

                    data_z_dry.append(src['z'])
                    data_z_wet.append(z_wet.cpu())
                    data_time.append(time_norm[time_ms])
                    data_fb.append(fb)
                    data_mix.append(mx)

                    count += 1
                    if count % 60 == 0:
                        print(f"    {count}/{total}")
                        clear_memory()

    print(f"  Total: {count} pairs")

    model = LatentDelay(n_channels=8, latent_dim=16, cond_dim=128, n_blocks=6).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    tensors = {
        'inputs': [
            torch.stack(data_z_dry).to(device),
            torch.tensor(data_time).float().to(device),
            torch.tensor(data_fb).float().to(device),
            torch.tensor(data_mix).float().to(device),
        ],
        'target': torch.stack(data_z_wet).to(device),
    }
    model = train_module(model, tensors, device, epochs=400, batch_size=16)

    torch.save({
        'model': model.state_dict(),
        'config': {'n_channels': 8, 'latent_dim': 16, 'cond_dim': 128, 'n_blocks': 6},
    }, str(out_dir / "latent_delay.pt"))

    # Test
    model.eval()
    src = sources[('saw', 220)]
    z_dry = src['z'].unsqueeze(0).to(device)
    _save_wav(out_dir / "dry.wav", src['audio'])

    for time_ms, fb in [(200, 0.3), (300, 0.5), (500, 0.7)]:
        tn = (time_ms - 50) / 450
        with torch.no_grad():
            z_wet = model(
                z_dry,
                torch.tensor([tn]).float().to(device),
                torch.tensor([fb]).float().to(device),
                torch.tensor([0.5]).float().to(device),
            )
        _save_wav(out_dir / f"delay_{time_ms}ms_fb{fb:.1f}_pred.wav",
                  decode_latent(dcae, z_wet.squeeze(0), device))
        _save_wav(out_dir / f"delay_{time_ms}ms_fb{fb:.1f}_gt.wav",
                  dsp_delay(src['audio'], time_ms, fb, 0.5))
        print(f"  {time_ms}ms fb={fb:.1f}: saved")

    print("  Delay DONE")
    clear_memory()
    return model


def train_reverb(dcae, sources, z_T, device):
    """Train the Reverb module."""
    print("\n" + "=" * 60)
    print("TRAINING: Reverb")
    print("=" * 60)

    out_dir = OUTPUT_BASE / "latent_reverb"
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = [0.3, 0.5, 0.7, 1.0]
    decays = [0.3, 0.5, 0.7, 0.85]
    mixes = [0.2, 0.4, 0.6]

    data_z_dry, data_z_wet = [], []
    data_size, data_decay, data_mix = [], [], []
    count = 0
    total = len(sources) * len(sizes) * len(decays) * len(mixes)

    for (wf, pitch), src in sources.items():
        for size in sizes:
            for decay in decays:
                for mx in mixes:
                    audio_wet = dsp_reverb(src['audio'], size, decay, mx)
                    z_wet = encode_audio(dcae, audio_wet, device)

                    data_z_dry.append(src['z'])
                    data_z_wet.append(z_wet.cpu())
                    data_size.append(size)
                    data_decay.append(decay)
                    data_mix.append(mx)

                    count += 1
                    if count % 48 == 0:
                        print(f"    {count}/{total}")
                        clear_memory()

    print(f"  Total: {count} pairs")

    model = LatentReverb(n_channels=8, latent_dim=16, cond_dim=128, n_blocks=6).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    tensors = {
        'inputs': [
            torch.stack(data_z_dry).to(device),
            torch.tensor(data_size).float().to(device),
            torch.tensor(data_decay).float().to(device),
            torch.tensor(data_mix).float().to(device),
        ],
        'target': torch.stack(data_z_wet).to(device),
    }
    model = train_module(model, tensors, device, epochs=400, batch_size=16)

    torch.save({
        'model': model.state_dict(),
        'config': {'n_channels': 8, 'latent_dim': 16, 'cond_dim': 128, 'n_blocks': 6},
    }, str(out_dir / "latent_reverb.pt"))

    # Test
    model.eval()
    src = sources[('saw', 220)]
    z_dry = src['z'].unsqueeze(0).to(device)
    _save_wav(out_dir / "dry.wav", src['audio'])

    for size, decay in [(0.3, 0.5), (0.7, 0.7), (1.0, 0.85)]:
        with torch.no_grad():
            z_wet = model(
                z_dry,
                torch.tensor([size]).float().to(device),
                torch.tensor([decay]).float().to(device),
                torch.tensor([0.5]).float().to(device),
            )
        label = f"size{size:.1f}_decay{decay:.2f}"
        _save_wav(out_dir / f"{label}_pred.wav",
                  decode_latent(dcae, z_wet.squeeze(0), device))
        _save_wav(out_dir / f"{label}_gt.wav",
                  dsp_reverb(src['audio'], size, decay, 0.5))
        print(f"  {label}: saved")

    print("  Reverb DONE")
    clear_memory()
    return model


def train_chorus(dcae, sources, z_T, device):
    """Train the Chorus module."""
    print("\n" + "=" * 60)
    print("TRAINING: Chorus")
    print("=" * 60)

    out_dir = OUTPUT_BASE / "latent_chorus"
    out_dir.mkdir(parents=True, exist_ok=True)

    rates = [0.5, 1.0, 2.0, 3.0, 5.0]
    depths_ms = [2.0, 5.0, 8.0]
    mixes = [0.3, 0.5, 0.7]

    rate_norm = {r: r / 5.0 for r in rates}
    depth_norm = {d: d / 10.0 for d in depths_ms}

    data_z_dry, data_z_wet = [], []
    data_rate, data_depth, data_mix = [], [], []
    count = 0
    total = len(sources) * len(rates) * len(depths_ms) * len(mixes)

    for (wf, pitch), src in sources.items():
        for rate in rates:
            for depth in depths_ms:
                for mx in mixes:
                    audio_wet = dsp_chorus(src['audio'], rate, depth, mx)
                    z_wet = encode_audio(dcae, audio_wet, device)

                    data_z_dry.append(src['z'])
                    data_z_wet.append(z_wet.cpu())
                    data_rate.append(rate_norm[rate])
                    data_depth.append(depth_norm[depth])
                    data_mix.append(mx)

                    count += 1
                    if count % 45 == 0:
                        print(f"    {count}/{total}")
                        clear_memory()

    print(f"  Total: {count} pairs")

    model = LatentChorus(n_channels=8, latent_dim=16, cond_dim=128, n_blocks=6).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    tensors = {
        'inputs': [
            torch.stack(data_z_dry).to(device),
            torch.tensor(data_rate).float().to(device),
            torch.tensor(data_depth).float().to(device),
            torch.tensor(data_mix).float().to(device),
        ],
        'target': torch.stack(data_z_wet).to(device),
    }
    model = train_module(model, tensors, device, epochs=400, batch_size=16)

    torch.save({
        'model': model.state_dict(),
        'config': {'n_channels': 8, 'latent_dim': 16, 'cond_dim': 128, 'n_blocks': 6},
    }, str(out_dir / "latent_chorus.pt"))

    # Test
    model.eval()
    src = sources[('saw', 220)]
    z_dry = src['z'].unsqueeze(0).to(device)
    _save_wav(out_dir / "dry.wav", src['audio'])

    for rate, depth in [(1.0, 5.0), (2.0, 5.0), (3.0, 8.0)]:
        rn = rate / 5.0
        dn = depth / 10.0
        with torch.no_grad():
            z_wet = model(
                z_dry,
                torch.tensor([rn]).float().to(device),
                torch.tensor([dn]).float().to(device),
                torch.tensor([0.5]).float().to(device),
            )
        label = f"rate{rate:.0f}_depth{depth:.0f}"
        _save_wav(out_dir / f"{label}_pred.wav",
                  decode_latent(dcae, z_wet.squeeze(0), device))
        _save_wav(out_dir / f"{label}_gt.wav",
                  dsp_chorus(src['audio'], rate, depth, 0.5))
        print(f"  {label}: saved")

    print("  Chorus DONE")
    clear_memory()
    return model


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("LATENT MODULAR SYNTH — Train All Modules")
    print("=" * 60)
    print("\nTraining 6 transform modules sequentially.")
    print("DCAE loaded once, shared across all training runs.\n")

    device = 'cuda'

    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_PATH,
        vocoder_checkpoint_path=VOCODER_PATH,
    ).to(device)
    dcae.eval()
    print("DCAE loaded!\n")

    # Shared source encoding
    sources, z_T = encode_sources(dcae, device)
    clear_memory()

    print(f"\n{'=' * 60}")
    print(f"Sources: {len(sources)}, z frames: T={z_T}")
    print(f"{'=' * 60}\n")

    # Train each module (skip already-trained ones)
    skip_done = os.environ.get('SKIP_DONE', '0') == '1'
    if not skip_done:
        train_wavefolder(dcae, sources, z_T, device)
        train_distortion(dcae, sources, z_T, device)
        train_ringmod(dcae, sources, z_T, device)
    else:
        print("Skipping wavefolder, distortion, ringmod (already done)")
    train_delay(dcae, sources, z_T, device)
    train_reverb(dcae, sources, z_T, device)
    train_chorus(dcae, sources, z_T, device)

    print("\n" + "=" * 60)
    print("ALL MODULES TRAINED")
    print("=" * 60)
    print("\nCheckpoints saved to test_outputs/latent_*/")
    print("\nComplete module inventory:")
    print("  Sources:    VCO (oscillator)")
    print("  Filters:    VCF (static), VCF temporal (envelope/LFO)")
    print("  Amplifiers: VCA (ADSR/tremolo)")
    print("  Mixing:     Mixer (blend two sources)")
    print("  Distortion: Wavefolder, Distortion/Saturation")
    print("  Modulation: Ring Modulator")
    print("  Time-based: Delay, Reverb, Chorus")
    print("  Control:    Envelope, LFO, S&H, Slew, Quantizer (math only)")
    print("\nAll modules share the ModSignal routing framework.")
    print("Same envelope → any module input via .route(amount, base)")


if __name__ == "__main__":
    main()
