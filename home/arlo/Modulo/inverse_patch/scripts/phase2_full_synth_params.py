#!/usr/bin/env python3
"""
Phase 2: Full Synth Parameters Latent Oscillator

Extends Phase 1 to handle full synth parameter set:
- Pitch (continuous)
- Cutoff (continuous)
- Resonance (continuous)
- Attack, Decay, Sustain, Release (ADSR envelope)
- Waveform type (saw, square, triangle, sine)

Training: 4^7 = 16384 would be too many, so we use Latin Hypercube Sampling
to get good coverage with ~500 samples.
"""

import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import signal
from scipy.stats import qmc

sys.stdout.reconfigure(line_buffering=True)

# Memory constraints
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
torch.backends.cudnn.benchmark = False

SAMPLE_RATE = 44100
DURATION = 2.0

DCAE_PATH = "/home/arlo/Data/ACE-Step"
if DCAE_PATH not in sys.path:
    sys.path.insert(0, DCAE_PATH)

DEFAULT_DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
DEFAULT_VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

from acestep.music_dcae.music_dcae_pipeline import MusicDCAE


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Parameter ranges
PARAM_RANGES = {
    'pitch': (110, 880),        # Hz (A2 to A5)
    'cutoff': (200, 8000),      # Hz
    'resonance': (0.5, 10.0),   # Q factor
    'attack': (0.001, 0.5),     # seconds
    'decay': (0.01, 0.5),       # seconds
    'sustain': (0.2, 1.0),      # level (0-1)
    'release': (0.05, 1.0),     # seconds
    'waveform': (0, 3),         # 0=saw, 1=square, 2=triangle, 3=sine
}

WAVEFORM_NAMES = ['saw', 'square', 'triangle', 'sine']


def generate_waveform(waveform_type: int, freq: float, duration: float, sr: int) -> np.ndarray:
    """Generate basic waveform."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    if waveform_type == 0:  # Saw
        signal_out = np.zeros_like(t)
        for h in range(1, 30):
            f = freq * h
            if f > sr / 2:
                break
            signal_out += ((-1) ** h) * np.sin(2 * np.pi * f * t) / h
        return signal_out

    elif waveform_type == 1:  # Square
        signal_out = np.zeros_like(t)
        for h in range(1, 30, 2):  # Odd harmonics only
            f = freq * h
            if f > sr / 2:
                break
            signal_out += np.sin(2 * np.pi * f * t) / h
        return signal_out

    elif waveform_type == 2:  # Triangle
        signal_out = np.zeros_like(t)
        for h in range(1, 30, 2):  # Odd harmonics
            f = freq * h
            if f > sr / 2:
                break
            signal_out += ((-1) ** ((h-1)//2)) * np.sin(2 * np.pi * f * t) / (h * h)
        return signal_out

    else:  # Sine
        return np.sin(2 * np.pi * freq * t)


def apply_resonant_filter(audio: np.ndarray, cutoff: float, resonance: float, sr: int) -> np.ndarray:
    """Apply resonant lowpass filter."""
    nyq = sr / 2
    cutoff_norm = min(cutoff / nyq, 0.99)

    # State variable filter approximation with resonance
    # Higher Q = more resonance peak at cutoff
    sos = signal.butter(2, cutoff_norm, btype='low', output='sos')

    # Add resonance by boosting near cutoff
    if resonance > 1.0:
        # Create a peak filter at cutoff
        bandwidth = cutoff / resonance
        bw_norm = min(bandwidth / nyq, 0.5)
        if cutoff_norm > 0.01:
            try:
                sos_peak = signal.iirpeak(cutoff_norm, resonance, output='sos')
                audio_peaked = signal.sosfilt(sos_peak, audio)
                audio = audio + (audio_peaked - audio) * min(resonance / 10, 0.8)
            except:
                pass  # Skip peak if params are invalid

    return signal.sosfilt(sos, audio).astype(np.float32)


def apply_adsr_envelope(audio: np.ndarray, attack: float, decay: float,
                        sustain: float, release: float, sr: int) -> np.ndarray:
    """Apply ADSR envelope."""
    n_samples = len(audio)
    t = np.arange(n_samples) / sr

    attack_samples = int(attack * sr)
    decay_samples = int(decay * sr)
    release_samples = int(release * sr)

    env = np.ones(n_samples, dtype=np.float32)

    # Attack
    if attack_samples > 0:
        env[:min(attack_samples, n_samples)] = np.linspace(0, 1, min(attack_samples, n_samples))

    # Decay
    decay_start = attack_samples
    decay_end = min(decay_start + decay_samples, n_samples)
    if decay_end > decay_start:
        env[decay_start:decay_end] = np.linspace(1, sustain, decay_end - decay_start)

    # Sustain
    sustain_end = max(0, n_samples - release_samples)
    if sustain_end > decay_end:
        env[decay_end:sustain_end] = sustain

    # Release
    if release_samples > 0 and sustain_end < n_samples:
        env[sustain_end:] = np.linspace(sustain, 0, n_samples - sustain_end)

    return audio * env


def generate_synth_sound(pitch: float, cutoff: float, resonance: float,
                         attack: float, decay: float, sustain: float, release: float,
                         waveform: int, duration: float = DURATION, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a complete synth sound with all parameters."""

    # Generate base waveform
    audio = generate_waveform(waveform, pitch, duration, sr)

    # Apply resonant filter
    audio = apply_resonant_filter(audio, cutoff, resonance, sr)

    # Apply ADSR envelope
    audio = apply_adsr_envelope(audio, attack, decay, sustain, release, sr)

    # Normalize
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.8

    return audio.astype(np.float32)


def normalize_params(params: Dict[str, float]) -> Dict[str, float]:
    """Normalize parameters to [0, 1] range."""
    normalized = {}
    for key, value in params.items():
        pmin, pmax = PARAM_RANGES[key]
        normalized[key] = (value - pmin) / (pmax - pmin)
    return normalized


def denormalize_params(params: Dict[str, float]) -> Dict[str, float]:
    """Denormalize parameters from [0, 1] range."""
    denormalized = {}
    for key, value in params.items():
        pmin, pmax = PARAM_RANGES[key]
        denormalized[key] = value * (pmax - pmin) + pmin
    return denormalized


class FullSynthOscillator(nn.Module):
    """Maps full synth params to latent z."""

    def __init__(self, n_params: int, n_channels: int, latent_dim: int, n_frames: int):
        super().__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.total_size = n_channels * latent_dim * n_frames

        # Larger network for more params
        self.net = nn.Sequential(
            nn.Linear(n_params, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, self.total_size)
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            params: [batch, n_params] all normalized to [0, 1]
        Returns:
            z: [batch, n_channels, latent_dim, n_frames]
        """
        z_flat = self.net(params)
        return z_flat.view(-1, self.n_channels, self.latent_dim, self.n_frames)


def load_dcae(device: str):
    """Load DCAE codec."""
    codec = MusicDCAE(
        source_sample_rate=SAMPLE_RATE,
        dcae_checkpoint_path=DEFAULT_DCAE_PATH,
        vocoder_checkpoint_path=DEFAULT_VOCODER_PATH,
    ).to(device)
    codec.eval()
    return codec


def encode_audio(codec, audio: np.ndarray, device='cuda'):
    """Encode audio to latent."""
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    audio_stereo = audio_tensor.expand(-1, 2, -1).to(device)
    audio_lengths = torch.tensor([audio_stereo.shape[-1]], device=device)

    with torch.no_grad():
        latent, _ = codec.encode(audio_stereo, audio_lengths=audio_lengths, sr=SAMPLE_RATE)

    return latent  # [1, n_channels, latent_dim, n_frames]


def decode_latent(codec, z, device='cuda'):
    """Decode latent to audio."""
    if z.dim() == 3:
        z = z.unsqueeze(0)

    audio_lengths = torch.tensor([int(DURATION * SAMPLE_RATE)], device=device)

    with torch.no_grad():
        sr, pred_wavs = codec.decode(z, audio_lengths=audio_lengths, sr=SAMPLE_RATE)

    return pred_wavs[0].mean(dim=0).cpu().numpy()


def generate_training_data(codec, device: str, n_samples: int = 512) -> Dict:
    """Generate training data using Latin Hypercube Sampling."""

    print(f"Generating {n_samples} training samples using LHS...")

    # Latin Hypercube Sampling for 8 parameters
    sampler = qmc.LatinHypercube(d=8, seed=42)
    lhs_samples = sampler.random(n=n_samples)

    data = {
        'z': [],
        'params': [],  # [n_samples, 8] normalized
        'params_raw': [],  # For reference
    }

    param_names = ['pitch', 'cutoff', 'resonance', 'attack', 'decay', 'sustain', 'release', 'waveform']

    for i, sample in enumerate(lhs_samples):
        if i % 50 == 0:
            print(f"  Sample {i}/{n_samples}...")
            clear_memory()

        # Denormalize to get actual values
        params_raw = {}
        for j, name in enumerate(param_names):
            pmin, pmax = PARAM_RANGES[name]
            if name == 'waveform':
                # Discrete: 0, 1, 2, or 3
                params_raw[name] = int(sample[j] * 3.99)
            else:
                params_raw[name] = sample[j] * (pmax - pmin) + pmin

        # Generate audio
        audio = generate_synth_sound(
            pitch=params_raw['pitch'],
            cutoff=params_raw['cutoff'],
            resonance=params_raw['resonance'],
            attack=params_raw['attack'],
            decay=params_raw['decay'],
            sustain=params_raw['sustain'],
            release=params_raw['release'],
            waveform=int(params_raw['waveform'])
        )

        # Encode
        z = encode_audio(codec, audio, device)

        # Store
        data['z'].append(z.cpu())
        data['params'].append(torch.tensor(sample, dtype=torch.float32))
        data['params_raw'].append(params_raw)

    # Stack
    data['z'] = torch.cat(data['z'], dim=0)
    data['params'] = torch.stack(data['params'], dim=0)

    print(f"  z shape: {data['z'].shape}")
    print(f"  params shape: {data['params'].shape}")

    return data


def train_oscillator(data: Dict, device: str, n_epochs: int = 1000) -> FullSynthOscillator:
    """Train the full synth oscillator."""

    n_channels = data['z'].shape[1]
    latent_dim = data['z'].shape[2]
    n_frames = data['z'].shape[3]
    n_params = data['params'].shape[1]

    print(f"Latent shape: [{n_channels}, {latent_dim}, {n_frames}]")
    print(f"Number of params: {n_params}")

    model = FullSynthOscillator(n_params, n_channels, latent_dim, n_frames).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    z_target = data['z'].to(device)
    params = data['params'].to(device)

    # Mini-batch training for larger dataset
    batch_size = 64
    n_batches = (len(params) + batch_size - 1) // batch_size

    print(f"\nTraining for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        # Shuffle
        perm = torch.randperm(len(params))

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, len(params))
            idx = perm[start:end]

            batch_params = params[idx]
            batch_z = z_target[idx]

            optimizer.zero_grad(set_to_none=True)

            z_pred = model(batch_params)
            loss = F.mse_loss(z_pred, batch_z)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:4d}: loss = {avg_loss:.6f}, lr = {scheduler.get_last_lr()[0]:.6f}")

        if epoch % 200 == 0:
            clear_memory()

    return model


def test_parameter_sweeps(model: FullSynthOscillator, codec, device: str, output_dir: Path):
    """Test sweeping each parameter independently."""

    print("\n" + "="*60)
    print("Parameter Sweep Tests")
    print("="*60)

    model.eval()

    # Base params (middle of each range)
    base_params = torch.tensor([
        0.5,  # pitch (middle)
        0.5,  # cutoff (middle)
        0.3,  # resonance (low-mid)
        0.2,  # attack (quick)
        0.3,  # decay (moderate)
        0.7,  # sustain (high)
        0.3,  # release (moderate)
        0.0,  # waveform (saw)
    ], dtype=torch.float32).to(device)

    param_names = ['pitch', 'cutoff', 'resonance', 'attack', 'decay', 'sustain', 'release', 'waveform']

    for param_idx, param_name in enumerate(param_names):
        print(f"\n  Sweeping {param_name}...")

        if param_name == 'waveform':
            values = [0.0, 0.33, 0.66, 1.0]  # saw, square, triangle, sine
            labels = ['saw', 'square', 'triangle', 'sine']
        else:
            values = [0.0, 0.25, 0.5, 0.75, 1.0]
            pmin, pmax = PARAM_RANGES[param_name]
            labels = [f"{pmin + v*(pmax-pmin):.2f}" for v in values]

        for val, label in zip(values, labels):
            params = base_params.clone()
            params[param_idx] = val

            with torch.no_grad():
                z = model(params.unsqueeze(0))
                audio = decode_latent(codec, z, device)

            filename = f"sweep_{param_name}_{label}.wav"
            torchaudio.save(str(output_dir / filename),
                          torch.from_numpy(audio).unsqueeze(0).float(), SAMPLE_RATE)

        print(f"    Saved {len(values)} files for {param_name}")


def test_waveform_variations(model: FullSynthOscillator, codec, device: str, output_dir: Path):
    """Test all waveform types with various settings."""

    print("\n" + "="*60)
    print("Waveform Comparison Tests")
    print("="*60)

    model.eval()

    # Test each waveform with same settings
    base_params = torch.tensor([
        0.3,   # pitch (A3 ~220Hz)
        0.4,   # cutoff
        0.2,   # resonance
        0.1,   # attack
        0.2,   # decay
        0.8,   # sustain
        0.3,   # release
        0.0,   # waveform (will vary)
    ], dtype=torch.float32).to(device)

    for wave_idx, wave_name in enumerate(WAVEFORM_NAMES):
        params = base_params.clone()
        params[7] = wave_idx / 3.0  # Normalize waveform to [0, 1]

        with torch.no_grad():
            z = model(params.unsqueeze(0))
            audio = decode_latent(codec, z, device)

        filename = f"waveform_{wave_name}.wav"
        torchaudio.save(str(output_dir / filename),
                       torch.from_numpy(audio).unsqueeze(0).float(), SAMPLE_RATE)
        print(f"  Saved {filename}")


def test_interpolation(model: FullSynthOscillator, codec, device: str, output_dir: Path):
    """Test smooth interpolation between parameter settings."""

    print("\n" + "="*60)
    print("Interpolation Tests")
    print("="*60)

    model.eval()

    # Interpolate from "dark bass" to "bright lead"
    dark_bass = torch.tensor([
        0.1,   # low pitch
        0.2,   # low cutoff
        0.5,   # medium resonance
        0.05,  # fast attack
        0.3,   # medium decay
        0.9,   # high sustain
        0.2,   # short release
        0.0,   # saw
    ], dtype=torch.float32).to(device)

    bright_lead = torch.tensor([
        0.8,   # high pitch
        0.9,   # high cutoff
        0.7,   # high resonance
        0.02,  # very fast attack
        0.1,   # short decay
        0.6,   # medium sustain
        0.5,   # medium release
        0.25,  # square
    ], dtype=torch.float32).to(device)

    print("  Dark bass → Bright lead interpolation:")
    for i, alpha in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        params = dark_bass * (1 - alpha) + bright_lead * alpha

        with torch.no_grad():
            z = model(params.unsqueeze(0))
            audio = decode_latent(codec, z, device)

        filename = f"interp_bass_to_lead_{int(alpha*100):03d}.wav"
        torchaudio.save(str(output_dir / filename),
                       torch.from_numpy(audio).unsqueeze(0).float(), SAMPLE_RATE)
    print("    Saved 5 interpolation files")


def test_novel_presets(model: FullSynthOscillator, codec, device: str, output_dir: Path):
    """Generate classic synth preset-style sounds."""

    print("\n" + "="*60)
    print("Novel Preset Tests")
    print("="*60)

    model.eval()

    presets = {
        'acid_bass': [0.2, 0.3, 0.9, 0.01, 0.2, 0.4, 0.1, 0.0],      # Saw, high reso
        'pad': [0.4, 0.5, 0.2, 0.8, 0.3, 0.9, 0.8, 0.75],             # Sine, slow attack
        'pluck': [0.5, 0.6, 0.4, 0.01, 0.05, 0.2, 0.3, 0.5],          # Triangle, fast decay
        'brass': [0.3, 0.7, 0.3, 0.1, 0.15, 0.85, 0.2, 0.0],          # Saw, medium attack
        'key': [0.6, 0.4, 0.2, 0.01, 0.3, 0.3, 0.5, 0.25],            # Square, plucky
        'sub_bass': [0.05, 0.15, 0.1, 0.02, 0.1, 0.95, 0.3, 1.0],     # Sine, very low
    }

    for name, params_list in presets.items():
        params = torch.tensor(params_list, dtype=torch.float32).to(device)

        with torch.no_grad():
            z = model(params.unsqueeze(0))
            audio = decode_latent(codec, z, device)

        filename = f"preset_{name}.wav"
        torchaudio.save(str(output_dir / filename),
                       torch.from_numpy(audio).unsqueeze(0).float(), SAMPLE_RATE)
        print(f"  Saved {filename}")


def main():
    print("="*60)
    print("Phase 2: Full Synth Parameters Latent Oscillator")
    print("="*60)

    # Setup
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.cuda.empty_cache()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/test_outputs/phase2_full_synth")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load codec
    print("\nLoading DCAE...")
    codec = load_dcae(device)
    print("DCAE loaded!")

    # Generate training data
    print("\n" + "="*60)
    print("Generating Training Data")
    print("="*60)
    data = generate_training_data(codec, device, n_samples=512)

    # Train
    print("\n" + "="*60)
    print("Training Full Synth Oscillator")
    print("="*60)
    model = train_oscillator(data, device, n_epochs=1000)

    # Save model
    torch.save({
        'model': model.state_dict(),
        'n_params': 8,
        'n_channels': data['z'].shape[1],
        'latent_dim': data['z'].shape[2],
        'n_frames': data['z'].shape[3],
        'param_ranges': PARAM_RANGES,
    }, str(output_dir / "full_synth_oscillator.pt"))
    print(f"\nSaved model to {output_dir / 'full_synth_oscillator.pt'}")

    # Run tests
    test_parameter_sweeps(model, codec, device, output_dir)
    clear_memory()

    test_waveform_variations(model, codec, device, output_dir)
    clear_memory()

    test_interpolation(model, codec, device, output_dir)
    clear_memory()

    test_novel_presets(model, codec, device, output_dir)

    print("\n" + "="*60)
    print("PHASE 2 COMPLETE")
    print("="*60)
    print(f"Outputs saved to: {output_dir}")
    print("\nParameters tested:")
    print("  - Pitch (110-880 Hz)")
    print("  - Cutoff (200-8000 Hz)")
    print("  - Resonance (0.5-10 Q)")
    print("  - Attack (1ms-500ms)")
    print("  - Decay (10ms-500ms)")
    print("  - Sustain (0.2-1.0)")
    print("  - Release (50ms-1s)")
    print("  - Waveform (saw/square/triangle/sine)")
    print("\nWhat to listen for:")
    print("  1. sweep_* files: Each param changes independently")
    print("  2. waveform_* files: Distinct timbres for each waveform")
    print("  3. interp_* files: Smooth transitions between sounds")
    print("  4. preset_* files: Recognizable synth sounds")


if __name__ == "__main__":
    main()
