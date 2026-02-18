#!/usr/bin/env python3
"""
Operation-Based Synthesizer

z → operation parameters → sine program → additive synthesis → audio

Uses the discovered z → operation mapping to create interpretable audio.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
import os

sys.path.insert(0, '/home/arlo/Data/ACE-Step')


# ============================================================================
# Z → OPERATION PARAMETER EXTRACTION
# ============================================================================

# From our mapping analysis:
# Harmonic presence: dims 48-57 (higher = more harmonic)
# f0 control: dims 115-118
# Harmonic centroid: dims 86-87, 102-103 (positive), 66-72 (negative)
# Decay control: dim 22
# Energy: dims 98, 115, 86

class ZToOperations:
    """Extract operation parameters from z latent."""

    def __init__(self):
        # Dim groups from mapping analysis
        self.harmonic_presence_dims = list(range(48, 58))  # 48-57
        self.f0_dims = [115, 116, 117, 118]
        self.centroid_pos_dims = [86, 87, 102, 103]
        self.centroid_neg_dims = [66, 67, 68, 69, 70, 71, 72]
        self.decay_dims = [22]
        self.energy_dims = [98, 115, 86, 87, 118]

    def extract(self, z):
        """
        Extract operation parameters from z.

        Args:
            z: [B, C, H, T] or [B, 128, T] latent

        Returns:
            dict of operation parameters
        """
        if z.dim() == 4:
            B, C, H, T = z.shape
            z_flat = z.reshape(B, 128, T)
        else:
            z_flat = z

        # Average over time
        z_mean = z_flat.mean(dim=-1)  # [B, 128]

        params = {}

        # 1. Harmonic presence (0-1)
        harm_vals = z_mean[:, self.harmonic_presence_dims].mean(dim=-1)
        # Normalize: observed range is roughly -3 to 0
        params['harmonic_presence'] = torch.sigmoid(harm_vals + 1.5)

        # 2. f0 (fundamental frequency, Hz)
        f0_vals = z_mean[:, self.f0_dims].mean(dim=-1)
        # Map to frequency range 50-500 Hz
        params['f0'] = 50 + 450 * torch.sigmoid(f0_vals)

        # 3. Harmonic centroid (controls partial distribution)
        centroid_pos = z_mean[:, self.centroid_pos_dims].mean(dim=-1)
        centroid_neg = z_mean[:, self.centroid_neg_dims].mean(dim=-1)
        centroid_val = centroid_pos - centroid_neg
        # Map to 1-8 (1 = fundamental dominant, 8 = upper partials dominant)
        params['harmonic_centroid'] = 1 + 7 * torch.sigmoid(centroid_val)

        # 4. Decay rate (0.01 - 1.0)
        decay_vals = z_mean[:, self.decay_dims].mean(dim=-1)
        params['decay_rate'] = 0.01 + 0.99 * torch.sigmoid(decay_vals)

        # 5. Overall energy
        energy_vals = z_mean[:, self.energy_dims].mean(dim=-1)
        params['energy'] = torch.sigmoid(energy_vals + 1)

        # 6. Number of harmonics (1-16)
        n_harm = (params['harmonic_presence'] * 15 + 1).int()
        params['n_harmonics'] = n_harm

        return params


# ============================================================================
# OPERATION → SINES
# ============================================================================

class OperationsToSines:
    """Convert operation parameters to sine parameters."""

    def __init__(self, n_sines=64, sr=44100):
        self.n_sines = n_sines
        self.sr = sr

    def generate(self, params, n_frames=128):
        """
        Generate (freqs, amps) from operation parameters.

        Args:
            params: dict from ZToOperations
            n_frames: number of time frames

        Returns:
            freqs: [B, T, n_sines] Hz
            amps: [B, T, n_sines]
        """
        B = params['f0'].shape[0]
        device = params['f0'].device

        freqs = torch.zeros(B, n_frames, self.n_sines, device=device)
        amps = torch.zeros(B, n_frames, self.n_sines, device=device)

        for b in range(B):
            f0 = params['f0'][b].item()
            n_harm = int(params['n_harmonics'][b].item())
            centroid = params['harmonic_centroid'][b].item()
            decay = params['decay_rate'][b].item()
            energy = params['energy'][b].item()

            # Generate harmonic series
            for h in range(min(n_harm, self.n_sines)):
                partial = h + 1  # 1, 2, 3, ...
                freq = f0 * partial

                # Amplitude: exponential decay with partial number, centered on centroid
                # High centroid = upper partials stronger
                amp_base = np.exp(-0.5 * ((partial - centroid) / 3) ** 2)
                amp_base *= energy

                # Time decay
                t = torch.linspace(0, 1, n_frames, device=device)
                amp_envelope = amp_base * torch.exp(-decay * t * 10)

                freqs[b, :, h] = freq
                amps[b, :, h] = amp_envelope

        return freqs, amps


# ============================================================================
# ADDITIVE SYNTHESIZER
# ============================================================================

class AdditiveSynth:
    """Generate audio from (freqs, amps)."""

    def __init__(self, sr=44100, hop_length=512):
        self.sr = sr
        self.hop_length = hop_length

    def synthesize(self, freqs, amps, phases=None):
        """
        Additive synthesis (vectorized).

        Args:
            freqs: [B, T, n_sines] Hz
            amps: [B, T, n_sines]
            phases: [B, n_sines] initial phases (optional)

        Returns:
            audio: [B, samples]
        """
        B, T, n_sines = freqs.shape
        n_samples = T * self.hop_length
        device = freqs.device

        # Upsample to sample rate
        freqs_up = F.interpolate(
            freqs.permute(0, 2, 1),  # [B, n_sines, T]
            size=n_samples,
            mode='linear',
            align_corners=True
        ).permute(0, 2, 1)  # [B, samples, n_sines]

        amps_up = F.interpolate(
            amps.permute(0, 2, 1),
            size=n_samples,
            mode='linear',
            align_corners=True
        ).permute(0, 2, 1)

        # Initialize phases
        if phases is None:
            phases = torch.zeros(B, n_sines, device=device)

        # Vectorized phase accumulation
        # phase[t] = phase[0] + cumsum(2*pi*freq*dt)
        dt = 1.0 / self.sr
        phase_increments = 2 * np.pi * freqs_up * dt  # [B, samples, n_sines]
        phase_accum = torch.cumsum(phase_increments, dim=1)  # [B, samples, n_sines]
        phase_accum = phase_accum + phases.unsqueeze(1)  # Add initial phases

        # Generate sines and sum
        sines = amps_up * torch.sin(phase_accum)  # [B, samples, n_sines]
        audio = sines.sum(dim=-1)  # [B, samples]

        # Normalize
        audio = audio / (audio.abs().max(dim=-1, keepdim=True)[0] + 1e-8) * 0.9

        return audio


# ============================================================================
# FULL PIPELINE
# ============================================================================

class OperationSynthesizer:
    """Full z → audio pipeline via operations."""

    def __init__(self, sr=44100):
        self.sr = sr
        self.z_to_ops = ZToOperations()
        self.ops_to_sines = OperationsToSines(n_sines=64, sr=sr)
        self.synth = AdditiveSynth(sr=sr, hop_length=512)

    def synthesize(self, z, n_frames=128):
        """
        z → operations → sines → audio

        Args:
            z: [B, C, H, T] or [B, 128, T] latent

        Returns:
            audio: [B, samples]
            program: dict of operation parameters (interpretable!)
        """
        # Step 1: z → operations
        params = self.z_to_ops.extract(z)

        # Step 2: operations → sines
        freqs, amps = self.ops_to_sines.generate(params, n_frames=n_frames)

        # Step 3: sines → audio
        audio = self.synth.synthesize(freqs, amps)

        # Return both audio and the interpretable program
        program = {
            'f0': params['f0'].cpu().numpy().tolist(),
            'n_harmonics': params['n_harmonics'].cpu().numpy().tolist(),
            'harmonic_centroid': params['harmonic_centroid'].cpu().numpy().tolist(),
            'decay_rate': params['decay_rate'].cpu().numpy().tolist(),
            'energy': params['energy'].cpu().numpy().tolist(),
        }

        return audio, program


# ============================================================================
# TEST
# ============================================================================

def test_pipeline():
    """Test the full pipeline."""
    import soundfile as sf
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load DCAE for comparison
    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8",
        vocoder_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    )
    dcae.dcae.to(device).eval()
    dcae.vocoder.to(device).eval()

    # Create synthesizer
    synth = OperationSynthesizer(sr=44100)

    # Test with random latents
    print("\nTesting with random latents...")
    for i in range(3):
        z = torch.randn(1, 8, 16, 32, device=device) * 0.5

        # Operation-based synthesis
        audio_ops, program = synth.synthesize(z)

        print(f"\nSample {i}:")
        print(f"  f0: {program['f0'][0]:.1f} Hz")
        print(f"  n_harmonics: {program['n_harmonics'][0]}")
        print(f"  centroid: {program['harmonic_centroid'][0]:.2f}")
        print(f"  decay: {program['decay_rate'][0]:.3f}")
        print(f"  energy: {program['energy'][0]:.3f}")

        # Save audio
        out_path = f'/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/outputs/operation_synth_{i}.wav'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, audio_ops[0].cpu().numpy(), 44100)
        print(f"  Saved: {out_path}")

        # DCAE comparison
        with torch.no_grad():
            z_denorm = z / dcae.scale_factor + dcae.shift_factor
            mel = dcae.dcae.decoder(z_denorm).mean(dim=1)
            mel_scaled = mel * 0.5 + 0.5
            mel_scaled = mel_scaled * (dcae.max_mel_value - dcae.min_mel_value) + dcae.min_mel_value
            audio_dcae = dcae.vocoder.decode(mel_scaled).squeeze()
            audio_dcae = audio_dcae / (audio_dcae.abs().max() + 1e-8) * 0.9

        dcae_path = f'/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/outputs/operation_synth_{i}_dcae.wav'
        sf.write(dcae_path, audio_dcae.cpu().numpy(), 44100)
        print(f"  DCAE: {dcae_path}")

    print("\nDone! Compare the operation-based synthesis with DCAE output.")


def test_with_real_latent():
    """Test with a real latent from the dataset."""
    import soundfile as sf
    import orjson
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load a real latent
    manifest_path = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json'
    with open(manifest_path, 'rb') as f:
        manifest = orjson.loads(f.read())

    # Find a sample with latent
    z = None
    for entry in manifest['entries'][:50]:
        path = entry['path']
        if 'drum' in path.lower():
            continue
        try:
            data = torch.load(path, weights_only=True, map_location='cpu')
            lat_path = data.get('latent_path')
            if lat_path and os.path.exists(lat_path):
                lat_data = torch.load(lat_path, weights_only=True, map_location='cpu')
                z = lat_data.get('latents', lat_data)
                if z.dim() == 3:
                    z = z.unsqueeze(0)
                z = z.to(device)
                print(f"Loaded latent from: {path}")
                print(f"  z shape: {z.shape}")
                break
        except:
            continue

    if z is None:
        print("No latent found!")
        return

    # Load DCAE
    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8",
        vocoder_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    )
    dcae.dcae.to(device).eval()
    dcae.vocoder.to(device).eval()

    # Synthesize
    synth = OperationSynthesizer(sr=44100)
    audio_ops, program = synth.synthesize(z)

    print(f"\nExtracted program:")
    print(f"  f0: {program['f0'][0]:.1f} Hz")
    print(f"  n_harmonics: {program['n_harmonics'][0]}")
    print(f"  centroid: {program['harmonic_centroid'][0]:.2f}")
    print(f"  decay: {program['decay_rate'][0]:.3f}")
    print(f"  energy: {program['energy'][0]:.3f}")

    # Save
    out_path = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/outputs/operation_synth_real.wav'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, audio_ops[0].cpu().numpy(), 44100)
    print(f"Saved: {out_path}")

    # DCAE comparison
    with torch.no_grad():
        z_denorm = z / dcae.scale_factor + dcae.shift_factor
        mel = dcae.dcae.decoder(z_denorm).mean(dim=1)
        mel_scaled = mel * 0.5 + 0.5
        mel_scaled = mel_scaled * (dcae.max_mel_value - dcae.min_mel_value) + dcae.min_mel_value
        audio_dcae = dcae.vocoder.decode(mel_scaled).squeeze()
        audio_dcae = audio_dcae / (audio_dcae.abs().max() + 1e-8) * 0.9

    dcae_path = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/outputs/operation_synth_real_dcae.wav'
    sf.write(dcae_path, audio_dcae.cpu().numpy(), 44100)
    print(f"DCAE: {dcae_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'real':
        test_with_real_latent()
    else:
        test_pipeline()
