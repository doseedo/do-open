#!/usr/bin/env python3
"""
Evaluate trained sine mapper by actually synthesizing audio.

The analytical loss is fast, but the real test is: does it SOUND right?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from pathlib import Path
import soundfile as sf


# ============ Model classes (must match training exactly) ============
class ResBlock(nn.Module):
    """Residual block with post-norm."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class SparseSineMapper(nn.Module):
    """Maps DCAE latent [B, T, 128] → sparse sine parameters."""

    def __init__(
        self,
        frame_dim: int = 128,
        max_sines: int = 512,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.frame_dim = frame_dim
        self.max_sines = max_sines
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.GELU(),
            *[ResBlock(hidden_dim) for _ in range(n_blocks)],
        )

        # Separate heads
        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_sines),
        )
        self.amp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_sines),
        )
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_sines),
        )

    def forward(self, z: torch.Tensor) -> dict:
        h = self.encoder(z)
        freqs = torch.sigmoid(self.freq_head(h)) * self.nyquist
        amps = torch.sigmoid(self.amp_head(h))
        phases = torch.sigmoid(self.phase_head(h)) * 2 * np.pi
        return {'freqs': freqs, 'amps': amps, 'phases': phases}


class DCEASinePipeline(nn.Module):
    """Wrapper that matches the saved checkpoint structure."""
    def __init__(self, max_sines=512, hidden_dim=512, n_blocks=4, sample_rate=44100):
        super().__init__()
        self.mapper = SparseSineMapper(
            frame_dim=128,
            max_sines=max_sines,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            sample_rate=sample_rate,
        )


def load_trained_mapper(checkpoint_dir: str, device: str = 'cuda'):
    """Load the trained sine mapper."""
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Infer config from state dict shapes
    state_dict = checkpoint['model_state_dict']
    hidden_dim = state_dict['mapper.encoder.0.weight'].shape[0]  # 512
    max_sines = state_dict['mapper.freq_head.2.weight'].shape[0]  # 512

    # Count ResBlocks (keys like mapper.encoder.2.net.0.weight)
    n_blocks = sum(1 for k in state_dict.keys() if 'encoder' in k and '.net.0.weight' in k)

    config = {
        'max_sines': max_sines,
        'hidden_dim': hidden_dim,
        'n_blocks': n_blocks,
    }

    # Create pipeline and load weights
    pipeline = DCEASinePipeline(
        max_sines=max_sines,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
    ).to(device)

    pipeline.load_state_dict(state_dict)
    pipeline.eval()

    print(f"Loaded mapper from {checkpoint_path}")
    print(f"  Config: {config}")
    print(f"  Best loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    return pipeline.mapper, config


def synthesize_sines(freqs, amps, phases, n_samples, sample_rate=48000):
    """
    Synthesize audio from sine parameters with proper phase accumulation.

    Key insight: The model was trained with analytical spectrum loss that
    ignores phases. So we need to accumulate phase based on frequency
    to get continuous, non-clicky output.

    Args:
        freqs: [B, T, n_sines] frequencies in Hz
        amps: [B, T, n_sines] amplitudes
        phases: [B, T, n_sines] phases (IGNORED - we accumulate instead)
        n_samples: number of output samples
        sample_rate: audio sample rate

    Returns:
        audio: [B, n_samples]
    """
    # Move to CPU for memory efficiency
    freqs = freqs.cpu().float()
    amps = amps.cpu().float()

    B, T, N = freqs.shape
    samples_per_frame = n_samples // T

    # Initialize cumulative phase for each sine [B, N]
    cumulative_phase = torch.zeros(B, N)

    audio_chunks = []

    for frame_idx in range(T):
        f = freqs[:, frame_idx, :]  # [B, N] frequencies for this frame
        a = amps[:, frame_idx, :]   # [B, N] amplitudes for this frame

        # Time within this frame
        t = torch.arange(samples_per_frame).float() / sample_rate  # [samples_per_frame]

        # Phase increment per sample: 2π * freq / sample_rate
        # For the whole frame: 2π * freq * t + starting_phase
        phase_increment = 2 * np.pi * f  # [B, N] radians per second

        # Generate sine for this frame with accumulated phase
        # phase(t) = cumulative_phase + 2π * freq * t
        frame_phases = cumulative_phase.unsqueeze(-1) + phase_increment.unsqueeze(-1) * t.view(1, 1, -1)
        # frame_phases: [B, N, samples_per_frame]

        frame_sines = a.unsqueeze(-1) * torch.sin(frame_phases)  # [B, N, samples_per_frame]
        frame_audio = frame_sines.sum(dim=1)  # [B, samples_per_frame]

        audio_chunks.append(frame_audio)

        # Update cumulative phase for next frame
        frame_duration = samples_per_frame / sample_rate
        cumulative_phase = cumulative_phase + phase_increment * frame_duration
        # Keep phase in [0, 2π] to avoid numerical issues
        cumulative_phase = cumulative_phase % (2 * np.pi)

    audio = torch.cat(audio_chunks, dim=-1)  # [B, T * samples_per_frame]

    # Trim or pad to exact length
    if audio.shape[-1] > n_samples:
        audio = audio[..., :n_samples]
    elif audio.shape[-1] < n_samples:
        audio = F.pad(audio, (0, n_samples - audio.shape[-1]))

    return audio


def load_eval_samples(manifest_path: str, n_samples: int = 5, device: str = 'cuda'):
    """Load a few samples from the training data for evaluation."""
    import orjson
    import random

    with open(manifest_path, 'rb') as f:
        manifest = orjson.loads(f.read())

    # Get entries list
    entries = manifest.get('entries', [])
    if not entries:
        # Fallback: try as dict
        entries = [{'audio_path': v.get('audio_path'), 'latent_path': v.get('latent_path'), 'name': k}
                   for k, v in manifest.items() if isinstance(v, dict) and 'audio_path' in v]

    # Filter to entries with latents
    entries_with_latent = [e for e in entries if e.get('has_latent', True) and e.get('latent_path')]

    # Randomly sample to get variety
    random.shuffle(entries_with_latent)

    samples = []
    for entry in entries_with_latent[:n_samples * 5]:  # Try more in case some fail
        if len(samples) >= n_samples:
            break

        try:
            latent_path = entry['latent_path']
            audio_path = entry['audio_path']

            # Load latent
            latent = torch.load(latent_path, map_location='cpu', weights_only=False)
            if isinstance(latent, dict):
                latent = latent.get('latent', latent.get('z', list(latent.values())[0]))

            # Load audio
            audio, sr = sf.read(audio_path)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)  # mono
            audio = torch.from_numpy(audio).float()

            # Validate shapes
            if latent.dim() == 4:  # [1, 8, 16, T]
                latent = latent.squeeze(0)  # [8, 16, T]
            if latent.shape[0] == 8 and latent.shape[1] == 16:
                name = entry.get('name', os.path.basename(audio_path))
                samples.append({
                    'latent': latent,
                    'audio': audio,
                    'name': name,
                    'sr': sr,
                    'group': entry.get('group', 'unknown'),
                })
        except Exception as e:
            continue

    print(f"Loaded {len(samples)} evaluation samples")
    return samples


def evaluate(
    mapper,
    samples: list,
    output_dir: str = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/eval_audio',
    device: str = 'cuda',
    sample_rate: int = 44100,
):
    """
    Generate comparison audio files.

    For each sample:
    1. Load paired (latent, audio) from training data
    2. Map latent to sine params
    3. Synthesize audio from sines
    4. Compare to ground truth audio
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nEvaluating {len(samples)} samples...")
    print(f"Output dir: {output_dir}")

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1}/{len(samples)}: {sample['name'][:50]} ---")

        latent = sample['latent'].to(device)  # [8, 16, T]
        audio_gt = sample['audio']  # [n_samples]

        # Prepare latent for mapper: [B, T, 128]
        C, H, T = latent.shape
        z_flat = latent.permute(2, 0, 1).reshape(1, T, C * H)  # [1, T, 128]

        # Get sine parameters
        with torch.no_grad():
            params = mapper(z_flat)
            freqs = params['freqs']
            amps = params['amps']
            phases = params['phases']

        # Use actual sample rate from file
        actual_sr = sample.get('sr', sample_rate)

        # Synthesize from sines
        audio_sines = synthesize_sines(
            freqs, amps, phases,
            n_samples=len(audio_gt),
            sample_rate=actual_sr
        )

        # Normalize
        audio_gt_np = audio_gt.numpy()
        audio_sines_np = audio_sines[0].cpu().numpy()

        audio_gt_np = audio_gt_np / (np.abs(audio_gt_np).max() + 1e-8) * 0.9
        audio_sines_np = audio_sines_np / (np.abs(audio_sines_np).max() + 1e-8) * 0.9

        # Save audio files
        gt_path = os.path.join(output_dir, f"sample_{i:02d}_original.wav")
        sines_path = os.path.join(output_dir, f"sample_{i:02d}_sines.wav")

        sf.write(gt_path, audio_gt_np, actual_sr)
        sf.write(sines_path, audio_sines_np, actual_sr)

        # Stats
        n_active = (amps > 0.1).float().sum(dim=-1).mean().item()
        active_freqs = freqs[amps > 0.1]
        if len(active_freqs) > 0:
            freq_range = (active_freqs.min().item(), active_freqs.max().item())
        else:
            freq_range = (0, 0)

        print(f"  Original: {gt_path}")
        print(f"  Sines:    {sines_path}")
        print(f"  Active sines: {n_active:.0f} (amp > 0.1)")
        print(f"  Freq range: {freq_range[0]:.0f} - {freq_range[1]:.0f} Hz")

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nListen and compare in: {output_dir}")
    print("  - *_original.wav: Ground truth audio")
    print("  - *_sines.wav: Reconstructed from learned sines")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                       default='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/dcae_sparse_sines')
    parser.add_argument('--manifest', type=str,
                       default='/home/arlo/gcs-bucket/Manifests/unified_manifest.json')
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("="*60)
    print("SINE MAPPER AUDIO EVALUATION")
    print("="*60)

    print("\nLoading trained mapper...")
    mapper, config = load_trained_mapper(args.checkpoint, args.device)

    print("\nLoading evaluation samples...")
    samples = load_eval_samples(args.manifest, args.n_samples, args.device)

    if not samples:
        print("ERROR: No samples loaded. Check manifest path.")
        sys.exit(1)

    evaluate(mapper, samples, device=args.device)
