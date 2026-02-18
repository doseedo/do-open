#!/usr/bin/env python3
"""
Evaluate latent sine model - synthesize actual audio to listen.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import soundfile as sf
import orjson

sys.path.insert(0, "/home/arlo/Data/ACE-Step")
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

SAMPLE_RATE = 44100


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(x + self.net(x))


class LatentSineMapper(nn.Module):
    def __init__(self, frame_dim=128, max_sines=64, hidden_dim=256, n_blocks=3, freq_min=20.0, freq_max=16000.0):
        super().__init__()
        self.max_sines = max_sines
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.encoder = nn.Sequential(nn.Linear(frame_dim, hidden_dim), nn.GELU(), *[ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.freq_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Linear(hidden_dim//2, max_sines))
        self.amp_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Linear(hidden_dim//2, max_sines))

    def forward(self, z):
        h = self.encoder(z)
        freq_logits = self.freq_head(h)
        log_freq = torch.sigmoid(freq_logits) * (np.log10(self.freq_max) - np.log10(self.freq_min)) + np.log10(self.freq_min)
        freqs = 10 ** log_freq
        amps = torch.sigmoid(self.amp_head(h))
        return {'freqs': freqs, 'amps': amps}


def synthesize_sines(freqs, amps, n_samples, sample_rate=44100):
    """Synthesize audio from sine params with phase accumulation."""
    freqs = freqs.cpu().float()
    amps = amps.cpu().float()
    B, T, N = freqs.shape
    samples_per_frame = n_samples // T

    cumulative_phase = torch.zeros(B, N)
    audio_chunks = []

    for frame_idx in range(T):
        f = freqs[:, frame_idx, :]
        a = amps[:, frame_idx, :]
        t = torch.arange(samples_per_frame).float() / sample_rate
        phase_increment = 2 * np.pi * f
        frame_phases = cumulative_phase.unsqueeze(-1) + phase_increment.unsqueeze(-1) * t.view(1, 1, -1)
        frame_sines = a.unsqueeze(-1) * torch.sin(frame_phases)
        frame_audio = frame_sines.sum(dim=1)
        audio_chunks.append(frame_audio)
        frame_duration = samples_per_frame / sample_rate
        cumulative_phase = (cumulative_phase + phase_increment * frame_duration) % (2 * np.pi)

    audio = torch.cat(audio_chunks, dim=-1)
    if audio.shape[-1] > n_samples:
        audio = audio[..., :n_samples]
    elif audio.shape[-1] < n_samples:
        audio = F.pad(audio, (0, n_samples - audio.shape[-1]))
    return audio


def main():
    device = 'cuda'

    print("Loading DCAE...")
    dcae = MusicDCAE(dcae_checkpoint_path=DCAE_PATH, vocoder_checkpoint_path=VOCODER_PATH).to(device)
    dcae.eval()

    print("Loading mapper...")
    ckpt = torch.load('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/latent_sines/best_model.pt', map_location=device)
    config = ckpt['config']
    mapper = LatentSineMapper(max_sines=config['max_sines'], freq_min=config['freq_min'], freq_max=config['freq_max']).to(device)
    mapper.load_state_dict(ckpt['model_state_dict'])
    mapper.eval()
    print(f"Config: {config}")
    print(f"Best loss: {ckpt['best_loss']:.4f}")

    # Load samples
    with open('/home/arlo/gcs-bucket/Manifests/unified_manifest.json', 'rb') as f:
        manifest = orjson.loads(f.read())
    entries = [e for e in manifest['entries'] if e.get('has_latent')][:10]

    output_dir = Path('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/eval_audio_latent')
    output_dir.mkdir(exist_ok=True)

    for i, entry in enumerate(entries[:3]):
        print(f"\nSample {i}: {entry['audio_path'].split('/')[-1]}")

        # Load original audio
        audio_gt, sr = sf.read(entry['audio_path'])
        if audio_gt.ndim == 2:
            audio_gt = audio_gt.mean(axis=1)

        # Load latent
        latent = torch.load(entry['latent_path'], map_location='cpu', weights_only=False)
        if isinstance(latent, dict):
            latent = latent.get('latents', latent.get('latent', list(latent.values())[0]))
        if latent.dim() == 4:
            latent = latent.squeeze(0)

        C, H, T = latent.shape
        latent = latent.to(device)

        # Predict sine params
        z_flat = latent.permute(2, 0, 1).reshape(1, T, C * H)
        with torch.no_grad():
            params = mapper(z_flat)

        freqs = params['freqs']
        amps = params['amps']

        # Synthesize
        audio_sines = synthesize_sines(freqs, amps, len(audio_gt), sr)

        # Stats
        n_active = (amps > 0.1).float().sum(dim=-1).mean().item()
        active_freqs = freqs[amps > 0.1]
        if len(active_freqs) > 0:
            print(f"  Active sines: {n_active:.0f}")
            print(f"  Freq range: {active_freqs.min():.0f} - {active_freqs.max():.0f} Hz")
        else:
            print(f"  Active sines: 0 (all amps < 0.1)")

        # Normalize and save
        audio_gt_np = audio_gt / (np.abs(audio_gt).max() + 1e-8) * 0.9
        audio_sines_np = audio_sines[0].numpy()
        audio_sines_np = audio_sines_np / (np.abs(audio_sines_np).max() + 1e-8) * 0.9

        sf.write(output_dir / f'sample_{i:02d}_original.wav', audio_gt_np, sr)
        sf.write(output_dir / f'sample_{i:02d}_sines.wav', audio_sines_np, sr)
        print(f"  Saved to {output_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
