#!/usr/bin/env python3
"""Render octave pairs to audio for listening."""

import os
import sys
import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/dø')


def load_dcae(device: str = 'cuda'):
    from do.pipeline_do import DoTrainComponents
    components = DoTrainComponents(
        checkpoint_dir="/home/arlo/Data/ACE-Step/checkpoints",
        device_id=int(device.split(':')[-1]) if ':' in device else 0
    )
    components.load_dcae()
    return components.music_dcae


def frame_to_latent_idx(frame: int, hop_size: int = 512, latent_hop: int = 2048) -> int:
    """Convert f0 frame index to latent time index."""
    # f0 is at hop_size (512), latent is at latent_hop (2048)
    # latent_idx = frame * hop_size / latent_hop
    return int(frame * hop_size / latent_hop)


@torch.no_grad()
def render_pairs(
    pairs_json: str = '/home/arlo/Data/pitchshift/v7/octave_pairs.json',
    output_dir: str = '/tmp/octave_pairs_audio',
    max_pairs: int = 22,
    device: str = 'cuda',
    context_frames: int = 50,  # Extra frames before/after segment
):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading DCAE...")
    dcae = load_dcae(device)

    print(f"Loading pairs from {pairs_json}...")
    with open(pairs_json) as f:
        data = json.load(f)

    pairs = data['pairs'][:max_pairs]
    print(f"Rendering {len(pairs)} pairs...")

    for i, pair in enumerate(tqdm(pairs)):
        low = pair['low']
        high = pair['high']
        match_pct = pair['match_ratio'] * 100
        same_file = pair.get('same_file', False)

        # Load full latents
        low_latent = torch.load(low['latent_path'], map_location='cpu', weights_only=False)
        high_latent = torch.load(high['latent_path'], map_location='cpu', weights_only=False)

        # Handle dict format
        if isinstance(low_latent, dict):
            low_latent = low_latent.get('latents', low_latent.get('latent'))
        if isinstance(high_latent, dict):
            high_latent = high_latent.get('latents', high_latent.get('latent'))

        # Add batch dim if needed
        if low_latent.dim() == 3:
            low_latent = low_latent.unsqueeze(0)
        if high_latent.dim() == 3:
            high_latent = high_latent.unsqueeze(0)

        # Extract segment from latent based on frame indices
        # Convert f0 frames to latent time indices
        low_start = frame_to_latent_idx(low['start_frame']) - context_frames
        low_end = frame_to_latent_idx(low['end_frame']) + context_frames
        high_start = frame_to_latent_idx(high['start_frame']) - context_frames
        high_end = frame_to_latent_idx(high['end_frame']) + context_frames

        # Clamp to valid range
        low_start = max(0, low_start)
        low_end = min(low_latent.shape[-1], low_end)
        high_start = max(0, high_start)
        high_end = min(high_latent.shape[-1], high_end)

        # Extract segments
        low_segment = low_latent[:, :, :, low_start:low_end]
        high_segment = high_latent[:, :, :, high_start:high_end]

        # Decode segments
        low_segment = low_segment.to(device)
        high_segment = high_segment.to(device)

        decode_out = dcae.decode(low_segment)
        if isinstance(decode_out, tuple):
            sr, wavs = decode_out
            low_audio = wavs[0].cpu()
        else:
            sr = 44100
            low_audio = decode_out[0].cpu()

        decode_out = dcae.decode(high_segment)
        if isinstance(decode_out, tuple):
            _, wavs = decode_out
            high_audio = wavs[0].cpu()
        else:
            high_audio = decode_out[0].cpu()

        # Normalize
        low_audio = low_audio / (low_audio.abs().max() + 1e-8) * 0.9
        high_audio = high_audio / (high_audio.abs().max() + 1e-8) * 0.9

        # Save
        same_str = "_SAMEFILE" if same_file else ""
        prefix = f"pair{i:02d}_match{match_pct:.0f}pct{same_str}"
        low_name = Path(low['audio_path']).stem[:20]
        high_name = Path(high['audio_path']).stem[:20]

        torchaudio.save(
            f"{output_dir}/{prefix}_LOW_midi{low['median_midi']:.0f}_{low_name}.wav",
            low_audio, sr
        )
        torchaudio.save(
            f"{output_dir}/{prefix}_HIGH_midi{high['median_midi']:.0f}_{high_name}.wav",
            high_audio, sr
        )

    print(f"\nSaved to: {output_dir}")
    print(f"Each pair has LOW and HIGH (octave up) segment versions")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=str, default='/home/arlo/Data/pitchshift/v7/octave_pairs.json')
    parser.add_argument('--output', type=str, default='/tmp/octave_pairs_audio')
    parser.add_argument('--max_pairs', type=int, default=22)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    render_pairs(args.pairs, args.output, args.max_pairs, args.device)
