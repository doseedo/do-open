#!/usr/bin/env python3
"""
Inference with AutoVC-style Register Disentanglement

Usage:
1. For pitch-shifted audio:
   - Encode to DCAE latent
   - Extract content with trained model
   - Get register embedding for TARGET pitch (not source pitch)
   - Decode to corrected latent
   - Decode latent to audio

2. For listening tests:
   - Process segments with different register sources
   - Compare reconstruction quality
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/dø')

from models_autovc import AutoVCLatent


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def load_model(checkpoint_path, device='cuda'):
    """Load trained AutoVC model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = AutoVCLatent(
        hidden_dim=config.get('hidden_dim', 256),
        bottleneck_dim=config['bottleneck_dim'],
        register_dim=config['register_dim'],
        downsample=config.get('downsample', 8),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    return model, config


def load_register_embeddings(embeddings_path, device='cuda'):
    """Load precomputed register embeddings."""
    data = torch.load(embeddings_path, map_location=device, weights_only=False)

    # Handle different formats
    if isinstance(data['embeddings'], list):
        # List format (sorted by bin)
        embeddings = [e.to(device) for e in data['embeddings']]
        sorted_bins = data.get('sorted_bins')
    else:
        # Dict format (keyed by pitch bin)
        embeddings = {k: v.to(device) for k, v in data['embeddings'].items()}
        sorted_bins = sorted(embeddings.keys())

    bin_size = data.get('bin_size') or data.get('pitch_bin_size', 2.0)

    return {
        'embeddings': embeddings,
        'sorted_bins': sorted_bins,
        'bin_size': bin_size,
    }


def get_register_embedding_for_pitch(register_data, target_midi):
    """Get the register embedding for a target pitch."""
    bin_size = register_data['bin_size']
    embeddings = register_data['embeddings']
    sorted_bins = register_data['sorted_bins']

    target_bin = int(target_midi / bin_size)

    # Find closest bin in sorted_bins
    closest_bin = sorted_bins[0]
    min_dist = abs(sorted_bins[0] - target_bin)
    for b in sorted_bins:
        if abs(b - target_bin) < min_dist:
            min_dist = abs(b - target_bin)
            closest_bin = b

    # Handle list vs dict embeddings
    if isinstance(embeddings, list):
        idx = sorted_bins.index(closest_bin)
        return embeddings[idx]
    else:
        return embeddings[closest_bin]


@torch.no_grad()
def correct_pitch_shift(model, latent, target_register_emb):
    """
    Correct pitch-shifted audio by swapping register embedding.

    Args:
        model: Trained AutoVC model
        latent: [B, C, H, T] DCAE latent of pitch-shifted audio
        target_register_emb: [1, register_dim] Register embedding for target pitch

    Returns:
        corrected: [B, C, H, T] Corrected DCAE latent
    """
    B = latent.shape[0]

    # Extract content from pitch-shifted audio
    content = model.encode_content(latent)

    # Expand register embedding to batch size
    target_register_emb = target_register_emb.expand(B, -1)

    # Decode with target register
    corrected = model.decode(content, target_register_emb)

    return corrected


def process_audio_file(
    audio_path,
    model,
    register_data,
    dcae,
    source_pitch,
    target_pitch,
    device='cuda',
):
    """
    Process an audio file through the correction pipeline.

    Args:
        audio_path: Path to input audio
        model: Trained AutoVC model
        register_data: Precomputed register embeddings
        dcae: DCAE model for encode/decode
        source_pitch: MIDI pitch of the shifted audio
        target_pitch: MIDI pitch we want it to sound like

    Returns:
        corrected_audio: Tensor of corrected audio
    """
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    if sr != 44100:
        audio = torchaudio.functional.resample(audio, sr, 44100)

    # Encode to DCAE latent
    audio = audio.unsqueeze(0).to(device)  # [1, channels, samples]
    if audio.shape[1] == 1:
        audio = audio.repeat(1, 2, 1)  # Stereo

    latent = dcae.encode(audio)  # [1, C, H, T]

    # Get register embedding for target pitch
    target_register_emb = get_register_embedding_for_pitch(register_data, target_pitch)

    # Correct
    corrected_latent = correct_pitch_shift(model, latent, target_register_emb)

    # Decode
    sr_out, wavs = dcae.decode(corrected_latent)
    corrected_audio = wavs[0]

    return corrected_audio, sr_out


def generate_listening_test(
    model,
    register_data,
    segments_json,
    output_dir,
    dcae,
    num_samples=10,
    device='cuda',
):
    """
    Generate listening test samples.

    For each sample:
    1. Original segment decoded directly
    2. Reconstruction (same register)
    3. Register swapped to different pitch range
    """
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    # Load segments
    with open(segments_json) as f:
        data = json.load(f)

    segments = []
    for group_id, segs in data.get('segments_by_group', {}).items():
        for seg in segs:
            seg_len = seg['end_frame'] - seg['start_frame']
            if seg_len >= 64:
                segments.append({
                    'latent_path': fix_path(seg['latent_path']),
                    'start_frame': seg['start_frame'],
                    'end_frame': seg['end_frame'],
                    'median_midi': seg['median_midi'],
                })

    # Sample diverse pitches
    segments.sort(key=lambda s: s['median_midi'])
    step = len(segments) // num_samples
    selected = [segments[i * step] for i in range(num_samples)]

    latent_cache = {}

    print(f"Generating {num_samples} listening test samples...")

    for i, seg in enumerate(tqdm(selected)):
        # Load latent
        path = seg['latent_path']
        if path not in latent_cache:
            ld = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(ld, dict):
                lat = ld.get('latents', ld.get('latent'))
            else:
                lat = ld
            if lat.dim() == 4:
                lat = lat.squeeze(0)
            latent_cache[path] = lat

        latent = latent_cache[path]
        start, end = seg['start_frame'], min(seg['end_frame'], latent.shape[-1])
        segment = latent[:, :, start:end].unsqueeze(0).to(device)

        source_pitch = seg['median_midi']

        # 1. Original (direct decode)
        sr, wavs = dcae.decode(segment)
        original_audio = wavs[0]
        original_path = os.path.join(output_dir, f"{i:02d}_original_pitch{source_pitch:.0f}.wav")
        torchaudio.save(original_path, original_audio.cpu(), sr)

        # 2. Reconstruction (same register)
        source_register = get_register_embedding_for_pitch(register_data, source_pitch)
        content = model.encode_content(segment)
        reconstructed = model.decode(content, source_register)
        sr, wavs = dcae.decode(reconstructed)
        recon_path = os.path.join(output_dir, f"{i:02d}_reconstructed_pitch{source_pitch:.0f}.wav")
        torchaudio.save(recon_path, wavs[0].cpu(), sr)

        # 3. Register swapped (+12 semitones target)
        target_pitch = source_pitch + 12
        if target_pitch > 96:
            target_pitch = source_pitch - 12
        target_register = get_register_embedding_for_pitch(register_data, target_pitch)
        swapped = model.decode(content, target_register)
        sr, wavs = dcae.decode(swapped)
        swapped_path = os.path.join(output_dir, f"{i:02d}_swapped_to{target_pitch:.0f}.wav")
        torchaudio.save(swapped_path, wavs[0].cpu(), sr)

        print(f"  Sample {i}: pitch {source_pitch:.0f} → {target_pitch:.0f}")

    print(f"\nListening test saved to: {output_dir}")
    print("\nFiles:")
    print("  XX_original_pitchNN.wav - Original segment decoded directly")
    print("  XX_reconstructed_pitchNN.wav - Passed through model (same register)")
    print("  XX_swapped_toNN.wav - Content preserved, register from different pitch")


def main():
    parser = argparse.ArgumentParser(description="AutoVC Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--register_embeddings', type=str, help='Register embeddings file')
    parser.add_argument('--segments', type=str, help='Segments JSON for listening test')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='listening_test',
                        choices=['listening_test', 'correct'])

    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, config = load_model(args.checkpoint, args.device)
    print(f"Model loaded (bottleneck={config['bottleneck_dim']}, register={config['register_dim']})")

    # Load register embeddings
    if args.register_embeddings:
        register_path = args.register_embeddings
    else:
        # Try to find in same directory as checkpoint
        register_path = os.path.join(os.path.dirname(args.checkpoint), 'register_embeddings.pt')

    print(f"Loading register embeddings from: {register_path}")
    register_data = load_register_embeddings(register_path, args.device)
    print(f"Loaded {len(register_data['embeddings'])} register bins")

    # Load DCAE
    print("Loading DCAE...")
    from do.pipeline_do import DoTrainComponents
    components = DoTrainComponents(
        checkpoint_dir="/home/arlo/Data/ACE-Step/checkpoints",
        device_id=0
    )
    components.load_dcae()
    dcae = components.music_dcae

    if args.mode == 'listening_test':
        if not args.segments:
            print("Error: --segments required for listening_test mode")
            return

        generate_listening_test(
            model=model,
            register_data=register_data,
            segments_json=args.segments,
            output_dir=args.output_dir,
            dcae=dcae,
            num_samples=args.num_samples,
            device=args.device,
        )
    else:
        print("Correction mode not yet implemented")


if __name__ == "__main__":
    main()
