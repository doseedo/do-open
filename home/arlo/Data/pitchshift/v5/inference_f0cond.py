#!/usr/bin/env python3
"""
Inference with F0-Conditioned Register Disentanglement
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

from models_f0cond import F0ConditionedModel


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def load_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = F0ConditionedModel(
        hidden_dim=config.get('hidden_dim', 256),
        bottleneck_dim=config['bottleneck_dim'],
        f0_dim=config['f0_dim'],
        register_dim=config['register_dim'],
        downsample=config.get('downsample', 8),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    return model, config


def load_register_embeddings(embeddings_path, device='cuda'):
    data = torch.load(embeddings_path, map_location=device, weights_only=False)

    if isinstance(data['embeddings'], list):
        embeddings = [e.to(device) for e in data['embeddings']]
        sorted_bins = list(range(len(embeddings)))
    else:
        embeddings = {k: v.to(device) for k, v in data['embeddings'].items()}
        sorted_bins = sorted(embeddings.keys())

    bin_size = data.get('bin_size') or data.get('pitch_bin_size', 2.0)

    return {
        'embeddings': embeddings,
        'sorted_bins': sorted_bins,
        'bin_size': bin_size,
    }


def get_register_embedding_for_pitch(register_data, target_midi):
    bin_size = register_data['bin_size']
    embeddings = register_data['embeddings']
    sorted_bins = register_data['sorted_bins']

    target_bin = int(target_midi / bin_size)

    closest_bin = sorted_bins[0]
    min_dist = abs(sorted_bins[0] - target_bin)
    for b in sorted_bins:
        if abs(b - target_bin) < min_dist:
            min_dist = abs(b - target_bin)
            closest_bin = b

    if isinstance(embeddings, list):
        idx = sorted_bins.index(closest_bin)
        return embeddings[idx]
    else:
        return embeddings[closest_bin]


def generate_listening_test(
    model,
    register_data,
    manifest_path,
    output_dir,
    dcae,
    instrument='trumpet',
    num_samples=10,
    device='cuda',
):
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get instrument entries with F0
    entries = []
    for e in manifest:
        if e.get('sub_group') != instrument:
            continue
        latent_path = fix_path(e.get('latent_path', ''))
        f0_path = fix_path(e.get('conditioning_paths', {}).get('f0', ''))
        if latent_path and f0_path and os.path.exists(latent_path) and os.path.exists(f0_path):
            entries.append({
                'latent_path': latent_path,
                'f0_path': f0_path,
            })

    # Sample diverse entries
    step = max(1, len(entries) // num_samples)
    selected = [entries[i * step] for i in range(min(num_samples, len(entries)))]

    print(f"Generating {len(selected)} listening test samples...")

    for i, entry in enumerate(tqdm(selected)):
        # Load latent
        latent_data = torch.load(entry['latent_path'], map_location='cpu', weights_only=False)
        if isinstance(latent_data, dict):
            latent = latent_data.get('latents', latent_data.get('latent'))
        else:
            latent = latent_data
        if latent.dim() == 4:
            latent = latent.squeeze(0)

        # Load F0
        f0 = np.load(entry['f0_path'])
        f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)

        # Align lengths
        T = min(latent.shape[-1], len(f0), 256)  # Cap at 256 frames
        latent = latent[:, :, :T].unsqueeze(0).to(device)
        f0_tensor = torch.from_numpy(f0[:T]).unsqueeze(0).to(device)

        # Get source pitch
        f0_valid = f0[f0 > 20]
        if len(f0_valid) == 0:
            continue
        source_midi = 12 * np.log2(np.median(f0_valid) / 440) + 69

        # 1. Original (direct decode)
        sr, wavs = dcae.decode(latent)
        original_audio = wavs[0]
        original_path = os.path.join(output_dir, f"{i:02d}_original_pitch{source_midi:.0f}.wav")
        torchaudio.save(original_path, original_audio.cpu(), sr)

        # 2. Reconstruction (same register)
        source_register = get_register_embedding_for_pitch(register_data, source_midi)
        with torch.no_grad():
            content = model.encode_content(latent)
            f0_emb = model.encode_f0(f0_tensor)
            reconstructed = model.decode(content, f0_emb, source_register, target_length=T)
        sr, wavs = dcae.decode(reconstructed)
        recon_path = os.path.join(output_dir, f"{i:02d}_reconstructed_pitch{source_midi:.0f}.wav")
        torchaudio.save(recon_path, wavs[0].cpu(), sr)

        # 3. Register swapped (+12 semitones)
        target_midi = source_midi + 12
        if target_midi > 96:
            target_midi = source_midi - 12
        target_register = get_register_embedding_for_pitch(register_data, target_midi)
        with torch.no_grad():
            swapped = model.decode(content, f0_emb, target_register, target_length=T)
        sr, wavs = dcae.decode(swapped)
        swapped_path = os.path.join(output_dir, f"{i:02d}_swapped_to{target_midi:.0f}.wav")
        torchaudio.save(swapped_path, wavs[0].cpu(), sr)

        print(f"  Sample {i}: pitch {source_midi:.0f} → {target_midi:.0f}")

    print(f"\nListening test saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="F0-Conditioned Inference")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--register_embeddings', type=str)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--instrument', type=str, default='trumpet')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, config = load_model(args.checkpoint, args.device)
    print(f"Model loaded (bottleneck={config['bottleneck_dim']}, f0={config['f0_dim']}, register={config['register_dim']})")

    # Load register embeddings
    if args.register_embeddings:
        register_path = args.register_embeddings
    else:
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

    generate_listening_test(
        model=model,
        register_data=register_data,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        dcae=dcae,
        instrument=args.instrument,
        num_samples=args.num_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
