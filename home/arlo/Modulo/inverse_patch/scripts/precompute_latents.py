#!/usr/bin/env python3
"""
Precompute DCAE latents for training data.

Processes audio pairs in batches, saves latents, uploads to GCS,
and creates updated manifest for training.

Usage:
    python -m inverse_afx.scripts.precompute_latents \
        --manifest /home/arlo/gcs-bucket/inverse_afx_training_data/manifest.json \
        --output-dir /mnt/models/latent_cache \
        --gcs-dir /home/arlo/gcs-bucket/inverse_afx_training_data/latent_pairs \
        --batch-size 5000
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio

# Add ACE-Step to path
sys.path.insert(0, "/home/arlo/Data/ACE-Step")

# Default checkpoint paths
DEFAULT_DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
DEFAULT_VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"


def load_dcae(device='cuda'):
    """Load DCAE codec."""
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

    print("Loading DCAE codec...")
    codec = MusicDCAE(
        source_sample_rate=44100,
        dcae_checkpoint_path=DEFAULT_DCAE_PATH,
        vocoder_checkpoint_path=DEFAULT_VOCODER_PATH,
    )
    codec = codec.to(device)
    codec.eval()
    for param in codec.parameters():
        param.requires_grad = False
    print("DCAE loaded.")
    return codec


def load_audio(path, target_sr=44100):
    """Load audio file and convert to stereo."""
    audio, sr = torchaudio.load(path)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)

    # Convert mono to stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    return audio


@torch.no_grad()
def encode_audio(codec, audio, device='cuda'):
    """Encode audio to DCAE latent."""
    # audio: [2, T] -> [1, 2, T]
    audio = audio.unsqueeze(0).to(device)
    audio = torch.clamp(audio, -1, 1)

    T = audio.shape[-1]
    audio_lengths = torch.tensor([T], device=device)

    latent, latent_lengths = codec.encode(audio, audio_lengths=audio_lengths, sr=44100)

    # Return [C, H, T'] (remove batch dim)
    return latent.squeeze(0).cpu(), {
        'audio_length': T,
        'latent_shape': list(latent.shape),
    }


def process_batch(
    codec,
    manifest_items,
    output_dir,
    device='cuda',
):
    """Process a batch of audio pairs and save latents."""
    results = []

    for item in tqdm(manifest_items, desc="Encoding"):
        item_id = item['id']
        wet_path = item['wet_path']
        dry_path = item['dry_path']

        try:
            # Load audio
            wet_audio = load_audio(wet_path)
            dry_audio = load_audio(dry_path)

            # Encode to latent
            wet_latent, wet_info = encode_audio(codec, wet_audio, device)
            dry_latent, dry_info = encode_audio(codec, dry_audio, device)

            # Save latents
            wet_latent_path = os.path.join(output_dir, f"{item_id}_wet.pt")
            dry_latent_path = os.path.join(output_dir, f"{item_id}_dry.pt")

            torch.save({
                'latent': wet_latent,
                'audio_length': wet_info['audio_length'],
                'latent_shape': wet_info['latent_shape'],
            }, wet_latent_path)

            torch.save({
                'latent': dry_latent,
                'audio_length': dry_info['audio_length'],
                'latent_shape': dry_info['latent_shape'],
            }, dry_latent_path)

            # Create result entry (preserves chain_spec, magnitude_tier, etc. from input)
            result = {
                'id': item_id,
                'wet_latent_path': wet_latent_path,
                'dry_latent_path': dry_latent_path,
                'chain_spec': item.get('chain_spec', []),
                'chain_length': item.get('chain_length', 0),
                'group': item.get('group', 'unknown'),
                'subgroup': item.get('subgroup', 'unknown'),
                'magnitude_tier': item.get('magnitude_tier', 'random'),
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing {item_id}: {e}")
            continue

    return results


def upload_to_gcs(local_dir, gcs_dir):
    """Copy files from local dir to GCS-mounted dir."""
    print(f"Uploading {local_dir} -> {gcs_dir}")

    os.makedirs(gcs_dir, exist_ok=True)

    files = list(Path(local_dir).glob("*.pt"))
    for f in tqdm(files, desc="Uploading"):
        dst = os.path.join(gcs_dir, f.name)
        shutil.copy2(f, dst)

    print(f"Uploaded {len(files)} files")


def update_manifest_paths(manifest_items, local_dir, gcs_dir):
    """Update manifest paths from local to GCS."""
    updated = []
    for item in manifest_items:
        item = item.copy()
        if 'wet_latent_path' in item:
            item['wet_latent_path'] = item['wet_latent_path'].replace(local_dir, gcs_dir)
        if 'dry_latent_path' in item:
            item['dry_latent_path'] = item['dry_latent_path'].replace(local_dir, gcs_dir)
        updated.append(item)
    return updated


def main():
    parser = argparse.ArgumentParser(description='Precompute DCAE latents')
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/gcs-bucket/inverse_afx_training_data/manifest.json',
                        help='Input manifest path')
    parser.add_argument('--output-dir', type=str,
                        default='/mnt/models/latent_cache',
                        help='Local output directory for latents')
    parser.add_argument('--gcs-dir', type=str,
                        default='/home/arlo/gcs-bucket/inverse_afx_training_data/latent_pairs',
                        help='GCS destination directory')
    parser.add_argument('--batch-size', type=int, default=5000,
                        help='Process and upload in batches of this size')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Start from this index (for resuming)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()

    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)
    print(f"Total items: {len(manifest)}")

    # Skip already processed
    manifest = manifest[args.start_idx:]
    print(f"Processing from index {args.start_idx}: {len(manifest)} items")

    # Create output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.gcs_dir, exist_ok=True)

    # Load DCAE
    codec = load_dcae(args.device)

    # Process in batches
    all_results = []
    num_batches = (len(manifest) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(manifest))
        batch = manifest[start:end]

        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1}/{num_batches} (items {args.start_idx + start}-{args.start_idx + end})")
        print(f"{'='*60}")

        # Process batch
        batch_results = process_batch(codec, batch, args.output_dir, args.device)

        # Upload to GCS
        upload_to_gcs(args.output_dir, args.gcs_dir)

        # Update paths for GCS
        batch_results = update_manifest_paths(batch_results, args.output_dir, args.gcs_dir)
        all_results.extend(batch_results)

        # Save intermediate manifest
        manifest_path = os.path.join(
            os.path.dirname(args.gcs_dir),
            f'manifest_latent_batch{batch_idx + 1}.json'
        )
        with open(manifest_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved intermediate manifest: {manifest_path}")

        # Clear local cache
        for f in Path(args.output_dir).glob("*.pt"):
            f.unlink()
        print("Cleared local cache")

        # Clear GPU cache
        torch.cuda.empty_cache()

    # Save final manifest
    final_manifest_path = os.path.join(
        os.path.dirname(args.gcs_dir),
        'manifest_latent.json'
    )
    with open(final_manifest_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Done! Final manifest: {final_manifest_path}")
    print(f"Total processed: {len(all_results)}")
    print(f"Latent pairs saved to: {args.gcs_dir}")


if __name__ == '__main__':
    main()
