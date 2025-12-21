#!/usr/bin/env python3
"""
Evaluate Mute Translator using pre-computed latents only.
No audio encoding needed - just decode the translated latents.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path

import torch
import torchaudio

# Add paths
sys.path.insert(0, '/home/arlo/do-repo')
sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

from models import MuteTranslator, MuteTranslatorLarge
from dataset import load_manifest


def load_translator(checkpoint_path: str, device: str = 'cuda'):
    """Load trained translator."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config', {})
    model_type = config.get('model_type', 'small')

    if model_type == 'large':
        model = MuteTranslatorLarge()
    else:
        model = MuteTranslator()

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded translator from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Training loss: {checkpoint.get('loss', '?')}")

    return model, checkpoint


def load_dcae(checkpoint_dir: str, device: str = 'cuda'):
    """Load DCAE for decoding."""
    from do.pipeline_do import DoTrainComponents

    components = DoTrainComponents(
        root=checkpoint_dir,
        device=device,
        dtype=torch.float32
    )
    components.load_dcae()
    return components


def decode_latent(dcae, latent: torch.Tensor, output_path: str, sample_rate: int = 48000):
    """Decode latent to audio."""
    with torch.no_grad():
        # Ensure correct shape [B, C, H, T]
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)

        # Decode
        audio = dcae.dcae.decode(latent)

        # audio shape: [B, 1, samples]
        audio = audio.squeeze(0)  # [1, samples]

        # Save
        torchaudio.save(output_path, audio.cpu(), sample_rate)

    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--manifest', type=str, default='./mute_manifest_fixed.json')
    parser.add_argument('--dcae_checkpoint', type=str,
                        default='/home/arlo/Data/ACE-Step/checkpoints')
    parser.add_argument('--output_dir', type=str, default='./eval_output')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find the DCAE path
    dcae_path = args.dcae_checkpoint
    snapshot_path = os.path.join(dcae_path, 'models--ACE-Step--ACE-Step-v1-3.5B/snapshots')
    if os.path.exists(snapshot_path):
        snapshots = os.listdir(snapshot_path)
        if snapshots:
            dcae_path = os.path.join(snapshot_path, snapshots[0])

    print(f"Loading DCAE from: {dcae_path}")
    dcae = load_dcae(dcae_path, args.device)

    print("Loading translator...")
    translator, checkpoint = load_translator(args.checkpoint, args.device)

    print("Loading manifest...")
    dry_entries, muted_entries = load_manifest(args.manifest)
    print(f"  Dry: {len(dry_entries)}, Muted: {len(muted_entries)}")

    # Filter to entries with existing latent files
    dry_entries = [e for e in dry_entries if os.path.exists(e.get('latent_path', ''))]
    muted_entries = [e for e in muted_entries if os.path.exists(e.get('latent_path', ''))]
    print(f"  Valid dry: {len(dry_entries)}, Valid muted: {len(muted_entries)}")

    results = []

    print("\n" + "=" * 60)
    print("EVALUATING TRANSLATOR")
    print("=" * 60)

    # Sample random dry entries
    samples = random.sample(dry_entries, min(args.num_samples, len(dry_entries)))

    for i, entry in enumerate(samples):
        latent_path = entry['latent_path']
        name = Path(entry['audio_path']).stem

        print(f"\n[Sample {i+1}/{args.num_samples}] {name}")

        sample_dir = output_dir / f"sample_{i+1}_{name[:30]}"
        sample_dir.mkdir(exist_ok=True)

        try:
            # Load latent
            latent_data = torch.load(latent_path, map_location=args.device)
            if isinstance(latent_data, dict):
                dry_latent = latent_data.get('latents', latent_data.get('latent'))
            else:
                dry_latent = latent_data

            # Shape: [C, H, T] -> [1, C, H, T]
            if dry_latent.dim() == 3:
                dry_latent = dry_latent.unsqueeze(0)
            dry_latent = dry_latent.to(args.device)

            print(f"  Dry latent shape: {dry_latent.shape}")

            # Translate
            print(f"  Translating dry -> muted...")
            with torch.no_grad():
                muted_latent = translator(dry_latent)

            print(f"  Muted latent shape: {muted_latent.shape}")

            # Decode both
            print(f"  Decoding dry (reconstructed)...")
            dry_output = sample_dir / "01_dry_reconstructed.wav"
            decode_latent(dcae, dry_latent, str(dry_output))

            print(f"  Decoding muted (translated)...")
            muted_output = sample_dir / "02_muted_translated.wav"
            decode_latent(dcae, muted_latent, str(muted_output))

            # Stats
            result = {
                'name': name,
                'dry_mean': dry_latent.mean().item(),
                'dry_std': dry_latent.std().item(),
                'muted_mean': muted_latent.mean().item(),
                'muted_std': muted_latent.std().item(),
                'diff_norm': (muted_latent - dry_latent).norm().item(),
                'dry_audio': str(dry_output),
                'muted_audio': str(muted_output),
                'success': True
            }
            results.append(result)

            print(f"  Dry  mean={result['dry_mean']:.3f}, std={result['dry_std']:.3f}")
            print(f"  Muted mean={result['muted_mean']:.3f}, std={result['muted_std']:.3f}")
            print(f"  Saved: {sample_dir}")

        except Exception as e:
            print(f"  Error: {e}")
            results.append({'name': name, 'success': False, 'error': str(e)})

    # Also decode a few real muted samples for comparison
    print("\n" + "=" * 60)
    print("REAL MUTED SAMPLES (for comparison)")
    print("=" * 60)

    real_samples = random.sample(muted_entries, min(3, len(muted_entries)))

    for i, entry in enumerate(real_samples):
        latent_path = entry['latent_path']
        name = Path(entry['audio_path']).stem

        print(f"\n[Real Muted {i+1}] {name}")

        sample_dir = output_dir / f"real_muted_{i+1}_{name[:30]}"
        sample_dir.mkdir(exist_ok=True)

        try:
            latent_data = torch.load(latent_path, map_location=args.device)
            if isinstance(latent_data, dict):
                muted_latent = latent_data.get('latents', latent_data.get('latent'))
            else:
                muted_latent = latent_data

            if muted_latent.dim() == 3:
                muted_latent = muted_latent.unsqueeze(0)
            muted_latent = muted_latent.to(args.device)

            print(f"  Latent shape: {muted_latent.shape}")
            print(f"  Decoding...")

            output_path = sample_dir / "real_muted.wav"
            decode_latent(dcae, muted_latent, str(output_path))

            print(f"  Mean={muted_latent.mean().item():.3f}, std={muted_latent.std().item():.3f}")
            print(f"  Saved: {output_path}")

        except Exception as e:
            print(f"  Error: {e}")

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': str(args.checkpoint),
            'num_samples': args.num_samples,
            'results': results
        }, f, indent=2)

    print(f"\n\nResults saved to: {results_path}")
    print(f"Audio samples in: {output_dir}")
    print("\nListen to the samples and compare:")
    print("  - 01_dry_reconstructed.wav: Original dry trumpet decoded")
    print("  - 02_muted_translated.wav: Translated to 'muted'")
    print("  - real_muted_*/real_muted.wav: Actual muted recordings for reference")


if __name__ == '__main__':
    main()
