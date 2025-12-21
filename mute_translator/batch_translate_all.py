#!/usr/bin/env python3
"""
Batch translate all dry trumpet samples with both OLD and NEW mute translator models.
Optimized for A100 with batched inference.
"""

import argparse
import json
import os
import sys
import time
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/mute_translator')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data/dø')

from models import MuteTranslator, MuteTranslatorWithEnvelope, MuteTranslatorDirect, MuteTranslatorAdaptive


def fix_mount_path(path: str) -> str:
    """Fix mount paths from /mnt/msdd/ to /mnt/msdd2/"""
    if path and '/mnt/msdd/' in path:
        return path.replace('/mnt/msdd/', '/mnt/msdd2/')
    return path


def load_dcae(checkpoint_dir: str, device: str = "cuda"):
    """Load DCAE for encoding/decoding audio."""
    from do.pipeline_do import DoTrainComponents
    comps = DoTrainComponents(checkpoint_dir=checkpoint_dir, dtype="float32")
    dcae = comps.load_dcae()
    dcae = dcae.to(device).eval()
    return dcae


def load_translator(checkpoint_path: str, device: str = 'cuda'):
    """Load translator model, detecting type from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Detect model type
    if 'alpha_bias' in state_dict:
        model = MuteTranslatorAdaptive()
        model_type = 'adaptive'
    elif 'envelope_mod.gain_bias' in state_dict:
        model = MuteTranslatorWithEnvelope()
        model_type = 'envelope'
    elif 'residual_scale' in state_dict:
        model = MuteTranslator()
        model_type = 'standard'
    elif 'input_proj.0.weight' in state_dict:
        in_ch = state_dict['input_proj.0.weight'].shape[1]
        if in_ch == 8:
            model = MuteTranslatorDirect()
            model_type = 'direct'
        else:
            raise ValueError(f"Unknown model type with input channels {in_ch}")
    elif 'input_proj.weight' in state_dict:
        model = MuteTranslator()
        model_type = 'standard'
    else:
        raise ValueError("Cannot detect model type from checkpoint")

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 0)
    print(f"  Loaded from epoch {epoch}, type: {model_type}")

    return model, model_type


def translate_with_model(model, model_type, dry_latent, device='cuda'):
    """Translate latent using appropriate method for model type"""
    with torch.no_grad():
        dry_latent = dry_latent.to(device)

        if model_type == 'envelope':
            energy = dry_latent.abs().mean(dim=(1, 2))
            amp = energy / (energy.max() + 1e-6)
            energy_diff = torch.zeros_like(energy)
            energy_diff[:, 1:] = energy[:, 1:] - energy[:, :-1]
            energy_diff = torch.clamp(energy_diff, min=0)
            mean = energy_diff.mean(dim=1, keepdim=True)
            std = energy_diff.std(dim=1, keepdim=True)
            onsets = (energy_diff > mean + std).float()
            muted_latent = model(dry_latent, onsets, amp)
        else:
            muted_latent = model(dry_latent)

        return muted_latent


def decode_and_save(dcae, latent: torch.Tensor, output_path: str):
    """Decode latent to audio and save."""
    with torch.no_grad():
        sr, pred_wavs = dcae.decode(latent)

    audio = pred_wavs[0]
    audio = audio / (audio.abs().max() + 1e-8) * 0.9

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, audio.cpu(), sr)
    return sr


def process_batch(entries, old_model, old_type, new_model, new_type, dcae, output_base, device, results):
    """Process a batch of entries efficiently."""
    latents = []
    valid_entries = []

    # Load all latents
    for entry in entries:
        latent_path = fix_mount_path(entry['latent_path'])
        try:
            latent_data = torch.load(latent_path, weights_only=True)
            if isinstance(latent_data, dict):
                latent = latent_data['latents']
            else:
                latent = latent_data
            if latent.dim() == 3:
                latent = latent.unsqueeze(0)
            latents.append(latent)
            valid_entries.append(entry)
        except Exception as e:
            print(f"  Error loading {latent_path}: {e}")
            continue

    if not latents:
        return

    # Batch translate - translator is lightweight, can do all at once
    for latent, entry in zip(latents, valid_entries):
        latent = latent.to(device)
        audio_path = fix_mount_path(entry['audio_path'])
        basename = os.path.basename(audio_path).replace('.wav', '')

        # Create unique output directory based on original path structure
        rel_path = audio_path.replace('/home/arlo/gcs-bucket/', '').replace('/mnt/msdd2/', '')
        sample_dir = os.path.join(output_base, os.path.dirname(rel_path), basename)
        os.makedirs(sample_dir, exist_ok=True)

        # Translate with both models
        old_translated = translate_with_model(old_model, old_type, latent, device)
        new_translated = translate_with_model(new_model, new_type, latent, device)

        # Decode all three (dry original, old translated, new translated)
        dry_path = os.path.join(sample_dir, "dry_original.wav")
        old_path = os.path.join(sample_dir, "old_model_translated.wav")
        new_path = os.path.join(sample_dir, "new_model_translated.wav")

        try:
            decode_and_save(dcae, latent, dry_path)
            decode_and_save(dcae, old_translated, old_path)
            decode_and_save(dcae, new_translated, new_path)

            results.append({
                'original_audio': audio_path,
                'original_latent': fix_mount_path(entry['latent_path']),
                'dry_decoded': dry_path,
                'old_model_translated': old_path,
                'new_model_translated': new_path,
                'basename': basename
            })
        except Exception as e:
            print(f"  Error decoding {basename}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Batch translate all dry trumpet samples')
    parser.add_argument('--old_checkpoint', default='./checkpoints/best.pt',
                        help='Path to old checkpoint')
    parser.add_argument('--new_checkpoint', default='./checkpoints_baseline_optimized/best.pt',
                        help='Path to new checkpoint')
    parser.add_argument('--dcae_checkpoint', default='/home/arlo/Data/ACE-Step/checkpoints',
                        help='Path to DCAE checkpoints directory')
    parser.add_argument('--manifest', default='./mute_manifest_deduped.json',
                        help='Path to manifest JSON')
    parser.add_argument('--output_dir', default='/mnt/msdd2/mute_translator_outputs',
                        help='Output directory for audio files')
    parser.add_argument('--output_json', default='/mnt/msdd2/mute_translator_outputs/translation_results.json',
                        help='Output JSON with all paths')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing (DCAE decode is sequential anyway)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip samples that already have output files')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to process (for testing)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load DCAE decoder
    print("Loading DCAE decoder...")
    start = time.time()
    dcae = load_dcae(args.dcae_checkpoint, args.device)
    print(f"  DCAE loaded in {time.time()-start:.1f}s")

    # Load both translator models
    print(f"\nLoading OLD model: {args.old_checkpoint}")
    old_model, old_type = load_translator(args.old_checkpoint, args.device)

    print(f"\nLoading NEW model: {args.new_checkpoint}")
    new_model, new_type = load_translator(args.new_checkpoint, args.device)

    # Load manifest and get dry trumpet entries
    print(f"\nLoading manifest: {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)

    dry_entries = [e for e in manifest
                   if e.get('sub_group') == 'trumpet'
                   and e.get('is_muted') == False
                   and e.get('latent_path')]

    print(f"  Found {len(dry_entries)} dry trumpet entries with latents")

    if args.limit:
        dry_entries = dry_entries[:args.limit]
        print(f"  Limited to {len(dry_entries)} samples")

    # Process all samples
    results = []
    start_time = time.time()

    print(f"\nProcessing {len(dry_entries)} samples...")
    print("="*60)

    for i, entry in enumerate(tqdm(dry_entries, desc="Translating")):
        audio_path = fix_mount_path(entry['audio_path'])
        basename = os.path.basename(audio_path).replace('.wav', '')

        # Create unique output directory
        rel_path = audio_path.replace('/home/arlo/gcs-bucket/', '').replace('/mnt/msdd2/', '')
        sample_dir = os.path.join(args.output_dir, os.path.dirname(rel_path), basename)

        # Check if already processed
        if args.skip_existing:
            new_path = os.path.join(sample_dir, "new_model_translated.wav")
            if os.path.exists(new_path):
                continue

        os.makedirs(sample_dir, exist_ok=True)

        # Load latent
        latent_path = fix_mount_path(entry['latent_path'])
        try:
            latent_data = torch.load(latent_path, weights_only=True)
            if isinstance(latent_data, dict):
                latent = latent_data['latents']
            else:
                latent = latent_data
            if latent.dim() == 3:
                latent = latent.unsqueeze(0)
            latent = latent.to(args.device)
        except Exception as e:
            print(f"\n  Error loading {latent_path}: {e}")
            continue

        # Translate with both models
        with torch.no_grad():
            old_translated = translate_with_model(old_model, old_type, latent, args.device)
            new_translated = translate_with_model(new_model, new_type, latent, args.device)

        # Decode all three
        dry_path = os.path.join(sample_dir, "dry_original.wav")
        old_path = os.path.join(sample_dir, "old_model_translated.wav")
        new_path = os.path.join(sample_dir, "new_model_translated.wav")

        try:
            decode_and_save(dcae, latent, dry_path)
            decode_and_save(dcae, old_translated, old_path)
            decode_and_save(dcae, new_translated, new_path)

            results.append({
                'original_audio': audio_path,
                'original_latent': latent_path,
                'dry_decoded': dry_path,
                'old_model_translated': old_path,
                'new_model_translated': new_path,
                'basename': basename
            })

            # Save incremental results every 100 samples
            if len(results) % 100 == 0:
                with open(args.output_json, 'w') as f:
                    json.dump({
                        'old_checkpoint': args.old_checkpoint,
                        'new_checkpoint': args.new_checkpoint,
                        'total_processed': len(results),
                        'samples': results
                    }, f, indent=2)

        except Exception as e:
            print(f"\n  Error decoding {basename}: {e}")
            continue

        # Clear GPU cache periodically
        if i % 50 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time

    # Final save
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"Processed: {len(results)} samples")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/len(results):.2f}s per sample)")
    print(f"Output directory: {args.output_dir}")

    # Save final results JSON
    final_results = {
        'old_checkpoint': os.path.abspath(args.old_checkpoint),
        'new_checkpoint': os.path.abspath(args.new_checkpoint),
        'total_processed': len(results),
        'processing_time_seconds': elapsed,
        'samples': results
    }

    with open(args.output_json, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"Results JSON: {args.output_json}")


if __name__ == '__main__':
    main()
