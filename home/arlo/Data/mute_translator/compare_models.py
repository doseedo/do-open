#!/usr/bin/env python3
"""
Compare old (pre-onset) vs new (envelope) mute translator models
Outputs: 1_dry_original.wav, 2_old_translated.wav, 3_new_translated.wav for each sample
"""

import argparse
import json
import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path

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
        # Has alpha_bias = adaptive mixing architecture
        print(f"  Detected: MuteTranslatorAdaptive (learnable mixing)")
        model = MuteTranslatorAdaptive()
        model_type = 'adaptive'
    elif 'envelope_mod.gain_bias' in state_dict:
        print(f"  Detected: MuteTranslatorWithEnvelope")
        model = MuteTranslatorWithEnvelope()
        model_type = 'envelope'
    elif 'residual_scale' in state_dict:
        # Has residual_scale = residual architecture (MuteTranslator)
        print(f"  Detected: MuteTranslator (standard residual)")
        model = MuteTranslator()
        model_type = 'standard'
    elif 'input_proj.0.weight' in state_dict:
        # Has input_proj.0 (Sequential) but no residual_scale = direct architecture
        in_ch = state_dict['input_proj.0.weight'].shape[1]
        if in_ch == 8:
            print(f"  Detected: MuteTranslatorDirect (non-residual)")
            model = MuteTranslatorDirect()
            model_type = 'direct'
        else:
            raise ValueError(f"Unknown model type with input channels {in_ch}")
    elif 'input_proj.weight' in state_dict:
        # Older single-conv input_proj - assume standard without residual_scale (legacy)
        in_ch = state_dict['input_proj.weight'].shape[1]
        if in_ch == 8:
            print(f"  Detected: MuteTranslator (standard, legacy)")
            model = MuteTranslator()
            model_type = 'standard'
        else:
            raise ValueError(f"Unknown model type with input channels {in_ch}")
    else:
        raise ValueError("Cannot detect model type from checkpoint")

    # Use strict=False to handle missing keys like dry_scale in older checkpoints
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('metrics', {}).get('loss', 0)
    print(f"  Loaded from epoch {epoch}, loss: {loss:.4f}")

    return model, model_type


def translate_with_model(model, model_type, dry_latent, device='cuda'):
    """Translate latent using appropriate method for model type"""
    with torch.no_grad():
        dry_latent = dry_latent.to(device)

        if model_type == 'envelope':
            # Generate synthetic onset/amp from latent energy
            energy = dry_latent.abs().mean(dim=(1, 2))  # [B, T]
            amp = energy / (energy.max() + 1e-6)

            # Onset detection from energy gradient
            energy_diff = torch.zeros_like(energy)
            energy_diff[:, 1:] = energy[:, 1:] - energy[:, :-1]
            energy_diff = torch.clamp(energy_diff, min=0)

            mean = energy_diff.mean(dim=1, keepdim=True)
            std = energy_diff.std(dim=1, keepdim=True)
            onsets = (energy_diff > mean + std).float()

            muted_latent = model(dry_latent, onsets, amp)
        else:
            # 'standard' (MuteTranslator) and 'direct' (MuteTranslatorDirect) both use same interface
            muted_latent = model(dry_latent)

        return muted_latent


def decode_latent(dcae, latent: torch.Tensor, output_path: str):
    """Decode latent to audio and save."""
    with torch.no_grad():
        sr, pred_wavs = dcae.decode(latent)

    # pred_wavs is a list of [2, T] tensors
    audio = pred_wavs[0]  # Get first (and only) batch item

    # Normalize
    audio = audio / (audio.abs().max() + 1e-8) * 0.9

    torchaudio.save(output_path, audio.cpu(), sr)
    return audio, sr


def spectral_centroid(audio_tensor, sr=44100):
    """Compute spectral centroid from audio tensor."""
    if isinstance(audio_tensor, torch.Tensor):
        audio = audio_tensor.squeeze().cpu().numpy()
    else:
        audio = audio_tensor.squeeze()

    if audio.ndim > 1:
        audio = audio.mean(axis=0)  # Mono

    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    return np.sum(freqs * fft) / (np.sum(fft) + 1e-6)


def main():
    parser = argparse.ArgumentParser(description='Compare old vs new mute translator models')
    parser.add_argument('--old_checkpoint', default='./checkpoints/best.pt',
                        help='Path to old (pre-onset) checkpoint')
    parser.add_argument('--new_checkpoint', default='./checkpoints_envelope_optimized/best.pt',
                        help='Path to new (envelope) checkpoint')
    parser.add_argument('--dcae_checkpoint', default='/home/arlo/Data/ACE-Step/checkpoints',
                        help='Path to DCAE checkpoints directory')
    parser.add_argument('--manifest', default='./mute_manifest_deduped.json',
                        help='Path to manifest JSON')
    parser.add_argument('--output_dir', default='./comparison_old_vs_new',
                        help='Output directory for audio files')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to compare')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--new_dry_scale', type=float, default=None,
                        help='Override dry_scale for NEW model (default: use checkpoint value)')
    parser.add_argument('--new_residual_scale', type=float, default=None,
                        help='Override residual_scale for NEW model (default: use checkpoint value)')
    parser.add_argument('--old_dry_scale', type=float, default=None,
                        help='Override dry_scale for OLD model (default: use checkpoint value)')
    parser.add_argument('--old_residual_scale', type=float, default=None,
                        help='Override residual_scale for OLD model (default: use checkpoint value)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load DCAE decoder
    print("Loading DCAE decoder...")
    dcae = load_dcae(args.dcae_checkpoint, args.device)
    print("  DCAE loaded")

    # Load both translator models
    print(f"\nLoading OLD model: {args.old_checkpoint}")
    old_model, old_type = load_translator(args.old_checkpoint, args.device)

    # Apply scale overrides to OLD model if specified
    if args.old_dry_scale is not None:
        if hasattr(old_model, 'dry_scale'):
            old_model.dry_scale.data.fill_(args.old_dry_scale)
            print(f"  Overriding dry_scale to {args.old_dry_scale}")
    if args.old_residual_scale is not None:
        if hasattr(old_model, 'residual_scale'):
            old_model.residual_scale.data.fill_(args.old_residual_scale)
            print(f"  Overriding residual_scale to {args.old_residual_scale}")

    print(f"\nLoading NEW model: {args.new_checkpoint}")
    new_model, new_type = load_translator(args.new_checkpoint, args.device)

    # Apply scale overrides to NEW model if specified
    if args.new_dry_scale is not None:
        if hasattr(new_model, 'dry_scale'):
            new_model.dry_scale.data.fill_(args.new_dry_scale)
            print(f"  Overriding dry_scale to {args.new_dry_scale}")
    if args.new_residual_scale is not None:
        if hasattr(new_model, 'residual_scale'):
            new_model.residual_scale.data.fill_(args.new_residual_scale)
            print(f"  Overriding residual_scale to {args.new_residual_scale}")

    # Load manifest and get dry trumpet entries
    print(f"\nLoading manifest: {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)

    dry_entries = [e for e in manifest
                   if e.get('sub_group') == 'trumpet'
                   and e.get('is_muted') == False
                   and e.get('latent_path')]

    print(f"  Found {len(dry_entries)} dry trumpet entries with latents")

    # Sample entries
    import random
    random.seed(args.seed)  # Seed for reproducible sampling
    selected = random.sample(dry_entries, min(args.num_samples, len(dry_entries)))

    results = []

    for i, entry in enumerate(selected):
        audio_path = fix_mount_path(entry['audio_path'])
        latent_path = fix_mount_path(entry['latent_path'])
        basename = os.path.basename(audio_path).replace('.wav', '')

        # Create sample directory
        sample_dir = os.path.join(args.output_dir, f"sample_{i:02d}_{basename[:30]}")
        os.makedirs(sample_dir, exist_ok=True)

        print(f"\n[{i+1}/{args.num_samples}] {basename}")

        # Load dry latent
        latent_data = torch.load(latent_path, weights_only=True)
        # Handle dict format (new) vs tensor format (old)
        if isinstance(latent_data, dict):
            dry_latent = latent_data['latents']
        else:
            dry_latent = latent_data
        if dry_latent.dim() == 3:
            dry_latent = dry_latent.unsqueeze(0)
        dry_latent = dry_latent.to(args.device)

        # Translate with both models
        print("  Translating with OLD model...")
        old_translated = translate_with_model(old_model, old_type, dry_latent, args.device)

        print("  Translating with NEW model...")
        new_translated = translate_with_model(new_model, new_type, dry_latent, args.device)

        # Decode all three to audio
        print("  Decoding to audio...")
        dry_path = os.path.join(sample_dir, "1_dry_original.wav")
        old_path = os.path.join(sample_dir, "2_old_model_translated.wav")
        new_path = os.path.join(sample_dir, "3_new_model_translated.wav")

        dry_audio, sr = decode_latent(dcae, dry_latent, dry_path)
        old_audio, _ = decode_latent(dcae, old_translated, old_path)
        new_audio, _ = decode_latent(dcae, new_translated, new_path)

        print(f"  Saved: {sample_dir}/")

        # Compute spectral metrics
        dry_cent = spectral_centroid(dry_audio, sr)
        old_cent = spectral_centroid(old_audio, sr)
        new_cent = spectral_centroid(new_audio, sr)

        print(f"  Centroid: dry={dry_cent:.0f}Hz, old={old_cent:.0f}Hz ({old_cent-dry_cent:+.0f}), new={new_cent:.0f}Hz ({new_cent-dry_cent:+.0f})")

        results.append({
            'sample': basename,
            'dry_centroid': float(dry_cent),
            'old_centroid': float(old_cent),
            'new_centroid': float(new_cent),
            'old_shift': float(old_cent - dry_cent),
            'new_shift': float(new_cent - dry_cent),
            'output_dir': sample_dir
        })

    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    avg_old_shift = np.mean([r['old_shift'] for r in results])
    avg_new_shift = np.mean([r['new_shift'] for r in results])

    print(f"Avg centroid shift (OLD model): {avg_old_shift:+.1f} Hz")
    print(f"Avg centroid shift (NEW model): {avg_new_shift:+.1f} Hz")
    print(f"\nNegative shift = darker/muted timbre (good)")

    # Save results JSON
    results_path = os.path.join(args.output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'old_checkpoint': args.old_checkpoint,
            'new_checkpoint': args.new_checkpoint,
            'samples': results,
            'summary': {
                'avg_old_centroid_shift': avg_old_shift,
                'avg_new_centroid_shift': avg_new_shift
            }
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"\nListen to files in: {args.output_dir}/")
    print("Each folder contains:")
    print("  1_dry_original.wav         - Original dry trumpet")
    print("  2_old_model_translated.wav - Translated with OLD model (pre-onset)")
    print("  3_new_model_translated.wav - Translated with NEW model (envelope)")


if __name__ == '__main__':
    main()
