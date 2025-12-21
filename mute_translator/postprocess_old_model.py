#!/usr/bin/env python3
"""
Post-process OLD model output to reduce dry bleed-through.
Approach: Subtract scaled dry latent from output in latent space.
"""

import argparse
import json
import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/mute_translator')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data/dø')

from models import MuteTranslator

def fix_mount_path(path: str) -> str:
    if path and '/mnt/msdd/' in path:
        return path.replace('/mnt/msdd/', '/mnt/msdd2/')
    return path

def load_dcae(checkpoint_dir: str, device: str = "cuda"):
    from do.pipeline_do import DoTrainComponents
    comps = DoTrainComponents(checkpoint_dir=checkpoint_dir, dtype="float32")
    dcae = comps.load_dcae()
    dcae = dcae.to(device).eval()
    return dcae

def load_translator(checkpoint_path: str, device: str = 'cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    model = MuteTranslator()
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

def decode_latent(dcae, latent: torch.Tensor, output_path: str):
    with torch.no_grad():
        sr, pred_wavs = dcae.decode(latent)
    audio = pred_wavs[0]
    audio = audio / (audio.abs().max() + 1e-8) * 0.9
    torchaudio.save(output_path, audio.cpu(), sr)
    return audio, sr

def spectral_centroid(audio_tensor, sr=44100):
    if isinstance(audio_tensor, torch.Tensor):
        audio = audio_tensor.squeeze().cpu().numpy()
    else:
        audio = audio_tensor.squeeze()
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    return np.sum(freqs * fft) / (np.sum(fft) + 1e-6)

def main():
    parser = argparse.ArgumentParser(description='Post-process OLD model to reduce dry bleed')
    parser.add_argument('--checkpoint', default='./checkpoints/best.pt')
    parser.add_argument('--dcae_checkpoint', default='/home/arlo/Data/ACE-Step/checkpoints')
    parser.add_argument('--manifest', default='./mute_manifest_deduped.json')
    parser.add_argument('--output_dir', default='./comparison_postprocess')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    # Post-processing parameters
    parser.add_argument('--dry_subtract', type=float, default=0.3,
                        help='How much dry to subtract from output (0-1)')
    parser.add_argument('--residual_boost', type=float, default=1.5,
                        help='Boost factor for residual component')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading DCAE decoder...")
    dcae = load_dcae(args.dcae_checkpoint, args.device)

    print(f"Loading model: {args.checkpoint}")
    model = load_translator(args.checkpoint, args.device)

    print(f"Loading manifest: {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)

    dry_entries = [e for e in manifest
                   if e.get('sub_group') == 'trumpet'
                   and e.get('is_muted') == False
                   and e.get('latent_path')]

    import random
    random.seed(args.seed)
    selected = random.sample(dry_entries, min(args.num_samples, len(dry_entries)))

    results = []

    for i, entry in enumerate(selected):
        latent_path = fix_mount_path(entry['latent_path'])
        basename = os.path.basename(entry['audio_path']).replace('.wav', '')

        sample_dir = os.path.join(args.output_dir, f"sample_{i:02d}_{basename[:30]}")
        os.makedirs(sample_dir, exist_ok=True)

        print(f"\n[{i+1}/{args.num_samples}] {basename}")

        # Load dry latent
        latent_data = torch.load(latent_path, weights_only=True)
        if isinstance(latent_data, dict):
            dry_latent = latent_data['latents']
        else:
            dry_latent = latent_data
        if dry_latent.dim() == 3:
            dry_latent = dry_latent.unsqueeze(0)
        dry_latent = dry_latent.to(args.device)

        with torch.no_grad():
            # Standard OLD model output
            old_output = model(dry_latent)
            
            # Post-processed: reduce dry, boost residual
            # old_output = dry_scale * dry + residual_scale * residual
            # We want: dry_scale_new * dry + residual_scale_new * residual
            # residual = (old_output - dry_scale * dry) / residual_scale
            
            dry_scale = model.dry_scale.item()
            res_scale = model.residual_scale.item()
            
            # Extract residual
            residual = (old_output - dry_scale * dry_latent) / (res_scale + 1e-6)
            
            # Recompose with new scales
            new_dry_scale = dry_scale * (1 - args.dry_subtract)
            new_res_scale = res_scale * args.residual_boost
            
            processed = new_dry_scale * dry_latent + new_res_scale * residual

        # Decode all versions
        print("  Decoding...")
        dry_path = os.path.join(sample_dir, "1_dry_original.wav")
        old_path = os.path.join(sample_dir, "2_old_model.wav")
        proc_path = os.path.join(sample_dir, f"3_processed_sub{args.dry_subtract}_boost{args.residual_boost}.wav")

        dry_audio, sr = decode_latent(dcae, dry_latent, dry_path)
        old_audio, _ = decode_latent(dcae, old_output, old_path)
        proc_audio, _ = decode_latent(dcae, processed, proc_path)

        # Compute centroids
        dry_cent = spectral_centroid(dry_audio, sr)
        old_cent = spectral_centroid(old_audio, sr)
        proc_cent = spectral_centroid(proc_audio, sr)

        print(f"  Centroid: dry={dry_cent:.0f}Hz, old={old_cent:.0f}Hz ({old_cent-dry_cent:+.0f}), proc={proc_cent:.0f}Hz ({proc_cent-dry_cent:+.0f})")
        print(f"  Scales: dry={new_dry_scale:.3f}, res={new_res_scale:.3f}")

        results.append({
            'sample': basename,
            'dry_centroid': float(dry_cent),
            'old_centroid': float(old_cent),
            'processed_centroid': float(proc_cent),
            'old_shift': float(old_cent - dry_cent),
            'processed_shift': float(proc_cent - dry_cent),
        })

    # Summary
    print("\n" + "="*60)
    print("POST-PROCESSING SUMMARY")
    print("="*60)
    print(f"dry_subtract={args.dry_subtract}, residual_boost={args.residual_boost}")
    
    avg_old = np.mean([r['old_shift'] for r in results])
    avg_proc = np.mean([r['processed_shift'] for r in results])
    
    print(f"Avg centroid shift (OLD): {avg_old:+.1f} Hz")
    print(f"Avg centroid shift (PROCESSED): {avg_proc:+.1f} Hz")
    print(f"\nOutput: {args.output_dir}")

if __name__ == '__main__':
    main()
