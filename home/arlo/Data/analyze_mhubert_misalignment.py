#!/usr/bin/env python3
"""Analyze mHuBERT alignment differences"""
import json
import torch
from pathlib import Path
from tqdm import tqdm
from collections import Counter

manifest_path = "/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS_SPKEMB_MHUBERT.json"
manifest = json.loads(Path(manifest_path).read_text())

differences = []
debug_count = {
    'no_paths': 0,
    'files_missing': 0,
    'load_failed': 0,
    'not_tensors': 0,
    'success': 0
}

for item in tqdm(manifest, desc="Analyzing alignment"):  # Check all samples
    if len(differences) >= 1000:  # Stop after 1000 valid samples
        break
    latent_path = item.get("latent_path")
    mhubert_path = item.get("mhubert_features_path")

    if not (latent_path and mhubert_path and mhubert_path != "null"):
        debug_count['no_paths'] += 1
        continue
    if not (Path(latent_path).exists() and Path(mhubert_path).exists()):
        debug_count['files_missing'] += 1
        continue

    try:
        latents = torch.load(latent_path, map_location='cpu')
        if isinstance(latents, dict):
            if 'latents' in latents:
                latents = latents['latents']
            elif 'latent' in latents:
                latents = latents['latent']

        mhubert_data = torch.load(mhubert_path, map_location='cpu')
        if isinstance(mhubert_data, dict):
            if 'aligned_features' in mhubert_data:
                mhubert_features = mhubert_data['aligned_features']
            elif 'features' in mhubert_data:
                mhubert_features = mhubert_data['features']
            else:
                mhubert_features = None
        else:
            mhubert_features = mhubert_data

        if not (isinstance(latents, torch.Tensor) and isinstance(mhubert_features, torch.Tensor)):
            debug_count['not_tensors'] += 1
            continue

        T_dcae = latents.shape[2] if len(latents.shape) == 3 else latents.shape[1]
        T_mhubert = mhubert_features.shape[0] if len(mhubert_features.shape) >= 1 else len(mhubert_features)
        diff = abs(T_dcae - T_mhubert)
        differences.append(diff)
        debug_count['success'] += 1
    except Exception as e:
        debug_count['load_failed'] += 1
        if debug_count['load_failed'] == 1:  # Print first error
            print(f"\nFirst error: {e}")
            print(f"Latent path: {latent_path}")
            print(f"mHuBERT path: {mhubert_path}")

print(f"\nDebug counts:")
for key, value in debug_count.items():
    print(f"  {key}: {value}")

if not differences:
    print("\nNo valid samples found!")
    exit(1)

differences.sort()
counter = Counter(differences)

print(f"\n{'='*60}")
print(f"Alignment Difference Statistics (first 1000 samples)")
print(f"{'='*60}")
print(f"Total analyzed: {len(differences)}")
print(f"Min difference: {min(differences)} frames")
print(f"Max difference: {max(differences)} frames")
print(f"Mean difference: {sum(differences)/len(differences):.1f} frames")
print(f"Median difference: {differences[len(differences)//2]} frames")
print()
print(f"Frequency distribution:")
for diff, count in sorted(counter.items())[:20]:
    print(f"  {diff:4d} frames: {count:4d} samples ({100*count/len(differences):5.1f}%)")
print(f"{'='*60}")
