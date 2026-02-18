#!/usr/bin/env python3
"""
Verify mHuBERT feature alignment with DCAE latents.
"""
import json
import torch
from pathlib import Path
from tqdm import tqdm

manifest_path = "/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS_SPKEMB_MHUBERT.json"
manifest = json.loads(Path(manifest_path).read_text())

mismatches = []
checked = 0
missing = 0

for item in tqdm(manifest[:100], desc="Checking alignment"):  # Check first 100
    latent_path = item.get("latent_path")
    mhubert_path = item.get("mhubert_features_path")

    if not latent_path or not mhubert_path:
        continue

    if not Path(latent_path).exists() or not Path(mhubert_path).exists():
        missing += 1
        continue

    try:
        # Load DCAE latents
        latents = torch.load(latent_path, map_location='cpu')
        if isinstance(latents, dict):
            if 'latents' in latents:
                latents = latents['latents']
            elif 'latent' in latents:
                latents = latents['latent']
            else:
                continue
        if not isinstance(latents, torch.Tensor):
            continue
        T_slow_dcae = latents.shape[2] if latents.dim() == 3 else latents.shape[1]

        # Load mHuBERT features
        mhubert_data = torch.load(mhubert_path, map_location='cpu')
        if 'aligned_features' in mhubert_data:
            mhubert_features = mhubert_data['aligned_features']
        elif 'features' in mhubert_data:
            mhubert_features = mhubert_data['features']
        else:
            continue

        T_slow_mhubert = mhubert_features.shape[0]

        # Check alignment
        diff = abs(T_slow_dcae - T_slow_mhubert)
        if diff > 2:  # Allow 2 frame tolerance
            mismatches.append({
                'audio': item.get('audio_path'),
                'dcae_frames': T_slow_dcae,
                'mhubert_frames': T_slow_mhubert,
                'diff': diff
            })

        checked += 1

    except Exception as e:
        print(f"Error: {e}")

print(f"\n{'='*60}")
print(f"Checked: {checked} files")
print(f"Missing: {missing} files")
print(f"Mismatches (>2 frames): {len(mismatches)}")

if mismatches:
    print(f"\n⚠️  Found {len(mismatches)} alignment issues:")
    for m in mismatches[:10]:
        print(f"  {Path(m['audio']).name}")
        print(f"    DCAE: {m['dcae_frames']} | mHuBERT: {m['mhubert_frames']} | Diff: {m['diff']}")
else:
    print("✅ All checked files are properly aligned!")
