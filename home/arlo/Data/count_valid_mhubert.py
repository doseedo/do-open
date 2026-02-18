#!/usr/bin/env python3
"""Count valid mHuBERT extractions across the full manifest"""
import json
import torch
from pathlib import Path
from tqdm import tqdm

manifest_path = "/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS_SPKEMB_MHUBERT.json"
manifest = json.loads(Path(manifest_path).read_text())

total = len(manifest)
has_mhubert_path = 0
mhubert_exists = 0
properly_aligned = 0
missing_files = 0
alignment_errors = []

tolerance = 2  # frames

for item in tqdm(manifest, desc="Checking mHuBERT"):
    mhubert_path = item.get("mhubert_features_path")

    # Count samples with mHuBERT path
    if mhubert_path and mhubert_path != "null":
        has_mhubert_path += 1

        # Check if file exists
        if Path(mhubert_path).exists():
            mhubert_exists += 1

            # Check alignment
            latent_path = item.get("latent_path")
            if latent_path and Path(latent_path).exists():
                try:
                    # Load DCAE latents
                    latents = torch.load(latent_path, map_location='cpu')
                    if isinstance(latents, dict):
                        if 'latents' in latents:
                            latents = latents['latents']
                        elif 'latent' in latents:
                            latents = latents['latent']

                    if isinstance(latents, torch.Tensor):
                        T_slow_dcae = latents.shape[2] if latents.dim() == 3 else latents.shape[1]

                        # Load mHuBERT features
                        mhubert_data = torch.load(mhubert_path, map_location='cpu')
                        if isinstance(mhubert_data, dict):
                            mhubert_features = mhubert_data.get('aligned_features') or mhubert_data.get('features')
                        else:
                            mhubert_features = mhubert_data

                        if isinstance(mhubert_features, torch.Tensor):
                            T_slow_mhubert = mhubert_features.shape[0]
                            diff = abs(T_slow_dcae - T_slow_mhubert)

                            if diff <= tolerance:
                                properly_aligned += 1
                            else:
                                alignment_errors.append({
                                    'audio': item.get('audio_path', ''),
                                    'dcae': T_slow_dcae,
                                    'mhubert': T_slow_mhubert,
                                    'diff': diff
                                })
                except Exception as e:
                    pass
        else:
            missing_files += 1

print(f"\n{'='*70}")
print(f"mHuBERT Feature Statistics")
print(f"{'='*70}")
print(f"Total samples in manifest:        {total}")
print(f"Samples with mHuBERT path:        {has_mhubert_path} ({100*has_mhubert_path/total:.1f}%)")
print(f"mHuBERT files exist:              {mhubert_exists} ({100*mhubert_exists/total:.1f}%)")
print(f"Missing mHuBERT files:            {missing_files}")
print(f"Properly aligned (≤{tolerance} frames):    {properly_aligned} ({100*properly_aligned/total:.1f}%)")
print(f"Misaligned (>{tolerance} frames):          {len(alignment_errors)} ({100*len(alignment_errors)/total:.1f}%)")
print(f"{'='*70}")

if alignment_errors:
    print(f"\nWorst alignment mismatches (top 10):")
    alignment_errors.sort(key=lambda x: x['diff'], reverse=True)
    for i, err in enumerate(alignment_errors[:10], 1):
        print(f"  {i}. {Path(err['audio']).name}")
        print(f"     DCAE: {err['dcae']} | mHuBERT: {err['mhubert']} | Diff: {err['diff']}")

print(f"\n✅ You have {properly_aligned} properly aligned mHuBERT features ready for training!")
