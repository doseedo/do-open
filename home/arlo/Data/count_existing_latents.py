#!/usr/bin/env python3
"""Count how many latents actually exist"""
import json
import torch
from pathlib import Path
from tqdm import tqdm

manifest_path = "/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS_SPKEMB_MHUBERT.json"
manifest = json.loads(Path(manifest_path).read_text())

total = len(manifest)
latents_exist = 0
mhubert_exist = 0
both_exist = 0
aligned = 0

for item in tqdm(manifest, desc="Checking files"):
    latent_path = item.get("latent_path")
    mhubert_path = item.get("mhubert_features_path")

    lat_exists = latent_path and Path(latent_path).exists()
    mhu_exists = mhubert_path and mhubert_path != "null" and Path(mhubert_path).exists()

    if lat_exists:
        latents_exist += 1
    if mhu_exists:
        mhubert_exist += 1
    if lat_exists and mhu_exists:
        both_exist += 1

        # Check alignment
        try:
            latents = torch.load(latent_path, map_location='cpu')
            if isinstance(latents, dict):
                if 'latents' in latents:
                    latents = latents['latents']
                elif 'latent' in latents:
                    latents = latents['latent']

            mhubert_data = torch.load(mhubert_path, map_location='cpu')
            if isinstance(mhubert_data, dict):
                mhubert_features = mhubert_data.get('aligned_features') or mhubert_data.get('features')
            else:
                mhubert_features = mhubert_data

            if isinstance(latents, torch.Tensor) and isinstance(mhubert_features, torch.Tensor):
                T_dcae = latents.shape[2] if latents.dim() == 3 else latents.shape[1]
                T_mhubert = mhubert_features.shape[0]
                if abs(T_dcae - T_mhubert) <= 2:
                    aligned += 1
        except Exception as e:
            pass

print(f"\n{'='*60}")
print(f"Total samples:              {total}")
print(f"Latents exist:              {latents_exist} ({100*latents_exist/total:.1f}%)")
print(f"mHuBERT exist:              {mhubert_exist} ({100*mhubert_exist/total:.1f}%)")
print(f"Both exist:                 {both_exist} ({100*both_exist/total:.1f}%)")
print(f"Properly aligned (≤2 fr):   {aligned} ({100*aligned/total:.1f}%)")
print(f"{'='*60}")
