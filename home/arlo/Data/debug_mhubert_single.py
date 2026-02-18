#!/usr/bin/env python3
"""Debug a single mHuBERT file"""
import json
import torch
from pathlib import Path

manifest_path = "/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS_SPKEMB_MHUBERT.json"
manifest = json.loads(Path(manifest_path).read_text())

# Get first item with mHuBERT
for item in manifest:
    mhubert_path = item.get("mhubert_features_path")
    if mhubert_path and mhubert_path != "null":
        break

print(f"Testing file: {item.get('audio_path')}")
print(f"Latent path: {item.get('latent_path')}")
print(f"mHuBERT path: {mhubert_path}")
print()

# Load DCAE latents
print("Loading DCAE latents...")
latents = torch.load(item.get('latent_path'), map_location='cpu')
print(f"  Type: {type(latents)}")
if isinstance(latents, dict):
    print(f"  Keys: {latents.keys()}")
    if 'latents' in latents:
        latents = latents['latents']
    elif 'latent' in latents:
        latents = latents['latent']
print(f"  Shape: {latents.shape}")
T_slow_dcae = latents.shape[2] if latents.dim() == 3 else latents.shape[1]
print(f"  T_slow_dcae: {T_slow_dcae}")
print()

# Load mHuBERT features
print("Loading mHuBERT features...")
mhubert_data = torch.load(mhubert_path, map_location='cpu')
print(f"  Type: {type(mhubert_data)}")
if isinstance(mhubert_data, dict):
    print(f"  Keys: {mhubert_data.keys()}")
    if 'aligned_features' in mhubert_data:
        mhubert_features = mhubert_data['aligned_features']
        print(f"  Using 'aligned_features'")
    elif 'features' in mhubert_data:
        mhubert_features = mhubert_data['features']
        print(f"  Using 'features'")
    else:
        print("  ERROR: No features found!")
        exit(1)
else:
    mhubert_features = mhubert_data

print(f"  Shape: {mhubert_features.shape}")
T_slow_mhubert = mhubert_features.shape[0]
print(f"  T_slow_mhubert: {T_slow_mhubert}")
print()

# Check alignment
diff = abs(T_slow_dcae - T_slow_mhubert)
print(f"Alignment:")
print(f"  DCAE frames: {T_slow_dcae}")
print(f"  mHuBERT frames: {T_slow_mhubert}")
print(f"  Difference: {diff}")
print(f"  Aligned (≤2 frames)? {diff <= 2}")
