#!/usr/bin/env python3
"""Analyze what features exist for samples missing latents"""
import json
from pathlib import Path
from tqdm import tqdm

manifest_path = "/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS_SPKEMB_MHUBERT.json"
manifest = json.loads(Path(manifest_path).read_text())

# Find samples without latents
missing_latent_samples = []
for item in tqdm(manifest, desc="Finding missing latents"):
    latent_path = item.get("latent_path")
    if not (latent_path and Path(latent_path).exists()):
        missing_latent_samples.append(item)

print(f"\nFound {len(missing_latent_samples)} samples without latents")
print("Checking what features they have...\n")

# Analyze what features exist
stats = {
    'audio_exists': 0,
    'encodec_exists': 0,
    'piano_roll_exists': 0,
    'mhubert_exists': 0,
    'speaker_emb_exists': 0,
    'vocal_conditioning_exists': 0
}

for item in tqdm(missing_latent_samples, desc="Analyzing features"):
    audio_path = item.get("audio_path")
    encodec_path = item.get("encodec_path")
    pr_path = item.get("piano_roll_path")
    mhubert_path = item.get("mhubert_features_path")
    spk_emb_path = item.get("speaker_emb_path")
    vocal_cond_paths = item.get("vocal_conditioning_paths") or {}

    if audio_path and Path(audio_path).exists():
        stats['audio_exists'] += 1

    if encodec_path and Path(encodec_path).exists():
        stats['encodec_exists'] += 1

    if pr_path and Path(pr_path).exists():
        stats['piano_roll_exists'] += 1

    if mhubert_path and mhubert_path != "null" and Path(mhubert_path).exists():
        stats['mhubert_exists'] += 1

    if spk_emb_path and spk_emb_path != "null" and Path(spk_emb_path).exists():
        stats['speaker_emb_exists'] += 1

    # Check if vocal conditioning exists
    if vocal_cond_paths:
        lyrics_data_path = vocal_cond_paths.get("lyrics_data")
        if lyrics_data_path and Path(lyrics_data_path).exists():
            stats['vocal_conditioning_exists'] += 1

total = len(missing_latent_samples)
print(f"\n{'='*70}")
print(f"Features for {total} samples WITHOUT latents:")
print(f"{'='*70}")
print(f"Audio files exist:           {stats['audio_exists']:5d} ({100*stats['audio_exists']/total:5.1f}%)")
print(f"EnCodec exists:              {stats['encodec_exists']:5d} ({100*stats['encodec_exists']/total:5.1f}%)")
print(f"Piano roll exists:           {stats['piano_roll_exists']:5d} ({100*stats['piano_roll_exists']/total:5.1f}%)")
print(f"mHuBERT exists:              {stats['mhubert_exists']:5d} ({100*stats['mhubert_exists']/total:5.1f}%)")
print(f"Speaker embedding exists:    {stats['speaker_emb_exists']:5d} ({100*stats['speaker_emb_exists']/total:5.1f}%)")
print(f"Vocal conditioning exists:   {stats['vocal_conditioning_exists']:5d} ({100*stats['vocal_conditioning_exists']/total:5.1f}%)")
print(f"{'='*70}")

# Show a few examples
print(f"\nExample missing latent paths (first 5):")
for i, item in enumerate(missing_latent_samples[:5]):
    print(f"\n{i+1}. Audio: {Path(item.get('audio_path', '')).name}")
    print(f"   Latent: {item.get('latent_path', 'None')}")
    print(f"   Exists: audio={Path(item.get('audio_path', '')).exists()}, "
          f"encodec={item.get('encodec_path') and Path(item.get('encodec_path')).exists()}, "
          f"mhubert={item.get('mhubert_features_path') and Path(item.get('mhubert_features_path', '')).exists()}")

print(f"\n💡 Recommendation:")
if stats['audio_exists'] > total * 0.9:
    print(f"   ✅ {stats['audio_exists']}/{total} audio files exist!")
    print(f"   You should extract DCAE latents for these files to use them in training.")
else:
    print(f"   ⚠️  Only {stats['audio_exists']}/{total} audio files exist.")
    print(f"   Consider cleaning the manifest to remove entries with missing audio.")
