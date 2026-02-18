#!/usr/bin/env python3
"""Analyze which samples are being filtered and why"""
import json
from pathlib import Path
from tqdm import tqdm

manifest_path = "/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS_SPKEMB_MHUBERT.json"
manifest = json.loads(Path(manifest_path).read_text())

total = len(manifest)
filter_reasons = {
    'no_latent': 0,
    'no_encodec': 0,
    'no_piano_roll': 0,
    'passed_filter': 0
}

for item in tqdm(manifest, desc="Analyzing filters"):
    latent_path = item.get("latent_path")
    encodec_path = item.get("encodec_path")
    pr_path = item.get("piano_roll_path")

    lat_ok = latent_path and Path(latent_path).exists()
    enc_ok = encodec_path and Path(encodec_path).exists()
    pr_ok = True if pr_path is None else (pr_path and Path(pr_path).exists())

    if not lat_ok:
        filter_reasons['no_latent'] += 1
    elif not enc_ok:
        filter_reasons['no_encodec'] += 1
    elif not pr_ok:
        filter_reasons['no_piano_roll'] += 1
    else:
        filter_reasons['passed_filter'] += 1

print(f"\n{'='*60}")
print(f"Dataset Filtering Analysis")
print(f"{'='*60}")
print(f"Total samples in manifest:     {total}")
print(f"\nFiltering reasons:")
print(f"  Missing latents:             {filter_reasons['no_latent']} ({100*filter_reasons['no_latent']/total:.1f}%)")
print(f"  Missing encodec:             {filter_reasons['no_encodec']} ({100*filter_reasons['no_encodec']/total:.1f}%)")
print(f"  Missing piano_roll:          {filter_reasons['no_piano_roll']} ({100*filter_reasons['no_piano_roll']/total:.1f}%)")
print(f"  Passed all filters:          {filter_reasons['passed_filter']} ({100*filter_reasons['passed_filter']/total:.1f}%)")
print(f"\nTotal filtered out:            {total - filter_reasons['passed_filter']}")
print(f"{'='*60}")
