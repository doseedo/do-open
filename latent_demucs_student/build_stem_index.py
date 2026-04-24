#!/usr/bin/env python3
"""Build a grouped stem index from manifest + Latents2 on GCS FUSE.

Audio paths in the manifest mirror Latents2 paths:
  /home/arlo/gcs-bucket/protools/DATE/New/Session/Audio Files/track.wav
  → /home/arlo/gcs-bucket/Latents2/protools/DATE/New/Session/Audio Files/track.vae.pt

Output: /scratch/latent_demucs_student/stem_group_index.pt
  {
    "guitar": [latent_path1, latent_path2, ...],
    "piano":  [...],
    ...
  }

Usage:
    python build_stem_index.py [--verify-pct 1]
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import torch

MANIFEST = "/home/arlo/gcs-bucket/Manifests/master_manifest_v2.3.json"
GCS_PREFIX = "/home/arlo/gcs-bucket/"
OUTPUT = "/scratch/latent_demucs_student/stem_group_index.pt"

# Groups useful for 6-stem augmentation, mapped to the 6-stem category
GROUP_TO_6STEM = {
    "guitar": "guitar",
    "plucked": "guitar",
    "piano": "piano",
    "keys": "piano",
    "organ": "piano",
    "strings": "other",
    "brass": "other",
    "winds": "other",
    "synth": "other",
    "percussion": "other",
    "mallets": "other",
    "fx": "other",
}


def audio_to_latent_candidates(audio_path):
    """Convert manifest audio path to possible Latents2 latent paths.

    Returns list of candidates — some files are .vae.pt, some are just .pt.
    """
    base = audio_path.replace(GCS_PREFIX, GCS_PREFIX + "Latents2/")
    stem = base.rsplit(".", 1)[0]  # strip .wav/.flac
    return [
        stem + ".vae.pt",
        stem + ".pt",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-pct", type=float, default=1.0,
                    help="% of paths to verify exist on FUSE (0=skip)")
    args = ap.parse_args()

    print("[index] Loading manifest...")
    t0 = time.time()
    m = json.load(open(MANIFEST))
    entries = m["entries"]
    print(f"[index] {len(entries)} entries ({time.time()-t0:.1f}s)")

    # Build grouped index
    by_group = defaultdict(list)     # original group → [latent_path]
    by_6stem = defaultdict(list)     # 6-stem category → [latent_path]
    skipped_group = 0
    skipped_mix = 0

    for audio_path, e in entries.items():
        group = e.get("group", "unknown")
        if group not in GROUP_TO_6STEM:
            skipped_group += 1
            continue
        if e.get("is_mix", False):
            skipped_mix += 1
            continue

        candidates = audio_to_latent_candidates(audio_path)
        by_group[group].append(candidates)
        by_6stem[GROUP_TO_6STEM[group]].append(candidates)

    total = sum(len(v) for v in by_group.values())
    print(f"[index] {total} stems across {len(by_group)} groups "
          f"(skipped {skipped_group} wrong-group, {skipped_mix} mixes)")
    print(f"\nBy original group:")
    for g in sorted(by_group, key=lambda x: -len(by_group[x])):
        print(f"  {g:15s} {len(by_group[g]):6d}")
    print(f"\nBy 6-stem category:")
    for g in sorted(by_6stem, key=lambda x: -len(by_6stem[x])):
        print(f"  {g:15s} {len(by_6stem[g]):6d}")

    # Verify a sample exists on FUSE (try each candidate)
    if args.verify_pct > 0:
        all_candidates = [c for paths in by_group.values() for c in paths]
        n_check = max(1, int(len(all_candidates) * args.verify_pct / 100))
        sample = random.Random(42).sample(all_candidates, min(n_check, len(all_candidates)))
        exists = 0
        for candidates in sample:
            if any(os.path.exists(p) for p in candidates):
                exists += 1
        print(f"\n[verify] {exists}/{len(sample)} checked stems found "
              f"({100*exists/len(sample):.1f}%)")

    # Save both granularities
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    torch.save({
        "by_group": dict(by_group),
        "by_6stem": dict(by_6stem),
    }, OUTPUT)
    print(f"\n[index] Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
