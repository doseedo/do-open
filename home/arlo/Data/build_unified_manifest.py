#!/usr/bin/env python3
"""
Build unified manifest combining:
- Audio paths from format_manifest.json
- Labels from combined_manifest.json
- Latent paths from GCS bucket (checking both .pt and .dcae.pt)

Efficient: builds latent file set first, then O(1) lookups.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import time

# Paths
GCS_BUCKET = Path("/home/arlo/gcs-bucket")
LATENTS_DIR = GCS_BUCKET / "Latents"
MANIFESTS_DIR = GCS_BUCKET / "Manifests"

FORMAT_MANIFEST = MANIFESTS_DIR / "format_manifest.json"
COMBINED_MANIFEST = MANIFESTS_DIR / "combined_manifest.json"
OUTPUT_MANIFEST = MANIFESTS_DIR / "unified_manifest.json"


def build_latent_index(latents_dir: Path) -> dict:
    """
    Build index of existing latent files.
    Returns: {relative_stem: extension} e.g. {"protools/.../track": ".dcae.pt"}
    """
    print("Building latent file index...")
    start = time.time()

    latent_index = {}
    count = 0

    # Walk the directory tree once
    for root, dirs, files in os.walk(latents_dir):
        for f in files:
            if f.endswith('.pt'):
                # Get path relative to Latents dir
                full_path = Path(root) / f
                rel_path = full_path.relative_to(latents_dir)

                # Extract stem (without .dcae.pt or .pt)
                if f.endswith('.dcae.pt'):
                    stem = str(rel_path)[:-8]  # Remove .dcae.pt
                    ext = '.dcae.pt'
                else:
                    stem = str(rel_path)[:-3]  # Remove .pt
                    ext = '.pt'

                # Prefer .dcae.pt if both exist
                if stem not in latent_index or ext == '.dcae.pt':
                    latent_index[stem] = ext

                count += 1
                if count % 50000 == 0:
                    print(f"  Indexed {count:,} latent files...")

    elapsed = time.time() - start
    print(f"  Indexed {count:,} latent files in {elapsed:.1f}s")
    return latent_index


def audio_path_to_latent_stem(audio_path: str) -> str:
    """
    Convert audio path to latent stem for index lookup.
    audio: protools/2025-03-28/New/session/Audio Files/track.wav
    stem:  protools/2025-03-28/New/session/Audio Files/track
    """
    # Handle both relative and absolute paths
    if audio_path.startswith('/home/arlo/gcs-bucket/'):
        audio_path = audio_path[len('/home/arlo/gcs-bucket/'):]

    # Remove extension
    p = Path(audio_path)
    return str(p.parent / p.stem)


def main():
    start_total = time.time()

    # Step 1: Build latent index
    latent_index = build_latent_index(LATENTS_DIR)

    # Step 2: Load combined_manifest for labels
    print(f"\nLoading labels from {COMBINED_MANIFEST.name}...")
    start = time.time()
    with open(COMBINED_MANIFEST, 'r') as f:
        combined = json.load(f)
    print(f"  Loaded {len(combined):,} label entries in {time.time()-start:.1f}s")

    # Build label lookup by normalized path
    label_lookup = {}
    for path, data in combined.items():
        # Normalize path for lookup
        if path.startswith('/home/arlo/gcs-bucket/'):
            norm_path = path[len('/home/arlo/gcs-bucket/'):]
        else:
            norm_path = path
        label_lookup[norm_path] = data

    # Step 3: Load format_manifest
    print(f"\nLoading audio entries from {FORMAT_MANIFEST.name}...")
    start = time.time()
    with open(FORMAT_MANIFEST, 'r') as f:
        format_data = json.load(f)
    entries = format_data.get('entries', [])
    print(f"  Loaded {len(entries):,} entries in {time.time()-start:.1f}s")

    # Step 4: Build unified manifest
    print("\nBuilding unified manifest...")
    start = time.time()

    unified_entries = []
    stats = defaultdict(int)

    for i, entry in enumerate(entries):
        audio_path = entry.get('path', '')

        # Get latent info
        latent_stem = audio_path_to_latent_stem(audio_path)
        latent_ext = latent_index.get(latent_stem)

        if latent_ext:
            latent_path = f"/home/arlo/gcs-bucket/Latents/{latent_stem}{latent_ext}"
            has_latent = True
            stats[f'latent_{latent_ext}'] += 1
        else:
            latent_path = None
            has_latent = False
            stats['no_latent'] += 1

        # Get labels
        labels = label_lookup.get(audio_path, {})
        group = labels.get('group', 'undefined')
        subgroup = labels.get('subgroup', 'undefined')

        if labels:
            stats['has_labels'] += 1
        else:
            stats['no_labels'] += 1

        stats[f'group_{group}'] += 1

        # Build unified entry
        unified_entry = {
            'audio_path': f"/home/arlo/gcs-bucket/{audio_path}",
            'latent_path': latent_path,
            'has_latent': has_latent,
            'group': group,
            'subgroup': subgroup,
            'source': entry.get('source', ''),
            'has_conditioning': entry.get('has_conditioning', False),
            'has_midi': entry.get('has_midi', False),
        }
        unified_entries.append(unified_entry)

        if (i + 1) % 100000 == 0:
            print(f"  Processed {i+1:,} entries...")

    print(f"  Processed {len(unified_entries):,} entries in {time.time()-start:.1f}s")

    # Step 5: Save unified manifest
    print(f"\nSaving to {OUTPUT_MANIFEST.name}...")
    start = time.time()

    output = {
        'generated_at': time.time(),
        'generated_at_iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'total_entries': len(unified_entries),
        'stats': {
            'with_latent': stats['latent_.dcae.pt'] + stats['latent_.pt'],
            'latent_dcae_pt': stats['latent_.dcae.pt'],
            'latent_pt': stats['latent_.pt'],
            'no_latent': stats['no_latent'],
            'has_labels': stats['has_labels'],
            'no_labels': stats['no_labels'],
        },
        'groups': {k.replace('group_', ''): v for k, v in stats.items() if k.startswith('group_')},
        'entries': unified_entries,
    }

    with open(OUTPUT_MANIFEST, 'w') as f:
        json.dump(output, f)

    print(f"  Saved in {time.time()-start:.1f}s")

    # Summary
    total_time = time.time() - start_total
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total entries:     {len(unified_entries):,}")
    print(f"With latent:       {output['stats']['with_latent']:,}")
    print(f"  - .dcae.pt:      {stats['latent_.dcae.pt']:,}")
    print(f"  - .pt:           {stats['latent_.pt']:,}")
    print(f"No latent:         {stats['no_latent']:,}")
    print(f"Has labels:        {stats['has_labels']:,}")
    print(f"No labels:         {stats['no_labels']:,}")
    print(f"\nTop groups:")
    for group, count in sorted(output['groups'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {group}: {count:,}")
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Output: {OUTPUT_MANIFEST}")


if __name__ == '__main__':
    main()
