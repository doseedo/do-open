#!/usr/bin/env python3
"""
Filter Missing Paths from Master Manifest

Removes entries where the audio file no longer exists.
"""

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/master_manifest.json")
OUTPUT_PATH = Path("/home/arlo/gcs-bucket/Manifests/master_manifest.json")  # Overwrite
BACKUP_PATH = Path("/home/arlo/gcs-bucket/Manifests/master_manifest_backup.json")

def check_exists(path):
    """Check if file exists (returns tuple for mapping)."""
    return path, os.path.exists(path)

def main():
    print("Loading manifest...")
    with open(MANIFEST_PATH) as f:
        data = json.load(f)

    entries = data.get('entries', {})
    total = len(entries)
    print(f"Total entries: {total:,}")

    # Check all paths in parallel (much faster for GCS)
    print("Checking file existence (this may take a few minutes)...")
    paths = list(entries.keys())

    valid_paths = set()
    missing_count = 0

    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(check_exists, p): p for p in paths}

        for i, future in enumerate(as_completed(futures)):
            path, exists = future.result()
            if exists:
                valid_paths.add(path)
            else:
                missing_count += 1

            if (i + 1) % 10000 == 0:
                print(f"  Checked {i+1:,}/{total:,} - {missing_count:,} missing so far")

    print(f"\nResults:")
    print(f"  Valid: {len(valid_paths):,}")
    print(f"  Missing: {missing_count:,}")

    # Filter entries
    filtered_entries = {p: e for p, e in entries.items() if p in valid_paths}

    # Update stats
    data['entries'] = filtered_entries
    data['stats']['total_entries'] = len(filtered_entries)
    data['stats']['filtered_missing'] = missing_count

    # Backup original
    print(f"\nBacking up to {BACKUP_PATH}...")
    with open(BACKUP_PATH, 'w') as f:
        json.dump({'entries': entries, 'stats': data.get('stats', {})}, f)

    # Write filtered
    print(f"Writing filtered manifest to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
