#!/usr/bin/env python3
"""
Export ground-truth labeled mixes from the monitor service API.
Outputs JSON file for use with learn_mix_transform_joint.py

Usage:
  python3 export_gt_labels.py --output gt_mix_labels.json
  python3 export_gt_labels.py --output gt_mix_labels.json --classes brass,strings,winds
"""

import json
import argparse
import requests
from pathlib import Path


def fetch_labeled_paths(page_size=500):
    """Fetch all labeled paths from the API."""
    all_items = []
    page = 1

    while True:
        try:
            resp = requests.get(
                f"http://localhost:8096/api/labeled-paths",
                params={"page": page, "pageSize": page_size}
            )
            resp.raise_for_status()
            data = resp.json()

            items = data.get('items', [])
            all_items.extend(items)

            total = data.get('total', 0)
            print(f"Fetched page {page}: {len(items)} items (total: {len(all_items)}/{total})")

            if len(all_items) >= total or not items:
                break

            page += 1
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break

    return all_items


def filter_mix_files(items, target_classes=None):
    """
    Filter for mix files (multi-instrument tracks).

    A file is considered a "mix" if:
    1. It has multiple labels, OR
    2. It contains "mix" in the filename/path

    If target_classes specified, only include files that have at least one of those classes.
    """
    mix_files = []

    for item in items:
        path = item.get('path', '')
        labels = item.get('labels', [])

        if not labels:
            continue

        # Check if it's a mix (multiple labels or "mix" in path)
        is_multi = len(labels) > 1
        has_mix_in_name = 'mix' in path.lower()

        # For training, we want files with our target classes
        if target_classes:
            has_target = any(lbl in target_classes for lbl in labels)
            if not has_target:
                continue

        if is_multi or has_mix_in_name or target_classes:
            # Convert audio path to latent path
            latent_path = path.replace('/gcs-bucket/protools/', '/gcs-bucket/Latents/protools/')
            latent_path = latent_path.replace('.wav', '.pt').replace('.mp3', '.pt')

            mix_files.append({
                'audio_path': path,
                'latent_path': latent_path,
                'labels': labels,
                'is_multi_label': is_multi
            })

    return mix_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='gt_mix_labels.json')
    parser.add_argument('--classes', type=str, default=None,
                        help='Comma-separated list of target classes (e.g., brass,strings,winds)')
    parser.add_argument('--min-labels', type=int, default=1,
                        help='Minimum number of labels per file')
    args = parser.parse_args()

    target_classes = None
    if args.classes:
        target_classes = [c.strip() for c in args.classes.split(',')]
        print(f"Filtering for target classes: {target_classes}")

    print("Fetching labeled paths from monitor service...")
    items = fetch_labeled_paths()
    print(f"\nTotal labeled items: {len(items)}")

    # Filter for mixes
    mix_files = filter_mix_files(items, target_classes)
    print(f"Files with target classes: {len(mix_files)}")

    # Filter by min labels
    if args.min_labels > 1:
        mix_files = [m for m in mix_files if len(m['labels']) >= args.min_labels]
        print(f"Files with >= {args.min_labels} labels: {len(mix_files)}")

    # Check which latent files exist
    existing = []
    for m in mix_files:
        latent_path = Path(m['latent_path'])
        if latent_path.exists():
            existing.append(m)
        else:
            # Try alternate path
            alt = m['latent_path'].replace('/Latents/protools/', '/Latents/')
            if Path(alt).exists():
                m['latent_path'] = alt
                existing.append(m)

    print(f"Files with existing latents: {len(existing)}")

    if not existing:
        print("No matching files found!")
        return

    # Convert to simple format for training
    output_data = {m['latent_path']: m['labels'] for m in existing}

    # Show label distribution
    label_counts = {}
    for m in existing:
        for lbl in m['labels']:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print(f"\nLabel distribution:")
    for lbl, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {lbl}: {count}")

    # Save
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(output_data)} labeled files to: {output_path}")
    print(f"\nTo train:")
    print(f"python3 learn_mix_transform_joint.py --gt-labels {output_path}")


if __name__ == "__main__":
    main()
