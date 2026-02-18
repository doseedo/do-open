#!/usr/bin/env python3
"""
Validate 25% of conditioning paths randomly to ensure they exist.
"""

import json
import random
from pathlib import Path

COND_TYPES = ["amp", "rbend", "rframe", "onsets", "f0", "f0_masked"]


def main():
    manifest_path = "./vocal_training_manifest_yamnet_filtered_REBUILT.json"

    print("="*80)
    print("Validating Conditioning Paths (25% Random Sample)")
    print("="*80)
    print(f"Manifest: {manifest_path}\n")

    # Load manifest
    print("Loading manifest...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    total_entries = len(manifest)
    print(f"Total entries: {total_entries}")

    # Random sample (25%)
    sample_size = int(total_entries * 0.25)
    sample_indices = random.sample(range(total_entries), sample_size)

    print(f"Validating {sample_size} random entries (25%)\n")

    stats = {
        'entries_checked': 0,
        'files_checked': 0,
        'files_exist': 0,
        'files_missing': 0,
        'by_type': {cond: {'exist': 0, 'missing': 0} for cond in COND_TYPES}
    }

    print("Checking files...")
    for i in sample_indices:
        entry = manifest[i]
        stats['entries_checked'] += 1

        cond_paths = entry.get("conditioning_paths", {})

        for cond_type in COND_TYPES:
            path = cond_paths.get(cond_type)
            if not path:
                continue

            stats['files_checked'] += 1

            if Path(path).exists():
                stats['files_exist'] += 1
                stats['by_type'][cond_type]['exist'] += 1
            else:
                stats['files_missing'] += 1
                stats['by_type'][cond_type]['missing'] += 1

    # Report
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Entries checked: {stats['entries_checked']} ({100*stats['entries_checked']/total_entries:.1f}%)")
    print(f"Files checked: {stats['files_checked']}")
    print(f"Files exist: {stats['files_exist']} ({100*stats['files_exist']/stats['files_checked']:.1f}%)")
    print(f"Files missing: {stats['files_missing']} ({100*stats['files_missing']/stats['files_checked']:.1f}%)")
    print()

    print("By conditioning type:")
    for cond_type in COND_TYPES:
        exist = stats['by_type'][cond_type]['exist']
        missing = stats['by_type'][cond_type]['missing']
        total = exist + missing

        if total > 0:
            exist_pct = 100 * exist / total
            print(f"  {cond_type:12s}: {exist:5d} exist, {missing:5d} missing ({exist_pct:.1f}% valid)")

    # Overall verdict
    print()
    if stats['files_exist'] > stats['files_missing']:
        print("✅ PASS: Majority of conditioning paths are valid")
    else:
        print("❌ FAIL: Majority of conditioning paths are missing")


if __name__ == "__main__":
    random.seed(42)  # Reproducible
    main()
