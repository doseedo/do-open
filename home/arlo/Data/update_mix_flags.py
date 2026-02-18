#!/usr/bin/env python3
"""
Update master manifest with is_mix=true for all known mix files.

Sources:
1. Multi-label corrections with 2+ instruments
2. Files tagged as "ensemble" or "full-track" in corrections
3. Files tagged as "ensemble" or "full-track" in manifest
4. Files with "mix" in the filename
5. Files tagged as "room" (capture multiple instruments)
"""

import orjson
import json
from pathlib import Path
from datetime import datetime

MASTER_MANIFEST = Path("/home/arlo/gcs-bucket/Manifests/master_manifest.json")
UNIFIED_MANIFEST = Path("/home/arlo/gcs-bucket/Manifests/unified_manifest.json")
CORRECTIONS_FILE = Path("/home/arlo/gcs-bucket/Manifests/corrections.json")

def main():
    print("=" * 60)
    print("UPDATING is_mix FLAGS IN MASTER MANIFEST")
    print("=" * 60)

    # Load master manifest
    print(f"\nLoading master manifest from {MASTER_MANIFEST}...")
    with open(MASTER_MANIFEST, 'rb') as f:
        master = orjson.loads(f.read())

    entries = master.get('entries', {})
    print(f"  Total entries: {len(entries)}")

    # Track what we're marking
    mix_paths = set()
    sources = {
        'multi_label_correction': set(),
        'ensemble_correction': set(),
        'fulltrack_correction': set(),
        'ensemble_manifest': set(),
        'fulltrack_manifest': set(),
        'room_manifest': set(),
        'mix_filename': set(),
        'already_marked': set(),
    }

    # 1. Load corrections - multi-label with 2+ instruments
    print(f"\nLoading corrections from {CORRECTIONS_FILE}...")
    with open(CORRECTIONS_FILE) as f:
        corrections = json.load(f)
    print(f"  Total corrections: {len(corrections)}")

    for path, data in corrections.items():
        # Multi-label with 2+ instruments in any region
        if data.get('multi_label'):
            regions = data.get('regions', [])
            for r in regions:
                if len(r.get('labels', [])) >= 2:
                    mix_paths.add(path)
                    sources['multi_label_correction'].add(path)
                    break

        # Tagged as ensemble
        if data.get('group') == 'ensemble':
            mix_paths.add(path)
            sources['ensemble_correction'].add(path)

        # Tagged as full-track
        if data.get('group') == 'full-track':
            mix_paths.add(path)
            sources['fulltrack_correction'].add(path)

    # 2. Check unified manifest for ensemble/full-track/room groups
    print(f"\nLoading unified manifest from {UNIFIED_MANIFEST}...")
    with open(UNIFIED_MANIFEST, 'rb') as f:
        unified = orjson.loads(f.read())

    unified_entries = unified.get('entries', [])
    print(f"  Total unified entries: {len(unified_entries)}")

    for entry in unified_entries:
        if not isinstance(entry, dict):
            continue

        audio_path = entry.get('audio_path', '')
        group = entry.get('group', '')

        if group == 'ensemble':
            mix_paths.add(audio_path)
            sources['ensemble_manifest'].add(audio_path)
        elif group == 'full-track':
            mix_paths.add(audio_path)
            sources['fulltrack_manifest'].add(audio_path)
        elif group == 'room':
            mix_paths.add(audio_path)
            sources['room_manifest'].add(audio_path)

    # 3. Check master manifest for ensemble/full-track/room and mix filenames
    for path, data in entries.items():
        if not isinstance(data, dict):
            continue

        group = data.get('group', '')

        # Already marked
        if data.get('is_mix'):
            sources['already_marked'].add(path)

        # Group-based
        if group == 'ensemble':
            mix_paths.add(path)
            sources['ensemble_manifest'].add(path)
        elif group == 'full-track':
            mix_paths.add(path)
            sources['fulltrack_manifest'].add(path)
        elif group == 'room':
            mix_paths.add(path)
            sources['room_manifest'].add(path)

        # Filename-based - check for 'mix' keyword
        fname_lower = path.lower()
        if 'mix' in fname_lower.split('/')[-1]:  # Only check filename, not full path
            mix_paths.add(path)
            sources['mix_filename'].add(path)

    # Print summary
    print("\n" + "=" * 60)
    print("MIX SOURCES FOUND:")
    print("=" * 60)
    print(f"  Multi-label corrections (2+ instruments): {len(sources['multi_label_correction'])}")
    print(f"  Ensemble corrections: {len(sources['ensemble_correction'])}")
    print(f"  Full-track corrections: {len(sources['fulltrack_correction'])}")
    print(f"  Ensemble in manifest: {len(sources['ensemble_manifest'])}")
    print(f"  Full-track in manifest: {len(sources['fulltrack_manifest'])}")
    print(f"  Room in manifest: {len(sources['room_manifest'])}")
    print(f"  'mix' in filename: {len(sources['mix_filename'])}")
    print(f"  Already marked is_mix: {len(sources['already_marked'])}")
    print(f"\n  TOTAL UNIQUE MIX PATHS: {len(mix_paths)}")

    # Update master manifest
    print("\n" + "=" * 60)
    print("UPDATING MASTER MANIFEST")
    print("=" * 60)

    updated_count = 0
    new_count = 0

    for path in mix_paths:
        if path in entries:
            if not entries[path].get('is_mix'):
                entries[path]['is_mix'] = True
                updated_count += 1
        else:
            # Path not in master manifest - might be in unified only
            # We'll skip these for now
            pass

    print(f"  Updated {updated_count} existing entries with is_mix=true")
    print(f"  (Paths not in master manifest: {len(mix_paths) - updated_count - len(sources['already_marked'])})")

    # Save backup
    backup_path = MASTER_MANIFEST.with_suffix('.json.bak')
    print(f"\nSaving backup to {backup_path}...")
    with open(backup_path, 'wb') as f:
        f.write(orjson.dumps(master, option=orjson.OPT_INDENT_2))

    # Update metadata
    master['is_mix_updated_at'] = datetime.now().isoformat()
    master['is_mix_count'] = sum(1 for e in entries.values() if isinstance(e, dict) and e.get('is_mix'))

    # Save updated manifest
    print(f"Saving updated manifest to {MASTER_MANIFEST}...")
    with open(MASTER_MANIFEST, 'wb') as f:
        f.write(orjson.dumps(master, option=orjson.OPT_INDENT_2))

    print(f"\nDone! Total is_mix=true: {master['is_mix_count']}")

    # Print some examples
    print("\nExample mix files marked:")
    examples = list(mix_paths)[:10]
    for p in examples:
        print(f"  {Path(p).name}")


if __name__ == "__main__":
    main()
