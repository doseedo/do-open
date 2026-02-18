#!/usr/bin/env python3
"""
Add technique tags to unified_manifest.json

CONSERVATIVE approach - only techniques that are stable per-file:
- Strings: pizz, arco, tremolo (from filename)
- Brass: muted, open (from mute_manifest only)
- Upright bass: pizz, arco (from filename)
- Drums: brushes, sticks (from filename)
- Voice: male, female (from vocal gendered manifest - lower weight)

NOT included (too variable within recordings):
- Guitar clean/distorted, palm mute, etc.
- Piano sustain/staccato
- Most playing articulations
"""

import re
from pathlib import Path
from collections import Counter
import orjson

# Paths
UNIFIED_MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/unified_manifest.json")
MUTE_MANIFEST_PATH = Path("/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json")
VOCAL_MANIFEST_PATH = Path("/home/arlo/Data/vocal_training_manifest_gendered.json")
OUTPUT_PATH = Path("/home/arlo/gcs-bucket/Manifests/unified_manifest.json")

# Technique patterns - CONSERVATIVE, only stable per-file techniques
# Format: (group, subgroup or None, pattern_regex, technique)
TECHNIQUE_PATTERNS = [
    # Strings - bow technique is usually consistent per take
    ('strings', 'violin', r'\bpizz\b|pizzicato|_pizz[_\.]', 'pizz'),
    ('strings', 'violin', r'\barco\b|_arco[_\.]', 'arco'),
    ('strings', 'violin', r'\btrem\b|tremolo|_trem[_\.]', 'tremolo'),
    ('strings', 'viola', r'\bpizz\b|pizzicato|_pizz[_\.]', 'pizz'),
    ('strings', 'viola', r'\barco\b|_arco[_\.]', 'arco'),
    ('strings', 'viola', r'\btrem\b|tremolo|_trem[_\.]', 'tremolo'),
    ('strings', 'cello', r'\bpizz\b|pizzicato|_pizz[_\.]', 'pizz'),
    ('strings', 'cello', r'\barco\b|_arco[_\.]', 'arco'),
    ('strings', 'cello', r'\btrem\b|tremolo|_trem[_\.]', 'tremolo'),
    ('strings', 'double_bass', r'\bpizz\b|pizzicato|_pizz[_\.]', 'pizz'),
    ('strings', 'double_bass', r'\barco\b|_arco[_\.]', 'arco'),
    ('strings', None, r'\bpizz\b|pizzicato|_pizz[_\.]', 'pizz'),
    ('strings', None, r'\barco\b|_arco[_\.]', 'arco'),
    ('strings', None, r'\btrem\b|tremolo|_trem[_\.]', 'tremolo'),

    # Bass (upright) - bow vs pluck is consistent per take
    ('bass', 'upright_bass', r'\bpizz\b|pizzicato|_pizz[_\.]', 'pizz'),
    ('bass', 'upright_bass', r'\barco\b|_arco[_\.]', 'arco'),

    # Drums - stick type is usually consistent per session
    ('drums', None, r'\bbrush\b|brushes|_brush[_\.]', 'brushes'),
    ('drums', None, r'\bsticks?\b|_stick[_\.]', 'sticks'),
    ('drums', None, r'\bmallets?\b|_mallet[_\.]', 'mallets'),
    ('drums', None, r'\brods?\b|_rod[_\.]|hot\s?rods?', 'rods'),
]

# Compile patterns
COMPILED_PATTERNS = [
    (group, subgroup, re.compile(pattern, re.IGNORECASE), technique)
    for group, subgroup, pattern, technique in TECHNIQUE_PATTERNS
]


def detect_technique_from_filename(filename: str, group: str, subgroup: str):
    """Detect technique from filename patterns."""
    filename_lower = filename.lower()

    for pat_group, pat_subgroup, pattern, technique in COMPILED_PATTERNS:
        # Must match group
        if pat_group != group:
            continue

        # If pattern specifies subgroup, must match (or subgroup is None for wildcard)
        if pat_subgroup is not None and pat_subgroup != subgroup:
            continue

        # Check pattern
        if pattern.search(filename_lower):
            return technique

    return None


def main():
    print("Loading manifests...")

    # Load unified manifest
    with open(UNIFIED_MANIFEST_PATH, 'rb') as f:
        manifest = orjson.loads(f.read())

    entries = manifest.get('entries', [])
    print(f"Unified manifest: {len(entries)} entries")

    # Load mute manifest (brass only)
    mute_lookup = {}
    if MUTE_MANIFEST_PATH.exists():
        with open(MUTE_MANIFEST_PATH, 'rb') as f:
            mute_data = orjson.loads(f.read())
        mute_lookup = {d['audio_path']: d.get('is_muted') for d in mute_data}
        print(f"Mute manifest: {len(mute_lookup)} entries")

    # Load vocal gender manifest
    vocal_gender_lookup = {}
    if VOCAL_MANIFEST_PATH.exists():
        with open(VOCAL_MANIFEST_PATH, 'rb') as f:
            vocal_data = orjson.loads(f.read())
        # Only include entries with actual gender labels
        for v in vocal_data:
            gender = v.get('gender') or v.get('predicted_gender')
            if gender in ('male', 'female'):
                vocal_gender_lookup[v['audio_path']] = gender
        print(f"Vocal gender manifest: {len(vocal_gender_lookup)} entries with gender")

    # Stats
    technique_counts = Counter()
    technique_sources = Counter()
    updated_count = 0

    print("\nProcessing entries...")
    for entry in entries:
        audio_path = entry.get('audio_path', '')
        group = entry.get('group', 'undefined')
        subgroup = entry.get('subgroup', 'undefined')
        filename = Path(audio_path).name if audio_path else ''

        technique = None
        source = None

        # 1. Brass: mute manifest (high confidence)
        if group == 'brass' and audio_path in mute_lookup:
            is_muted = mute_lookup[audio_path]
            if is_muted is True:
                technique = 'muted'
                source = 'mute_manifest'
            elif is_muted is False:
                technique = 'open'
                source = 'mute_manifest'

        # 2. Voice: gender manifest (model-predicted, mark source)
        elif group == 'voice' and audio_path in vocal_gender_lookup:
            technique = vocal_gender_lookup[audio_path]
            source = 'vocal_gender_model'

        # 3. Filename detection for strings, bass, drums
        elif group in ('strings', 'bass', 'drums') and group not in ('undefined', 'unknown'):
            detected = detect_technique_from_filename(filename, group, subgroup)
            if detected:
                technique = detected
                source = 'filename'

        # Update entry
        if technique:
            entry['technique'] = technique
            technique_counts[f"{group}/{subgroup}/{technique}"] += 1
            technique_sources[source] += 1
            updated_count += 1
        else:
            entry['technique'] = None

    print(f"\nUpdated {updated_count} entries with technique labels")
    print(f"\nSources: {dict(technique_sources)}")

    print(f"\nTechnique combinations:")
    for combo, count in technique_counts.most_common(30):
        print(f"  {combo}: {count}")

    # Update manifest metadata
    from datetime import datetime
    manifest['technique_tags_added_at'] = datetime.now().isoformat()
    manifest['technique_sources'] = dict(technique_sources)

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

    print("Done!")


if __name__ == '__main__':
    main()
