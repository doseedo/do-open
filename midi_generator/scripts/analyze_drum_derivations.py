#!/usr/bin/env python
"""
Analyze drum derivations from discovery results to determine if the drum
handling issue is actually causing problems.

This checks:
1. Are drums being derived from other drums?
2. What transforms are used for drum derivations?
3. Are there suspicious pitch-based drum derivations?
4. What's the error rate for drum vs. melodic derivations?

Usage:
    python scripts/analyze_drum_derivations.py --results-dir ./full_corpus_discovery_results/20251124_085033

Author: Agent - Drum Analysis
"""

import sys
import os
import json
import argparse
from collections import Counter, defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '1_approaches', 'transform_based'))


def is_drum_track(track_id: str) -> bool:
    """
    Heuristic to detect if a track is drums based on track ID.

    Args:
        track_id: Track identifier string

    Returns:
        True if likely a drum track
    """
    track_id_lower = track_id.lower()
    drum_keywords = ['drum', 'percussion', 'perc', 'drums', 'kit', 'beats']
    return any(keyword in track_id_lower for keyword in drum_keywords)


def analyze_iteration_results(iteration_file: str):
    """
    Analyze a single iteration's results for drum handling.

    Args:
        iteration_file: Path to iteration JSON file

    Returns:
        dict: Analysis results
    """
    with open(iteration_file, 'r') as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"Analyzing: {os.path.basename(iteration_file)}")
    print(f"{'='*70}")

    # Get compositions
    compositions = data.get('all_compositions', [])

    # Classify compositions
    pitch_transforms = {'transpose_semitone', 'inversion'}
    rhythm_transforms = {'time_shift', 'time_scale', 'retrograde'}

    compositions_by_type = {
        'pitch_only': [],
        'rhythm_only': [],
        'mixed': [],
        'other': []
    }

    for comp_str, freq in compositions[:100]:  # Top 100
        # Parse composition string
        has_pitch = any(pt in comp_str for pt in pitch_transforms)
        has_rhythm = any(rt in comp_str for rt in rhythm_transforms)

        if has_pitch and not has_rhythm:
            compositions_by_type['pitch_only'].append((comp_str, freq))
        elif has_rhythm and not has_pitch:
            compositions_by_type['rhythm_only'].append((comp_str, freq))
        elif has_pitch and has_rhythm:
            compositions_by_type['mixed'].append((comp_str, freq))
        else:
            compositions_by_type['other'].append((comp_str, freq))

    # Print summary
    print(f"\n## Composition Type Distribution (Top 100):")
    print(f"  Pitch-only:   {len(compositions_by_type['pitch_only'])} compositions")
    print(f"  Rhythm-only:  {len(compositions_by_type['rhythm_only'])} compositions")
    print(f"  Mixed:        {len(compositions_by_type['mixed'])} compositions")
    print(f"  Other:        {len(compositions_by_type['other'])} compositions")

    # Show top rhythm-only (likely drum patterns)
    print(f"\n## Top Rhythm-Only Compositions (Likely Drum Patterns):")
    for i, (comp, freq) in enumerate(compositions_by_type['rhythm_only'][:10], 1):
        print(f"  {i}. {comp}")
        print(f"     Frequency: {freq}")

    # Show suspicious pitch-only with high frequency
    print(f"\n## Pitch-Only Compositions (Should NOT be drums):")
    for i, (comp, freq) in enumerate(compositions_by_type['pitch_only'][:10], 1):
        print(f"  {i}. {comp}")
        print(f"     Frequency: {freq}")

    # Statistics
    stats = data.get('statistics', {})
    print(f"\n## Statistics:")
    print(f"  Total objects:      {stats.get('total_objects', 'N/A')}")
    print(f"  Total derivations:  {stats.get('total_derivations', 'N/A')}")
    print(f"  Derivation rate:    {stats.get('derivation_rate', 0)*100:.1f}%")
    print(f"  Total compositions: {stats.get('total_compositions', 'N/A')}")

    return {
        'iteration': data.get('iteration'),
        'compositions_by_type': compositions_by_type,
        'stats': stats
    }


def check_drum_consistency(corpus_path: str):
    """
    Check if drums use consistent note mappings across the corpus.

    This helps determine if pitch transform false matches are likely.

    Args:
        corpus_path: Path to MIDI corpus
    """
    print(f"\n{'='*70}")
    print("DRUM NOTE CONSISTENCY CHECK")
    print(f"{'='*70}")

    try:
        import mido
    except ImportError:
        print("  ⚠️  mido not available, skipping drum consistency check")
        return

    drum_notes_by_file = defaultdict(Counter)

    print(f"\nScanning MIDI files for drum notes (channel 9)...")

    midi_count = 0
    for root, dirs, files in os.walk(corpus_path):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                if midi_count >= 50:  # Sample 50 files
                    break

                try:
                    midi = mido.MidiFile(os.path.join(root, file))

                    # Extract drum notes (channel 9)
                    for track in midi.tracks:
                        for msg in track:
                            if msg.type == 'note_on' and hasattr(msg, 'channel'):
                                if msg.channel == 9:  # Drum channel
                                    drum_notes_by_file[file][msg.note] += 1

                    midi_count += 1

                except Exception:
                    continue

        if midi_count >= 50:
            break

    if not drum_notes_by_file:
        print("  ℹ️  No drum tracks found in sample")
        return

    print(f"\n  Analyzed {len(drum_notes_by_file)} files with drums")

    # Find most common drum notes across corpus
    all_drum_notes = Counter()
    for notes in drum_notes_by_file.values():
        all_drum_notes.update(notes)

    print(f"\n## Most Common Drum Notes (across sample):")
    gm_drum_map = {
        35: 'Acoustic Bass Drum',
        36: 'Bass Drum 1',
        38: 'Acoustic Snare',
        40: 'Electric Snare',
        42: 'Closed Hi-Hat',
        44: 'Pedal Hi-Hat',
        46: 'Open Hi-Hat',
        49: 'Crash Cymbal 1',
        51: 'Ride Cymbal 1',
        57: 'Crash Cymbal 2',
    }

    for note, count in all_drum_notes.most_common(15):
        instrument = gm_drum_map.get(note, f'Unknown ({note})')
        print(f"  Note {note:3d}: {count:5d} hits - {instrument}")

    # Check consistency
    note_variance = len(all_drum_notes)
    if note_variance > 30:
        print(f"\n  ⚠️  HIGH VARIANCE: {note_variance} different drum notes used")
        print(f"      This suggests either:")
        print(f"      - Rich drum patterns (good)")
        print(f"      - Mixed drum mappings (could cause false matches)")
    else:
        print(f"\n  ✓ CONSISTENT: {note_variance} drum notes used (standard GM mapping)")
        print(f"      False pitch matches unlikely with standard mapping")


def main():
    parser = argparse.ArgumentParser(description='Analyze drum handling in discovery results')
    parser.add_argument('--results-dir', required=True,
                        help='Path to discovery results directory')
    parser.add_argument('--corpus-path',
                        default='/home/arlo/do-repo/midi_generator/midi_corpus/big_band',
                        help='Path to MIDI corpus (for consistency check)')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("DRUM DERIVATION ANALYSIS")
    print(f"{'='*70}")
    print(f"\nResults directory: {args.results_dir}")

    # Find all iteration files
    iteration_files = []
    for file in os.listdir(args.results_dir):
        if file.startswith('iteration_') and file.endswith('.json'):
            iteration_files.append(os.path.join(args.results_dir, file))

    iteration_files.sort()

    if not iteration_files:
        print(f"\n⚠️  No iteration files found in {args.results_dir}")
        return

    print(f"Found {len(iteration_files)} iteration files\n")

    # Analyze each iteration
    results = []
    for iteration_file in iteration_files:
        try:
            result = analyze_iteration_results(iteration_file)
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  Error analyzing {iteration_file}: {e}")

    # Check drum consistency
    if os.path.exists(args.corpus_path):
        check_drum_consistency(args.corpus_path)

    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*70}")

    total_rhythm = sum(len(r['compositions_by_type']['rhythm_only'])
                      for r in results if r)
    total_pitch = sum(len(r['compositions_by_type']['pitch_only'])
                     for r in results if r)

    print(f"\nTotal rhythm-only compositions: {total_rhythm}")
    print(f"Total pitch-only compositions:  {total_pitch}")

    if total_rhythm > total_pitch * 0.2:
        print(f"\n✓ GOOD: Rhythm patterns are being discovered")
        print(f"  Drum pattern discovery appears to be working")
    else:
        print(f"\n⚠️  LOW: Few rhythm-only patterns found")
        print(f"  Drums may not be deriving well")

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    print("""
Based on this analysis:

1. If rhythm-only compositions are common:
   ✓ Drum patterns ARE being discovered correctly
   ✓ Pitch transforms likely have high error (not matching)
   → Optimization (skip pitch transforms) is optional

2. If pitch-only dominates and rhythm-only is rare:
   ⚠️  Drums may be treated as sources (no derivations)
   ⚠️  OR drums absent from corpus
   → Check if corpus actually has drums

3. If seeing suspicious pitch transforms on drums:
   ❌ False matches occurring
   → Implement drum-aware filtering (priority)

Run this after discovery completes to assess actual impact.
""")


if __name__ == '__main__':
    main()
