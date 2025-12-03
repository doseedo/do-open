#!/usr/bin/env python
"""
Multi-View Pattern Discovery runner.

Generates pairs from multiple musical perspectives (orchestration, temporal,
cross-track, cross-piece), mines compounds on each view, then unifies via MDL.

Usage:
    python scripts/run_multi_view.py --max-files 25 --output-dir ./vocab_first_v8

Author: Multi-View MDL Discovery
"""

import sys
import os
import argparse
import mido
import json
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '1_approaches', 'transform_based'))

from core.hierarchical_corpus import HierarchicalMIDICorpus
from discovery.emergent_hierarchy import EmergentHierarchyDiscovery
from discovery.multi_view_discovery import MultiViewDiscovery, MusicalObject, ViewType

import torch
import numpy as np

# Fixed scales and meters
STANDARD_SCALES = [16, 32, 64, 128, 256]
ALLOWED_METERS = {(4, 4), (2, 4), (3, 4), (6, 8)}


def get_midi_time_signatures(midi_file):
    time_sigs = set()
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                time_sigs.add((msg.numerator, msg.denominator))
    if not time_sigs:
        time_sigs.add((4, 4))
    return time_sigs


def has_allowed_meter(midi_file):
    return get_midi_time_signatures(midi_file).issubset(ALLOWED_METERS)


def convert_objects_to_musical_objects(objects: list) -> list:
    """
    Convert EmergentHierarchy MusicalObjects to MultiView MusicalObjects.

    The input objects are dataclass instances with:
    - piece_id, track_id, start_time, end_time, tensor (T, F)

    We extract pitch class, interval, contour, and rhythm features
    from the tensor representation.
    """
    musical_objects = []

    for obj in objects:
        # EmergentHierarchy MusicalObject has these attributes directly
        piece_id = getattr(obj, 'piece_id', 'unknown')
        track_id = getattr(obj, 'track_id', 'track_0')
        start_time = getattr(obj, 'start_time', 0)
        end_time = getattr(obj, 'end_time', start_time + 64)
        tensor = getattr(obj, 'tensor', None)

        # Convert start/end time to bar indices (assuming 16 ticks per bar)
        bar_start = start_time // 16
        bar_end = end_time // 16

        # Extract features from tensor
        # The tensor is (T, F) where F typically includes pitch info
        if tensor is not None and len(tensor) > 0:
            tensor_np = np.array(tensor)

            # Sum across time to get aggregate pitch profile
            if len(tensor_np.shape) == 2:
                time_sum = tensor_np.sum(axis=0)  # (F,)
            else:
                time_sum = tensor_np

            # Extract pitch classes (first 12 or 128 dimensions)
            if len(time_sum) >= 128:
                # Piano roll format: 128 pitches
                pitch_classes = torch.zeros(12)
                for i in range(128):
                    pitch_classes[i % 12] += time_sum[i]
            elif len(time_sum) >= 12:
                # Already pitch class format
                pitch_classes = torch.tensor(time_sum[:12], dtype=torch.float32)
            else:
                pitch_classes = torch.zeros(12)

            if pitch_classes.sum() > 0:
                pitch_classes = pitch_classes / pitch_classes.sum()

            # Compute interval profile from tensor
            intervals = torch.zeros(12)
            if len(tensor_np.shape) == 2 and tensor_np.shape[0] > 1:
                # Look at consecutive time steps
                for t in range(tensor_np.shape[0] - 1):
                    # Find active pitches at each timestep
                    active_t = np.where(tensor_np[t] > 0)[0]
                    active_t1 = np.where(tensor_np[t + 1] > 0)[0]

                    if len(active_t) > 0 and len(active_t1) > 0:
                        # Melodic interval between highest notes
                        interval = abs(active_t1.max() - active_t.max()) % 12
                        intervals[interval] += 1

            if intervals.sum() > 0:
                intervals = intervals / intervals.sum()

            # Melodic contour from tensor
            contour = torch.zeros(8)
            if len(tensor_np.shape) == 2:
                # Sample 9 time points for 8 contour values
                sample_points = np.linspace(0, tensor_np.shape[0] - 1, 9).astype(int)
                pitches = []
                for t in sample_points:
                    active = np.where(tensor_np[t] > 0)[0]
                    if len(active) > 0:
                        pitches.append(active.mean())
                    elif pitches:
                        pitches.append(pitches[-1])
                    else:
                        pitches.append(60)

                for i in range(min(8, len(pitches) - 1)):
                    contour[i] = np.tanh((pitches[i + 1] - pitches[i]) / 12)

            # Rhythm pattern from tensor
            rhythm = torch.zeros(16)
            if len(tensor_np.shape) == 2:
                # Sample 16 time bins
                bin_size = max(1, tensor_np.shape[0] // 16)
                for i in range(16):
                    start_idx = i * bin_size
                    end_idx = min((i + 1) * bin_size, tensor_np.shape[0])
                    if start_idx < tensor_np.shape[0]:
                        rhythm[i] = 1.0 if tensor_np[start_idx:end_idx].sum() > 0 else 0.0
        else:
            # Empty features
            pitch_classes = torch.zeros(12)
            intervals = torch.zeros(12)
            contour = torch.zeros(8)
            rhythm = torch.zeros(16)

        # Guess instrument family from track name
        instrument_family = "unknown"
        track_name = str(track_id).lower()
        if any(x in track_name for x in ['trumpet', 'trombone', 'horn', 'brass', 'tuba']):
            instrument_family = "brass"
        elif any(x in track_name for x in ['sax', 'clarinet', 'flute', 'oboe', 'reed']):
            instrument_family = "reed"
        elif any(x in track_name for x in ['piano', 'keys']):
            instrument_family = "piano"
        elif any(x in track_name for x in ['bass']):
            instrument_family = "bass"
        elif any(x in track_name for x in ['drum', 'percussion']):
            instrument_family = "drums"
        elif any(x in track_name for x in ['guitar']):
            instrument_family = "guitar"
        elif any(x in track_name for x in ['string', 'violin', 'viola', 'cello']):
            instrument_family = "strings"

        musical_obj = MusicalObject(
            piece_id=str(piece_id),
            track_id=str(track_id),
            bar_start=int(bar_start),
            bar_end=int(bar_end),
            pitch_classes=pitch_classes,
            intervals=intervals,
            contour=contour,
            rhythm=rhythm,
            instrument_family=instrument_family
        )
        musical_objects.append(musical_obj)

    return musical_objects


def main():
    parser = argparse.ArgumentParser(description='Multi-View Pattern Discovery')
    parser.add_argument('--corpus-path', default='/home/arlo/do-repo/midi_generator/midi_corpus/big_band')
    parser.add_argument('--max-files', type=int, default=25)
    parser.add_argument('--max-error', type=float, default=0.15)
    parser.add_argument('--samples-per-view', type=int, default=50000)
    parser.add_argument('--match-threshold', type=float, default=0.15)
    parser.add_argument('--scale', type=int, default=16)
    parser.add_argument('--output-dir', default='./vocab_first_v8')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("MULTI-VIEW PATTERN DISCOVERY")
    print(f"{'='*70}\n")

    print(f"Corpus path: {args.corpus_path}")
    print(f"Max files: {args.max_files}")
    print(f"Max error: {args.max_error}")
    print(f"Samples per view: {args.samples_per_view}")
    print(f"Match threshold: {args.match_threshold}")
    print(f"Output directory: {args.output_dir}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load MIDI files
    print("Loading MIDI files...")
    midi_files = []
    file_paths = []

    for root, dirs, files in os.walk(args.corpus_path):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                file_paths.append(os.path.join(root, file))
                if len(file_paths) >= args.max_files:
                    break
        if len(file_paths) >= args.max_files:
            break

    print(f"Found {len(file_paths)} MIDI files")

    # Load with meter filter
    for path in file_paths:
        try:
            midi = mido.MidiFile(path)
            if has_allowed_meter(midi):
                midi_files.append(midi)
        except Exception as e:
            print(f"  [!] Failed to load {os.path.basename(path)}: {e}")

    print(f"Loaded {len(midi_files)} MIDI files (after meter filter)\n")

    if not midi_files:
        print("No valid MIDI files loaded!")
        return

    # Convert to hierarchical representation
    print("Converting to hierarchical representation...")
    corpus_loader = HierarchicalMIDICorpus()
    corpus = corpus_loader.load_corpus_hierarchical(midi_files, verbose=True)

    # Extract objects using EmergentHierarchyDiscovery
    print("\nExtracting objects...")
    discovery = EmergentHierarchyDiscovery(
        scales=STANDARD_SCALES,
        max_error=args.max_error
    )
    objects = discovery.extract_objects(corpus, verbose=True)
    print(f"Extracted {len(objects)} objects")

    if not objects:
        print("No objects extracted!")
        return

    # Convert to MusicalObjects for multi-view
    print("\nConverting to MusicalObjects for multi-view analysis...")
    musical_objects = convert_objects_to_musical_objects(objects)
    print(f"Converted {len(musical_objects)} objects")

    # Show distribution
    pieces = {}
    families = {}
    for obj in musical_objects:
        pieces[obj.piece_id] = pieces.get(obj.piece_id, 0) + 1
        families[obj.instrument_family] = families.get(obj.instrument_family, 0) + 1

    print(f"\nPieces: {len(pieces)}")
    for piece_id, count in sorted(pieces.items())[:10]:
        print(f"  {piece_id}: {count}")
    if len(pieces) > 10:
        print(f"  ... and {len(pieces) - 10} more")

    print(f"\nInstrument families:")
    for family, count in sorted(families.items(), key=lambda x: -x[1]):
        print(f"  {family}: {count}")

    # Run multi-view discovery
    print("\n" + "="*70)
    print("Running Multi-View Discovery...")
    print("="*70)

    start_time = time.time()

    mv_discovery = MultiViewDiscovery(
        samples_per_view=args.samples_per_view,
        match_threshold=args.match_threshold
    )
    mv_discovery.add_objects(musical_objects)

    results = mv_discovery.discover()

    total_time = time.time() - start_time

    # Save results
    output = {
        'parameters': vars(args),
        'total_time_seconds': total_time,
        'total_objects': len(musical_objects),
        'pieces': len(pieces),
        'vocabulary': results['vocabulary'],
        'compound_stats': results['compound_stats'],
        'derivations_count': results['derivations_count']
    }

    output_path = os.path.join(args.output_dir, 'multi_view_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY BY VIEW")
    print(f"{'='*70}")

    for view_type in ViewType:
        print(f"\n{view_type.name}:")
        view_compounds = []
        for name, stats in results['compound_stats'].items():
            if view_type.name in stats['views']:
                view_compounds.append((name, stats['views'][view_type.name]))

        for name, count in sorted(view_compounds, key=lambda x: -x[1])[:5]:
            print(f"  {name}: {count}")

    print(f"\n{'='*70}")
    print("CROSS-VIEW PATTERNS (found in multiple views)")
    print(f"{'='*70}")

    cross_view_patterns = []
    for name, stats in results['compound_stats'].items():
        views_present = list(stats['views'].keys())
        if len(views_present) >= 2:
            view_str = ", ".join(f"{v}: {stats['views'][v]}" for v in views_present)
            cross_view_patterns.append((name, stats['total_matches'], view_str))

    for name, total, view_str in sorted(cross_view_patterns, key=lambda x: -x[1])[:20]:
        print(f"  {name}: {total} total [{view_str}]")

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
