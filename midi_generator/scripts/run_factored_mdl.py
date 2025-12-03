#!/usr/bin/env python
"""
Test script for factored MDL discovery.

This implements the Lewinian multi-space approach where objects are
factored into independent components (rhythm, pitch, contour, etc.)
and transforms operate on components independently.

Usage:
    python scripts/run_factored_mdl.py --max-files 25

Author: Factored MDL Test
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
from discovery.factored_mdl import run_factored_mdl

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


def main():
    parser = argparse.ArgumentParser(description='Factored MDL Discovery')
    parser.add_argument('--corpus-path', default='/home/arlo/do-repo/midi_generator/midi_corpus/big_band')
    parser.add_argument('--max-files', type=int, default=25)
    parser.add_argument('--min-group-size', type=int, default=3, help='Min objects per pattern group')
    parser.add_argument('--output-dir', default='./factored_mdl_results')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("FACTORED MDL DISCOVERY")
    print(f"{'='*70}\n")

    print(f"Corpus path: {args.corpus_path}")
    print(f"Max files: {args.max_files}")
    print(f"Min group size: {args.min_group_size}")
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

    # Convert to hierarchical representation
    print("Converting to hierarchical representation...")
    corpus_loader = HierarchicalMIDICorpus()
    corpus = corpus_loader.load_corpus_hierarchical(midi_files, verbose=True)

    # Extract objects using EmergentHierarchyDiscovery
    print("\nExtracting objects...")
    discovery = EmergentHierarchyDiscovery(
        scales=STANDARD_SCALES,
        max_error=0.1  # Not used directly in factored version
    )
    objects = discovery.extract_objects(corpus, verbose=True)
    print(f"Extracted {len(objects)} objects")

    # Run factored MDL
    start_time = time.time()

    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.npz')
    result = run_factored_mdl(
        objects=objects,
        min_group_size=args.min_group_size,
        verbose=True,
        save_checkpoint_path=checkpoint_path
    )

    total_time = time.time() - start_time

    # Save results
    output = {
        'parameters': vars(args),
        'total_time_seconds': total_time,
        'stats': result.stats,
        'component_stats': result.component_stats,
        'vocabulary': [t.name for t in result.vocabulary],
        'rhythm_groups': result.rhythm_groups,
        'contour_groups': result.contour_groups,
        # New features
        'compression': result.stats.get('compression', {}),
        'within_component_transforms': result.stats.get('within_component_transforms', {}),
        'factored_vs_atomic': result.stats.get('factored_vs_atomic', {}),
        'reconstruction_quality': result.stats.get('reconstruction_quality', {}),
        # Reconstruction data for MIDI DNA editor
        'reconstruction_data': result.reconstruction_data,
    }

    output_path = os.path.join(args.output_dir, 'factored_mdl_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Results saved to: {output_path}")

    # Print comparison metrics
    print(f"\n{'='*70}")
    print("KEY METRICS")
    print(f"{'='*70}")
    print(f"  Total objects: {result.stats['total_objects']}")
    print(f"  Valid objects (with notes): {result.stats['valid_objects']}")
    print(f"  Unique rhythm patterns: {result.rhythm_groups}")
    print(f"  Unique pitch contours: {result.contour_groups}")
    print(f"  Objects assigned (strict): {result.stats['total_assigned']} ({result.stats['derivation_rate']*100:.1f}%)")
    print(f"  Vocabulary size: {result.stats['vocabulary_size']}")
    print(f"\n  Transform-aware patterns:")
    print(f"    Total patterns: {result.stats.get('transform_aware_patterns', 0)}")
    print(f"    Cross-component: {result.stats.get('cross_component_patterns', 0)}")
    print(f"    Same-source: {result.stats.get('same_source_patterns', 0)}")
    print(f"    Legacy cross-patterns: {result.stats.get('cross_patterns', 0)}")

    # New metrics from the 3 features
    compression = result.stats.get('compression', {})
    if compression:
        print(f"\n  COMPRESSION (Explicit Bit Counting):")
        print(f"    Total compression ratio: {compression.get('total_ratio', 1.0):.2f}x")
        print(f"    Literal bits: {compression.get('total_literal_bits', 0):.0f}")
        print(f"    Compressed bits: {compression.get('total_compressed_bits', 0):.0f}")
        per_comp = compression.get('per_component', {})
        for comp_name, ratio in per_comp.items():
            if comp_name != 'total':
                print(f"    {comp_name}: {ratio:.2f}x")

    within_transforms = result.stats.get('within_component_transforms', {})
    if within_transforms:
        print(f"\n  WITHIN-COMPONENT TRANSFORMS:")
        for comp_name, stats in within_transforms.items():
            print(f"    {comp_name}: {stats.get('canonical', 0)} canonical, "
                  f"{stats.get('derived', 0)} derived, "
                  f"{stats.get('compression', 1.0):.2f}x compression")

    fva = result.stats.get('factored_vs_atomic', {})
    if fva:
        print(f"\n  FACTORED vs ATOMIC:")
        print(f"    Factored: {fva.get('factored_bits', 0):.0f} bits ({fva.get('factored_ratio', 1.0):.2f}x)")
        print(f"    Atomic: {fva.get('atomic_bits', 0):.0f} bits ({fva.get('atomic_ratio', 1.0):.2f}x)")
        print(f"    Winner: {fva.get('winner', 'unknown').upper()}")
        print(f"    Advantage: {fva.get('advantage_ratio', 1.0):.2f}x")

    recon = result.stats.get('reconstruction_quality', {})
    if recon:
        print(f"\n  RECONSTRUCTION QUALITY:")
        print(f"    Note accuracy: {recon.get('note_accuracy', 0)*100:.1f}%")
        print(f"    Exact matches: {recon.get('exact_matches', 0)}/{recon.get('objects_verified', 0)} ({recon.get('exact_match_rate', 0)*100:.1f}%)")
        print(f"    Total notes: {recon.get('total_notes', 0)}, Reconstructed: {recon.get('reconstructed_notes', 0)}")
        print(f"    Average MSE: {recon.get('average_mse', 0):.6f}")


if __name__ == '__main__':
    main()
