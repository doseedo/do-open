#!/usr/bin/env python
"""
Test script for vocabulary-first MDL discovery.

This implements the Lewinian approach: find transform vocabulary that
minimizes total corpus description length.

Usage:
    python scripts/run_vocabulary_first.py --max-files 25

Author: Vocabulary-First MDL Test
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
from discovery.vocabulary_first_mdl import run_vocabulary_first_mdl, run_vocabulary_first_hybrid

import numpy as np


def build_faiss_indices(objects, verbose=True, use_gpu=True):
    """
    Build FAISS indices for all scales present in objects.

    Uses IVF index (instead of flat) to avoid GPU cublas errors on large matrices.
    IVF is approximate but much faster and avoids the matrix size limitations.

    Returns:
        Dict[scale -> (index_gpu, index_cpu, index_to_object, F)]
    """
    try:
        import faiss
    except ImportError:
        print("FAISS not available, skipping index building")
        return {}

    # Check GPU availability
    gpu_available = False
    res = None
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            gpu_available = True
        except Exception as e:
            if verbose:
                print(f"  GPU FAISS not available: {e}")

    # Group objects by scale
    by_scale = {}
    for obj in objects:
        scale = obj.tensor.shape[0]
        if scale not in by_scale:
            by_scale[scale] = []
        by_scale[scale].append(obj)

    scale_indices = {}
    for scale, scale_objects in by_scale.items():
        if len(scale_objects) < 2:
            continue

        # Stack tensors
        vectors = np.stack([obj.tensor.flatten().astype(np.float32) for obj in scale_objects])
        n_vectors, F = vectors.shape

        # Build object lookup
        index_to_object = {i: obj for i, obj in enumerate(scale_objects)}

        # Choose index type based on dataset size
        # For small datasets, use flat index; for large, use IVF to avoid cublas errors
        if n_vectors < 1000:
            # Small dataset: flat index is fine
            index = faiss.IndexFlatL2(F)
            index.add(vectors)
            index_type = "Flat"
        else:
            # Large dataset: use IVF to avoid GPU cublas matrix size issues
            # nlist = number of clusters (sqrt rule is common)
            nlist = min(int(np.sqrt(n_vectors)), 256)  # Cap at 256 clusters
            nlist = max(nlist, 4)  # At least 4 clusters

            quantizer = faiss.IndexFlatL2(F)
            index = faiss.IndexIVFFlat(quantizer, F, nlist)

            # Train on the vectors
            index.train(vectors)
            index.add(vectors)

            # Set nprobe for search (higher = more accurate but slower)
            index.nprobe = min(nlist // 2, 32)  # Search ~50% of clusters, max 32
            index_type = f"IVF{nlist}"

        # Try to move to GPU for speed
        index_gpu = None
        if gpu_available and res is not None:
            try:
                index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
                if verbose:
                    print(f"  Scale {scale}: {n_vectors} objects, {F} features ({index_type}, GPU)")
            except Exception as e:
                if verbose:
                    print(f"  Scale {scale}: {n_vectors} objects, {F} features ({index_type}, CPU - GPU failed: {e})")
        else:
            if verbose:
                print(f"  Scale {scale}: {n_vectors} objects, {F} features ({index_type}, CPU)")

        scale_indices[scale] = (index_gpu, index, index_to_object, F)

    return scale_indices

# Fixed scales and meters (from run_emergent_discovery.py)
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


def load_initial_transforms():
    """Load initial irreducible transform primitives."""
    primitives = [
        # PITCH
        {'name': 'transpose_semitone', 'amount': 7},
        {'name': 'transpose_semitone', 'amount': -7},
        {'name': 'transpose_semitone', 'amount': 12},
        {'name': 'transpose_semitone', 'amount': -12},
        {'name': 'transpose_semitone', 'amount': 5},
        {'name': 'transpose_semitone', 'amount': -5},
        {'name': 'transpose_semitone', 'amount': 3},
        {'name': 'transpose_semitone', 'amount': -3},
        {'name': 'transpose_semitone', 'amount': 2},
        {'name': 'transpose_semitone', 'amount': -2},
        {'name': 'inversion', 'amount': 60},

        # TIME
        {'name': 'time_shift', 'amount': 16},
        {'name': 'time_shift', 'amount': -16},
        {'name': 'time_shift', 'amount': 32},
        {'name': 'time_shift', 'amount': -32},
        {'name': 'time_shift', 'amount': 64},
        {'name': 'time_shift', 'amount': -64},
        {'name': 'retrograde', 'amount': 0},

        # DYNAMICS
        {'name': 'velocity_scale', 'amount': 0.5},
        {'name': 'velocity_scale', 'amount': 0.7},
        {'name': 'velocity_scale', 'amount': 0.8},
        {'name': 'velocity_scale', 'amount': 1.2},
        {'name': 'velocity_scale', 'amount': 1.5},
    ]
    return primitives


def main():
    parser = argparse.ArgumentParser(description='Vocabulary-First MDL Discovery')
    parser.add_argument('--corpus-path', default='/home/arlo/do-repo/midi_generator/midi_corpus/big_band')
    parser.add_argument('--max-files', type=int, default=25)
    parser.add_argument('--max-error', type=float, default=0.03)
    parser.add_argument('--max-depth', type=int, default=2, help='Max compound depth')
    parser.add_argument('--min-frequency', type=int, default=10)
    parser.add_argument('--scale', type=int, default=16, help='Scale for Phase 1 mining')
    parser.add_argument('--output-dir', default='./vocab_first_results')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid FAISS+compound mode')
    parser.add_argument('--k-neighbors', type=int, default=5, help='FAISS k neighbors (hybrid mode)')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("VOCABULARY-FIRST MDL DISCOVERY")
    print(f"{'='*70}\n")

    print(f"Corpus path: {args.corpus_path}")
    print(f"Max files: {args.max_files}")
    print(f"Max error: {args.max_error}")
    print(f"Max compound depth: {args.max_depth}")
    print(f"Min frequency: {args.min_frequency}")
    print(f"Scale for mining: {args.scale}")
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
        max_error=args.max_error
    )
    objects = discovery.extract_objects(corpus, verbose=True)
    print(f"Extracted {len(objects)} objects")

    # Load primitives
    primitives = load_initial_transforms()
    print(f"Loaded {len(primitives)} primitive transforms\n")

    # Run vocabulary-first MDL
    start_time = time.time()

    if args.hybrid:
        # Hybrid mode: use FAISS to find pairs, then identify compounds
        # First need to build FAISS indices
        print("Building FAISS indices for hybrid mode...")
        scale_indices = build_faiss_indices(objects, verbose=True)

        result = run_vocabulary_first_hybrid(
            objects=objects,
            primitives=primitives,
            scale_indices=scale_indices,
            max_error=args.max_error,
            max_depth=args.max_depth,
            min_frequency=args.min_frequency,
            k_neighbors=args.k_neighbors,
            verbose=True
        )
    else:
        # Standard mode: same-timestep cross-track pairs
        result = run_vocabulary_first_mdl(
            objects=objects,
            primitives=primitives,
            scale_indices=None,  # No FAISS for now
            max_error=args.max_error,
            max_depth=args.max_depth,
            min_frequency=args.min_frequency,
            scale=args.scale,
            verbose=True
        )

    total_time = time.time() - start_time

    # Save results
    output = {
        'parameters': vars(args),
        'total_time_seconds': total_time,
        'stats': result.stats,
        'vocabulary': [c.name for c in result.vocabulary if c.depth > 1],
        'top_compounds': result.stats.get('compound_usage', {})
    }

    output_path = os.path.join(args.output_dir, 'vocab_first_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
