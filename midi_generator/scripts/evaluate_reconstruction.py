#!/usr/bin/env python
"""
Evaluate reconstruction quality of discovered derivation graph.

This measures end-to-end reconstruction error by:
1. Following each object's derivation chain back to source
2. Reconstructing by applying all transforms in sequence
3. Measuring MSE between reconstructed and original

This reveals whether long chains compound error or preserve quality.

Usage:
    python scripts/evaluate_reconstruction.py --test-run-path emergent_optimized_cpu_test.log

Author: Agent - Reconstruction Evaluation
"""

import sys
import os
import argparse
import pickle
import numpy as np
from typing import Dict, List, Tuple
import mido

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '1_approaches', 'transform_based'))

from core.hierarchical_corpus import HierarchicalMIDICorpus
from discovery.emergent_hierarchy import EmergentHierarchyDiscovery, MusicalObject, Derivation
from core.numpy_transforms import NumpyTransformLibrary


def load_initial_transforms():
    """Load initial irreducible transform primitives."""
    primitives = [
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
        {'name': 'retrograde', 'amount': 0},
        {'name': 'time_scale', 'amount': 2.0},
        {'name': 'time_scale', 'amount': 0.5},
        {'name': 'time_shift', 'amount': 16},
        {'name': 'time_shift', 'amount': -16},
        {'name': 'time_shift', 'amount': 32},
        {'name': 'time_shift', 'amount': -32},
    ]
    return primitives


def reconstruct_from_source(
    target: MusicalObject,
    graph: Dict[MusicalObject, Derivation],
    lib: NumpyTransformLibrary,
    max_depth: int = 50
) -> Tuple[np.ndarray, List[str], int]:
    """
    Reconstruct an object by following its derivation chain to source.

    Args:
        target: Object to reconstruct
        graph: Derivation graph
        lib: Transform library
        max_depth: Maximum chain length to follow

    Returns:
        (reconstructed_tensor, path_taken, depth)
    """
    # Trace path back to source
    path = []
    current = target
    depth = 0

    while current in graph and depth < max_depth:
        derivation = graph[current]
        path.append(derivation)
        current = derivation.source
        depth += 1

    # Reverse path (now goes from source → target)
    path.reverse()

    # Start with source
    reconstructed = current.tensor.copy()
    transform_names = []

    # Apply each transform in sequence
    for derivation in path:
        reconstructed = np.expand_dims(reconstructed, 0)
        reconstructed = lib.apply_transform(
            reconstructed,
            derivation.transform_name,
            derivation.transform_amount
        )[0]
        transform_names.append(f"{derivation.transform_name}({derivation.transform_amount})")

    return reconstructed, transform_names, depth


def evaluate_reconstruction_quality(
    objects: List[MusicalObject],
    graph: Dict[MusicalObject, Derivation],
    sources: set,
    verbose: bool = True
) -> Dict:
    """
    Evaluate end-to-end reconstruction quality.

    Args:
        objects: All musical objects
        graph: Derivation graph
        sources: Source objects (roots)
        verbose: Print detailed statistics

    Returns:
        Dictionary of metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print("RECONSTRUCTION QUALITY EVALUATION")
        print(f"{'='*70}\n")

    lib = NumpyTransformLibrary()

    reconstruction_errors = []
    direct_errors = []  # Error from immediate parent
    depths = []
    failed_reconstructions = 0

    if verbose:
        print(f"Evaluating {len(objects)} objects...")
        print(f"  {len(sources)} sources")
        print(f"  {len(graph)} derivations\n")

    for i, obj in enumerate(objects):
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(objects)} objects...")

        # Skip sources (perfect reconstruction)
        if obj in sources:
            reconstruction_errors.append(0.0)
            direct_errors.append(0.0)
            depths.append(0)
            continue

        # Reconstruct from source
        try:
            reconstructed, path, depth = reconstruct_from_source(obj, graph, lib)

            # Compute end-to-end reconstruction error
            mse = np.mean((obj.tensor - reconstructed) ** 2)
            reconstruction_errors.append(mse)
            depths.append(depth)

            # Also get direct parent error (from graph)
            if obj in graph:
                direct_errors.append(graph[obj].error)
            else:
                direct_errors.append(0.0)

        except Exception as e:
            if verbose:
                print(f"  [!] Failed to reconstruct {obj}: {str(e)}")
            failed_reconstructions += 1
            reconstruction_errors.append(float('inf'))
            direct_errors.append(float('inf'))
            depths.append(0)

    # Compute statistics (excluding failed)
    valid_reconstruction_errors = [e for e in reconstruction_errors if e != float('inf')]
    valid_direct_errors = [e for e in direct_errors if e != float('inf')]
    valid_depths = [d for d, e in zip(depths, reconstruction_errors) if e != float('inf')]

    stats = {
        'total_objects': len(objects),
        'num_sources': len(sources),
        'num_derived': len(graph),
        'failed_reconstructions': failed_reconstructions,

        # Reconstruction errors
        'reconstruction_mse_mean': np.mean(valid_reconstruction_errors),
        'reconstruction_mse_median': np.median(valid_reconstruction_errors),
        'reconstruction_mse_max': np.max(valid_reconstruction_errors),
        'reconstruction_mse_std': np.std(valid_reconstruction_errors),

        # Direct errors (from graph)
        'direct_mse_mean': np.mean(valid_direct_errors),
        'direct_mse_median': np.median(valid_direct_errors),
        'direct_mse_max': np.max(valid_direct_errors),

        # Depth statistics
        'depth_mean': np.mean(valid_depths),
        'depth_median': np.median(valid_depths),
        'depth_max': np.max(valid_depths),

        # Error amplification
        'error_amplification': np.mean(valid_reconstruction_errors) / (np.mean(valid_direct_errors) + 1e-10),
    }

    if verbose:
        print(f"\n{'='*70}")
        print("RECONSTRUCTION STATISTICS")
        print(f"{'='*70}\n")

        print("Coverage:")
        print(f"  Total objects: {stats['total_objects']}")
        print(f"  Sources: {stats['num_sources']}")
        print(f"  Derived: {stats['num_derived']}")
        print(f"  Failed reconstructions: {stats['failed_reconstructions']}\n")

        print("Depth Statistics:")
        print(f"  Mean depth: {stats['depth_mean']:.2f}")
        print(f"  Median depth: {stats['depth_median']:.1f}")
        print(f"  Max depth: {stats['depth_max']}\n")

        print("Direct Derivation Error (parent → child):")
        print(f"  Mean MSE: {stats['direct_mse_mean']:.6f}")
        print(f"  Median MSE: {stats['direct_mse_median']:.6f}")
        print(f"  Max MSE: {stats['direct_mse_max']:.6f}\n")

        print("End-to-End Reconstruction Error (source → target):")
        print(f"  Mean MSE: {stats['reconstruction_mse_mean']:.6f}")
        print(f"  Median MSE: {stats['reconstruction_mse_median']:.6f}")
        print(f"  Max MSE: {stats['reconstruction_mse_max']:.6f}")
        print(f"  Std MSE: {stats['reconstruction_mse_std']:.6f}\n")

        print("Error Amplification:")
        print(f"  Amplification factor: {stats['error_amplification']:.2f}x")
        print(f"  (End-to-end error / Direct error)")

        # Quality interpretation
        print(f"\n{'='*70}")
        print("QUALITY INTERPRETATION")
        print(f"{'='*70}\n")

        avg_recon_error = stats['reconstruction_mse_mean']

        if avg_recon_error < 0.001:
            quality = "EXCELLENT"
            interpretation = "Very high fidelity - chains preserve quality well"
        elif avg_recon_error < 0.01:
            quality = "GOOD"
            interpretation = "Good fidelity - acceptable reconstruction quality"
        elif avg_recon_error < 0.05:
            quality = "FAIR"
            interpretation = "Moderate fidelity - some degradation in long chains"
        elif avg_recon_error < 0.1:
            quality = "POOR"
            interpretation = "Low fidelity - significant error compounding"
        else:
            quality = "VERY POOR"
            interpretation = "Very low fidelity - severe error compounding"

        print(f"Overall Quality: {quality}")
        print(f"  {interpretation}\n")

        # Error distribution
        print("Error Distribution:")
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(valid_reconstruction_errors, p)
            print(f"  {p}th percentile: {val:.6f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate reconstruction quality')
    parser.add_argument('--corpus-path', default='/home/arlo/do-repo/midi_generator/midi_corpus/big_band',
                        help='Path to MIDI corpus')
    parser.add_argument('--max-files', type=int, default=30,
                        help='Maximum files to process')
    parser.add_argument('--scales', nargs='+', type=int, default=[64, 128, 256],
                        help='Segment scales to extract (timesteps)')
    parser.add_argument('--max-error', type=float, default=0.03,
                        help='Maximum error for valid derivation')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("RECONSTRUCTION QUALITY EVALUATION")
    print(f"{'='*70}\n")

    # Load corpus
    print("Loading MIDI corpus...")
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

    for i, path in enumerate(file_paths):
        try:
            midi = mido.MidiFile(path)
            midi_files.append(midi)
        except Exception as e:
            print(f"  [!] Failed to load {os.path.basename(path)}: {str(e)}")

    print(f"✓ Loaded {len(midi_files)} MIDI files\n")

    # Convert to hierarchical representation
    print("Converting to hierarchical representation...")
    corpus_loader = HierarchicalMIDICorpus()
    corpus = corpus_loader.load_corpus_hierarchical(midi_files, verbose=False)

    # Run discovery
    print("Running emergent hierarchy discovery...")
    primitives = load_initial_transforms()

    discovery = EmergentHierarchyDiscovery(
        scales=args.scales,
        max_error=args.max_error,
        min_path_frequency=2
    )

    results = discovery.run_full_discovery(
        corpus, primitives, verbose=False,
        use_gpu=False, same_piece_only=True
    )

    print(f"✓ Discovery complete")
    print(f"  Objects: {len(results['objects'])}")
    print(f"  Derivations: {len(results['graph'])}")
    print(f"  Sources: {len(results['sources'])}")

    # Evaluate reconstruction quality
    stats = evaluate_reconstruction_quality(
        results['objects'],
        results['graph'],
        results['sources'],
        verbose=True
    )

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
