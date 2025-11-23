#!/usr/bin/env python -u
"""
Start discovery pipeline using CPU multiprocessing - UNBUFFERED OUTPUT VERSION.

Expected performance on 60-core instance (56 workers):
- ~30-60 seconds per iteration
- ~15-25 minutes for 12 iterations total
- 20-30× faster than GPU approach

Usage:
    python -u scripts/start_discovery_cpu_unbuffered.py --iterations 12 --cores 56
"""

import sys
import os
import argparse
import numpy as np

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '1_approaches', 'transform_based'))

from discovery.cpu_discovery_pipeline import CPUDiscoveryPipeline
from core.tensor_representation import TensorMIDICorpus
import mido


def load_midi_corpus_as_numpy(corpus_path: str, max_files: int = None):
    """Load MIDI corpus and convert to numpy arrays."""
    print(f"\n{'='*70}", flush=True)
    print("LOADING CORPUS", flush=True)
    print(f"{'='*70}", flush=True)

    # Load MIDI files
    midi_files = []
    file_paths = []

    for root, dirs, files in os.walk(corpus_path):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                file_paths.append(os.path.join(root, file))
                if max_files and len(file_paths) >= max_files:
                    break
        if max_files and len(file_paths) >= max_files:
            break

    print(f"Loading {len(file_paths)} MIDI files...", flush=True)

    for i, path in enumerate(file_paths):
        try:
            midi = mido.MidiFile(path)
            midi_files.append(midi)

            if (i + 1) % 100 == 0:
                print(f"  Loaded {i+1}/{len(file_paths)} files...", flush=True)

        except Exception as e:
            print(f"  [!] Warning: Failed to load {os.path.basename(path)}: {str(e)}", flush=True)

    print(f"\n✓ Loaded {len(midi_files)} MIDI files successfully\n", flush=True)

    # Convert to tensors then numpy
    print("Converting to numpy arrays...", flush=True)
    converter = TensorMIDICorpus()
    corpus_list = []

    for i, midi in enumerate(midi_files):
        try:
            tensor = converter.midi_to_tensor(midi)
            corpus_list.append(tensor.cpu().numpy())

            if (i + 1) % 100 == 0:
                print(f"  Converted {i+1}/{len(midi_files)} files...", flush=True)

        except Exception as e:
            print(f"  [!] Warning: Failed to convert file: {str(e)}", flush=True)

    # Stack into single array
    corpus = np.stack(corpus_list, axis=0)
    print(f"✓ Corpus shape: {corpus.shape}", flush=True)
    print(f"  Memory: {corpus.nbytes / 1e9:.2f} GB\n", flush=True)

    return corpus


def load_initial_transforms():
    """Load initial irreducible transform primitives."""
    print(f"{'='*70}", flush=True)
    print("LOADING BASE PRIMITIVES", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Define 14 irreducible primitives
    primitives = [
        {'name': 'transpose_semitone', 'amount': 7},
        {'name': 'transpose_semitone', 'amount': -7},
        {'name': 'inversion', 'amount': 60},
        {'name': 'retrograde', 'amount': 0},
        {'name': 'time_scale', 'amount': 2.0},
        {'name': 'time_scale', 'amount': 0.5},
        {'name': 'time_shift', 'amount': 16},
        {'name': 'time_shift', 'amount': -16},
        {'name': 'velocity_scale', 'amount': 1.5},
        {'name': 'velocity_scale', 'amount': 0.7},
        {'name': 'quantize_16th', 'amount': 0},
        {'name': 'quantize_8th', 'amount': 0},
        {'name': 'instrument_filter', 'amount': 0.0},  # Piano
        {'name': 'instrument_filter', 'amount': 0.5},  # Strings
    ]

    names = ', '.join([t['name'] for t in primitives[:5]])
    print(f"✓ Loaded {len(primitives)} irreducible primitives", flush=True)
    print(f"  Transform names: {names}...\n", flush=True)

    return primitives


def main():
    parser = argparse.ArgumentParser(description='CPU-accelerated discovery pipeline')
    parser.add_argument('--corpus-path', type=str,
                        default='/home/arlo/do-repo/midi_generator/midi_corpus/big_band',
                        help='Path to MIDI corpus directory')
    parser.add_argument('--iterations', type=int, default=12,
                        help='Number of discovery iterations')
    parser.add_argument('--max-transforms-per-iteration', type=int, default=50,
                        help='Maximum new transforms to add per iteration')
    parser.add_argument('--target-quality', type=float, default=0.95,
                        help='Target reconstruction quality')
    parser.add_argument('--cores', type=int, default=56,
                        help='Number of CPU cores to use (default: 56)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum MIDI files to load (for testing)')
    parser.add_argument('--adaptive-threshold', action='store_true',
                        help='Use adaptive threshold based on improvement distribution (recommended)')
    parser.add_argument('--threshold', type=float, default=0.0001,
                        help='Fixed threshold value (ignored if --adaptive-threshold is set)')

    args = parser.parse_args()

    print(f"\n{'='*70}", flush=True)
    print("NEURAL PROGRAM SYNTHESIS - CPU DISCOVERY PIPELINE", flush=True)
    print(f"{'='*70}\n", flush=True)

    print("Configuration:", flush=True)
    print(f"  Corpus path: {args.corpus_path}", flush=True)
    print(f"  Iterations: {args.iterations}", flush=True)
    print(f"  Target quality: {args.target_quality:.1%}", flush=True)
    print(f"  Max transforms/iteration: {args.max_transforms_per_iteration}", flush=True)
    print(f"  CPU cores: {args.cores}", flush=True)
    print(f"  Threshold mode: {'Adaptive (top 5%)' if args.adaptive_threshold else f'Fixed ({args.threshold})'}\n", flush=True)

    # Load corpus
    corpus = load_midi_corpus_as_numpy(args.corpus_path, max_files=args.max_files)

    # Load initial transforms
    initial_transforms = load_initial_transforms()

    # Create pipeline
    pipeline = CPUDiscoveryPipeline(
        n_cores=args.cores,
        adaptive_threshold=args.adaptive_threshold,
        initial_threshold=args.threshold
    )

    print(f"{'='*70}", flush=True)
    print(f"RUNNING {args.iterations} DISCOVERY ITERATIONS", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Run discovery
    results = pipeline.run_full_discovery(
        corpus=corpus,
        initial_transforms=initial_transforms,
        target_quality=args.target_quality,
        max_iterations=args.iterations,
        max_transforms_per_iteration=args.max_transforms_per_iteration
    )

    # Save results
    output_path = 'discovery_results_cpu.npy'
    np.save(output_path, results, allow_pickle=True)
    print(f"\n✓ Results saved to {output_path}", flush=True)


if __name__ == '__main__':
    main()
