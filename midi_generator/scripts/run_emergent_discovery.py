#!/usr/bin/env python
"""
Full iterative emergent hierarchy discovery pipeline.

This runs multiple iterations of discovery:
    Iteration 1: Discover derivations with primitives
    Iteration 2: Add discovered compositions → discover higher-order patterns
    Iteration 3+: Continue until convergence

The transform library grows organically as the algorithm discovers
compositional structure in the corpus.

Usage:
    python scripts/run_emergent_discovery.py --corpus-path /path/to/corpus --max-iterations 5 --gpu

Author: Agent - Emergent Hierarchy Discovery
"""

import sys
import os
import argparse
import mido
import json
import time
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '1_approaches', 'transform_based'))

from core.hierarchical_corpus import HierarchicalMIDICorpus
from discovery.emergent_hierarchy import EmergentHierarchyDiscovery
from core.numpy_transforms import NumpyTransformLibrary

# Fixed scales for standard meters (4/4, 2/4, 3/4, 6/8)
# 16=full bar (4/4), 32=2 bars, etc.
# Note: scales 8 and 12 removed - cause GPU cuBLAS errors with large corpora
STANDARD_SCALES = [16, 32, 64, 128, 256]

# Allowed time signatures (numerator, denominator)
ALLOWED_METERS = {(4, 4), (2, 4), (3, 4), (6, 8)}


def get_midi_time_signatures(midi_file: mido.MidiFile) -> set:
    """Extract all time signatures from a MIDI file."""
    time_sigs = set()
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                time_sigs.add((msg.numerator, msg.denominator))
    # If no time signature found, assume 4/4
    if not time_sigs:
        time_sigs.add((4, 4))
    return time_sigs


def has_allowed_meter(midi_file: mido.MidiFile) -> bool:
    """Check if MIDI file only contains allowed time signatures."""
    time_sigs = get_midi_time_signatures(midi_file)
    return time_sigs.issubset(ALLOWED_METERS)


def load_initial_transforms():
    """
    Load initial irreducible transform primitives.

    Includes:
    - Pitch transforms (transpose, inversion)
    - Time transforms (shift, scale, retrograde)
    - Dynamics (velocity_scale)
    - Rhythm (quantize)
    - Structure (repeat, fragment)
    - Multitrack (track_derive) - special handling
    """
    primitives = [
        # === PITCH TRANSFORMS ===
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

        # === TIME TRANSFORMS ===
        {'name': 'time_shift', 'amount': 16},
        {'name': 'time_shift', 'amount': -16},
        {'name': 'time_shift', 'amount': 32},
        {'name': 'time_shift', 'amount': -32},
        {'name': 'time_shift', 'amount': 64},
        {'name': 'time_shift', 'amount': -64},
        {'name': 'time_scale', 'amount': 2.0},
        {'name': 'time_scale', 'amount': 0.5},
        {'name': 'time_scale', 'amount': 4.0},
        {'name': 'time_scale', 'amount': 0.25},
        {'name': 'retrograde', 'amount': 0},

        # === DYNAMICS ===
        {'name': 'velocity_scale', 'amount': 0.5},
        {'name': 'velocity_scale', 'amount': 0.7},
        {'name': 'velocity_scale', 'amount': 0.8},
        {'name': 'velocity_scale', 'amount': 1.2},
        {'name': 'velocity_scale', 'amount': 1.5},

        # === RHYTHM ===
        {'name': 'quantize_16th', 'amount': 16},
        {'name': 'quantize_8th', 'amount': 8},

        # === STRUCTURE ===
        # ConcatSeq: Universal concatenation primitive (replaces old 'concatenate')
        # Handles both homogeneous (repetition) and heterogeneous (composition)
        #
        # For homogeneous patterns [x,x,x]: amount = n (repeat factor)
        # For heterogeneous patterns [x,y,z]: handled separately via valid_splits
        #
        # Cross-scale: target=256 with n=2 searches source at scale=128
        {'name': 'concat_seq', 'amount': 2},  # Target = ConcatSeq([x, x])
        {'name': 'concat_seq', 'amount': 3},  # Target = ConcatSeq([x, x, x])
        {'name': 'concat_seq', 'amount': 4},  # Target = ConcatSeq([x, x, x, x])
        {'name': 'concat_seq', 'amount': 8},  # Target = ConcatSeq([x, x, x, x, x, x, x, x])

        # === MULTITRACK ===
        # Note: TrackDerive is handled specially in the discovery loop
        # It requires access to track names and cross-track comparison
        # See build_derivation_graph_with_multitrack() for implementation
    ]
    return primitives


def parse_composition_path(path_str: str) -> List[Dict]:
    """
    Parse a composition path string into list of transforms.

    Example: "time_shift(-16) ∘ transpose_semitone(12)"
    Returns: [{'name': 'time_shift', 'amount': -16}, {'name': 'transpose_semitone', 'amount': 12}]
    """
    transforms = []
    parts = [p.strip() for p in path_str.split('∘')]

    for part in parts:
        # Parse "transform_name(amount)"
        if '(' in part and ')' in part:
            name = part[:part.index('(')]
            amount_str = part[part.index('(')+1:part.index(')')]
            try:
                amount = float(amount_str)
                transforms.append({'name': name, 'amount': amount})
            except ValueError:
                continue

    return transforms


def compositions_to_transforms(
    compositions: List[tuple],
    min_frequency: int = 10,
    max_compositions: int = 50
) -> List[Dict]:
    """
    Convert discovered compositions to transform representations.

    Args:
        compositions: List of (path_string, frequency) tuples
        min_frequency: Minimum frequency to be added to library
        max_compositions: Maximum compositions to add per iteration

    Returns:
        List of composition transform dictionaries
    """
    composition_transforms = []

    for path_str, freq in compositions:
        if freq < min_frequency:
            continue

        if len(composition_transforms) >= max_compositions:
            break

        # Parse the composition path
        transform_seq = parse_composition_path(path_str)

        if len(transform_seq) >= 2:  # Only multi-step compositions
            composition_transforms.append({
                'name': f'composition_{len(composition_transforms)}',
                'type': 'composition',
                'transforms': transform_seq,
                'frequency': freq,
                'path': path_str
            })

    return composition_transforms


def main():
    parser = argparse.ArgumentParser(description='Full iterative emergent hierarchy discovery')
    parser.add_argument('--corpus-path', default='/home/arlo/do-repo/midi_generator/midi_corpus/big_band',
                        help='Path to MIDI corpus')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum files to process (None = all)')
    parser.add_argument('--scales', nargs='+', type=int, default=None,
                        help='Segment scales to extract (timesteps). Default: [8,12,16,32,64,128,256]')
    parser.add_argument('--max-error', type=float, default=0.03,
                        help='Maximum error for valid derivation')
    parser.add_argument('--max-iterations', type=int, default=5,
                        help='Maximum discovery iterations')
    parser.add_argument('--min-composition-frequency', type=int, default=10,
                        help='Minimum frequency for composition to be added')
    parser.add_argument('--max-compositions-per-iteration', type=int, default=50,
                        help='Maximum compositions to add per iteration')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration if available')
    parser.add_argument('--cross-piece', action='store_true',
                        help='Allow cross-piece derivation (slower, may take days)')
    parser.add_argument('--faiss-cross-piece', action='store_true',
                        help='Use FAISS-accelerated cross-piece MDL (recommended, ~30 min)')
    parser.add_argument('--faiss-unified', action='store_true',
                        help='Use unified FAISS discovery (skips iteration 1, no OOM)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel CPU workers (default: auto-detect all cores)')
    parser.add_argument('--output-dir', default='./discovery_results',
                        help='Directory to save results')
    args = parser.parse_args()

    # Handle scales: use standard fixed scales for common meters
    if args.scales is None:
        args.scales = STANDARD_SCALES.copy()
        print(f"\n[STANDARD SCALES] Using fixed scales for 4/4, 2/4, 3/4, 6/8 meters")
        print(f"  Scales: {args.scales}")

    print(f"\n{'='*70}")
    print("ITERATIVE EMERGENT HIERARCHY DISCOVERY")
    print(f"{'='*70}\n")

    print(f"Corpus path: {args.corpus_path}")
    print(f"Max files: {args.max_files if args.max_files else 'all'}")
    print(f"Scales: {args.scales[:5]}... ({len(args.scales)} scales)" if len(args.scales) > 5 else f"Scales: {args.scales}")
    print(f"Max error: {args.max_error}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Min composition frequency: {args.min_composition_frequency}")
    print(f"GPU acceleration: {args.gpu}")
    print(f"Cross-piece derivation: {args.cross_piece}")
    print(f"FAISS unified mode: {args.faiss_unified}")
    print(f"Output directory: {args.output_dir}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load MIDI files
    print("Loading MIDI files...")
    midi_files = []
    file_paths = []

    for root, dirs, files in os.walk(args.corpus_path):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                file_paths.append(os.path.join(root, file))
                if args.max_files and len(file_paths) >= args.max_files:
                    break
        if args.max_files and len(file_paths) >= args.max_files:
            break

    print(f"Found {len(file_paths)} MIDI files")

    # Parallel MIDI loading with meter filter
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    def load_midi_file(path):
        """Load single MIDI file, return (midi, path, skip_reason) or (None, path, error)"""
        try:
            midi = mido.MidiFile(path)
            # Check meter filter
            if not has_allowed_meter(midi):
                time_sigs = get_midi_time_signatures(midi)
                return (None, path, f"unsupported meter: {time_sigs}")
            return (midi, path, None)
        except Exception as e:
            return (None, path, str(e))

    n_workers = min(12, len(file_paths))
    loaded_count = 0
    skipped_meter = 0
    lock = threading.Lock()

    print(f"  Using {n_workers} parallel workers...")
    print(f"  Filtering for meters: {sorted(ALLOWED_METERS)}")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(load_midi_file, p): p for p in file_paths}
        for future in as_completed(futures):
            midi, path, error = future.result()
            if midi is not None:
                midi_files.append(midi)
            elif error and error.startswith("unsupported meter"):
                skipped_meter += 1
            else:
                print(f"  [!] Failed to load {os.path.basename(path)}: {error}")

            with lock:
                loaded_count += 1
                if loaded_count % 100 == 0:
                    print(f"  Loaded {loaded_count}/{len(file_paths)} files...")

    print(f"\n✓ Loaded {len(midi_files)} MIDI files successfully")
    if skipped_meter > 0:
        print(f"  Skipped {skipped_meter} files with unsupported meters\n")
    else:
        print()

    # Convert to hierarchical representation
    print("Converting to hierarchical representation...")
    corpus_loader = HierarchicalMIDICorpus()
    corpus = corpus_loader.load_corpus_hierarchical(midi_files, verbose=True)

    # Initialize discovery engine
    discovery = EmergentHierarchyDiscovery(
        scales=args.scales,
        max_error=args.max_error,
        min_path_frequency=args.min_composition_frequency
    )

    # Load initial primitives
    all_transforms = load_initial_transforms()
    print(f"\n✓ Loaded {len(all_transforms)} primitive transforms\n")

    # Track iteration history and state for incremental mode
    iteration_history = []
    start_time = time.time()

    # FAISS UNIFIED MODE: Skip iteration 1 entirely, use FAISS for everything
    if args.faiss_unified:
        print(f"\n{'='*70}")
        print("UNIFIED FAISS DISCOVERY (SKIPPING ITERATION 1)")
        print(f"{'='*70}")
        print("Using FAISS for ALL derivation discovery - no OOM, no piece boundaries.")
        print("This is the recommended approach for large corpora.\n")

        # Step 1: Extract all objects (still needed)
        print(f"\n{'='*70}")
        print("STEP 1: MULTI-SCALE OBJECT EXTRACTION")
        print(f"{'='*70}")
        print(f"Scales: {args.scales} timesteps")

        objects = discovery.extract_objects(corpus, verbose=True)

        print(f"✓ Extracted {len(objects)} objects")

        # Step 2: Run unified FAISS discovery
        try:
            from discovery.faiss_cross_piece import run_unified_faiss_discovery

            faiss_graph, faiss_sources, faiss_stats = run_unified_faiss_discovery(
                objects=objects,
                transforms=all_transforms,
                max_error=args.max_error,
                max_iterations=args.max_iterations,
                verbose=True
            )

            # Populate results structure to match standard mode
            results = {
                'objects': objects,
                'graph': faiss_graph,
                'sources': faiss_sources,
                'statistics': {
                    'total_objects': len(objects),
                    'total_derivations': len(faiss_graph),
                    'derivation_rate': len(faiss_graph) / len(objects) if objects else 0,
                    'total_sources': len(faiss_sources)
                },
                'compositions': [],  # Will find compositions from graph
                'meta_patterns': []
            }

            # Find compositions from the FAISS graph
            compositions, _ = discovery.find_compositions_from_paths(faiss_graph, verbose=True)
            results['compositions'] = compositions

            # Record FAISS iteration stats
            for stat in faiss_stats:
                iteration_history.append({
                    'iteration': stat.iteration,
                    'transform_count': len(all_transforms),
                    'new_compositions': stat.new_derivations,
                    'total_derivations': len(faiss_graph),
                    'derivation_rate': results['statistics']['derivation_rate'],
                    'total_compositions_discovered': len(compositions),
                    'total_sources': stat.total_sources,
                    'paths_shortened': stat.paths_shortened,
                    'avg_path_length': stat.avg_path_length,
                    'total_description_length': stat.total_description_length,
                    'time_seconds': stat.time_seconds,
                    'is_faiss_unified': True
                })

            # Set prev_ variables for final summary
            prev_sources = faiss_sources
            prev_graph = faiss_graph
            prev_objects = objects

            print(f"\n✓ FAISS unified discovery complete!")
            print(f"  Final sources: {len(faiss_sources)}")
            print(f"  Final derivations: {len(faiss_graph)}")
            if faiss_stats:
                print(f"  Final avg path length: {faiss_stats[-1].avg_path_length:.2f}")

            # V2: E-Graph Meta-Pattern Analysis
            try:
                from discovery.egraph_metapatterns import analyze_v1_results

                print(f"\n{'='*70}")
                print("V2: META-PATTERN ABSTRACTION")
                print(f"{'='*70}")

                v2_output_path = os.path.join(args.output_dir, 'meta_patterns.json')
                v2_discovery = analyze_v1_results(
                    faiss_graph,
                    output_path=v2_output_path,
                    verbose=True
                )

                # Add meta-patterns to results
                results['meta_patterns'] = [
                    p.to_dict() for p in v2_discovery.get_all_patterns()
                ]

                print(f"\n✓ V2 meta-pattern analysis complete!")
                print(f"  Discovered {len(results['meta_patterns'])} meta-patterns")
                print(f"  Saved to: {v2_output_path}")

            except ImportError as e:
                print(f"\n⚠ V2 meta-pattern module not available: {e}")
            except Exception as e:
                print(f"\n⚠ V2 meta-pattern analysis failed: {e}")

        except ImportError as e:
            print(f"\n⚠ FAISS not available: {e}")
            print("Install with: pip install faiss-gpu")
            print("Falling back to standard discovery...")
            args.faiss_unified = False
        except Exception as e:
            import traceback
            print(f"\n⚠ FAISS unified discovery failed: {e}")
            traceback.print_exc()
            print("Falling back to standard discovery...")
            args.faiss_unified = False

    # State for incremental discovery (standard mode)
    if not args.faiss_unified:
        prev_objects = None
        prev_graph = None
        prev_sources = None
        prev_transform_count = len(all_transforms)

    # Iterative discovery loop (standard mode - skipped if FAISS unified succeeded)
    for iteration in range(args.max_iterations):
        if args.faiss_unified:
            # Already done with FAISS unified, skip standard loop
            break
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{args.max_iterations}")
        print(f"{'='*70}")
        print(f"Current transform library size: {len(all_transforms)}")
        print(f"  Primitives: {len(load_initial_transforms())}")
        print(f"  Compositions: {len(all_transforms) - len(load_initial_transforms())}\n")

        iter_start = time.time()

        # Determine if we can use incremental mode
        if iteration == 0:
            # First iteration: full discovery
            new_transforms_for_testing = None
            results = discovery.run_full_discovery(
                corpus, all_transforms, verbose=True,
                use_gpu=args.gpu,
                same_piece_only=not args.cross_piece,
                num_workers=args.num_workers
            )
        else:
            # Subsequent iterations: incremental mode
            # Only test the NEW transforms added in previous iteration
            new_transforms_for_testing = all_transforms[prev_transform_count:]

            print(f"INCREMENTAL MODE: Testing {len(new_transforms_for_testing)} new transforms")
            print(f"                  against {len(prev_sources)} source objects\n")

            results = discovery.run_full_discovery(
                corpus, all_transforms, verbose=True,
                use_gpu=args.gpu,
                same_piece_only=not args.cross_piece,
                num_workers=args.num_workers,
                existing_objects=prev_objects,
                existing_graph=prev_graph,
                existing_sources=prev_sources,
                new_transforms_only=new_transforms_for_testing
            )

        # Extract compositions
        compositions = results['compositions']
        new_compositions = compositions_to_transforms(
            compositions,
            min_frequency=args.min_composition_frequency,
            max_compositions=args.max_compositions_per_iteration
        )

        # Cross-track relationships are now discovered in the main graph
        # (via is_cross_track field in Derivation objects)
        cross_track_patterns = []

        iter_time = time.time() - iter_start

        # Compute MDL stats for display (before formal recording)
        graph = results['graph']
        if graph:
            avg_path_length_display = sum(d.path_length for d in graph.values()) / len(graph)
        else:
            avg_path_length_display = 0
        paths_shortened_display = getattr(discovery, '_last_paths_shortened', 0)

        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1} RESULTS")
        print(f"{'='*70}")
        print(f"Discovered compositions: {len(new_compositions)}")
        print(f"Time: {iter_time/60:.1f} minutes")
        if iteration > 0:
            print(f"Paths shortened (MDL improvement): {paths_shortened_display}")
            print(f"Average path length: {avg_path_length_display:.2f}\n")
        else:
            print()

        if new_compositions:
            print("Top 10 new compositions:")
            for i, comp in enumerate(new_compositions[:10]):
                print(f"  {i+1}. {comp['path']}")
                print(f"     Frequency: {comp['frequency']} occurrences\n")
        else:
            print("No new compositions discovered above threshold.")

        # Compute MDL statistics
        graph = results['graph']
        if graph:
            avg_path_length = sum(d.path_length for d in graph.values()) / len(graph)
            total_description_length = sum(d.path_length for d in graph.values())
        else:
            avg_path_length = 0
            total_description_length = 0

        paths_shortened = getattr(discovery, '_last_paths_shortened', 0)

        # Record iteration stats
        iteration_history.append({
            'iteration': iteration + 1,
            'transform_count': len(all_transforms),
            'new_compositions': len(new_compositions),
            'total_derivations': results['statistics']['total_derivations'],
            'derivation_rate': results['statistics']['derivation_rate'],
            'total_compositions_discovered': len(compositions),
            'total_sources': results['statistics']['total_sources'],
            'paths_shortened': paths_shortened,
            'avg_path_length': avg_path_length,
            'total_description_length': total_description_length,
            'time_seconds': iter_time
        })

        # Save iteration results
        iteration_output = {
            'iteration': iteration + 1,
            'transforms': all_transforms,
            'statistics': results['statistics'],
            'new_compositions': new_compositions,
            'all_compositions': compositions[:100],  # Top 100
            'meta_patterns': len(results['meta_patterns']),
            'cross_track_patterns': cross_track_patterns if iteration == 0 else []
        }

        output_path = os.path.join(args.output_dir, f'iteration_{iteration+1}.json')
        with open(output_path, 'w') as f:
            json.dump(iteration_output, f, indent=2)
        print(f"✓ Saved iteration results to {output_path}")

        # Update state for next iteration
        prev_objects = results['objects']
        prev_graph = results['graph']
        prev_sources = results['sources']
        prev_transform_count = len(all_transforms)

        # Check convergence (early stopping)
        if len(new_compositions) == 0:
            print(f"\n{'='*70}")
            print("CONVERGENCE REACHED (NO NEW COMPOSITIONS)")
            print(f"{'='*70}")
            print("No new compositions discovered above threshold.")
            print("Discovery has converged!")
            break

        # Check if sources stabilized AND no paths shortened (early stopping with MDL)
        if iteration > 0 and len(prev_sources) == results['statistics']['total_sources'] and paths_shortened == 0:
            print(f"\n{'='*70}")
            print("CONVERGENCE REACHED (SOURCES STABILIZED + NO MDL IMPROVEMENT)")
            print(f"{'='*70}")
            print(f"Source count unchanged: {len(prev_sources)} sources")
            print(f"Paths shortened: {paths_shortened}")
            print("No further progress possible - discovery has converged!")
            break

        # Add new compositions to transform library
        all_transforms.extend(new_compositions)
        print(f"\n✓ Added {len(new_compositions)} compositions to transform library")
        print(f"New library size: {len(all_transforms)}")

    total_time = time.time() - start_time

    # FAISS cross-piece MDL optimization (optional)
    faiss_stats = []
    if args.faiss_cross_piece:
        print(f"\n{'='*70}")
        print("STARTING FAISS CROSS-PIECE MDL OPTIMIZATION")
        print(f"{'='*70}")
        print("This will find cross-piece derivations and further optimize MDL...")

        try:
            from discovery.faiss_cross_piece import run_cross_piece_mdl

            faiss_graph, faiss_sources, faiss_stats = run_cross_piece_mdl(
                objects=results['objects'],
                transforms=all_transforms,
                existing_graph=results['graph'],
                existing_sources=results['sources'],
                max_error=args.max_error,
                max_iterations=10,
                verbose=True
            )

            # Update results with FAISS optimization
            results['graph'] = faiss_graph
            results['sources'] = faiss_sources
            results['statistics']['total_sources'] = len(faiss_sources)
            results['statistics']['total_derivations'] = len(faiss_graph)
            results['statistics']['derivation_rate'] = len(faiss_graph) / len(results['objects'])

            # Add FAISS stats to iteration history
            for stat in faiss_stats:
                iteration_history.append({
                    'iteration': len(iteration_history) + 1,
                    'transform_count': len(all_transforms),
                    'new_compositions': 0,
                    'total_derivations': len(faiss_graph),
                    'derivation_rate': results['statistics']['derivation_rate'],
                    'total_compositions_discovered': 0,
                    'total_sources': stat.total_sources,
                    'paths_shortened': stat.paths_shortened,
                    'avg_path_length': stat.avg_path_length,
                    'total_description_length': stat.total_description_length,
                    'time_seconds': stat.time_seconds,
                    'is_faiss_cross_piece': True
                })

            print(f"\n✓ FAISS cross-piece MDL optimization complete!")
            print(f"  Sources reduced: {len(prev_sources)} → {len(faiss_sources)}")
            print(f"  Final avg path length: {faiss_stats[-1].avg_path_length:.2f}")

        except ImportError as e:
            print(f"\n⚠ FAISS not available: {e}")
            print("Install with: pip install faiss-gpu")
            print("Skipping cross-piece optimization...")
        except Exception as e:
            print(f"\n⚠ FAISS cross-piece failed: {e}")
            print("Continuing with within-piece results...")

    total_time = time.time() - start_time

    # Final summary
    print(f"\n{'='*70}")
    print("DISCOVERY COMPLETE")
    print(f"{'='*70}")
    print(f"Total iterations: {len(iteration_history)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per iteration: {total_time/len(iteration_history)/60:.1f} minutes")
    print(f"Final transform library size: {len(all_transforms)}")
    print(f"  Primitives: {len(load_initial_transforms())}")
    print(f"  Discovered compositions: {len(all_transforms) - len(load_initial_transforms())}")

    # Save final results
    final_results = {
        'corpus_path': args.corpus_path,
        'num_files': len(midi_files),
        'total_iterations': len(iteration_history),
        'total_time_seconds': total_time,
        'final_transform_count': len(all_transforms),
        'iteration_history': iteration_history,
        'final_transforms': all_transforms,
        'parameters': vars(args)
    }

    final_path = os.path.join(args.output_dir, 'final_results.json')
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✓ Saved final results to {final_path}")
    print(f"✓ All iteration results saved to {args.output_dir}/")

    print(f"\n{'='*70}")
    print("SUCCESS")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
