#!/usr/bin/env python3
"""
Discovery Pipeline Startup Script

Runs the first iteration of neural program synthesis discovery on the big_band corpus.

Usage:
    python scripts/start_discovery.py [--batch-size 1731] [--iterations 1] [--device cuda]

Expected time:
    - GPU: 10-30 minutes per iteration
    - CPU: 8-12 hours per iteration (not recommended)

Author: Agent 8 - GPU Tensorization
"""

import sys
import argparse
from pathlib import Path
import mido
import torch
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "1_approaches/transform_based"))

from discovery.gpu_discovery_pipeline import GPUDiscoveryPipeline
from core.minimal_theoretical_base import get_minimal_base
from core.gpu_memory_manager import GPUMemoryManager


def transforms_to_dicts(transforms):
    """Convert SpaceLevelTransform objects to dict format."""
    result = []
    for t in transforms:
        # Check if it's already a dict
        if isinstance(t, dict):
            result.append(t)
        else:
            # Convert SpaceLevelTransform to dict
            result.append({
                'name': t.name,
                'amount': getattr(t, 'amount', 1.0)
            })
    return result


def main():
    parser = argparse.ArgumentParser(description='Start discovery pipeline')
    parser.add_argument('--corpus-path', type=str,
                       default=None,
                       help='Path to MIDI corpus directory')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Number of files to process (default: all that fit in memory)')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of discovery iterations to run (default: 1)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: cuda or cpu (default: cuda)')
    parser.add_argument('--max-time-steps', type=int, default=2000,
                       help='Maximum time steps per piece (default: 2000)')
    parser.add_argument('--max-transforms-per-iteration', type=int, default=50,
                       help='Number of best transforms to keep per iteration (default: 50)')
    parser.add_argument('--checkpoint-dir', type=str,
                       default=None,
                       help='Directory to save checkpoints')
    parser.add_argument('--target-quality', type=float, default=0.99,
                       help='Target reconstruction quality to reach (default: 0.99)')

    args = parser.parse_args()

    # Determine paths relative to project root
    project_root = Path(__file__).parent.parent

    if args.corpus_path is None:
        args.corpus_path = str(project_root / "midi_corpus/big_band")

    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(project_root / "checkpoints")

    print("="*70)
    print("NEURAL PROGRAM SYNTHESIS - DISCOVERY PIPELINE")
    print("="*70)
    print()
    print("Configuration:")
    print(f"  Corpus path: {args.corpus_path}")
    print(f"  Device: {args.device}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Target quality: {args.target_quality:.1%}")
    print(f"  Max transforms/iteration: {args.max_transforms_per_iteration}")
    print()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[!] WARNING: CUDA requested but not available!")
        print("    Falling back to CPU (will be MUCH slower)")
        print("    Install PyTorch with CUDA:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print()
        args.device = 'cpu'

    # Find all MIDI files
    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        print(f"[!] ERROR: Corpus path not found: {corpus_path}")
        print("    Please verify the path or use --corpus-path")
        return 1

    all_midi_files = sorted(corpus_path.glob("*.mid"))
    total_files = len(all_midi_files)

    if total_files == 0:
        print(f"[!] ERROR: No MIDI files found in {corpus_path}")
        return 1

    print(f"Found {total_files:,} MIDI files in corpus\n")

    # Memory check (GPU only)
    if args.device == 'cuda':
        mem_manager = GPUMemoryManager(device='cuda', reserve_gb=10.0)

        if args.batch_size is None:
            # Auto-determine batch size
            batch_size, recommendation = mem_manager.recommend_batch_size(
                total_pieces=total_files,
                num_transforms=500,
                max_time_steps=args.max_time_steps,
                num_features=133
            )
            print(f"Memory recommendation: {recommendation}\n")
            args.batch_size = batch_size
        else:
            print(f"Using specified batch size: {args.batch_size:,}\n")
    else:
        # CPU mode - use smaller batch
        if args.batch_size is None:
            args.batch_size = min(100, total_files)
            print(f"[CPU Mode] Using batch size: {args.batch_size}")
            print("(CPU is slow - consider using smaller batch or GPU)\n")

    # Load corpus
    print("="*70)
    print("LOADING CORPUS")
    print("="*70)

    midi_files = []
    print(f"Loading {min(args.batch_size, total_files):,} MIDI files...")

    for i, midi_path in enumerate(all_midi_files[:args.batch_size]):
        try:
            midi_files.append(mido.MidiFile(midi_path))
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1:,}/{min(args.batch_size, total_files):,} files...")
        except Exception as e:
            print(f"  [!] Warning: Failed to load {midi_path.name}: {e}")

    print(f"\n✓ Loaded {len(midi_files):,} MIDI files successfully")
    print()

    # Load base primitives
    print("="*70)
    print("LOADING BASE PRIMITIVES")
    print("="*70)

    base_primitives_objects = get_minimal_base()
    base_primitives = transforms_to_dicts(base_primitives_objects)

    print(f"\n✓ Loaded {len(base_primitives)} irreducible primitives")
    print(f"  Transform names: {', '.join([t['name'] for t in base_primitives[:5]])}...")
    print()

    # Initialize pipeline
    pipeline = GPUDiscoveryPipeline(
        device=args.device,
        batch_size=args.batch_size,
        chunk_size=1000,
        lambda_sparsity=0.1
    )

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Run discovery
    if args.iterations == 1:
        print("="*70)
        print("RUNNING SINGLE DISCOVERY ITERATION")
        print("="*70)
        print()

        # Load corpus to GPU
        corpus_tensor = pipeline.load_corpus_to_gpu(midi_files)

        # Run iteration
        results = pipeline.discovery_iteration_gpu(
            corpus_tensor=corpus_tensor,
            existing_transforms=base_primitives,
            target_quality=args.target_quality,
            max_new_transforms=args.max_transforms_per_iteration
        )

        # Build final transform list
        all_transforms = base_primitives + results['new_transforms']

        print(f"\n{'='*70}")
        print("ITERATION 1 COMPLETE")
        print(f"{'='*70}")
        print(f"Dictionary size: {len(base_primitives)} → {len(all_transforms)} transforms")
        print(f"Quality: {results['quality']:.1%}")
        print(f"Sparsity: {results['sparsity']:.1f} transforms/piece")
        print(f"Candidates tested: {results['candidates_tested']}")
        print(f"Time elapsed: {results['time']:.1f}s ({results['time']/60:.1f}m)")
        print()

        # Save checkpoint
        metrics = {
            'iteration': 1,
            'quality': results['quality'],
            'sparsity': results['sparsity'],
            'dict_size': len(all_transforms),
            'new_transforms': len(results['new_transforms']),
            'time_seconds': results['time']
        }

        with open(checkpoint_dir / 'iteration_1_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        torch.save({
            'transforms': all_transforms,
            'metrics': metrics
        }, checkpoint_dir / 'iteration_1_dict.pt')

        print(f"✓ Results saved to {checkpoint_dir}/")
        print(f"  - iteration_1_metrics.json")
        print(f"  - iteration_1_dict.pt")

    else:
        print("="*70)
        print(f"RUNNING {args.iterations} DISCOVERY ITERATIONS")
        print("="*70)
        print()

        results = pipeline.run_full_discovery(
            midi_files=midi_files,
            initial_transforms=base_primitives,
            target_quality=args.target_quality,
            max_iterations=args.iterations,
            max_transforms_per_iteration=args.max_transforms_per_iteration
        )

        print(f"\n{'='*70}")
        print("FULL DISCOVERY COMPLETE")
        print(f"{'='*70}")
        print(f"Final dictionary size: {len(results['all_transforms'])} transforms")
        print(f"Final quality: {results['final_quality']:.1%}")
        print(f"Total iterations: {results['iterations']}")
        print(f"Total time: {results['total_time']/60:.1f} minutes")
        print()

        # Print iteration summary
        print("Iteration Summary:")
        for i, iter_info in enumerate(results['iteration_history'], 1):
            print(f"  Iteration {i}: {iter_info['total_transforms']} transforms, " +
                  f"quality={iter_info['quality']:.1%}, " +
                  f"{iter_info['time']:.1f}s")
        print()

        # Save final results
        with open(checkpoint_dir / 'final_metrics.json', 'w') as f:
            json.dump({
                'final_quality': results['final_quality'],
                'iterations': results['iterations'],
                'total_time': results['total_time'],
                'iteration_history': results['iteration_history']
            }, f, indent=2)

        torch.save({
            'transforms': results['all_transforms'],
            'metrics': results
        }, checkpoint_dir / 'final_dict.pt')

        print(f"✓ Results saved to {checkpoint_dir}/")
        print(f"  - final_metrics.json")
        print(f"  - final_dict.pt")

    print()
    print("="*70)
    print("DISCOVERY COMPLETE")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Run V2 abstraction for hierarchical patterns")
    print("  2. Generate new music using learned transforms")
    print("  3. Analyze discovered patterns")
    print()
    print("See DISCOVERY_QUICKSTART.md for details")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
