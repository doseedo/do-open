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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "1_approaches/transform_based"))

from core.tensor_representation import load_corpus_to_gpu, TensorMIDICorpus
from core.gpu_memory_manager import GPUMemoryManager
from discovery.gpu_discovery_pipeline import discovery_iteration_gpu, run_full_discovery
from core.minimal_theoretical_base import get_base_primitives


def main():
    parser = argparse.ArgumentParser(description='Start discovery pipeline')
    parser.add_argument('--corpus-path', type=str,
                       default='/home/user/Do/midi_generator/midi_corpus/big_band',
                       help='Path to MIDI corpus directory')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Number of files to process (default: all that fit in memory)')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of discovery iterations to run (default: 1)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: cuda or cpu (default: cuda)')
    parser.add_argument('--max-time-steps', type=int, default=2000,
                       help='Maximum time steps per piece (default: 2000)')
    parser.add_argument('--num-candidates', type=int, default=10000,
                       help='Number of composition candidates to test (default: 10000)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Number of best transforms to keep per iteration (default: 50)')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='/home/user/Do/midi_generator/checkpoints',
                       help='Directory to save checkpoints')

    args = parser.parse_args()

    print("="*70)
    print("NEURAL PROGRAM SYNTHESIS - DISCOVERY PIPELINE")
    print("="*70)
    print()
    print("Configuration:")
    print(f"  Corpus path: {args.corpus_path}")
    print(f"  Device: {args.device}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Candidates per iteration: {args.num_candidates:,}")
    print(f"  Top-k per iteration: {args.top_k}")
    print()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[!] WARNING: CUDA requested but not available!")
        print("    Falling back to CPU (will be MUCH slower)")
        print("    Install PyTorch with CUDA:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print()
        args.device = 'cpu'

    # Memory check (GPU only)
    if args.device == 'cuda':
        mem_manager = GPUMemoryManager(device='cuda', reserve_gb=10.0)

        # Find all MIDI files
        corpus_path = Path(args.corpus_path)
        all_midi_files = sorted(corpus_path.glob("*.mid"))
        total_files = len(all_midi_files)

        print(f"Found {total_files:,} MIDI files in corpus\n")

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
        # CPU mode - just count files
        corpus_path = Path(args.corpus_path)
        all_midi_files = sorted(corpus_path.glob("*.mid"))
        total_files = len(all_midi_files)

        print(f"Found {total_files:,} MIDI files in corpus")

        if args.batch_size is None:
            args.batch_size = min(100, total_files)  # Default to 100 for CPU
            print(f"[CPU Mode] Using batch size: {args.batch_size}")
            print("(CPU is slow - consider using smaller batch or GPU)\n")

    # Load corpus
    print("="*70)
    print("STEP 1: LOADING CORPUS TO GPU")
    print("="*70)

    midi_files = [mido.MidiFile(f) for f in all_midi_files[:args.batch_size]]

    print(f"Loading {len(midi_files):,} MIDI files to {args.device}...")
    corpus_tensor, converter = load_corpus_to_gpu(
        midi_files,
        max_time_steps=args.max_time_steps,
        device=args.device
    )

    print(f"\n✓ Corpus loaded successfully!")
    print(f"  Shape: {corpus_tensor.shape}")
    print(f"  Device: {corpus_tensor.device}")
    print(f"  Memory: {corpus_tensor.element_size() * corpus_tensor.nelement() / 1e9:.2f} GB")
    print()

    # Load base primitives
    print("="*70)
    print("STEP 2: LOADING BASE PRIMITIVES")
    print("="*70)

    base_primitives = get_base_primitives()

    print(f"\n✓ Loaded {len(base_primitives)} irreducible primitives:")
    print("  Transpositional: T₁, T₁⁻¹, T₁⁷, T₁⁻⁷, T₁¹², T₁⁻¹²")
    print("  Neo-Riemannian: P, L, R, PLR, LPR, RPL")
    print("  Inversion: I")
    print("  Rhythm: augmentation (×2), diminution (×½)")
    print("  Multitrack: instrument_filter, instrument_derive")
    print()

    # Run discovery
    if args.iterations == 1:
        print("="*70)
        print("STEP 3: RUNNING DISCOVERY ITERATION")
        print("="*70)

        new_dict, metrics = discovery_iteration_gpu(
            corpus_tensor=corpus_tensor,
            current_dict=base_primitives,
            num_candidates=args.num_candidates,
            top_k=args.top_k,
            device=args.device
        )

        print(f"\n{'='*70}")
        print("ITERATION 1 COMPLETE")
        print(f"{'='*70}")
        print(f"Dictionary size: {len(base_primitives)} → {len(new_dict)} transforms")
        print(f"Sparsity: {metrics['sparsity_mean']:.1f} transforms/piece")
        print(f"Reconstruction error: {metrics['reconstruction_error']:.4f}")
        print(f"Time elapsed: {metrics['time_elapsed']:.1f}s ({metrics['time_elapsed']/60:.1f}m)")
        print()

        # Save checkpoint
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        import json
        with open(checkpoint_dir / 'iteration_1_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Metrics saved to {checkpoint_dir / 'iteration_1_metrics.json'}")

    else:
        print("="*70)
        print(f"STEP 3: RUNNING {args.iterations} DISCOVERY ITERATIONS")
        print("="*70)

        final_dict, all_metrics = run_full_discovery(
            corpus_tensor=corpus_tensor,
            initial_dict=base_primitives,
            max_iterations=args.iterations,
            target_dict_size=17 + (args.iterations * args.top_k),
            num_candidates=args.num_candidates,
            top_k=args.top_k,
            device=args.device,
            save_checkpoint_every=max(1, args.iterations // 4),
            checkpoint_dir=args.checkpoint_dir
        )

        print(f"\n{'='*70}")
        print("FULL DISCOVERY COMPLETE")
        print(f"{'='*70}")
        print(f"Final dictionary size: {len(final_dict)} transforms")
        print(f"Total time: {sum(m['time_elapsed'] for m in all_metrics)/3600:.2f} hours")
        print()

        # Print iteration summary
        print("Iteration Summary:")
        for i, metrics in enumerate(all_metrics, 1):
            print(f"  Iteration {i}: {metrics['dict_size']} transforms, " +
                  f"{metrics['time_elapsed']/60:.1f}m, " +
                  f"error={metrics['reconstruction_error']:.4f}")
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


if __name__ == '__main__':
    main()
