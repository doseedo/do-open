#!/usr/bin/env python3
"""
Parallel Training Script for Modular Semantic Discovery Pipeline
================================================================

This script trains all 6 modular encoders (5 domain + 1 cross-dimensional)
to discover 120 interpretable musical parameters.

Features:
- Parallel training across multiple GPUs/CPUs
- Checkpoint management and resumption
- Progress monitoring with TensorBoard
- Automatic resource allocation
- Training time estimation

Usage:
    # Basic training (CPU, sequential)
    python train_modular_pipeline.py --corpus /path/to/midi --output /path/to/output

    # Parallel training on multiple GPUs
    python train_modular_pipeline.py --corpus /path/to/midi --output /path/to/output \\
        --parallel --devices cuda:0 cuda:1 cuda:2

    # Resume from checkpoint
    python train_modular_pipeline.py --corpus /path/to/midi --output /path/to/output \\
        --resume /path/to/checkpoint

    # Quick test run (10 files, 5 epochs)
    python train_modular_pipeline.py --corpus /path/to/midi --output /path/to/output \\
        --quick-test

Author: Agent 8 - Integration Pipeline Builder
Date: November 21, 2025
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import time
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from midi_generator.learning.modular_discovery_pipeline import (
        ModularSemanticDiscoveryPipeline,
        ModularPipelineConfig,
        MusicalDNA
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("❌ Error: ModularSemanticDiscoveryPipeline not available")


# =============================================================================
# Training Configurations
# =============================================================================

def get_quick_test_config(corpus_dir: Path, output_dir: Path) -> ModularPipelineConfig:
    """Configuration for quick testing (10 files, 5 epochs)"""
    return ModularPipelineConfig(
        midi_corpus_dir=corpus_dir,
        output_dir=output_dir,
        max_files=10,
        max_epochs=5,
        batch_size=16,
        early_stopping_patience=3,
        train_encoders_parallel=False,
        verbose=True
    )


def get_standard_config(corpus_dir: Path, output_dir: Path, parallel: bool = True) -> ModularPipelineConfig:
    """Standard training configuration"""
    return ModularPipelineConfig(
        midi_corpus_dir=corpus_dir,
        output_dir=output_dir,
        max_files=None,  # Use all files
        max_epochs=100,
        batch_size=64,
        early_stopping_patience=10,
        train_encoders_parallel=parallel,
        num_parallel_workers=5,
        device="cuda" if parallel else "cpu",
        verbose=True,
        log_to_tensorboard=True
    )


def get_gpu_config(
    corpus_dir: Path,
    output_dir: Path,
    devices: List[str]
) -> ModularPipelineConfig:
    """Multi-GPU training configuration"""
    return ModularPipelineConfig(
        midi_corpus_dir=corpus_dir,
        output_dir=output_dir,
        max_files=None,
        max_epochs=100,
        batch_size=128,  # Larger batch size for multi-GPU
        early_stopping_patience=10,
        train_encoders_parallel=True,
        num_parallel_workers=len(devices),
        device=devices[0],
        devices=devices,
        verbose=True,
        log_to_tensorboard=True
    )


# =============================================================================
# Training Functions
# =============================================================================

def train_pipeline(config: ModularPipelineConfig, resume_from: Optional[Path] = None):
    """
    Train modular pipeline.

    Args:
        config: Pipeline configuration
        resume_from: Optional checkpoint directory to resume from
    """
    print("="*70)
    print("MODULAR SEMANTIC DISCOVERY - PARALLEL TRAINING")
    print("="*70)

    # Print configuration
    print("\n📋 Configuration:")
    print(f"  Corpus: {config.midi_corpus_dir}")
    print(f"  Output: {config.output_dir}")
    print(f"  Max files: {config.max_files or 'all'}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Parallel training: {config.train_encoders_parallel}")
    if config.devices:
        print(f"  Devices: {', '.join(config.devices)}")
    else:
        print(f"  Device: {config.device}")

    # Print weight sparsity settings
    print(f"\n🔬 Weight Sparsity (Superposition Reduction):")
    if config.enable_weight_sparsity:
        print(f"  Status: ENABLED")
        print(f"  Target sparsity: {config.sparsity_ratio} ({config.sparsity_ratio*100:.2f}% of weights kept)")
        print(f"  Initial sparsity: {config.initial_sparsity} ({config.initial_sparsity*100:.1f}% of weights kept)")
        print(f"  Warmup epochs: {config.sparsity_warmup_epochs}")
    else:
        print(f"  Status: DISABLED (use --enable-weight-sparsity to enable)")

    # Create pipeline
    print("\n🏗️  Initializing pipeline...")
    pipeline = ModularSemanticDiscoveryPipeline(config)

    # Resume from checkpoint if specified
    if resume_from is not None:
        print(f"\n♻️  Resuming from checkpoint: {resume_from}")
        pipeline.load(resume_from)
    else:
        # Create encoders
        pipeline.create_encoders()

    # Train
    print("\n🎓 Starting training...")
    start_time = time.time()

    try:
        pipeline.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("   Saving checkpoint...")
        pipeline.save()
        print("   ✅ Checkpoint saved")
        return
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        print("   Saving checkpoint...")
        pipeline.save()
        print("   ✅ Checkpoint saved")
        raise

    training_time = time.time() - start_time

    # Save trained pipeline
    print("\n💾 Saving trained pipeline...")
    pipeline.save()

    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"  Total time: {training_time/3600:.2f} hours")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Discovered parameters: 120")
    print("="*70)


def test_extraction(pipeline_dir: Path, test_midi: Path):
    """
    Test DNA extraction on a MIDI file.

    Args:
        pipeline_dir: Directory containing trained pipeline
        test_midi: Path to test MIDI file
    """
    print("\n🧬 Testing DNA extraction...")

    # Load pipeline
    config = ModularPipelineConfig(
        midi_corpus_dir=Path("/tmp"),  # Dummy
        output_dir=pipeline_dir
    )
    pipeline = ModularSemanticDiscoveryPipeline(config)
    pipeline.load(pipeline_dir)

    # Extract DNA
    dna = pipeline.extract_dna(test_midi)

    # Print DNA
    print("\n📊 Extracted Musical DNA:")
    print(f"  Harmony params: {len(dna.harmony_params)} values")
    print(f"    Range: [{dna.harmony_params.min():.3f}, {dna.harmony_params.max():.3f}]")
    print(f"  Rhythm params: {len(dna.rhythm_params)} values")
    print(f"    Range: [{dna.rhythm_params.min():.3f}, {dna.rhythm_params.max():.3f}]")
    print(f"  Form params: {len(dna.form_params)} values")
    print(f"    Range: [{dna.form_params.min():.3f}, {dna.form_params.max():.3f}]")
    print(f"  Orchestration params: {len(dna.orchestration_params)} values")
    print(f"    Range: [{dna.orchestration_params.min():.3f}, {dna.orchestration_params.max():.3f}]")
    print(f"  Texture params: {len(dna.texture_params)} values")
    print(f"    Range: [{dna.texture_params.min():.3f}, {dna.texture_params.max():.3f}]")
    print(f"  Cross-dimensional params: {len(dna.cross_params)} values")
    print(f"    Range: [{dna.cross_params.min():.3f}, {dna.cross_params.max():.3f}]")

    # Save DNA
    dna_path = pipeline_dir / f"dna_{test_midi.stem}.json"
    dna.save(dna_path)
    print(f"\n✅ DNA saved to {dna_path}")


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Modular Semantic Discovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_modular_pipeline.py --corpus ./corpus --output ./output

  # Parallel GPU training
  python train_modular_pipeline.py --corpus ./corpus --output ./output \\
      --parallel --devices cuda:0 cuda:1

  # Quick test
  python train_modular_pipeline.py --corpus ./corpus --output ./output --quick-test

  # Resume training
  python train_modular_pipeline.py --corpus ./corpus --output ./output \\
      --resume ./output/checkpoint
        """
    )

    # Required arguments
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to MIDI corpus directory"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output directory"
    )

    # Training mode
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode (10 files, 5 epochs)"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel training of domain encoders"
    )

    parser.add_argument(
        "--devices",
        nargs="+",
        help="Devices for training (e.g., cuda:0 cuda:1 cpu)"
    )

    # Resumption
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume training from checkpoint directory"
    )

    # Testing
    parser.add_argument(
        "--test-extraction",
        type=Path,
        help="Test DNA extraction on specified MIDI file (requires trained pipeline)"
    )

    # Training parameters
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of MIDI files to use"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )

    # Weight sparsity parameters (for superposition reduction)
    parser.add_argument(
        "--enable-weight-sparsity",
        action="store_true",
        help="Enable weight sparsity for superposition reduction (default: off)"
    )

    parser.add_argument(
        "--sparsity-ratio",
        type=float,
        default=0.001,
        help="Target sparsity ratio - proportion of weights to keep (default: 0.001 = 0.1%%)"
    )

    parser.add_argument(
        "--initial-sparsity",
        type=float,
        default=0.5,
        help="Initial sparsity ratio at start of training (default: 0.5 = 50%%)"
    )

    parser.add_argument(
        "--sparsity-warmup-epochs",
        type=int,
        default=50,
        help="Number of epochs to gradually increase sparsity (default: 50)"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Check if pipeline is available
    if not PIPELINE_AVAILABLE:
        print("❌ ModularSemanticDiscoveryPipeline not available")
        print("   Please ensure the pipeline module is installed")
        sys.exit(1)

    # Check corpus directory exists
    if not args.corpus.exists():
        print(f"❌ Corpus directory not found: {args.corpus}")
        sys.exit(1)

    # Test extraction mode
    if args.test_extraction:
        if not args.output.exists():
            print(f"❌ Pipeline directory not found: {args.output}")
            sys.exit(1)
        if not args.test_extraction.exists():
            print(f"❌ Test MIDI file not found: {args.test_extraction}")
            sys.exit(1)

        test_extraction(args.output, args.test_extraction)
        return

    # Create configuration
    if args.quick_test:
        config = get_quick_test_config(args.corpus, args.output)
    elif args.devices:
        config = get_gpu_config(args.corpus, args.output, args.devices)
    elif args.parallel:
        config = get_standard_config(args.corpus, args.output, parallel=True)
    else:
        config = get_standard_config(args.corpus, args.output, parallel=False)

    # Override with command line arguments
    if args.max_files:
        config.max_files = args.max_files
    if args.epochs:
        config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    # Weight sparsity parameters (default: off)
    if args.enable_weight_sparsity:
        config.enable_weight_sparsity = True
        config.sparsity_ratio = args.sparsity_ratio
        config.initial_sparsity = args.initial_sparsity
        config.sparsity_warmup_epochs = args.sparsity_warmup_epochs

    # Train pipeline
    train_pipeline(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
