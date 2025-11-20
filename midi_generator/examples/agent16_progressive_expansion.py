#!/usr/bin/env python3
"""
Agent 16: Progressive Expansion - Advanced Demo
================================================

This demo showcases progressive, iterative system expansion where the system
continuously improves by analyzing diverse musical datasets.

Key Concepts Demonstrated:
1. Multi-epoch expansion (continuous learning)
2. Dataset diversity sampling
3. Convergence detection
4. Quality tracking over time
5. Parameter accumulation strategy
6. Active learning (intelligent file selection)

Use Cases:
- Bootstrap system from scratch on large dataset
- Continuous improvement deployment
- Genre-specific capability expansion
- Quality-driven development cycle

Usage:
    # Progressive expansion on dataset
    python agent16_progressive_expansion.py \
        --dataset path/to/midi/files \
        --epochs 10 \
        --files-per-epoch 20 \
        --mock

    # Real mode with API
    export ANTHROPIC_API_KEY='your-key'
    python agent16_progressive_expansion.py \
        --dataset path/to/midi/files \
        --real

Author: Agent 16 - Progressive Expansion Demo
License: MIT
"""

import sys
import os
from pathlib import Path
import argparse
import glob
import random
import time
from datetime import datetime
from typing import List, Dict
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_generator.orchestration import (
    ExpansionOrchestrator,
    ExpansionWorkflowResult
)


class ProgressiveExpansionManager:
    """
    Manages progressive expansion across multiple epochs
    """

    def __init__(self, orchestrator: ExpansionOrchestrator, output_dir: Path = None):
        self.orchestrator = orchestrator
        self.output_dir = output_dir or Path('expansion_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epoch_results: List[Dict] = []
        self.total_parameters_added = 0
        self.quality_history = []

    def run_progressive_expansion(self,
                                   midi_files: List[Path],
                                   max_epochs: int = 10,
                                   files_per_epoch: int = 20,
                                   convergence_threshold: float = 0.01,
                                   max_expansions_per_file: int = 2) -> Dict:
        """
        Run progressive expansion across multiple epochs

        Args:
            midi_files: List of MIDI files to analyze
            max_epochs: Maximum number of epochs
            files_per_epoch: Files to process per epoch
            convergence_threshold: Stop if avg improvement < this
            max_expansions_per_file: Max new params per file

        Returns:
            Summary dictionary
        """
        print(f"\n{'='*80}")
        print(f"PROGRESSIVE EXPANSION - STARTING")
        print(f"{'='*80}")
        print(f"\n📝 Configuration:")
        print(f"   Dataset size: {len(midi_files)} files")
        print(f"   Max epochs: {max_epochs}")
        print(f"   Files per epoch: {files_per_epoch}")
        print(f"   Convergence threshold: {convergence_threshold:.3f}")
        print(f"   Max expansions per file: {max_expansions_per_file}")
        print(f"   Output directory: {self.output_dir}")

        start_time = time.time()

        for epoch in range(1, max_epochs + 1):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch}/{max_epochs}")
            print(f"{'='*80}")

            # Select files for this epoch
            epoch_files = self._select_epoch_files(
                midi_files,
                files_per_epoch,
                epoch
            )

            print(f"\n📂 Selected {len(epoch_files)} files for epoch {epoch}")

            # Run expansion on selected files
            epoch_start = time.time()

            results = self.orchestrator.batch_expand_from_dataset(
                midi_files=epoch_files,
                max_expansions_per_file=max_expansions_per_file
            )

            epoch_time = time.time() - epoch_start

            # Analyze epoch results
            epoch_summary = self._analyze_epoch_results(
                epoch,
                results,
                epoch_time
            )

            self.epoch_results.append(epoch_summary)

            # Print epoch summary
            self._print_epoch_summary(epoch_summary)

            # Check convergence
            if epoch_summary['avg_improvement'] < convergence_threshold:
                print(f"\n✅ CONVERGED!")
                print(f"   Average improvement ({epoch_summary['avg_improvement']:.3f}) ")
                print(f"   below threshold ({convergence_threshold:.3f})")
                break

            # Save intermediate results
            self._save_checkpoint(epoch)

        # Final summary
        total_time = time.time() - start_time
        summary = self._generate_final_summary(total_time)

        # Save final results
        self._save_final_results(summary)

        # Print final summary
        self._print_final_summary(summary)

        return summary

    def _select_epoch_files(self,
                           midi_files: List[Path],
                           n_files: int,
                           epoch: int) -> List[Path]:
        """
        Select files for epoch using intelligent sampling

        Strategy:
        - Epoch 1: Random diverse sample
        - Later epochs: Mix of random + worst reconstructing files
        """
        n_files = min(n_files, len(midi_files))

        if epoch == 1:
            # Random sample for first epoch
            return random.sample(midi_files, n_files)
        else:
            # TODO: Implement active learning
            # - Analyze which files have worst reconstruction
            # - Prioritize those for expansion
            # For now, just random sample
            return random.sample(midi_files, n_files)

    def _analyze_epoch_results(self,
                               epoch: int,
                               results: Dict[str, ExpansionWorkflowResult],
                               epoch_time: float) -> Dict:
        """Analyze results from one epoch"""

        successful = [r for r in results.values() if r.success]
        failed = [r for r in results.values() if not r.success]

        total_params = sum(len(r.expansions_deployed) for r in successful)
        avg_improvement = (
            sum(r.quality_improvement for r in successful) / len(successful)
            if successful else 0.0
        )
        avg_initial = (
            sum(r.initial_quality for r in successful) / len(successful)
            if successful else 0.0
        )
        avg_final = (
            sum(r.final_quality for r in successful) / len(successful)
            if successful else 0.0
        )

        self.total_parameters_added += total_params
        self.quality_history.append({
            'epoch': epoch,
            'avg_quality': avg_final,
            'avg_improvement': avg_improvement
        })

        return {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'files_processed': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'parameters_added': total_params,
            'total_parameters': self.total_parameters_added,
            'avg_improvement': avg_improvement,
            'avg_initial_quality': avg_initial,
            'avg_final_quality': avg_final,
            'epoch_time': epoch_time
        }

    def _print_epoch_summary(self, summary: Dict):
        """Print epoch summary"""
        print(f"\n{'─'*80}")
        print(f"EPOCH {summary['epoch']} SUMMARY")
        print(f"{'─'*80}")
        print(f"\n📊 Results:")
        print(f"   Files processed: {summary['files_processed']}")
        print(f"   Successful: {summary['successful']} ✅")
        print(f"   Failed: {summary['failed']} ❌")
        print(f"   Parameters added: {summary['parameters_added']}")
        print(f"   Total parameters: {summary['total_parameters']}")
        print(f"\n📈 Quality:")
        print(f"   Avg improvement: {summary['avg_improvement']:+.3f}")
        print(f"   Avg quality: {summary['avg_initial_quality']:.3f} → {summary['avg_final_quality']:.3f}")
        print(f"\n⏱  Time: {summary['epoch_time']:.1f}s")

    def _generate_final_summary(self, total_time: float) -> Dict:
        """Generate final summary"""

        total_epochs = len(self.epoch_results)
        total_files = sum(e['files_processed'] for e in self.epoch_results)
        total_successful = sum(e['successful'] for e in self.epoch_results)
        total_failed = sum(e['failed'] for e in self.epoch_results)

        # Quality progression
        initial_quality = self.quality_history[0]['avg_quality'] if self.quality_history else 0.0
        final_quality = self.quality_history[-1]['avg_quality'] if self.quality_history else 0.0
        total_improvement = final_quality - initial_quality

        # Time statistics
        avg_time_per_epoch = total_time / total_epochs if total_epochs > 0 else 0.0
        total_epoch_time = sum(e['epoch_time'] for e in self.epoch_results)

        return {
            'total_epochs': total_epochs,
            'total_files_processed': total_files,
            'total_successful': total_successful,
            'total_failed': total_failed,
            'total_parameters_added': self.total_parameters_added,
            'initial_quality': initial_quality,
            'final_quality': final_quality,
            'total_quality_improvement': total_improvement,
            'total_time': total_time,
            'avg_time_per_epoch': avg_time_per_epoch,
            'total_expansion_time': total_epoch_time,
            'quality_history': self.quality_history,
            'epoch_details': self.epoch_results
        }

    def _print_final_summary(self, summary: Dict):
        """Print final summary"""
        print(f"\n{'='*80}")
        print(f"PROGRESSIVE EXPANSION COMPLETE")
        print(f"{'='*80}")

        print(f"\n📊 Overall Statistics:")
        print(f"   Total epochs: {summary['total_epochs']}")
        print(f"   Total files: {summary['total_files_processed']}")
        print(f"   Successful: {summary['total_successful']} ✅")
        print(f"   Failed: {summary['total_failed']} ❌")
        print(f"   Success rate: {summary['total_successful']/summary['total_files_processed']*100:.1f}%")

        print(f"\n🎯 Parameter Expansion:")
        print(f"   Total parameters added: {summary['total_parameters_added']}")
        print(f"   Avg per epoch: {summary['total_parameters_added']/summary['total_epochs']:.1f}")
        print(f"   Avg per file: {summary['total_parameters_added']/summary['total_successful']:.2f}")

        print(f"\n📈 Quality Progression:")
        print(f"   Initial quality: {summary['initial_quality']:.3f}")
        print(f"   Final quality: {summary['final_quality']:.3f}")
        print(f"   Total improvement: {summary['total_quality_improvement']:+.3f} ({summary['total_quality_improvement']*100:+.1f}%)")

        print(f"\n⏱  Time Statistics:")
        print(f"   Total time: {summary['total_time']:.1f}s ({summary['total_time']/60:.1f}m)")
        print(f"   Expansion time: {summary['total_expansion_time']:.1f}s")
        print(f"   Avg per epoch: {summary['avg_time_per_epoch']:.1f}s")

        # Quality progression table
        if summary['quality_history']:
            print(f"\n📊 Quality Progression by Epoch:")
            print(f"   {'Epoch':<8} {'Quality':<10} {'Improvement':<12}")
            print(f"   {'-'*30}")
            for entry in summary['quality_history']:
                print(f"   {entry['epoch']:<8} {entry['avg_quality']:<10.3f} {entry['avg_improvement']:+.3f}")

    def _save_checkpoint(self, epoch: int):
        """Save checkpoint after epoch"""
        checkpoint_file = self.output_dir / f'epoch_{epoch}_checkpoint.json'

        checkpoint_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'epoch_results': self.epoch_results,
            'total_parameters_added': self.total_parameters_added,
            'quality_history': self.quality_history
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def _save_final_results(self, summary: Dict):
        """Save final results"""
        results_file = self.output_dir / 'progressive_expansion_results.json'

        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Results saved to: {results_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Agent 16: Progressive Expansion Demo'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='examples/test_inputs',
        help='Path to MIDI dataset directory'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='**/*.mid',
        help='Glob pattern for MIDI files (default: **/*.mid)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Maximum number of epochs (default: 5)'
    )
    parser.add_argument(
        '--files-per-epoch',
        type=int,
        default=10,
        help='Files to process per epoch (default: 10)'
    )
    parser.add_argument(
        '--convergence',
        type=float,
        default=0.01,
        help='Convergence threshold (default: 0.01)'
    )
    parser.add_argument(
        '--max-expansions',
        type=int,
        default=2,
        help='Max expansions per file (default: 2)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='expansion_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Run in mock mode (no API calls)'
    )
    parser.add_argument(
        '--real',
        action='store_true',
        help='Run with real API (requires ANTHROPIC_API_KEY)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Print header
    print(f"\n{'='*80}")
    print(f"Agent 16: Progressive Expansion Demo")
    print(f"{'='*80}")

    # Determine mode
    if args.real:
        mock_mode = False
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("\n❌ Error: ANTHROPIC_API_KEY environment variable not set")
            print("   Set it with: export ANTHROPIC_API_KEY='your-key'")
            sys.exit(1)
    else:
        mock_mode = True
        api_key = None

    print(f"\n⚙️  Configuration:")
    print(f"   Mode: {'🤖 Real API' if not mock_mode else '🎭 Mock'}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Pattern: {args.pattern}")
    print(f"   Random seed: {args.seed}")

    # Find MIDI files
    print(f"\n📂 Scanning for MIDI files...")
    dataset_path = Path(args.dataset)

    if dataset_path.is_file():
        midi_files = [dataset_path]
    else:
        search_pattern = str(dataset_path / args.pattern)
        midi_files = [Path(f) for f in glob.glob(search_pattern, recursive=True)]

    if not midi_files:
        print(f"❌ No MIDI files found matching pattern: {search_pattern}")
        print(f"\n💡 Tips:")
        print(f"   - Check the dataset path exists")
        print(f"   - Try different pattern (e.g., '*.mid' or '**/*.midi')")
        print(f"   - Use --dataset to specify correct directory")
        sys.exit(1)

    print(f"   ✅ Found {len(midi_files)} MIDI files")

    # Initialize orchestrator
    print(f"\n📦 Initializing Expansion Orchestrator...")
    orchestrator = ExpansionOrchestrator(
        api_key=api_key,
        mock_mode=mock_mode
    )
    print(f"   ✅ Orchestrator initialized")

    # Initialize progressive expansion manager
    manager = ProgressiveExpansionManager(
        orchestrator=orchestrator,
        output_dir=Path(args.output)
    )

    try:
        # Run progressive expansion
        summary = manager.run_progressive_expansion(
            midi_files=midi_files,
            max_epochs=args.epochs,
            files_per_epoch=args.files_per_epoch,
            convergence_threshold=args.convergence,
            max_expansions_per_file=args.max_expansions
        )

        print(f"\n{'='*80}")
        print(f"✅ Progressive expansion completed successfully!")
        print(f"{'='*80}")

        print(f"\n📚 Next steps:")
        print(f"   1. Review results in: {args.output}/")
        print(f"   2. Analyze quality progression")
        print(f"   3. Deploy successful parameters")
        print(f"   4. Run validation tests")

        sys.exit(0)

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Progressive expansion interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during progressive expansion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
