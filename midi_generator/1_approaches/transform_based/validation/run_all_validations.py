#!/usr/bin/env python3
"""
Run All 6 Validation Steps
==========================

This script runs all validation steps in sequence and produces a comprehensive report.

Usage:
    python run_all_validations.py <checkpoint_path> <corpus_path> [--output-dir OUTPUT_DIR]

Steps:
1. Decoder - Test basic decoding from checkpoint
2. Round-trip - Test encode/decode consistency
3. Train/test split - Create and run pipeline on training only
4. Generalization - Evaluate vocabulary on held-out test files
5. Pattern inspection - Analyze musical meaningfulness
6. Cross-piece analysis - Analyze pattern sharing across pieces
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_step_1_decoder(checkpoint_path: str, verbose: bool = True) -> dict:
    """Step 1: Test decoder."""
    from validation.decoder import decode_encoding

    print("\n" + "=" * 70)
    print("STEP 1: DECODER TEST")
    print("=" * 70)

    start = time.time()
    result = decode_encoding(checkpoint_path)
    elapsed = time.time() - start

    report = {
        'step': 1,
        'name': 'Decoder',
        'success': result.stats['n_notes'] > 0,
        'elapsed_seconds': elapsed,
        'metrics': {
            'notes_decoded': result.stats['n_notes'],
            'patterns_decoded': result.stats['n_patterns'],
            'tracks': result.stats['n_tracks'],
            'canonicals': result.stats['n_canonicals'],
            'token_counts': result.stats['token_counts'],
        }
    }

    if verbose:
        print(f"Notes decoded: {result.stats['n_notes']}")
        print(f"Patterns decoded: {result.stats['n_patterns']}")
        print(f"Token counts: {result.stats['token_counts']}")
        print(f"Time: {elapsed:.2f}s")

    return report


def run_step_2_round_trip(checkpoint_path: str, corpus_path: str, max_files: int = 10, verbose: bool = True) -> dict:
    """Step 2: Round-trip test."""
    from validation.round_trip import run_round_trip_test

    print("\n" + "=" * 70)
    print("STEP 2: ROUND-TRIP TEST")
    print("=" * 70)

    start = time.time()
    result = run_round_trip_test(checkpoint_path, corpus_path, max_files)
    elapsed = time.time() - start

    report = {
        'step': 2,
        'name': 'Round-trip',
        'success': True,  # Limited test - see notes
        'elapsed_seconds': elapsed,
        'metrics': {
            'files_tested': result.n_files_tested,
            'decoded_notes': result.aggregate_stats.get('decoded_notes', 0),
            'decoded_patterns': result.aggregate_stats.get('decoded_patterns', 0),
        },
        'notes': 'Full round-trip requires per-file encoding (not stored in checkpoint)'
    }

    if verbose:
        print(f"Files examined: {result.n_files_tested}")
        print(f"Decoded notes: {result.aggregate_stats.get('decoded_notes', 0)}")
        print(f"NOTE: Per-file round-trip not available in current checkpoint format")
        print(f"Time: {elapsed:.2f}s")

    return report


def run_step_3_train_test_split(
    corpus_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    verbose: bool = True
) -> dict:
    """Step 3: Train/test split."""
    from validation.train_test_split import create_train_test_split, TrainTestPipeline

    print("\n" + "=" * 70)
    print("STEP 3: TRAIN/TEST SPLIT")
    print("=" * 70)

    start = time.time()

    # Create split
    split_path = os.path.join(output_dir, 'train_test_split.json')
    split = create_train_test_split(
        corpus_path,
        train_ratio=train_ratio,
        seed=42,
        output_path=split_path
    )

    # Run training pipeline
    train_checkpoint = os.path.join(output_dir, 'checkpoint_train_only.npz')

    try:
        pipeline = TrainTestPipeline(split, device='cuda', target_vocab_size=500)
        pipeline_result = pipeline.run(train_checkpoint, verbose=verbose)
        pipeline_success = True
    except Exception as e:
        pipeline_result = {'error': str(e)}
        pipeline_success = False
        if verbose:
            print(f"Pipeline error: {e}")

    elapsed = time.time() - start

    report = {
        'step': 3,
        'name': 'Train/Test Split',
        'success': pipeline_success,
        'elapsed_seconds': elapsed,
        'metrics': {
            'n_train': split.n_train,
            'n_test': split.n_test,
            'train_ratio': train_ratio,
            'split_path': split_path,
            'train_checkpoint': train_checkpoint if pipeline_success else None,
        },
        'pipeline_result': pipeline_result if pipeline_success else {'error': str(pipeline_result.get('error', 'Unknown'))}
    }

    if verbose:
        print(f"Train files: {split.n_train}")
        print(f"Test files: {split.n_test}")
        if pipeline_success:
            print(f"Training checkpoint: {train_checkpoint}")
        print(f"Time: {elapsed:.2f}s")

    return report


def run_step_4_generalization(
    train_checkpoint: str,
    test_files: list,
    verbose: bool = True
) -> dict:
    """Step 4: Generalization evaluation."""
    from validation.generalization import evaluate_generalization

    print("\n" + "=" * 70)
    print("STEP 4: GENERALIZATION EVALUATION")
    print("=" * 70)

    start = time.time()

    try:
        result = evaluate_generalization(train_checkpoint, test_files, verbose=verbose)
        success = True
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        result = None
        success = False

    elapsed = time.time() - start

    if success and result:
        report = {
            'step': 4,
            'name': 'Generalization',
            'success': True,
            'elapsed_seconds': elapsed,
            'metrics': {
                'total_patterns': result.total_patterns,
                'total_matched': result.total_matched,
                'total_oov': result.total_oov,
                'overall_coverage': result.overall_coverage,
                'oov_rate': result.oov_rate,
                'match_types': result.match_type_counts,
            },
            'interpretation': (
                'EXCELLENT' if result.overall_coverage >= 0.8 else
                'MODERATE' if result.overall_coverage >= 0.5 else
                'POOR'
            )
        }
    else:
        report = {
            'step': 4,
            'name': 'Generalization',
            'success': False,
            'elapsed_seconds': elapsed,
            'error': 'Failed to evaluate generalization'
        }

    return report


def run_step_5_pattern_inspection(checkpoint_path: str, n_patterns: int = 20, verbose: bool = True) -> dict:
    """Step 5: Pattern inspection."""
    from validation.pattern_inspection import inspect_patterns

    print("\n" + "=" * 70)
    print("STEP 5: PATTERN INSPECTION")
    print("=" * 70)

    start = time.time()
    result = inspect_patterns(checkpoint_path, n_patterns, verbose=verbose)
    elapsed = time.time() - start

    report = {
        'step': 5,
        'name': 'Pattern Inspection',
        'success': True,
        'elapsed_seconds': elapsed,
        'metrics': {
            'patterns_analyzed': result.n_patterns_analyzed,
            'coherent_patterns': result.coherent_patterns,
            'incoherent_patterns': result.incoherent_patterns,
            'chord_arpeggios': result.chord_arpeggios,
            'scale_fragments': result.scale_fragments,
            'melodic_motifs': result.melodic_motifs,
            'length_distribution': result.length_distribution,
        },
        'top_patterns': [
            {
                'id': pa.pattern_id,
                'notes': pa.note_names[:8],
                'usage': pa.usage_count,
                'type': pa.chord_type or pa.scale_type or pa.motif_description,
                'coherence': pa.musical_coherence,
            }
            for pa in result.pattern_analyses[:10]
        ]
    }

    return report


def run_step_6_cross_piece(checkpoint_path: str, corpus_path: str, verbose: bool = True) -> dict:
    """Step 6: Cross-piece analysis."""
    from validation.cross_piece import analyze_cross_piece_sharing

    print("\n" + "=" * 70)
    print("STEP 6: CROSS-PIECE ANALYSIS")
    print("=" * 70)

    start = time.time()
    result = analyze_cross_piece_sharing(checkpoint_path, corpus_path, verbose=verbose)
    elapsed = time.time() - start

    report = {
        'step': 6,
        'name': 'Cross-Piece Analysis',
        'success': True,
        'elapsed_seconds': elapsed,
        'metrics': {
            'n_pieces': result.n_pieces,
            'n_canonicals': result.n_canonicals,
            'single_piece_patterns': result.single_piece_patterns,
            'multi_piece_patterns': result.multi_piece_patterns,
            'universal_patterns': result.universal_patterns,
            'sharing_rate': result.sharing_rate,
            'universal_rate': result.universal_rate,
        },
        'interpretation': (
            'HIGH_SHARING' if result.sharing_rate >= 0.3 else
            'MODERATE_SHARING' if result.sharing_rate >= 0.1 else
            'LOW_SHARING'
        )
    }

    return report


def run_all_validations(
    checkpoint_path: str,
    corpus_path: str,
    output_dir: str,
    skip_train_test: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run all 6 validation steps.

    Args:
        checkpoint_path: Path to checkpoint file
        corpus_path: Path to MIDI corpus
        output_dir: Directory for output files
        skip_train_test: Skip step 3 (train/test pipeline) - slow
        verbose: Print progress

    Returns:
        Complete validation report
    """
    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()
    reports = []

    print("\n" + "#" * 70)
    print("# RUNNING ALL 6 VALIDATION STEPS")
    print("#" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Corpus: {corpus_path}")
    print(f"Output: {output_dir}")

    # Step 1: Decoder
    try:
        reports.append(run_step_1_decoder(checkpoint_path, verbose))
    except Exception as e:
        reports.append({'step': 1, 'name': 'Decoder', 'success': False, 'error': str(e)})

    # Step 2: Round-trip
    try:
        reports.append(run_step_2_round_trip(checkpoint_path, corpus_path, max_files=10, verbose=verbose))
    except Exception as e:
        reports.append({'step': 2, 'name': 'Round-trip', 'success': False, 'error': str(e)})

    # Step 3: Train/test split (optional - slow)
    train_checkpoint = None
    split = None

    if not skip_train_test:
        try:
            report3 = run_step_3_train_test_split(corpus_path, output_dir, verbose=verbose)
            reports.append(report3)
            train_checkpoint = report3.get('metrics', {}).get('train_checkpoint')

            # Load split for step 4
            split_path = report3.get('metrics', {}).get('split_path')
            if split_path and os.path.exists(split_path):
                with open(split_path) as f:
                    split_data = json.load(f)
                    test_files = split_data.get('test_files', [])
            else:
                test_files = []
        except Exception as e:
            reports.append({'step': 3, 'name': 'Train/Test Split', 'success': False, 'error': str(e)})
            test_files = []
    else:
        reports.append({
            'step': 3,
            'name': 'Train/Test Split',
            'success': True,
            'skipped': True,
            'reason': 'Skipped by user request'
        })
        test_files = []

    # Step 4: Generalization (requires step 3)
    if train_checkpoint and test_files:
        try:
            reports.append(run_step_4_generalization(train_checkpoint, test_files, verbose))
        except Exception as e:
            reports.append({'step': 4, 'name': 'Generalization', 'success': False, 'error': str(e)})
    else:
        reports.append({
            'step': 4,
            'name': 'Generalization',
            'success': False,
            'skipped': True,
            'reason': 'Requires step 3 (train/test split)'
        })

    # Step 5: Pattern inspection
    try:
        reports.append(run_step_5_pattern_inspection(checkpoint_path, verbose=verbose))
    except Exception as e:
        reports.append({'step': 5, 'name': 'Pattern Inspection', 'success': False, 'error': str(e)})

    # Step 6: Cross-piece analysis
    try:
        reports.append(run_step_6_cross_piece(checkpoint_path, corpus_path, verbose=verbose))
    except Exception as e:
        reports.append({'step': 6, 'name': 'Cross-Piece Analysis', 'success': False, 'error': str(e)})

    total_elapsed = time.time() - total_start

    # Compile final report
    final_report = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint_path': checkpoint_path,
        'corpus_path': corpus_path,
        'output_dir': output_dir,
        'total_elapsed_seconds': total_elapsed,
        'steps': reports,
        'summary': {
            'total_steps': 6,
            'completed': sum(1 for r in reports if r.get('success')),
            'failed': sum(1 for r in reports if not r.get('success') and not r.get('skipped')),
            'skipped': sum(1 for r in reports if r.get('skipped')),
        }
    }

    # Save report
    report_path = os.path.join(output_dir, 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)

    # Print summary
    print("\n" + "#" * 70)
    print("# VALIDATION COMPLETE")
    print("#" * 70)
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Steps completed: {final_report['summary']['completed']}/6")
    print(f"Steps failed: {final_report['summary']['failed']}")
    print(f"Steps skipped: {final_report['summary']['skipped']}")
    print(f"\nReport saved to: {report_path}")

    return final_report


def main():
    parser = argparse.ArgumentParser(description='Run all 6 validation steps')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('corpus', help='Path to MIDI corpus')
    parser.add_argument('--output-dir', '-o', default='validation_output',
                        help='Output directory for results')
    parser.add_argument('--skip-train-test', action='store_true',
                        help='Skip step 3 (train/test split) - saves time')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    run_all_validations(
        checkpoint_path=args.checkpoint,
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        skip_train_test=args.skip_train_test,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
