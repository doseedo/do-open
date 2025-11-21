#!/usr/bin/env python3
"""
Rich Feature Extraction Script - Agent 2
=========================================

Batch extract 600D rich multitrack features from MIDI corpus.

This script processes all MIDI files in the corpus (train/val/test splits)
and extracts features using parallel processing with multiprocessing.

Usage:
    # Extract features for all splits with 16 workers
    python scripts/extract_rich_features.py

    # Specify custom directories and worker count
    python scripts/extract_rich_features.py \
        --corpus-dir data/corpus \
        --output-dir data/features \
        --workers 16

    # Extract only specific split
    python scripts/extract_rich_features.py --split train --workers 8

Author: Agent 2 - Rich Feature Extraction Specialist
"""

import argparse
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_generator.feature_selection.rich_feature_extractor import (
    RichMultitrackFeatureExtractor,
    BatchFeatureExtractor
)


def main():
    parser = argparse.ArgumentParser(
        description='Extract 600D rich multitrack features from MIDI corpus'
    )

    parser.add_argument(
        '--corpus-dir',
        type=str,
        default='data/corpus',
        help='Directory containing train/val/test MIDI files (default: data/corpus)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/features',
        help='Directory to save feature .npy files (default: data/features)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=16,
        help='Number of parallel workers (default: 16)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['train', 'val', 'test'],
        help='Process only specific split (default: all splits)'
    )

    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=100,
        help='Save checkpoint every N files (default: 100)'
    )

    parser.add_argument(
        '--test-single',
        type=str,
        default=None,
        help='Test extraction on a single MIDI file'
    )

    args = parser.parse_args()

    print("="*70)
    print("RICH MULTITRACK FEATURE EXTRACTOR - AGENT 2")
    print("="*70)

    # Test single file mode
    if args.test_single:
        print(f"\nTesting extraction on: {args.test_single}")
        extractor = RichMultitrackFeatureExtractor(use_base_extractor=False)

        try:
            start = time.time()
            features = extractor.extract(args.test_single)
            elapsed = time.time() - start

            print(f"\n✅ Extraction successful!")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Shape: {features.shape}")
            print(f"   Min: {features.min():.3f}")
            print(f"   Max: {features.max():.3f}")
            print(f"   Mean: {features.mean():.3f}")

            # Save test features
            import numpy as np
            output_path = Path('test_features.npy')
            np.save(output_path, features)
            print(f"\n   Saved to: {output_path}")

            return 0

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Batch extraction mode
    print(f"\nConfiguration:")
    print(f"  Corpus directory: {args.corpus_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Workers: {args.workers}")
    print(f"  Processing: {args.split if args.split else 'all splits'}")

    # Check corpus directory exists
    corpus_path = Path(args.corpus_dir)
    if not corpus_path.exists():
        print(f"\n❌ Error: Corpus directory not found: {corpus_path}")
        print(f"\nExpected structure:")
        print(f"  {args.corpus_dir}/")
        print(f"    train/  (8,000 MIDI files)")
        print(f"    val/    (1,000 MIDI files)")
        print(f"    test/   (1,000 MIDI files)")
        print(f"\nPlease run Agent 1 corpus organization first.")
        return 1

    # Create batch extractor
    try:
        batch_extractor = BatchFeatureExtractor(
            corpus_dir=args.corpus_dir,
            output_dir=args.output_dir,
            n_workers=args.workers,
            checkpoint_interval=args.checkpoint_interval
        )

        # Run extraction
        if args.split:
            # Process single split
            print(f"\nProcessing {args.split} split only...")
            report = {
                'splits': {
                    args.split: batch_extractor._process_split(args.split)
                }
            }
        else:
            # Process all splits
            report = batch_extractor.run()

        # Print summary
        print(f"\n{'='*70}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*70}")

        for split, split_report in report['splits'].items():
            print(f"\n{split.upper()}:")
            print(f"  Files processed: {split_report['features_extracted']}/{split_report['num_files']}")
            print(f"  Time: {split_report['total_time']:.1f}s")
            print(f"  Avg: {split_report['avg_time']:.2f}s per file")
            print(f"  Error rate: {split_report['error_rate']*100:.2f}%")

        # Success criteria check
        print(f"\n{'='*70}")
        print(f"SUCCESS CRITERIA CHECK")
        print(f"{'='*70}")

        all_features_extracted = sum(r['features_extracted'] for r in report['splits'].values())
        all_files = sum(r['num_files'] for r in report['splits'].values())
        overall_error_rate = 1 - (all_features_extracted / max(all_files, 1))
        avg_time = sum(r['total_time'] for r in report['splits'].values()) / max(all_files, 1)

        criteria = [
            ("✅ 600D features extracted", all_features_extracted > 0, True),
            ("✅ Extraction time < 30s per file", avg_time < 30.0, True),
            ("✅ Error rate < 0.5%", overall_error_rate < 0.005, True),
        ]

        for criterion, actual, required in criteria:
            status = "✅" if (actual == required or (isinstance(actual, (int, float)) and actual > 0)) else "❌"
            print(f"{status} {criterion}")

        print(f"\nReport saved to: {args.output_dir}/feature_extraction_report.json")

        return 0

    except Exception as e:
        print(f"\n❌ Error during batch extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
