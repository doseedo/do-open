"""
Complete Corpus Extraction Script
==================================

Extracts BOTH features (200D) and parameters (50) from entire MIDI corpus.

Usage:
    python scripts/extract_corpus_complete.py \
        --corpus midi_corpus/big_band/ \
        --output labeled_dataset_complete.json
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_generator.parameters.hierarchical_extractor_v2 import HierarchicalParameterExtractorV2


def extract_corpus(corpus_dir: Path, output_file: Path, verbose: bool = False):
    """
    Extract complete dataset from MIDI corpus.

    Args:
        corpus_dir: Directory containing MIDI files
        output_file: Output JSON file path
        verbose: Print detailed extraction info
    """
    print(f"\n{'='*80}")
    print(f"COMPLETE CORPUS EXTRACTION")
    print(f"{'='*80}")
    print(f"Corpus: {corpus_dir}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    # Initialize extractor
    print("Initializing extractor...")
    extractor = HierarchicalParameterExtractorV2(verbose=verbose)

    # Find all MIDI files
    midi_files = list(corpus_dir.glob("**/*.mid")) + list(corpus_dir.glob("**/*.MID"))
    print(f"Found {len(midi_files)} MIDI files\n")

    if len(midi_files) == 0:
        print(f"❌ No MIDI files found in {corpus_dir}")
        return

    # Extract from each file
    labeled_data = []
    errors = []

    for midi_file in tqdm(midi_files, desc="Extracting"):
        try:
            # Extract features + parameters
            extraction = extractor.extract_complete(str(midi_file))

            # Verify format
            assert 'features' in extraction, "Missing 'features' key"
            assert 'parameters' in extraction, "Missing 'parameters' key"
            assert len(extraction['features']) == 200, f"Expected 200 features, got {len(extraction['features'])}"

            # Verify parameter counts
            params = extraction['parameters']
            level1_count = len(params['level1_global'])
            level2_count = sum(len(v) for v in params['level2_universal'].values())
            level3_count = sum(len(v) for v in params['level3_genre_specific'].values())

            assert level1_count == 8, f"Level 1: expected 8, got {level1_count}"
            assert level2_count == 20, f"Level 2: expected 20, got {level2_count}"
            assert level3_count == 22, f"Level 3: expected 22, got {level3_count}"

            # Add to dataset
            labeled_data.append({
                'file_id': midi_file.stem,
                'file_path': str(midi_file),
                'features': extraction['features'],
                'parameters': extraction['parameters'],
                'metadata': extraction['metadata']
            })

        except Exception as e:
            errors.append((midi_file.name, str(e)))
            if verbose:
                print(f"\n❌ Error processing {midi_file.name}: {e}")

    # Save dataset
    print(f"\n\nSaving dataset to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(labeled_data, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Successfully extracted: {len(labeled_data)} files")
    print(f"❌ Errors: {len(errors)} files")

    if errors:
        print(f"\nErrors:")
        for file, error in errors[:10]:  # Show first 10 errors
            print(f"  - {file}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Verify first sample
    if labeled_data:
        sample = labeled_data[0]
        print(f"\n✅ Sample Verification:")
        print(f"   File: {sample['file_id']}")
        print(f"   Features: {len(sample['features'])}D")
        params_total = (
            len(sample['parameters']['level1_global']) +
            sum(len(v) for v in sample['parameters']['level2_universal'].values()) +
            sum(len(v) for v in sample['parameters']['level3_genre_specific'].values())
        )
        print(f"   Parameters: {params_total} total")
        print(f"   Genre: {sample['parameters']['level1_global'].get('genre.primary', 'unknown')}")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract complete dataset from MIDI corpus')
    parser.add_argument('--corpus', type=str, required=True, help='Path to MIDI corpus directory')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--verbose', action='store_true', help='Print detailed extraction info')

    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    output_file = Path(args.output)

    if not corpus_dir.exists():
        print(f"❌ Corpus directory not found: {corpus_dir}")
        sys.exit(1)

    extract_corpus(corpus_dir, output_file, args.verbose)
