#!/usr/bin/env python3
"""
Filter manifest based on YAMNet labels.
Remove entries with specific issues or warnings.
"""

import json
from pathlib import Path
from typing import List, Set
from collections import Counter

def analyze_yamnet_manifest(manifest_path: str):
    """Analyze YAMNet labeled manifest and show statistics."""

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"Analyzing: {manifest_path}")
    print("=" * 70)

    total = len(manifest)
    processed = sum(1 for e in manifest if e.get('yamnet_labels', {}).get('status') == 'success')
    failed = sum(1 for e in manifest if e.get('yamnet_labels', {}).get('status') == 'error')

    print(f"\nTotal entries: {total}")
    print(f"Successfully labeled: {processed}")
    print(f"Failed: {failed}")

    # Count warnings
    all_warnings = []
    entries_with_warnings = 0

    for entry in manifest:
        labels = entry.get('yamnet_labels', {})
        warnings = labels.get('warnings', [])
        if warnings:
            entries_with_warnings += 1
            all_warnings.extend(warnings)

    print(f"\nEntries with warnings: {entries_with_warnings} ({100*entries_with_warnings/total:.1f}%)")

    if all_warnings:
        warning_counts = Counter(all_warnings)
        print("\nTop warnings:")
        for warning, count in warning_counts.most_common(15):
            print(f"  - {warning}: {count} files ({100*count/total:.1f}%)")

    # Top classes
    top_classes = []
    for entry in manifest:
        labels = entry.get('yamnet_labels', {})
        if labels.get('top_class'):
            top_classes.append(labels['top_class'])

    if top_classes:
        class_counts = Counter(top_classes)
        print("\nTop predicted classes:")
        for cls, count in class_counts.most_common(10):
            print(f"  - {cls}: {count} files ({100*count/total:.1f}%)")

    # Confidence stats
    confidences = [e.get('yamnet_labels', {}).get('top_score', 0)
                  for e in manifest if e.get('yamnet_labels', {}).get('top_score')]

    if confidences:
        import numpy as np
        print(f"\nConfidence statistics:")
        print(f"  - Mean: {np.mean(confidences):.3f}")
        print(f"  - Median: {np.median(confidences):.3f}")
        print(f"  - Min: {np.min(confidences):.3f}")
        print(f"  - Max: {np.max(confidences):.3f}")

def filter_manifest(input_path: str, output_path: str,
                   exclude_warnings: List[str] = None,
                   exclude_classes: List[str] = None,
                   min_confidence: float = None,
                   require_vocal: bool = True):
    """
    Filter manifest based on YAMNet labels.

    Args:
        input_path: Input manifest with YAMNet labels
        output_path: Output filtered manifest
        exclude_warnings: List of warning keywords to filter out
        exclude_classes: List of top class keywords to filter out
        min_confidence: Minimum top score confidence
        require_vocal: Require vocal content in top predictions
    """

    with open(input_path, 'r') as f:
        manifest = json.load(f)

    print(f"Filtering manifest: {input_path}")
    print("=" * 70)

    original_count = len(manifest)
    filtered = []
    removal_reasons = Counter()

    vocal_keywords = [
        'speech', 'singing', 'voice', 'vocal', 'female singing',
        'male singing', 'child singing', 'choir', 'chant', 'yodel',
        'male speech', 'female speech', 'narration', 'conversation'
    ]

    for entry in manifest:
        labels = entry.get('yamnet_labels', {})

        # Skip failed entries
        if labels.get('status') != 'success':
            removal_reasons['Failed YAMNet processing'] += 1
            continue

        # Check confidence
        if min_confidence is not None:
            top_score = labels.get('top_score', 0)
            if top_score < min_confidence:
                removal_reasons[f'Low confidence (<{min_confidence})'] += 1
                continue

        # Check for excluded warnings
        if exclude_warnings:
            warnings = labels.get('warnings', [])
            has_excluded_warning = False

            for exclude in exclude_warnings:
                if any(exclude.lower() in w.lower() for w in warnings):
                    removal_reasons[f'Warning: {exclude}'] += 1
                    has_excluded_warning = True
                    break

            if has_excluded_warning:
                continue

        # Check for excluded classes
        if exclude_classes:
            top_class = labels.get('top_class', '').lower()
            has_excluded_class = False

            for exclude in exclude_classes:
                if exclude.lower() in top_class:
                    removal_reasons[f'Class: {exclude}'] += 1
                    has_excluded_class = True
                    break

            if has_excluded_class:
                continue

        # Check for vocal content
        if require_vocal:
            top_preds = labels.get('top_predictions', [])
            has_vocal = False

            for pred in top_preds[:5]:  # Check top 5
                pred_class = pred.get('class', '').lower()
                if any(vk in pred_class for vk in vocal_keywords):
                    has_vocal = True
                    break

            if not has_vocal:
                removal_reasons['No vocal content'] += 1
                continue

        # Passed all filters
        filtered.append(entry)

    # Save filtered manifest
    with open(output_path, 'w') as f:
        json.dump(filtered, f, indent=2)

    # Print summary
    removed = original_count - len(filtered)
    print(f"\nFiltering complete:")
    print(f"  Original entries: {original_count}")
    print(f"  Filtered entries: {len(filtered)}")
    print(f"  Removed: {removed} ({100*removed/original_count:.1f}%)")

    if removal_reasons:
        print("\nRemoval reasons:")
        for reason, count in removal_reasons.most_common():
            print(f"  - {reason}: {count}")

    print(f"\nFiltered manifest saved to: {output_path}")

def interactive_filter():
    """Interactive filtering with prompts."""

    print("YAMNet Manifest Filtering Tool")
    print("=" * 70)

    input_path = input("\nInput manifest path [./vocal_training_manifest_yamnet_labeled.json]: ").strip()
    if not input_path:
        input_path = "./vocal_training_manifest_yamnet_labeled.json"

    if not Path(input_path).exists():
        print(f"Error: {input_path} not found")
        return

    # Analyze first
    print("\nAnalyzing manifest...")
    analyze_yamnet_manifest(input_path)

    # Get filter criteria
    print("\n" + "=" * 70)
    print("Filter Criteria")
    print("=" * 70)

    output_path = input("\nOutput manifest path [./vocal_training_manifest_filtered_clean.json]: ").strip()
    if not output_path:
        output_path = "./vocal_training_manifest_filtered_clean.json"

    exclude_warnings = []
    print("\nExclude entries with warnings containing these keywords (comma-separated):")
    print("  Examples: 'music,guitar,piano,noise,static'")
    exclude_input = input("  Keywords [none]: ").strip()
    if exclude_input:
        exclude_warnings = [w.strip() for w in exclude_input.split(',')]

    exclude_classes = []
    print("\nExclude entries where top class contains these keywords (comma-separated):")
    print("  Examples: 'music,noise,silence'")
    exclude_input = input("  Keywords [none]: ").strip()
    if exclude_input:
        exclude_classes = [w.strip() for w in exclude_input.split(',')]

    min_conf_input = input("\nMinimum confidence threshold (0.0-1.0) [0.3]: ").strip()
    min_confidence = float(min_conf_input) if min_conf_input else 0.3

    require_vocal_input = input("\nRequire vocal content in top 5 predictions? (y/n) [y]: ").strip().lower()
    require_vocal = require_vocal_input != 'n'

    # Run filtering
    print("\nFiltering...")
    filter_manifest(
        input_path=input_path,
        output_path=output_path,
        exclude_warnings=exclude_warnings if exclude_warnings else None,
        exclude_classes=exclude_classes if exclude_classes else None,
        min_confidence=min_confidence,
        require_vocal=require_vocal
    )

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Filter manifest by YAMNet labels")
    ap.add_argument("--input", type=str,
                    default="./vocal_training_manifest_yamnet_labeled.json",
                    help="Input labeled manifest")
    ap.add_argument("--output", type=str,
                    default="./vocal_training_manifest_filtered_clean.json",
                    help="Output filtered manifest")
    ap.add_argument("--analyze_only", action="store_true",
                    help="Only analyze, don't filter")
    ap.add_argument("--interactive", action="store_true",
                    help="Interactive mode with prompts")
    ap.add_argument("--exclude_warnings", type=str,
                    help="Comma-separated warning keywords to exclude")
    ap.add_argument("--exclude_classes", type=str,
                    help="Comma-separated class keywords to exclude")
    ap.add_argument("--min_confidence", type=float, default=0.3,
                    help="Minimum confidence threshold")
    ap.add_argument("--require_vocal", action="store_true", default=True,
                    help="Require vocal content")

    args = ap.parse_args()

    if args.interactive:
        interactive_filter()
    elif args.analyze_only:
        analyze_yamnet_manifest(args.input)
    else:
        exclude_warnings = args.exclude_warnings.split(',') if args.exclude_warnings else None
        exclude_classes = args.exclude_classes.split(',') if args.exclude_classes else None

        filter_manifest(
            input_path=args.input,
            output_path=args.output,
            exclude_warnings=exclude_warnings,
            exclude_classes=exclude_classes,
            min_confidence=args.min_confidence,
            require_vocal=args.require_vocal
        )
