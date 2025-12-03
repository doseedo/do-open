#!/usr/bin/env python3
"""
Validation script for discovered transforms.
Tests the quality of transform discovery on random corpus files.
"""

import argparse
import json
import os
import sys
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mido import MidiFile
from tqdm import tqdm


def load_discovery_results(results_dir: str) -> Dict:
    """Load discovery results from output directory."""
    results_path = Path(results_dir) / "final_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    # Load meta patterns if available
    meta_path = Path(results_dir) / "meta_patterns.json"
    if meta_path.exists():
        with open(meta_path) as f:
            results['meta_patterns'] = json.load(f)

    return results


def load_midi_as_piano_roll(midi_path: str, resolution: int = 16) -> np.ndarray:
    """Load MIDI file as piano roll representation."""
    try:
        mid = MidiFile(midi_path)
    except Exception as e:
        raise ValueError(f"Failed to load MIDI: {e}")

    # Get ticks per beat
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_step = ticks_per_beat // (resolution // 4)  # 16th note resolution

    # Collect all notes
    notes = []
    for track in mid.tracks:
        current_time = 0
        active_notes = {}

        for msg in track:
            current_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = {
                    'start': current_time,
                    'velocity': msg.velocity
                }
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    note_data = active_notes.pop(msg.note)
                    notes.append({
                        'pitch': msg.note,
                        'start': note_data['start'],
                        'end': current_time,
                        'velocity': note_data['velocity']
                    })

    if not notes:
        return np.zeros((128, 100))

    # Convert to piano roll
    max_time = max(n['end'] for n in notes)
    num_steps = max(1, int(max_time / ticks_per_step) + 1)

    piano_roll = np.zeros((128, num_steps))

    for note in notes:
        start_step = int(note['start'] / ticks_per_step)
        end_step = int(note['end'] / ticks_per_step)
        piano_roll[note['pitch'], start_step:end_step] = note['velocity'] / 127.0

    return piano_roll


def apply_transform(piano_roll: np.ndarray, transform: Dict) -> np.ndarray:
    """Apply a transform to piano roll."""
    name = transform['name']
    amount = transform.get('amount', 0)

    result = piano_roll.copy()

    if name == 'transpose_semitone':
        # Shift pitch dimension
        amount = int(amount)
        if amount > 0:
            result = np.zeros_like(piano_roll)
            result[amount:128, :] = piano_roll[:128-amount, :]
        elif amount < 0:
            result = np.zeros_like(piano_roll)
            result[:128+amount, :] = piano_roll[-amount:, :]

    elif name == 'time_shift':
        # Shift time dimension
        amount = int(amount)
        if amount > 0 and amount < result.shape[1]:
            new_roll = np.zeros_like(result)
            new_roll[:, amount:] = result[:, :-amount]
            result = new_roll
        elif amount < 0 and -amount < result.shape[1]:
            new_roll = np.zeros_like(result)
            new_roll[:, :amount] = result[:, -amount:]
            result = new_roll

    elif name == 'time_scale':
        # Scale time dimension
        scale = float(amount)
        if scale > 0:
            from scipy.ndimage import zoom
            new_width = int(result.shape[1] * scale)
            if new_width > 0:
                result = zoom(result, (1.0, scale), order=0)

    elif name == 'velocity_scale':
        # Scale velocity values
        scale = float(amount)
        result = np.clip(result * scale, 0, 1)

    elif name == 'retrograde':
        # Reverse in time
        result = result[:, ::-1]

    elif name == 'inversion':
        # Invert around pivot
        pivot = int(amount)
        active_mask = result > 0
        new_result = np.zeros_like(result)
        for pitch in range(128):
            if np.any(result[pitch, :] > 0):
                new_pitch = 2 * pivot - pitch
                if 0 <= new_pitch < 128:
                    new_result[new_pitch, :] = np.maximum(new_result[new_pitch, :], result[pitch, :])
        result = new_result

    return result


def compute_similarity(roll1: np.ndarray, roll2: np.ndarray) -> float:
    """Compute similarity between two piano rolls."""
    # Ensure same shape
    min_time = min(roll1.shape[1], roll2.shape[1])
    r1 = roll1[:, :min_time]
    r2 = roll2[:, :min_time]

    # Binary similarity (Jaccard-like)
    active1 = r1 > 0.1
    active2 = r2 > 0.1

    intersection = np.sum(active1 & active2)
    union = np.sum(active1 | active2)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def find_best_transform_match(
    source: np.ndarray,
    target: np.ndarray,
    transforms: List[Dict],
    max_error: float = 0.03
) -> Tuple[Optional[Dict], float]:
    """Find best transform that maps source to target."""
    best_transform = None
    best_similarity = 0.0

    for transform in transforms:
        transformed = apply_transform(source, transform)
        similarity = compute_similarity(transformed, target)

        if similarity > best_similarity:
            best_similarity = similarity
            best_transform = transform

    # Check if similarity meets threshold
    if best_similarity >= (1.0 - max_error):
        return best_transform, best_similarity

    return None, best_similarity


def validate_on_file(
    midi_path: str,
    transforms: List[Dict],
    scales: List[int] = [16, 32, 64, 128]
) -> Dict:
    """Validate transforms on a single MIDI file."""
    results = {
        'file': os.path.basename(midi_path),
        'scales': {},
        'overall_derivation_rate': 0.0,
        'transform_coverage': {}
    }

    try:
        piano_roll = load_midi_as_piano_roll(midi_path)
    except Exception as e:
        results['error'] = str(e)
        return results

    total_objects = 0
    total_derived = 0
    transform_usage = {}

    for scale in scales:
        # Extract segments at this scale
        num_segments = piano_roll.shape[1] // scale
        if num_segments < 2:
            continue

        segments = []
        for i in range(num_segments):
            start = i * scale
            end = start + scale
            segments.append(piano_roll[:, start:end])

        # Try to derive each segment from others
        derived_count = 0
        for i, target in enumerate(segments):
            for j, source in enumerate(segments):
                if i == j:
                    continue

                match, similarity = find_best_transform_match(
                    source, target, transforms
                )

                if match:
                    derived_count += 1
                    name = match['name']
                    transform_usage[name] = transform_usage.get(name, 0) + 1
                    break

        total_objects += len(segments)
        total_derived += derived_count

        results['scales'][scale] = {
            'segments': len(segments),
            'derived': derived_count,
            'rate': derived_count / len(segments) if segments else 0
        }

    if total_objects > 0:
        results['overall_derivation_rate'] = total_derived / total_objects

    results['transform_coverage'] = transform_usage

    return results


def main():
    parser = argparse.ArgumentParser(description='Validate discovery results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing discovery results')
    parser.add_argument('--corpus-path', type=str, required=True,
                        help='Path to MIDI corpus')
    parser.add_argument('--num-files', type=int, default=10,
                        help='Number of random files to validate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for validation results')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for file selection')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TRANSFORM DISCOVERY VALIDATION")
    print("=" * 60)

    # Load discovery results
    print(f"\nLoading results from: {args.results_dir}")
    results = load_discovery_results(args.results_dir)

    transforms = results.get('final_transforms', [])
    print(f"  Transforms loaded: {len(transforms)}")

    if 'meta_patterns' in results:
        meta = results['meta_patterns']
        print(f"  Meta-patterns: {meta.get('total_patterns', 0)}")

    # Find MIDI files
    corpus_path = Path(args.corpus_path)
    midi_files = list(corpus_path.glob("**/*.mid")) + list(corpus_path.glob("**/*.midi"))
    print(f"\nFound {len(midi_files)} MIDI files in corpus")

    if not midi_files:
        print("ERROR: No MIDI files found!")
        return 1

    # Select random files
    random.seed(args.random_seed)
    num_files = min(args.num_files, len(midi_files))
    selected_files = random.sample(midi_files, num_files)

    print(f"Selected {num_files} files for validation")
    print("\n" + "-" * 60)

    # Validate each file
    validation_results = []
    scales = [16, 32, 64, 128]

    for midi_path in tqdm(selected_files, desc="Validating"):
        result = validate_on_file(str(midi_path), transforms, scales)
        validation_results.append(result)

    # Aggregate results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    total_rate = 0
    scale_rates = {s: [] for s in scales}
    transform_usage = {}
    errors = []

    for result in validation_results:
        if 'error' in result:
            errors.append(result)
            continue

        total_rate += result['overall_derivation_rate']

        for scale, data in result['scales'].items():
            if scale in scale_rates:
                scale_rates[scale].append(data['rate'])

        for name, count in result['transform_coverage'].items():
            transform_usage[name] = transform_usage.get(name, 0) + count

    valid_count = len(validation_results) - len(errors)

    print(f"\nFiles validated: {valid_count}/{num_files}")
    if errors:
        print(f"Files with errors: {len(errors)}")

    if valid_count > 0:
        avg_rate = total_rate / valid_count
        print(f"\nOverall Derivation Rate: {avg_rate:.2%}")

        print("\nDerivation Rate by Scale:")
        for scale in sorted(scale_rates.keys()):
            rates = scale_rates[scale]
            if rates:
                avg = sum(rates) / len(rates)
                print(f"  Scale {scale:4d}: {avg:.2%} (n={len(rates)})")

        print("\nTransform Usage:")
        sorted_usage = sorted(transform_usage.items(), key=lambda x: -x[1])
        for name, count in sorted_usage[:10]:
            print(f"  {name}: {count}")

    # Quality metrics
    print("\n" + "-" * 60)
    print("QUALITY METRICS")
    print("-" * 60)

    quality_score = avg_rate if valid_count > 0 else 0

    if quality_score >= 0.8:
        quality_label = "EXCELLENT"
    elif quality_score >= 0.6:
        quality_label = "GOOD"
    elif quality_score >= 0.4:
        quality_label = "FAIR"
    else:
        quality_label = "NEEDS IMPROVEMENT"

    print(f"\nQuality Score: {quality_score:.2%} ({quality_label})")
    print(f"Transform Library Size: {len(transforms)}")
    print(f"Active Transforms: {len(transform_usage)}")

    # Save results
    output_data = {
        'results_dir': args.results_dir,
        'corpus_path': args.corpus_path,
        'num_files_validated': valid_count,
        'overall_derivation_rate': avg_rate if valid_count > 0 else 0,
        'quality_score': quality_score,
        'quality_label': quality_label,
        'scale_rates': {k: sum(v)/len(v) if v else 0 for k, v in scale_rates.items()},
        'transform_usage': transform_usage,
        'file_results': validation_results,
        'transforms': transforms
    }

    if args.output:
        output_path = args.output
    else:
        output_path = str(Path(args.results_dir) / "validation_results.json")

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
