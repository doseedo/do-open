#!/usr/bin/env python3
"""
Run YAMNet audio classification on vocal training manifest.
Creates a labeled manifest for reviewing audio quality and content.
"""

import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from typing import Dict, List, Tuple
import librosa
import soundfile as sf
from tqdm import tqdm
import traceback

# YAMNet expects 16kHz mono audio
YAMNET_SR = 16000

def load_yamnet_model():
    """Load YAMNet model from TensorFlow Hub."""
    print("Loading YAMNet model...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    print("✅ YAMNet model loaded")
    return model

def load_yamnet_class_names():
    """Load YAMNet class names."""
    class_map_path = tf.keras.utils.get_file(
        'yamnet_class_map.csv',
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    )

    class_names = []
    with open(class_map_path) as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                class_names.append(parts[2])  # Display name is in column 2

    return class_names

def load_audio_for_yamnet(audio_path: str, max_duration: float = 30.0) -> np.ndarray:
    """
    Load audio file and prepare for YAMNet.

    Args:
        audio_path: Path to audio file
        max_duration: Maximum duration to process (seconds)

    Returns:
        Audio waveform resampled to 16kHz, limited to max_duration
    """
    try:
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=None, mono=True)

        # Limit duration to avoid memory issues
        max_samples = int(max_duration * sr)
        if len(waveform) > max_samples:
            # Take from middle section (often has the most content)
            start = (len(waveform) - max_samples) // 2
            waveform = waveform[start:start + max_samples]

        # Resample to 16kHz if needed
        if sr != YAMNET_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=YAMNET_SR)

        return waveform

    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

def analyze_audio_with_yamnet(model, waveform: np.ndarray, class_names: List[str],
                               top_k: int = 10) -> Dict:
    """
    Run YAMNet on audio and get top predictions.

    Args:
        model: YAMNet model
        waveform: Audio waveform at 16kHz
        class_names: List of YAMNet class names
        top_k: Number of top predictions to return

    Returns:
        Dict with predictions and scores
    """
    try:
        # Run YAMNet
        scores, embeddings, spectrogram = model(waveform)

        # Average scores across time
        mean_scores = np.mean(scores.numpy(), axis=0)

        # Get top K predictions
        top_indices = np.argsort(mean_scores)[-top_k:][::-1]

        predictions = []
        for idx in top_indices:
            predictions.append({
                'class': class_names[idx],
                'score': float(mean_scores[idx]),
                'percentage': float(mean_scores[idx] * 100)
            })

        # Get max score per frame for confidence analysis
        max_scores_per_frame = np.max(scores.numpy(), axis=1)

        return {
            'predictions': predictions,
            'top_class': class_names[top_indices[0]],
            'top_score': float(mean_scores[top_indices[0]]),
            'mean_confidence': float(np.mean(max_scores_per_frame)),
            'min_confidence': float(np.min(max_scores_per_frame)),
            'num_frames': int(scores.shape[0])
        }

    except Exception as e:
        print(f"Error in YAMNet analysis: {e}")
        traceback.print_exc()
        return None

def check_for_issues(predictions: List[Dict], top_class: str) -> List[str]:
    """
    Check for potential issues in the audio based on YAMNet predictions.

    Returns:
        List of warning messages
    """
    warnings = []

    # Get all predicted classes with scores > 0.1
    significant_classes = [p['class'] for p in predictions if p['score'] > 0.1]

    # Check for non-vocal content
    non_vocal_indicators = [
        'Music', 'Musical instrument', 'Plucked string instrument',
        'Guitar', 'Piano', 'Drum', 'Percussion', 'Bass guitar',
        'Synthesizer', 'Electronic music', 'Ambient music'
    ]

    for indicator in non_vocal_indicators:
        if any(indicator.lower() in c.lower() for c in significant_classes):
            warnings.append(f"Contains {indicator.lower()}")

    # Check for noise/interference
    noise_indicators = [
        'Noise', 'Static', 'Hum', 'White noise', 'Pink noise',
        'Crackle', 'Click', 'Pop', 'Distortion'
    ]

    for indicator in noise_indicators:
        if any(indicator.lower() in c.lower() for c in significant_classes):
            warnings.append(f"Contains {indicator.lower()}")

    # Check for speech/singing
    vocal_indicators = [
        'Speech', 'Singing', 'Voice', 'Vocal', 'Female singing',
        'Male singing', 'Child singing', 'Choir', 'Chant', 'Yodeling'
    ]

    has_vocal = any(any(vi.lower() in c.lower() for vi in vocal_indicators)
                    for c in significant_classes[:3])  # Check top 3

    if not has_vocal:
        warnings.append("No clear vocal content detected")

    # Check for environmental sounds
    env_indicators = [
        'Wind', 'Rain', 'Thunder', 'Water', 'Bird', 'Dog', 'Cat',
        'Traffic', 'Car', 'Engine', 'Airplane', 'Door', 'Footsteps'
    ]

    for indicator in env_indicators:
        if any(indicator.lower() in c.lower() for c in significant_classes):
            warnings.append(f"Environmental sound: {indicator.lower()}")

    return warnings

def process_manifest_with_yamnet(input_manifest_path: str, output_manifest_path: str,
                                  max_files: int = None, skip_existing: bool = True):
    """
    Process manifest with YAMNet and create labeled version.

    Args:
        input_manifest_path: Path to input manifest
        output_manifest_path: Path to output labeled manifest
        max_files: Maximum number of files to process (None = all)
        skip_existing: Skip processing if output exists and has yamnet_labels
    """

    # Load input manifest
    print(f"Loading manifest from: {input_manifest_path}")
    with open(input_manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"Total entries: {len(manifest)}")

    # Check if we should resume from existing output
    processed_indices = set()
    if skip_existing and Path(output_manifest_path).exists():
        print(f"Found existing output, checking for processed entries...")
        with open(output_manifest_path, 'r') as f:
            existing = json.load(f)

        for idx, entry in enumerate(existing):
            if 'yamnet_labels' in entry and entry['yamnet_labels']:
                processed_indices.add(idx)

        print(f"Found {len(processed_indices)} already processed entries")
        manifest = existing  # Start from existing to preserve labels

    # Load YAMNet
    model = load_yamnet_model()
    class_names = load_yamnet_class_names()
    print(f"Loaded {len(class_names)} YAMNet classes")

    # Process files
    if max_files:
        manifest = manifest[:max_files]

    failed_count = 0
    warning_count = 0

    print(f"\nProcessing {len(manifest)} entries...")

    for idx, entry in enumerate(tqdm(manifest, desc="Labeling audio")):
        # Skip if already processed
        if idx in processed_indices:
            continue

        audio_path = entry.get('audio_path', '')

        if not audio_path or not Path(audio_path).exists():
            entry['yamnet_labels'] = {
                'status': 'error',
                'error': 'File not found'
            }
            failed_count += 1
            continue

        # Load audio
        waveform = load_audio_for_yamnet(audio_path, max_duration=30.0)

        if waveform is None:
            entry['yamnet_labels'] = {
                'status': 'error',
                'error': 'Failed to load audio'
            }
            failed_count += 1
            continue

        # Run YAMNet
        result = analyze_audio_with_yamnet(model, waveform, class_names, top_k=10)

        if result is None:
            entry['yamnet_labels'] = {
                'status': 'error',
                'error': 'YAMNet processing failed'
            }
            failed_count += 1
            continue

        # Check for issues
        warnings = check_for_issues(result['predictions'], result['top_class'])
        if warnings:
            warning_count += 1

        # Add to manifest
        entry['yamnet_labels'] = {
            'status': 'success',
            'top_predictions': result['predictions'][:5],  # Top 5
            'all_predictions': result['predictions'],      # All top 10
            'top_class': result['top_class'],
            'top_score': result['top_score'],
            'confidence_stats': {
                'mean': result['mean_confidence'],
                'min': result['min_confidence']
            },
            'warnings': warnings,
            'num_frames': result['num_frames']
        }

        # Save periodically
        if (idx + 1) % 100 == 0:
            with open(output_manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(f"\nSaved checkpoint at {idx + 1} entries")

    # Final save
    with open(output_manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("YAMNet Labeling Complete!")
    print("=" * 70)
    print(f"Total processed: {len(manifest)}")
    print(f"Failed: {failed_count}")
    print(f"Entries with warnings: {warning_count}")
    print(f"\nOutput saved to: {output_manifest_path}")

    # Show common warnings
    all_warnings = []
    for entry in manifest:
        labels = entry.get('yamnet_labels', {})
        if labels.get('warnings'):
            all_warnings.extend(labels['warnings'])

    if all_warnings:
        from collections import Counter
        warning_counts = Counter(all_warnings)
        print("\nMost common warnings:")
        for warning, count in warning_counts.most_common(10):
            print(f"  - {warning}: {count} files")

def create_review_report(labeled_manifest_path: str, report_path: str):
    """Create a human-readable review report."""

    with open(labeled_manifest_path, 'r') as f:
        manifest = json.load(f)

    lines = []
    lines.append("YAMNet Audio Review Report")
    lines.append("=" * 80)
    lines.append("")

    # Summary
    total = len(manifest)
    with_warnings = sum(1 for e in manifest if e.get('yamnet_labels', {}).get('warnings'))
    failed = sum(1 for e in manifest if e.get('yamnet_labels', {}).get('status') == 'error')

    lines.append(f"Total entries: {total}")
    lines.append(f"Entries with warnings: {with_warnings} ({100*with_warnings/total:.1f}%)")
    lines.append(f"Failed: {failed}")
    lines.append("")

    # Entries with warnings
    lines.append("=" * 80)
    lines.append("ENTRIES WITH WARNINGS (Review these first)")
    lines.append("=" * 80)
    lines.append("")

    for idx, entry in enumerate(manifest):
        labels = entry.get('yamnet_labels', {})
        warnings = labels.get('warnings', [])

        if warnings:
            audio_path = entry.get('audio_path', '')
            top_class = labels.get('top_class', 'Unknown')
            top_score = labels.get('top_score', 0)

            lines.append(f"[{idx}] {Path(audio_path).name}")
            lines.append(f"    Path: {audio_path}")
            lines.append(f"    Top prediction: {top_class} ({top_score*100:.1f}%)")
            lines.append(f"    Warnings: {', '.join(warnings)}")

            # Show top predictions
            top_preds = labels.get('top_predictions', [])
            if top_preds:
                pred_str = ', '.join([f"{p['class']} ({p['percentage']:.1f}%)"
                                     for p in top_preds[:3]])
                lines.append(f"    Top 3: {pred_str}")
            lines.append("")

    # Write report
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n📋 Review report saved to: {report_path}")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run YAMNet labeling on vocal training manifest")
    ap.add_argument("--input_manifest", type=str,
                    default="./vocal_training_manifest_with_alternates.json",
                    help="Input manifest path")
    ap.add_argument("--output_manifest", type=str,
                    default="./vocal_training_manifest_yamnet_labeled.json",
                    help="Output labeled manifest path")
    ap.add_argument("--max_files", type=int, default=None,
                    help="Maximum number of files to process (for testing)")
    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="Skip already processed entries")
    ap.add_argument("--create_report", action="store_true", default=True,
                    help="Create human-readable review report")

    args = ap.parse_args()

    # Process manifest
    process_manifest_with_yamnet(
        input_manifest_path=args.input_manifest,
        output_manifest_path=args.output_manifest,
        max_files=args.max_files,
        skip_existing=args.skip_existing
    )

    # Create review report
    if args.create_report:
        report_path = args.output_manifest.replace('.json', '_review.txt')
        create_review_report(args.output_manifest, report_path)
