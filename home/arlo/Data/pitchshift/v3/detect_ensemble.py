#!/usr/bin/env python3
"""
Detect ensemble/polyphonic recordings vs clean solo trumpet recordings.

Indicators of room ensemble recordings:
1. F0 values outside trumpet range (MIDI < 52 or > 86)
2. High pitch variance (f0 jumping due to multiple instruments)
3. High amplitude during unvoiced frames (bleed from other instruments)
4. Very low voiced ratio (f0 extractor confused by polyphony)

Uses muted trumpet recordings as ground truth for clean solo characteristics.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def fix_path(path: str) -> str:
    """Fix mount paths."""
    if not path:
        return path
    replacements = [
        ('/mnt/msdd/', '/mnt/msdd2/'),
        ('/home/arlo/gcs-bucket/', '/mnt/gcs-bucket/'),
    ]
    for old, new in replacements:
        if old in path:
            path = path.replace(old, new)
    return path


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to MIDI, handling zeros."""
    hz = np.asarray(hz)
    result = np.zeros_like(hz, dtype=float)
    valid = hz > 0
    if valid.any():
        result[valid] = 69 + 12 * np.log2(hz[valid] / 440.0)
    return result


def analyze_recording(f0_path: str, amp_path: str) -> Dict:
    """
    Analyze a recording's conditioning data for ensemble indicators.

    Returns dict with metrics and flags.

    Key insight: MIDI 36 is the f0 extractor's failure floor, not real pitches.
    Clean solo recordings can have f0 extraction failures too.
    The best indicator of ensemble/bleed is high amplitude during unvoiced frames.
    """
    f0_path = fix_path(f0_path)
    amp_path = fix_path(amp_path)

    if not os.path.exists(f0_path) or not os.path.exists(amp_path):
        return {'valid': False, 'error': 'files_not_found'}

    try:
        f0 = np.load(f0_path)
        amp = np.load(amp_path)
    except Exception as e:
        return {'valid': False, 'error': str(e)}

    # Basic stats
    valid_f0 = f0[f0 > 0]
    voiced_mask = f0 > 0
    voiced_ratio = voiced_mask.mean()

    if len(valid_f0) < 10:
        return {
            'valid': True,
            'voiced_ratio': voiced_ratio,
            'flags': ['very_low_voiced'],
            'score': 50,  # Suspicious but not definitive
        }

    midi = hz_to_midi(valid_f0)

    # Filter out f0 extractor floor values (MIDI < 40 is likely extraction failure)
    # The extractor often outputs ~36 when it fails
    F0_FLOOR = 40
    real_midi = midi[midi >= F0_FLOOR]

    if len(real_midi) < 10:
        # Most pitches are at the floor - extraction failed
        return {
            'valid': True,
            'voiced_ratio': voiced_ratio,
            'midi_min': float(midi.min()),
            'midi_max': float(midi.max()),
            'flags': ['f0_extraction_failed'],
            'score': 30,
        }

    # Metrics using filtered MIDI (excluding floor values)
    metrics = {
        'valid': True,
        'voiced_ratio': voiced_ratio,
        'midi_min': float(real_midi.min()),
        'midi_max': float(real_midi.max()),
        'midi_mean': float(real_midi.mean()),
        'midi_std': float(real_midi.std()),
        'midi_raw_min': float(midi.min()),  # Include raw for debugging
        'midi_raw_max': float(midi.max()),
        'f0_floor_ratio': float((midi < F0_FLOOR).mean()),  # How much is at floor
    }

    # Trumpet range: roughly MIDI 52 (E3) to 86 (D6)
    # But allow some flexibility - pros can go higher
    TRUMPET_MIN = 50
    TRUMPET_MAX = 96  # High C and above is rare but possible

    # Check for out-of-range pitches (using filtered MIDI)
    below_range = (real_midi < TRUMPET_MIN).mean()
    above_range = (real_midi > TRUMPET_MAX).mean()
    metrics['below_range_ratio'] = float(below_range)
    metrics['above_range_ratio'] = float(above_range)

    # Pitch variance in windows (instability indicator)
    # Use filtered MIDI for more accurate variance
    window_size = 10
    hop = 5
    variances = []
    for i in range(0, len(real_midi) - window_size, hop):
        window = real_midi[i:i+window_size]
        variances.append(np.var(window))

    metrics['mean_pitch_variance'] = float(np.mean(variances)) if variances else 0
    metrics['max_pitch_variance'] = float(np.max(variances)) if variances else 0

    # Amplitude during unvoiced frames (bleed indicator) - THIS IS KEY
    unvoiced_mask = ~voiced_mask
    if unvoiced_mask.any() and len(amp) == len(f0):
        unvoiced_amp = amp[unvoiced_mask]
        metrics['unvoiced_amp_mean'] = float(unvoiced_amp.mean())
        metrics['unvoiced_amp_max'] = float(unvoiced_amp.max())
        # High unvoiced amplitude relative to overall is suspicious
        metrics['unvoiced_amp_ratio'] = float(unvoiced_amp.mean() / (amp.mean() + 1e-6))
    else:
        metrics['unvoiced_amp_mean'] = 0
        metrics['unvoiced_amp_max'] = 0
        metrics['unvoiced_amp_ratio'] = 0

    # Flag detection - recalibrated based on verified clean recordings
    flags = []
    score = 0

    # Flag 1: High unvoiced amplitude (bleed) - PRIMARY INDICATOR
    # Clean recordings have unvoiced_amp_ratio < 0.05
    # Ensemble recordings have higher background noise
    if metrics['unvoiced_amp_ratio'] > 0.15:
        flags.append('high_unvoiced_amplitude')
        score += 40
    elif metrics['unvoiced_amp_ratio'] > 0.08:
        flags.append('moderate_unvoiced_amplitude')
        score += 20

    # Flag 2: Out of range pitches (using filtered values)
    # Only flag if significant portion is out of range
    if below_range > 0.2:  # More than 20% below trumpet range
        flags.append('pitches_below_range')
        score += 25
    if above_range > 0.1:  # More than 10% above (rare for trumpet)
        flags.append('pitches_above_range')
        score += 15

    # Flag 3: Very high pitch variance - but raise threshold
    # Clean trumpet can have variance up to 50 during runs/phrases
    if metrics['mean_pitch_variance'] > 80:
        flags.append('high_pitch_variance')
        score += 20

    # Flag 4: Low voiced ratio with high amplitude
    if voiced_ratio < 0.2 and amp.mean() > 0.1:
        flags.append('low_voiced_high_amp')
        score += 25

    # Flag 5: Very wide pitch range (but use filtered range)
    pitch_range = metrics['midi_max'] - metrics['midi_min']
    if pitch_range > 36:  # More than 3 octaves is suspicious
        flags.append('very_wide_pitch_range')
        score += 10

    metrics['flags'] = flags
    metrics['score'] = score

    return metrics


def analyze_segments_json(segments_json: str, output_path: str = None) -> Dict:
    """
    Analyze all recordings in the V3 segments JSON.
    """
    print(f"Loading segments: {segments_json}")
    with open(segments_json) as f:
        data = json.load(f)

    # Collect unique recordings (by f0 path)
    recordings = {}
    for group_id, segments in data['segments_by_group'].items():
        for seg in segments:
            f0_path = seg['f0_path']
            if f0_path not in recordings:
                # Find amp path (same directory, same base name)
                amp_path = f0_path.replace('.f0.npy', '.amp.npy')
                recordings[f0_path] = {
                    'f0_path': f0_path,
                    'amp_path': amp_path,
                    'audio_path': seg.get('audio_path', ''),
                    'latent_path': seg.get('latent_path', ''),
                    'segment_count': 0,
                }
            recordings[f0_path]['segment_count'] += 1

    print(f"Found {len(recordings)} unique recordings")

    # Analyze each
    results = []
    flagged = []

    for f0_path, info in tqdm(recordings.items(), desc="Analyzing"):
        metrics = analyze_recording(info['f0_path'], info['amp_path'])

        result = {
            **info,
            **metrics,
        }
        results.append(result)

        if metrics.get('score', 0) >= 25:
            flagged.append(result)

    # Sort flagged by score
    flagged.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total recordings: {len(recordings)}")
    print(f"Flagged as potential ensemble: {len(flagged)}")

    # Group by flag type
    flag_counts = defaultdict(int)
    for r in flagged:
        for flag in r.get('flags', []):
            flag_counts[flag] += 1

    print(f"\nFlag breakdown:")
    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        print(f"  {flag}: {count}")

    # Show top flagged
    print(f"\nTop flagged recordings:")
    for r in flagged[:20]:
        audio_name = os.path.basename(r.get('audio_path', r.get('f0_path', 'unknown')))
        print(f"  Score {r['score']:3d}: {audio_name}")
        print(f"           Flags: {r.get('flags', [])}")
        if r.get('midi_min') and r.get('midi_max'):
            print(f"           MIDI: {r['midi_min']:.0f}-{r['midi_max']:.0f}, voiced: {r.get('voiced_ratio', 0):.1%}")

    # Save results
    output = {
        'total_recordings': len(recordings),
        'flagged_count': len(flagged),
        'flag_counts': dict(flag_counts),
        'flagged_recordings': flagged,
        'all_recordings': results,
    }

    if output_path:
        print(f"\nSaving results to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    return output


def analyze_with_ground_truth(
    segments_json: str,
    mute_manifest: str,
    output_path: str = None,
) -> Dict:
    """
    Analyze using muted recordings as ground truth for clean solo characteristics.
    """
    print("Loading ground truth (muted trumpet recordings)...")
    with open(mute_manifest) as f:
        mute_data = json.load(f)

    # Get muted trumpet entries
    muted = [e for e in mute_data if e.get('is_muted') and e.get('sub_group') == 'trumpet']
    print(f"Found {len(muted)} muted trumpet recordings as ground truth")

    # Analyze ground truth to establish baseline
    gt_metrics = []
    for entry in tqdm(muted, desc="Analyzing ground truth"):
        cond = entry.get('conditioning_paths', {})
        f0_path = cond.get('f0', '')
        amp_path = cond.get('amp', '')

        if f0_path and amp_path:
            metrics = analyze_recording(f0_path, amp_path)
            if metrics.get('valid'):
                gt_metrics.append(metrics)

    if gt_metrics:
        # Calculate ground truth statistics
        gt_stats = {
            'voiced_ratio_mean': np.mean([m['voiced_ratio'] for m in gt_metrics]),
            'voiced_ratio_std': np.std([m['voiced_ratio'] for m in gt_metrics]),
            'pitch_variance_mean': np.mean([m['mean_pitch_variance'] for m in gt_metrics]),
            'pitch_variance_std': np.std([m['mean_pitch_variance'] for m in gt_metrics]),
            'unvoiced_amp_mean': np.mean([m['unvoiced_amp_ratio'] for m in gt_metrics]),
            'unvoiced_amp_std': np.std([m['unvoiced_amp_ratio'] for m in gt_metrics]),
        }
        print(f"\nGround truth statistics:")
        for k, v in gt_stats.items():
            print(f"  {k}: {v:.4f}")

    # Now analyze target segments
    return analyze_segments_json(segments_json, output_path)


def main():
    parser = argparse.ArgumentParser(description="Detect ensemble recordings in trumpet segments")

    parser.add_argument('--segments', type=str,
                        default='/home/arlo/Data/pitchshift/v3/trumpet_segments.json',
                        help='Path to segments JSON')
    parser.add_argument('--mute_manifest', type=str,
                        default='/home/arlo/Data/mute_translator/mute_manifest_deduped.json',
                        help='Path to mute manifest (ground truth)')
    parser.add_argument('--output', type=str,
                        default='/home/arlo/Data/pitchshift/v3/ensemble_detection_results.json',
                        help='Output path for results')
    parser.add_argument('--use_gt', action='store_true',
                        help='Use ground truth analysis')

    args = parser.parse_args()

    if args.use_gt:
        analyze_with_ground_truth(args.segments, args.mute_manifest, args.output)
    else:
        analyze_segments_json(args.segments, args.output)


if __name__ == "__main__":
    main()
