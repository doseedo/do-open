#!/usr/bin/env python3
"""
Run mix classifier v3 on mix latents and generate temporal predictions.
V3 is trained on segment-level data with timestamps for better temporal accuracy.

Usage:
  python3 run_mix_classifier_v3.py --input-dir /home/arlo/gcs-bucket/Latents --limit 500
"""

import torch
import torch.nn as nn
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

MODEL_PATH = Path("/home/arlo/Data/mix_classifier/mix_classifier_v3.pt")
OUTPUT_DIR = Path("/home/arlo/Data/mix_classifier")
FRAMES_PER_SEC = 10.77  # ACE-Step latent frame rate


class MLPTransform(nn.Module):
    def __init__(self, input_dim=384, output_dim=384, hidden_dims=[1024, 1024]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TargetClassifier(nn.Module):
    """Simple classifier matching v3 training architecture."""
    def __init__(self, input_dim=384, hidden_dim=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def load_model(model_path):
    """Load the v3 mix classifier (segment-trained)."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    transform = MLPTransform(384, 384, checkpoint['transform_hidden_dims'])
    transform.load_state_dict(checkpoint['transform_state'])
    transform.eval()

    # Infer hidden_dim from classifier weights
    classifier_state = checkpoint['classifier_state']
    hidden_dim = classifier_state['net.0.bias'].shape[0]  # First layer output dim
    num_classes = len(checkpoint['target_classes'])

    classifier = TargetClassifier(384, hidden_dim, num_classes)
    classifier.load_state_dict(classifier_state)
    classifier.eval()

    return {
        'transform': transform,
        'classifier': classifier,
        'X_mean': checkpoint['X_mean'],
        'X_std': checkpoint['X_std'],
        'target_classes': checkpoint['target_classes']
    }


def extract_segment_features(latent, start_frame, end_frame):
    """Extract features from a time segment using multi-pool."""
    segment = latent[:, :, start_frame:end_frame]

    pools = []
    for method in ['mean', 'std', 'max']:
        if method == 'mean':
            pooled = segment.mean(dim=-1)
        elif method == 'std':
            pooled = segment.std(dim=-1) if segment.shape[-1] > 1 else torch.zeros_like(segment.mean(dim=-1))
        elif method == 'max':
            pooled = segment.max(dim=-1)[0]
        pools.append(pooled.flatten())
    return torch.cat(pools)


def classify_temporal_v3(model, latent, segment_sec=2.0, hop_sec=0.5, threshold=0.5):
    """
    Run temporal classification using overlapping segments.
    Uses smaller hop for finer temporal resolution.
    """
    transform = model['transform']
    classifier = model['classifier']
    X_mean = model['X_mean']
    X_std = model['X_std']
    target_classes = model['target_classes']

    T = latent.shape[-1]
    segment_frames = int(segment_sec * FRAMES_PER_SEC)
    hop_frames = max(1, int(hop_sec * FRAMES_PER_SEC))

    results = []
    frame = 0

    while frame < T:
        end_frame = min(frame + segment_frames, T)
        if end_frame - frame < segment_frames // 2:
            break  # Skip too-short final segment

        features = extract_segment_features(latent, frame, end_frame)
        features_norm = (features - X_mean) / X_std

        with torch.no_grad():
            transformed = transform(features_norm.unsqueeze(0)).squeeze(0)
            logits = classifier(transformed.unsqueeze(0)).squeeze(0)
            probs = torch.sigmoid(logits)

        start_sec = frame / FRAMES_PER_SEC
        end_sec = end_frame / FRAMES_PER_SEC

        # Get active classes above threshold
        active_classes = []
        probabilities = {}
        for i, cls in enumerate(target_classes):
            p = probs[i].item()
            probabilities[cls] = round(p, 3)
            if p >= threshold:
                active_classes.append(cls)

        results.append({
            'start_sec': round(start_sec, 2),
            'end_sec': round(end_sec, 2),
            'start_frame': frame,
            'end_frame': end_frame,
            'probabilities': probabilities,
            'active_classes': active_classes
        })

        frame += hop_frames

    return results


def merge_temporal_regions_v3(temporal_results, target_classes, min_duration=0.5, gap_tolerance=1.0):
    """
    Merge consecutive segments with same classes into regions.
    More aggressive merging for cleaner output.
    """
    if not temporal_results:
        return []

    merged = []

    for cls in target_classes:
        # Find all segments where this class is active
        active_segments = [
            (t['start_sec'], t['end_sec'], t['probabilities'].get(cls, 0))
            for t in temporal_results
            if cls in t.get('active_classes', [])
        ]

        if not active_segments:
            continue

        # Merge consecutive segments
        current_start = active_segments[0][0]
        current_end = active_segments[0][1]
        current_probs = [active_segments[0][2]]

        for start, end, prob in active_segments[1:]:
            if start <= current_end + gap_tolerance:
                current_end = max(current_end, end)
                current_probs.append(prob)
            else:
                duration = current_end - current_start
                if duration >= min_duration:
                    merged.append({
                        'class': cls,
                        'start_sec': current_start,
                        'end_sec': current_end,
                        'avg_confidence': round(sum(current_probs) / len(current_probs), 3),
                        'duration': round(duration, 2)
                    })
                current_start = start
                current_end = end
                current_probs = [prob]

        duration = current_end - current_start
        if duration >= min_duration:
            merged.append({
                'class': cls,
                'start_sec': current_start,
                'end_sec': current_end,
                'avg_confidence': round(sum(current_probs) / len(current_probs), 3),
                'duration': round(duration, 2)
            })

    # Sort by start time
    merged.sort(key=lambda x: (x['start_sec'], x['class']))

    # Group overlapping regions by time into multi-class regions
    grouped = []
    i = 0
    while i < len(merged):
        start = merged[i]['start_sec']
        end = merged[i]['end_sec']
        classes = [merged[i]['class']]
        confidences = {merged[i]['class']: merged[i]['avg_confidence']}

        # Find overlapping regions
        j = i + 1
        while j < len(merged) and merged[j]['start_sec'] < end:
            overlap_start = max(start, merged[j]['start_sec'])
            overlap_end = min(end, merged[j]['end_sec'])
            overlap = overlap_end - overlap_start
            # Require significant overlap
            if overlap > 0.5:
                classes.append(merged[j]['class'])
                confidences[merged[j]['class']] = merged[j]['avg_confidence']
                end = max(end, merged[j]['end_sec'])
            j += 1

        grouped.append({
            'start_sec': round(start, 2),
            'end_sec': round(end, 2),
            'classes': sorted(set(classes)),
            'avg_confidence': confidences
        })
        i = j if j > i + 1 else i + 1

    return grouped


def load_latent(path):
    """Load latent from .pt file."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        return data.get('latents', data.get('latent'))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='/home/arlo/gcs-bucket/Latents',
                        help='Directory with latent files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--model', type=str, default=str(MODEL_PATH),
                        help='Model checkpoint path')
    parser.add_argument('--limit', type=int, default=500,
                        help='Max files to process')
    parser.add_argument('--segment-sec', type=float, default=2.0)
    parser.add_argument('--hop-sec', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--gap-tolerance', type=float, default=1.0)
    parser.add_argument('--min-duration', type=float, default=0.5)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(OUTPUT_DIR / 'mix_classifier_v3_temporal.json')

    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    target_classes = model['target_classes']
    print(f"Target classes: {target_classes}")

    # Find latent files
    latent_dir = Path(args.input_dir)
    latent_files = list(latent_dir.rglob("*.pt"))

    # Filter out Demucs stems (we want original mixes only)
    latent_files = [f for f in latent_files if 'Demucs' not in str(f)]

    if args.limit:
        latent_files = latent_files[:args.limit]

    print(f"Processing {len(latent_files)} files...")

    results = []
    detection_counts = defaultdict(int)

    for i, f in enumerate(latent_files):
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(latent_files)}] Processed...")

        try:
            latent = load_latent(f)
            if latent is None:
                continue

            # Run temporal classification
            temporal = classify_temporal_v3(
                model, latent,
                segment_sec=args.segment_sec,
                hop_sec=args.hop_sec,
                threshold=args.threshold
            )

            # Merge into regions
            merged = merge_temporal_regions_v3(
                temporal, target_classes,
                min_duration=args.min_duration,
                gap_tolerance=args.gap_tolerance
            )

            # Get detected instruments
            detected = list(set(cls for m in merged for cls in m['classes']))
            for cls in detected:
                detection_counts[cls] += 1

            # Convert latent path to audio path
            audio_path = str(f).replace('/Latents/', '/').replace('.pt', '.wav')

            results.append({
                'path': str(f),
                'audio_path': audio_path,
                'filename': f.stem,
                'detected': detected,
                'temporal': temporal,
                'merged': merged,
                'duration': latent.shape[-1] / FRAMES_PER_SEC
            })

        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

    # Save results
    output_data = {
        'model': str(args.model),
        'model_version': 'v3',
        'training_type': 'segment-level',
        'total': len(results),
        'target_classes': target_classes,
        'threshold': args.threshold,
        'segment_sec': args.segment_sec,
        'hop_sec': args.hop_sec,
        'gap_tolerance': args.gap_tolerance,
        'min_duration': args.min_duration,
        'detection_counts': dict(detection_counts),
        'generated_at': datetime.now().isoformat(),
        'results': results
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*50}")
    print(f"V3 Model (segment-trained) - Processed {len(results)} files")
    print(f"\nDetection counts:")
    for cls, count in sorted(detection_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
