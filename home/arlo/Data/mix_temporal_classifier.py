#!/usr/bin/env python3
"""
Temporal multi-label classifier for mix audio.
Uses learned transform from mix features to stem space,
then applies solo classifier to detect instruments directly from mix.

Classes: bass, brass, drums, guitar, piano, strings, voice, winds
(Can detect multiple instruments at each time window)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Paths
MIX_CLASSIFIER_PATH = Path("/home/arlo/Data/mix_classifier/mix_classifier.pt")
OUTPUT_DIR = Path("/home/arlo/Data/mix_classifier")

# Target classes for mix detection
TARGET_CLASSES = ['bass', 'brass', 'drums', 'guitar', 'piano', 'strings', 'voice', 'winds']

# Frame rate for temporal windows
FRAMES_PER_SEC = 10.77  # ACE-Step latent frame rate


class MixClassifier:
    """Classifier that works directly on mix audio latents."""

    def __init__(self, checkpoint_path=MIX_CLASSIFIER_PATH):
        data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        self.transform_type = data.get('transform_type', 'linear')

        if self.transform_type == 'mlp_v2':
            # V2: Fresh classifier head (not using solo classifier)
            self.mlp_transform = self._build_mlp_transform(
                data['transform_state'],
                data.get('transform_hidden_dims', [1024, 1024])
            )
            self.mlp_transform.eval()
            self.mlp_X_mean = data['X_mean']
            self.mlp_X_std = data['X_std']
            self.target_classifier = self._build_target_classifier(
                data['classifier_state'],
                len(data['target_classes'])
            )
            self.target_classifier.eval()
            self.classes = data['target_classes']
            self.use_v2 = True
            return

        self.use_v2 = False

        if self.transform_type in ('mlp', 'mlp_joint'):
            # Load MLP transform
            self.mlp_transform = self._build_mlp_transform(
                data['mlp_state'],
                data.get('mlp_hidden_dims', [512, 512])
            )
            self.mlp_transform.eval()
            self.mlp_X_mean = data['mlp_X_mean']
            self.mlp_X_std = data['mlp_X_std']
        else:
            # Linear transform
            self.W = data['transform_W']
            self.b = data['transform_b']

        self.solo_mean = data['solo_mean']
        self.solo_std = data['solo_std']
        self.solo_classes = data['solo_classes']

        # Build model
        self.model = self._build_model(data['solo_model_state'])
        self.model.eval()

        # Get indices for target classes
        self.target_indices = [
            self.solo_classes.index(c) for c in TARGET_CLASSES
            if c in self.solo_classes
        ]
        self.classes = [c for c in TARGET_CLASSES if c in self.solo_classes]

    def _build_mlp_transform(self, state_dict, hidden_dims):
        """Build MLP transform model."""
        class MLPTransform(nn.Module):
            def __init__(self, input_dim=384, output_dim=384, hidden_dims=[512, 512]):
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

        model = MLPTransform(384, 384, hidden_dims)
        model.load_state_dict(state_dict)
        return model

    def _build_target_classifier(self, state_dict, num_classes):
        """Build target classifier for v2 model."""
        class TargetClassifier(nn.Module):
            def __init__(self, input_dim=384, hidden_dim=256, num_classes=3):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, num_classes)
                )
            def forward(self, x):
                return self.net(x)

        model = TargetClassifier(384, 256, num_classes)
        model.load_state_dict(state_dict)
        return model

    def _build_model(self, state_dict):
        class Classifier(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(384, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes)
                )
            def forward(self, x):
                return self.net(x)

        model = Classifier(len(self.solo_classes))
        model.load_state_dict(state_dict)
        return model

    def extract_window_features(self, latent, start_frame, window_frames):
        """Extract features from a time window."""
        end_frame = min(start_frame + window_frames, latent.shape[-1])
        window = latent[:, :, start_frame:end_frame]

        pools = []
        for method in ['mean', 'std', 'max']:
            if method == 'mean':
                pooled = window.mean(dim=-1)
            elif method == 'std':
                pooled = window.std(dim=-1) if window.shape[-1] > 1 else torch.zeros_like(window.mean(dim=-1))
            elif method == 'max':
                pooled = window.max(dim=-1)[0]
            pools.append(pooled.flatten())
        return torch.cat(pools)

    def classify_window(self, features, threshold=0.3):
        """Classify a single window."""
        # V2 model: use fresh classifier head
        if getattr(self, 'use_v2', False):
            features_norm = (features - self.mlp_X_mean) / self.mlp_X_std
            with torch.no_grad():
                transformed = self.mlp_transform(features_norm.unsqueeze(0)).squeeze(0)
                logits = self.target_classifier(transformed.unsqueeze(0)).squeeze(0)
                probs = torch.sigmoid(logits)

            result = {}
            active = []
            for i, cls in enumerate(self.classes):
                p = probs[i].item()
                result[cls] = p
                if p >= threshold:
                    active.append(cls)
            return result, active

        # Original model: transform to stem space
        if self.transform_type in ('mlp', 'mlp_joint'):
            features_norm = (features - self.mlp_X_mean) / self.mlp_X_std
            with torch.no_grad():
                stem_feat = self.mlp_transform(features_norm.unsqueeze(0)).squeeze(0)
        else:
            stem_feat = features @ self.W.T + self.b

        # Normalize
        stem_feat_norm = (stem_feat - self.solo_mean) / self.solo_std

        # Run through model
        with torch.no_grad():
            logits = self.model(stem_feat_norm.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze()

        # Get target class probabilities
        result = {}
        active = []
        for i, cls in enumerate(self.classes):
            idx = self.solo_classes.index(cls)
            p = probs[idx].item()
            result[cls] = p
            if p >= threshold:
                active.append(cls)

        return result, active

    def classify_temporal(self, latent, window_sec=1.0, hop_sec=0.5, threshold=0.3):
        """Run temporal classification on a latent."""
        T = latent.shape[-1]
        window_frames = int(window_sec * FRAMES_PER_SEC)
        hop_frames = int(hop_sec * FRAMES_PER_SEC)

        results = []
        frame = 0
        while frame < T:
            features = self.extract_window_features(latent, frame, window_frames)
            probs, active = self.classify_window(features, threshold)

            start_sec = frame / FRAMES_PER_SEC
            end_sec = min(frame + window_frames, T) / FRAMES_PER_SEC

            results.append({
                'start_sec': round(start_sec, 2),
                'end_sec': round(end_sec, 2),
                'probabilities': {k: round(v, 3) for k, v in probs.items()},
                'active_classes': active
            })

            frame += hop_frames

        return results


def merge_temporal_regions(temporal_results, gap_tolerance=2.0):
    """Merge consecutive windows with same classes."""
    if not temporal_results:
        return []

    merged = []

    for cls in TARGET_CLASSES:
        # Find all windows where this class is active
        active_windows = [
            (t['start_sec'], t['end_sec'], t['probabilities'].get(cls, 0))
            for t in temporal_results
            if cls in t.get('active_classes', [])
        ]

        if not active_windows:
            continue

        # Merge consecutive windows
        current_start = active_windows[0][0]
        current_end = active_windows[0][1]
        current_probs = [active_windows[0][2]]

        for start, end, prob in active_windows[1:]:
            if start <= current_end + gap_tolerance:
                current_end = end
                current_probs.append(prob)
            else:
                merged.append({
                    'class': cls,
                    'start_sec': current_start,
                    'end_sec': current_end,
                    'avg_confidence': round(sum(current_probs) / len(current_probs), 3)
                })
                current_start = start
                current_end = end
                current_probs = [prob]

        merged.append({
            'class': cls,
            'start_sec': current_start,
            'end_sec': current_end,
            'avg_confidence': round(sum(current_probs) / len(current_probs), 3)
        })

    # Sort by start time
    merged.sort(key=lambda x: (x['start_sec'], x['class']))
    return merged


def run_batch(args):
    """Run batch classification on mix latents."""
    classifier = MixClassifier(checkpoint_path=args.model)
    print(f"Loaded classifier ({classifier.transform_type}) with classes: {classifier.classes}")

    # Find latent files
    latent_dir = Path(args.input_dir)
    latent_files = list(latent_dir.rglob("*.pt"))

    if args.limit:
        latent_files = latent_files[:args.limit]

    print(f"Processing {len(latent_files)} files...")

    results = []
    detection_counts = defaultdict(int)

    for i, f in enumerate(latent_files):
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(latent_files)}] Processed...")

        try:
            data = torch.load(f, map_location='cpu', weights_only=False)
            latent = data['latents'] if isinstance(data, dict) else data

            temporal = classifier.classify_temporal(
                latent,
                window_sec=args.window_sec,
                hop_sec=args.hop_sec,
                threshold=args.threshold
            )

            merged = merge_temporal_regions(temporal, gap_tolerance=args.gap_tolerance)

            # Count detections
            detected = list(set(m['class'] for m in merged))
            for cls in detected:
                detection_counts[cls] += 1

            results.append({
                'path': str(f),
                'filename': f.stem,
                'detected_instruments': detected,
                'temporal': temporal,
                'merged': merged
            })

        except Exception as e:
            results.append({
                'path': str(f),
                'error': str(e)
            })

    # Save results
    output = {
        'total': len(results),
        'classes': classifier.classes,
        'threshold': args.threshold,
        'detection_counts': dict(detection_counts),
        'results': results
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Processed {len(results)} files")
    print(f"\nDetection counts:")
    for cls, count in sorted(detection_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Mix audio temporal classifier')
    parser.add_argument('--mode', choices=['batch', 'single'], default='batch')
    parser.add_argument('--input-dir', type=str, help='Directory with latent files')
    parser.add_argument('--input', type=str, help='Single latent file')
    parser.add_argument('--output', type=str, default='mix_temporal_results.json')
    parser.add_argument('--model', type=str, default=str(MIX_CLASSIFIER_PATH),
                        help='Path to mix classifier checkpoint')
    parser.add_argument('--window-sec', type=float, default=2.0)
    parser.add_argument('--hop-sec', type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--gap-tolerance', type=float, default=3.0)
    parser.add_argument('--limit', type=int, help='Limit number of files')

    args = parser.parse_args()

    if args.mode == 'batch':
        if not args.input_dir:
            print("Error: --input-dir required for batch mode")
            return
        run_batch(args)
    else:
        if not args.input:
            print("Error: --input required for single mode")
            return
        # Single file classification
        classifier = MixClassifier(checkpoint_path=args.model)
        data = torch.load(args.input, map_location='cpu', weights_only=False)
        latent = data['latents'] if isinstance(data, dict) else data

        temporal = classifier.classify_temporal(
            latent,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            threshold=args.threshold
        )
        merged = merge_temporal_regions(temporal)

        print(f"\nDetected instruments: {list(set(m['class'] for m in merged))}")
        print(f"\nMerged regions:")
        for m in merged:
            print(f"  {m['class']}: {m['start_sec']:.1f}s - {m['end_sec']:.1f}s ({m['avg_confidence']:.2f})")


if __name__ == "__main__":
    main()
