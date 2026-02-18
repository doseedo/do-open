#!/usr/bin/env python3
"""
Run instrument classifier on each demucs-separated stem.
Verifies what's actually in each stem rather than assuming based on energy.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

PROGRESS_FILE = "/home/arlo/Data/demucs_latent_progress.json"
CLASSIFIER_PATH = "/home/arlo/Data/latent_classifier/model.pt"
OUTPUT_FILE = "/home/arlo/Data/mix_classifier/stems_classified.json"
FRAMES_PER_SEC = 10.77

DEMUCS_STEMS = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']

# Expected mappings (what we expect each stem to contain)
EXPECTED_CLASSES = {
    'vocals': ['voice'],
    'drums': ['drums', 'percussion'],
    'bass': ['bass'],
    'guitar': ['guitar'],
    'piano': ['piano', 'organ', 'synth'],
    'other': ['brass', 'strings', 'winds', 'synth', 'mallets', 'plucked']
}


class InstrumentClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_classes=13):
        super().__init__()
        # Architecture: 384 -> 256 -> 256 -> 128 -> 13
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),          # 0
            nn.BatchNorm1d(hidden_dim),                # 1
            nn.GELU(),                                  # 2
            nn.Dropout(0.2),                           # 3
            nn.Linear(hidden_dim, hidden_dim),         # 4
            nn.BatchNorm1d(hidden_dim),                # 5
            nn.GELU(),                                  # 6
            nn.Dropout(0.2),                           # 7
            nn.Linear(hidden_dim, hidden_dim // 2),    # 8
            nn.BatchNorm1d(hidden_dim // 2),           # 9
            nn.GELU(),                                  # 10
            nn.Dropout(0.2),                           # 11
            nn.Linear(hidden_dim // 2, num_classes)    # 12
        )

    def forward(self, x):
        return self.net(x)


def load_classifier():
    checkpoint = torch.load(CLASSIFIER_PATH, map_location='cpu', weights_only=False)

    model = InstrumentClassifier(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return {
        'model': model,
        'classes': list(checkpoint['label_encoder_classes']),
        'mean': checkpoint['mean'],
        'std': checkpoint['std']
    }


def extract_features(latent):
    """Extract features using multi-pool."""
    pools = []
    for method in ['mean', 'std', 'max']:
        if method == 'mean':
            pooled = latent.mean(dim=-1)
        elif method == 'std':
            pooled = latent.std(dim=-1) if latent.shape[-1] > 1 else torch.zeros_like(latent.mean(dim=-1))
        elif method == 'max':
            pooled = latent.max(dim=-1)[0]
        pools.append(pooled.flatten())
    return torch.cat(pools)


def classify_stem(classifier, latent_path, hop_frames=50, silence_threshold=0.1):
    """Classify a stem temporally."""
    data = torch.load(latent_path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        latent = data.get('latents', data.get('latent', list(data.values())[0]))
    else:
        latent = data

    if latent.dim() == 2:
        latent = latent.unsqueeze(0)

    model = classifier['model']
    classes = classifier['classes']
    mean = classifier['mean']
    std = classifier['std']

    T = latent.shape[-1]
    results = []

    with torch.no_grad():
        for start in range(0, T, hop_frames):
            end = min(start + hop_frames, T)
            if end - start < 10:
                continue

            chunk = latent[:, :, start:end]
            energy = float(chunk.abs().mean())

            if energy < silence_threshold:
                results.append({
                    'start_sec': start / FRAMES_PER_SEC,
                    'end_sec': end / FRAMES_PER_SEC,
                    'energy': energy,
                    'silent': True,
                    'predictions': {}
                })
                continue

            features = extract_features(chunk)
            features = (features - mean) / (std + 1e-8)

            logits = model(features.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze()

            top_probs, top_idx = probs.topk(3)
            predictions = {classes[idx]: float(top_probs[i]) for i, idx in enumerate(top_idx.tolist())}

            results.append({
                'start_sec': start / FRAMES_PER_SEC,
                'end_sec': end / FRAMES_PER_SEC,
                'energy': energy,
                'silent': False,
                'predictions': predictions,
                'top_class': classes[top_idx[0].item()],
                'top_confidence': float(top_probs[0])
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=0.3)
    args = parser.parse_args()

    # Load progress
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)

    processed = progress.get('processed', [])
    print(f"Found {len(processed)} files with demucs stems")

    if args.limit:
        processed = processed[:args.limit]

    # Load classifier
    print("Loading classifier...")
    classifier = load_classifier()
    print(f"Classes: {classifier['classes']}")

    results = []
    errors = []

    for item in tqdm(processed, desc="Classifying stems"):
        orig_path = item['path']
        output_dir = Path(item['output_dir'])

        try:
            stem_results = {}
            timeline = []

            # Classify each stem
            for stem in DEMUCS_STEMS:
                stem_path = output_dir / f"{stem}.pt"
                if not stem_path.exists():
                    continue

                temporal = classify_stem(classifier, stem_path)

                # Aggregate
                all_preds = {}
                active_segments = [t for t in temporal if not t.get('silent')]

                for t in active_segments:
                    for cls, prob in t.get('predictions', {}).items():
                        if cls not in all_preds:
                            all_preds[cls] = []
                        all_preds[cls].append(prob)

                avg_preds = {cls: np.mean(probs) for cls, probs in all_preds.items()}
                top_class = max(avg_preds, key=avg_preds.get) if avg_preds else None

                # Check if classification matches expected
                expected = EXPECTED_CLASSES.get(stem, [])
                matches_expected = top_class in expected if top_class else False

                # Find best expected class if top doesn't match
                expected_pred = None
                expected_conf = 0
                for exp_cls in expected:
                    if exp_cls in avg_preds and avg_preds[exp_cls] > expected_conf:
                        expected_pred = exp_cls
                        expected_conf = avg_preds[exp_cls]

                # Use expected class if reasonable, else use top class only if high confidence
                HIGH_CONF_THRESHOLD = 0.7
                if matches_expected:
                    final_class = top_class
                    final_conf = avg_preds.get(top_class, 0)
                elif expected_conf > 0.2:  # Expected class has some signal
                    final_class = expected_pred
                    final_conf = expected_conf
                elif avg_preds.get(top_class, 0) > HIGH_CONF_THRESHOLD:  # Very confident in unexpected
                    final_class = top_class
                    final_conf = avg_preds.get(top_class, 0)
                else:
                    final_class = None  # Not confident enough
                    final_conf = 0

                stem_results[stem] = {
                    'top_class': top_class,
                    'top_confidence': avg_preds.get(top_class, 0) if top_class else 0,
                    'final_class': final_class,
                    'final_confidence': final_conf,
                    'expected_class': expected_pred,
                    'expected_confidence': expected_conf,
                    'all_predictions': avg_preds,
                    'matches_expected': matches_expected,
                    'temporal': temporal
                }

            # Build unified timeline using final_class (filtered)
            other_temporal = stem_results.get('other', {}).get('temporal', [])

            for i, t in enumerate(other_temporal):
                if t.get('silent'):
                    continue

                instruments = []

                # Check each stem at this time - use final_class
                for stem in DEMUCS_STEMS:
                    stem_data = stem_results.get(stem, {})
                    final_class = stem_data.get('final_class')
                    final_conf = stem_data.get('final_confidence', 0)

                    if final_class and final_conf > 0.2:
                        instruments.append({
                            'instrument': final_class,
                            'confidence': final_conf,
                            'stem': stem
                        })

                if instruments:
                    timeline.append({
                        'start': t['start_sec'],
                        'end': t['end_sec'],
                        'instruments': instruments
                    })

            results.append({
                'original_path': orig_path,
                'output_dir': str(output_dir),
                'stems': stem_results,
                'timeline': timeline,
                'detected_instruments': list(set(
                    s['final_class'] for s in stem_results.values()
                    if s.get('final_class') and s.get('final_confidence', 0) > args.threshold
                ))
            })

        except Exception as e:
            errors.append({'path': orig_path, 'error': str(e)})

    # Save
    output = {
        'total': len(results),
        'errors': len(errors),
        'classifier_classes': classifier['classes'],
        'results': results
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"Processed: {len(results)}, Errors: {len(errors)}")

    # Summary
    if results:
        from collections import Counter
        all_detected = []
        for r in results:
            all_detected.extend(r['detected_instruments'])
        print(f"\nDetected instruments: {dict(Counter(all_detected))}")


if __name__ == "__main__":
    main()
