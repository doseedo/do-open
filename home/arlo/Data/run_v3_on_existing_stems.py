#!/usr/bin/env python3
"""
Run V3 classifier on existing demucs "other" stems.
Uses the ~2974 already-separated files from demucs_latent_progress.json
"""

import json
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

PROGRESS_FILE = "/home/arlo/Data/demucs_latent_progress.json"
V3_MODEL = "/home/arlo/Data/mix_classifier/mix_classifier_v3.pt"
OUTPUT_FILE = "/home/arlo/Data/mix_classifier/v3_other_stem_results.json"
FRAMES_PER_SEC = 10.77

class MLPTransform(torch.nn.Module):
    def __init__(self, input_dim=384, output_dim=384, hidden_dims=[1024, 1024]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, h_dim),
                torch.nn.LayerNorm(h_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TargetClassifier(torch.nn.Module):
    def __init__(self, input_dim=384, hidden_dim=64, num_classes=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def load_v3_model():
    checkpoint = torch.load(V3_MODEL, map_location='cpu', weights_only=False)

    transform = MLPTransform(384, 384, checkpoint['transform_hidden_dims'])
    transform.load_state_dict(checkpoint['transform_state'])
    transform.eval()

    classifier_state = checkpoint['classifier_state']
    hidden_dim = classifier_state['net.0.bias'].shape[0]
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

def classify_latent(model_dict, latent_path, hop_frames=50, silence_threshold=0.1):
    """Run V3 classifier on a latent file, return temporal predictions."""
    data = torch.load(latent_path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        latent = data.get('latents', data.get('latent', list(data.values())[0]))
    else:
        latent = data

    # Shape should be [8, 16, T] or similar
    if latent.dim() == 2:
        latent = latent.unsqueeze(0)

    transform = model_dict['transform']
    classifier = model_dict['classifier']
    X_mean = model_dict['X_mean']
    X_std = model_dict['X_std']
    target_classes = model_dict['target_classes']

    results = []
    T = latent.shape[-1]

    with torch.no_grad():
        for start in range(0, T, hop_frames):
            end = min(start + hop_frames, T)
            if end - start < 10:
                continue

            # Check energy - skip silent segments
            chunk = latent[:, :, start:end]
            energy = float(chunk.abs().mean())
            if energy < silence_threshold:
                results.append({
                    'start_sec': start / FRAMES_PER_SEC,
                    'end_sec': end / FRAMES_PER_SEC,
                    'predictions': {cls: 0.0 for cls in target_classes},
                    'skipped': True,
                    'energy': energy
                })
                continue

            # Extract multi-pool features (mean, std, max)
            features = extract_segment_features(latent, start, end)

            # Normalize
            features = (features - X_mean) / (X_std + 1e-8)
            features = features.unsqueeze(0)

            # Transform then classify
            transformed = transform(features)
            logits = classifier(transformed)
            probs = torch.sigmoid(logits).squeeze().tolist()

            if isinstance(probs, float):
                probs = [probs]

            results.append({
                'start_sec': start / FRAMES_PER_SEC,
                'end_sec': end / FRAMES_PER_SEC,
                'energy': energy,
                'predictions': {cls: float(p) for cls, p in zip(target_classes, probs)}
            })

    return results, target_classes

DEMUCS_STEMS = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']

def compute_stem_energy(latent_path, hop_frames=50, silence_threshold=0.1):
    """Compute temporal energy for a demucs stem latent."""
    data = torch.load(latent_path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        latent = data.get('latents', data.get('latent', list(data.values())[0]))
    else:
        latent = data

    if latent.dim() == 2:
        latent = latent.unsqueeze(0)

    T = latent.shape[-1]
    results = []

    for start in range(0, T, hop_frames):
        end = min(start + hop_frames, T)
        if end - start < 5:
            continue
        chunk = latent[:, :, start:end]
        energy = float(chunk.abs().mean())
        is_silent = energy < silence_threshold
        results.append({
            'start_sec': start / FRAMES_PER_SEC,
            'end_sec': end / FRAMES_PER_SEC,
            'energy': energy,
            'silent': is_silent
        })

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--silence-threshold', type=float, default=0.1, help='Energy below this is silent')
    args = parser.parse_args()

    # Load progress file
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)

    processed = progress.get('processed', [])
    print(f"Found {len(processed)} processed files with demucs stems")

    if args.limit:
        processed = processed[:args.limit]
        print(f"Limited to {len(processed)} files")

    # Load model
    print("Loading V3 model...")
    model_dict = load_v3_model()
    target_classes = model_dict['target_classes']
    print(f"Target classes: {target_classes}")

    results = []
    errors = []

    for item in tqdm(processed, desc="Processing mixes"):
        orig_path = item['path']
        output_dir = Path(item['output_dir'])

        # Check all stems exist
        other_latent = output_dir / "other.pt"
        if not other_latent.exists():
            errors.append({'path': orig_path, 'error': 'other.pt not found'})
            continue

        try:
            # 1. Compute energy for all 6 demucs stems
            stem_energies = {}
            stem_temporal = {}
            for stem in DEMUCS_STEMS:
                stem_path = output_dir / f"{stem}.pt"
                if stem_path.exists():
                    temporal_e = compute_stem_energy(stem_path, silence_threshold=args.silence_threshold)
                    energies = [t['energy'] for t in temporal_e if not t.get('silent')]
                    stem_energies[stem] = float(np.mean(energies)) if energies else 0.0
                    stem_temporal[stem] = temporal_e
                else:
                    stem_energies[stem] = 0.0
                    stem_temporal[stem] = []

            # 2. Run V3 classifier on "other" stem for brass/strings/winds
            v3_temporal, _ = classify_latent(model_dict, other_latent, silence_threshold=args.silence_threshold)

            all_preds = {cls: [] for cls in target_classes}
            for t in v3_temporal:
                if t.get('skipped'):
                    continue  # Skip silent segments
                for cls, p in t['predictions'].items():
                    all_preds[cls].append(p)

            v3_agg = {cls: float(np.mean(vals)) if vals else 0.0 for cls, vals in all_preds.items()}
            v3_detected = [cls for cls, p in v3_agg.items() if p >= args.threshold]

            # 3. Combine into final result with all 8 categories
            # Demucs: vocals, drums, bass, guitar, piano (energy-based presence)
            # V3: brass, strings, winds (classifier probability)
            all_categories = {}
            for stem in ['vocals', 'drums', 'bass', 'guitar', 'piano']:
                all_categories[stem] = stem_energies.get(stem, 0.0)
            for cls in target_classes:
                all_categories[cls] = v3_agg.get(cls, 0.0)

            results.append({
                'original_path': orig_path,
                'output_dir': str(output_dir),
                'stem_energies': stem_energies,
                'v3_predictions': v3_agg,
                'v3_detected': v3_detected,
                'all_categories': all_categories,
                'stem_temporal': stem_temporal,
                'v3_temporal': v3_temporal
            })
        except Exception as e:
            errors.append({'path': orig_path, 'error': str(e)})

    # Save results
    all_8_categories = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'brass', 'strings', 'winds']
    output = {
        'total_processed': len(results),
        'total_errors': len(errors),
        'threshold': args.threshold,
        'demucs_stems': DEMUCS_STEMS,
        'v3_classes': target_classes,
        'all_categories': all_8_categories,
        'results': results,
        'errors': errors[:100]
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Processed: {len(results)}, Errors: {len(errors)}")

    # Summary
    if results:
        from collections import Counter
        v3_detected = []
        for r in results:
            v3_detected.extend(r['v3_detected'])
        counts = Counter(v3_detected)
        print(f"\nV3 detection counts (brass/strings/winds at threshold={args.threshold}):")
        for cls in target_classes:
            print(f"  {cls}: {counts.get(cls, 0)}")

        # Stem energy stats
        print(f"\nDemucs stem avg energies:")
        for stem in ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']:
            energies = [r['stem_energies'].get(stem, 0) for r in results]
            avg = np.mean(energies) if energies else 0
            print(f"  {stem}: {avg:.4f}")

if __name__ == "__main__":
    main()
