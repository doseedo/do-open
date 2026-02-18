#!/usr/bin/env python3
"""
Multi-label classifier for Demucs "other" stem.

Detects: brass, strings, winds, synth (includes organ)
Excludes: percussion (-> drums), plucked (-> guitar)

Trained on:
1. Single-label GT from manifest (45k files)
2. Multi-label GT from corrections (16 examples)
3. Synthetic multi-label by mixing latents

Usage:
    # Train
    python3 other_stem_classifier.py --mode train --synthetic 5000

    # Classify with 1s windows
    python3 other_stem_classifier.py --mode temporal --input /path/to/other.pt

    # Batch classify
    python3 other_stem_classifier.py --mode batch --input-dir /path/to/latents/
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score
import random
from datetime import datetime

# Configuration
GCS_BASE = Path("/home/arlo/gcs-bucket")
LATENTS_BASE = GCS_BASE / "Latents"
COMBINED_MANIFEST = GCS_BASE / "Manifests/combined_manifest.json"
FORMAT_MANIFEST = GCS_BASE / "Manifests/format_manifest.json"
CORRECTIONS_PATH = GCS_BASE / "Manifests/corrections.json"
OUTPUT_DIR = Path("/home/arlo/Data/other_classifier")

# Cache for has_latent lookup
_HAS_LATENT_CACHE = None


def load_has_latent_lookup():
    """Load pre-computed has_latent lookup from format_manifest.json for O(1) checks."""
    global _HAS_LATENT_CACHE
    if _HAS_LATENT_CACHE is not None:
        return _HAS_LATENT_CACHE

    print("Loading has_latent lookup from format_manifest.json...")
    with open(FORMAT_MANIFEST) as f:
        fmt = json.load(f)

    # Build set of paths that have latents (paths are relative in format_manifest)
    _HAS_LATENT_CACHE = set()
    for entry in fmt.get('entries', []):
        if entry.get('has_latent') == True:  # Explicit True check (can be "skipped" string)
            # Convert relative path to absolute
            abs_path = str(GCS_BASE / entry['path'])
            _HAS_LATENT_CACHE.add(abs_path)

    print(f"  Loaded {len(_HAS_LATENT_CACHE)} paths with latents")
    return _HAS_LATENT_CACHE

# Default classes for "other" stem
# Can be overridden with --train-classes
DEFAULT_CLASSES = ['brass', 'strings', 'winds', 'synth']
CLASSES = DEFAULT_CLASSES.copy()
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def set_classes(class_list):
    """Update global CLASSES and CLASS_TO_IDX."""
    global CLASSES, CLASS_TO_IDX, GROUP_MAP
    CLASSES = class_list
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    # Update GROUP_MAP to only include active classes
    GROUP_MAP = {c: c for c in CLASSES}
    if 'synth' in CLASSES:
        GROUP_MAP['organ'] = 'synth'

# Mapping for consolidation
GROUP_MAP = {
    'brass': 'brass',
    'strings': 'strings',
    'winds': 'winds',
    'synth': 'synth',
    'organ': 'synth',  # organ -> synth
    # Excluded from this classifier:
    # 'percussion': excluded (use drums stem)
    # 'plucked': excluded (use guitar stem)
}

# Latent params
LATENT_SHAPE = (8, 16)  # [8, 16, T]
# ACE-Step latent frame rate: ~11.73 fps for demucs stems (704 frames / 60s max)
# Original audio latents use different rate, but demucs stems are resampled
FRAMES_PER_SEC = 704 / 60  # ~11.73 fps for demucs-extracted latents
WINDOW_FRAMES = int(1.0 * FRAMES_PER_SEC)  # 1 second window (~12 frames)


class MultiLabelOtherDataset(Dataset):
    """Dataset for multi-label other stem classification."""

    def __init__(self, samples, window_frames=WINDOW_FRAMES):
        """
        samples: list of (latent_path, labels) where labels is set of class names
        """
        self.samples = samples
        self.window_frames = window_frames
        self.num_classes = len(CLASSES)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        latent_path, labels = self.samples[idx]

        # Load latent
        try:
            data = torch.load(latent_path, map_location='cpu', weights_only=False)
            # Latent files are dicts with 'latents' key
            latent = data['latents'] if isinstance(data, dict) else data
        except:
            # Return zeros if load fails
            latent = torch.zeros(8, 16, self.window_frames)

        # Random window if latent is longer
        T = latent.shape[-1]
        if T > self.window_frames:
            start = random.randint(0, T - self.window_frames)
            latent = latent[:, :, start:start + self.window_frames]
        elif T < self.window_frames:
            # Pad with zeros
            pad = torch.zeros(8, 16, self.window_frames - T)
            latent = torch.cat([latent, pad], dim=-1)

        # Pool to fixed features: mean, std, max across time
        features = self._pool_latent(latent)

        # Multi-hot label
        label_vec = torch.zeros(self.num_classes)
        for lbl in labels:
            if lbl in CLASS_TO_IDX:
                label_vec[CLASS_TO_IDX[lbl]] = 1.0

        return features, label_vec

    def _pool_latent(self, latent):
        """Pool [8, 16, T] latent to fixed-size feature vector."""
        # Flatten to [128, T]
        flat = latent.reshape(-1, latent.shape[-1])

        # Pool across time
        mean_feat = flat.mean(dim=-1)
        std_feat = flat.std(dim=-1)
        max_feat = flat.max(dim=-1)[0]

        # Concatenate: 128 * 3 = 384 features
        return torch.cat([mean_feat, std_feat, max_feat])


class SyntheticMixer:
    """Generate synthetic multi-label samples by mixing latents."""

    def __init__(self, single_label_paths: dict, output_dir: Path):
        """
        single_label_paths: {class_name: [list of latent paths]}
        """
        self.paths_by_class = single_label_paths
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, num_samples: int) -> list:
        """Generate synthetic multi-label samples."""
        samples = []
        classes = list(self.paths_by_class.keys())

        for i in range(num_samples):
            # Randomly pick 2 or 3 classes to mix
            num_mix = random.choices([2, 3], weights=[0.7, 0.3])[0]
            mix_classes = random.sample(classes, min(num_mix, len(classes)))

            # Load latents
            latents = []
            for cls in mix_classes:
                if self.paths_by_class[cls]:
                    path = random.choice(self.paths_by_class[cls])
                    try:
                        data = torch.load(path, map_location='cpu', weights_only=False)
                        # Latent files are dicts with 'latents' key
                        lat = data['latents'] if isinstance(data, dict) else data
                        latents.append(lat)
                    except:
                        pass

            if len(latents) < 2:
                continue

            # Mix latents with random weights
            mixed = self._mix_latents(latents)

            # Save synthetic latent
            out_path = self.output_dir / f"synthetic_{i:05d}.pt"
            torch.save(mixed, out_path)

            samples.append((out_path, set(mix_classes)))

            if (i + 1) % 500 == 0:
                print(f"  Generated {i + 1}/{num_samples} synthetic samples")

        return samples

    def _mix_latents(self, latents: list) -> torch.Tensor:
        """Mix multiple latents together."""
        # Find minimum length across all latents
        min_len = min(lat.shape[-1] for lat in latents)

        # Use at least WINDOW_FRAMES if possible, otherwise use min_len
        target_len = min(min_len, WINDOW_FRAMES) if min_len < WINDOW_FRAMES else WINDOW_FRAMES

        # Trim/pad all latents to target length
        processed = []
        for lat in latents:
            if lat.shape[-1] > target_len:
                # Trim
                lat = lat[:, :, :target_len]
            elif lat.shape[-1] < target_len:
                # Pad with zeros
                pad = torch.zeros(8, 16, target_len - lat.shape[-1])
                lat = torch.cat([lat, pad], dim=-1)
            processed.append(lat)

        # Random mixing weights
        weights = torch.rand(len(processed))
        weights = weights / weights.sum()

        # Weighted sum
        mixed = torch.zeros_like(processed[0])
        for w, lat in zip(weights, processed):
            mixed += w * lat

        return mixed


class MultiLabelClassifier(nn.Module):
    """Multi-label classifier with sigmoid outputs."""

    def __init__(self, input_dim=384, num_classes=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)  # Raw logits, apply sigmoid for probabilities


def load_single_label_data(manifest_path: Path, latents_base: Path, limit_per_class: int = None):
    """Load single-label samples from manifest."""
    # Load has_latent lookup for O(1) checks instead of slow file existence
    has_latent_set = load_has_latent_lookup()

    print("Loading manifest...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    paths_by_class = defaultdict(list)

    for path, meta in manifest.items():
        if not isinstance(meta, dict):
            continue

        group = meta.get('group', '')
        mapped = GROUP_MAP.get(group)
        if not mapped:
            continue

        # Check if latent exists using O(1) lookup
        if path not in has_latent_set:
            continue

        # Build latent path (audio.wav -> audio.pt, not audio.wav.pt)
        rel_path = path.replace(str(GCS_BASE) + '/', '')
        # Remove .wav extension and add .pt
        if rel_path.endswith('.wav'):
            rel_path = rel_path[:-4]
        latent_path = latents_base / (rel_path + '.pt')
        paths_by_class[mapped].append(latent_path)

    # Apply limit per class if specified
    if limit_per_class:
        for cls in paths_by_class:
            if len(paths_by_class[cls]) > limit_per_class:
                paths_by_class[cls] = random.sample(paths_by_class[cls], limit_per_class)

    print(f"Loaded single-label samples:")
    for cls, paths in paths_by_class.items():
        print(f"  {cls}: {len(paths)}")

    # Convert to samples list
    samples = []
    for cls, paths in paths_by_class.items():
        for p in paths:
            samples.append((p, {cls}))

    return samples, dict(paths_by_class)


def load_multilabel_gt(corrections_path: Path, latents_base: Path):
    """Load multi-label ground truth from corrections."""
    # Load has_latent lookup for O(1) checks
    has_latent_set = load_has_latent_lookup()

    print("Loading multi-label corrections...")
    with open(corrections_path) as f:
        corrections = json.load(f)

    samples = []

    for path, data in corrections.items():
        regions = data.get('regions', [])
        if not regions:
            continue

        # Check if latent exists using O(1) lookup
        if path not in has_latent_set:
            continue

        # Check for "other" group labels
        for region in regions:
            labels = set(region.get('labels', []))
            mapped_labels = set()
            for lbl in labels:
                if lbl in GROUP_MAP:
                    mapped_labels.add(GROUP_MAP[lbl])

            if len(mapped_labels) >= 1:  # At least one "other" class
                # Build latent path (audio.wav -> audio.pt)
                rel_path = path.replace(str(GCS_BASE) + '/', '')
                if rel_path.endswith('.wav'):
                    rel_path = rel_path[:-4]
                latent_path = latents_base / (rel_path + '.pt')
                samples.append((latent_path, mapped_labels))

    print(f"Loaded {len(samples)} multi-label GT samples")
    return samples


def train(args):
    """Train the multi-label classifier."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Load data
    single_samples, paths_by_class = load_single_label_data(
        COMBINED_MANIFEST, LATENTS_BASE,
        limit_per_class=args.limit_per_class
    )

    multi_samples = load_multilabel_gt(CORRECTIONS_PATH, LATENTS_BASE)

    # Generate synthetic multi-label
    synthetic_samples = []
    if args.synthetic > 0:
        print(f"\nGenerating {args.synthetic} synthetic multi-label samples...")
        mixer = SyntheticMixer(paths_by_class, OUTPUT_DIR / "synthetic")
        synthetic_samples = mixer.generate(args.synthetic)
        print(f"Generated {len(synthetic_samples)} synthetic samples")

    # Combine all samples
    all_samples = single_samples + multi_samples + synthetic_samples
    random.shuffle(all_samples)

    # Train/val split
    split_idx = int(0.9 * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # Create datasets
    train_dataset = MultiLabelOtherDataset(train_samples)
    val_dataset = MultiLabelOtherDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Model
    model = MultiLabelClassifier(input_dim=384, num_classes=len(CLASSES))
    model.to(device)

    # Loss and optimizer (BCE for multi-label)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Training loop
    best_f1 = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Calculate F1
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        scheduler.step(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"F1: {f1:.3f}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'model_state': model.state_dict(),
                'classes': CLASSES,
                'class_to_idx': CLASS_TO_IDX,
                'input_dim': 384,
                'num_classes': len(CLASSES),
                'best_f1': best_f1,
                'trained_at': datetime.now().isoformat()
            }, OUTPUT_DIR / "model.pt")
            print(f"  Saved best model (F1={best_f1:.3f})")

    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0))


def classify_temporal(latent_path: Path, model, device, window_sec=1.0, hop_sec=0.5, threshold=0.5):
    """Classify latent in temporal windows."""
    latent = torch.load(latent_path, map_location='cpu', weights_only=True)

    window_frames = int(window_sec * FRAMES_PER_SEC)
    hop_frames = int(hop_sec * FRAMES_PER_SEC)

    T = latent.shape[-1]
    results = []

    model.eval()
    with torch.no_grad():
        for start in range(0, T - window_frames + 1, hop_frames):
            end = start + window_frames
            window = latent[:, :, start:end]

            # Pool to features
            flat = window.reshape(-1, window.shape[-1])
            mean_feat = flat.mean(dim=-1)
            std_feat = flat.std(dim=-1)
            max_feat = flat.max(dim=-1)[0]
            features = torch.cat([mean_feat, std_feat, max_feat]).unsqueeze(0).to(device)

            # Predict
            logits = model(features)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            # Get active classes
            active = [CLASSES[i] for i, p in enumerate(probs) if p > threshold]

            results.append({
                'start_sec': start / FRAMES_PER_SEC,
                'end_sec': end / FRAMES_PER_SEC,
                'active_classes': active,
                'probabilities': {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
            })

    return results


def merge_temporal_regions(temporal_results, gap_tolerance=2.0, exclude_classes=None):
    """Merge consecutive temporal regions with same classes.

    Args:
        temporal_results: List of {start_sec, end_sec, active_classes, probabilities}
        gap_tolerance: Max gap (seconds) to merge across
        exclude_classes: Set of classes to exclude from results (e.g., {'synth'})

    Returns:
        List of merged regions: {start_sec, end_sec, classes, avg_confidence}
    """
    if not temporal_results:
        return []

    exclude_classes = exclude_classes or set()

    # Filter out excluded classes and empty results
    filtered = []
    for r in temporal_results:
        classes = [c for c in r['active_classes'] if c not in exclude_classes]
        if classes:
            filtered.append({
                'start_sec': r['start_sec'],
                'end_sec': r['end_sec'],
                'classes': set(classes),
                'probs': {c: r['probabilities'].get(c, 0) for c in classes}
            })

    if not filtered:
        return []

    # Sort by start time
    filtered.sort(key=lambda x: x['start_sec'])

    # Merge consecutive regions with same classes
    merged = []
    current = {
        'start_sec': filtered[0]['start_sec'],
        'end_sec': filtered[0]['end_sec'],
        'classes': filtered[0]['classes'],
        'prob_sums': dict(filtered[0]['probs']),
        'count': 1
    }

    for r in filtered[1:]:
        # Check if can merge: same classes and within gap tolerance
        gap = r['start_sec'] - current['end_sec']
        same_classes = r['classes'] == current['classes']

        if same_classes and gap <= gap_tolerance:
            # Extend current region
            current['end_sec'] = r['end_sec']
            for c, p in r['probs'].items():
                current['prob_sums'][c] = current['prob_sums'].get(c, 0) + p
            current['count'] += 1
        else:
            # Save current and start new
            merged.append({
                'start_sec': current['start_sec'],
                'end_sec': current['end_sec'],
                'classes': sorted(current['classes']),
                'avg_confidence': {c: current['prob_sums'][c] / current['count']
                                   for c in current['classes']}
            })
            current = {
                'start_sec': r['start_sec'],
                'end_sec': r['end_sec'],
                'classes': r['classes'],
                'prob_sums': dict(r['probs']),
                'count': 1
            }

    # Don't forget last region
    merged.append({
        'start_sec': current['start_sec'],
        'end_sec': current['end_sec'],
        'classes': sorted(current['classes']),
        'avg_confidence': {c: current['prob_sums'][c] / current['count']
                           for c in current['classes']}
    })

    return merged


def run_temporal(args):
    """Run temporal classification on a latent file."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(OUTPUT_DIR / "model.pt", map_location='cpu', weights_only=False)
    model = MultiLabelClassifier(
        input_dim=checkpoint['input_dim'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    # Run classification
    results = classify_temporal(
        Path(args.input), model, device,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        threshold=args.threshold
    )

    # Print results
    print(f"\nTemporal classification: {args.input}")
    print("="*50)

    for r in results:
        if r['active_classes']:
            classes_str = ', '.join(r['active_classes'])
            print(f"{r['start_sec']:.1f}s - {r['end_sec']:.1f}s: {classes_str}")

    # Summary
    all_detected = set()
    for r in results:
        all_detected.update(r['active_classes'])

    print(f"\nDetected instruments: {sorted(all_detected)}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to: {args.output}")


def run_batch(args):
    """Run temporal classification on all stems from manifest."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    # Filter to target stem type
    stem_filter = args.stem_filter
    all_paths = []
    for stem_type, paths in manifest.get('by_stem', {}).items():
        if stem_filter == 'all' or stem_type == stem_filter:
            all_paths.extend(paths)

    print(f"Found {len(all_paths)} '{stem_filter}' stems to classify")

    # Load model
    checkpoint = torch.load(OUTPUT_DIR / "model.pt", map_location='cpu', weights_only=False)
    model = MultiLabelClassifier(
        input_dim=checkpoint['input_dim'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    classes = checkpoint['classes']
    # Update global CLASSES to match model
    set_classes(classes)
    print(f"Model classes: {classes}")

    # Parse exclude classes
    exclude_classes = set()
    if args.exclude_classes:
        exclude_classes = set(c.strip() for c in args.exclude_classes.split(','))
        print(f"Excluding classes: {exclude_classes}")

    # Process each file
    results = []
    for i, path in enumerate(all_paths):
        try:
            temporal = classify_temporal(
                Path(path), model, device,
                window_sec=args.window_sec,
                hop_sec=args.hop_sec,
                threshold=args.threshold
            )

            # Merge regions if requested
            if args.merge_regions:
                merged = merge_temporal_regions(
                    temporal,
                    gap_tolerance=args.gap_tolerance,
                    exclude_classes=exclude_classes
                )
            else:
                merged = None

            # Aggregate detected classes (excluding filtered ones)
            all_detected = set()
            for r in temporal:
                for c in r['active_classes']:
                    if c not in exclude_classes:
                        all_detected.add(c)

            results.append({
                'path': path,
                'detected': sorted(all_detected),
                'temporal': temporal,
                'merged': merged  # Will be None if not merging
            })

            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(all_paths)}] Processed...")

        except Exception as e:
            results.append({'path': path, 'error': str(e)})

    # Summary
    print(f"\n{'='*50}")
    print(f"Processed {len(results)} files")

    # Count detections
    from collections import Counter
    detection_counts = Counter()
    for r in results:
        for cls in r.get('detected', []):
            detection_counts[cls] += 1

    print("\nDetection counts:")
    for cls, cnt in detection_counts.most_common():
        print(f"  {cls}: {cnt}")

    # Save
    output_path = args.output or (OUTPUT_DIR / "batch_temporal_results.json")
    output_data = {
        'total': len(results),
        'stem_filter': stem_filter,
        'detection_counts': dict(detection_counts),
        'results': results
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Multi-label other stem classifier')
    parser.add_argument('--mode', choices=['train', 'temporal', 'batch'], required=True)

    # Training args
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--synthetic', type=int, default=5000, help='Number of synthetic multi-label samples')
    parser.add_argument('--limit-per-class', type=int, default=2000, help='Max samples per class for training')

    # Inference args
    parser.add_argument('--input', type=str, help='Input latent file or directory')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--manifest', type=str, help='Manifest JSON for batch mode')
    parser.add_argument('--stem-filter', type=str, default='other', help='Stem type to filter (default: other)')
    parser.add_argument('--window-sec', type=float, default=1.0)
    parser.add_argument('--hop-sec', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--exclude-classes', type=str, default='', help='Comma-separated classes to exclude (e.g., synth)')
    parser.add_argument('--gap-tolerance', type=float, default=3.0, help='Max gap (seconds) to merge regions across')
    parser.add_argument('--merge-regions', action='store_true', help='Merge consecutive same-class regions')
    parser.add_argument('--train-classes', type=str, default='', help='Comma-separated classes to train on (default: brass,strings,winds,synth)')

    args = parser.parse_args()

    if args.mode == 'train':
        # Set custom classes if provided
        if args.train_classes:
            classes = [c.strip() for c in args.train_classes.split(',') if c.strip()]
            set_classes(classes)
            print(f"Training on classes: {CLASSES}")
        train(args)
    elif args.mode == 'temporal':
        if not args.input:
            print("Error: --input required for temporal mode")
            return
        run_temporal(args)
    elif args.mode == 'batch':
        if not args.manifest:
            print("Error: --manifest required for batch mode")
            return
        run_batch(args)


if __name__ == "__main__":
    main()
