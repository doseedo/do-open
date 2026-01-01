#!/usr/bin/env python3
"""
Latent Space Instrument Classifier

Uses pre-computed ACE-Step latents for fast, accurate instrument classification.
Much faster than audio-based classification since latents are already extracted.

Usage:
  # Train classifier from manifest (recommended - uses paths with latents)
  python latent_instrument_classifier.py --mode train \
    --manifest /home/arlo/gcs-bucket/Manifests/combined_manifest.json \
    --output-dir /home/arlo/Data/latent_classifier

  # Classify undefined files
  python latent_instrument_classifier.py --mode classify \
    --model /home/arlo/Data/latent_classifier/model.pt \
    --input /home/arlo/undefined_audio_paths.txt \
    --output /home/arlo/Data/latent_classifier/predictions.json

  # Full pipeline
  python latent_instrument_classifier.py --mode full \
    --manifest /home/arlo/gcs-bucket/Manifests/combined_manifest.json \
    --undefined /home/arlo/undefined_audio_paths.txt \
    --output-dir /home/arlo/Data/latent_classifier
"""

import argparse
import json
import pickle
import logging
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ===================== CONFIGURATION =====================

LATENTS_ROOT = Path("/home/arlo/gcs-bucket/Latents")
AUDIO_ROOT = Path("/home/arlo/gcs-bucket")

# Confidence thresholds
CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.65

# Training settings
MAX_SAMPLES_PER_CLASS = 3000
MIN_SAMPLES_PER_CLASS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
HIDDEN_DIM = 256

# Feature extraction - pooling strategies
POOL_METHODS = ['mean', 'std', 'max']  # Results in 8*16*3 = 384 features


# ===================== PATH CONVERSION =====================

def audio_path_to_latent_path(audio_path: str) -> Path:
    """Convert audio file path to corresponding latent path."""
    audio_path = Path(audio_path)

    # Find the relative path from gcs-bucket
    try:
        rel_path = audio_path.relative_to(AUDIO_ROOT)
    except ValueError:
        # If not relative to AUDIO_ROOT, try to find 'protools' in path
        parts = audio_path.parts
        if 'protools' in parts:
            idx = parts.index('protools')
            rel_path = Path(*parts[idx:])
        elif 'protoolsA' in parts:
            idx = parts.index('protoolsA')
            rel_path = Path(*parts[idx:])
        else:
            # Just use the full path after gcs-bucket if present
            if 'gcs-bucket' in parts:
                idx = parts.index('gcs-bucket')
                rel_path = Path(*parts[idx+1:])
            else:
                rel_path = audio_path

    # Change extension to .pt
    latent_path = LATENTS_ROOT / rel_path.with_suffix('.pt')
    return latent_path


def load_latent(latent_path: Path) -> Optional[torch.Tensor]:
    """Load latent tensor from file."""
    try:
        data = torch.load(latent_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            return data['latents']  # [8, 16, T]
        return data
    except Exception as e:
        return None


def pool_latent(latent: torch.Tensor) -> torch.Tensor:
    """Pool latent [8, 16, T] to fixed-size feature vector."""
    # latent shape: [8, 16, T] - 8 codebook groups, 16 channels, T time steps
    features = []

    if 'mean' in POOL_METHODS:
        features.append(latent.mean(dim=-1))  # [8, 16]
    if 'std' in POOL_METHODS:
        features.append(latent.std(dim=-1))   # [8, 16]
    if 'max' in POOL_METHODS:
        features.append(latent.max(dim=-1)[0])  # [8, 16]
    if 'min' in POOL_METHODS:
        features.append(latent.min(dim=-1)[0])  # [8, 16]

    # Concatenate and flatten: [8, 16, num_pools] -> [8*16*num_pools]
    stacked = torch.stack(features, dim=-1)  # [8, 16, num_pools]
    return stacked.flatten()


# ===================== DATASET =====================

class LatentDataset(Dataset):
    """Dataset for loading latent features."""

    def __init__(self, audio_paths: List[str], labels: List[int],
                 cache: Optional[Dict] = None):
        self.audio_paths = audio_paths
        self.labels = labels
        self.cache = cache if cache is not None else {}

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        # Check cache first
        if audio_path in self.cache:
            features = self.cache[audio_path]
        else:
            latent_path = audio_path_to_latent_path(audio_path)
            latent = load_latent(latent_path)

            if latent is None:
                # Return zeros if latent not found
                features = torch.zeros(8 * 16 * len(POOL_METHODS))
            else:
                features = pool_latent(latent)

            self.cache[audio_path] = features

        return features, label


# ===================== MODEL =====================

class InstrumentClassifier(nn.Module):
    """MLP classifier for instrument classification."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = HIDDEN_DIM):
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

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ===================== FEATURE EXTRACTION =====================

def extract_features_batch(audio_paths: List[str],
                          batch_size: int = 100,
                          num_workers: int = 8) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Extract pooled features from latents in parallel."""

    features_list = []
    valid_paths = []
    failed_paths = []

    total = len(audio_paths)
    logging.info(f"Extracting features from {total} files...")
    start_time = datetime.now()

    def process_one(audio_path: str) -> Tuple[Optional[torch.Tensor], str]:
        latent_path = audio_path_to_latent_path(audio_path)
        latent = load_latent(latent_path)
        if latent is not None:
            return pool_latent(latent), audio_path
        return None, audio_path

    # Process in parallel using threads (I/O bound)
    processed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, p): p for p in audio_paths}

        for future in as_completed(futures):
            features, path = future.result()
            if features is not None:
                features_list.append(features)
                valid_paths.append(path)
            else:
                failed_paths.append(path)

            processed += 1
            if processed % 500 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = processed / elapsed if elapsed > 0 else 0
                logging.info(f"  {processed}/{total} ({rate:.1f}/s)")

    elapsed = (datetime.now() - start_time).total_seconds()
    logging.info(f"  Complete: {len(valid_paths)} success, {len(failed_paths)} failed ({elapsed:.1f}s)")

    if len(features_list) == 0:
        return torch.tensor([]), [], failed_paths

    return torch.stack(features_list), valid_paths, failed_paths


# ===================== TRAINING =====================

def train_classifier(manifest_path: Path, output_dir: Path,
                    num_workers: int = 8, device: str = 'cuda') -> Dict:
    """Train the latent-space instrument classifier using manifest labels."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    device = torch.device(device)
    logging.info(f"Using device: {device}")

    # Load manifest
    logging.info(f"Loading manifest from {manifest_path}...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Group by label (using 'group' field from manifest)
    # Only include paths that actually have latents (verify existence)
    file_label_pairs = []
    class_counts = Counter()
    skipped_undefined = 0
    skipped_no_latent = 0

    logging.info("Checking latent existence for manifest paths...")
    total_checked = 0
    for audio_path, meta in manifest.items():
        total_checked += 1
        if total_checked % 50000 == 0:
            logging.info(f"  Checked {total_checked}/{len(manifest)}...")

        # Get group label
        if isinstance(meta, dict):
            label = meta.get('group', 'undefined')
        else:
            label = 'undefined'

        if label == 'undefined' or not label:
            skipped_undefined += 1
            continue

        # Quick filter: latents only exist for /New/ paths, not /Prev/
        if '/New/' not in audio_path:
            skipped_no_latent += 1
            continue

        # Verify latent actually exists
        latent_path = audio_path_to_latent_path(audio_path)
        if not latent_path.exists():
            skipped_no_latent += 1
            continue

        file_label_pairs.append((audio_path, label))
        class_counts[label] += 1

    logging.info(f"Loaded {len(file_label_pairs)} labeled files with verified latents")
    logging.info(f"  Skipped {skipped_undefined} undefined, {skipped_no_latent} without latents")

    # Apply class limits
    import random
    random.seed(42)

    filtered_pairs = []
    class_samples = defaultdict(list)

    for path, label in file_label_pairs:
        class_samples[label].append(path)

    for label, paths in class_samples.items():
        if len(paths) < MIN_SAMPLES_PER_CLASS:
            logging.warning(f"  Skipping {label}: only {len(paths)} samples")
            continue

        if len(paths) > MAX_SAMPLES_PER_CLASS:
            paths = random.sample(paths, MAX_SAMPLES_PER_CLASS)

        for path in paths:
            filtered_pairs.append((path, label))

    file_label_pairs = filtered_pairs
    class_counts = Counter(label for _, label in file_label_pairs)

    logging.info(f"Loaded {len(file_label_pairs)} files across {len(class_counts)} classes:")
    for label, count in class_counts.most_common():
        logging.info(f"  {label}: {count}")

    # Extract features
    audio_paths = [p for p, _ in file_label_pairs]
    labels = [l for _, l in file_label_pairs]

    X, valid_paths, failed = extract_features_batch(audio_paths, num_workers=num_workers)

    if len(X) < 100:
        raise ValueError(f"Not enough samples: {len(X)}")

    # Filter labels to match valid paths
    path_to_label = dict(file_label_pairs)
    valid_labels = [path_to_label[p] for p in valid_paths]

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(valid_labels)
    y = torch.tensor(y, dtype=torch.long)

    num_classes = len(label_encoder.classes_)
    input_dim = X.shape[1]

    logging.info(f"Feature dim: {input_dim}, Classes: {num_classes}")

    # Train/val/test split
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.15,
                                            stratify=y.numpy(), random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1,
                                           stratify=y[train_idx].numpy(), random_state=42)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Normalize features
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Compute class weights for imbalanced data
    class_counts_arr = np.bincount(y_train.numpy())
    class_weights = 1.0 / (class_counts_arr + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Initialize model
    model = InstrumentClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    logging.info(f"\nTraining for {NUM_EPOCHS} epochs...")
    best_val_acc = 0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            train_total += len(batch_y)

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                val_total += len(batch_y)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"  Epoch {epoch+1}/{NUM_EPOCHS}: "
                        f"train_loss={train_loss/train_total:.4f}, "
                        f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluate on test set
    model.eval()
    X_test_dev = X_test.to(device)

    with torch.no_grad():
        logits = model(X_test_dev)
        probs = F.softmax(logits, dim=1)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_proba = probs.cpu().numpy()

    y_test_np = y_test.numpy()

    logging.info("\n" + "=" * 60)
    logging.info("CLASSIFICATION REPORT")
    logging.info("=" * 60)
    report = classification_report(y_test_np, y_pred,
                                   target_names=label_encoder.classes_,
                                   output_dict=True)
    logging.info("\n" + classification_report(y_test_np, y_pred,
                                              target_names=label_encoder.classes_))

    # High-confidence accuracy
    max_proba = np.max(y_proba, axis=1)
    for threshold in [0.9, 0.8, 0.7]:
        mask = max_proba >= threshold
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == y_test_np[mask])
            logging.info(f"Accuracy at {threshold:.0%} confidence: {acc:.1%} ({np.sum(mask)} samples)")

    # Save model
    model_data = {
        'model_state': best_model_state,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'hidden_dim': HIDDEN_DIM,
        'mean': mean,
        'std': std,
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'pool_methods': POOL_METHODS,
        'training_stats': {
            'total_samples': len(X),
            'failed_samples': len(failed),
            'test_accuracy': report['accuracy'],
            'best_val_accuracy': best_val_acc,
            'class_counts': dict(Counter(valid_labels))
        },
        'trained_at': datetime.now().isoformat()
    }

    model_path = output_dir / 'model.pt'
    torch.save(model_data, model_path)
    logging.info(f"\nModel saved to {model_path}")

    # Also save label encoder separately for compatibility
    with open(output_dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    return model_data


# ===================== CLASSIFICATION =====================

def classify_undefined(model_path: Path, undefined_paths: List[str],
                       output_path: Path, num_workers: int = 8,
                       device: str = 'cuda',
                       manifest_path: Path = None) -> Dict:
    """Classify undefined files using trained model.

    If manifest_path is provided, extract undefined paths from manifest directly.
    """

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    # If manifest provided, extract undefined paths with latents
    if manifest_path and manifest_path.exists():
        logging.info(f"Loading undefined paths from manifest: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)

        undefined_paths = []
        skipped_no_latent = 0
        for path, meta in manifest.items():
            if not isinstance(meta, dict):
                continue
            if meta.get('group') != 'undefined':
                continue
            if '/New/' not in path:
                continue
            # Check latent exists
            latent_path = audio_path_to_latent_path(path)
            if latent_path.exists():
                undefined_paths.append(path)
            else:
                skipped_no_latent += 1

        logging.info(f"Found {len(undefined_paths)} undefined paths with latents")
        logging.info(f"Skipped {skipped_no_latent} without latents")

    # Load model
    logging.info(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)

    input_dim = model_data['input_dim']
    num_classes = model_data['num_classes']
    hidden_dim = model_data.get('hidden_dim', HIDDEN_DIM)
    mean = model_data['mean']
    std = model_data['std']
    classes = model_data['label_encoder_classes']

    model = InstrumentClassifier(input_dim, num_classes, hidden_dim)
    model.load_state_dict(model_data['model_state'])
    model.to(device)
    model.eval()

    logging.info(f"Model classes: {classes}")

    if len(undefined_paths) == 0:
        logging.error("No undefined paths to classify!")
        return {'predictions': [], 'failed_paths': []}

    # Extract features
    X, valid_paths, failed = extract_features_batch(undefined_paths, num_workers=num_workers)

    if len(X) == 0:
        logging.error("No features extracted!")
        return {}

    # Normalize
    X = (X - mean) / std

    # Predict in batches
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            batch = X[i:i+BATCH_SIZE].to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    y_pred = all_probs.argmax(axis=1)
    max_proba = all_probs.max(axis=1)

    # Decode predictions
    y_pred_labels = [classes[i] for i in y_pred]

    # Categorize by confidence
    predictions = {
        'high_confidence': [],
        'medium_confidence': [],
        'low_confidence': []
    }

    results = []
    for i, (path, label, conf) in enumerate(zip(valid_paths, y_pred_labels, max_proba)):
        result = {
            'path': path,
            'predicted_group': label,
            'confidence': float(conf),
            'all_probabilities': {c: float(p) for c, p in zip(classes, all_probs[i])}
        }
        results.append(result)

        if conf >= CONFIDENCE_HIGH:
            predictions['high_confidence'].append(result)
        elif conf >= CONFIDENCE_MEDIUM:
            predictions['medium_confidence'].append(result)
        else:
            predictions['low_confidence'].append(result)

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("CLASSIFICATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"Total classified: {len(results)}")
    logging.info(f"  High confidence (>={CONFIDENCE_HIGH:.0%}): {len(predictions['high_confidence'])}")
    logging.info(f"  Medium confidence ({CONFIDENCE_MEDIUM:.0%}-{CONFIDENCE_HIGH:.0%}): {len(predictions['medium_confidence'])}")
    logging.info(f"  Low confidence (<{CONFIDENCE_MEDIUM:.0%}): {len(predictions['low_confidence'])}")
    logging.info(f"  Failed to load latent: {len(failed)}")

    # Predictions by class
    high_by_class = Counter(r['predicted_group'] for r in predictions['high_confidence'])
    logging.info("\nHigh-confidence predictions by class:")
    for cls, count in high_by_class.most_common():
        logging.info(f"  {cls}: {count}")

    # Save results
    output_data = {
        'predictions': results,
        'failed_paths': failed,
        'summary': {
            'total': len(results),
            'high_confidence': len(predictions['high_confidence']),
            'medium_confidence': len(predictions['medium_confidence']),
            'low_confidence': len(predictions['low_confidence']),
            'failed': len(failed),
            'by_class': dict(Counter(r['predicted_group'] for r in results))
        },
        'thresholds': {
            'high': CONFIDENCE_HIGH,
            'medium': CONFIDENCE_MEDIUM
        },
        'classified_at': datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"\nPredictions saved to {output_path}")

    return output_data


def update_manifest(manifest_path: Path, predictions_path: Path,
                    min_confidence: float = CONFIDENCE_MEDIUM) -> int:
    """Update manifest with predictions above confidence threshold."""

    logging.info(f"Updating manifest with predictions (min confidence: {min_confidence:.0%})...")

    with open(manifest_path) as f:
        manifest = json.load(f)

    with open(predictions_path) as f:
        pred_data = json.load(f)

    updated = 0
    for pred in pred_data['predictions']:
        if pred['confidence'] >= min_confidence:
            path = pred['path']
            group = pred['predicted_group']

            manifest[path] = {
                'group': group,
                'subgroup': 'undefined',
                'filename': Path(path).name,
                'classifier_confidence': pred['confidence'],
                'labeling_method': 'latent_classifier'
            }
            updated += 1

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logging.info(f"Updated {updated} entries in manifest")
    return updated


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(
        description='Latent-space instrument classifier (GPU-accelerated)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--mode', choices=['train', 'classify', 'full'], required=True)
    parser.add_argument('--manifest', type=str,
                        help='Manifest JSON with audio paths and group labels')
    parser.add_argument('--undefined', type=str,
                        help='File with undefined audio paths to classify')
    parser.add_argument('--model', type=str,
                        help='Path to trained model')
    parser.add_argument('--output-manifest', type=str,
                        help='Manifest to update with predictions')
    parser.add_argument('--output-dir', type=str, default='./latent_classifier_output')
    parser.add_argument('--workers', type=int, default=12,
                        help='Parallel workers for loading latents')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--min-confidence', type=float, default=CONFIDENCE_MEDIUM)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'train':
        if not args.manifest:
            parser.error("--manifest required for train mode")
        train_classifier(
            Path(args.manifest),
            output_dir,
            num_workers=args.workers,
            device=args.device
        )

    elif args.mode == 'classify':
        if not args.model:
            parser.error("--model required for classify mode")
        if not args.undefined and not args.manifest:
            parser.error("--undefined or --manifest required for classify mode")

        # Load undefined paths from file or use manifest directly
        undefined_paths = []
        manifest_path = None

        if args.undefined:
            with open(args.undefined) as f:
                undefined_paths = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        elif args.manifest:
            # Pass manifest path to classify_undefined, it will extract undefined paths
            manifest_path = Path(args.manifest)

        predictions_path = output_dir / 'predictions.json'
        classify_undefined(
            Path(args.model),
            undefined_paths,
            predictions_path,
            num_workers=args.workers,
            device=args.device,
            manifest_path=manifest_path
        )

        if args.output_manifest:
            update_manifest(Path(args.output_manifest), predictions_path, args.min_confidence)

    elif args.mode == 'full':
        if not args.manifest or not args.undefined:
            parser.error("--manifest and --undefined required")

        logging.info("=" * 60)
        logging.info("PHASE 1: TRAINING")
        logging.info("=" * 60)
        train_classifier(
            Path(args.manifest),
            output_dir,
            num_workers=args.workers,
            device=args.device
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info("\n" + "=" * 60)
        logging.info("PHASE 2: CLASSIFICATION")
        logging.info("=" * 60)
        with open(args.undefined) as f:
            undefined_paths = [l.strip() for l in f if l.strip() and not l.startswith('#')]

        predictions_path = output_dir / 'predictions.json'
        classify_undefined(
            output_dir / 'model.pt',
            undefined_paths,
            predictions_path,
            num_workers=args.workers,
            device=args.device
        )

        if args.output_manifest:
            logging.info("\n" + "=" * 60)
            logging.info("PHASE 3: UPDATE MANIFEST")
            logging.info("=" * 60)
            update_manifest(Path(args.output_manifest), predictions_path, args.min_confidence)


if __name__ == '__main__':
    main()
