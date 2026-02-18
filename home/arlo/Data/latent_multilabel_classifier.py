#!/usr/bin/env python3
"""
Multi-label Temporal Instrument Classifier

Detects multiple instruments simultaneously and tracks changes over time.
Uses sigmoid activation per class (not softmax) to allow multiple labels.

Key differences from single-label classifier:
- Output: sigmoid per class (independent probabilities)
- Loss: BCEWithLogitsLoss (binary cross-entropy per class)
- Prediction: threshold each class (default 0.5)
- Training: works with both single-label and multi-label data

Usage:
  # Train from manifest + corrections (uses multi-label data where available)
  python latent_multilabel_classifier.py --mode train \
    --manifest /home/arlo/gcs-bucket/Manifests/unified_manifest.json \
    --corrections /home/arlo/gcs-bucket/Manifests/corrections.json \
    --output-dir /home/arlo/Data/multilabel_classifier

  # Classify files with multi-label + temporal output
  python latent_multilabel_classifier.py --mode classify \
    --model /home/arlo/Data/multilabel_classifier/model.pt \
    --manifest /home/arlo/gcs-bucket/Manifests/unified_manifest.json \
    --output-dir /home/arlo/Data/multilabel_classifier

  # Temporal analysis on specific files
  python latent_multilabel_classifier.py --mode temporal \
    --model /home/arlo/Data/multilabel_classifier/model.pt \
    --manifest /home/arlo/gcs-bucket/Manifests/unified_manifest.json \
    --output-dir /home/arlo/Data/multilabel_classifier
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

# ===================== CONFIGURATION =====================

LATENTS_ROOT = Path("/home/arlo/gcs-bucket/Latents")
AUDIO_ROOT = Path("/home/arlo/gcs-bucket")

# Training settings
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
HIDDEN_DIM = 256

# Feature pooling
POOL_METHODS = ['mean', 'std', 'max']

# Multi-label settings
LABEL_THRESHOLD = 0.5  # Threshold for positive prediction
MIN_SAMPLES_PER_CLASS = 50
MAX_SAMPLES_PER_CLASS = 15000

# Excluded from classification (meta-labels, not instruments)
EXCLUDED_CLASSES = {'undefined', 'room', 'click', 'silent', 'junk', 'fx', 'ensemble', 'full-track', 'review_vocals', 'dialogue'}

# Merge small/variant groups into main groups
GROUP_MERGES = {
    'plucked': 'guitar',
    'drums_roomy': 'drums',
    'e-drums': 'drums',
    'e_drums': 'drums',
}

def normalize_group(group: str) -> str:
    """Normalize group name by applying merges."""
    return GROUP_MERGES.get(group, group)

# Temporal settings
LATENT_FRAMES_PER_SEC = 44100 / 512  # ~86.13 frames/sec
DEFAULT_WINDOW_SEC = 2.0
DEFAULT_HOP_SEC = 1.0


# ===================== PATH UTILITIES =====================

def audio_path_to_latent_path(audio_path: str) -> Optional[Path]:
    """Convert audio path to latent path."""
    audio_path = Path(audio_path)

    try:
        rel_path = audio_path.relative_to(AUDIO_ROOT)
    except ValueError:
        parts = audio_path.parts
        for marker in ['protools', 'protoolsA']:
            if marker in parts:
                idx = parts.index(marker)
                rel_path = Path(*parts[idx:])
                break
        else:
            if 'gcs-bucket' in parts:
                idx = parts.index('gcs-bucket')
                rel_path = Path(*parts[idx+1:])
            else:
                rel_path = audio_path

    stem = rel_path.with_suffix('')
    dcae_path = LATENTS_ROOT / f"{stem}.dcae.pt"
    pt_path = LATENTS_ROOT / f"{stem}.pt"

    if dcae_path.exists():
        return dcae_path
    elif pt_path.exists():
        return pt_path
    return None


def load_latent(latent_path: Path) -> Optional[torch.Tensor]:
    """Load latent tensor."""
    try:
        data = torch.load(latent_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            return data['latents']
        return data
    except Exception:
        return None


def pool_latent(latent: torch.Tensor) -> torch.Tensor:
    """Pool latent [8, 16, T] to fixed-size features."""
    features = []

    if 'mean' in POOL_METHODS:
        features.append(latent.mean(dim=-1))
    if 'std' in POOL_METHODS:
        features.append(latent.std(dim=-1))
    if 'max' in POOL_METHODS:
        features.append(latent.max(dim=-1)[0])

    stacked = torch.stack(features, dim=-1)
    return stacked.flatten()


def window_latent(latent: torch.Tensor, window_sec: float = DEFAULT_WINDOW_SEC,
                  hop_sec: float = DEFAULT_HOP_SEC) -> List[Tuple[float, float, torch.Tensor]]:
    """Split latent into overlapping time windows."""
    T = latent.shape[-1]
    window_frames = int(window_sec * LATENT_FRAMES_PER_SEC)
    hop_frames = int(hop_sec * LATENT_FRAMES_PER_SEC)

    window_frames = max(window_frames, 32)
    hop_frames = max(hop_frames, 16)

    windows = []
    for start in range(0, T - window_frames + 1, hop_frames):
        end = start + window_frames
        window = latent[:, :, start:end]
        start_sec = start / LATENT_FRAMES_PER_SEC
        end_sec = end / LATENT_FRAMES_PER_SEC
        windows.append((start_sec, end_sec, window))

    if len(windows) == 0 and T > 0:
        windows.append((0, T / LATENT_FRAMES_PER_SEC, latent))

    return windows


# ===================== MODEL =====================

class MultiLabelClassifier(nn.Module):
    """Multi-label MLP classifier with sigmoid output per class."""

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

            nn.Linear(hidden_dim // 2, num_classes),
            # No activation - BCEWithLogitsLoss applies sigmoid
        )

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        """Get probabilities (applies sigmoid)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x, threshold=LABEL_THRESHOLD):
        """Get binary predictions."""
        proba = self.predict_proba(x)
        return (proba >= threshold).float()


# ===================== DATA LOADING =====================

def load_training_data(
    manifest_path: Path,
    corrections_path: Path,
    num_workers: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """
    Load training data from manifest and corrections.

    Returns:
        X: feature tensor [N, feature_dim]
        Y: multi-hot label tensor [N, num_classes]
        valid_paths: list of audio paths
        classes: list of class names
    """
    logging.info(f"Loading manifest from {manifest_path}...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    logging.info(f"Loading corrections from {corrections_path}...")
    corrections = {}
    if corrections_path.exists():
        with open(corrections_path) as f:
            corrections = json.load(f)

    # Collect all labels to build class list
    all_labels = set()
    entries = manifest.get('entries', [])

    for entry in entries:
        group = normalize_group(entry.get('group', 'undefined'))
        if group not in EXCLUDED_CLASSES:
            all_labels.add(group)

    # Also add labels from multi-label corrections
    for corr in corrections.values():
        if corr.get('multi_label') and corr.get('regions'):
            for region in corr['regions']:
                for label in region.get('labels', []):
                    label = normalize_group(label)
                    if label not in EXCLUDED_CLASSES:
                        all_labels.add(label)
        elif corr.get('group'):
            group = normalize_group(corr['group'])
            if group not in EXCLUDED_CLASSES:
                all_labels.add(group)

    classes = sorted(all_labels)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    logging.info(f"Found {num_classes} classes: {classes}")

    # Build training samples
    # Format: (audio_path, labels_list, start_sec, end_sec)
    # For single-label entries: labels_list has one item, start/end are None (whole file)
    # For multi-label regions: labels_list has multiple items, start/end define the region
    samples = []

    # Process manifest entries (single-label)
    for entry in entries:
        audio_path = entry.get('audio_path', '')
        if not entry.get('has_latent'):
            continue

        group = normalize_group(entry.get('group', 'undefined'))
        if group in EXCLUDED_CLASSES:
            continue

        # Check if this path has a correction
        if audio_path in corrections:
            corr = corrections[audio_path]
            if corr.get('multi_label') and corr.get('regions'):
                # Use multi-label regions instead
                for region in corr['regions']:
                    labels = [normalize_group(l) for l in region.get('labels', []) if normalize_group(l) not in EXCLUDED_CLASSES]
                    if labels:
                        samples.append((audio_path, labels, region.get('start'), region.get('end')))
            elif corr.get('group'):
                # Use corrected single label
                corrected_group = normalize_group(corr['group'])
                if corrected_group not in EXCLUDED_CLASSES:
                    samples.append((audio_path, [corrected_group], None, None))
        else:
            # Use manifest label
            samples.append((audio_path, [group], None, None))

    logging.info(f"Collected {len(samples)} training samples")

    # Count samples per class (labels already normalized)
    class_counts = Counter()
    for _, labels, _, _ in samples:
        for label in labels:
            class_counts[label] += 1

    # Filter out classes with too few samples
    valid_classes = {cls for cls, count in class_counts.items() if count >= MIN_SAMPLES_PER_CLASS}
    removed_classes = set(classes) - valid_classes
    if removed_classes:
        logging.info(f"Removing classes with < {MIN_SAMPLES_PER_CLASS} samples: {removed_classes}")
        classes = sorted(valid_classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        num_classes = len(classes)
        # Filter samples to only include valid classes
        samples = [(p, [l for l in labels if l in valid_classes], s, e)
                   for p, labels, s, e in samples if any(l in valid_classes for l in labels)]
        logging.info(f"After filtering: {len(samples)} samples, {num_classes} classes")

    logging.info("Final class distribution:")
    for cls, count in class_counts.most_common():
        if cls in valid_classes:
            logging.info(f"  {cls}: {count}")

    # Balance classes (limit overrepresented, but keep multi-label samples)
    random.seed(42)
    balanced_samples = []
    class_sample_counts = defaultdict(int)

    # First, add all multi-label samples (they're rare and valuable)
    multilabel_samples = [s for s in samples if len(s[1]) > 1]
    balanced_samples.extend(multilabel_samples)
    for _, labels, _, _ in multilabel_samples:
        for label in labels:
            class_sample_counts[label] += 1

    # Then add single-label samples up to limit
    single_label_samples = [s for s in samples if len(s[1]) == 1]
    random.shuffle(single_label_samples)

    for sample in single_label_samples:
        label = sample[1][0]
        if class_sample_counts[label] < MAX_SAMPLES_PER_CLASS:
            balanced_samples.append(sample)
            class_sample_counts[label] += 1

    samples = balanced_samples
    logging.info(f"After balancing: {len(samples)} samples")
    logging.info(f"  Multi-label samples: {len(multilabel_samples)}")

    # Extract features
    def process_sample(sample):
        audio_path, labels, start_sec, end_sec = sample
        latent_path = audio_path_to_latent_path(audio_path)
        if latent_path is None:
            return None

        latent = load_latent(latent_path)
        if latent is None:
            return None

        # If region specified, extract that portion
        if start_sec is not None and end_sec is not None:
            start_frame = int(start_sec * LATENT_FRAMES_PER_SEC)
            end_frame = int(end_sec * LATENT_FRAMES_PER_SEC)
            if end_frame > latent.shape[-1]:
                end_frame = latent.shape[-1]
            if end_frame - start_frame < 32:
                # Too short, use whole file
                pass
            else:
                latent = latent[:, :, start_frame:end_frame]

        features = pool_latent(latent)

        # Create multi-hot label vector
        label_vec = torch.zeros(num_classes)
        for label in labels:
            if label in class_to_idx:
                label_vec[class_to_idx[label]] = 1.0

        return features, label_vec, audio_path

    logging.info("Extracting features...")
    features_list = []
    labels_list = []
    valid_paths = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_sample, s): s for s in samples}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                features, label_vec, path = result
                features_list.append(features)
                labels_list.append(label_vec)
                valid_paths.append(path)

            if (i + 1) % 1000 == 0:
                logging.info(f"  Processed {i+1}/{len(samples)}...")

    logging.info(f"Extracted {len(features_list)} features")

    X = torch.stack(features_list)
    Y = torch.stack(labels_list)

    return X, Y, valid_paths, classes


# ===================== TRAINING =====================

def train_classifier(
    manifest_path: Path,
    corrections_path: Path,
    output_dir: Path,
    num_workers: int = 8,
    device: str = 'cuda'
) -> Dict:
    """Train the multi-label classifier."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)
    logging.info(f"Using device: {device}")

    # Load data
    X, Y, paths, classes = load_training_data(manifest_path, corrections_path, num_workers)

    if len(X) < 100:
        raise ValueError(f"Not enough training data: {len(X)} samples")

    num_classes = len(classes)
    input_dim = X.shape[1]

    logging.info(f"Feature dim: {input_dim}, Classes: {num_classes}")

    # Normalize features
    mean = X.mean(dim=0)
    std = X.std(dim=0) + 1e-8
    X_norm = (X - mean) / std

    # Train/test split
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)

    X_train, Y_train = X_norm[train_idx], Y[train_idx]
    X_test, Y_test = X_norm[test_idx], Y[test_idx]

    # Calculate pos_weight for class imbalance
    # pos_weight[i] = num_negative / num_positive for class i
    pos_counts = Y_train.sum(dim=0)
    neg_counts = len(Y_train) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=50.0)  # Limit extreme weights
    pos_weight = pos_weight.to(device)

    logging.info(f"Pos weights (min/max): {pos_weight.min():.1f} / {pos_weight.max():.1f}")

    # DataLoader
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = MultiLabelClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop
    logging.info(f"Training for {NUM_EPOCHS} epochs...")
    best_f1 = 0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_y)

        scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test.to(device))
            test_probs = torch.sigmoid(test_logits).cpu()
            test_preds = (test_probs >= LABEL_THRESHOLD).float()

            # Calculate metrics
            # Exact match accuracy (all labels correct)
            exact_match = (test_preds == Y_test).all(dim=1).float().mean().item()

            # Per-sample F1
            tp = (test_preds * Y_test).sum(dim=1)
            fp = (test_preds * (1 - Y_test)).sum(dim=1)
            fn = ((1 - test_preds) * Y_test).sum(dim=1)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            mean_f1 = f1.mean().item()

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"  Epoch {epoch+1}/{NUM_EPOCHS}: loss={train_loss/len(train_dataset):.4f}, "
                        f"exact_match={exact_match:.3f}, mean_f1={mean_f1:.3f}")

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_probs >= LABEL_THRESHOLD).astype(int)
        Y_test_np = Y_test.numpy().astype(int)

    # Per-class metrics
    logging.info("\n" + "=" * 60)
    logging.info("MULTI-LABEL CLASSIFICATION RESULTS")
    logging.info("=" * 60)

    # Calculate per-class precision/recall/f1
    class_metrics = []
    for i, cls in enumerate(classes):
        tp = ((test_preds[:, i] == 1) & (Y_test_np[:, i] == 1)).sum()
        fp = ((test_preds[:, i] == 1) & (Y_test_np[:, i] == 0)).sum()
        fn = ((test_preds[:, i] == 0) & (Y_test_np[:, i] == 1)).sum()
        tn = ((test_preds[:, i] == 0) & (Y_test_np[:, i] == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = int(Y_test_np[:, i].sum())

        class_metrics.append({
            'class': cls,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        })

    # Sort by F1
    class_metrics.sort(key=lambda x: -x['f1'])

    logging.info("\nPer-class metrics (sorted by F1):")
    logging.info(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    logging.info("-" * 55)
    for m in class_metrics:
        logging.info(f"{m['class']:<15} {m['precision']:>10.3f} {m['recall']:>10.3f} "
                    f"{m['f1']:>10.3f} {m['support']:>10}")

    # Multi-label specific metrics
    # Hamming loss (fraction of wrong labels)
    hamming = (test_preds != Y_test_np).mean()

    # Exact match ratio
    exact_match = (test_preds == Y_test_np).all(axis=1).mean()

    logging.info(f"\nOverall metrics:")
    logging.info(f"  Hamming loss: {hamming:.4f}")
    logging.info(f"  Exact match ratio: {exact_match:.4f}")
    logging.info(f"  Best mean F1: {best_f1:.4f}")

    # Save model
    model_data = {
        'model_state': best_state,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'hidden_dim': HIDDEN_DIM,
        'mean': mean,
        'std': std,
        'classes': classes,
        'class_to_idx': {c: i for i, c in enumerate(classes)},
        'pool_methods': POOL_METHODS,
        'label_threshold': LABEL_THRESHOLD,
        'training_stats': {
            'num_samples': len(X),
            'num_multilabel': int((Y.sum(dim=1) > 1).sum()),
            'best_f1': best_f1,
            'exact_match': exact_match,
            'hamming_loss': hamming,
        },
        'class_metrics': class_metrics,
        'trained_at': datetime.now().isoformat()
    }

    model_path = output_dir / 'model.pt'
    torch.save(model_data, model_path)
    logging.info(f"\nModel saved to {model_path}")

    return model_data


# ===================== CLASSIFICATION =====================

def classify_files(
    model_path: Path,
    audio_paths: List[str],
    output_path: Path,
    threshold: float = LABEL_THRESHOLD,
    num_workers: int = 8,
    device: str = 'cuda'
) -> Dict:
    """Classify files with multi-label output."""

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device_obj = torch.device(device)

    # Load model
    logging.info(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)

    input_dim = model_data['input_dim']
    num_classes = model_data['num_classes']
    hidden_dim = model_data.get('hidden_dim', HIDDEN_DIM)
    mean = model_data['mean']
    std = model_data['std']
    classes = model_data['classes']

    model = MultiLabelClassifier(input_dim, num_classes, hidden_dim)
    model.load_state_dict(model_data['model_state'])
    model.to(device_obj)
    model.eval()

    logging.info(f"Classifying {len(audio_paths)} files...")

    # Extract features
    def process_path(path):
        latent_path = audio_path_to_latent_path(path)
        if latent_path is None:
            return path, None
        latent = load_latent(latent_path)
        if latent is None:
            return path, None
        return path, pool_latent(latent)

    features_list = []
    valid_paths = []
    failed = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_path, p) for p in audio_paths]
        for i, future in enumerate(as_completed(futures)):
            path, features = future.result()
            if features is not None:
                features_list.append(features)
                valid_paths.append(path)
            else:
                failed.append(path)
            if (i + 1) % 1000 == 0:
                logging.info(f"  Processed {i+1}/{len(audio_paths)}...")

    if len(features_list) == 0:
        return {'results': [], 'failed': failed}

    # Predict
    X = torch.stack(features_list)
    X_norm = (X - mean) / std

    with torch.no_grad():
        logits = model(X_norm.to(device_obj))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)

    # Build results
    results = []
    for i, path in enumerate(valid_paths):
        pred_labels = [classes[j] for j in range(num_classes) if preds[i, j] == 1]
        probs_dict = {classes[j]: float(probs[i, j]) for j in range(num_classes)}

        # Sort by probability
        top_probs = sorted(probs_dict.items(), key=lambda x: -x[1])[:5]

        result = {
            'path': path,
            'filename': Path(path).name,
            'predicted_labels': pred_labels,
            'num_labels': len(pred_labels),
            'is_multilabel': len(pred_labels) > 1,
            'top_probabilities': dict(top_probs),
            'all_probabilities': probs_dict,
        }
        results.append(result)

    # Summary
    multilabel_count = sum(1 for r in results if r['is_multilabel'])
    logging.info(f"\nClassified {len(results)} files:")
    logging.info(f"  Single-label: {len(results) - multilabel_count}")
    logging.info(f"  Multi-label: {multilabel_count}")

    # Label distribution
    label_counts = Counter()
    for r in results:
        for label in r['predicted_labels']:
            label_counts[label] += 1

    logging.info("\nLabel distribution:")
    for label, count in label_counts.most_common(15):
        logging.info(f"  {label}: {count}")

    # Save results
    output_data = {
        'threshold': threshold,
        'total': len(results),
        'multilabel_count': multilabel_count,
        'label_distribution': dict(label_counts),
        'results': results,
        'failed': failed,
        'classified_at': datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"\nResults saved to {output_path}")

    return output_data


# ===================== TEMPORAL CLASSIFICATION =====================

def classify_temporal(
    model_path: Path,
    audio_paths: List[str],
    output_path: Path,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    threshold: float = LABEL_THRESHOLD,
    num_workers: int = 8,
    device: str = 'cuda'
) -> Dict:
    """Classify files with temporal windowing - detect instrument changes over time."""

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device_obj = torch.device(device)

    # Load model
    logging.info(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)

    input_dim = model_data['input_dim']
    num_classes = model_data['num_classes']
    hidden_dim = model_data.get('hidden_dim', HIDDEN_DIM)
    mean = model_data['mean']
    std = model_data['std']
    classes = model_data['classes']

    model = MultiLabelClassifier(input_dim, num_classes, hidden_dim)
    model.load_state_dict(model_data['model_state'])
    model.to(device_obj)
    model.eval()

    logging.info(f"Processing {len(audio_paths)} files with temporal analysis...")
    logging.info(f"  Window: {window_sec}s, Hop: {hop_sec}s, Threshold: {threshold}")

    results = []
    temporal_changes = []
    failed = []

    for i, audio_path in enumerate(audio_paths):
        if (i + 1) % 100 == 0:
            logging.info(f"  Processed {i+1}/{len(audio_paths)}...")

        # Load latent
        latent_path = audio_path_to_latent_path(audio_path)
        if latent_path is None:
            failed.append({'path': audio_path, 'reason': 'no_latent'})
            continue

        latent = load_latent(latent_path)
        if latent is None:
            failed.append({'path': audio_path, 'reason': 'load_failed'})
            continue

        # Window the latent
        windows = window_latent(latent, window_sec, hop_sec)

        if len(windows) == 0:
            failed.append({'path': audio_path, 'reason': 'too_short'})
            continue

        # Pool all windows
        window_features = torch.stack([pool_latent(w[2]) for w in windows])
        window_features_norm = (window_features - mean) / std

        # Predict all windows
        with torch.no_grad():
            logits = model(window_features_norm.to(device_obj))
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

        # Build per-window results
        window_results = []
        all_labels = set()
        for j, (start_sec, end_sec, _) in enumerate(windows):
            window_labels = [classes[k] for k in range(num_classes) if preds[j, k] == 1]
            all_labels.update(window_labels)

            window_results.append({
                'start': round(start_sec, 2),
                'end': round(end_sec, 2),
                'labels': window_labels,
                'probabilities': {classes[k]: round(float(probs[j, k]), 3)
                                 for k in range(num_classes) if probs[j, k] > 0.1}
            })

        # Detect temporal changes (group consecutive windows with same labels)
        segments = []
        current_labels = None
        current_start = None

        for wr in window_results:
            labels_set = frozenset(wr['labels'])
            if labels_set != current_labels:
                if current_labels is not None and len(current_labels) > 0:
                    segments.append({
                        'start': current_start,
                        'end': wr['start'],
                        'labels': list(current_labels)
                    })
                current_labels = labels_set
                current_start = wr['start']

        # Don't forget last segment
        if current_labels is not None and len(current_labels) > 0:
            segments.append({
                'start': current_start,
                'end': window_results[-1]['end'],
                'labels': list(current_labels)
            })

        # Aggregate file-level predictions
        # Use mean probability across windows, then threshold
        mean_probs = probs.mean(axis=0)
        file_labels = [classes[k] for k in range(num_classes) if mean_probs[k] >= threshold]

        # Calculate label durations
        label_durations = defaultdict(float)
        for seg in segments:
            duration = seg['end'] - seg['start']
            for label in seg['labels']:
                label_durations[label] += duration

        total_duration = window_results[-1]['end'] if window_results else 0

        result = {
            'path': audio_path,
            'filename': Path(audio_path).name,
            'total_duration': round(total_duration, 2),
            'file_labels': file_labels,
            'all_detected_labels': list(all_labels),
            'is_multilabel': len(all_labels) > 1,
            'has_temporal_change': len(segments) > 1,
            'num_segments': len(segments),
            'segments': segments,
            'label_durations': dict(label_durations),
            'windows': window_results,
            'mean_probabilities': {classes[k]: round(float(mean_probs[k]), 3)
                                   for k in range(num_classes) if mean_probs[k] > 0.1}
        }
        results.append(result)

        if result['has_temporal_change']:
            temporal_changes.append(result)

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("TEMPORAL MULTI-LABEL CLASSIFICATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"Total processed: {len(results)}")
    logging.info(f"  Multi-label (any window): {sum(1 for r in results if r['is_multilabel'])}")
    logging.info(f"  Temporal changes detected: {len(temporal_changes)}")
    logging.info(f"  Failed: {len(failed)}")

    if temporal_changes:
        logging.info("\nFiles with temporal instrument changes:")
        for r in temporal_changes[:15]:
            seg_summary = " -> ".join([f"{'+'.join(s['labels'])}({s['end']-s['start']:.1f}s)"
                                       for s in r['segments'][:4]])
            if len(r['segments']) > 4:
                seg_summary += f" ... ({len(r['segments'])} segments)"
            logging.info(f"  {r['filename']}: {seg_summary}")

    # Save results
    output_data = {
        'settings': {
            'window_sec': window_sec,
            'hop_sec': hop_sec,
            'threshold': threshold,
        },
        'summary': {
            'total': len(results),
            'multilabel': sum(1 for r in results if r['is_multilabel']),
            'temporal_changes': len(temporal_changes),
            'failed': len(failed),
        },
        'temporal_changes': temporal_changes,
        'all_results': results,
        'failed': failed,
        'analyzed_at': datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"\nResults saved to {output_path}")

    return output_data


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-label temporal instrument classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--mode', choices=['train', 'classify', 'temporal'], required=True)
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/gcs-bucket/Manifests/unified_manifest.json',
                        help='Manifest JSON with audio paths and labels')
    parser.add_argument('--corrections', type=str,
                        default='/home/arlo/gcs-bucket/Manifests/corrections.json',
                        help='Corrections JSON with multi-label data')
    parser.add_argument('--model', type=str,
                        help='Path to trained model')
    parser.add_argument('--output-dir', type=str,
                        default='/home/arlo/Data/multilabel_classifier')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--threshold', type=float, default=LABEL_THRESHOLD,
                        help='Threshold for positive prediction')
    parser.add_argument('--window-sec', type=float, default=DEFAULT_WINDOW_SEC,
                        help='Window size in seconds for temporal mode')
    parser.add_argument('--hop-sec', type=float, default=DEFAULT_HOP_SEC,
                        help='Hop size in seconds for temporal mode')
    parser.add_argument('--group', type=str,
                        help='Filter to specific group')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of files to process')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'train':
        train_classifier(
            Path(args.manifest),
            Path(args.corrections),
            output_dir,
            num_workers=args.workers,
            device=args.device
        )

    elif args.mode == 'classify':
        if not args.model:
            args.model = output_dir / 'model.pt'

        # Get paths from manifest
        logging.info(f"Loading paths from manifest: {args.manifest}")
        with open(args.manifest) as f:
            manifest = json.load(f)

        audio_paths = []
        for entry in manifest.get('entries', []):
            if not entry.get('has_latent'):
                continue
            # Skip excluded groups (silent, junk, etc.)
            if entry.get('group') in EXCLUDED_CLASSES:
                continue
            if args.group and entry.get('group') != args.group:
                continue
            audio_paths.append(entry['audio_path'])

        if args.limit > 0:
            audio_paths = audio_paths[:args.limit]

        logging.info(f"Found {len(audio_paths)} paths to classify")

        classify_files(
            Path(args.model),
            audio_paths,
            output_dir / 'predictions.json',
            threshold=args.threshold,
            num_workers=args.workers,
            device=args.device
        )

    elif args.mode == 'temporal':
        if not args.model:
            args.model = output_dir / 'model.pt'

        # Get paths from manifest
        logging.info(f"Loading paths from manifest: {args.manifest}")
        with open(args.manifest) as f:
            manifest = json.load(f)

        audio_paths = []
        for entry in manifest.get('entries', []):
            if not entry.get('has_latent'):
                continue
            # Skip excluded groups (silent, junk, etc.)
            if entry.get('group') in EXCLUDED_CLASSES:
                continue
            if args.group and entry.get('group') != args.group:
                continue
            audio_paths.append(entry['audio_path'])

        if args.limit > 0:
            audio_paths = audio_paths[:args.limit]

        logging.info(f"Found {len(audio_paths)} paths to analyze")

        classify_temporal(
            Path(args.model),
            audio_paths,
            output_dir / 'temporal_analysis.json',
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            threshold=args.threshold,
            num_workers=args.workers,
            device=args.device
        )


if __name__ == '__main__':
    main()
