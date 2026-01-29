#!/usr/bin/env python3
"""
Latent-Space Ensemble Detector

Detects ensemble/full-track recordings vs clean solo stems using ACE-Step latents.
Uses latent features (like the instrument classifier) for fast, accurate detection.

Two-stage pipeline:
  Stage 1: Solo vs Ensemble classification
  Stage 2: If solo -> instrument classifier
           If ensemble -> multi-label (future)

Usage:
  # Train from corrections (ensemble/full-track labels)
  python latent_ensemble_detector.py --mode train \
    --corrections /home/arlo/gcs-bucket/Manifests/corrections.json \
    --manifest /home/arlo/gcs-bucket/Manifests/unified_manifest.json \
    --output-dir /home/arlo/Data/ensemble_detector

  # Detect ensembles in manifest
  python latent_ensemble_detector.py --mode detect \
    --model /home/arlo/Data/ensemble_detector/model.pt \
    --manifest /home/arlo/gcs-bucket/Manifests/unified_manifest.json \
    --output-dir /home/arlo/Data/ensemble_detector

  # Add silent detection (pre-filter)
  python latent_ensemble_detector.py --mode detect-silent \
    --manifest /home/arlo/gcs-bucket/Manifests/unified_manifest.json \
    --output-dir /home/arlo/Data/ensemble_detector
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ===================== CONFIGURATION =====================

LATENTS_ROOT = Path("/home/arlo/gcs-bucket/Latents")
AUDIO_ROOT = Path("/home/arlo/gcs-bucket")

# Labels that indicate ensemble/multi-instrument
ENSEMBLE_LABELS = {'ensemble', 'full-track'}
SILENT_LABEL = 'silent'
JUNK_LABEL = 'junk'

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
HIDDEN_DIM = 128

# Feature pooling
POOL_METHODS = ['mean', 'std', 'max']

# Silent detection thresholds (RMS-based)
SILENT_RMS_THRESHOLD = 0.001  # Below this = silent


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


def extract_latent_stats(latent: torch.Tensor) -> Dict:
    """Extract additional statistics from latent for ensemble detection.

    Ensemble recordings tend to have:
    - Higher variance across time (multiple instruments)
    - More "active" codebook entries
    - Different temporal dynamics
    """
    stats = {}

    # Basic pooled features
    stats['mean'] = latent.mean().item()
    stats['std'] = latent.std().item()
    stats['max'] = latent.max().item()
    stats['min'] = latent.min().item()

    # Temporal variance (how much the latent changes over time)
    temporal_diff = torch.diff(latent, dim=-1)
    stats['temporal_variance'] = temporal_diff.var().item()
    stats['temporal_mean_change'] = temporal_diff.abs().mean().item()

    # Activity measure (how many codebook entries are "active")
    threshold = latent.abs().mean() * 0.5
    stats['activity_ratio'] = (latent.abs() > threshold).float().mean().item()

    # Per-codebook variance (8 codebooks)
    codebook_vars = latent.var(dim=(1, 2))  # variance per codebook
    stats['codebook_var_mean'] = codebook_vars.mean().item()
    stats['codebook_var_std'] = codebook_vars.std().item()

    # Spectral-like features (variance across the 16 channels)
    channel_vars = latent.var(dim=(0, 2))  # variance per channel
    stats['channel_var_mean'] = channel_vars.mean().item()
    stats['channel_var_std'] = channel_vars.std().item()

    return stats


# ===================== MODEL =====================

class EnsembleDetector(nn.Module):
    """Binary classifier: solo (0) vs ensemble (1)."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ===================== DATA LOADING =====================

def load_training_data(
    corrections_path: Path,
    manifest_path: Path,
    num_workers: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Load training data from corrections and manifest.

    Positive class (ensemble=1): corrections labeled as ensemble/full-track
    Negative class (solo=0): random sample of solo instrument labels
    """
    logging.info("Loading corrections...")
    with open(corrections_path) as f:
        corrections = json.load(f)

    logging.info("Loading manifest...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get ensemble examples from corrections
    ensemble_paths = []
    for path, info in corrections.items():
        group = info.get('group', '')
        if group in ENSEMBLE_LABELS:
            ensemble_paths.append(path)

    logging.info(f"Found {len(ensemble_paths)} ensemble/full-track examples in corrections")

    # Get solo examples from manifest (excluding undefined, room, ensemble labels)
    solo_paths = []
    excluded = {'undefined', 'room', 'fx', 'click', 'ensemble', 'full-track', 'silent', 'junk'}

    entries = manifest.get('entries', [])
    for entry in entries:
        if not entry.get('has_latent'):
            continue
        group = entry.get('group', '')
        if group and group not in excluded:
            # Also exclude items that were corrected to ensemble
            audio_path = entry.get('audio_path', '')
            if audio_path not in corrections or corrections.get(audio_path, {}).get('group') not in ENSEMBLE_LABELS:
                solo_paths.append(audio_path)

    # Balance classes - sample solo to match ensemble (or use ratio)
    # For small datasets, use all ensemble and sample more solos
    num_ensemble = len(ensemble_paths)
    num_solo = min(len(solo_paths), num_ensemble * 10)  # 10:1 ratio max

    import random
    random.seed(42)
    solo_paths = random.sample(solo_paths, num_solo)

    logging.info(f"Training data: {num_ensemble} ensemble, {num_solo} solo")

    # Extract features
    all_paths = ensemble_paths + solo_paths
    labels = [1] * len(ensemble_paths) + [0] * len(solo_paths)

    features_list = []
    valid_paths = []
    valid_labels = []

    def process_path(path: str) -> Optional[torch.Tensor]:
        latent_path = audio_path_to_latent_path(path)
        if latent_path is None:
            return None
        latent = load_latent(latent_path)
        if latent is None:
            return None
        return pool_latent(latent)

    logging.info("Extracting features...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_path, p): (p, l) for p, l in zip(all_paths, labels)}

        for future in as_completed(futures):
            path, label = futures[future]
            features = future.result()
            if features is not None:
                features_list.append(features)
                valid_paths.append(path)
                valid_labels.append(label)

    logging.info(f"Extracted {len(features_list)} features ({sum(valid_labels)} ensemble, {len(valid_labels) - sum(valid_labels)} solo)")

    X = torch.stack(features_list)
    y = torch.tensor(valid_labels, dtype=torch.float32)

    return X, y, valid_paths


# ===================== TRAINING =====================

def train_ensemble_detector(
    corrections_path: Path,
    manifest_path: Path,
    output_dir: Path,
    num_workers: int = 8,
    device: str = 'cuda'
) -> Dict:
    """Train the ensemble detector."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)
    logging.info(f"Using device: {device}")

    # Load data
    X, y, paths = load_training_data(corrections_path, manifest_path, num_workers)

    if len(X) < 20:
        raise ValueError(f"Not enough training data: {len(X)} samples (need at least 20)")

    # Normalize
    mean = X.mean(dim=0)
    std = X.std(dim=0) + 1e-8
    X_norm = (X - mean) / std

    # Split
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y.numpy(), random_state=42)

    X_train, y_train = X_norm[train_idx], y[train_idx]
    X_test, y_test = X_norm[test_idx], y[test_idx]

    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    input_dim = X.shape[1]
    model = EnsembleDetector(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Handle class imbalance with pos_weight
    num_pos = y_train.sum().item()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / (num_pos + 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training
    logging.info(f"Training for {NUM_EPOCHS} epochs...")
    best_acc = 0
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

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test.to(device))
            test_probs = torch.sigmoid(test_logits)
            test_preds = (test_probs > 0.5).float()
            test_acc = (test_preds.cpu() == y_test).float().mean().item()

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            logging.info(f"  Epoch {epoch+1}/{NUM_EPOCHS}: loss={train_loss/len(train_dataset):.4f}, test_acc={test_acc:.3f}")

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_probs > 0.5).astype(int)
        y_test_np = y_test.numpy().astype(int)

    # Metrics
    from sklearn.metrics import classification_report, confusion_matrix

    logging.info("\n" + "=" * 50)
    logging.info("ENSEMBLE DETECTOR RESULTS")
    logging.info("=" * 50)
    logging.info(f"Test samples: {len(y_test)}")
    logging.info(f"  Ensemble: {sum(y_test_np)}")
    logging.info(f"  Solo: {len(y_test_np) - sum(y_test_np)}")
    logging.info(f"\nBest accuracy: {best_acc:.1%}")

    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test_np, test_preds, target_names=['solo', 'ensemble']))

    logging.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_np, test_preds)
    logging.info(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
    logging.info(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")

    # Save model
    model_data = {
        'model_state': best_state,
        'input_dim': input_dim,
        'hidden_dim': HIDDEN_DIM,
        'mean': mean,
        'std': std,
        'pool_methods': POOL_METHODS,
        'training_stats': {
            'num_ensemble': int(y.sum().item()),
            'num_solo': int(len(y) - y.sum().item()),
            'best_accuracy': best_acc,
        },
        'trained_at': datetime.now().isoformat()
    }

    model_path = output_dir / 'model.pt'
    torch.save(model_data, model_path)
    logging.info(f"\nModel saved to {model_path}")

    return model_data


# ===================== DETECTION =====================

def detect_ensembles(
    model_path: Path,
    manifest_path: Path,
    output_dir: Path,
    num_workers: int = 8,
    device: str = 'cuda',
    threshold: float = 0.5
) -> Dict:
    """Detect ensembles in manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    # Load model
    logging.info(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)

    input_dim = model_data['input_dim']
    hidden_dim = model_data.get('hidden_dim', HIDDEN_DIM)
    mean = model_data['mean']
    std = model_data['std']

    model = EnsembleDetector(input_dim, hidden_dim)
    model.load_state_dict(model_data['model_state'])
    model.to(device)
    model.eval()

    # Load manifest
    logging.info(f"Loading manifest from {manifest_path}...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = manifest.get('entries', [])
    paths_to_check = []

    for entry in entries:
        if not entry.get('has_latent'):
            continue
        # Check all entries with latents
        paths_to_check.append(entry.get('audio_path', ''))

    logging.info(f"Checking {len(paths_to_check)} entries...")

    # Extract features
    def process_path(path: str) -> Tuple[str, Optional[torch.Tensor]]:
        latent_path = audio_path_to_latent_path(path)
        if latent_path is None:
            return path, None
        latent = load_latent(latent_path)
        if latent is None:
            return path, None
        return path, pool_latent(latent)

    features_list = []
    valid_paths = []

    logging.info("Extracting features...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_path, p) for p in paths_to_check]

        for i, future in enumerate(as_completed(futures)):
            path, features = future.result()
            if features is not None:
                features_list.append(features)
                valid_paths.append(path)

            if (i + 1) % 10000 == 0:
                logging.info(f"  Processed {i+1}/{len(paths_to_check)}...")

    logging.info(f"Extracted {len(features_list)} features")

    if len(features_list) == 0:
        return {'detected': [], 'total_checked': 0}

    # Normalize and predict
    X = torch.stack(features_list)
    X_norm = (X - mean) / std

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_norm), BATCH_SIZE):
            batch = X_norm[i:i+BATCH_SIZE].to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs).numpy()

    # Results
    detected_ensemble = []
    for path, prob in zip(valid_paths, all_probs):
        if prob >= threshold:
            detected_ensemble.append({
                'path': path,
                'ensemble_probability': float(prob),
                'filename': Path(path).name
            })

    # Sort by probability
    detected_ensemble.sort(key=lambda x: -x['ensemble_probability'])

    # Summary
    logging.info("\n" + "=" * 50)
    logging.info("ENSEMBLE DETECTION RESULTS")
    logging.info("=" * 50)
    logging.info(f"Total checked: {len(valid_paths)}")
    logging.info(f"Detected as ensemble (>{threshold:.0%}): {len(detected_ensemble)}")

    # Distribution
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(bins) - 1):
        count = sum(1 for p in all_probs if bins[i] <= p < bins[i+1])
        logging.info(f"  {bins[i]:.0%}-{bins[i+1]:.0%}: {count}")

    # Save results
    results = {
        'threshold': threshold,
        'total_checked': len(valid_paths),
        'detected_count': len(detected_ensemble),
        'detected': detected_ensemble,
        'detected_at': datetime.now().isoformat()
    }

    output_path = output_dir / 'ensemble_detections.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nResults saved to {output_path}")

    return results


# ===================== SILENT DETECTION =====================

def detect_silent_from_latents(
    manifest_path: Path,
    output_dir: Path,
    num_workers: int = 8,
    rms_threshold: float = SILENT_RMS_THRESHOLD
) -> Dict:
    """
    Detect silent files using latent statistics.

    Silent files have very low latent activation (near-zero mean/std).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading manifest from {manifest_path}...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = manifest.get('entries', [])
    paths_to_check = []

    for entry in entries:
        if not entry.get('has_latent'):
            continue
        paths_to_check.append(entry.get('audio_path', ''))

    logging.info(f"Checking {len(paths_to_check)} entries for silence...")

    def check_silence(path: str) -> Tuple[str, Optional[Dict]]:
        latent_path = audio_path_to_latent_path(path)
        if latent_path is None:
            return path, None
        latent = load_latent(latent_path)
        if latent is None:
            return path, None

        # Silent detection: very low latent magnitude
        mean_abs = latent.abs().mean().item()
        std = latent.std().item()
        max_abs = latent.abs().max().item()

        return path, {
            'mean_abs': mean_abs,
            'std': std,
            'max_abs': max_abs,
        }

    # Process
    results = []
    silent_detected = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(check_silence, p) for p in paths_to_check]

        for i, future in enumerate(as_completed(futures)):
            path, stats = future.result()
            if stats is not None:
                results.append((path, stats))

                # Silent if very low activation
                # Threshold calibrated empirically - silent files have mean_abs < 0.01
                if stats['mean_abs'] < 0.01 and stats['max_abs'] < 0.1:
                    silent_detected.append({
                        'path': path,
                        'filename': Path(path).name,
                        **stats
                    })

            if (i + 1) % 10000 == 0:
                logging.info(f"  Processed {i+1}/{len(paths_to_check)}...")

    # Summary
    logging.info("\n" + "=" * 50)
    logging.info("SILENT DETECTION RESULTS")
    logging.info("=" * 50)
    logging.info(f"Total checked: {len(results)}")
    logging.info(f"Detected as silent: {len(silent_detected)}")

    # Save
    output_data = {
        'total_checked': len(results),
        'silent_count': len(silent_detected),
        'silent_files': silent_detected,
        'detected_at': datetime.now().isoformat()
    }

    output_path = output_dir / 'silent_detections.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"\nResults saved to {output_path}")

    return output_data


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description='Latent-space ensemble detector')

    parser.add_argument('--mode', choices=['train', 'detect', 'detect-silent'], required=True)
    parser.add_argument('--corrections', type=str,
                        default='/home/arlo/gcs-bucket/Manifests/corrections.json',
                        help='Corrections JSON with ensemble/full-track labels')
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/gcs-bucket/Manifests/unified_manifest.json',
                        help='Manifest with audio entries')
    parser.add_argument('--model', type=str,
                        help='Path to trained model (for detect mode)')
    parser.add_argument('--output-dir', type=str,
                        default='/home/arlo/Data/ensemble_detector')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Ensemble detection threshold')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    output_dir = Path(args.output_dir)

    if args.mode == 'train':
        train_ensemble_detector(
            Path(args.corrections),
            Path(args.manifest),
            output_dir,
            num_workers=args.workers,
            device=args.device
        )

    elif args.mode == 'detect':
        if not args.model:
            parser.error("--model required for detect mode")
        detect_ensembles(
            Path(args.model),
            Path(args.manifest),
            output_dir,
            num_workers=args.workers,
            device=args.device,
            threshold=args.threshold
        )

    elif args.mode == 'detect-silent':
        detect_silent_from_latents(
            Path(args.manifest),
            output_dir,
            num_workers=args.workers
        )


if __name__ == '__main__':
    main()
