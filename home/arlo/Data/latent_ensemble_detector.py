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

# Instrument groups for conditioning
INSTRUMENT_GROUPS = [
    'guitar', 'drums', 'piano', 'bass', 'voice', 'strings',
    'brass', 'winds', 'synth', 'organ', 'percussion', 'mallets',
    'plucked', 'dialogue', 'fx', 'unknown'
]
GROUP_TO_IDX = {g: i for i, g in enumerate(INSTRUMENT_GROUPS)}
NUM_GROUPS = len(INSTRUMENT_GROUPS)

# Bleed instruments (what can bleed into a recording)
BLEED_INSTRUMENTS = [
    'drums', 'bass', 'guitar', 'piano', 'voice', 'strings',
    'brass', 'winds', 'percussion', 'organ', 'synth', 'click'
]
BLEED_TO_IDX = {g: i for i, g in enumerate(BLEED_INSTRUMENTS)}
NUM_BLEED = len(BLEED_INSTRUMENTS)


class EnsembleDetectorV2(nn.Module):
    """
    3-class classifier with group conditioning:
      - 0: isolated (clean single instrument)
      - 1: bleed (main instrument + quiet background)
      - 2: mix (multiple instruments at similar levels)

    Also predicts which instruments are bleeding (multi-label).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        num_groups: int = NUM_GROUPS,
        group_embed_dim: int = 16,
        num_bleed_instruments: int = NUM_BLEED
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_embed_dim = group_embed_dim

        # Group embedding
        self.group_embedding = nn.Embedding(num_groups, group_embed_dim)

        # Main network (latent features + group embedding)
        combined_dim = input_dim + group_embed_dim
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Classification head: isolated (0) / bleed (1) / mix (2)
        self.classifier = nn.Linear(hidden_dim // 2, 3)

        # Bleed instrument prediction head (multi-label)
        self.bleed_predictor = nn.Linear(hidden_dim // 2, num_bleed_instruments)

    def forward(self, x, group_idx):
        """
        Args:
            x: [B, input_dim] latent features
            group_idx: [B] group indices
        Returns:
            class_logits: [B, 3] - isolated/bleed/mix
            bleed_logits: [B, num_bleed] - which instruments bleeding
        """
        # Get group embedding
        group_emb = self.group_embedding(group_idx)  # [B, group_embed_dim]

        # Concatenate features
        combined = torch.cat([x, group_emb], dim=-1)

        # Backbone
        features = self.backbone(combined)

        # Outputs
        class_logits = self.classifier(features)
        bleed_logits = self.bleed_predictor(features)

        return class_logits, bleed_logits


class EnsembleDetector(nn.Module):
    """Legacy binary classifier for backwards compatibility."""

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

def load_training_data_v2(
    corrections_path: Path,
    manifest_path: Path,
    num_workers: int = 8,
    max_samples_per_class: int = 15000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Load training data with 3-class labels and group conditioning.

    Classes:
      - 0: isolated (clean single instrument, no bleed)
      - 1: bleed (main instrument + quiet background instruments)
      - 2: mix (multiple instruments at similar levels)

    Returns:
      features: [N, 384] latent features
      labels: [N] class labels (0/1/2)
      group_indices: [N] instrument group indices for conditioning
      bleed_targets: [N, num_bleed] multi-label bleed instruments
      paths: list of audio paths
    """
    import random
    random.seed(42)

    logging.info("Loading corrections...")
    with open(corrections_path) as f:
        corrections = json.load(f)

    logging.info("Loading manifest...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Build lookups from manifest
    manifest_lookup = {}
    entries = manifest.get('entries', [])
    for entry in entries:
        if isinstance(entry, dict):
            audio_path = entry.get('audio_path', '')
            manifest_lookup[audio_path] = entry

    # Collect samples by class
    isolated_samples = []  # class 0: clean solo
    bleed_samples = []     # class 1: has bleed
    mix_samples = []       # class 2: full mix

    solo_groups = {'guitar', 'drums', 'piano', 'bass', 'voice', 'strings',
                   'brass', 'winds', 'synth', 'organ', 'percussion', 'mallets',
                   'plucked', 'dialogue'}
    mix_correction_groups = {'ensemble', 'full-track', 'room'}

    # Process corrections first (highest quality labels)
    for path, info in corrections.items():
        group = info.get('group', '')
        is_multi_label = info.get('multi_label', False)
        has_bleed = info.get('has_bleed', False)
        bleed_instruments = info.get('bleed_instruments', [])

        # Check if has latent
        manifest_entry = manifest_lookup.get(path, {})
        if not manifest_entry.get('has_latent', False):
            latent_path = audio_path_to_latent_path(path)
            if latent_path is None:
                continue

        # Get group for conditioning (use corrected group or manifest group)
        cond_group = group if group in solo_groups else manifest_entry.get('group', 'unknown')

        sample = {
            'path': path,
            'group': cond_group,
            'bleed_instruments': bleed_instruments,
        }

        if is_multi_label:
            # Multi-label = mix (class 2)
            regions = info.get('regions', [])
            for r in regions:
                if len(r.get('labels', [])) >= 2:
                    mix_samples.append(sample)
                    break
        elif group in mix_correction_groups:
            # Explicitly labeled as mix (class 2)
            mix_samples.append(sample)
        elif has_bleed and group in solo_groups:
            # Has bleed but main instrument identified (class 1)
            bleed_samples.append(sample)
        elif group in solo_groups:
            # Clean solo (class 0)
            isolated_samples.append(sample)

    logging.info(f"From corrections: {len(isolated_samples)} isolated, {len(bleed_samples)} bleed, {len(mix_samples)} mix")

    # Add mix samples from manifest (is_mix=True)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not entry.get('has_latent'):
            continue

        audio_path = entry.get('audio_path', '')

        # Skip if already in corrections
        if audio_path in corrections:
            continue

        if entry.get('is_mix'):
            group = entry.get('group', 'unknown')
            mix_samples.append({
                'path': audio_path,
                'group': group,
                'bleed_instruments': [],
            })

    logging.info(f"Total: {len(isolated_samples)} isolated, {len(bleed_samples)} bleed, {len(mix_samples)} mix")

    # Balance classes (but keep all bleed since we have few)
    num_isolated = min(len(isolated_samples), max_samples_per_class)
    num_bleed = len(bleed_samples)  # Keep all bleed samples
    num_mix = min(len(mix_samples), max_samples_per_class)

    if len(isolated_samples) > num_isolated:
        isolated_samples = random.sample(isolated_samples, num_isolated)
    if len(mix_samples) > num_mix:
        mix_samples = random.sample(mix_samples, num_mix)

    logging.info(f"After balancing: {len(isolated_samples)} isolated, {len(bleed_samples)} bleed, {len(mix_samples)} mix")

    # Combine all samples with labels
    all_samples = []
    for s in isolated_samples:
        s['label'] = 0
        all_samples.append(s)
    for s in bleed_samples:
        s['label'] = 1
        all_samples.append(s)
    for s in mix_samples:
        s['label'] = 2
        all_samples.append(s)

    # Extract features
    def process_sample(sample):
        latent_path = audio_path_to_latent_path(sample['path'])
        if latent_path is None:
            return None
        latent = load_latent(latent_path)
        if latent is None:
            return None

        features = pool_latent(latent)
        group_idx = GROUP_TO_IDX.get(sample['group'], GROUP_TO_IDX['unknown'])

        # Build bleed target vector
        bleed_target = torch.zeros(NUM_BLEED)
        for inst in sample.get('bleed_instruments', []):
            if inst in BLEED_TO_IDX:
                bleed_target[BLEED_TO_IDX[inst]] = 1.0

        return {
            'features': features,
            'label': sample['label'],
            'group_idx': group_idx,
            'bleed_target': bleed_target,
            'path': sample['path'],
        }

    logging.info("Extracting features...")
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_sample, s): s for s in all_samples}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    logging.info(f"Extracted {len(results)} samples")

    # Stack into tensors
    features = torch.stack([r['features'] for r in results])
    labels = torch.tensor([r['label'] for r in results], dtype=torch.long)
    group_indices = torch.tensor([r['group_idx'] for r in results], dtype=torch.long)
    bleed_targets = torch.stack([r['bleed_target'] for r in results])
    paths = [r['path'] for r in results]

    # Log class distribution
    for i, name in enumerate(['isolated', 'bleed', 'mix']):
        count = (labels == i).sum().item()
        logging.info(f"  Class {i} ({name}): {count}")

    return features, labels, group_indices, bleed_targets, paths


def load_training_data(
    corrections_path: Path,
    manifest_path: Path,
    num_workers: int = 8,
    max_samples_per_class: int = 15000
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Legacy data loader for binary classification (backwards compatibility).
    """
    import random
    random.seed(42)

    logging.info("Loading corrections...")
    with open(corrections_path) as f:
        corrections = json.load(f)

    logging.info("Loading manifest...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Build lookup for has_latent from manifest
    has_latent_lookup = {}
    entries = manifest.get('entries', [])
    for entry in entries:
        if isinstance(entry, dict):
            audio_path = entry.get('audio_path', '')
            has_latent_lookup[audio_path] = entry.get('has_latent', False)

    # Collect mix paths from manifest (is_mix=True)
    mix_paths = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not entry.get('has_latent'):
            continue
        if entry.get('is_mix'):
            mix_paths.add(entry.get('audio_path', ''))

    logging.info(f"Found {len(mix_paths)} is_mix=True in manifest")

    # Collect solo paths from corrections (single-label, verified instruments)
    solo_paths = []
    solo_groups = {'guitar', 'drums', 'piano', 'bass', 'voice', 'strings',
                   'brass', 'winds', 'synth', 'organ', 'percussion', 'mallets',
                   'plucked', 'dialogue'}
    mix_correction_groups = {'ensemble', 'full-track', 'room'}

    for path, info in corrections.items():
        group = info.get('group', '')
        is_multi_label = info.get('multi_label', False)

        # Check if has latent
        if not has_latent_lookup.get(path, False):
            latent_path = audio_path_to_latent_path(path)
            if latent_path is None:
                continue

        if is_multi_label:
            # Multi-label = mix
            regions = info.get('regions', [])
            for r in regions:
                if len(r.get('labels', [])) >= 2:
                    mix_paths.add(path)
                    break
        elif group in mix_correction_groups:
            # Explicitly labeled as mix
            mix_paths.add(path)
        elif group in solo_groups:
            # Single instrument correction = verified solo
            solo_paths.append(path)

    mix_paths = list(mix_paths)
    logging.info(f"Total mix examples: {len(mix_paths)}")
    logging.info(f"Solo examples from corrections: {len(solo_paths)}")

    # Balance classes for training
    num_mix = min(len(mix_paths), max_samples_per_class)
    num_solo = min(len(solo_paths), max_samples_per_class)

    # Sample if needed
    if len(mix_paths) > num_mix:
        mix_paths = random.sample(mix_paths, num_mix)
    if len(solo_paths) > num_solo:
        solo_paths = random.sample(solo_paths, num_solo)

    logging.info(f"Training data: {len(mix_paths)} mix/ensemble, {len(solo_paths)} solo")

    # Extract features
    all_paths = mix_paths + solo_paths
    labels = [1] * len(mix_paths) + [0] * len(solo_paths)

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

    logging.info(f"Extracted {len(features_list)} features ({sum(valid_labels)} mix, {len(valid_labels) - sum(valid_labels)} solo)")

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
            'num_mix': int(y.sum().item()),
            'num_solo': int(len(y) - y.sum().item()),
            'best_accuracy': best_acc,
        },
        'trained_at': datetime.now().isoformat()
    }

    model_path = output_dir / 'model.pt'
    torch.save(model_data, model_path)
    logging.info(f"\nModel saved to {model_path}")

    return model_data


def train_ensemble_detector_v2(
    corrections_path: Path,
    manifest_path: Path,
    output_dir: Path,
    num_workers: int = 8,
    device: str = 'cuda'
) -> Dict:
    """Train V2 ensemble detector with group conditioning and 3-class output."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)
    logging.info(f"Using device: {device}")

    # Load data with 3-class labels
    X, y, group_indices, bleed_targets, paths = load_training_data_v2(
        corrections_path, manifest_path, num_workers
    )

    if len(X) < 20:
        raise ValueError(f"Not enough training data: {len(X)} samples (need at least 20)")

    # Normalize features
    mean = X.mean(dim=0)
    std = X.std(dim=0) + 1e-8
    X_norm = (X - mean) / std

    # Split (stratified by class)
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y.numpy(), random_state=42)

    X_train = X_norm[train_idx]
    y_train = y[train_idx]
    group_train = group_indices[train_idx]
    bleed_train = bleed_targets[train_idx]

    X_test = X_norm[test_idx]
    y_test = y[test_idx]
    group_test = group_indices[test_idx]
    bleed_test = bleed_targets[test_idx]

    # DataLoader
    train_dataset = TensorDataset(X_train, y_train, group_train, bleed_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    input_dim = X.shape[1]
    model = EnsembleDetectorV2(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_groups=NUM_GROUPS,
        group_embed_dim=16,
        num_bleed_instruments=NUM_BLEED
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Class weights for imbalanced data
    class_counts = torch.bincount(y_train, minlength=3).float()
    class_weights = (1.0 / (class_counts + 1)).to(device)
    class_weights = class_weights / class_weights.sum() * 3  # Normalize

    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    criterion_bleed = nn.BCEWithLogitsLoss()

    # Training
    logging.info(f"Training V2 for {NUM_EPOCHS} epochs...")
    logging.info(f"  Class weights: isolated={class_weights[0]:.2f}, bleed={class_weights[1]:.2f}, mix={class_weights[2]:.2f}")

    best_acc = 0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0

        for batch_x, batch_y, batch_group, batch_bleed in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_group = batch_group.to(device)
            batch_bleed = batch_bleed.to(device)

            optimizer.zero_grad()

            class_logits, bleed_logits = model(batch_x, batch_group)

            # Combined loss
            loss_class = criterion_class(class_logits, batch_y)
            loss_bleed = criterion_bleed(bleed_logits, batch_bleed)

            # Weight bleed loss lower since we have few examples
            loss = loss_class + 0.1 * loss_bleed

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_y)

        # Evaluate
        model.eval()
        with torch.no_grad():
            class_logits, bleed_logits = model(X_test.to(device), group_test.to(device))
            test_preds = class_logits.argmax(dim=1).cpu()
            test_acc = (test_preds == y_test).float().mean().item()

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
        class_logits, bleed_logits = model(X_test.to(device), group_test.to(device))
        test_probs = F.softmax(class_logits, dim=1).cpu().numpy()
        test_preds = class_logits.argmax(dim=1).cpu().numpy()
        y_test_np = y_test.numpy()

    # Metrics
    from sklearn.metrics import classification_report, confusion_matrix

    logging.info("\n" + "=" * 50)
    logging.info("ENSEMBLE DETECTOR V2 RESULTS")
    logging.info("=" * 50)

    class_names = ['isolated', 'bleed', 'mix']
    for i, name in enumerate(class_names):
        count = (y_test_np == i).sum()
        logging.info(f"  {name}: {count}")

    logging.info(f"\nBest accuracy: {best_acc:.1%}")

    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test_np, test_preds, target_names=class_names))

    logging.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_np, test_preds)
    logging.info(f"  Rows=true, Cols=predicted")
    logging.info(f"  {class_names}")
    for i, row in enumerate(cm):
        logging.info(f"  {class_names[i]}: {row}")

    # Save model
    model_data = {
        'model_state': best_state,
        'model_version': 'v2',
        'input_dim': input_dim,
        'hidden_dim': HIDDEN_DIM,
        'num_groups': NUM_GROUPS,
        'group_embed_dim': 16,
        'num_bleed_instruments': NUM_BLEED,
        'mean': mean,
        'std': std,
        'pool_methods': POOL_METHODS,
        'instrument_groups': INSTRUMENT_GROUPS,
        'bleed_instruments': BLEED_INSTRUMENTS,
        'training_stats': {
            'num_isolated': int((y == 0).sum().item()),
            'num_bleed': int((y == 1).sum().item()),
            'num_mix': int((y == 2).sum().item()),
            'best_accuracy': best_acc,
        },
        'trained_at': datetime.now().isoformat()
    }

    model_path = output_dir / 'model_v2.pt'
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


def detect_ensembles_v2(
    model_path: Path,
    manifest_path: Path,
    output_dir: Path,
    num_workers: int = 8,
    device: str = 'cuda',
) -> Dict:
    """Detect ensembles using V2 model (3-class with group conditioning)."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    # Load model
    logging.info(f"Loading V2 model from {model_path}...")
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)

    input_dim = model_data['input_dim']
    hidden_dim = model_data.get('hidden_dim', HIDDEN_DIM)
    num_groups = model_data.get('num_groups', NUM_GROUPS)
    group_embed_dim = model_data.get('group_embed_dim', 16)
    num_bleed = model_data.get('num_bleed_instruments', NUM_BLEED)
    mean = model_data['mean']
    std = model_data['std']

    model = EnsembleDetectorV2(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_groups=num_groups,
        group_embed_dim=group_embed_dim,
        num_bleed_instruments=num_bleed
    )
    model.load_state_dict(model_data['model_state'])
    model.to(device)
    model.eval()

    # Load manifest
    logging.info(f"Loading manifest from {manifest_path}...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = manifest.get('entries', [])

    # Build list of (path, group) pairs
    samples_to_check = []
    for entry in entries:
        if not entry.get('has_latent'):
            continue
        audio_path = entry.get('audio_path', '')
        group = entry.get('group', 'unknown')
        samples_to_check.append({'path': audio_path, 'group': group})

    logging.info(f"Checking {len(samples_to_check)} entries...")

    # Extract features
    def process_sample(sample):
        latent_path = audio_path_to_latent_path(sample['path'])
        if latent_path is None:
            return None
        latent = load_latent(latent_path)
        if latent is None:
            return None

        features = pool_latent(latent)
        group_idx = GROUP_TO_IDX.get(sample['group'], GROUP_TO_IDX['unknown'])

        return {
            'path': sample['path'],
            'features': features,
            'group_idx': group_idx,
            'group': sample['group'],
        }

    results = []
    logging.info("Extracting features...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_sample, s) for s in samples_to_check]

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                results.append(result)

            if (i + 1) % 10000 == 0:
                logging.info(f"  Processed {i+1}/{len(samples_to_check)}...")

    logging.info(f"Extracted {len(results)} features")

    if len(results) == 0:
        return {'detected': [], 'total_checked': 0}

    # Stack and normalize
    X = torch.stack([r['features'] for r in results])
    group_indices = torch.tensor([r['group_idx'] for r in results], dtype=torch.long)
    X_norm = (X - mean) / std

    # Predict in batches
    all_class_probs = []
    all_bleed_probs = []

    with torch.no_grad():
        for i in range(0, len(X_norm), BATCH_SIZE):
            batch_x = X_norm[i:i+BATCH_SIZE].to(device)
            batch_group = group_indices[i:i+BATCH_SIZE].to(device)

            class_logits, bleed_logits = model(batch_x, batch_group)

            class_probs = F.softmax(class_logits, dim=1).cpu()
            bleed_probs = torch.sigmoid(bleed_logits).cpu()

            all_class_probs.append(class_probs)
            all_bleed_probs.append(bleed_probs)

    all_class_probs = torch.cat(all_class_probs).numpy()
    all_bleed_probs = torch.cat(all_bleed_probs).numpy()

    # Build output
    class_names = ['isolated', 'bleed', 'mix']
    bleed_names = model_data.get('bleed_instruments', BLEED_INSTRUMENTS)

    detected_isolated = []
    detected_bleed = []
    detected_mix = []

    for i, r in enumerate(results):
        probs = all_class_probs[i]
        bleed_probs = all_bleed_probs[i]
        pred_class = int(probs.argmax())

        entry = {
            'path': r['path'],
            'filename': Path(r['path']).name,
            'group': r['group'],
            'predicted_class': class_names[pred_class],
            'class_probabilities': {
                'isolated': float(probs[0]),
                'bleed': float(probs[1]),
                'mix': float(probs[2]),
            },
        }

        # Add bleed instruments if predicted as bleed or mix
        if pred_class >= 1:
            bleed_instruments = []
            for j, prob in enumerate(bleed_probs):
                if prob > 0.3:  # Threshold for bleed instrument
                    bleed_instruments.append({
                        'instrument': bleed_names[j],
                        'probability': float(prob)
                    })
            entry['bleed_instruments'] = bleed_instruments

        if pred_class == 0:
            detected_isolated.append(entry)
        elif pred_class == 1:
            detected_bleed.append(entry)
        else:
            detected_mix.append(entry)

    # Sort by confidence
    detected_isolated.sort(key=lambda x: -x['class_probabilities']['isolated'])
    detected_bleed.sort(key=lambda x: -x['class_probabilities']['bleed'])
    detected_mix.sort(key=lambda x: -x['class_probabilities']['mix'])

    # Summary
    logging.info("\n" + "=" * 50)
    logging.info("ENSEMBLE DETECTION V2 RESULTS")
    logging.info("=" * 50)
    logging.info(f"Total checked: {len(results)}")
    logging.info(f"  Isolated: {len(detected_isolated)}")
    logging.info(f"  Bleed: {len(detected_bleed)}")
    logging.info(f"  Mix: {len(detected_mix)}")

    # Save results
    output_data = {
        'model_version': 'v2',
        'total_checked': len(results),
        'counts': {
            'isolated': len(detected_isolated),
            'bleed': len(detected_bleed),
            'mix': len(detected_mix),
        },
        'isolated': detected_isolated,
        'bleed': detected_bleed,
        'mix': detected_mix,
        'detected_at': datetime.now().isoformat()
    }

    output_path = output_dir / 'ensemble_detections_v2.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"\nResults saved to {output_path}")

    return output_data


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

    parser.add_argument('--mode', choices=['train', 'train-v2', 'detect', 'detect-v2', 'detect-silent'], required=True,
                        help='train: binary (solo/mix), train-v2: 3-class with group conditioning (isolated/bleed/mix)')
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

    elif args.mode == 'train-v2':
        train_ensemble_detector_v2(
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

    elif args.mode == 'detect-v2':
        model_path = args.model or (output_dir / 'model_v2.pt')
        if not Path(model_path).exists():
            parser.error(f"Model not found: {model_path}. Run --mode train-v2 first.")
        detect_ensembles_v2(
            Path(model_path),
            Path(args.manifest),
            output_dir,
            num_workers=args.workers,
            device=args.device,
        )

    elif args.mode == 'detect-silent':
        detect_silent_from_latents(
            Path(args.manifest),
            output_dir,
            num_workers=args.workers
        )


if __name__ == '__main__':
    main()
