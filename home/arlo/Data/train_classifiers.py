#!/usr/bin/env python3
"""
Unified classifier training: mix, group, and subgroup classifiers.

Pipeline order matters — mix classifier runs first:
  1. Mix classifier      — multi-label: detects which instruments are present
  2. Mix detection       — runs inference on full manifest to find mixes
  3. Group classifier    — classifies ISOLATED audio into instrument groups
  4. Subgroup classifier — classifies within each group (e.g. electric_bass vs upright_bass)

Group and subgroup only train on isolated tracks (mixes excluded via Step 2).
All use ACE-Step latents (384-dim via mean/std/max pooling from 8x16 DCAE latents).

Usage:
  # Train all classifiers (mix → detect → group → subgroup)
  python3 train_classifiers.py

  # Train only mix classifier + detection
  python3 train_classifiers.py --stage mix

  # Train only group classifier (uses existing ensemble_detections.json)
  python3 train_classifiers.py --stage group

  # Train only subgroup classifiers (uses existing ensemble_detections.json)
  python3 train_classifiers.py --stage subgroup

  # Custom settings
  python3 train_classifiers.py --epochs 50 --device cuda --workers 12
"""

import argparse
import logging
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import orjson
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ===================== PATHS =====================

MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/master_manifest.json")
CORRECTIONS_PATH = Path("/home/arlo/gcs-bucket/Manifests/corrections.json")
LATENTS_ROOT = Path("/home/arlo/gcs-bucket/Latents")
AUDIO_ROOT = Path("/home/arlo/gcs-bucket")

GROUP_OUTPUT_DIR = Path("/home/arlo/Data/latent_classifier")
SUBGROUP_OUTPUT_DIR = Path("/home/arlo/Data/subgroup_classifiers")
MIX_OUTPUT_DIR = Path("/home/arlo/Data/mix_classifier")
ENSEMBLE_OUTPUT_DIR = Path("/home/arlo/Data/ensemble_detector")
ENSEMBLE_DETECTIONS_PATH = ENSEMBLE_OUTPUT_DIR / "ensemble_detections.json"

# ===================== CONFIG =====================

# Group classifier
GROUP_EXCLUDED = {'undefined', 'room', 'fx', 'click', 'silent', 'junk',
                  'review_vocals', 'ensemble', 'full-track', 'noise_hiss'}
GROUP_MAX_SAMPLES = 15000
GROUP_MIN_SAMPLES = 100

# Subgroup classifier
GROUPS_WITH_SUBGROUPS = {
    'brass': ['french_horn', 'trombone', 'trumpet', 'tuba', 'flugelhorn', 'brass_section'],
    'strings': ['cello', 'viola', 'violin', 'double_bass', 'string_section'],
    'winds': ['clarinet', 'flute', 'oboe', 'saxophone', 'bassoon'],
    'bass': ['electric_bass', 'upright_bass', 'synth_bass'],
    'guitar': ['acoustic_guitar', 'electric_guitar', 'classical_guitar'],
    'piano': ['grand_piano', 'upright_piano', 'electric_piano'],
}
SUBGROUP_MIN_SAMPLES = 5
SUBGROUP_MIN_CLASSES = 2

# Mix classifier — target classes to detect in mixes
MIX_TARGET_CLASSES = ['bass', 'brass', 'drums', 'guitar', 'piano', 'strings', 'voice', 'winds']
MIX_MIN_SOLO_SAMPLES = 50
MIX_DETECT_THRESHOLD = 0.3  # Per-class threshold for mix detection
MIX_MIN_CLASSES_FOR_MIX = 2  # Need 2+ classes above threshold to be a mix

# Shared
HIDDEN_DIM = 256
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LATENT_SILENT_THRESHOLD = 0.01

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')


# ===================== MODELS =====================

class GroupClassifier(nn.Module):
    """3-layer MLP for instrument group classification."""
    def __init__(self, input_dim=384, num_classes=10, hidden_dim=256):
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
        )

    def forward(self, x):
        return self.net(x)


class SubgroupClassifier(nn.Module):
    """2-layer MLP for within-group subgroup classification."""
    def __init__(self, input_dim=384, hidden_dim=256, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MixClassifier(nn.Module):
    """MLP transform from mix latent space to solo space + multi-label head."""
    def __init__(self, input_dim=384, hidden_dim=256, num_classes=8):
        super().__init__()
        # Transform: mix features -> solo-like features
        self.transform = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, input_dim),
        )
        # Multi-label classifier head
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        transformed = self.transform(x)
        return self.head(transformed)


# ===================== FEATURE EXTRACTION =====================

def audio_path_to_latent_path(audio_path: str) -> Optional[Path]:
    """Convert audio path to latent path, checking .dcae.pt then .pt."""
    audio_path = Path(audio_path)
    try:
        rel_path = audio_path.relative_to(AUDIO_ROOT)
    except ValueError:
        parts = audio_path.parts
        for prefix in ('protools', 'protoolsA'):
            if prefix in parts:
                idx = parts.index(prefix)
                rel_path = Path(*parts[idx:])
                break
        else:
            if 'gcs-bucket' in parts:
                idx = parts.index('gcs-bucket')
                rel_path = Path(*parts[idx + 1:])
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


def load_and_pool(audio_path: str) -> Optional[torch.Tensor]:
    """Load latent and extract 384-dim feature vector (mean+std+max pooling)."""
    latent_path = audio_path_to_latent_path(audio_path)
    if latent_path is None:
        return None
    try:
        data = torch.load(latent_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            latent = data.get('latents', data.get('latent'))
        else:
            latent = data

        if latent is None or latent.numel() == 0:
            return None

        # Mask silent frames
        if latent.shape[-1] > 1:
            energy = torch.sqrt((latent ** 2).mean(dim=(0, 1)))
            mask = energy > LATENT_SILENT_THRESHOLD
            if mask.sum() > 0:
                latent = latent[:, :, mask]

        # Multi-pool: mean + std + max -> 384 dims
        pools = [
            latent.mean(dim=-1),  # [8, 16]
            latent.std(dim=-1) if latent.shape[-1] > 1 else torch.zeros_like(latent.mean(dim=-1)),
            latent.max(dim=-1)[0],  # [8, 16]
        ]
        return torch.stack(pools, dim=-1).flatten()  # [384]
    except Exception:
        return None


def extract_features(paths: List[str], num_workers: int = 8, desc: str = "Extracting") -> Tuple[torch.Tensor, List[str]]:
    """Extract features in parallel. Returns (features, valid_paths)."""
    features_list = []
    valid_paths = []
    processed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_and_pool, p): p for p in paths}
        for future in as_completed(futures):
            path = futures[future]
            feat = future.result()
            if feat is not None:
                features_list.append(feat)
                valid_paths.append(path)
            processed += 1
            if processed % 1000 == 0:
                logging.info(f"  {desc}: {processed}/{len(paths)}")

    if not features_list:
        return torch.tensor([]), []

    logging.info(f"  {desc}: {len(valid_paths)}/{len(paths)} extracted")
    return torch.stack(features_list), valid_paths


# ===================== DATA LOADING =====================

def load_manifest() -> dict:
    """Load master manifest (dict keyed by audio path)."""
    logging.info(f"Loading manifest: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, 'rb') as f:
        data = orjson.loads(f.read())

    entries = data.get('entries', data)
    # Handle list format (consolidated/unified)
    if isinstance(entries, list):
        entries = {e.get('audio_path', e.get('path', '')): e for e in entries if isinstance(e, dict)}
    logging.info(f"  {len(entries):,} entries")
    return entries


def load_corrections() -> dict:
    """Load corrections file."""
    if not CORRECTIONS_PATH.exists():
        return {}
    with open(CORRECTIONS_PATH, 'rb') as f:
        corrections = orjson.loads(f.read())
    logging.info(f"  {len(corrections):,} corrections loaded")
    return corrections


def get_label(path: str, entry: dict, corrections: dict) -> Tuple[str, str]:
    """Get (group, subgroup) for a path, applying corrections."""
    group = entry.get('group', 'undefined')
    subgroup = entry.get('subgroup', '')

    if path in corrections:
        corr = corrections[path]
        if not corr.get('multi_label') and corr.get('group'):
            group = corr['group']
        if corr.get('subgroup'):
            subgroup = corr['subgroup']

    return group, subgroup


# ===================== STAGE 2: GROUP CLASSIFIER =====================

def train_group_classifier(manifest: dict, corrections: dict, epochs: int,
                           device: str, num_workers: int,
                           mix_paths: set = None) -> dict:
    """Train instrument group classifier on isolated tracks only."""
    logging.info("\n" + "=" * 60)
    logging.info("STAGE 2: GROUP CLASSIFIER (isolated only)")
    logging.info("=" * 60)

    if mix_paths is None:
        mix_paths = _load_existing_detections()

    GROUP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect labeled paths (skip mixes, excluded classes)
    path_to_label = {}
    for path, entry in manifest.items():
        group, _ = get_label(path, entry, corrections)

        if group in GROUP_EXCLUDED:
            continue

        # Skip mixes — detected by mix classifier + filename heuristics
        if path in mix_paths:
            continue
        is_mix = entry.get('is_mix', False)
        fname_lower = path.lower()
        if is_mix or 'mix' in fname_lower or '/room' in fname_lower:
            continue

        path_to_label[path] = group

    class_counts = Counter(path_to_label.values())
    logging.info(f"Total labeled paths: {len(path_to_label):,}")

    # Apply class limits
    random.seed(42)
    filtered = {}
    class_samples = defaultdict(list)
    for p, l in path_to_label.items():
        class_samples[l].append(p)

    for label, paths in class_samples.items():
        if len(paths) < GROUP_MIN_SAMPLES:
            logging.info(f"  Skipping {label}: only {len(paths)} samples")
            continue
        if len(paths) > GROUP_MAX_SAMPLES:
            paths = random.sample(paths, GROUP_MAX_SAMPLES)
        for p in paths:
            filtered[p] = label

    class_counts = Counter(filtered.values())
    logging.info(f"After filtering: {len(filtered):,} samples, {len(class_counts)} classes:")
    for label, count in class_counts.most_common():
        logging.info(f"  {label}: {count:,}")

    # Extract features
    paths = list(filtered.keys())
    features, valid_paths = extract_features(paths, num_workers, "Group features")

    if len(features) < 100:
        logging.error("Not enough features extracted")
        return {}

    labels = [filtered[p] for p in valid_paths]

    # Encode labels
    classes = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = torch.tensor([class_to_idx[l] for l in labels], dtype=torch.long)
    num_classes = len(classes)

    logging.info(f"Features: {features.shape}, Classes: {num_classes}")

    # Train/val/test split (stratified)
    indices = np.arange(len(features))
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=y.numpy(), random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=y[train_idx].numpy(), random_state=42)

    X_train, y_train = features[train_idx], y[train_idx]
    X_val, y_val = features[val_idx], y[val_idx]
    X_test, y_test = features[test_idx], y[test_idx]

    # Normalize
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0).clamp(min=1e-8)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Class weights
    counts = np.bincount(y_train.numpy(), minlength=num_classes)
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    # Train
    model = GroupClassifier(384, num_classes, HIDDEN_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += len(yb)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"  Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val: {val_acc:.4f} | Best: {best_val_acc:.4f}")

    # Test eval
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).argmax(1).cpu().numpy()
    y_test_np = y_test.numpy()

    from sklearn.metrics import classification_report
    logging.info("\n" + classification_report(y_test_np, preds, target_names=classes))

    # Save
    checkpoint = {
        'model_state': best_state,
        'input_dim': 384,
        'num_classes': num_classes,
        'hidden_dim': HIDDEN_DIM,
        'mean': mean,
        'std': std,
        'label_encoder_classes': classes,
        'pool_methods': ['mean', 'std', 'max'],
        'training_stats': {
            'total_samples': len(features),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'best_val_accuracy': best_val_acc,
            'class_counts': dict(class_counts),
        },
        'trained_at': datetime.now().isoformat(),
    }
    save_path = GROUP_OUTPUT_DIR / "model.pt"
    torch.save(checkpoint, save_path)
    logging.info(f"Saved group classifier to {save_path} (val acc: {best_val_acc:.4f})")

    return {'stage': 'group', 'classes': classes, 'val_acc': best_val_acc, 'samples': len(features)}


# ===================== STAGE 3: SUBGROUP CLASSIFIERS =====================

def train_subgroup_classifiers(manifest: dict, corrections: dict, epochs: int,
                               device: str, num_workers: int,
                               mix_paths: set = None) -> List[dict]:
    """Train per-group subgroup classifiers on isolated tracks only."""
    logging.info("\n" + "=" * 60)
    logging.info("STAGE 3: SUBGROUP CLASSIFIERS (isolated only)")
    logging.info("=" * 60)

    if mix_paths is None:
        mix_paths = _load_existing_detections()

    SUBGROUP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for group, valid_subgroups in GROUPS_WITH_SUBGROUPS.items():
        # Collect paths with known subgroups (isolated only)
        path_to_subgroup = {}
        for path, entry in manifest.items():
            if path in mix_paths:
                continue
            g, sg = get_label(path, entry, corrections)
            if g == group and sg in valid_subgroups:
                path_to_subgroup[path] = sg

        if not path_to_subgroup:
            logging.info(f"  {group}: no data, skipping")
            continue

        # Filter classes with enough samples
        class_counts = Counter(path_to_subgroup.values())
        valid_classes = sorted([c for c, n in class_counts.items() if n >= SUBGROUP_MIN_SAMPLES])

        if len(valid_classes) < SUBGROUP_MIN_CLASSES:
            logging.info(f"  {group}: only {len(valid_classes)} valid classes, skipping")
            continue

        path_to_subgroup = {p: s for p, s in path_to_subgroup.items() if s in valid_classes}
        logging.info(f"\n  {group}: {len(path_to_subgroup)} samples, classes: {dict(Counter(path_to_subgroup.values()))}")

        # Extract features
        paths = list(path_to_subgroup.keys())
        features, valid_paths = extract_features(paths, num_workers, f"{group} features")

        if len(features) < SUBGROUP_MIN_CLASSES * SUBGROUP_MIN_SAMPLES:
            logging.info(f"  {group}: only {len(features)} valid features, skipping")
            continue

        labels = [valid_classes.index(path_to_subgroup[p]) for p in valid_paths]
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        num_classes = len(valid_classes)

        # Train/val split
        n = len(labels)
        indices = torch.randperm(n)
        split = int(0.8 * n)
        train_idx, val_idx = indices[:split], indices[split:]

        X_train, y_train = features[train_idx], labels_tensor[train_idx]
        X_val, y_val = features[val_idx], labels_tensor[val_idx]

        mean = X_train.mean(dim=0)
        std = X_train.std(dim=0).clamp(min=1e-6)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

        # Weighted sampling
        cc = Counter(y_train.numpy().tolist())
        w = torch.tensor([1.0 / cc.get(i, 1) for i in range(num_classes)])
        sampler = WeightedRandomSampler(w[y_train], len(y_train))

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=min(64, len(X_train)), sampler=sampler, drop_last=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=min(64, len(X_val)))

        model = SubgroupClassifier(384, HIDDEN_DIM, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state = None

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total += len(yb)
            val_acc = correct / total if total > 0 else 0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logging.info(f"    Epoch {epoch+1}/{epochs} | Val: {val_acc:.4f} | Best: {best_val_acc:.4f}")

        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        checkpoint = {
            'group': group,
            'model_state': best_state,
            'input_dim': 384,
            'num_classes': num_classes,
            'hidden_dim': HIDDEN_DIM,
            'mean': mean,
            'std': std,
            'classes': valid_classes,
            'best_val_accuracy': best_val_acc,
            'trained_at': datetime.now().isoformat(),
        }
        torch.save(checkpoint, SUBGROUP_OUTPUT_DIR / f"{group}_subgroup_model.pt")
        logging.info(f"  Saved {group} subgroup model (val acc: {best_val_acc:.4f})")

        results.append({'stage': 'subgroup', 'group': group, 'classes': valid_classes,
                        'val_acc': best_val_acc, 'samples': len(features)})

    return results


# ===================== STAGE 1: MIX CLASSIFIER =====================

def train_mix_classifier(manifest: dict, corrections: dict, epochs: int,
                         device: str, num_workers: int) -> dict:
    """Train multi-label mix classifier.

    Training strategy:
    - Solo tracks: single label (the instrument group)
    - Mix tracks: multi-label from corrections or from session metadata
    Uses BCEWithLogitsLoss for multi-label classification.
    """
    logging.info("\n" + "=" * 60)
    logging.info("STAGE 1: MIX CLASSIFIER")
    logging.info("=" * 60)

    MIX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classes = MIX_TARGET_CLASSES
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    # Collect training data: both solo tracks (single label) and mixes (multi-label)
    path_to_labels = {}  # path -> set of class indices

    for path, entry in manifest.items():
        group, _ = get_label(path, entry, corrections)
        is_mix = entry.get('is_mix', False)

        if group in GROUP_EXCLUDED:
            continue

        if is_mix:
            # For mixes, check if corrections provide multi-label info
            if path in corrections:
                corr = corrections[path]
                if corr.get('multi_label') and corr.get('labels'):
                    label_set = set()
                    for lbl in corr['labels']:
                        if lbl in class_to_idx:
                            label_set.add(class_to_idx[lbl])
                    if label_set:
                        path_to_labels[path] = label_set
                        continue

            # Mix without multi-label correction — skip for training
            continue
        else:
            # Solo track: single label
            if group in class_to_idx:
                path_to_labels[path] = {class_to_idx[group]}

    # Cap solo samples per class
    class_paths = defaultdict(list)
    for p, label_set in path_to_labels.items():
        if len(label_set) == 1:
            cls_idx = next(iter(label_set))
            class_paths[cls_idx].append(p)

    random.seed(42)
    capped_paths = {}
    for cls_idx, paths in class_paths.items():
        if len(paths) < MIX_MIN_SOLO_SAMPLES:
            logging.info(f"  Skipping {classes[cls_idx]}: only {len(paths)} solo samples")
            continue
        if len(paths) > GROUP_MAX_SAMPLES:
            paths = random.sample(paths, GROUP_MAX_SAMPLES)
        for p in paths:
            capped_paths[p] = path_to_labels[p]

    # Add multi-label entries (not capped)
    for p, label_set in path_to_labels.items():
        if len(label_set) > 1:
            capped_paths[p] = label_set

    logging.info(f"Mix training samples: {len(capped_paths):,}")
    multi_label_count = sum(1 for v in capped_paths.values() if len(v) > 1)
    logging.info(f"  Multi-label: {multi_label_count:,}, Solo: {len(capped_paths) - multi_label_count:,}")

    # Extract features
    paths = list(capped_paths.keys())
    features, valid_paths = extract_features(paths, num_workers, "Mix features")

    if len(features) < 100:
        logging.error("Not enough features for mix classifier")
        return {}

    # Build multi-label targets
    y = torch.zeros(len(valid_paths), num_classes)
    for i, p in enumerate(valid_paths):
        for cls_idx in capped_paths[p]:
            y[i, cls_idx] = 1.0

    # Train/val split
    n = len(features)
    indices = torch.randperm(n)
    split = int(0.85 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = features[train_idx], y[train_idx]
    X_val, y_val = features[val_idx], y[val_idx]

    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0).clamp(min=1e-8)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = MixClassifier(384, HIDDEN_DIM, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validate with F1
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = torch.sigmoid(model(xb)).cpu()
                all_preds.append(preds)
                all_targets.append(yb)

        preds_cat = torch.cat(all_preds) > 0.5
        targets_cat = torch.cat(all_targets).bool()

        # Per-class F1
        tp = (preds_cat & targets_cat).float().sum(0)
        fp = (preds_cat & ~targets_cat).float().sum(0)
        fn = (~preds_cat & targets_cat).float().sum(0)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        macro_f1 = f1.mean().item()

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"  Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | F1: {macro_f1:.4f} | Best: {best_val_f1:.4f}")

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Per-class results
    logging.info("\nPer-class F1:")
    for i, cls in enumerate(classes):
        logging.info(f"  {cls}: F1={f1[i]:.4f} P={precision[i]:.4f} R={recall[i]:.4f}")

    checkpoint = {
        'transform_type': 'mlp_v2',
        'model_state': best_state,
        'input_dim': 384,
        'num_classes': num_classes,
        'hidden_dim': HIDDEN_DIM,
        'X_mean': mean,
        'X_std': std,
        'target_classes': classes,
        'best_val_f1': best_val_f1,
        'trained_at': datetime.now().isoformat(),
    }
    save_path = MIX_OUTPUT_DIR / "mix_classifier.pt"
    torch.save(checkpoint, save_path)
    logging.info(f"Saved mix classifier to {save_path} (F1: {best_val_f1:.4f})")

    return {'stage': 'mix', 'classes': classes, 'val_f1': best_val_f1, 'samples': len(features)}


# ===================== MIX DETECTION (INFERENCE) =====================

def detect_mixes(manifest: dict, device: str, num_workers: int) -> set:
    """Run mix classifier inference on full manifest to identify mixes.

    Files with 2+ instrument classes above threshold are detected as mixes.
    Saves results to ensemble_detections.json for group/subgroup filtering.
    Returns set of detected mix paths.
    """
    logging.info("\n" + "=" * 60)
    logging.info("MIX DETECTION (inference on full manifest)")
    logging.info("=" * 60)

    model_path = MIX_OUTPUT_DIR / "mix_classifier.pt"
    if not model_path.exists():
        logging.warning(f"Mix classifier not found at {model_path}, skipping detection")
        return _load_existing_detections()

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    classes = checkpoint['target_classes']
    num_classes = len(classes)

    model = MixClassifier(384, checkpoint.get('hidden_dim', HIDDEN_DIM), num_classes)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()

    mean = checkpoint['X_mean'].to(device)
    std = checkpoint['X_std'].to(device)

    # Collect all paths with potential latents
    all_paths = [p for p in manifest.keys()
                 if manifest[p].get('group', 'undefined') not in GROUP_EXCLUDED]
    logging.info(f"Checking {len(all_paths):,} entries...")

    # Extract features
    features, valid_paths = extract_features(all_paths, num_workers, "Mix detection")

    if len(features) == 0:
        logging.warning("No features extracted for mix detection")
        return _load_existing_detections()

    # Run inference in batches
    X = features.to(device)
    X_norm = (X - mean) / std

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_norm), BATCH_SIZE):
            batch = X_norm[i:i + BATCH_SIZE]
            logits = model(batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs)

    # Detect mixes: 2+ classes above threshold
    detected = []
    for i, (path, probs) in enumerate(zip(valid_paths, all_probs)):
        above = (probs >= MIX_DETECT_THRESHOLD).sum().item()
        if above >= MIX_MIN_CLASSES_FOR_MIX:
            max_prob = probs.max().item()
            detected.append({
                'path': path,
                'ensemble_probability': float(max_prob),
                'filename': Path(path).name,
                'num_classes_detected': above,
            })

    detected.sort(key=lambda x: -x['ensemble_probability'])
    detected_paths = {d['path'] for d in detected}

    logging.info(f"Detected {len(detected):,} mixes out of {len(valid_paths):,} checked")

    # Save
    ENSEMBLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'threshold': MIX_DETECT_THRESHOLD,
        'min_classes': MIX_MIN_CLASSES_FOR_MIX,
        'total_checked': len(valid_paths),
        'detected_count': len(detected),
        'detected': detected,
        'detected_at': datetime.now().isoformat(),
    }
    with open(ENSEMBLE_DETECTIONS_PATH, 'wb') as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    logging.info(f"Saved detections to {ENSEMBLE_DETECTIONS_PATH}")

    return detected_paths


def _load_existing_detections() -> set:
    """Load existing ensemble_detections.json if available."""
    if ENSEMBLE_DETECTIONS_PATH.exists():
        with open(ENSEMBLE_DETECTIONS_PATH, 'rb') as f:
            data = orjson.loads(f.read())
        paths = {e['path'] for e in data.get('detected', [])}
        logging.info(f"Loaded {len(paths):,} existing mix detections from {ENSEMBLE_DETECTIONS_PATH}")
        return paths
    logging.warning("No existing ensemble_detections.json found — group/subgroup will use filename heuristics only")
    return set()


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description='Train group, subgroup, and mix classifiers')
    parser.add_argument('--stage', choices=['all', 'group', 'subgroup', 'mix'], default='all')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device: {device}")
    logging.info(f"Epochs: {args.epochs}")

    manifest = load_manifest()
    corrections = load_corrections()

    results = []
    mix_paths = set()

    # Stage 1: Mix classifier (trains first — group/subgroup depend on its detections)
    if args.stage in ('all', 'mix'):
        r = train_mix_classifier(manifest, corrections, args.epochs, device, args.workers)
        if r:
            results.append(r)
        # Run inference to detect mixes in full manifest
        mix_paths = detect_mixes(manifest, device, args.workers)
    else:
        # Load existing detections for group/subgroup filtering
        mix_paths = _load_existing_detections()

    # Stage 2: Group classifier (isolated tracks only)
    if args.stage in ('all', 'group'):
        r = train_group_classifier(manifest, corrections, args.epochs, device, args.workers,
                                   mix_paths=mix_paths)
        if r:
            results.append(r)

    # Stage 3: Subgroup classifiers (isolated tracks only)
    if args.stage in ('all', 'subgroup'):
        rs = train_subgroup_classifiers(manifest, corrections, args.epochs, device, args.workers,
                                        mix_paths=mix_paths)
        results.extend(rs)

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("TRAINING SUMMARY")
    logging.info("=" * 60)
    for r in results:
        stage = r.get('stage', '?')
        if stage == 'group':
            logging.info(f"  Group: {len(r['classes'])} classes, {r['samples']:,} samples, val_acc={r['val_acc']:.4f}")
        elif stage == 'subgroup':
            logging.info(f"  Subgroup/{r['group']}: {r['classes']}, {r['samples']:,} samples, val_acc={r['val_acc']:.4f}")
        elif stage == 'mix':
            logging.info(f"  Mix: {len(r['classes'])} classes, {r['samples']:,} samples, val_f1={r['val_f1']:.4f}")

    logging.info("\nDone!")


if __name__ == "__main__":
    main()
