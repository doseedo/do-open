#!/usr/bin/env python3
"""
Train per-group subgroup classifiers.

Usage:
  python3 train_subgroup_classifiers.py --manifest /path/to/manifest.json --corrections /path/to/corrections.json
  python3 train_subgroup_classifiers.py --epochs 50 --device cuda
"""

import argparse
import logging
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import orjson
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ===================== CONFIG =====================

MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/unified_manifest.json")
CORRECTIONS_PATH = Path("/home/arlo/gcs-bucket/Manifests/corrections.json")
OUTPUT_DIR = Path("/home/arlo/Data/subgroup_classifiers")
LATENTS_ROOT = Path("/home/arlo/gcs-bucket/Latents")
AUDIO_ROOT = Path("/home/arlo/gcs-bucket")

GROUPS_WITH_SUBGROUPS = {
    'brass': ['french_horn', 'trombone', 'trumpet', 'tuba', 'flugelhorn', 'brass_section'],
    'strings': ['cello', 'viola', 'violin', 'double_bass', 'string_section'],
    'winds': ['clarinet', 'flute', 'oboe', 'saxophone', 'bassoon'],
    'bass': ['electric_bass', 'upright_bass', 'synth_bass'],
    'guitar': ['acoustic_guitar', 'electric_guitar', 'classical_guitar'],
    'piano': ['grand_piano', 'upright_piano', 'electric_piano'],
}

MIN_SAMPLES_PER_CLASS = 5
MIN_CLASSES = 2

# ===================== MODEL =====================

class SubgroupClassifier(nn.Module):
    """Same architecture as run_subgroup_classifiers.py for checkpoint compat."""
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
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ===================== FEATURE EXTRACTION =====================

LATENT_SILENT_THRESHOLD = 0.01

def audio_path_to_latent_path(audio_path: str) -> Optional[Path]:
    """Convert audio file path to corresponding latent path."""
    audio_path = Path(audio_path)
    try:
        rel_path = audio_path.relative_to(AUDIO_ROOT)
    except ValueError:
        parts = audio_path.parts
        if 'protools' in parts:
            idx = parts.index('protools')
            rel_path = Path(*parts[idx:])
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


def load_and_extract(audio_path: str) -> Optional[torch.Tensor]:
    """Load latent and extract 384-dim feature vector."""
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
        stacked = torch.stack(pools, dim=-1)  # [8, 16, 3]
        return stacked.flatten()  # [384]
    except Exception:
        return None


# ===================== DATA LOADING =====================

def load_manifest(manifest_path: Path) -> dict:
    """Load manifest with orjson."""
    logging.info(f"Loading manifest: {manifest_path}")
    with open(manifest_path, 'rb') as f:
        data = orjson.loads(f.read())
    entries = data.get('entries', data) if isinstance(data, dict) else data
    if isinstance(entries, list):
        entries = {e.get('path', ''): e for e in entries if isinstance(e, dict) and 'path' in e}
    logging.info(f"  {len(entries)} entries")
    return entries


def load_corrections(corrections_path: Path) -> dict:
    """Load corrections."""
    if not corrections_path.exists():
        return {}
    with open(corrections_path, 'rb') as f:
        corrections = orjson.loads(f.read())
    logging.info(f"  {len(corrections)} corrections")
    return corrections


def build_group_data(
    group: str,
    valid_subgroups: List[str],
    manifest: dict,
    corrections: dict,
    num_workers: int = 12,
) -> Tuple[torch.Tensor, List[int], List[str], List[str]]:
    """Build training data for a single group.

    Returns: (features_tensor, labels, classes, paths)
    """
    # Collect paths with subgroup labels
    path_to_subgroup = {}

    # From manifest: entries matching this group with a known subgroup
    for path, entry in manifest.items():
        entry_group = entry.get('group', '')
        entry_subgroup = entry.get('subgroup', '')
        if entry_group == group and entry_subgroup in valid_subgroups:
            path_to_subgroup[path] = entry_subgroup

    # Corrections override manifest
    for path, corr in corrections.items():
        corr_group = corr.get('group', '')
        corr_subgroup = corr.get('subgroup', '')
        if corr_group == group and corr_subgroup in valid_subgroups:
            path_to_subgroup[path] = corr_subgroup
        elif corr_group == group and corr_subgroup not in valid_subgroups:
            # Correction says different group or unknown subgroup — remove if it was there
            path_to_subgroup.pop(path, None)

    if not path_to_subgroup:
        return None, None, None, None

    # Filter to classes with enough samples
    class_counts = Counter(path_to_subgroup.values())
    valid_classes = sorted([c for c, n in class_counts.items() if n >= MIN_SAMPLES_PER_CLASS])

    if len(valid_classes) < MIN_CLASSES:
        return None, None, None, None

    # Filter to valid classes
    path_to_subgroup = {p: s for p, s in path_to_subgroup.items() if s in valid_classes}

    logging.info(f"  {group}: {len(path_to_subgroup)} samples across {valid_classes}")

    # Extract features in parallel
    paths = list(path_to_subgroup.keys())
    features_list = []
    valid_paths = []

    processed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_and_extract, p): p for p in paths}
        for future in as_completed(futures):
            path = futures[future]
            feat = future.result()
            if feat is not None:
                features_list.append(feat)
                valid_paths.append(path)
            processed += 1
            if processed % 500 == 0:
                logging.info(f"    {processed}/{len(paths)} extracted")

    if len(features_list) < MIN_CLASSES * MIN_SAMPLES_PER_CLASS:
        logging.warning(f"  {group}: only {len(features_list)} valid samples, skipping")
        return None, None, None, None

    features = torch.stack(features_list)
    labels = [valid_classes.index(path_to_subgroup[p]) for p in valid_paths]

    logging.info(f"  {group}: {len(features)} features extracted, classes: {dict(Counter(valid_classes[l] for l in labels))}")

    return features, labels, valid_classes, valid_paths


# ===================== TRAINING =====================

def train_one_group(
    group: str,
    features: torch.Tensor,
    labels: List[int],
    classes: List[str],
    epochs: int,
    device: str,
    output_dir: Path,
) -> dict:
    """Train subgroup classifier for one group."""
    logging.info(f"\n{'='*50}")
    logging.info(f"Training {group} subgroup classifier")
    logging.info(f"  Classes: {classes}")
    logging.info(f"  Samples: {len(labels)}")
    logging.info(f"{'='*50}")

    num_classes = len(classes)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Train/val split (80/20)
    n = len(labels)
    indices = torch.randperm(n)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = features[train_idx], labels_tensor[train_idx]
    X_val, y_val = features[val_idx], labels_tensor[val_idx]

    # Normalize
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0).clamp(min=1e-6)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Weighted sampling for class imbalance
    class_counts = Counter(y_train.numpy().tolist())
    weights = torch.tensor([1.0 / class_counts.get(i, 1) for i in range(num_classes)])
    sample_weights = weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=min(64, len(X_train)),
        sampler=sampler,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=min(64, len(X_val)),
        shuffle=False,
    )

    model = SubgroupClassifier(input_dim=384, hidden_dim=256, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        val_acc = correct / total if total > 0 else 0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(f"  Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")

    # Save checkpoint
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    checkpoint = {
        'group': group,
        'model_state': best_state,
        'input_dim': 384,
        'num_classes': num_classes,
        'hidden_dim': 256,
        'mean': mean,
        'std': std,
        'classes': classes,
        'best_val_accuracy': best_val_acc,
        'trained_at': datetime.now().isoformat(),
    }

    save_path = output_dir / f"{group}_subgroup_model.pt"
    torch.save(checkpoint, save_path)
    logging.info(f"  Saved {save_path} | Best val accuracy: {best_val_acc:.4f}")

    return {
        'group': group,
        'classes': classes,
        'samples': len(labels),
        'best_val_accuracy': best_val_acc,
    }


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description='Train per-group subgroup classifiers')
    parser.add_argument('--manifest', type=str, default=str(MANIFEST_PATH))
    parser.add_argument('--corrections', type=str, default=str(CORRECTIONS_PATH))
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        datefmt='%H:%M:%S')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device: {device}")

    manifest = load_manifest(Path(args.manifest))
    corrections = load_corrections(Path(args.corrections))

    results = []
    for group, valid_subgroups in GROUPS_WITH_SUBGROUPS.items():
        features, labels, classes, paths = build_group_data(
            group, valid_subgroups, manifest, corrections, num_workers=args.workers
        )
        if features is None:
            logging.warning(f"Skipping {group}: not enough data")
            continue

        result = train_one_group(
            group, features, labels, classes,
            epochs=args.epochs, device=device, output_dir=output_dir,
        )
        results.append(result)

    # Summary
    logging.info("\n" + "=" * 50)
    logging.info("TRAINING SUMMARY")
    logging.info("=" * 50)
    for r in results:
        logging.info(f"  {r['group']}: {r['classes']} | {r['samples']} samples | acc: {r['best_val_accuracy']:.4f}")

    logging.info("Training complete")


if __name__ == "__main__":
    main()
