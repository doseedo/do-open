#!/usr/bin/env python3
"""
Latent Subgroup Classifier

Classifies audio files into subgroups within a known group.
E.g., within "brass" -> trumpet, trombone, french_horn, tuba

Usage:
  # Train all subgroup classifiers
  python latent_subgroup_classifier.py --mode train \
    --manifest /home/arlo/gcs-bucket/Manifests/combined_manifest.json \
    --output-dir /home/arlo/Data/subgroup_classifiers

  # Classify undefined subgroups within a specific group
  python latent_subgroup_classifier.py --mode classify \
    --group brass \
    --model-dir /home/arlo/Data/subgroup_classifiers \
    --undefined /home/arlo/undefined_audio_paths.txt \
    --output-dir /home/arlo/Data/subgroup_classifiers
"""

import argparse
import json
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
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ===================== CONFIGURATION =====================

LATENTS_ROOT = Path("/home/arlo/gcs-bucket/Latents")
AUDIO_ROOT = Path("/home/arlo/gcs-bucket")

CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.65

MAX_SAMPLES_PER_CLASS = 3000
MIN_SAMPLES_PER_CLASS = 30  # Lower threshold for subgroups
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
HIDDEN_DIM = 256

POOL_METHODS = ['mean', 'std', 'max']

# Groups with meaningful subgroups (exclude 'undefined' heavy ones)
TRAINABLE_GROUPS = {
    'brass': ['trumpet', 'trombone', 'french_horn', 'tuba'],
    'guitar': ['electric_guitar', 'acoustic_guitar'],
    'piano': ['acoustic_piano', 'keys', 'electric_piano'],
    'strings': ['violin', 'cello', 'viola'],
    'winds': ['sax', 'clarinet', 'flute', 'oboe'],
    'bass': ['electric_bass', 'upright_bass', 'synth_bass'],
}


# ===================== PATH CONVERSION =====================

def audio_path_to_latent_path(audio_path: str) -> Path:
    audio_path = Path(audio_path)
    try:
        rel_path = audio_path.relative_to(AUDIO_ROOT)
    except ValueError:
        parts = audio_path.parts
        if 'gcs-bucket' in parts:
            idx = parts.index('gcs-bucket')
            rel_path = Path(*parts[idx+1:])
        else:
            rel_path = audio_path
    return LATENTS_ROOT / rel_path.with_suffix('.pt')


def load_latent(latent_path: Path) -> Optional[torch.Tensor]:
    try:
        data = torch.load(latent_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            return data['latents']
        return data
    except Exception:
        return None


def pool_latent(latent: torch.Tensor) -> torch.Tensor:
    features = []
    if 'mean' in POOL_METHODS:
        features.append(latent.mean(dim=-1))
    if 'std' in POOL_METHODS:
        features.append(latent.std(dim=-1))
    if 'max' in POOL_METHODS:
        features.append(latent.max(dim=-1)[0])
    stacked = torch.stack(features, dim=-1)
    return stacked.flatten()


# ===================== MODEL =====================

class SubgroupClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = HIDDEN_DIM):
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
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ===================== FEATURE EXTRACTION =====================

def extract_features_batch(audio_paths: List[str], num_workers: int = 8) -> Tuple[torch.Tensor, List[str], List[str]]:
    features_list = []
    valid_paths = []
    failed_paths = []

    def process_one(audio_path: str) -> Tuple[Optional[torch.Tensor], str]:
        latent_path = audio_path_to_latent_path(audio_path)
        latent = load_latent(latent_path)
        if latent is not None:
            return pool_latent(latent), audio_path
        return None, audio_path

    total = len(audio_paths)
    start_time = datetime.now()
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
            if processed % 1000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = processed / elapsed if elapsed > 0 else 0
                logging.info(f"  {processed}/{total} ({rate:.1f}/s)")

    elapsed = (datetime.now() - start_time).total_seconds()
    logging.info(f"  Complete: {len(valid_paths)} success, {len(failed_paths)} failed ({elapsed:.1f}s)")

    if len(features_list) == 0:
        return torch.tensor([]), [], failed_paths

    return torch.stack(features_list), valid_paths, failed_paths


# ===================== TRAINING =====================

def train_subgroup_classifier(group: str, subgroups: List[str],
                               manifest: Dict, output_dir: Path,
                               num_workers: int = 8, device: str = 'cuda') -> Optional[Dict]:
    """Train a subgroup classifier for a specific group."""

    logging.info(f"\n{'='*60}")
    logging.info(f"Training subgroup classifier for: {group}")
    logging.info(f"Subgroups: {subgroups}")
    logging.info(f"{'='*60}")

    # Collect paths for this group's subgroups
    file_label_pairs = []
    for audio_path, meta in manifest.items():
        if not isinstance(meta, dict):
            continue
        if meta.get('group') != group:
            continue
        subgroup = meta.get('subgroup', 'undefined')
        if subgroup not in subgroups:
            continue
        if '/New/' not in audio_path:
            continue
        # Verify latent exists
        latent_path = audio_path_to_latent_path(audio_path)
        if latent_path.exists():
            file_label_pairs.append((audio_path, subgroup))

    if len(file_label_pairs) < 100:
        logging.warning(f"Not enough samples for {group}: {len(file_label_pairs)}")
        return None

    # Count and filter by class
    class_counts = Counter(label for _, label in file_label_pairs)
    logging.info(f"Found {len(file_label_pairs)} samples:")
    for sg, count in class_counts.most_common():
        logging.info(f"  {sg}: {count}")

    # Apply limits
    import random
    random.seed(42)

    class_samples = defaultdict(list)
    for path, label in file_label_pairs:
        class_samples[label].append(path)

    filtered_pairs = []
    for label, paths in class_samples.items():
        if len(paths) < MIN_SAMPLES_PER_CLASS:
            logging.warning(f"  Skipping {label}: only {len(paths)} samples")
            continue
        if len(paths) > MAX_SAMPLES_PER_CLASS:
            paths = random.sample(paths, MAX_SAMPLES_PER_CLASS)
        for path in paths:
            filtered_pairs.append((path, label))

    if len(filtered_pairs) < 50:
        logging.warning(f"Not enough filtered samples for {group}")
        return None

    # Extract features
    audio_paths = [p for p, _ in filtered_pairs]
    labels = [l for _, l in filtered_pairs]

    logging.info(f"Extracting features for {len(audio_paths)} samples...")
    X, valid_paths, failed = extract_features_batch(audio_paths, num_workers=num_workers)

    if len(X) < 50:
        logging.warning(f"Not enough valid samples for {group}: {len(X)}")
        return None

    # Match labels to valid paths
    path_to_label = dict(filtered_pairs)
    valid_labels = [path_to_label[p] for p in valid_paths]

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(valid_labels)
    y = torch.tensor(y, dtype=torch.long)

    num_classes = len(label_encoder.classes_)
    input_dim = X.shape[1]

    if num_classes < 2:
        logging.warning(f"Only {num_classes} class for {group}, skipping")
        return None

    logging.info(f"Training with {len(X)} samples, {num_classes} classes")

    # Split
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.15,
                                            stratify=y.numpy(), random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1,
                                           stratify=y[train_idx].numpy(), random_state=42)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Normalize
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Class weights
    class_counts_arr = np.bincount(y_train.numpy())
    class_weights = 1.0 / (class_counts_arr + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Model
    model = SubgroupClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_val_acc = 0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                val_total += len(batch_y)

        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            logging.info(f"  Epoch {epoch+1}: val_acc={val_acc:.3f}")

    # Evaluate
    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        logits = model(X_test.to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()

    y_test_np = y_test.numpy()

    logging.info(f"\nClassification Report for {group}:")
    logging.info(classification_report(y_test_np, y_pred, target_names=label_encoder.classes_))

    # Save model
    model_data = {
        'group': group,
        'model_state': best_model_state,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'hidden_dim': HIDDEN_DIM,
        'mean': mean,
        'std': std,
        'classes': label_encoder.classes_.tolist(),
        'best_val_accuracy': best_val_acc,
        'trained_at': datetime.now().isoformat()
    }

    model_path = output_dir / f'{group}_subgroup_model.pt'
    torch.save(model_data, model_path)
    logging.info(f"Model saved to {model_path}")

    return model_data


def train_all_subgroup_classifiers(manifest_path: Path, output_dir: Path,
                                    num_workers: int = 8, device: str = 'cuda'):
    """Train subgroup classifiers for all groups with meaningful subgroups."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, using CPU")
        device = 'cpu'

    logging.info(f"Loading manifest from {manifest_path}...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    trained_models = {}

    for group, subgroups in TRAINABLE_GROUPS.items():
        model_data = train_subgroup_classifier(
            group, subgroups, manifest, output_dir,
            num_workers=num_workers, device=device
        )
        if model_data:
            trained_models[group] = model_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary
    summary = {
        'groups_trained': list(trained_models.keys()),
        'models': {g: {'classes': m['classes'], 'accuracy': m['best_val_accuracy']}
                   for g, m in trained_models.items()},
        'trained_at': datetime.now().isoformat()
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"\n{'='*60}")
    logging.info("TRAINING COMPLETE")
    logging.info(f"{'='*60}")
    logging.info(f"Trained {len(trained_models)} subgroup classifiers:")
    for group, model in trained_models.items():
        logging.info(f"  {group}: {model['classes']} (acc: {model['best_val_accuracy']:.1%})")


# ===================== CLASSIFICATION =====================

def classify_subgroups(group: str, model_dir: Path, undefined_paths: List[str],
                       output_path: Path, num_workers: int = 8, device: str = 'cuda',
                       manifest_path: Path = None) -> Dict:
    """Classify undefined subgroups for a specific group.

    If manifest_path is provided, extract paths with this group but undefined subgroup.
    """

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # If manifest provided, extract paths for this group with undefined subgroup
    if manifest_path and manifest_path.exists():
        logging.info(f"Loading paths from manifest for group '{group}' with undefined subgroup...")
        with open(manifest_path) as f:
            manifest = json.load(f)

        undefined_paths = []
        skipped_no_latent = 0
        for path, meta in manifest.items():
            if not isinstance(meta, dict):
                continue
            if meta.get('group') != group:
                continue
            if meta.get('subgroup', 'undefined') != 'undefined':
                continue  # Already has subgroup
            if '/New/' not in path:
                continue
            # Check latent exists
            latent_path = audio_path_to_latent_path(path)
            if latent_path.exists():
                undefined_paths.append(path)
            else:
                skipped_no_latent += 1

        logging.info(f"Found {len(undefined_paths)} '{group}' paths with undefined subgroup and latents")
        logging.info(f"Skipped {skipped_no_latent} without latents")

    model_path = model_dir / f'{group}_subgroup_model.pt'
    if not model_path.exists():
        raise ValueError(f"No model found for group: {group}")

    logging.info(f"Loading model for {group}...")
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)

    input_dim = model_data['input_dim']
    num_classes = model_data['num_classes']
    mean = model_data['mean']
    std = model_data['std']
    classes = model_data['classes']

    model = SubgroupClassifier(input_dim, num_classes)
    model.load_state_dict(model_data['model_state'])
    model.to(device)
    model.eval()

    logging.info(f"Subgroup classes: {classes}")

    if len(undefined_paths) == 0:
        logging.warning(f"No paths to classify for {group}")
        return {'predictions': [], 'group': group}

    # Extract features
    X, valid_paths, failed = extract_features_batch(undefined_paths, num_workers=num_workers)

    if len(X) == 0:
        logging.error("No features extracted!")
        return {}

    # Normalize and predict
    X = (X - mean) / std

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

    # Build results
    results = []
    for i, (path, pred_idx, conf) in enumerate(zip(valid_paths, y_pred, max_proba)):
        results.append({
            'path': path,
            'group': group,
            'predicted_subgroup': classes[pred_idx],
            'confidence': float(conf),
            'all_probabilities': {c: float(p) for c, p in zip(classes, all_probs[i])}
        })

    # Summary
    high_conf = [r for r in results if r['confidence'] >= CONFIDENCE_HIGH]
    med_conf = [r for r in results if CONFIDENCE_MEDIUM <= r['confidence'] < CONFIDENCE_HIGH]
    low_conf = [r for r in results if r['confidence'] < CONFIDENCE_MEDIUM]

    logging.info(f"\nClassification Results for {group}:")
    logging.info(f"  High confidence (>={CONFIDENCE_HIGH:.0%}): {len(high_conf)}")
    logging.info(f"  Medium confidence: {len(med_conf)}")
    logging.info(f"  Low confidence: {len(low_conf)}")
    logging.info(f"  Failed to load: {len(failed)}")

    by_subgroup = Counter(r['predicted_subgroup'] for r in high_conf)
    logging.info(f"\nHigh-confidence by subgroup:")
    for sg, count in by_subgroup.most_common():
        logging.info(f"  {sg}: {count}")

    # Save
    output_data = {
        'group': group,
        'predictions': results,
        'failed_paths': failed,
        'summary': {
            'total': len(results),
            'high_confidence': len(high_conf),
            'medium_confidence': len(med_conf),
            'low_confidence': len(low_conf),
            'by_subgroup': dict(Counter(r['predicted_subgroup'] for r in results))
        },
        'classified_at': datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"\nSaved to {output_path}")

    return output_data


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(
        description='Latent-space subgroup classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--mode', choices=['train', 'classify'], required=True)
    parser.add_argument('--manifest', type=str,
                        help='Manifest JSON (for training, or classify mode to find entries with group but undefined subgroup)')
    parser.add_argument('--group', type=str,
                        help='Group to classify subgroups for (e.g., brass, guitar). Use "all" to classify all groups.')
    parser.add_argument('--model-dir', type=str,
                        help='Directory with trained models')
    parser.add_argument('--undefined', type=str,
                        help='File with undefined audio paths (optional if --manifest provided in classify mode)')
    parser.add_argument('--output-dir', type=str, default='./subgroup_classifiers')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

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
        train_all_subgroup_classifiers(
            Path(args.manifest),
            output_dir,
            num_workers=args.workers,
            device=args.device
        )

    elif args.mode == 'classify':
        if not args.group or not args.model_dir:
            parser.error("--group and --model-dir required for classify mode")
        if not args.undefined and not args.manifest:
            parser.error("Either --undefined or --manifest required for classify mode")

        # Load undefined paths from file if provided
        undefined_paths = []
        if args.undefined:
            with open(args.undefined) as f:
                undefined_paths = [l.strip() for l in f if l.strip() and not l.startswith('#')]

        manifest_path = Path(args.manifest) if args.manifest else None

        # Handle "all" groups
        groups_to_classify = list(TRAINABLE_GROUPS.keys()) if args.group == 'all' else [args.group]

        for group in groups_to_classify:
            logging.info(f"\n{'='*60}")
            logging.info(f"Classifying subgroups for: {group}")
            logging.info(f"{'='*60}")

            output_path = output_dir / f'{group}_subgroup_predictions.json'
            try:
                classify_subgroups(
                    group,
                    Path(args.model_dir),
                    undefined_paths,
                    output_path,
                    num_workers=args.workers,
                    device=args.device,
                    manifest_path=manifest_path
                )
            except ValueError as e:
                logging.warning(f"Skipping {group}: {e}")
                continue


if __name__ == '__main__':
    main()
