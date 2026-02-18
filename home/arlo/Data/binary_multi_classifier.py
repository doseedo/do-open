#!/usr/bin/env python3
"""
Binary Multi-Instrument Classifier

Classifies audio as single-instrument or multi-instrument based on latent features.
Uses corrections data to identify training samples.

Usage:
  # Train from corrections
  python binary_multi_classifier.py --mode train \
    --corrections /home/arlo/gcs-bucket/Manifests/corrections.json \
    --output-dir /home/arlo/Data/binary_classifier

  # Classify files
  python binary_multi_classifier.py --mode classify \
    --model /home/arlo/Data/binary_classifier/model.pt \
    --predictions /home/arlo/Data/latent_classifier/predictions.json \
    --output /home/arlo/Data/binary_classifier/multi_predictions.json

  # Filter predictions - add is_multi field to existing predictions
  python binary_multi_classifier.py --mode filter \
    --model /home/arlo/Data/binary_classifier/model.pt \
    --predictions /home/arlo/Data/latent_classifier/predictions.json
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ===================== CONFIGURATION =====================

LATENTS_ROOT = Path("/home/arlo/gcs-bucket/Latents")
AUDIO_ROOT = Path("/home/arlo/gcs-bucket")

# Feature extraction settings (same as main classifier)
POOL_METHODS = ['mean', 'std', 'max']
SILENT_FRAME_THRESHOLD = 0.01

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
HIDDEN_DIM = 128  # Smaller model for binary task

# Groups that are inherently multi-instrument
MULTI_GROUPS = {'ensemble', 'full-track'}

# Groups to exclude (not useful for single/multi distinction)
EXCLUDED_GROUPS = {'silent', 'junk', 'undefined', 'room', 'fx', 'click'}


# ===================== PATH CONVERSION =====================

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
        elif 'protoolsA' in parts:
            idx = parts.index('protoolsA')
            rel_path = Path(*parts[idx:])
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
    """Load latent tensor from file."""
    try:
        data = torch.load(latent_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            return data['latents']
        return data
    except Exception:
        return None


def detect_silent_frames(latent: torch.Tensor, threshold: float = SILENT_FRAME_THRESHOLD) -> torch.Tensor:
    """Detect silent frames in latent based on energy."""
    energy = torch.sqrt((latent ** 2).mean(dim=(0, 1)))
    return energy > threshold


def pool_latent(latent: torch.Tensor, mask_silent: bool = True) -> torch.Tensor:
    """Pool latent [8, 16, T] to fixed-size feature vector."""
    if mask_silent and latent.shape[-1] > 1:
        non_silent_mask = detect_silent_frames(latent)
        if non_silent_mask.sum() > 0:
            latent = latent[:, :, non_silent_mask]

    features = []
    if 'mean' in POOL_METHODS:
        features.append(latent.mean(dim=-1))
    if 'std' in POOL_METHODS:
        features.append(latent.std(dim=-1))
    if 'max' in POOL_METHODS:
        features.append(latent.max(dim=-1)[0])

    stacked = torch.stack(features, dim=-1)
    return stacked.flatten()


# ===================== DATASET =====================

class BinaryDataset(Dataset):
    """Dataset for binary single/multi classification."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ===================== MODEL =====================

class BinaryClassifier(nn.Module):
    """Simple MLP for binary classification."""

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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ===================== DATA LOADING =====================

def load_training_data(corrections_path: Path, manifest_path: Optional[Path] = None) -> Tuple[List[str], List[int]]:
    """Load training data from corrections.

    Single (label=0): corrections without bleed_instruments and not in MULTI_GROUPS
    Multi (label=1): corrections with bleed_instruments OR in MULTI_GROUPS

    Returns:
        paths: List of audio paths
        labels: List of labels (0=single, 1=multi)
    """
    with open(corrections_path) as f:
        corrections = json.load(f)

    paths = []
    labels = []

    single_count = 0
    multi_count = 0

    for path, corr in corrections.items():
        group = corr.get('group', '')
        bleed = corr.get('bleed_instruments', [])

        # Skip excluded groups
        if group in EXCLUDED_GROUPS:
            continue

        # Determine if multi-instrument
        is_multi = bool(bleed) or group in MULTI_GROUPS

        paths.append(path)
        labels.append(1 if is_multi else 0)

        if is_multi:
            multi_count += 1
        else:
            single_count += 1

    logging.info(f"Loaded from corrections: {single_count} single, {multi_count} multi")

    # If we have a manifest, we can add more verified single-instrument samples
    # from high-confidence predictions that match labels
    if manifest_path and manifest_path.exists():
        logging.info("Loading additional samples from manifest...")
        # This could be expanded later to add more training data

    return paths, labels


def extract_features(paths: List[str], num_workers: int = 8) -> Tuple[torch.Tensor, List[str], List[int]]:
    """Extract features from paths, returning only successful extractions."""

    features_list = []
    valid_paths = []
    valid_indices = []

    total = len(paths)
    logging.info(f"Extracting features from {total} files...")

    def process_one(idx_path):
        idx, path = idx_path
        latent_path = audio_path_to_latent_path(path)
        if latent_path is None:
            return None, idx
        latent = load_latent(latent_path)
        if latent is not None:
            return pool_latent(latent), idx
        return None, idx

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, (i, p)): i for i, p in enumerate(paths)}

        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 100 == 0:
                logging.info(f"  Progress: {done}/{total}")

            feature, idx = future.result()
            if feature is not None:
                features_list.append(feature)
                valid_paths.append(paths[idx])
                valid_indices.append(idx)

    if not features_list:
        return torch.tensor([]), [], []

    features = torch.stack(features_list)
    logging.info(f"Extracted {len(features)} features successfully ({len(paths) - len(features)} failed)")

    return features, valid_paths, valid_indices


# ===================== TRAINING =====================

def train_model(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    output_dir: Path,
    device: str = 'cuda'
):
    """Train binary classifier."""

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device_obj = torch.device(device)

    input_dim = train_features.shape[1]
    logging.info(f"Training binary classifier: input_dim={input_dim}")
    logging.info(f"Train: {len(train_labels)} samples ({train_labels.sum().item()} multi)")
    logging.info(f"Val: {len(val_labels)} samples ({val_labels.sum().item()} multi)")

    # Compute normalization
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0)
    std[std < 1e-6] = 1.0

    train_features_norm = (train_features - mean) / std
    val_features_norm = (val_features - mean) / std

    # Create datasets
    train_dataset = BinaryDataset(train_features_norm, train_labels.float())
    val_dataset = BinaryDataset(val_features_norm, val_labels.float())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Create model
    model = BinaryClassifier(input_dim).to(device_obj)

    # Class weights for imbalanced data
    num_single = (train_labels == 0).sum().item()
    num_multi = (train_labels == 1).sum().item()
    pos_weight = torch.tensor([num_single / max(num_multi, 1)]).to(device_obj)
    logging.info(f"Pos weight (multi): {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features = features.to(device_obj)
            labels = labels.to(device_obj)

            optimizer.zero_grad()

            # Get logits (before sigmoid)
            logits = model.net[:-1](features).squeeze(-1)  # Skip sigmoid
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device_obj)
                labels = labels.to(device_obj)

                logits = model.net[:-1](features).squeeze(-1)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'hidden_dim': HIDDEN_DIM,
                'mean': mean,
                'std': std,
                'epoch': epoch,
                'val_acc': val_acc,
            }, output_dir / 'model.pt')

        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            logging.info(f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}, "
                        f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    logging.info(f"\nBest validation accuracy: {best_val_acc:.3f} at epoch {best_epoch}")

    # Final evaluation
    model.load_state_dict(torch.load(output_dir / 'model.pt', weights_only=False)['model_state_dict'])
    model.eval()

    val_preds = []
    val_true = []
    val_probs = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device_obj)
            probs = model(features)
            preds = (probs > 0.5).float()

            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.numpy())
            val_probs.extend(probs.cpu().numpy())

    # Classification report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(val_true, val_preds, target_names=['single', 'multi']))

    # Confusion matrix
    cm = confusion_matrix(val_true, val_preds)
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("              single  multi")
    print(f"Actual single   {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"Actual multi    {cm[1,0]:4d}   {cm[1,1]:4d}")

    return model


# ===================== CLASSIFICATION =====================

def classify_files(
    model_path: Path,
    paths: List[str],
    device: str = 'cuda',
    num_workers: int = 8
) -> List[Dict]:
    """Classify files as single/multi instrument."""

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device_obj = torch.device(device)

    # Load model
    logging.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    input_dim = checkpoint['input_dim']
    hidden_dim = checkpoint.get('hidden_dim', HIDDEN_DIM)
    mean = checkpoint['mean']
    std = checkpoint['std']

    model = BinaryClassifier(input_dim, hidden_dim).to(device_obj)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extract features
    features, valid_paths, valid_indices = extract_features(paths, num_workers)

    if len(features) == 0:
        logging.warning("No features extracted!")
        return []

    # Normalize
    features_norm = (features - mean) / std

    # Classify
    results = []
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(features_norm), batch_size):
            batch = features_norm[i:i+batch_size].to(device_obj)
            probs = model(batch).cpu().numpy()

            for j, prob in enumerate(probs):
                idx = i + j
                results.append({
                    'path': valid_paths[idx],
                    'is_multi': bool(prob > 0.5),
                    'multi_probability': float(prob),
                    'confidence': float(max(prob, 1 - prob))
                })

    return results


def filter_predictions(
    model_path: Path,
    predictions_path: Path,
    output_path: Optional[Path] = None,
    device: str = 'cuda'
):
    """Add is_multi field to existing predictions."""

    logging.info(f"Loading predictions from {predictions_path}...")
    with open(predictions_path) as f:
        pred_data = json.load(f)

    predictions = pred_data.get('predictions', [])
    paths = [p['path'] for p in predictions]

    logging.info(f"Classifying {len(paths)} files...")
    results = classify_files(model_path, paths, device)

    # Create lookup
    multi_lookup = {r['path']: r for r in results}

    # Update predictions
    multi_count = 0
    for pred in predictions:
        path = pred['path']
        if path in multi_lookup:
            pred['is_multi'] = multi_lookup[path]['is_multi']
            pred['multi_probability'] = multi_lookup[path]['multi_probability']
            if pred['is_multi']:
                multi_count += 1
        else:
            pred['is_multi'] = None  # Couldn't classify

    logging.info(f"Found {multi_count} multi-instrument files ({multi_count/len(predictions)*100:.1f}%)")

    # Add to summary
    if 'summary' in pred_data:
        pred_data['summary']['multi_instrument'] = multi_count
        pred_data['summary']['single_instrument'] = len(predictions) - multi_count

    # Save
    if output_path is None:
        output_path = predictions_path

    with open(output_path, 'w') as f:
        json.dump(pred_data, f, indent=2)

    logging.info(f"Saved to {output_path}")

    return pred_data


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description='Binary Multi-Instrument Classifier')
    parser.add_argument('--mode', choices=['train', 'classify', 'filter'], required=True)
    parser.add_argument('--corrections', type=Path,
                       default=Path('/home/arlo/gcs-bucket/Manifests/corrections.json'))
    parser.add_argument('--predictions', type=Path,
                       default=Path('/home/arlo/Data/latent_classifier/predictions.json'))
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output-dir', type=Path,
                       default=Path('/home/arlo/Data/binary_classifier'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=8)

    args = parser.parse_args()

    if args.mode == 'train':
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        paths, labels = load_training_data(args.corrections)

        if len(paths) < 20:
            logging.error(f"Not enough training data: {len(paths)} samples")
            return

        # Extract features
        features, valid_paths, valid_indices = extract_features(paths, args.workers)
        valid_labels = torch.tensor([labels[i] for i in valid_indices])

        if len(features) < 20:
            logging.error(f"Not enough features extracted: {len(features)}")
            return

        # Split
        train_idx, val_idx = train_test_split(
            range(len(features)),
            test_size=0.2,
            stratify=valid_labels.numpy(),
            random_state=42
        )

        train_features = features[train_idx]
        train_labels = valid_labels[train_idx]
        val_features = features[val_idx]
        val_labels = valid_labels[val_idx]

        # Train
        train_model(train_features, train_labels, val_features, val_labels,
                   args.output_dir, args.device)

    elif args.mode == 'classify':
        if args.model is None:
            args.model = args.output_dir / 'model.pt'

        with open(args.predictions) as f:
            pred_data = json.load(f)

        paths = [p['path'] for p in pred_data.get('predictions', [])]
        results = classify_files(args.model, paths, args.device, args.workers)

        output = args.output or args.output_dir / 'multi_predictions.json'
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Saved {len(results)} results to {output}")

    elif args.mode == 'filter':
        if args.model is None:
            args.model = args.output_dir / 'model.pt'

        filter_predictions(args.model, args.predictions, args.output, args.device)


if __name__ == '__main__':
    main()
