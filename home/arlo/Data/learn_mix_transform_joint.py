#!/usr/bin/env python3
"""
Joint training: MLP transform with semi-supervised domain adaptation.

Two losses:
1. L_transform: MSE between transformed mix features and stem features (from ~7500 Demucs pairs)
2. L_classify: BCE between classifier predictions and GT labels (from ~51 labeled mixes)

The GT labels act as anchor points to guide the transform toward features that
actually classify brass/strings/winds correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import argparse

# Paths
GCS_BASE = Path("/home/arlo/gcs-bucket")
LATENTS_BASE = GCS_BASE / "Latents"
V2_LATENTS_BASE = GCS_BASE / "LatentDemucsV2"
SOLO_MODEL_PATH = Path("/home/arlo/Data/latent_classifier/model.pt")
OUTPUT_DIR = Path("/home/arlo/Data/mix_classifier")

# Feature extraction (same as classifier)
POOL_METHODS = ['mean', 'std', 'max']


class MLPTransform(nn.Module):
    """MLP to transform mix features to stem-like features."""
    def __init__(self, input_dim=384, output_dim=384, hidden_dims=[1024, 1024]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SoloClassifierHead(nn.Module):
    """Recreation of the solo classifier architecture for inference."""
    def __init__(self, state_dict, input_dim=384, hidden_dim=256, num_classes=13):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

        # Load weights
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


def extract_features(latent):
    """Extract 384-dim features from latent."""
    pools = []
    for method in POOL_METHODS:
        if method == 'mean':
            pooled = latent.mean(dim=-1)
        elif method == 'std':
            pooled = latent.std(dim=-1)
        elif method == 'max':
            pooled = latent.max(dim=-1)[0]
        pools.append(pooled.flatten())
    return torch.cat(pools)


def load_latent(path):
    """Load latent from .pt file."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        return data.get('latents', data.get('latent'))
    return data


def find_matched_pairs():
    """Find V2 entries that have corresponding original mix latents."""
    pairs = []
    for stem_dir in V2_LATENTS_BASE.rglob("other.pt"):
        stem_parent = stem_dir.parent
        rel_path = str(stem_parent).replace(str(V2_LATENTS_BASE) + "/", "")
        mix_latent_path = LATENTS_BASE / (rel_path + ".pt")
        if mix_latent_path.exists():
            pairs.append({
                'mix_latent': mix_latent_path,
                'stem_dir': stem_parent,
                'name': stem_parent.name
            })
    return pairs


def extract_all_stem_pairs(pairs, limit=None):
    """Extract features from ALL 6 Demucs stems."""
    all_mix_features = []
    all_stem_features = []
    stem_counts = defaultdict(int)
    stems_to_use = ["drums", "bass", "vocals", "other", "guitar", "piano"]

    if limit:
        pairs = pairs[:limit]

    for i, pair in enumerate(pairs):
        try:
            mix_latent = load_latent(pair['mix_latent'])
            if mix_latent is None:
                continue
            mix_feat = extract_features(mix_latent)

            for stem_name in stems_to_use:
                stem_path = pair['stem_dir'] / f"{stem_name}.pt"
                if not stem_path.exists():
                    continue
                stem_latent = load_latent(stem_path)
                if stem_latent is None:
                    continue
                stem_energy = stem_latent.abs().mean()
                if stem_energy < 0.01:
                    continue
                stem_feat = extract_features(stem_latent)
                all_mix_features.append(mix_feat)
                all_stem_features.append(stem_feat)
                stem_counts[stem_name] += 1
        except Exception:
            continue

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(pairs)} files, {len(all_mix_features)} pairs so far")

    print(f"\nPairs per stem: {dict(stem_counts)}")
    print(f"Total pairs: {len(all_mix_features)}")

    if not all_mix_features:
        return None, None
    return torch.stack(all_mix_features), torch.stack(all_stem_features)


def load_gt_labeled_mixes(gt_labels_path, solo_classes):
    """
    Load ground-truth labeled mix files.

    Expected JSON format:
    {
        "/path/to/latent.pt": ["brass", "strings"],
        "/path/to/another.pt": ["winds", "piano"],
        ...
    }

    Or from API:
    [
        {"path": "/path/to/audio.wav", "labels": ["brass", "strings"]},
        ...
    ]
    """
    with open(gt_labels_path) as f:
        data = json.load(f)

    # Handle list format (from API)
    if isinstance(data, list):
        gt_dict = {}
        for item in data:
            path = item.get('path', '')
            labels = item.get('labels', [])
            # Convert audio path to latent path
            latent_path = path.replace('/protools/', '/Latents/protools/')
            latent_path = latent_path.replace('.wav', '.pt').replace('.mp3', '.pt')
            if labels:
                gt_dict[latent_path] = labels
        data = gt_dict

    # Build class to index mapping
    class_to_idx = {c: i for i, c in enumerate(solo_classes)}

    features = []
    labels = []
    paths_used = []

    for path, label_list in data.items():
        latent_path = Path(path)
        if not latent_path.exists():
            # Try finding in Latents dir
            alt_path = Path(str(path).replace('/gcs-bucket/protools/', '/gcs-bucket/Latents/protools/'))
            if alt_path.exists():
                latent_path = alt_path
            else:
                continue

        try:
            latent = load_latent(latent_path)
            if latent is None:
                continue
            feat = extract_features(latent)

            # Create multi-hot label vector
            label_vec = torch.zeros(len(solo_classes))
            for lbl in label_list:
                if lbl in class_to_idx:
                    label_vec[class_to_idx[lbl]] = 1.0

            features.append(feat)
            labels.append(label_vec)
            paths_used.append(str(latent_path))
        except Exception as e:
            continue

    if not features:
        return None, None, []

    return torch.stack(features), torch.stack(labels), paths_used


def train_joint(
    mix_feats, stem_feats,
    gt_mix_feats, gt_labels,
    solo_classifier, solo_mean, solo_std,
    target_indices,
    epochs=500, lr=0.001, lambda_classify=1.0,
    hidden_dims=[1024, 1024]
):
    """
    Joint training with transform loss + classification loss.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Split transform data
    n = len(mix_feats)
    indices = torch.randperm(n)
    train_idx = indices[:int(0.9 * n)]
    val_idx = indices[int(0.9 * n):]

    X_train, Y_train = mix_feats[train_idx], stem_feats[train_idx]
    X_val, Y_val = mix_feats[val_idx], stem_feats[val_idx]

    # Normalize
    X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-6
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std

    # Normalize GT mix features the same way
    gt_mix_norm = (gt_mix_feats - X_mean) / X_std

    # Move to device
    X_train_norm = X_train_norm.to(device)
    Y_train = Y_train.to(device)
    X_val_norm = X_val_norm.to(device)
    Y_val = Y_val.to(device)
    gt_mix_norm = gt_mix_norm.to(device)
    gt_labels = gt_labels.to(device)
    solo_mean = solo_mean.to(device)
    solo_std = solo_std.to(device)

    # Models
    transform = MLPTransform(384, 384, hidden_dims).to(device)
    solo_classifier = solo_classifier.to(device)
    solo_classifier.eval()  # Freeze solo classifier
    for p in solo_classifier.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(transform.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')
    best_state = None

    batch_size = 256
    gt_batch_size = min(16, len(gt_mix_norm))  # Smaller batches for GT

    target_indices_t = torch.tensor(target_indices, device=device)

    for epoch in range(epochs):
        transform.train()

        # Shuffle transform data
        perm = torch.randperm(len(X_train_norm), device=device)
        total_transform_loss = 0
        total_classify_loss = 0
        n_batches = 0

        for i in range(0, len(X_train_norm), batch_size):
            batch_idx = perm[i:i+batch_size]
            x_batch = X_train_norm[batch_idx]
            y_batch = Y_train[batch_idx]

            # Sample GT batch
            gt_perm = torch.randperm(len(gt_mix_norm), device=device)[:gt_batch_size]
            gt_x = gt_mix_norm[gt_perm]
            gt_y = gt_labels[gt_perm]

            optimizer.zero_grad()

            # Loss 1: Transform reconstruction
            pred_stem = transform(x_batch)
            loss_transform = F.mse_loss(pred_stem, y_batch)

            # Loss 2: Classification on GT mixes
            gt_transformed = transform(gt_x)
            # Normalize for solo classifier
            gt_transformed_normed = (gt_transformed - solo_mean) / (solo_std + 1e-6)
            logits = solo_classifier(gt_transformed_normed)

            # Only supervise on target classes (brass, strings, winds)
            target_logits = logits[:, target_indices_t]
            target_labels = gt_y[:, target_indices_t]
            loss_classify = F.binary_cross_entropy_with_logits(target_logits, target_labels)

            # Combined loss
            loss = loss_transform + lambda_classify * loss_classify
            loss.backward()
            optimizer.step()

            total_transform_loss += loss_transform.item()
            total_classify_loss += loss_classify.item()
            n_batches += 1

        scheduler.step()

        # Validation
        transform.eval()
        with torch.no_grad():
            val_pred = transform(X_val_norm)
            val_loss = F.mse_loss(val_pred, Y_val).item()

            # Also check classification accuracy on GT
            gt_trans = transform(gt_mix_norm)
            gt_trans_normed = (gt_trans - solo_mean) / (solo_std + 1e-6)
            gt_logits = solo_classifier(gt_trans_normed)
            gt_probs = torch.sigmoid(gt_logits[:, target_indices_t])
            gt_preds = (gt_probs > 0.5).float()
            gt_targets = gt_labels[:, target_indices_t]
            accuracy = (gt_preds == gt_targets).float().mean().item()

        # Track best by combined metric
        combined_val = val_loss  # Could also include classify loss
        if combined_val < best_val_loss:
            best_val_loss = combined_val
            best_state = {k: v.cpu().clone() for k, v in transform.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: transform_loss={total_transform_loss/n_batches:.4f}, "
                  f"classify_loss={total_classify_loss/n_batches:.4f}, "
                  f"val_loss={val_loss:.4f}, gt_acc={accuracy:.3f}")

    # Load best
    transform.load_state_dict(best_state)
    transform.eval()

    # Final eval
    with torch.no_grad():
        gt_trans = transform(gt_mix_norm)
        gt_trans_normed = (gt_trans - solo_mean) / (solo_std + 1e-6)
        gt_logits = solo_classifier(gt_trans_normed)
        gt_probs = torch.sigmoid(gt_logits[:, target_indices_t])
        gt_preds = (gt_probs > 0.5).float()
        gt_targets = gt_labels[:, target_indices_t]

        # Per-class metrics
        target_classes = ['brass', 'strings', 'winds']
        print("\nFinal GT accuracy per class:")
        for i, cls in enumerate(target_classes):
            cls_preds = gt_preds[:, i]
            cls_targets = gt_targets[:, i]
            correct = (cls_preds == cls_targets).sum().item()
            total = len(cls_targets)
            tp = ((cls_preds == 1) & (cls_targets == 1)).sum().item()
            fp = ((cls_preds == 1) & (cls_targets == 0)).sum().item()
            fn = ((cls_preds == 0) & (cls_targets == 1)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"  {cls}: acc={correct/total:.3f}, prec={precision:.3f}, rec={recall:.3f}")

    X_mean = X_mean.cpu()
    X_std = X_std.cpu()

    return transform.cpu(), X_mean, X_std, best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-labels', type=str, required=True,
                        help='Path to JSON file with GT labeled mixes')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--hidden', type=int, nargs='+', default=[1024, 1024])
    parser.add_argument('--lambda-classify', type=float, default=1.0,
                        help='Weight for classification loss')
    parser.add_argument('--limit-pairs', type=int, default=None,
                        help='Limit number of Demucs pairs for faster testing')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("JOINT TRAINING: TRANSFORM + CLASSIFICATION")
    print("=" * 60)
    print(f"GT labels: {args.gt_labels}")
    print(f"Lambda classify: {args.lambda_classify}")
    print(f"Hidden dims: {args.hidden}")

    # Load solo classifier
    print("\nLoading solo classifier...")
    solo_checkpoint = torch.load(SOLO_MODEL_PATH, map_location='cpu', weights_only=False)
    solo_classes = list(solo_checkpoint['label_encoder_classes'])
    solo_state = solo_checkpoint['model_state']
    solo_mean = solo_checkpoint['mean']
    solo_std = solo_checkpoint['std']
    print(f"Solo classes: {solo_classes}")

    # Find target class indices
    target_classes = ['brass', 'strings', 'winds']
    target_indices = [solo_classes.index(c) for c in target_classes if c in solo_classes]
    print(f"Target indices: {target_indices} -> {[solo_classes[i] for i in target_indices]}")

    # Recreate solo classifier
    solo_classifier = SoloClassifierHead(solo_state, input_dim=384, hidden_dim=256,
                                          num_classes=len(solo_classes))

    # Load GT labeled mixes
    print(f"\nLoading GT labeled mixes from {args.gt_labels}...")
    gt_feats, gt_labels, gt_paths = load_gt_labeled_mixes(args.gt_labels, solo_classes)
    if gt_feats is None:
        print("No GT labeled mixes found!")
        return
    print(f"Loaded {len(gt_feats)} GT labeled mixes")

    # Show GT label distribution
    label_counts = gt_labels.sum(0)
    print("GT label distribution:")
    for i, cls in enumerate(solo_classes):
        if label_counts[i] > 0:
            print(f"  {cls}: {int(label_counts[i])}")

    # Find Demucs pairs
    print("\nFinding Demucs pairs...")
    pairs = find_matched_pairs()
    print(f"Found {len(pairs)} matched pairs")

    # Extract paired features
    print("\nExtracting paired features from all stems...")
    mix_feats, stem_feats = extract_all_stem_pairs(pairs, limit=args.limit_pairs)
    if mix_feats is None:
        print("Failed to extract paired features!")
        return
    print(f"Extracted {len(mix_feats)} paired samples")

    # Train jointly
    print(f"\nTraining joint model (epochs={args.epochs}, lambda={args.lambda_classify})...")
    transform, X_mean, X_std, val_loss = train_joint(
        mix_feats, stem_feats,
        gt_feats, gt_labels,
        solo_classifier, solo_mean, solo_std,
        target_indices,
        epochs=args.epochs,
        lambda_classify=args.lambda_classify,
        hidden_dims=args.hidden
    )
    print(f"\nBest val_loss: {val_loss:.4f}")

    # Save model
    model_path = OUTPUT_DIR / "mix_classifier_joint.pt"
    torch.save({
        'transform_type': 'mlp_joint',
        'mlp_state': transform.state_dict(),
        'mlp_X_mean': X_mean,
        'mlp_X_std': X_std,
        'mlp_hidden_dims': args.hidden,
        'solo_model_state': solo_state,
        'solo_mean': solo_mean,
        'solo_std': solo_std,
        'solo_classes': solo_classes,
        'target_classes': target_classes,
        'target_indices': target_indices,
        'lambda_classify': args.lambda_classify,
        'num_gt_samples': len(gt_feats),
        'num_transform_pairs': len(mix_feats),
    }, model_path)
    print(f"\nSaved to: {model_path}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print("\nTo evaluate:")
    print(f"python3 mix_temporal_classifier.py --mode batch --input-dir /home/arlo/gcs-bucket/Latents \\")
    print(f"  --limit 50 --model {model_path} --threshold 0.25")


if __name__ == "__main__":
    main()
