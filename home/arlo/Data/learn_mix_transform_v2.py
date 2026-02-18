#!/usr/bin/env python3
"""
Joint training v2: Train transform + new classification head together.
Does NOT use the broken solo classifier for brass/strings/winds.
Instead, trains a fresh classifier head on the target classes directly.

Two losses:
1. L_transform: MSE between transformed mix features and stem features (Demucs pairs)
2. L_classify: BCE for target classes using a NEW trainable classification head

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
OUTPUT_DIR = Path("/home/arlo/Data/mix_classifier")

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


class TargetClassifier(nn.Module):
    """Fresh classifier head for target classes (brass/strings/winds).

    Simplified architecture to prevent overfitting with small training set.
    """
    def __init__(self, input_dim=384, hidden_dim=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),  # Higher dropout to prevent overfitting
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


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


def load_negative_examples(corrections_path, target_classes, max_negatives=50):
    """Load negative examples from corrections.json - files labeled with non-target classes."""
    with open(corrections_path) as f:
        corrections = json.load(f)

    target_set = set(target_classes)
    negatives = []

    for audio_path, data in corrections.items():
        labels_in_file = set()
        for region in data.get('regions', []):
            labels_in_file.update(region.get('labels', []))

        # Has labels but NOT target classes - this is a true negative
        if labels_in_file and not labels_in_file.intersection(target_set):
            # Convert audio path to latent path
            latent_path = audio_path.replace('/home/arlo/gcs-bucket/', '/home/arlo/gcs-bucket/Latents/')
            latent_path = latent_path.replace('.wav', '.pt').replace('.mp3', '.pt')

            if Path(latent_path).exists():
                negatives.append((latent_path, list(labels_in_file)))

    return negatives[:max_negatives]


def load_gt_labeled_files(gt_labels_path, target_classes, include_negatives=True, corrections_path=None):
    """Load GT labeled files with multi-hot encoding for target classes."""
    with open(gt_labels_path) as f:
        data = json.load(f)

    # Handle dict format {path: [labels]}
    if isinstance(data, dict):
        gt_dict = data
    else:
        # Handle list format
        gt_dict = {}
        for item in data:
            path = item.get('path', item.get('latent_path', ''))
            labels = item.get('labels', [])
            if labels:
                gt_dict[path] = labels

    class_to_idx = {c: i for i, c in enumerate(target_classes)}

    features = []
    labels = []
    paths_used = []
    positive_count = 0

    for path, label_list in gt_dict.items():
        latent_path = Path(path)
        if not latent_path.exists():
            continue

        try:
            latent = load_latent(latent_path)
            if latent is None:
                continue
            feat = extract_features(latent)

            # Create multi-hot label vector for target classes only
            label_vec = torch.zeros(len(target_classes))
            for lbl in label_list:
                if lbl in class_to_idx:
                    label_vec[class_to_idx[lbl]] = 1.0

            # Only include if at least one target class is present
            if label_vec.sum() > 0:
                features.append(feat)
                labels.append(label_vec)
                paths_used.append(str(latent_path))
                positive_count += 1
        except Exception:
            continue

    # Load negative examples to balance the training
    if include_negatives and corrections_path and Path(corrections_path).exists():
        negatives = load_negative_examples(corrections_path, target_classes, max_negatives=positive_count)
        print(f"Loading {len(negatives)} negative examples to balance {positive_count} positive examples")

        for latent_path, _ in negatives:
            try:
                latent = load_latent(Path(latent_path))
                if latent is None:
                    continue
                feat = extract_features(latent)

                # All zeros for negative examples
                label_vec = torch.zeros(len(target_classes))
                features.append(feat)
                labels.append(label_vec)
                paths_used.append(str(latent_path))
            except Exception:
                continue

    if not features:
        return None, None, []

    return torch.stack(features), torch.stack(labels), paths_used


def train_joint_v2(
    mix_feats, stem_feats,
    gt_mix_feats, gt_labels,
    target_classes,
    epochs=500, lr=0.001, lambda_classify=5.0,
    hidden_dims=[1024, 1024]
):
    """Joint training with transform loss + fresh classification head."""
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

    # Split GT into train/val (80/20)
    n_gt = len(gt_mix_feats)
    gt_indices = torch.randperm(n_gt)
    gt_train_idx = gt_indices[:int(0.8 * n_gt)]
    gt_val_idx = gt_indices[int(0.8 * n_gt):]

    gt_train_feats = gt_mix_feats[gt_train_idx]
    gt_train_labels = gt_labels[gt_train_idx]
    gt_val_feats = gt_mix_feats[gt_val_idx]
    gt_val_labels = gt_labels[gt_val_idx]

    # Normalize GT features
    gt_train_norm = (gt_train_feats - X_mean) / X_std
    gt_val_norm = (gt_val_feats - X_mean) / X_std

    # Move to device
    X_train_norm = X_train_norm.to(device)
    Y_train = Y_train.to(device)
    X_val_norm = X_val_norm.to(device)
    Y_val = Y_val.to(device)
    gt_train_norm = gt_train_norm.to(device)
    gt_train_labels = gt_train_labels.to(device)
    gt_val_norm = gt_val_norm.to(device)
    gt_val_labels = gt_val_labels.to(device)

    # Models - BOTH trainable
    # Using smaller classifier (64 hidden) to prevent overfitting with small dataset
    transform = MLPTransform(384, 384, hidden_dims).to(device)
    classifier = TargetClassifier(384, 64, len(target_classes)).to(device)

    # Combined optimizer with stronger regularization
    optimizer = torch.optim.AdamW(
        list(transform.parameters()) + list(classifier.parameters()),
        lr=lr, weight_decay=0.1  # Increased weight decay for regularization
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_f1 = 0
    best_state = None

    batch_size = 256
    gt_batch_size = min(32, len(gt_train_norm))

    for epoch in range(epochs):
        transform.train()
        classifier.train()

        perm = torch.randperm(len(X_train_norm), device=device)
        total_transform_loss = 0
        total_classify_loss = 0
        n_batches = 0

        for i in range(0, len(X_train_norm), batch_size):
            batch_idx = perm[i:i+batch_size]
            x_batch = X_train_norm[batch_idx]
            y_batch = Y_train[batch_idx]

            # Sample GT batch
            gt_perm = torch.randperm(len(gt_train_norm), device=device)[:gt_batch_size]
            gt_x = gt_train_norm[gt_perm]
            gt_y = gt_train_labels[gt_perm]

            optimizer.zero_grad()

            # Loss 1: Transform reconstruction
            pred_stem = transform(x_batch)
            loss_transform = F.mse_loss(pred_stem, y_batch)

            # Loss 2: Classification on GT - use transformed features
            # Use class weights: higher weight for negative class (absence) to reduce over-prediction
            gt_transformed = transform(gt_x)
            logits = classifier(gt_transformed)
            # pos_weight > 1 increases recall, < 1 increases precision
            # We want higher precision, so use lower pos_weight for classes that co-occur
            pos_weight = torch.tensor([0.5, 0.5, 0.5], device=device)  # Reduce FP by weighting negatives higher
            loss_classify = F.binary_cross_entropy_with_logits(logits, gt_y, pos_weight=pos_weight)

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
        classifier.eval()
        with torch.no_grad():
            # Transform val loss
            val_pred = transform(X_val_norm)
            val_loss = F.mse_loss(val_pred, Y_val).item()

            # Classification metrics on GT val
            gt_val_trans = transform(gt_val_norm)
            gt_val_logits = classifier(gt_val_trans)
            gt_val_probs = torch.sigmoid(gt_val_logits)
            gt_val_preds = (gt_val_probs > 0.5).float()

            # Per-class F1
            f1_scores = []
            for i, cls in enumerate(target_classes):
                tp = ((gt_val_preds[:, i] == 1) & (gt_val_labels[:, i] == 1)).sum().float()
                fp = ((gt_val_preds[:, i] == 1) & (gt_val_labels[:, i] == 0)).sum().float()
                fn = ((gt_val_preds[:, i] == 0) & (gt_val_labels[:, i] == 1)).sum().float()
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                f1_scores.append(f1.item())

            avg_f1 = sum(f1_scores) / len(f1_scores)

        # Track best by F1
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            best_state = {
                'transform': {k: v.cpu().clone() for k, v in transform.state_dict().items()},
                'classifier': {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            }

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: trans_loss={total_transform_loss/n_batches:.4f}, "
                  f"cls_loss={total_classify_loss/n_batches:.4f}, "
                  f"val_loss={val_loss:.4f}, F1={avg_f1:.3f} "
                  f"({', '.join(f'{c}={f:.2f}' for c, f in zip(target_classes, f1_scores))})")

    # Load best
    transform.load_state_dict(best_state['transform'])
    classifier.load_state_dict(best_state['classifier'])
    transform.eval()
    classifier.eval()

    # Final eval on full GT
    gt_all_norm = (gt_mix_feats - X_mean) / X_std
    gt_all_norm = gt_all_norm.to(device)
    gt_labels_dev = gt_labels.to(device)

    with torch.no_grad():
        gt_trans = transform(gt_all_norm)
        gt_logits = classifier(gt_trans)
        gt_probs = torch.sigmoid(gt_logits)
        gt_preds = (gt_probs > 0.5).float()

        print("\nFinal results on all GT data:")
        for i, cls in enumerate(target_classes):
            tp = ((gt_preds[:, i] == 1) & (gt_labels_dev[:, i] == 1)).sum().item()
            fp = ((gt_preds[:, i] == 1) & (gt_labels_dev[:, i] == 0)).sum().item()
            fn = ((gt_preds[:, i] == 0) & (gt_labels_dev[:, i] == 1)).sum().item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"  {cls}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} (TP={tp} FP={fp} FN={fn})")

    X_mean = X_mean.cpu()
    X_std = X_std.cpu()

    return transform.cpu(), classifier.cpu(), X_mean, X_std, best_val_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-labels', type=str, required=True,
                        help='Path to JSON file with GT labeled files')
    parser.add_argument('--corrections', type=str,
                        default='/home/arlo/gcs-bucket/Manifests/corrections.json',
                        help='Path to corrections.json for negative examples')
    parser.add_argument('--no-negatives', action='store_true',
                        help='Disable negative example loading')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--hidden', type=int, nargs='+', default=[1024, 1024])
    parser.add_argument('--lambda-classify', type=float, default=5.0)
    parser.add_argument('--limit-pairs', type=int, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_classes = ['brass', 'strings', 'winds']

    print("=" * 60)
    print("JOINT TRAINING V2: TRANSFORM + NEW CLASSIFIER HEAD")
    print("=" * 60)
    print(f"Target classes: {target_classes}")
    print(f"Lambda classify: {args.lambda_classify}")

    # Load GT labeled files (with negative examples for balancing)
    print(f"\nLoading GT labels from {args.gt_labels}...")
    corrections_path = None if args.no_negatives else args.corrections
    gt_feats, gt_labels, gt_paths = load_gt_labeled_files(
        args.gt_labels, target_classes,
        include_negatives=not args.no_negatives,
        corrections_path=corrections_path
    )
    if gt_feats is None:
        print("No GT labeled files found!")
        return
    print(f"Loaded {len(gt_feats)} GT labeled files")

    # Show label distribution
    label_counts = gt_labels.sum(0)
    # Count negative examples (all zeros)
    negative_count = (gt_labels.sum(1) == 0).sum().item()
    positive_count = len(gt_labels) - negative_count
    print(f"GT label distribution ({positive_count} positive, {negative_count} negative):")
    for i, cls in enumerate(target_classes):
        print(f"  {cls}: {int(label_counts[i])}")

    # Find Demucs pairs
    print("\nFinding Demucs pairs...")
    pairs = find_matched_pairs()
    print(f"Found {len(pairs)} matched pairs")

    # Extract paired features
    print("\nExtracting paired features...")
    mix_feats, stem_feats = extract_all_stem_pairs(pairs, limit=args.limit_pairs)
    if mix_feats is None:
        print("Failed to extract paired features!")
        return
    print(f"Extracted {len(mix_feats)} paired samples")

    # Train
    print(f"\nTraining (epochs={args.epochs}, lambda={args.lambda_classify})...")
    transform, classifier, X_mean, X_std, best_f1 = train_joint_v2(
        mix_feats, stem_feats,
        gt_feats, gt_labels,
        target_classes,
        epochs=args.epochs,
        lambda_classify=args.lambda_classify,
        hidden_dims=args.hidden
    )
    print(f"\nBest validation F1: {best_f1:.3f}")

    # Save
    model_path = OUTPUT_DIR / "mix_classifier_v2.pt"
    torch.save({
        'transform_type': 'mlp_v2',
        'transform_state': transform.state_dict(),
        'classifier_state': classifier.state_dict(),
        'transform_hidden_dims': args.hidden,
        'X_mean': X_mean,
        'X_std': X_std,
        'target_classes': target_classes,
        'lambda_classify': args.lambda_classify,
        'num_gt_samples': len(gt_feats),
        'num_transform_pairs': len(mix_feats),
    }, model_path)
    print(f"\nSaved to: {model_path}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
