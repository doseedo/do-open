#!/usr/bin/env python3
"""
V3: Segment-level training using timestamped regions from corrections.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse

LATENTS_BASE = Path("/home/arlo/gcs-bucket/Latents")
V2_LATENTS_BASE = Path("/home/arlo/gcs-bucket/LatentDemucsV2")
CORRECTIONS_PATH = Path("/home/arlo/gcs-bucket/Manifests/corrections.json")
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
    """Classifier head for target classes."""
    def __init__(self, input_dim=384, hidden_dim=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def extract_features(latent, start_frame=None, end_frame=None):
    """Extract 384-dim features from latent, optionally for a specific time range."""
    if start_frame is not None and end_frame is not None:
        latent = latent[..., start_frame:end_frame]

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
        return data.get('latents', data.get('latent')), data.get('latent_duration', data.get('original_duration', 0))
    return data, 0


def load_segment_data(corrections_path, target_classes):
    """Load segment-level training data from corrections with regions."""
    with open(corrections_path) as f:
        corrections = json.load(f)

    segments = []

    for audio_path, info in corrections.items():
        if not isinstance(info, dict) or 'regions' not in info or not info['regions']:
            continue

        # Build latent path
        latent_path = audio_path.replace('/gcs-bucket/', '/gcs-bucket/Latents/')
        latent_path = latent_path.rsplit('.', 1)[0] + '.pt'

        if not Path(latent_path).exists():
            continue

        duration = info.get('duration', 0)
        if duration <= 0:
            continue

        # Load latent to get frame rate
        try:
            latent, latent_duration = load_latent(latent_path)
            if latent is None or latent_duration <= 0:
                continue

            n_frames = latent.shape[-1]
            fps = n_frames / latent_duration

            # Extract features for each region with target classes
            for region in info['regions']:
                labels = region.get('labels', [])
                relevant_labels = [l for l in labels if l in target_classes]

                # Skip regions without target classes (they become negative examples implicitly)
                # but we also want negative examples
                start_time = region.get('start', 0)
                end_time = region.get('end', duration)

                start_frame = int(start_time * fps)
                end_frame = min(int(end_time * fps), n_frames)

                if end_frame <= start_frame:
                    continue

                # Create multi-hot label
                label_vec = torch.zeros(len(target_classes))
                for lbl in relevant_labels:
                    idx = target_classes.index(lbl)
                    label_vec[idx] = 1.0

                feat = extract_features(latent, start_frame, end_frame)

                segments.append({
                    'features': feat,
                    'labels': label_vec,
                    'path': latent_path,
                    'start': start_time,
                    'end': end_time,
                    'is_positive': len(relevant_labels) > 0
                })

        except Exception as e:
            print(f"Error loading {latent_path}: {e}")
            continue

    return segments


def find_matched_pairs():
    """Find V2 entries that have corresponding original mix latents."""
    pairs = []
    for stem_dir in V2_LATENTS_BASE.rglob("other.pt"):
        stem_parent = stem_dir.parent
        rel_path = str(stem_parent).replace(str(V2_LATENTS_BASE) + "/", "")
        mix_latent_path = LATENTS_BASE / (rel_path + ".pt")
        if mix_latent_path.exists():
            pairs.append((mix_latent_path, stem_parent))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda-classify', type=float, default=5.0)
    parser.add_argument('--output', type=str, default='/home/arlo/Data/mix_classifier/mix_classifier_v3.pt')
    args = parser.parse_args()

    target_classes = ['brass', 'strings', 'winds']

    print("=" * 60)
    print("SEGMENT-LEVEL TRAINING V3")
    print("=" * 60)
    print(f"Target classes: {target_classes}")
    print(f"Lambda classify: {args.lambda_classify}")

    # Load segment data
    print(f"\nLoading segment data from corrections...")
    segments = load_segment_data(CORRECTIONS_PATH, target_classes)

    positive_segments = [s for s in segments if s['is_positive']]
    negative_segments = [s for s in segments if not s['is_positive']]

    print(f"Loaded {len(segments)} segments ({len(positive_segments)} positive, {len(negative_segments)} negative)")

    if len(positive_segments) == 0:
        print("No positive segments found!")
        return

    # Stack features and labels
    features = torch.stack([s['features'] for s in positive_segments])
    labels = torch.stack([s['labels'] for s in positive_segments])

    # Add negative examples (balance with positives)
    if negative_segments:
        n_neg = min(len(negative_segments), len(positive_segments))
        import random
        random.shuffle(negative_segments)
        neg_feats = torch.stack([s['features'] for s in negative_segments[:n_neg]])
        neg_labels = torch.stack([s['labels'] for s in negative_segments[:n_neg]])
        features = torch.cat([features, neg_feats])
        labels = torch.cat([labels, neg_labels])

    print(f"\nTotal training samples: {len(features)} ({len(positive_segments)} positive, {len(features) - len(positive_segments)} negative)")

    # Label distribution
    print("Label distribution:")
    for i, cls in enumerate(target_classes):
        count = (labels[:, i] == 1).sum().item()
        print(f"  {cls}: {count}")

    # Find Demucs pairs for transform training
    print("\nFinding Demucs pairs...")
    pairs = find_matched_pairs()
    print(f"Found {len(pairs)} matched pairs")

    if len(pairs) == 0:
        print("No pairs found!")
        return

    # Extract paired features
    print("\nExtracting paired features...")
    X_pairs = []  # mix features
    Y_pairs = []  # stem features (target for transform)

    stem_counts = {}
    for mix_path, stem_dir in pairs:
        try:
            mix_latent, _ = load_latent(mix_path)
            if mix_latent is None:
                continue
            mix_feat = extract_features(mix_latent)

            for stem_name in ['drums', 'bass', 'vocals', 'other', 'guitar', 'piano']:
                stem_path = stem_dir / f"{stem_name}.pt"
                if not stem_path.exists():
                    continue

                stem_latent, _ = load_latent(stem_path)
                if stem_latent is None:
                    continue
                stem_feat = extract_features(stem_latent)

                X_pairs.append(mix_feat)
                Y_pairs.append(stem_feat)
                stem_counts[stem_name] = stem_counts.get(stem_name, 0) + 1

        except Exception:
            continue

    print(f"Pairs per stem: {stem_counts}")
    print(f"Total pairs: {len(X_pairs)}")

    if len(X_pairs) == 0:
        print("No pairs extracted!")
        return

    X_pairs = torch.stack(X_pairs)
    Y_pairs = torch.stack(Y_pairs)

    # Normalize
    X_mean = X_pairs.mean(0)
    X_std = X_pairs.std(0) + 1e-6
    X_norm = (X_pairs - X_mean) / X_std

    features_norm = (features - X_mean) / X_std

    # Train/val split
    n_train = int(0.8 * len(X_norm))
    X_train = X_norm[:n_train]
    Y_train = Y_pairs[:n_train]
    X_val = X_norm[n_train:]
    Y_val = Y_pairs[n_train:]

    n_cls_train = int(0.8 * len(features_norm))
    cls_train_feats = features_norm[:n_cls_train]
    cls_train_labels = labels[:n_cls_train]
    cls_val_feats = features_norm[n_cls_train:]
    cls_val_labels = labels[n_cls_train:]

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")

    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    cls_train_feats = cls_train_feats.to(device)
    cls_train_labels = cls_train_labels.to(device)
    cls_val_feats = cls_val_feats.to(device)
    cls_val_labels = cls_val_labels.to(device)

    transform = MLPTransform().to(device)
    classifier = TargetClassifier(num_classes=len(target_classes)).to(device)

    optimizer = torch.optim.AdamW(
        list(transform.parameters()) + list(classifier.parameters()),
        lr=args.lr, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val_f1 = 0
    best_state = None

    batch_size = 256
    cls_batch_size = min(32, len(cls_train_feats))

    print(f"\nTraining for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        transform.train()
        classifier.train()

        perm = torch.randperm(len(X_train), device=device)
        total_trans_loss = 0
        total_cls_loss = 0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i+batch_size]
            x_batch = X_train[batch_idx]
            y_batch = Y_train[batch_idx]

            # Sample classification batch
            cls_perm = torch.randperm(len(cls_train_feats), device=device)[:cls_batch_size]
            cls_x = cls_train_feats[cls_perm]
            cls_y = cls_train_labels[cls_perm]

            optimizer.zero_grad()

            # Transform loss
            pred_stem = transform(x_batch)
            loss_trans = F.mse_loss(pred_stem, y_batch)

            # Classification loss
            cls_transformed = transform(cls_x)
            logits = classifier(cls_transformed)
            loss_cls = F.binary_cross_entropy_with_logits(logits, cls_y)

            loss = loss_trans + args.lambda_classify * loss_cls
            loss.backward()
            optimizer.step()

            total_trans_loss += loss_trans.item()
            total_cls_loss += loss_cls.item()
            n_batches += 1

        scheduler.step()

        # Validation
        transform.eval()
        classifier.eval()
        with torch.no_grad():
            val_pred = transform(X_val)
            val_loss = F.mse_loss(val_pred, Y_val).item()

            cls_val_trans = transform(cls_val_feats)
            cls_val_logits = classifier(cls_val_trans)
            cls_val_probs = torch.sigmoid(cls_val_logits)
            cls_val_preds = (cls_val_probs > 0.5).float()

            # Per-class F1
            f1_scores = []
            for i, cls in enumerate(target_classes):
                tp = ((cls_val_preds[:, i] == 1) & (cls_val_labels[:, i] == 1)).sum().float()
                fp = ((cls_val_preds[:, i] == 1) & (cls_val_labels[:, i] == 0)).sum().float()
                fn = ((cls_val_preds[:, i] == 0) & (cls_val_labels[:, i] == 1)).sum().float()
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                f1_scores.append(f1.item())

            avg_f1 = sum(f1_scores) / len(f1_scores)

        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            best_state = {
                'transform': {k: v.cpu().clone() for k, v in transform.state_dict().items()},
                'classifier': {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            }

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: trans={total_trans_loss/n_batches:.4f}, cls={total_cls_loss/n_batches:.4f}, "
                  f"val_loss={val_loss:.4f}, F1={avg_f1:.3f} ({', '.join(f'{c}={f:.2f}' for c, f in zip(target_classes, f1_scores))})")

    # Load best
    if best_state:
        transform.load_state_dict(best_state['transform'])
        classifier.load_state_dict(best_state['classifier'])

    transform.eval()
    classifier.eval()

    # Final eval on all segment data
    all_feats = features_norm.to(device)
    all_labels = labels.to(device)

    with torch.no_grad():
        all_trans = transform(all_feats)
        all_logits = classifier(all_trans)
        all_probs = torch.sigmoid(all_logits)
        all_preds = (all_probs > 0.5).float()

    print(f"\nFinal results on all segment data:")
    for i, cls in enumerate(target_classes):
        tp = ((all_preds[:, i] == 1) & (all_labels[:, i] == 1)).sum().item()
        fp = ((all_preds[:, i] == 1) & (all_labels[:, i] == 0)).sum().item()
        fn = ((all_preds[:, i] == 0) & (all_labels[:, i] == 1)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {cls}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} (TP={tp} FP={fp} FN={fn})")

    print(f"\nBest validation F1: {best_val_f1:.3f}")

    # Save
    save_dict = {
        'transform_type': 'MLPTransform',
        'transform_state': transform.state_dict(),
        'classifier_state': classifier.state_dict(),
        'transform_hidden_dims': [1024, 1024],
        'X_mean': X_mean,
        'X_std': X_std,
        'target_classes': target_classes,
        'lambda_classify': args.lambda_classify,
        'num_segments': len(features),
        'num_transform_pairs': len(X_pairs)
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, args.output)
    print(f"\nSaved to: {args.output}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
