#!/usr/bin/env python3
"""
Learn transform from isolated stem features to mix features.
Uses ALL 6 Demucs stems as bridge to learn a general mix->isolated mapping.

Then apply transform to brass/strings/winds weights from solo classifier
to create mix-space classifier for those classes.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

# Paths
GCS_BASE = Path("/home/arlo/gcs-bucket")
LATENTS_BASE = GCS_BASE / "Latents"
V2_LATENTS_BASE = GCS_BASE / "LatentDemucsV2"
SOLO_MODEL_PATH = Path("/home/arlo/Data/latent_classifier/model.pt")
OUTPUT_DIR = Path("/home/arlo/Data/mix_classifier")

# Feature extraction (same as classifier)
POOL_METHODS = ['mean', 'std', 'max']

def extract_features(latent):
    """Extract 384-dim features from latent."""
    # latent shape: [8, 16, T]
    pools = []
    for method in POOL_METHODS:
        if method == 'mean':
            pooled = latent.mean(dim=-1)
        elif method == 'std':
            pooled = latent.std(dim=-1)
        elif method == 'max':
            pooled = latent.max(dim=-1)[0]
        pools.append(pooled.flatten())
    return torch.cat(pools)  # [384]


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
        stem_parent = stem_dir.parent  # e.g., .../Audio Files/filename/

        # Build path to original mix latent
        rel_path = str(stem_parent).replace(str(V2_LATENTS_BASE) + "/", "")
        mix_latent_path = LATENTS_BASE / (rel_path + ".pt")

        if mix_latent_path.exists():
            pairs.append({
                'mix_latent': mix_latent_path,
                'stem_dir': stem_parent,
                'name': stem_parent.name
            })

    return pairs


def extract_paired_features(pairs, stem_name='guitar'):
    """Extract aligned features from mix and stem latents (single stem)."""
    mix_features = []
    stem_features = []

    for pair in pairs:
        stem_path = pair['stem_dir'] / f"{stem_name}.pt"

        if not stem_path.exists():
            continue

        try:
            # Load latents
            mix_latent = load_latent(pair['mix_latent'])
            stem_latent = load_latent(stem_path)

            if mix_latent is None or stem_latent is None:
                continue

            # Check if stem has content (not silent)
            stem_energy = stem_latent.abs().mean()
            if stem_energy < 0.01:  # Skip silent stems
                continue

            # Extract whole-file features
            mix_feat = extract_features(mix_latent)
            stem_feat = extract_features(stem_latent)

            mix_features.append(mix_feat)
            stem_features.append(stem_feat)

        except Exception as e:
            continue

    if not mix_features:
        return None, None

    return torch.stack(mix_features), torch.stack(stem_features)


def extract_all_stem_pairs(pairs):
    """Extract features from ALL 6 Demucs stems for a more general transform."""
    all_mix_features = []
    all_stem_features = []
    stem_counts = defaultdict(int)

    stems_to_use = ["drums", "bass", "vocals", "other", "guitar", "piano"]

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

                # Skip silent stems
                stem_energy = stem_latent.abs().mean()
                if stem_energy < 0.01:
                    continue

                stem_feat = extract_features(stem_latent)

                all_mix_features.append(mix_feat)
                all_stem_features.append(stem_feat)
                stem_counts[stem_name] += 1

        except Exception as e:
            continue

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(pairs)} files, {len(all_mix_features)} pairs so far")

    print(f"\nPairs per stem: {dict(stem_counts)}")
    print(f"Total pairs: {len(all_mix_features)}")

    if not all_mix_features:
        return None, None

    return torch.stack(all_mix_features), torch.stack(all_stem_features)


def learn_transform(mix_features, stem_features):
    """Learn linear transform: stem_features ≈ W @ mix_features + b"""
    # Use least squares to learn W and b
    # stem = W @ mix + b
    # Add bias by appending 1s to mix features

    X = mix_features.numpy()  # [N, 384]
    Y = stem_features.numpy()  # [N, 384]

    # Add bias term
    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])  # [N, 385]

    # Solve: Y = X_bias @ W_full.T
    # W_full is [384, 385] = [W, b]
    W_full, residuals, rank, s = np.linalg.lstsq(X_bias, Y, rcond=None)

    # W_full is [385, 384], we want W [384, 384] and b [384]
    W = W_full[:-1, :].T  # [384, 384]
    b = W_full[-1, :]     # [384]

    # Compute reconstruction error
    Y_pred = X @ W.T + b
    mse = np.mean((Y - Y_pred) ** 2)

    return torch.tensor(W, dtype=torch.float32), torch.tensor(b, dtype=torch.float32), mse


def learn_mlp_transform(mix_features, stem_features, epochs=500, lr=0.001, hidden_dims=[512, 512]):
    """Learn MLP transform: stem_features ≈ MLP(mix_features)"""

    class MLPTransform(nn.Module):
        def __init__(self, input_dim=384, output_dim=384, hidden_dims=[512, 512]):
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

    # Split into train/val
    n = len(mix_features)
    indices = torch.randperm(n)
    train_idx = indices[:int(0.9 * n)]
    val_idx = indices[int(0.9 * n):]

    X_train, Y_train = mix_features[train_idx], stem_features[train_idx]
    X_val, Y_val = mix_features[val_idx], stem_features[val_idx]

    # Normalize inputs
    X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-6
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std

    # Model
    model = MLPTransform(384, 384, hidden_dims)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        batch_size = 256
        perm = torch.randperm(len(X_train_norm))
        total_loss = 0

        for i in range(0, len(X_train_norm), batch_size):
            batch_idx = perm[i:i+batch_size]
            x_batch = X_train_norm[batch_idx]
            y_batch = Y_train[batch_idx]

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = nn.functional.mse_loss(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_norm)
            val_loss = nn.functional.mse_loss(val_pred, Y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: train_loss={total_loss/(len(X_train_norm)//batch_size+1):.4f}, val_loss={val_loss:.4f}")

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Final MSE on full data
    X_all_norm = (mix_features - X_mean) / X_std
    with torch.no_grad():
        pred_all = model(X_all_norm)
        final_mse = nn.functional.mse_loss(pred_all, stem_features).item()

    print(f"  Best val_loss: {best_val_loss:.4f}, Final MSE: {final_mse:.4f}")

    return model, X_mean, X_std, final_mse


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlp', action='store_true', help='Use MLP transform instead of linear')
    parser.add_argument('--epochs', type=int, default=500, help='MLP training epochs')
    parser.add_argument('--hidden', type=int, nargs='+', default=[512, 512], help='MLP hidden dims')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"LEARNING MIX -> STEM FEATURE TRANSFORM ({'MLP' if args.mlp else 'Linear'})")
    print("=" * 60)

    # Find matched pairs
    print("\nFinding matched pairs...")
    pairs = find_matched_pairs()
    print(f"Found {len(pairs)} matched pairs")

    if len(pairs) < 50:
        print("Not enough pairs! Need at least 50.")
        return

    # Load solo classifier to get class weights
    print("\nLoading solo classifier...")
    solo_checkpoint = torch.load(SOLO_MODEL_PATH, map_location='cpu', weights_only=False)
    solo_classes = solo_checkpoint['label_encoder_classes']
    print(f"Solo classifier classes: {solo_classes}")

    # Extract paired features using ALL 6 stems (not just guitar)
    print("\nExtracting paired features from ALL 6 Demucs stems...")
    mix_feats, stem_feats = extract_all_stem_pairs(pairs)

    if mix_feats is None:
        print("Failed to extract paired features!")
        return

    print(f"Extracted {len(mix_feats)} paired samples (from all stems)")

    # Learn transform
    if args.mlp:
        print(f"\nLearning MLP transform (hidden={args.hidden}, epochs={args.epochs})...")
        mlp_model, X_mean, X_std, mse = learn_mlp_transform(
            mix_feats, stem_feats,
            epochs=args.epochs,
            hidden_dims=args.hidden
        )
        print(f"MLP Transform learned. MSE: {mse:.6f}")

        # Save MLP transform
        transform_path = OUTPUT_DIR / "mix_to_stem_transform_mlp.pt"
        torch.save({
            'mlp_state': mlp_model.state_dict(),
            'X_mean': X_mean,
            'X_std': X_std,
            'hidden_dims': args.hidden,
            'mse': mse,
            'transform_type': 'mlp',
            'bridge_classes': ['drums', 'bass', 'vocals', 'other', 'guitar', 'piano'],
            'num_samples': len(mix_feats)
        }, transform_path)
        print(f"\nMLP Transform saved to {transform_path}")
    else:
        print("\nLearning linear transform...")
        W, b, mse = learn_transform(mix_feats, stem_feats)
        print(f"Transform learned. MSE: {mse:.6f}")
        print(f"W shape: {W.shape}, b shape: {b.shape}")

        # Save linear transform
        transform_path = OUTPUT_DIR / "mix_to_stem_transform.pt"
        torch.save({
            'W': W,
            'b': b,
            'mse': mse,
            'transform_type': 'linear',
            'bridge_classes': ['drums', 'bass', 'vocals', 'other', 'guitar', 'piano'],
            'num_samples': len(mix_feats)
        }, transform_path)
        print(f"\nTransform saved to {transform_path}")

    # Now create mix-space classifier for brass/strings/winds
    print("\n" + "=" * 60)
    print("CREATING MIX-SPACE CLASSIFIER")
    print("=" * 60)

    # Get solo classifier's final layer weights
    # The classifier is: features -> hidden -> hidden -> logits
    # We need to transform the input features, then use same weights

    solo_state = solo_checkpoint['model_state']
    solo_mean = solo_checkpoint['mean']
    solo_std = solo_checkpoint['std']

    # Find indices for brass, strings, winds in solo classifier
    target_classes = ['brass', 'strings', 'winds']
    target_indices = [list(solo_classes).index(c) for c in target_classes if c in solo_classes]
    print(f"Target classes in solo: {[solo_classes[i] for i in target_indices]}")

    # The approach:
    # 1. Mix features -> transform -> stem-space features
    # 2. Normalize with solo mean/std
    # 3. Run through solo classifier

    # Save the combined model info
    if args.mlp:
        mix_classifier_data = {
            'transform_type': 'mlp',
            'mlp_state': mlp_model.state_dict(),
            'mlp_X_mean': X_mean,
            'mlp_X_std': X_std,
            'mlp_hidden_dims': args.hidden,
            'solo_model_state': solo_state,
            'solo_mean': solo_mean,
            'solo_std': solo_std,
            'solo_classes': list(solo_classes),
            'target_classes': target_classes,
            'target_indices': target_indices,
        }
        mix_classifier_path = OUTPUT_DIR / "mix_classifier_mlp.pt"
    else:
        mix_classifier_data = {
            'transform_type': 'linear',
            'transform_W': W,
            'transform_b': b,
            'solo_model_state': solo_state,
            'solo_mean': solo_mean,
            'solo_std': solo_std,
            'solo_classes': list(solo_classes),
            'target_classes': target_classes,
            'target_indices': target_indices,
        }
        mix_classifier_path = OUTPUT_DIR / "mix_classifier.pt"

    torch.save(mix_classifier_data, mix_classifier_path)
    print(f"Mix classifier saved to {mix_classifier_path}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nTo use: load mix_classifier.pt, apply transform to mix features,")
    print(f"then run through solo classifier and extract target class logits.")


if __name__ == "__main__":
    main()
