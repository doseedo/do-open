#!/usr/bin/env python3
"""
Run subgroup classifiers on all relevant groups in master manifest.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

MANIFEST_PATH = "/home/arlo/gcs-bucket/Manifests/master_manifest.json"
MODELS_DIR = Path("/home/arlo/Data/subgroup_classifiers")
LATENTS_BASE = Path("/home/arlo/gcs-bucket/Latents")
OUTPUT_FILE = "/home/arlo/Data/subgroup_classifiers/all_predictions.json"

GROUPS_WITH_SUBGROUPS = ['brass', 'strings', 'winds', 'bass', 'guitar', 'piano']


class SubgroupClassifier(nn.Module):
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


def load_model(group):
    """Load subgroup classifier for a group."""
    model_path = MODELS_DIR / f"{group}_subgroup_model.pt"
    if not model_path.exists():
        return None

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = SubgroupClassifier(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return {
        'model': model,
        'classes': checkpoint['classes'],
        'mean': checkpoint['mean'],
        'std': checkpoint['std']
    }


def extract_features(latent):
    """Extract 384-dim features from latent using multi-pool."""
    pools = []
    for method in ['mean', 'std', 'max']:
        if method == 'mean':
            pooled = latent.mean(dim=-1)
        elif method == 'std':
            pooled = latent.std(dim=-1) if latent.shape[-1] > 1 else torch.zeros_like(latent.mean(dim=-1))
        elif method == 'max':
            pooled = latent.max(dim=-1)[0]
        pools.append(pooled.flatten())
    return torch.cat(pools)


def classify_file(model_dict, latent_path):
    """Classify a single file."""
    data = torch.load(latent_path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        latent = data.get('latents', data.get('latent'))
    else:
        latent = data

    features = extract_features(latent)
    features = (features - model_dict['mean']) / (model_dict['std'] + 1e-8)

    with torch.no_grad():
        logits = model_dict['model'](features.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1).squeeze()

    pred_idx = probs.argmax().item()
    pred_class = model_dict['classes'][pred_idx]
    confidence = probs[pred_idx].item()

    return {
        'predicted': pred_class,
        'confidence': confidence,
        'all_probs': {c: float(probs[i]) for i, c in enumerate(model_dict['classes'])}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=MANIFEST_PATH)
    parser.add_argument('--output', default=OUTPUT_FILE)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)

    entries = manifest.get('entries', manifest)  # Handle both formats
    if not isinstance(entries, dict):
        # Convert list format to dict
        entries = {e.get('path'): e for e in entries if 'path' in e}

    print(f"Found {len(entries)} entries")

    # Load all models
    models = {}
    for group in GROUPS_WITH_SUBGROUPS:
        m = load_model(group)
        if m:
            models[group] = m
            print(f"Loaded {group} model: {m['classes']}")

    # Collect files by group (don't check existence upfront - too slow on GCS)
    files_by_group = {g: [] for g in GROUPS_WITH_SUBGROUPS}
    for path, entry in entries.items():
        group = entry.get('group')
        if group in files_by_group:
            # Derive latent path from audio path
            latent_path = path.replace('/gcs-bucket/protools', '/gcs-bucket/Latents/protools')
            latent_path = latent_path.rsplit('.', 1)[0] + '.dcae.pt'
            files_by_group[group].append({
                'path': path,
                'latent_path': latent_path,
                'current_subgroup': entry.get('subgroup')
            })

    for g in GROUPS_WITH_SUBGROUPS:
        print(f"  {g}: {len(files_by_group[g])} files")

    # Run classifiers
    all_results = {}
    for group, files in files_by_group.items():
        if group not in models:
            continue

        if args.limit:
            files = files[:args.limit]

        print(f"\nClassifying {len(files)} {group} files...")
        results = []

        errors = 0
        for f in tqdm(files, desc=group):
            try:
                if not Path(f['latent_path']).exists():
                    errors += 1
                    continue
                pred = classify_file(models[group], f['latent_path'])
                results.append({
                    'path': f['path'],
                    'latent_path': f['latent_path'],
                    'current_subgroup': f['current_subgroup'],
                    **pred
                })
            except Exception as e:
                errors += 1

        print(f"  {len(results)} classified, {errors} errors/missing")

        all_results[group] = results

        # Quick stats
        if results:
            from collections import Counter
            preds = [r.get('predicted') for r in results if 'predicted' in r]
            counts = Counter(preds)
            print(f"  Predictions: {dict(counts)}")

    # Save
    output = {
        'groups': GROUPS_WITH_SUBGROUPS,
        'models_loaded': list(models.keys()),
        'results': all_results
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
