#!/usr/bin/env python3
"""
Run the 3-stage classifier pipeline on undefined files in the manifest.

Pipeline:
  1. Mix classifier   — multi-label: detects if file is a mix (2+ instruments)
  2. Group classifier  — classifies isolated (non-mix) tracks into instrument groups
  3. Subgroup classifiers — classifies within each group (e.g. electric_bass vs upright_bass)

Uses pre-trained checkpoints from train_classifiers.py.

Usage:
  python3 classify_undefined.py
  python3 classify_undefined.py --device cuda --workers 12 --batch-size 256
  python3 classify_undefined.py --dry-run  # just count, don't classify
"""

import argparse
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import orjson
import torch
import torch.nn as nn

# ===================== PATHS =====================

MERGED_MANIFEST = Path("/home/arlo/gcs-bucket/Manifests/merged_manifest_v2.json")
FORMAT_MANIFEST = Path("/home/arlo/gcs-bucket/Manifests/format_manifest_clean.json")
LATENTS_ROOT = Path("/home/arlo/gcs-bucket/Latents")
AUDIO_ROOT = Path("/home/arlo/gcs-bucket")

MIX_MODEL_PATH = Path("/home/arlo/Data/mix_classifier/mix_classifier.pt")
GROUP_MODEL_PATH = Path("/home/arlo/Data/latent_classifier/model.pt")
SUBGROUP_DIR = Path("/home/arlo/Data/subgroup_classifiers")

OUTPUT_PATH = Path("/home/arlo/gcs-bucket/Manifests/undefined_predictions.json")

# ===================== CONFIG =====================

MIX_DETECT_THRESHOLD = 0.3
MIX_MIN_CLASSES_FOR_MIX = 2
LATENT_SILENT_THRESHOLD = 0.01
CONFIDENCE_THRESHOLD = 0.4  # minimum confidence to assign a group prediction

# Synth correction: the group classifier's synth class is contaminated with vocals
# (84% of its training data was mislabeled voice files). Use the mix classifier's
# voice score to cross-check: if mix says voice > threshold, override synth -> voice.
SYNTH_VOICE_OVERRIDE_THRESHOLD = 0.8  # mix classifier voice prob to override synth

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')


# ===================== MODELS (must match train_classifiers.py) =====================

class GroupClassifier(nn.Module):
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
    def __init__(self, input_dim=384, hidden_dim=256, num_classes=8):
        super().__init__()
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
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.transform(x))


# ===================== FEATURE EXTRACTION =====================

def audio_path_to_latent_path(audio_path: str) -> Optional[Path]:
    """Convert audio path to latent path."""
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
            latent.mean(dim=-1),
            latent.std(dim=-1) if latent.shape[-1] > 1 else torch.zeros_like(latent.mean(dim=-1)),
            latent.max(dim=-1)[0],
        ]
        return torch.stack(pools, dim=-1).flatten()
    except Exception:
        return None


# ===================== LOADING =====================

def get_undefined_paths_with_latents() -> List[str]:
    """Get undefined audio paths that have latents available."""
    logging.info("Loading manifests...")

    # Build set of relative paths that have latents (from format manifest)
    with open(FORMAT_MANIFEST, 'rb') as f:
        fm = orjson.loads(f.read())
    latent_rel_paths = {e['path'] for e in fm['entries'] if e.get('has_latent')}
    logging.info(f"  Format manifest: {len(latent_rel_paths):,} paths with latents")

    # Load merged manifest, find undefined entries
    with open(MERGED_MANIFEST, 'rb') as f:
        mm = orjson.loads(f.read())

    audio_root_str = str(AUDIO_ROOT) + '/'
    undefined_paths = []
    for path, entry in mm['entries'].items():
        if entry.get('group') != 'undefined':
            continue
        # Convert to relative path to check latent availability
        rel = path[len(audio_root_str):] if path.startswith(audio_root_str) else path
        if rel in latent_rel_paths:
            undefined_paths.append(path)

    logging.info(f"  Undefined with latents: {len(undefined_paths):,}")
    return undefined_paths


def load_models(device: str):
    """Load all classifier checkpoints."""
    models = {}

    # Mix classifier
    logging.info("Loading mix classifier...")
    ckpt = torch.load(MIX_MODEL_PATH, map_location='cpu', weights_only=False)
    mix_model = MixClassifier(
        ckpt['input_dim'], ckpt.get('hidden_dim', 256), ckpt['num_classes']
    )
    mix_model.load_state_dict(ckpt['model_state'])
    mix_model.to(device).eval()
    models['mix'] = {
        'model': mix_model,
        'mean': ckpt['X_mean'].to(device),
        'std': ckpt['X_std'].to(device),
        'classes': ckpt['target_classes'],
    }
    logging.info(f"  Mix classes: {ckpt['target_classes']}")

    # Group classifier
    logging.info("Loading group classifier...")
    ckpt = torch.load(GROUP_MODEL_PATH, map_location='cpu', weights_only=False)
    group_model = GroupClassifier(
        ckpt['input_dim'], ckpt['num_classes'], ckpt.get('hidden_dim', 256)
    )
    group_model.load_state_dict(ckpt['model_state'])
    group_model.to(device).eval()
    models['group'] = {
        'model': group_model,
        'mean': ckpt['mean'].to(device),
        'std': ckpt['std'].to(device),
        'classes': ckpt['label_encoder_classes'],
    }
    logging.info(f"  Group classes: {ckpt['label_encoder_classes']}")

    # Subgroup classifiers
    models['subgroups'] = {}
    for sg_path in sorted(SUBGROUP_DIR.glob("*_subgroup_model.pt")):
        ckpt = torch.load(sg_path, map_location='cpu', weights_only=False)
        group = ckpt['group']
        sg_model = SubgroupClassifier(
            ckpt['input_dim'], ckpt.get('hidden_dim', 256), ckpt['num_classes']
        )
        sg_model.load_state_dict(ckpt['model_state'])
        sg_model.to(device).eval()
        models['subgroups'][group] = {
            'model': sg_model,
            'mean': ckpt['mean'].to(device),
            'std': ckpt['std'].to(device),
            'classes': ckpt['classes'],
        }
        logging.info(f"  Subgroup [{group}]: {ckpt['classes']}")

    return models


# ===================== INFERENCE =====================

def classify_batch(features: torch.Tensor, paths: List[str],
                   models: dict, device: str, batch_size: int) -> List[dict]:
    """Run 3-stage classification on a batch of features."""
    results = []
    n = len(features)

    mix_info = models['mix']
    group_info = models['group']
    subgroup_models = models['subgroups']

    # Normalize for each model
    X_mix = (features.to(device) - mix_info['mean']) / mix_info['std']
    X_group = (features.to(device) - group_info['mean']) / group_info['std']

    # Stage 1: Mix detection
    mix_probs_all = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = X_mix[i:i + batch_size]
            logits = mix_info['model'](batch)
            probs = torch.sigmoid(logits).cpu()
            mix_probs_all.append(probs)
    mix_probs_all = torch.cat(mix_probs_all)

    # Stage 2: Group classification
    group_logits_all = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = X_group[i:i + batch_size]
            logits = group_info['model'](batch)
            group_logits_all.append(logits.cpu())
    group_logits_all = torch.cat(group_logits_all)
    group_probs_all = torch.softmax(group_logits_all, dim=1)

    # Pre-compute subgroup predictions per group
    subgroup_preds = {}
    for group, sg_info in subgroup_models.items():
        X_sg = (features.to(device) - sg_info['mean']) / sg_info['std']
        sg_logits = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = X_sg[i:i + batch_size]
                logits = sg_info['model'](batch)
                sg_logits.append(logits.cpu())
        sg_logits = torch.cat(sg_logits)
        sg_probs = torch.softmax(sg_logits, dim=1)
        subgroup_preds[group] = sg_probs

    # Combine results per file
    for i in range(n):
        mix_probs = mix_probs_all[i]
        classes_above = (mix_probs >= MIX_DETECT_THRESHOLD).sum().item()
        is_mix = classes_above >= MIX_MIN_CLASSES_FOR_MIX

        # Mix instrument breakdown
        mix_instruments = {}
        for j, cls in enumerate(mix_info['classes']):
            if mix_probs[j].item() >= MIX_DETECT_THRESHOLD:
                mix_instruments[cls] = round(mix_probs[j].item(), 4)

        if is_mix:
            result = {
                'audio_path': paths[i],
                'predicted_group': 'mix',
                'is_mix': True,
                'mix_instruments': mix_instruments,
                'mix_confidence': round(max(mix_probs.tolist()), 4),
                'subgroup': None,
                'subgroup_confidence': None,
            }
        else:
            # Group prediction
            group_probs = group_probs_all[i]
            top_idx = group_probs.argmax().item()
            top_conf = group_probs[top_idx].item()
            predicted_group = group_info['classes'][top_idx]

            # Top-2 for context
            sorted_indices = group_probs.argsort(descending=True)
            top2_group = group_info['classes'][sorted_indices[1].item()]
            top2_conf = group_probs[sorted_indices[1].item()].item()

            # Cross-check: synth class is contaminated with vocals in training data.
            # If group classifier says synth but mix classifier's voice score is high,
            # override to voice (the mix classifier was trained on clean data).
            synth_overridden = False
            if predicted_group == 'synth':
                voice_idx = mix_info['classes'].index('voice') if 'voice' in mix_info['classes'] else -1
                if voice_idx >= 0:
                    mix_voice_prob = mix_probs[voice_idx].item()
                    if mix_voice_prob >= SYNTH_VOICE_OVERRIDE_THRESHOLD:
                        predicted_group = 'voice'
                        synth_overridden = True

            # Subgroup prediction if available
            subgroup = None
            subgroup_conf = None
            if predicted_group in subgroup_preds:
                sg_probs = subgroup_preds[predicted_group][i]
                sg_idx = sg_probs.argmax().item()
                sg_conf = sg_probs[sg_idx].item()
                subgroup = subgroup_models[predicted_group]['classes'][sg_idx]
                subgroup_conf = round(sg_conf, 4)

            result = {
                'audio_path': paths[i],
                'predicted_group': predicted_group if top_conf >= CONFIDENCE_THRESHOLD else 'low_confidence',
                'group_confidence': round(top_conf, 4),
                'group_runner_up': top2_group,
                'group_runner_up_confidence': round(top2_conf, 4),
                'is_mix': False,
                'mix_instruments': mix_instruments if mix_instruments else None,
                'subgroup': subgroup,
                'subgroup_confidence': subgroup_conf,
                'synth_overridden': synth_overridden,
            }

        results.append(result)

    return results


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description='Classify undefined files using 3-stage pipeline')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--dry-run', action='store_true', help='Just count files, do not classify')
    parser.add_argument('--output', type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device: {device}")

    # Get undefined files with latents
    undefined_paths = get_undefined_paths_with_latents()

    if args.dry_run:
        logging.info(f"Dry run: {len(undefined_paths):,} undefined files with latents to classify")
        return

    if not undefined_paths:
        logging.info("No undefined files with latents to classify")
        return

    # Load models
    models = load_models(device)

    # Extract features
    logging.info(f"\nExtracting features for {len(undefined_paths):,} files...")
    features_list = []
    valid_paths = []
    processed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(load_and_pool, p): p for p in undefined_paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                feat = future.result()
                if feat is not None:
                    features_list.append(feat)
                    valid_paths.append(path)
                else:
                    failed += 1
            except Exception:
                failed += 1
            processed += 1
            if processed % 2000 == 0:
                logging.info(f"  Features: {processed:,}/{len(undefined_paths):,} ({len(features_list):,} valid, {failed:,} failed)")

    logging.info(f"  Features done: {len(features_list):,} valid, {failed:,} failed out of {len(undefined_paths):,}")

    if not features_list:
        logging.error("No features extracted!")
        return

    features = torch.stack(features_list)

    # Run 3-stage classification
    logging.info(f"\nRunning 3-stage classification on {len(features):,} files...")
    results = classify_batch(features, valid_paths, models, device, args.batch_size)

    # Summary
    group_counts = Counter(r['predicted_group'] for r in results)
    mix_count = sum(1 for r in results if r['is_mix'])
    synth_override_count = sum(1 for r in results if r.get('synth_overridden'))

    logging.info(f"\n{'='*60}")
    logging.info(f"CLASSIFICATION RESULTS")
    logging.info(f"{'='*60}")
    logging.info(f"Total classified: {len(results):,}")
    logging.info(f"Detected mixes: {mix_count:,}")
    logging.info(f"Synth -> voice overrides: {synth_override_count:,}")
    logging.info(f"\nGroup distribution:")
    for g, c in group_counts.most_common():
        logging.info(f"  {g}: {c:,}")

    subgroup_counts = Counter(r['subgroup'] for r in results if r['subgroup'])
    if subgroup_counts:
        logging.info(f"\nSubgroup distribution:")
        for sg, c in subgroup_counts.most_common():
            logging.info(f"  {sg}: {c:,}")

    # Save
    output_path = Path(args.output)
    output = {
        'generated_at': datetime.now().isoformat(),
        'total_classified': len(results),
        'total_undefined': len(undefined_paths),
        'features_failed': failed,
        'mix_threshold': MIX_DETECT_THRESHOLD,
        'min_classes_for_mix': MIX_MIN_CLASSES_FOR_MIX,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'group_distribution': dict(group_counts.most_common()),
        'predictions': results,
    }
    with open(output_path, 'wb') as f:
        f.write(orjson.dumps(output, option=orjson.OPT_INDENT_2))

    logging.info(f"\nSaved {len(results):,} predictions to {output_path}")
    logging.info("Done!")


if __name__ == "__main__":
    main()
