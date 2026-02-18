#!/usr/bin/env python3
"""
Train a temporal multi-label classifier to predict which Demucs stems are active
in a mix recording, using mix latents as input and stem energy as ground truth.

Training data: ~2960 files with Demucs-separated stem latents + original mix latents.
For each windowed segment of the mix latent, the label is which stems have energy > threshold.

Inference: given any mix latent, predict temporal presence of 6 stems (drums, bass, vocals, guitar, piano, other).

Usage:
  python3 train_stem_predictor.py --mode train
  python3 train_stem_predictor.py --mode predict --manifest /path/to/manifest.json --output predictions.json
  python3 train_stem_predictor.py --mode predict --input /path/to/mix_latent.pt
"""

import argparse
import logging
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import orjson
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===================== CONFIG =====================

DEMUCS_LATENTS_ROOT = Path("/home/arlo/gcs-bucket/LatentDemucsV2")
MIX_LATENTS_ROOT = Path("/home/arlo/gcs-bucket/Latents")
PROGRESS_FILE = Path("/home/arlo/Data/demucs_latent_progress.json")
OUTPUT_DIR = Path("/home/arlo/Data/stem_predictor")
AUDIO_ROOT = Path("/home/arlo/gcs-bucket")

STEMS = ['drums', 'bass', 'vocals', 'guitar', 'piano', 'other']
FRAMES_PER_SEC = 10.77
WINDOW_FRAMES = 50  # ~4.6 seconds
HOP_FRAMES = 25     # ~2.3 seconds (50% overlap for training)
ENERGY_THRESHOLD = 0.10  # stem active if energy > this
SILENCE_THRESHOLD = 0.05  # skip mix windows below this

# ===================== MODEL =====================

class StemPredictor(nn.Module):
    """Multi-label temporal classifier: mix features -> 6 stem presence."""
    def __init__(self, input_dim=384, hidden_dim=256, num_stems=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_stems),
        )

    def forward(self, x):
        return self.net(x)  # raw logits, use sigmoid for probs


# ===================== FEATURE EXTRACTION =====================

def extract_window_features(latent: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Extract 384-dim features from a time window using multi-pool."""
    segment = latent[:, :, start:end]
    mean_pool = segment.mean(dim=-1)
    std_pool = segment.std(dim=-1) if segment.shape[-1] > 1 else torch.zeros_like(mean_pool)
    max_pool = segment.max(dim=-1)[0]
    return torch.cat([mean_pool.flatten(), std_pool.flatten(), max_pool.flatten()])


def compute_stem_energy(latent: torch.Tensor, start: int, end: int) -> float:
    """Compute energy for a stem latent window."""
    segment = latent[:, :, start:end]
    return float(segment.abs().mean())


def load_latent(path: Path) -> Optional[torch.Tensor]:
    """Load a latent tensor from file."""
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            return data.get('latents', data.get('latent'))
        return data
    except:
        return None


# ===================== DATA BUILDING =====================

def find_training_pairs() -> List[Dict]:
    """Find all mix latent + stem latent pairs."""
    # Load progress file to find which files have been separated
    if not PROGRESS_FILE.exists():
        logging.error(f"Progress file not found: {PROGRESS_FILE}")
        return []

    with open(PROGRESS_FILE, 'rb') as f:
        progress = orjson.loads(f.read())

    pairs = []
    completed = progress.get('completed', {})
    logging.info(f"Checking {len(completed)} completed Demucs separations...")

    for audio_path, info in completed.items():
        stem_dir = info.get('latent_dir')
        if not stem_dir:
            continue
        stem_dir = Path(stem_dir)

        # Check all 6 stems exist
        stem_paths = {}
        all_exist = True
        for stem in STEMS:
            sp = stem_dir / f"{stem}.pt"
            if sp.exists():
                stem_paths[stem] = sp
            else:
                all_exist = False
                break

        if not all_exist:
            continue

        # Find corresponding mix latent
        audio_p = Path(audio_path)
        try:
            rel_path = audio_p.relative_to(AUDIO_ROOT)
        except ValueError:
            parts = audio_p.parts
            if 'protools' in parts:
                idx = parts.index('protools')
                rel_path = Path(*parts[idx:])
            else:
                continue

        stem = rel_path.with_suffix('')
        mix_latent = MIX_LATENTS_ROOT / f"{stem}.dcae.pt"
        if not mix_latent.exists():
            mix_latent = MIX_LATENTS_ROOT / f"{stem}.pt"
        if not mix_latent.exists():
            continue

        pairs.append({
            'audio_path': audio_path,
            'mix_latent': mix_latent,
            'stem_paths': stem_paths,
        })

    logging.info(f"Found {len(pairs)} complete training pairs")
    return pairs


def build_windowed_dataset(pairs: List[Dict], max_pairs: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build windowed training data from all pairs.

    Returns: (features [N, 384], labels [N, 6])
    """
    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    all_features = []
    all_labels = []
    skipped = 0
    processed = 0

    for i, pair in enumerate(pairs):
        if (i + 1) % 200 == 0:
            logging.info(f"  Processing {i+1}/{len(pairs)}, {len(all_features)} windows so far")

        # Load mix latent
        mix_latent = load_latent(pair['mix_latent'])
        if mix_latent is None:
            skipped += 1
            continue

        if mix_latent.dim() == 2:
            mix_latent = mix_latent.unsqueeze(0)

        T_mix = mix_latent.shape[-1]
        if T_mix < WINDOW_FRAMES:
            skipped += 1
            continue

        # Load all stem latents
        stem_latents = {}
        ok = True
        for stem_name, stem_path in pair['stem_paths'].items():
            sl = load_latent(stem_path)
            if sl is None:
                ok = False
                break
            if sl.dim() == 2:
                sl = sl.unsqueeze(0)
            stem_latents[stem_name] = sl
        if not ok:
            skipped += 1
            continue

        # Window through the mix and compute labels from stem energies
        for start in range(0, T_mix - WINDOW_FRAMES + 1, HOP_FRAMES):
            end = start + WINDOW_FRAMES

            # Check mix energy
            mix_energy = compute_stem_energy(mix_latent, start, end)
            if mix_energy < SILENCE_THRESHOLD:
                continue

            # Extract mix features
            features = extract_window_features(mix_latent, start, end)

            # Compute stem labels: energy > threshold = active
            # Stem latents may have different T due to Demucs processing
            # Map mix frames to stem frames proportionally
            labels = []
            for stem_name in STEMS:
                sl = stem_latents[stem_name]
                T_stem = sl.shape[-1]
                # Proportional mapping
                s_start = int(start * T_stem / T_mix)
                s_end = int(end * T_stem / T_mix)
                s_end = min(s_end, T_stem)
                s_start = min(s_start, T_stem - 1)
                if s_end <= s_start:
                    labels.append(0.0)
                else:
                    energy = compute_stem_energy(sl, s_start, s_end)
                    labels.append(1.0 if energy > ENERGY_THRESHOLD else 0.0)

            all_features.append(features)
            all_labels.append(torch.tensor(labels))
        processed += 1

    logging.info(f"Built {len(all_features)} windows from {processed} files ({skipped} skipped)")

    if not all_features:
        return torch.tensor([]), torch.tensor([])

    return torch.stack(all_features), torch.stack(all_labels)


# ===================== TRAINING =====================

def train(args):
    logging.info("=" * 50)
    logging.info("STEM PREDICTOR TRAINING")
    logging.info("=" * 50)

    device = args.device if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find training pairs
    pairs = find_training_pairs()
    if not pairs:
        logging.error("No training pairs found!")
        return

    random.shuffle(pairs)

    # Build dataset
    logging.info("Building windowed training data...")
    features, labels = build_windowed_dataset(pairs, max_pairs=args.limit)
    if features.numel() == 0:
        logging.error("No training data!")
        return

    logging.info(f"Dataset: {features.shape[0]} windows, {features.shape[1]} features, {labels.shape[1]} stems")

    # Label distribution
    for i, stem in enumerate(STEMS):
        active = labels[:, i].sum().item()
        logging.info(f"  {stem}: {int(active)}/{len(labels)} ({100*active/len(labels):.1f}% active)")

    # Train/val split
    n = len(features)
    indices = torch.randperm(n)
    split = int(0.85 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]

    # Normalize
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0).clamp(min=1e-6)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Compute pos_weight for class imbalance
    pos_counts = y_train.sum(dim=0)
    neg_counts = len(y_train) - pos_counts
    pos_weight = (neg_counts / pos_counts.clamp(min=1)).to(device)
    logging.info(f"Pos weights: {dict(zip(STEMS, pos_weight.tolist()))}")

    # Dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = StemPredictor(input_dim=384, hidden_dim=256, num_stems=len(STEMS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0.0
    best_state = None

    for epoch in range(args.epochs):
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
        all_preds = []
        all_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                preds = (torch.sigmoid(logits) > 0.5).cpu().float()
                all_preds.append(preds)
                all_true.append(y_batch)

        all_preds = torch.cat(all_preds)
        all_true = torch.cat(all_true)

        # Per-stem F1
        per_stem_f1 = []
        for i in range(len(STEMS)):
            tp = ((all_preds[:, i] == 1) & (all_true[:, i] == 1)).sum().float()
            fp = ((all_preds[:, i] == 1) & (all_true[:, i] == 0)).sum().float()
            fn = ((all_preds[:, i] == 0) & (all_true[:, i] == 1)).sum().float()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            per_stem_f1.append(f1.item())

        macro_f1 = np.mean(per_stem_f1)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            stem_f1_str = " | ".join(f"{s}:{f:.3f}" for s, f in zip(STEMS, per_stem_f1))
            logging.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss/len(train_loader):.4f} | F1: {macro_f1:.4f} | {stem_f1_str}")

    # Save
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    checkpoint = {
        'model_state': best_state,
        'stems': STEMS,
        'input_dim': 384,
        'hidden_dim': 256,
        'num_stems': len(STEMS),
        'mean': mean,
        'std': std,
        'best_f1': best_f1,
        'window_frames': WINDOW_FRAMES,
        'hop_frames': HOP_FRAMES,
        'frames_per_sec': FRAMES_PER_SEC,
        'energy_threshold': ENERGY_THRESHOLD,
        'trained_at': datetime.now().isoformat(),
    }

    save_path = OUTPUT_DIR / "model.pt"
    torch.save(checkpoint, save_path)
    logging.info(f"Saved to {save_path} | Best macro F1: {best_f1:.4f}")
    logging.info("Training complete")


# ===================== PREDICTION =====================

def load_model(model_path: Path, device: str = 'cpu'):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = StemPredictor(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_stems=checkpoint['num_stems'],
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    return {
        'model': model,
        'mean': checkpoint['mean'].to(device),
        'std': checkpoint['std'].to(device),
        'stems': checkpoint['stems'],
        'window_frames': checkpoint['window_frames'],
        'frames_per_sec': checkpoint['frames_per_sec'],
    }


def predict_file(model_dict: dict, latent_path: Path, hop_frames: int = 25, threshold: float = 0.5) -> Dict:
    """Predict stem presence over time for a single mix latent."""
    latent = load_latent(latent_path)
    if latent is None:
        return None

    if latent.dim() == 2:
        latent = latent.unsqueeze(0)

    model = model_dict['model']
    mean = model_dict['mean']
    std = model_dict['std']
    stems = model_dict['stems']
    win = model_dict['window_frames']
    fps = model_dict['frames_per_sec']

    T = latent.shape[-1]
    if T < win:
        return None

    temporal = []
    with torch.no_grad():
        for start in range(0, T - win + 1, hop_frames):
            end = start + win
            energy = compute_stem_energy(latent, start, end)
            if energy < SILENCE_THRESHOLD:
                temporal.append({
                    'start_sec': round(start / fps, 2),
                    'end_sec': round(end / fps, 2),
                    'active': [],
                    'probs': {s: 0.0 for s in stems},
                })
                continue

            features = extract_window_features(latent, start, end)
            features = (features - mean) / (std + 1e-8)
            features = features.unsqueeze(0).to(next(model.parameters()).device)

            logits = model(features)
            probs = torch.sigmoid(logits).squeeze().cpu()

            active = [s for s, p in zip(stems, probs) if p > threshold]
            temporal.append({
                'start_sec': round(start / fps, 2),
                'end_sec': round(end / fps, 2),
                'active': active,
                'probs': {s: round(float(p), 3) for s, p in zip(stems, probs)},
            })

    # Merge contiguous active regions per stem
    merged = {}
    for stem in stems:
        regions = []
        current = None
        for t in temporal:
            if stem in t['active']:
                if current is None:
                    current = {'start': t['start_sec'], 'end': t['end_sec'],
                               'confs': [t['probs'][stem]]}
                else:
                    current['end'] = t['end_sec']
                    current['confs'].append(t['probs'][stem])
            else:
                if current is not None:
                    regions.append({
                        'start_sec': current['start'],
                        'end_sec': current['end'],
                        'avg_confidence': round(np.mean(current['confs']), 3),
                        'duration': round(current['end'] - current['start'], 2),
                    })
                    current = None
        if current is not None:
            regions.append({
                'start_sec': current['start'],
                'end_sec': current['end'],
                'avg_confidence': round(np.mean(current['confs']), 3),
                'duration': round(current['end'] - current['start'], 2),
            })
        if regions:
            merged[stem] = regions

    return {
        'path': str(latent_path),
        'duration_sec': round(T / fps, 2),
        'detected_stems': list(merged.keys()),
        'temporal': temporal,
        'merged': merged,
    }


def predict(args):
    """Run prediction on mix latents."""
    device = args.device if torch.cuda.is_available() else 'cpu'
    model_path = OUTPUT_DIR / "model.pt"
    if not model_path.exists():
        logging.error(f"No model found at {model_path}")
        return

    model_dict = load_model(model_path, device)
    logging.info(f"Loaded model: stems={model_dict['stems']}, window={model_dict['window_frames']} frames")

    if args.input:
        # Single file prediction
        result = predict_file(model_dict, Path(args.input), threshold=args.threshold)
        if result:
            print(json.dumps(result, indent=2))
        return

    # Batch prediction from manifest
    manifest_path = Path(args.manifest) if args.manifest else Path("/home/arlo/gcs-bucket/Manifests/unified_manifest.json")
    with open(manifest_path, 'rb') as f:
        manifest_data = orjson.loads(f.read())

    entries = manifest_data.get('entries', manifest_data) if isinstance(manifest_data, dict) else manifest_data
    if isinstance(entries, list):
        entries = {e.get('path', ''): e for e in entries if isinstance(e, dict)}

    # Filter to mix files (or all if not filtering)
    paths = []
    for path, entry in entries.items():
        # Derive latent path
        audio_p = Path(path)
        try:
            rel_path = audio_p.relative_to(AUDIO_ROOT)
        except ValueError:
            parts = audio_p.parts
            if 'protools' in parts:
                idx = parts.index('protools')
                rel_path = Path(*parts[idx:])
            else:
                continue
        stem = rel_path.with_suffix('')
        latent = MIX_LATENTS_ROOT / f"{stem}.dcae.pt"
        if not latent.exists():
            latent = MIX_LATENTS_ROOT / f"{stem}.pt"
        if latent.exists():
            paths.append((path, latent))

    if args.limit:
        paths = paths[:args.limit]

    logging.info(f"Predicting {len(paths)} files...")

    results = []
    for i, (audio_path, latent_path) in enumerate(paths):
        if (i + 1) % 500 == 0:
            logging.info(f"  {i+1}/{len(paths)}")
        result = predict_file(model_dict, latent_path, threshold=args.threshold)
        if result:
            result['audio_path'] = audio_path
            results.append(result)

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "predictions.json"
    output_data = {
        'model': str(model_path),
        'stems': STEMS,
        'threshold': args.threshold,
        'total': len(results),
        'predictions': results,
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f)

    logging.info(f"Saved {len(results)} predictions to {output_path}")
    logging.info("Training complete")  # trigger monitor service detection


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description='Stem Predictor: predict temporal instrument presence from mix latents')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--limit', type=int, default=0, help='Limit training pairs (0=all)')
    parser.add_argument('--input', type=str, default=None, help='Single latent file to predict')
    parser.add_argument('--manifest', type=str, default=None, help='Manifest for batch prediction')
    parser.add_argument('--output', type=str, default=None, help='Output predictions file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        datefmt='%H:%M:%S')

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)


if __name__ == "__main__":
    main()
