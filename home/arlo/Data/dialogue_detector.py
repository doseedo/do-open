#!/usr/bin/env python3
"""
Temporal dialogue detector using 1-second latent windows.

Distinguishes dialogue (speech/talking) from singing/vocals and other audio.
Trained on:
1. Single-label dialogue corrections
2. Multi-label dialogue regions (usually at beginning/end of files)
3. Hard negatives from voice group (singing - hardest to distinguish)
4. Random other groups as easy negatives

Key insight: Dialogue vs vocals is distinguished by LOCAL temporal shape,
not full-file statistics. Speech has different rhythm/prosody than singing.

Usage:
    # Train
    python3 dialogue_detector.py --mode train --voice-ratio 3.0

    # Detect dialogue in 1s windows
    python3 dialogue_detector.py --mode detect --input /path/to/latent.pt

    # Batch detect
    python3 dialogue_detector.py --mode batch --input-dir /path/to/latents/
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, precision_recall_fscore_support
import random
from datetime import datetime

# Configuration
GCS_BASE = Path("/home/arlo/gcs-bucket")
LATENTS_BASE = GCS_BASE / "Latents"
COMBINED_MANIFEST = GCS_BASE / "Manifests/combined_manifest.json"
FORMAT_MANIFEST = GCS_BASE / "Manifests/format_manifest.json"
CORRECTIONS_PATH = GCS_BASE / "Manifests/corrections.json"
OUTPUT_DIR = Path("/home/arlo/Data/dialogue_detector")

# Binary classification: dialogue vs non-dialogue
CLASSES = ['non_dialogue', 'dialogue']

# Cache for has_latent lookup
_HAS_LATENT_CACHE = None


def load_has_latent_lookup():
    """Load pre-computed has_latent lookup from format_manifest.json for O(1) checks."""
    global _HAS_LATENT_CACHE
    if _HAS_LATENT_CACHE is not None:
        return _HAS_LATENT_CACHE

    print("Loading has_latent lookup from format_manifest.json...")
    with open(FORMAT_MANIFEST) as f:
        fmt = json.load(f)

    # Build set of paths that have latents (paths are relative in format_manifest)
    _HAS_LATENT_CACHE = set()
    for entry in fmt.get('entries', []):
        if entry.get('has_latent') == True:  # Explicit True check (can be "skipped" string)
            # Convert relative path to absolute
            abs_path = str(GCS_BASE / entry['path'])
            _HAS_LATENT_CACHE.add(abs_path)

    print(f"  Loaded {len(_HAS_LATENT_CACHE)} paths with latents")
    return _HAS_LATENT_CACHE

# Latent params
LATENT_SHAPE = (8, 16)  # [8, 16, T]
FRAMES_PER_SEC = 44100 / 512  # ~86.13 fps
WINDOW_FRAMES = int(1.0 * FRAMES_PER_SEC)  # 1 second window (~86 frames)


class TemporalDialogueDataset(Dataset):
    """
    Dataset that extracts 1-second windows from latents for temporal training.

    For dialogue files: extracts windows from dialogue regions
    For non-dialogue: extracts random windows
    """

    def __init__(self, samples, windows_per_sample=5, augment=True):
        """
        samples: list of (latent_path, is_dialogue, region_info)
            region_info: None for full file, or (start_sec, end_sec) for specific region
        """
        self.samples = samples
        self.windows_per_sample = windows_per_sample
        self.augment = augment
        self.window_frames = WINDOW_FRAMES

    def __len__(self):
        return len(self.samples) * self.windows_per_sample

    def __getitem__(self, idx):
        sample_idx = idx // self.windows_per_sample
        latent_path, is_dialogue, region_info = self.samples[sample_idx]

        # Load latent
        try:
            data = torch.load(latent_path, map_location='cpu', weights_only=False)
            # Latent files are dicts with 'latents' key
            latent = data['latents'] if isinstance(data, dict) else data
        except:
            latent = torch.zeros(8, 16, self.window_frames)

        T = latent.shape[-1]

        # Determine valid range for window extraction
        if region_info is not None:
            # Extract from specific region (dialogue region)
            start_sec, end_sec = region_info
            start_frame = int(start_sec * FRAMES_PER_SEC)
            end_frame = int(end_sec * FRAMES_PER_SEC)
            start_frame = max(0, min(start_frame, T - self.window_frames))
            end_frame = min(T, end_frame)
        else:
            # Use full file
            start_frame = 0
            end_frame = T

        # Extract random window from valid range
        valid_range = end_frame - start_frame - self.window_frames
        if valid_range > 0:
            offset = random.randint(0, valid_range)
            window_start = start_frame + offset
        else:
            window_start = start_frame

        window_start = max(0, min(window_start, T - self.window_frames))
        window = latent[:, :, window_start:window_start + self.window_frames]

        # Pad if needed
        if window.shape[-1] < self.window_frames:
            pad = torch.zeros(8, 16, self.window_frames - window.shape[-1])
            window = torch.cat([window, pad], dim=-1)

        # Extract temporal features (preserves local shape)
        features = self._extract_temporal_features(window)

        # Augmentation
        if self.augment and random.random() < 0.3:
            features = features + torch.randn_like(features) * 0.01

        label = torch.tensor([1.0 if is_dialogue else 0.0])

        return features, label

    def _extract_temporal_features(self, window):
        """
        Extract features that capture temporal/local shape.

        Instead of just pooling, we preserve some temporal structure
        to capture rhythm/prosody differences between speech and singing.
        """
        # window: [8, 16, T] where T = ~86 frames (1 second)

        # 1. Global stats (mean, std, max) - 384 features
        flat = window.reshape(-1, window.shape[-1])  # [128, T]
        global_mean = flat.mean(dim=-1)
        global_std = flat.std(dim=-1)
        global_max = flat.max(dim=-1)[0]
        global_feats = torch.cat([global_mean, global_std, global_max])  # 384

        # 2. Temporal dynamics - split into 4 quarters, get delta between them
        T = window.shape[-1]
        quarter = T // 4
        quarters = [window[:, :, i*quarter:(i+1)*quarter] for i in range(4)]
        quarter_means = [q.reshape(-1, q.shape[-1]).mean(dim=-1) for q in quarters]

        # Deltas between quarters (captures rhythm/changes)
        delta1 = quarter_means[1] - quarter_means[0]  # 128
        delta2 = quarter_means[2] - quarter_means[1]  # 128
        delta3 = quarter_means[3] - quarter_means[2]  # 128
        temporal_feats = torch.cat([delta1, delta2, delta3])  # 384

        # 3. Variance of frame-to-frame changes (speech is more "choppy")
        frame_diffs = flat[:, 1:] - flat[:, :-1]  # [128, T-1]
        diff_var = frame_diffs.var(dim=-1)  # 128

        # Combine all features: 384 + 384 + 128 = 896
        return torch.cat([global_feats, temporal_feats, diff_var])


def load_dialogue_data(corrections_path: Path, latents_base: Path):
    """Load dialogue samples from corrections."""
    # Load has_latent lookup for O(1) checks
    has_latent_set = load_has_latent_lookup()

    print("Loading dialogue corrections...")
    with open(corrections_path) as f:
        corrections = json.load(f)

    dialogue_samples = []

    for path, data in corrections.items():
        # Check if latent exists using O(1) lookup
        if path not in has_latent_set:
            continue

        # Build latent path
        rel_path = path.replace(str(GCS_BASE) + '/', '')
        latent_path = latents_base / ((rel_path[:-4] if rel_path.endswith('.wav') else rel_path) + '.pt')

        regions = data.get('regions', [])
        group = data.get('group', '')

        if not regions:
            # Single-label
            if group == 'dialogue':
                dialogue_samples.append((latent_path, True, None))
        else:
            # Multi-label - find dialogue regions
            for region in regions:
                labels = region.get('labels', [])
                if 'dialogue' in labels:
                    start = region.get('start', 0)
                    end = region.get('end', 0)
                    if end - start >= 0.5:  # At least 0.5 seconds
                        dialogue_samples.append((latent_path, True, (start, end)))

    print(f"  Loaded {len(dialogue_samples)} dialogue samples")
    return dialogue_samples


def load_voice_negatives(manifest_path: Path, latents_base: Path, limit: int):
    """Load voice group samples as hard negatives."""
    # Load has_latent lookup for O(1) checks
    has_latent_set = load_has_latent_lookup()

    print("Loading voice samples (hard negatives)...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    voice_samples = []

    for path, meta in manifest.items():
        if not isinstance(meta, dict):
            continue
        if meta.get('group') != 'voice':
            continue

        # Check if latent exists using O(1) lookup
        if path not in has_latent_set:
            continue

        rel_path = path.replace(str(GCS_BASE) + '/', '')
        latent_path = latents_base / ((rel_path[:-4] if rel_path.endswith('.wav') else rel_path) + '.pt')
        voice_samples.append((latent_path, False, None))

    # Shuffle and limit
    random.shuffle(voice_samples)
    voice_samples = voice_samples[:limit]

    print(f"  Loaded {len(voice_samples)} voice samples")
    return voice_samples


def load_other_negatives(manifest_path: Path, latents_base: Path, limit: int):
    """Load random other group samples as easy negatives."""
    # Load has_latent lookup for O(1) checks
    has_latent_set = load_has_latent_lookup()

    print("Loading other samples (easy negatives)...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Exclude voice and dialogue-heavy groups
    exclude_groups = {'voice', 'undefined', 'silent'}

    other_samples = []

    for path, meta in manifest.items():
        if not isinstance(meta, dict):
            continue
        group = meta.get('group', '')
        if group in exclude_groups:
            continue

        # Check if latent exists using O(1) lookup
        if path not in has_latent_set:
            continue

        rel_path = path.replace(str(GCS_BASE) + '/', '')
        latent_path = latents_base / ((rel_path[:-4] if rel_path.endswith('.wav') else rel_path) + '.pt')
        other_samples.append((latent_path, False, None))

    # Shuffle and limit
    random.shuffle(other_samples)
    other_samples = other_samples[:limit]

    print(f"  Loaded {len(other_samples)} other samples")
    return other_samples


class DialogueDetector(nn.Module):
    """Binary classifier for dialogue detection."""

    def __init__(self, input_dim=896, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(args):
    """Train the dialogue detector."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Load dialogue samples
    dialogue_samples = load_dialogue_data(CORRECTIONS_PATH, LATENTS_BASE)
    num_dialogue = len(dialogue_samples)

    # Calculate negative sample counts
    # voice_ratio controls how many voice samples relative to dialogue
    num_voice = int(num_dialogue * args.voice_ratio)
    num_other = int(num_dialogue * args.other_ratio)

    # Load negatives
    voice_samples = load_voice_negatives(COMBINED_MANIFEST, LATENTS_BASE, num_voice)
    other_samples = load_other_negatives(COMBINED_MANIFEST, LATENTS_BASE, num_other)

    # Combine all samples
    all_samples = dialogue_samples + voice_samples + other_samples
    random.shuffle(all_samples)

    print(f"\nDataset composition:")
    print(f"  Dialogue (positive): {len(dialogue_samples)}")
    print(f"  Voice (hard negative): {len(voice_samples)}")
    print(f"  Other (easy negative): {len(other_samples)}")
    print(f"  Total: {len(all_samples)}")

    # Train/val split
    split_idx = int(0.9 * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # Create datasets with temporal windowing
    train_dataset = TemporalDialogueDataset(
        train_samples,
        windows_per_sample=args.windows_per_sample,
        augment=True
    )
    val_dataset = TemporalDialogueDataset(
        val_samples,
        windows_per_sample=3,  # Fewer for validation
        augment=False
    )

    print(f"\nTrain windows: {len(train_dataset)}")
    print(f"Val windows: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Model
    model = DialogueDetector(input_dim=896, hidden_dim=256)
    model.to(device)

    # Loss with class weighting (dialogue is minority class)
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_f1 = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.squeeze().to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.squeeze().to(device)

                logits = model(features)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )

        scheduler.step(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Loss: {train_loss/len(train_loader):.4f}/{val_loss/len(val_loader):.4f}, "
              f"P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'model_state': model.state_dict(),
                'input_dim': 896,
                'hidden_dim': 256,
                'best_f1': best_f1,
                'precision': precision,
                'recall': recall,
                'trained_at': datetime.now().isoformat()
            }, OUTPUT_DIR / "model.pt")
            print(f"  Saved best model (F1={best_f1:.3f})")

    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    print(f"Best F1: {best_f1:.3f}")
    print(classification_report(all_labels, all_preds, target_names=['non_dialogue', 'dialogue']))


def detect_dialogue(latent_path: Path, model, device, window_sec=1.0, hop_sec=0.5, threshold=0.5):
    """Detect dialogue in temporal windows."""
    data = torch.load(latent_path, map_location='cpu', weights_only=False)
    latent = data['latents'] if isinstance(data, dict) else data

    window_frames = int(window_sec * FRAMES_PER_SEC)
    hop_frames = int(hop_sec * FRAMES_PER_SEC)

    T = latent.shape[-1]
    results = []

    # Create dummy dataset to use feature extraction
    dummy_dataset = TemporalDialogueDataset([], augment=False)

    model.eval()
    with torch.no_grad():
        for start in range(0, max(1, T - window_frames + 1), hop_frames):
            end = min(start + window_frames, T)
            window = latent[:, :, start:end]

            # Pad if needed
            if window.shape[-1] < window_frames:
                pad = torch.zeros(8, 16, window_frames - window.shape[-1])
                window = torch.cat([window, pad], dim=-1)

            # Extract features
            features = dummy_dataset._extract_temporal_features(window)
            features = features.unsqueeze(0).to(device)

            # Predict
            logit = model(features)
            prob = torch.sigmoid(logit).item()

            results.append({
                'start_sec': start / FRAMES_PER_SEC,
                'end_sec': end / FRAMES_PER_SEC,
                'is_dialogue': prob > threshold,
                'probability': prob
            })

    return results


def run_detect(args):
    """Run dialogue detection on a latent file."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(OUTPUT_DIR / "model.pt", map_location='cpu', weights_only=False)
    model = DialogueDetector(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    # Run detection
    results = detect_dialogue(
        Path(args.input), model, device,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        threshold=args.threshold
    )

    # Print results
    print(f"\nDialogue detection: {args.input}")
    print("="*50)

    dialogue_regions = []
    current_start = None

    for r in results:
        if r['is_dialogue']:
            if current_start is None:
                current_start = r['start_sec']
            current_end = r['end_sec']
        else:
            if current_start is not None:
                dialogue_regions.append((current_start, current_end))
                current_start = None

    # Don't forget last region
    if current_start is not None:
        dialogue_regions.append((current_start, results[-1]['end_sec']))

    if dialogue_regions:
        print("Dialogue detected at:")
        for start, end in dialogue_regions:
            print(f"  {start:.1f}s - {end:.1f}s")

        total_dialogue = sum(end - start for start, end in dialogue_regions)
        total_duration = results[-1]['end_sec']
        print(f"\nTotal dialogue: {total_dialogue:.1f}s / {total_duration:.1f}s ({100*total_dialogue/total_duration:.1f}%)")
    else:
        print("No dialogue detected")

    # Save results
    if args.output:
        output_data = {
            'windows': results,
            'dialogue_regions': [{'start': s, 'end': e} for s, e in dialogue_regions],
            'threshold': args.threshold
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description='Temporal dialogue detector')
    parser.add_argument('--mode', choices=['train', 'detect', 'batch'], required=True)

    # Training args
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--voice-ratio', type=float, default=3.0,
                        help='Ratio of voice samples to dialogue (hard negatives)')
    parser.add_argument('--other-ratio', type=float, default=1.0,
                        help='Ratio of other samples to dialogue (easy negatives)')
    parser.add_argument('--windows-per-sample', type=int, default=10,
                        help='Number of windows to extract per sample during training')
    parser.add_argument('--pos-weight', type=float, default=2.0,
                        help='Positive class weight for imbalanced data')

    # Inference args
    parser.add_argument('--input', type=str, help='Input latent file or directory')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--window-sec', type=float, default=1.0)
    parser.add_argument('--hop-sec', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'detect':
        if not args.input:
            print("Error: --input required for detect mode")
            return
        run_detect(args)
    elif args.mode == 'batch':
        print("Batch mode not yet implemented")


if __name__ == "__main__":
    main()
