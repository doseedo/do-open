#!/usr/bin/env python3
"""
Latent Space Instrument Classifier

Uses pre-computed ACE-Step latents for fast, accurate instrument classification.
Much faster than audio-based classification since latents are already extracted.

Usage:
  # Train classifier from manifest (recommended - uses paths with latents)
  python latent_instrument_classifier.py --mode train \
    --manifest /home/arlo/gcs-bucket/Manifests/combined_manifest.json \
    --output-dir /home/arlo/Data/latent_classifier

  # Classify undefined files
  python latent_instrument_classifier.py --mode classify \
    --model /home/arlo/Data/latent_classifier/model.pt \
    --manifest /home/arlo/gcs-bucket/Manifests/combined_manifest.json \
    --output-dir /home/arlo/Data/latent_classifier

  # Validate labels & flag potential mislabels
  python latent_instrument_classifier.py --mode validate \
    --model /home/arlo/Data/latent_classifier/model.pt \
    --manifest /home/arlo/gcs-bucket/Manifests/combined_manifest.json \
    --output-dir /home/arlo/Data/latent_classifier

  # Full pipeline (train + classify undefined)
  python latent_instrument_classifier.py --mode full \
    --manifest /home/arlo/gcs-bucket/Manifests/combined_manifest.json \
    --undefined /home/arlo/undefined_audio_paths.txt \
    --output-dir /home/arlo/Data/latent_classifier

  # Temporal analysis - detect instrument changes over time
  python latent_instrument_classifier.py --mode temporal \
    --model /home/arlo/Data/latent_classifier/model.pt \
    --manifest /home/arlo/gcs-bucket/Manifests/unified_manifest.json \
    --output-dir /home/arlo/Data/latent_classifier \
    --window-sec 2.0 --hop-sec 1.0

  # Temporal analysis on specific group
  python latent_instrument_classifier.py --mode temporal \
    --model /home/arlo/Data/latent_classifier/model.pt \
    --manifest /home/arlo/gcs-bucket/Manifests/unified_manifest.json \
    --group ensemble \
    --output-dir /home/arlo/Data/latent_classifier

Validation flags entries as:
  - disagreement: classifier confident in different class than label
  - uncertain: high entropy (possible ensemble/ambiguous sound)
  - outlier: feature vector far from class centroid

Temporal analysis:
  - Splits audio into overlapping time windows
  - Classifies each window independently
  - Detects if instruments change over time (temporal ensemble)
  - Outputs segments with timestamps and confidence
"""

import argparse
import json
import pickle
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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ===================== CONFIGURATION =====================

LATENTS_ROOT = Path("/home/arlo/gcs-bucket/Latents")
AUDIO_ROOT = Path("/home/arlo/gcs-bucket")
FORMAT_MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/format_manifest.json")

# Confidence thresholds
CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.65

# Training settings
MAX_SAMPLES_PER_CLASS = 15000  # Use more data, class weights handle imbalance
MIN_SAMPLES_PER_CLASS = 100
# Filter these out from training (meta-groups, not instruments)
# Note: 'dialogue' is a valid class for speech detection
EXCLUDED_CLASSES = {'undefined', 'room', 'fx', 'click', 'silent', 'junk', 'review_vocals', 'ensemble', 'full-track'}

# Silent frame detection threshold (frames with energy below this are masked)
SILENT_FRAME_THRESHOLD = 0.01  # RMS threshold for considering a frame silent
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
HIDDEN_DIM = 256

# Feature extraction - pooling strategies
POOL_METHODS = ['mean', 'std', 'max']  # Results in 8*16*3 = 384 features


# ===================== LATENT EXISTENCE LOOKUP =====================

# Cached has_latent lookup from format_manifest.json
_has_latent_cache: Optional[Dict[str, bool]] = None

def load_has_latent_lookup() -> Dict[str, bool]:
    """Load has_latent data from format_manifest.json.

    Returns dict mapping audio path -> bool (has latent).
    Cached after first load.
    """
    global _has_latent_cache
    if _has_latent_cache is not None:
        return _has_latent_cache

    if not FORMAT_MANIFEST_PATH.exists():
        logging.warning(f"Format manifest not found: {FORMAT_MANIFEST_PATH}")
        _has_latent_cache = {}
        return _has_latent_cache

    logging.info(f"Loading has_latent lookup from {FORMAT_MANIFEST_PATH}...")
    with open(FORMAT_MANIFEST_PATH) as f:
        format_manifest = json.load(f)

    _has_latent_cache = {}
    entries = format_manifest.get('entries', format_manifest) if isinstance(format_manifest, dict) else format_manifest
    for entry in entries:
        if isinstance(entry, dict) and 'path' in entry:
            _has_latent_cache[entry['path']] = entry.get('has_latent', False)

    logging.info(f"  Loaded {len(_has_latent_cache)} entries, {sum(1 for v in _has_latent_cache.values() if v)} have latents")
    return _has_latent_cache


# ===================== PATH CONVERSION =====================

def audio_path_to_latent_path(audio_path: str) -> Optional[Path]:
    """Convert audio file path to corresponding latent path.

    Checks for both .dcae.pt (new) and .pt (old) extensions.
    Returns None if no latent file exists.
    """
    audio_path = Path(audio_path)

    # Find the relative path from gcs-bucket
    try:
        rel_path = audio_path.relative_to(AUDIO_ROOT)
    except ValueError:
        # If not relative to AUDIO_ROOT, try to find 'protools' in path
        parts = audio_path.parts
        if 'protools' in parts:
            idx = parts.index('protools')
            rel_path = Path(*parts[idx:])
        elif 'protoolsA' in parts:
            idx = parts.index('protoolsA')
            rel_path = Path(*parts[idx:])
        else:
            # Just use the full path after gcs-bucket if present
            if 'gcs-bucket' in parts:
                idx = parts.index('gcs-bucket')
                rel_path = Path(*parts[idx+1:])
            else:
                rel_path = audio_path

    # Check for both extensions - prefer .dcae.pt (newer format)
    stem = rel_path.with_suffix('')
    dcae_path = LATENTS_ROOT / f"{stem}.dcae.pt"
    pt_path = LATENTS_ROOT / f"{stem}.pt"

    if dcae_path.exists():
        return dcae_path
    elif pt_path.exists():
        return pt_path
    else:
        return None


def load_latent(latent_path: Path) -> Optional[torch.Tensor]:
    """Load latent tensor from file."""
    try:
        data = torch.load(latent_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            return data['latents']  # [8, 16, T]
        return data
    except Exception as e:
        return None


def detect_silent_frames(latent: torch.Tensor, threshold: float = SILENT_FRAME_THRESHOLD) -> torch.Tensor:
    """Detect silent frames in latent based on energy.

    Args:
        latent: [8, 16, T] tensor
        threshold: RMS threshold for considering a frame silent

    Returns:
        Boolean mask [T] where True = non-silent frame
    """
    # Compute energy per frame (RMS across all channels)
    # latent shape: [8, 16, T]
    energy = torch.sqrt((latent ** 2).mean(dim=(0, 1)))  # [T]
    return energy > threshold


def pool_latent(latent: torch.Tensor, mask_silent: bool = True) -> torch.Tensor:
    """Pool latent [8, 16, T] to fixed-size feature vector.

    Args:
        latent: [8, 16, T] tensor
        mask_silent: If True, exclude silent frames from pooling
    """
    # latent shape: [8, 16, T] - 8 codebook groups, 16 channels, T time steps

    # Mask silent frames if requested
    if mask_silent and latent.shape[-1] > 1:
        non_silent_mask = detect_silent_frames(latent)
        if non_silent_mask.sum() > 0:
            # Only keep non-silent frames
            latent = latent[:, :, non_silent_mask]
        # If all frames are silent, use original (will classify as silent/undefined)

    features = []

    if 'mean' in POOL_METHODS:
        features.append(latent.mean(dim=-1))  # [8, 16]
    if 'std' in POOL_METHODS:
        features.append(latent.std(dim=-1))   # [8, 16]
    if 'max' in POOL_METHODS:
        features.append(latent.max(dim=-1)[0])  # [8, 16]
    if 'min' in POOL_METHODS:
        features.append(latent.min(dim=-1)[0])  # [8, 16]

    # Concatenate and flatten: [8, 16, num_pools] -> [8*16*num_pools]
    stacked = torch.stack(features, dim=-1)  # [8, 16, num_pools]
    return stacked.flatten()


# ===================== TEMPORAL WINDOWING =====================

# ACE-Step latent frame rate: 44100 / 512 ≈ 86.13 frames/sec
LATENT_FRAMES_PER_SEC = 44100 / 512

def window_latent(latent: torch.Tensor, window_sec: float = 2.0,
                  hop_sec: float = 1.0) -> List[Tuple[float, float, torch.Tensor]]:
    """Split latent into overlapping time windows.

    Args:
        latent: [8, 16, T] tensor
        window_sec: window size in seconds
        hop_sec: hop size in seconds

    Returns:
        List of (start_sec, end_sec, window_latent) tuples
    """
    T = latent.shape[-1]
    window_frames = int(window_sec * LATENT_FRAMES_PER_SEC)
    hop_frames = int(hop_sec * LATENT_FRAMES_PER_SEC)

    # Ensure minimum window size
    window_frames = max(window_frames, 32)
    hop_frames = max(hop_frames, 16)

    windows = []
    for start in range(0, T - window_frames + 1, hop_frames):
        end = start + window_frames
        window = latent[:, :, start:end]

        start_sec = start / LATENT_FRAMES_PER_SEC
        end_sec = end / LATENT_FRAMES_PER_SEC
        windows.append((start_sec, end_sec, window))

    # Handle short files - use whole latent as single window
    if len(windows) == 0 and T > 0:
        windows.append((0, T / LATENT_FRAMES_PER_SEC, latent))

    return windows


def classify_temporal(
    latent: torch.Tensor,
    model: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    classes: List[str],
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
    device: str = 'cpu'
) -> List[Dict]:
    """Classify latent in time windows.

    Args:
        latent: [8, 16, T] tensor
        model: trained classifier
        mean, std: normalization params
        classes: class names
        window_sec: window size in seconds
        hop_sec: hop size in seconds

    Returns:
        List of dicts with start, end, predicted_class, confidence, all_probs
    """
    windows = window_latent(latent, window_sec, hop_sec)

    if len(windows) == 0:
        return []

    # Pool all windows
    features = torch.stack([pool_latent(w[2]) for w in windows])

    # Normalize
    features = (features - mean) / std

    # Predict
    model.eval()
    with torch.no_grad():
        features = features.to(device)
        logits = model(features)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    results = []
    for i, (start_sec, end_sec, _) in enumerate(windows):
        pred_idx = probs[i].argmax()
        results.append({
            'start': round(start_sec, 2),
            'end': round(end_sec, 2),
            'predicted_class': classes[pred_idx],
            'confidence': float(probs[i][pred_idx]),
            'all_probs': {c: float(p) for c, p in zip(classes, probs[i])}
        })

    return results


def detect_temporal_changes(
    temporal_results: List[Dict],
    min_confidence: float = 0.6,
    min_duration_sec: float = 1.0
) -> Dict:
    """Analyze temporal classification results to detect instrument changes.

    Args:
        temporal_results: output from classify_temporal
        min_confidence: minimum confidence to count a prediction
        min_duration_sec: minimum duration for an instrument to be counted

    Returns:
        Dict with analysis results
    """
    if not temporal_results:
        return {'is_temporal_ensemble': False, 'reason': 'no_results'}

    # Filter by confidence
    confident = [r for r in temporal_results if r['confidence'] >= min_confidence]

    if not confident:
        return {'is_temporal_ensemble': None, 'reason': 'low_confidence'}

    # Group consecutive predictions by class
    segments = []
    current_class = None
    current_start = None
    current_end = None
    current_confidences = []

    for r in confident:
        if r['predicted_class'] != current_class:
            # Save previous segment
            if current_class is not None:
                duration = current_end - current_start
                if duration >= min_duration_sec:
                    segments.append({
                        'class': current_class,
                        'start': current_start,
                        'end': current_end,
                        'duration': round(duration, 2),
                        'avg_confidence': round(np.mean(current_confidences), 3)
                    })
            # Start new segment
            current_class = r['predicted_class']
            current_start = r['start']
            current_confidences = [r['confidence']]

        current_end = r['end']
        current_confidences.append(r['confidence'])

    # Don't forget last segment
    if current_class is not None:
        duration = current_end - current_start
        if duration >= min_duration_sec:
            segments.append({
                'class': current_class,
                'start': current_start,
                'end': current_end,
                'duration': round(duration, 2),
                'avg_confidence': round(np.mean(current_confidences), 3)
            })

    # Analyze segments
    unique_classes = list(set(s['class'] for s in segments))
    total_duration = temporal_results[-1]['end'] if temporal_results else 0

    # Calculate time per instrument
    class_durations = {}
    for s in segments:
        cls = s['class']
        class_durations[cls] = class_durations.get(cls, 0) + s['duration']

    # Determine if temporal ensemble
    is_temporal_ensemble = len(unique_classes) > 1

    result = {
        'is_temporal_ensemble': is_temporal_ensemble,
        'total_duration': round(total_duration, 2),
        'num_segments': len(segments),
        'unique_instruments': unique_classes,
        'num_instruments': len(unique_classes),
        'segments': segments,
        'instrument_durations': class_durations,
    }

    # Add dominant instrument
    if class_durations:
        dominant = max(class_durations.items(), key=lambda x: x[1])
        result['dominant_instrument'] = dominant[0]
        result['dominant_duration'] = round(dominant[1], 2)
        result['dominant_ratio'] = round(dominant[1] / total_duration, 2) if total_duration > 0 else 0

    return result


def segments_to_ui_regions(segments: List[Dict]) -> List[Dict]:
    """Convert temporal segments to UI-compatible regions format.

    Args:
        segments: List of {'class': str, 'start': float, 'end': float, ...}

    Returns:
        List of {'labels': [str], 'start': float, 'end': float} for UI display
    """
    regions = []
    for seg in segments:
        regions.append({
            'labels': [seg['class']],  # UI expects array of labels
            'start': seg['start'],
            'end': seg['end'],
            'confidence': seg.get('avg_confidence', 0)
        })
    return regions


def run_temporal_classification(
    model_path: Path,
    audio_paths: List[str],
    output_path: Path,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
    min_confidence: float = 0.6,
    num_workers: int = 8,
    device: str = 'cuda'
) -> Dict:
    """Run temporal classification on a list of audio files.

    Detects if instruments change over time within each file.
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device_obj = torch.device(device)

    # Load model
    logging.info(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)

    input_dim = model_data['input_dim']
    num_classes = model_data['num_classes']
    hidden_dim = model_data.get('hidden_dim', HIDDEN_DIM)
    mean = model_data['mean']
    std = model_data['std']
    classes = model_data['label_encoder_classes']

    model = InstrumentClassifier(input_dim, num_classes, hidden_dim)
    model.load_state_dict(model_data['model_state'])
    model.to(device_obj)
    model.eval()

    logging.info(f"Processing {len(audio_paths)} files with temporal analysis...")
    logging.info(f"  Window: {window_sec}s, Hop: {hop_sec}s, Min confidence: {min_confidence}")

    results = []
    temporal_ensembles = []
    single_instrument = []
    failed = []

    for i, audio_path in enumerate(audio_paths):
        if (i + 1) % 100 == 0:
            logging.info(f"  Processed {i+1}/{len(audio_paths)}...")

        # Load latent
        latent_path = audio_path_to_latent_path(audio_path)
        if latent_path is None:
            failed.append({'path': audio_path, 'reason': 'no_latent'})
            continue

        latent = load_latent(latent_path)
        if latent is None:
            failed.append({'path': audio_path, 'reason': 'load_failed'})
            continue

        # Temporal classification
        temporal = classify_temporal(
            latent, model, mean, std, classes,
            window_sec=window_sec, hop_sec=hop_sec, device=device
        )

        # Detect changes
        analysis = detect_temporal_changes(temporal, min_confidence=min_confidence)

        # Convert segments to UI regions format
        ui_regions = segments_to_ui_regions(analysis.get('segments', []))

        result = {
            'path': audio_path,
            'filename': Path(audio_path).name,
            'temporal_windows': temporal,
            'analysis': analysis,
            'regions': ui_regions,  # UI-compatible format
            'predicted_labels': analysis.get('unique_instruments', []),
            'is_temporal_ensemble': analysis.get('is_temporal_ensemble', False),
            'dominant_instrument': analysis.get('dominant_instrument'),
        }
        results.append(result)

        if analysis.get('is_temporal_ensemble'):
            temporal_ensembles.append(result)
        elif analysis.get('is_temporal_ensemble') == False:
            single_instrument.append(result)

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("TEMPORAL CLASSIFICATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"Total processed: {len(results)}")
    logging.info(f"  Single instrument: {len(single_instrument)}")
    logging.info(f"  Temporal ensemble (instrument changes): {len(temporal_ensembles)}")
    logging.info(f"  Failed: {len(failed)}")

    if temporal_ensembles:
        logging.info("\nTemporal ensembles found:")
        for r in temporal_ensembles[:20]:
            instruments = r['analysis']['unique_instruments']
            logging.info(f"  {r['filename']}: {' -> '.join(instruments)}")

    # Save results
    output_data = {
        'settings': {
            'window_sec': window_sec,
            'hop_sec': hop_sec,
            'min_confidence': min_confidence,
        },
        'summary': {
            'total': len(results),
            'single_instrument': len(single_instrument),
            'temporal_ensemble': len(temporal_ensembles),
            'failed': len(failed),
        },
        'temporal_ensembles': temporal_ensembles,
        'single_instrument': single_instrument,
        'failed': failed,
        'all_results': results,
        'analyzed_at': datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"\nResults saved to {output_path}")

    return output_data


# ===================== DATASET =====================

class LatentDataset(Dataset):
    """Dataset for loading latent features."""

    def __init__(self, audio_paths: List[str], labels: List[int],
                 cache: Optional[Dict] = None):
        self.audio_paths = audio_paths
        self.labels = labels
        self.cache = cache if cache is not None else {}

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        # Check cache first
        if audio_path in self.cache:
            features = self.cache[audio_path]
        else:
            latent_path = audio_path_to_latent_path(audio_path)

            if latent_path is None:
                # Return zeros if latent not found
                features = torch.zeros(8 * 16 * len(POOL_METHODS))
            else:
                latent = load_latent(latent_path)
                if latent is None:
                    features = torch.zeros(8 * 16 * len(POOL_METHODS))
                else:
                    features = pool_latent(latent)

            self.cache[audio_path] = features

        return features, label


# ===================== MODEL =====================

class InstrumentClassifier(nn.Module):
    """MLP classifier for instrument classification."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = HIDDEN_DIM):
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

            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ===================== FEATURE EXTRACTION =====================

def extract_features_batch(audio_paths: List[str],
                          batch_size: int = 100,
                          num_workers: int = 8) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Extract pooled features from latents in parallel."""

    features_list = []
    valid_paths = []
    failed_paths = []

    total = len(audio_paths)
    logging.info(f"Extracting features from {total} files...")
    start_time = datetime.now()

    def process_one(audio_path: str) -> Tuple[Optional[torch.Tensor], str]:
        latent_path = audio_path_to_latent_path(audio_path)
        if latent_path is None:
            return None, audio_path
        latent = load_latent(latent_path)
        if latent is not None:
            return pool_latent(latent), audio_path
        return None, audio_path

    # Process in parallel using threads (I/O bound)
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
            if processed % 500 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = processed / elapsed if elapsed > 0 else 0
                logging.info(f"  {processed}/{total} ({rate:.1f}/s)")

    elapsed = (datetime.now() - start_time).total_seconds()
    logging.info(f"  Complete: {len(valid_paths)} success, {len(failed_paths)} failed ({elapsed:.1f}s)")

    if len(features_list) == 0:
        return torch.tensor([]), [], failed_paths

    return torch.stack(features_list), valid_paths, failed_paths


# ===================== TRAINING =====================

def train_classifier(manifest_path: Path, output_dir: Path,
                    corrections_path: Path = None, num_workers: int = 8, device: str = 'cuda',
                    exclude_file: Path = None) -> Dict:
    """Train the latent-space instrument classifier using manifest labels.

    Args:
        manifest_path: Path to unified manifest JSON
        output_dir: Output directory for model and stats
        corrections_path: Optional path to corrections JSON (overrides manifest labels)
        num_workers: Number of parallel workers for feature extraction
        device: 'cuda' or 'cpu'
        exclude_file: Optional file with paths to exclude from training (one per line)
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load exclude paths if provided
    exclude_paths = set()
    if exclude_file and Path(exclude_file).exists():
        logging.info(f"Loading exclude paths from {exclude_file}...")
        with open(exclude_file) as f:
            exclude_paths = set(line.strip() for line in f if line.strip())
        logging.info(f"  Will exclude {len(exclude_paths)} paths from training")

    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    device = torch.device(device)
    logging.info(f"Using device: {device}")

    # Load manifest
    logging.info(f"Loading manifest from {manifest_path}...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load corrections if provided
    corrections = {}
    if corrections_path and corrections_path.exists():
        logging.info(f"Loading corrections from {corrections_path}...")
        with open(corrections_path) as f:
            corrections = json.load(f)
        logging.info(f"  Loaded {len(corrections)} corrections")

    # Load ensemble detector results to filter out mix files
    # PIPELINE: Group classifier is for ISOLATED files only (Phase 2a)
    ensemble_mix_paths = set()
    ensemble_file = Path("/home/arlo/Data/ensemble_detector/ensemble_detections.json")
    if ensemble_file.exists():
        logging.info(f"Loading ensemble detector results to exclude mix files...")
        try:
            with open(ensemble_file) as f:
                ensemble_data = json.load(f)
            for entry in ensemble_data.get("detected", []):
                ensemble_mix_paths.add(entry.get("path", ""))
            logging.info(f"  Will exclude {len(ensemble_mix_paths)} detected mix files from training")
        except Exception as e:
            logging.warning(f"  Could not load ensemble detector: {e}")

    # Support both formats:
    # - combined_manifest: dict keyed by path, value has 'group'
    # - unified_manifest: has 'entries' list with 'audio_path', 'group', 'latent_path'
    is_unified = 'entries' in manifest and isinstance(manifest.get('entries'), list)

    # Group by label (using 'group' field from manifest)
    # Only include paths that actually have latents (verify existence)
    file_label_pairs = []
    class_counts = Counter()
    skipped_undefined = 0
    skipped_no_latent = 0

    if is_unified:
        logging.info("Detected unified manifest format")
        entries = manifest['entries']
        total_checked = 0
        for entry in entries:
            total_checked += 1
            if total_checked % 50000 == 0:
                logging.info(f"  Checked {total_checked}/{len(entries)}...")

            audio_path = entry.get('audio_path', '')
            label = entry.get('group', 'undefined')
            has_latent = entry.get('has_latent', False)
            latent_path_str = entry.get('latent_path')

            # Apply correction if exists (single-label only for this classifier)
            if audio_path in corrections:
                corr = corrections[audio_path]
                # Use corrected group if not multi-label
                if not corr.get('multi_label') and corr.get('group'):
                    label = corr['group']

            if not label or label in EXCLUDED_CLASSES:
                skipped_undefined += 1
                continue

            # Skip mix files (likely multi-instrument)
            # Check both filename patterns AND ensemble detector results
            fname_lower = audio_path.lower()
            if 'mix' in fname_lower or '/room' in fname_lower or '_room' in fname_lower:
                skipped_undefined += 1
                continue

            # PIPELINE: Skip files detected as mix by ensemble detector (Phase 1)
            if audio_path in ensemble_mix_paths:
                skipped_undefined += 1
                continue

            if not has_latent or not latent_path_str:
                skipped_no_latent += 1
                continue

            # Skip excluded paths
            if audio_path in exclude_paths:
                continue

            file_label_pairs.append((audio_path, label))
            class_counts[label] += 1
    else:
        logging.info("Detected combined manifest format (dict keyed by path)")
        # Load has_latent lookup from format_manifest.json
        has_latent_lookup = load_has_latent_lookup()
        logging.info("Processing manifest entries...")
        total_checked = 0
        for audio_path, meta in manifest.items():
            total_checked += 1
            if total_checked % 50000 == 0:
                logging.info(f"  Checked {total_checked}/{len(manifest)}...")

            # Get group label
            if isinstance(meta, dict):
                label = meta.get('group', 'undefined')
            else:
                label = 'undefined'

            # Apply correction if exists
            if audio_path in corrections:
                corr = corrections[audio_path]
                if not corr.get('multi_label') and corr.get('group'):
                    label = corr['group']

            if not label or label in EXCLUDED_CLASSES:
                skipped_undefined += 1
                continue

            # Skip mix files (likely multi-instrument)
            fname_lower = audio_path.lower()
            if 'mix' in fname_lower or '/room' in fname_lower or '_room' in fname_lower:
                skipped_undefined += 1
                continue

            # PIPELINE: Skip files detected as mix by ensemble detector (Phase 1)
            if audio_path in ensemble_mix_paths:
                skipped_undefined += 1
                continue

            # Check for latent - first from meta, then from lookup (try relative path)
            has_latent = meta.get('has_latent', False) if isinstance(meta, dict) else False
            if not has_latent:
                rel_path = audio_path.replace('/home/arlo/gcs-bucket/', '')
                has_latent = has_latent_lookup.get(rel_path, False)
            if not has_latent:
                skipped_no_latent += 1
                continue

            # Skip excluded paths
            if audio_path in exclude_paths:
                continue

            file_label_pairs.append((audio_path, label))
            class_counts[label] += 1

    logging.info(f"Loaded {len(file_label_pairs)} labeled files with verified latents")
    logging.info(f"  Skipped {skipped_undefined} undefined, {skipped_no_latent} without latents")

    # Apply class limits
    import random
    random.seed(42)

    filtered_pairs = []
    class_samples = defaultdict(list)

    for path, label in file_label_pairs:
        class_samples[label].append(path)

    for label, paths in class_samples.items():
        if len(paths) < MIN_SAMPLES_PER_CLASS:
            logging.warning(f"  Skipping {label}: only {len(paths)} samples")
            continue

        if len(paths) > MAX_SAMPLES_PER_CLASS:
            paths = random.sample(paths, MAX_SAMPLES_PER_CLASS)

        for path in paths:
            filtered_pairs.append((path, label))

    file_label_pairs = filtered_pairs
    class_counts = Counter(label for _, label in file_label_pairs)

    logging.info(f"Loaded {len(file_label_pairs)} files across {len(class_counts)} classes:")
    for label, count in class_counts.most_common():
        logging.info(f"  {label}: {count}")

    # Extract features
    audio_paths = [p for p, _ in file_label_pairs]
    labels = [l for _, l in file_label_pairs]

    X, valid_paths, failed = extract_features_batch(audio_paths, num_workers=num_workers)

    if len(X) < 100:
        raise ValueError(f"Not enough samples: {len(X)}")

    # Filter labels to match valid paths
    path_to_label = dict(file_label_pairs)
    valid_labels = [path_to_label[p] for p in valid_paths]

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(valid_labels)
    y = torch.tensor(y, dtype=torch.long)

    num_classes = len(label_encoder.classes_)
    input_dim = X.shape[1]

    logging.info(f"Feature dim: {input_dim}, Classes: {num_classes}")

    # Train/val/test split
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.15,
                                            stratify=y.numpy(), random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1,
                                           stratify=y[train_idx].numpy(), random_state=42)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Normalize features
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Compute class weights for imbalanced data
    class_counts_arr = np.bincount(y_train.numpy())
    class_weights = 1.0 / (class_counts_arr + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Initialize model
    model = InstrumentClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    logging.info(f"\nTraining for {NUM_EPOCHS} epochs...")
    best_val_acc = 0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            train_total += len(batch_y)

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                val_total += len(batch_y)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"  Epoch {epoch+1}/{NUM_EPOCHS}: "
                        f"train_loss={train_loss/train_total:.4f}, "
                        f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluate on test set
    model.eval()
    X_test_dev = X_test.to(device)

    with torch.no_grad():
        logits = model(X_test_dev)
        probs = F.softmax(logits, dim=1)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_proba = probs.cpu().numpy()

    y_test_np = y_test.numpy()

    logging.info("\n" + "=" * 60)
    logging.info("CLASSIFICATION REPORT")
    logging.info("=" * 60)
    report = classification_report(y_test_np, y_pred,
                                   target_names=label_encoder.classes_,
                                   output_dict=True)
    logging.info("\n" + classification_report(y_test_np, y_pred,
                                              target_names=label_encoder.classes_))

    # High-confidence accuracy
    max_proba = np.max(y_proba, axis=1)
    for threshold in [0.9, 0.8, 0.7]:
        mask = max_proba >= threshold
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == y_test_np[mask])
            logging.info(f"Accuracy at {threshold:.0%} confidence: {acc:.1%} ({np.sum(mask)} samples)")

    # Save model
    model_data = {
        'model_state': best_model_state,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'hidden_dim': HIDDEN_DIM,
        'mean': mean,
        'std': std,
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'pool_methods': POOL_METHODS,
        'training_stats': {
            'total_samples': len(X),
            'failed_samples': len(failed),
            'test_accuracy': report['accuracy'],
            'best_val_accuracy': best_val_acc,
            'class_counts': dict(Counter(valid_labels))
        },
        'trained_at': datetime.now().isoformat()
    }

    model_path = output_dir / 'model.pt'
    torch.save(model_data, model_path)
    logging.info(f"\nModel saved to {model_path}")

    # Also save label encoder separately for compatibility
    with open(output_dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    return model_data


# ===================== CLASSIFICATION =====================

def classify_undefined(model_path: Path, undefined_paths: List[str],
                       output_path: Path, num_workers: int = 8,
                       device: str = 'cuda',
                       manifest_path: Path = None) -> Dict:
    """Classify undefined files using trained model.

    If manifest_path is provided, extract undefined paths from manifest directly.
    """

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    # If manifest provided, extract undefined paths with latents
    if manifest_path and manifest_path.exists():
        logging.info(f"Loading undefined paths from manifest: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Support both formats
        is_unified = 'entries' in manifest and isinstance(manifest.get('entries'), list)

        undefined_paths = []
        skipped_no_latent = 0

        if is_unified:
            logging.info("Detected unified manifest format")
            for entry in manifest['entries']:
                label = entry.get('group', 'undefined')
                if label != 'undefined':
                    continue
                if not entry.get('has_latent') or not entry.get('latent_path'):
                    skipped_no_latent += 1
                    continue
                undefined_paths.append(entry['audio_path'])
        else:
            # Load has_latent lookup from format_manifest.json
            has_latent_lookup = load_has_latent_lookup()
            for path, meta in manifest.items():
                if not isinstance(meta, dict):
                    continue
                if meta.get('group') != 'undefined':
                    continue
                # Check latent exists using lookup
                if has_latent_lookup.get(path, False):
                    undefined_paths.append(path)
                else:
                    skipped_no_latent += 1

        logging.info(f"Found {len(undefined_paths)} undefined paths with latents")
        logging.info(f"Skipped {skipped_no_latent} without latents")

    # Load model
    logging.info(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)

    input_dim = model_data['input_dim']
    num_classes = model_data['num_classes']
    hidden_dim = model_data.get('hidden_dim', HIDDEN_DIM)
    mean = model_data['mean']
    std = model_data['std']
    classes = model_data['label_encoder_classes']

    model = InstrumentClassifier(input_dim, num_classes, hidden_dim)
    model.load_state_dict(model_data['model_state'])
    model.to(device)
    model.eval()

    logging.info(f"Model classes: {classes}")

    if len(undefined_paths) == 0:
        logging.error("No undefined paths to classify!")
        return {'predictions': [], 'failed_paths': []}

    # Extract features
    X, valid_paths, failed = extract_features_batch(undefined_paths, num_workers=num_workers)

    if len(X) == 0:
        logging.error("No features extracted!")
        return {}

    # Normalize
    X = (X - mean) / std

    # Predict in batches
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

    # Decode predictions
    y_pred_labels = [classes[i] for i in y_pred]

    # Detect silent files by checking if pooled feature energy is very low
    # X is already normalized, so check raw features before normalization
    X_raw = X * std + mean  # Denormalize to get raw pooled features
    feature_energy = (X_raw ** 2).mean(dim=1).numpy()  # RMS energy per file
    SILENT_ENERGY_THRESHOLD = 0.001  # Very low energy = silent file

    # Categorize by confidence
    predictions = {
        'high_confidence': [],
        'medium_confidence': [],
        'low_confidence': [],
        'silent': []  # New category for detected silent files
    }

    results = []
    silent_count = 0
    for i, (path, label, conf) in enumerate(zip(valid_paths, y_pred_labels, max_proba)):
        # Check if file is silent based on feature energy
        is_silent = feature_energy[i] < SILENT_ENERGY_THRESHOLD

        if is_silent:
            result = {
                'path': path,
                'predicted_group': 'silent',
                'confidence': 1.0,  # High confidence it's silent
                'all_probabilities': {'silent': 1.0},
                'detected_silent': True
            }
            predictions['silent'].append(result)
            silent_count += 1
        else:
            result = {
                'path': path,
                'predicted_group': label,
                'confidence': float(conf),
                'all_probabilities': {c: float(p) for c, p in zip(classes, all_probs[i])}
            }
            if conf >= CONFIDENCE_HIGH:
                predictions['high_confidence'].append(result)
            elif conf >= CONFIDENCE_MEDIUM:
                predictions['medium_confidence'].append(result)
            else:
                predictions['low_confidence'].append(result)

        results.append(result)

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("CLASSIFICATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"Total classified: {len(results)}")
    logging.info(f"  High confidence (>={CONFIDENCE_HIGH:.0%}): {len(predictions['high_confidence'])}")
    logging.info(f"  Medium confidence ({CONFIDENCE_MEDIUM:.0%}-{CONFIDENCE_HIGH:.0%}): {len(predictions['medium_confidence'])}")
    logging.info(f"  Low confidence (<{CONFIDENCE_MEDIUM:.0%}): {len(predictions['low_confidence'])}")
    logging.info(f"  Detected silent: {silent_count}")
    logging.info(f"  Failed to load latent: {len(failed)}")

    # Predictions by class
    high_by_class = Counter(r['predicted_group'] for r in predictions['high_confidence'])
    logging.info("\nHigh-confidence predictions by class:")
    for cls, count in high_by_class.most_common():
        logging.info(f"  {cls}: {count}")

    # Save results
    output_data = {
        'total': len(results),  # Put total at top for fast header reading
        'predictions': results,
        'failed_paths': failed,
        'summary': {
            'total': len(results),
            'high_confidence': len(predictions['high_confidence']),
            'medium_confidence': len(predictions['medium_confidence']),
            'low_confidence': len(predictions['low_confidence']),
            'silent': silent_count,
            'failed': len(failed),
            'by_class': dict(Counter(r['predicted_group'] for r in results))
        },
        'thresholds': {
            'high': CONFIDENCE_HIGH,
            'medium': CONFIDENCE_MEDIUM
        },
        'classified_at': datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"\nPredictions saved to {output_path}")

    return output_data


# ===================== VALIDATION / MISLABEL DETECTION =====================

def validate_labels(model_path: Path, manifest_path: Path, output_dir: Path,
                   num_workers: int = 8, device: str = 'cuda',
                   disagreement_threshold: float = 0.7,
                   entropy_threshold: float = 1.5,
                   outlier_percentile: float = 95) -> Dict:
    """
    Validate existing labels and flag potential mislabels.

    Flags entries based on:
    1. High-confidence disagreement: classifier confident in different class
    2. High entropy: classifier uncertain (possible ensemble/ambiguous)
    3. Feature outlier: entry far from its class centroid

    Args:
        model_path: Path to trained model
        manifest_path: Manifest with labeled entries
        output_dir: Where to save validation results
        disagreement_threshold: Min confidence to flag disagreement (default 0.7)
        entropy_threshold: Max entropy before flagging uncertainty (default 1.5)
        outlier_percentile: Percentile for outlier detection (default 95)
    """
    from scipy.spatial.distance import cdist

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device_obj = torch.device(device)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logging.info(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)

    input_dim = model_data['input_dim']
    num_classes = model_data['num_classes']
    hidden_dim = model_data.get('hidden_dim', HIDDEN_DIM)
    mean = model_data['mean']
    std = model_data['std']
    classes = model_data['label_encoder_classes']

    model = InstrumentClassifier(input_dim, num_classes, hidden_dim)
    model.load_state_dict(model_data['model_state'])
    model.to(device_obj)
    model.eval()

    # Load manifest
    logging.info(f"Loading manifest from {manifest_path}...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Support both formats
    is_unified = 'entries' in manifest and isinstance(manifest.get('entries'), list)

    # Get labeled entries (excluding undefined and room)
    labeled_entries = []

    if is_unified:
        logging.info("Detected unified manifest format")
        for entry in manifest['entries']:
            label = entry.get('group', 'undefined')
            if label in ['undefined', 'room']:
                continue
            if label not in classes:
                continue
            if not entry.get('has_latent') or not entry.get('latent_path'):
                continue
            labeled_entries.append((entry['audio_path'], label))
    else:
        # Load has_latent lookup from format_manifest.json
        has_latent_lookup = load_has_latent_lookup()
        for path, meta in manifest.items():
            if not isinstance(meta, dict):
                continue
            label = meta.get('group', 'undefined')
            if label in ['undefined', 'room']:  # Skip these
                continue
            if label not in classes:  # Skip classes not in training
                continue
            # Check latent exists - first from meta, then from lookup (try relative path)
            has_latent = meta.get('has_latent', False)
            if not has_latent:
                # Try relative path for lookup
                rel_path = path.replace('/home/arlo/gcs-bucket/', '')
                has_latent = has_latent_lookup.get(rel_path, False)
            if has_latent:
                labeled_entries.append((path, label))

    logging.info(f"Found {len(labeled_entries):,} labeled entries with latents to validate")

    if len(labeled_entries) == 0:
        logging.error("No labeled entries found!")
        return {}

    # Extract features
    audio_paths = [p for p, _ in labeled_entries]
    true_labels = [l for _, l in labeled_entries]

    X, valid_paths, failed = extract_features_batch(audio_paths, num_workers=num_workers)

    if len(X) == 0:
        logging.error("No features extracted!")
        return {}

    # Filter labels to match valid paths
    path_to_label = dict(labeled_entries)
    valid_labels = [path_to_label[p] for p in valid_paths]

    # Normalize features
    X_norm = (X - mean) / std

    # Get predictions
    logging.info("Running inference...")
    all_probs = []
    all_features = []

    with torch.no_grad():
        for i in range(0, len(X_norm), BATCH_SIZE):
            batch = X_norm[i:i+BATCH_SIZE].to(device_obj)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_features.append(batch.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_features = torch.cat(all_features, dim=0).numpy()

    # Calculate entropy for each prediction
    entropy = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=1)

    # Calculate class centroids
    logging.info("Computing class centroids for outlier detection...")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    class_features = {c: [] for c in classes}

    for i, label in enumerate(valid_labels):
        class_features[label].append(all_features[i])

    centroids = {}
    class_distances = {}
    for c in classes:
        if len(class_features[c]) > 0:
            centroids[c] = np.mean(class_features[c], axis=0)
            # Calculate distances to centroid for this class
            dists = cdist([centroids[c]], class_features[c], metric='euclidean')[0]
            class_distances[c] = dists

    # Calculate outlier thresholds per class
    outlier_thresholds = {}
    for c, dists in class_distances.items():
        if len(dists) > 10:
            outlier_thresholds[c] = np.percentile(dists, outlier_percentile)
        else:
            outlier_thresholds[c] = float('inf')

    # Analyze each entry
    logging.info("Analyzing entries for potential mislabels...")

    flagged_disagreement = []
    flagged_uncertain = []
    flagged_outlier = []
    all_results = []

    for i, (path, true_label) in enumerate(zip(valid_paths, valid_labels)):
        probs = all_probs[i]
        pred_idx = probs.argmax()
        pred_label = classes[pred_idx]
        pred_conf = probs[pred_idx]
        entry_entropy = entropy[i]

        # Calculate distance to true class centroid
        if true_label in centroids:
            dist_to_centroid = np.linalg.norm(all_features[i] - centroids[true_label])
            is_outlier = dist_to_centroid > outlier_thresholds.get(true_label, float('inf'))
        else:
            dist_to_centroid = None
            is_outlier = False

        # Determine flags
        flags = []
        flag_reasons = []

        # Flag 1: High-confidence disagreement
        if pred_label != true_label and pred_conf >= disagreement_threshold:
            flags.append('disagreement')
            flag_reasons.append(f"Labeled '{true_label}' but classifier says '{pred_label}' ({pred_conf:.1%})")
            flagged_disagreement.append({
                'path': path,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': float(pred_conf),
                'entropy': float(entry_entropy),
            })

        # Flag 2: High uncertainty
        if entry_entropy >= entropy_threshold:
            flags.append('uncertain')
            top3 = sorted(zip(classes, probs), key=lambda x: -x[1])[:3]
            top3_str = ', '.join([f"{c}:{p:.1%}" for c, p in top3])
            flag_reasons.append(f"High uncertainty (entropy={entry_entropy:.2f}): {top3_str}")
            flagged_uncertain.append({
                'path': path,
                'true_label': true_label,
                'entropy': float(entry_entropy),
                'top_predictions': {c: float(p) for c, p in top3},
            })

        # Flag 3: Outlier in feature space
        if is_outlier:
            flags.append('outlier')
            flag_reasons.append(f"Outlier in '{true_label}' class (dist={dist_to_centroid:.2f}, threshold={outlier_thresholds.get(true_label, 0):.2f})")
            flagged_outlier.append({
                'path': path,
                'true_label': true_label,
                'distance_to_centroid': float(dist_to_centroid),
                'threshold': float(outlier_thresholds.get(true_label, 0)),
            })

        result = {
            'path': path,
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': float(pred_conf),
            'entropy': float(entry_entropy),
            'distance_to_centroid': float(dist_to_centroid) if dist_to_centroid else None,
            'flags': flags,
            'flag_reasons': flag_reasons,
        }
        all_results.append(result)

    # Summary
    total_flagged = len(set(r['path'] for r in all_results if r['flags']))

    logging.info("\n" + "=" * 60)
    logging.info("VALIDATION SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Total validated: {len(all_results):,}")
    logging.info(f"Total flagged: {total_flagged:,} ({100*total_flagged/len(all_results):.1f}%)")
    logging.info(f"  - Disagreement (confident misprediction): {len(flagged_disagreement):,}")
    logging.info(f"  - Uncertain (high entropy): {len(flagged_uncertain):,}")
    logging.info(f"  - Outlier (far from class centroid): {len(flagged_outlier):,}")

    # Disagreement breakdown by class
    if flagged_disagreement:
        logging.info("\nDisagreement breakdown (true -> predicted):")
        confusion = Counter((r['true_label'], r['predicted_label']) for r in flagged_disagreement)
        for (true, pred), count in confusion.most_common(20):
            logging.info(f"  {true} -> {pred}: {count}")

    # Save results
    output_data = {
        'summary': {
            'total_validated': len(all_results),
            'total_flagged': total_flagged,
            'flagged_disagreement': len(flagged_disagreement),
            'flagged_uncertain': len(flagged_uncertain),
            'flagged_outlier': len(flagged_outlier),
            'thresholds': {
                'disagreement_confidence': disagreement_threshold,
                'entropy': entropy_threshold,
                'outlier_percentile': outlier_percentile,
            }
        },
        'flagged_disagreement': flagged_disagreement,
        'flagged_uncertain': flagged_uncertain,
        'flagged_outlier': flagged_outlier,
        'all_results': all_results,
        'validated_at': datetime.now().isoformat(),
    }

    output_path = output_dir / 'validation_results.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"\nResults saved to {output_path}")

    # Also save just the flagged paths for easy review
    flagged_paths_file = output_dir / 'flagged_paths.txt'
    with open(flagged_paths_file, 'w') as f:
        for r in all_results:
            if r['flags']:
                f.write(f"{r['path']}\t{r['true_label']}\t{','.join(r['flags'])}\t{r['predicted_label']}:{r['confidence']:.2f}\n")
    logging.info(f"Flagged paths saved to {flagged_paths_file}")

    return output_data


def update_manifest(manifest_path: Path, predictions_path: Path,
                    min_confidence: float = CONFIDENCE_MEDIUM) -> int:
    """Update manifest with predictions above confidence threshold."""

    logging.info(f"Updating manifest with predictions (min confidence: {min_confidence:.0%})...")

    with open(manifest_path) as f:
        manifest = json.load(f)

    with open(predictions_path) as f:
        pred_data = json.load(f)

    updated = 0
    for pred in pred_data['predictions']:
        if pred['confidence'] >= min_confidence:
            path = pred['path']
            group = pred['predicted_group']

            manifest[path] = {
                'group': group,
                'subgroup': 'undefined',
                'filename': Path(path).name,
                'classifier_confidence': pred['confidence'],
                'labeling_method': 'latent_classifier'
            }
            updated += 1

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logging.info(f"Updated {updated} entries in manifest")
    return updated


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(
        description='Latent-space instrument classifier (GPU-accelerated)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--mode', choices=['train', 'classify', 'validate', 'temporal', 'full'], required=True)
    parser.add_argument('--manifest', type=str,
                        help='Manifest JSON with audio paths and group labels')
    parser.add_argument('--corrections', type=str,
                        help='Corrections JSON to override manifest labels during training')
    parser.add_argument('--undefined', type=str,
                        help='File with undefined audio paths to classify')
    parser.add_argument('--model', type=str,
                        help='Path to trained model')
    parser.add_argument('--output-manifest', type=str,
                        help='Manifest to update with predictions')
    parser.add_argument('--output-dir', type=str, default='./latent_classifier_output')
    parser.add_argument('--workers', type=int, default=12,
                        help='Parallel workers for loading latents')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--min-confidence', type=float, default=CONFIDENCE_MEDIUM)
    parser.add_argument('--disagreement-threshold', type=float, default=0.7,
                        help='Min confidence to flag label disagreement (validate mode)')
    parser.add_argument('--entropy-threshold', type=float, default=1.5,
                        help='Max entropy before flagging as uncertain (validate mode)')
    parser.add_argument('--outlier-percentile', type=float, default=95,
                        help='Percentile for outlier detection (validate mode)')
    # Temporal mode options
    parser.add_argument('--window-sec', type=float, default=2.0,
                        help='Window size in seconds for temporal mode')
    parser.add_argument('--hop-sec', type=float, default=1.0,
                        help='Hop size in seconds for temporal mode')
    parser.add_argument('--group', type=str,
                        help='Filter to specific group for temporal analysis')
    parser.add_argument('--exclude-file', type=str,
                        help='File with paths to exclude from training (one per line)')

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
        corrections_path = Path(args.corrections) if args.corrections else None
        exclude_file = Path(args.exclude_file) if args.exclude_file else None
        train_classifier(
            Path(args.manifest),
            output_dir,
            corrections_path=corrections_path,
            num_workers=args.workers,
            device=args.device,
            exclude_file=exclude_file
        )

    elif args.mode == 'classify':
        if not args.model:
            parser.error("--model required for classify mode")
        if not args.undefined and not args.manifest:
            parser.error("--undefined or --manifest required for classify mode")

        # Load undefined paths from file or use manifest directly
        undefined_paths = []
        manifest_path = None

        if args.undefined:
            with open(args.undefined) as f:
                undefined_paths = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        elif args.manifest:
            # Pass manifest path to classify_undefined, it will extract undefined paths
            manifest_path = Path(args.manifest)

        predictions_path = output_dir / 'predictions.json'
        classify_undefined(
            Path(args.model),
            undefined_paths,
            predictions_path,
            num_workers=args.workers,
            device=args.device,
            manifest_path=manifest_path
        )

        if args.output_manifest:
            update_manifest(Path(args.output_manifest), predictions_path, args.min_confidence)

    elif args.mode == 'validate':
        if not args.model:
            parser.error("--model required for validate mode")
        if not args.manifest:
            parser.error("--manifest required for validate mode")

        validate_labels(
            Path(args.model),
            Path(args.manifest),
            output_dir,
            num_workers=args.workers,
            device=args.device,
            disagreement_threshold=args.disagreement_threshold,
            entropy_threshold=args.entropy_threshold,
            outlier_percentile=args.outlier_percentile,
        )

    elif args.mode == 'temporal':
        if not args.model:
            parser.error("--model required for temporal mode")
        if not args.manifest and not args.undefined:
            parser.error("--manifest or --undefined required for temporal mode")

        # Get paths to analyze
        audio_paths = []

        if args.undefined:
            with open(args.undefined) as f:
                audio_paths = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        elif args.manifest:
            logging.info(f"Loading paths from manifest: {args.manifest}")
            with open(args.manifest) as f:
                manifest = json.load(f)

            # Support unified manifest format
            if 'entries' in manifest:
                for entry in manifest['entries']:
                    if not entry.get('has_latent'):
                        continue
                    # Optional: filter by group
                    if args.group and entry.get('group') != args.group:
                        continue
                    audio_paths.append(entry['audio_path'])
            else:
                # Load has_latent lookup from format_manifest.json
                has_latent_lookup = load_has_latent_lookup()
                for path, meta in manifest.items():
                    if isinstance(meta, dict):
                        if args.group and meta.get('group') != args.group:
                            continue
                    # Only include paths with latents
                    if has_latent_lookup.get(path, False):
                        audio_paths.append(path)

            logging.info(f"Found {len(audio_paths)} paths to analyze")

        if len(audio_paths) == 0:
            logging.error("No audio paths to analyze!")
        else:
            # Limit for testing if needed
            # audio_paths = audio_paths[:1000]

            temporal_output = output_dir / 'temporal_analysis.json'
            run_temporal_classification(
                Path(args.model),
                audio_paths,
                temporal_output,
                window_sec=args.window_sec,
                hop_sec=args.hop_sec,
                min_confidence=args.min_confidence,
                num_workers=args.workers,
                device=args.device
            )

    elif args.mode == 'full':
        if not args.manifest or not args.undefined:
            parser.error("--manifest and --undefined required")

        logging.info("=" * 60)
        logging.info("PHASE 1: TRAINING")
        logging.info("=" * 60)
        exclude_file = Path(args.exclude_file) if args.exclude_file else None
        train_classifier(
            Path(args.manifest),
            output_dir,
            num_workers=args.workers,
            device=args.device,
            exclude_file=exclude_file
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info("\n" + "=" * 60)
        logging.info("PHASE 2: CLASSIFICATION")
        logging.info("=" * 60)
        with open(args.undefined) as f:
            undefined_paths = [l.strip() for l in f if l.strip() and not l.startswith('#')]

        predictions_path = output_dir / 'predictions.json'
        classify_undefined(
            output_dir / 'model.pt',
            undefined_paths,
            predictions_path,
            num_workers=args.workers,
            device=args.device
        )

        if args.output_manifest:
            logging.info("\n" + "=" * 60)
            logging.info("PHASE 3: UPDATE MANIFEST")
            logging.info("=" * 60)
            update_manifest(Path(args.output_manifest), predictions_path, args.min_confidence)


if __name__ == '__main__':
    main()
