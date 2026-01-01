#!/usr/bin/env python3
"""
Robust Multi-Class Instrument Classifier

Handles noisy labels through:
1. Cross-validation based filtering (identify mislabeled samples)
2. Iterative refinement
3. Confidence-based prediction tiers

Usage:
  # Train classifier (with label cleaning)
  python robust_instrument_classifier.py --mode train \
    --labeled-dir /home/arlo/Data/new_labels/categorized \
    --output-dir /home/arlo/Data/classifier

  # Classify undefined files
  python robust_instrument_classifier.py --mode classify \
    --model /home/arlo/Data/classifier/instrument_model.pkl \
    --input /home/arlo/undefined_audio_paths.txt \
    --output /home/arlo/Data/classifier/predictions.json

  # Full pipeline (train + classify + update manifest)
  python robust_instrument_classifier.py --mode full \
    --labeled-dir /home/arlo/Data/new_labels/categorized \
    --undefined /home/arlo/undefined_audio_paths.txt \
    --manifest /home/arlo/gcs-bucket/Manifests/combined_manifest.json \
    --output-dir /home/arlo/Data/classifier
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import json
import pickle
import tempfile
import subprocess
import warnings
import gc
import resource
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import numpy as np

# Memory limit per process (4GB)
MAX_MEMORY_MB = 4000

def limit_memory():
    """Set memory limit for child processes."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_MB * 1024 * 1024, hard))
    except Exception:
        pass
from scipy.fft import rfft, rfftfreq
from scipy.fftpack import dct
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

warnings.filterwarnings('ignore')

# ===================== CONFIGURATION =====================

# Confidence thresholds for prediction tiers
CONFIDENCE_HIGH = 0.85      # Apply label confidently
CONFIDENCE_MEDIUM = 0.65    # Apply label but flag for review
CONFIDENCE_LOW = 0.65       # Keep as undefined

# Cross-validation folds for label cleaning
CV_FOLDS = 5
MISLABEL_THRESHOLD = 3  # Misclassified in >= this many folds = likely mislabeled

# Feature extraction settings
TARGET_SR = 22050
N_FFT = 2048
N_MFCC = 13
HOP_LENGTH = 512

# Training settings
MAX_SAMPLES_PER_CLASS = 2000  # Limit for faster training (reduced from 5000)
MIN_SAMPLES_PER_CLASS = 50    # Minimum to include class
BATCH_SIZE = 200  # Process in batches to limit memory

FREQUENCY_BANDS = {
    'sub_bass': (20, 80),
    'bass': (80, 200),
    'low_mid': (200, 500),
    'mid': (500, 1500),
    'high_mid': (1500, 4000),
    'high': (4000, 8000),
    'ultra_high': (8000, 11025)
}


# ===================== FEATURE EXTRACTION =====================

def load_audio_sox(audio_path: str, target_sr: int = TARGET_SR) -> Optional[np.ndarray]:
    """Load audio using sox (avoids librosa dependency)."""
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            'sox', audio_path,
            '-t', 'raw', '-e', 'float', '-b', '32', '-c', '1',
            '-r', str(target_sr), tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None

        audio = np.fromfile(tmp_path, dtype=np.float32)
        return audio
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def compute_mfcc(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Compute MFCCs manually."""
    n_mels = 40
    fmin, fmax = 80, sr // 2

    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    if n_frames < 1:
        return np.zeros((N_MFCC, 1))

    # STFT with windowing
    frames = []
    window = np.hanning(N_FFT)
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = audio[start:start + N_FFT]
        if len(frame) < N_FFT:
            frame = np.pad(frame, (0, N_FFT - len(frame)))
        frames.append(frame * window)

    frames = np.array(frames)
    spectrum = np.abs(np.fft.rfft(frames, axis=1)) ** 2

    # Mel filterbank
    freqs = np.fft.rfftfreq(N_FFT, 1/sr)
    mel_low = 2595 * np.log10(1 + fmin / 700)
    mel_high = 2595 * np.log10(1 + fmax / 700)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    filterbank = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        for j, f in enumerate(freqs):
            if left <= f < center:
                filterbank[i, j] = (f - left) / (center - left)
            elif center <= f < right:
                filterbank[i, j] = (right - f) / (right - center)

    mel_spec = np.dot(spectrum, filterbank.T)
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)

    mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :N_MFCC]
    return mfcc.T


def extract_features(audio_path: str) -> Optional[np.ndarray]:
    """Extract feature vector for classification."""
    audio = load_audio_sox(audio_path)

    if audio is None or len(audio) < TARGET_SR * 0.3:
        return None

    # Trim silence
    threshold = np.max(np.abs(audio)) * 0.01
    above = np.abs(audio) > threshold
    if np.any(above):
        indices = np.where(above)[0]
        start = max(0, indices[0] - int(TARGET_SR * 0.05))
        end = min(len(audio), indices[-1] + int(TARGET_SR * 0.05))
        audio = audio[start:end]

    if len(audio) < TARGET_SR * 0.2:
        return None

    features = []

    # 1. MFCCs (mean + std over time)
    mfcc = compute_mfcc(audio)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2. Delta MFCCs
    if mfcc.shape[1] > 2:
        delta_mfcc = np.diff(mfcc, axis=1)
        features.extend(np.mean(delta_mfcc, axis=1))
        features.extend(np.std(delta_mfcc, axis=1))
    else:
        features.extend([0] * 26)

    # 3. Spectral features
    chunk = audio[:min(len(audio), N_FFT * 10)]
    spectrum = rfft(chunk, n=N_FFT)
    freqs = rfftfreq(N_FFT, 1/TARGET_SR)
    magnitude = np.abs(spectrum)

    # Centroid
    if np.sum(magnitude) > 1e-10:
        centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
    else:
        centroid = 0
    features.append(centroid)

    # Rolloff (85%)
    cumsum = np.cumsum(magnitude ** 2)
    total = cumsum[-1]
    if total > 1e-10:
        rolloff_idx = np.searchsorted(cumsum, 0.85 * total)
        rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
    else:
        rolloff = 0
    features.append(rolloff)

    # Band energies (normalized)
    band_energies = []
    total_energy = 0
    for name, (low, high) in FREQUENCY_BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        energy = np.mean(magnitude[mask] ** 2) if np.any(mask) else 0
        band_energies.append(energy)
        total_energy += energy

    if total_energy > 0:
        band_energies = [e / total_energy for e in band_energies]
    features.extend(band_energies)

    # Spectral contrast (std in each band)
    for name, (low, high) in FREQUENCY_BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            features.append(np.std(magnitude[mask]))
        else:
            features.append(0)

    # Spectral flatness
    geo_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
    arith_mean = np.mean(magnitude)
    flatness = geo_mean / (arith_mean + 1e-10)
    features.append(flatness)

    # Zero crossing rate
    zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2
    features.append(zcr)

    # RMS energy
    rms = np.sqrt(np.mean(audio ** 2))
    features.append(rms)

    return np.array(features)


def get_feature_names() -> List[str]:
    """Get feature names for interpretability."""
    names = []
    for i in range(N_MFCC):
        names.append(f'mfcc_{i}_mean')
    for i in range(N_MFCC):
        names.append(f'mfcc_{i}_std')
    for i in range(N_MFCC):
        names.append(f'delta_mfcc_{i}_mean')
    for i in range(N_MFCC):
        names.append(f'delta_mfcc_{i}_std')
    names.extend(['centroid', 'rolloff'])
    names.extend([f'band_{name}' for name in FREQUENCY_BANDS.keys()])
    names.extend([f'contrast_{name}' for name in FREQUENCY_BANDS.keys()])
    names.extend(['flatness', 'zcr', 'rms'])
    return names


# ===================== PARALLEL PROCESSING =====================

def _extract_single(args) -> Tuple[Optional[np.ndarray], str, str]:
    """Worker function for parallel feature extraction."""
    audio_path, label = args
    try:
        features = extract_features(audio_path)
        return features, label, audio_path
    except Exception as e:
        return None, label, audio_path


def extract_features_parallel(file_label_pairs: List[Tuple[str, str]],
                              workers: int = 4,
                              desc: str = "Extracting features") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract features in parallel with batched processing to limit memory."""
    features_list = []
    labels_list = []
    paths_list = []
    failed = 0
    total = len(file_label_pairs)

    logging.info(f"{desc}: {total} files with {workers} workers (batch size: {BATCH_SIZE})")
    start_time = datetime.now()
    processed = 0

    # Process in batches to limit memory usage
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = file_label_pairs[batch_start:batch_end]

        with ProcessPoolExecutor(max_workers=workers, initializer=limit_memory) as executor:
            futures = {executor.submit(_extract_single, pair): pair for pair in batch}

            for future in as_completed(futures):
                try:
                    features, label, path = future.result(timeout=60)
                    if features is not None:
                        features_list.append(features)
                        labels_list.append(label)
                        paths_list.append(path)
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1

                processed += 1

        # Progress logging
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (total - processed) / rate if rate > 0 else 0
        logging.info(f"  {processed}/{total} ({rate:.1f}/s, ~{remaining/60:.1f}m remaining, {failed} failed)")

        # Force garbage collection between batches
        gc.collect()

    elapsed = (datetime.now() - start_time).total_seconds()
    logging.info(f"  Complete: {len(features_list)} success, {failed} failed ({elapsed:.1f}s)")

    if len(features_list) == 0:
        return np.array([]), np.array([]), []

    return np.array(features_list), np.array(labels_list), paths_list


# ===================== LABEL CLEANING =====================

def identify_mislabeled(X: np.ndarray, y: np.ndarray, paths: List[str],
                        n_folds: int = CV_FOLDS) -> Tuple[Set[int], Dict]:
    """
    Use cross-validation to identify likely mislabeled samples.
    Returns indices of samples misclassified in >= MISLABEL_THRESHOLD folds.
    """
    logging.info(f"Identifying mislabeled samples using {n_folds}-fold CV...")

    misclassified_counts = np.zeros(len(y), dtype=int)
    fold_predictions = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=2  # Limit threads to reduce memory
        )
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_val_scaled)
        wrong = y_pred != y_val
        misclassified_counts[val_idx[wrong]] += 1

        logging.info(f"  Fold {fold+1}: {np.sum(wrong)}/{len(val_idx)} misclassified")

    # Find likely mislabeled
    mislabeled_idx = set(np.where(misclassified_counts >= MISLABEL_THRESHOLD)[0])

    # Analyze mislabeled by class
    mislabeled_by_class = defaultdict(list)
    for idx in mislabeled_idx:
        mislabeled_by_class[y[idx]].append(paths[idx])

    stats = {
        'total_mislabeled': len(mislabeled_idx),
        'by_class': {k: len(v) for k, v in mislabeled_by_class.items()},
        'threshold': MISLABEL_THRESHOLD,
        'folds': n_folds
    }

    logging.info(f"\nMislabeled detection results:")
    logging.info(f"  Total flagged: {len(mislabeled_idx)} ({100*len(mislabeled_idx)/len(y):.1f}%)")
    for cls, count in sorted(stats['by_class'].items(), key=lambda x: -x[1])[:10]:
        logging.info(f"    {cls}: {count}")

    return mislabeled_idx, stats


# ===================== TRAINING =====================

def train_classifier(labeled_dir: Path, output_dir: Path,
                     workers: int = 8, clean_labels: bool = True) -> Dict:
    """Train instrument classifier with optional label cleaning."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labeled files from categorized directory
    logging.info("Loading labeled files...")
    file_label_pairs = []
    class_counts = Counter()

    for txt_file in labeled_dir.glob('*.txt'):
        label = txt_file.stem
        if label == 'undefined':
            continue  # Skip undefined for training

        with open(txt_file) as f:
            paths = [line.strip() for line in f if line.strip()]

        # Limit samples per class
        if len(paths) > MAX_SAMPLES_PER_CLASS:
            import random
            random.seed(42)
            paths = random.sample(paths, MAX_SAMPLES_PER_CLASS)

        if len(paths) < MIN_SAMPLES_PER_CLASS:
            logging.warning(f"  Skipping {label}: only {len(paths)} samples (min: {MIN_SAMPLES_PER_CLASS})")
            continue

        for path in paths:
            file_label_pairs.append((path, label))
        class_counts[label] = len(paths)

    logging.info(f"Loaded {len(file_label_pairs)} files across {len(class_counts)} classes:")
    for label, count in class_counts.most_common():
        logging.info(f"  {label}: {count}")

    # Extract features
    X, y, paths = extract_features_parallel(file_label_pairs, workers=workers,
                                            desc="Extracting training features")

    if len(X) < 100:
        raise ValueError(f"Not enough samples for training: {len(X)}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Clean labels if requested
    mislabeled_idx = set()
    cleaning_stats = {}
    if clean_labels:
        mislabeled_idx, cleaning_stats = identify_mislabeled(X, y, paths)

        # Save mislabeled paths for review
        mislabeled_file = output_dir / 'mislabeled_samples.json'
        mislabeled_data = {y[i]: paths[i] for i in mislabeled_idx}
        by_class = defaultdict(list)
        for i in mislabeled_idx:
            by_class[y[i]].append(paths[i])
        with open(mislabeled_file, 'w') as f:
            json.dump(dict(by_class), f, indent=2)
        logging.info(f"Saved mislabeled samples to {mislabeled_file}")

    # Filter out mislabeled
    if mislabeled_idx:
        clean_mask = np.array([i not in mislabeled_idx for i in range(len(X))])
        X_clean = X[clean_mask]
        y_clean = y[clean_mask]
        paths_clean = [p for i, p in enumerate(paths) if i not in mislabeled_idx]
        logging.info(f"Training on {len(X_clean)} samples (removed {len(mislabeled_idx)} likely mislabeled)")
    else:
        X_clean, y_clean, paths_clean = X, y, paths

    y_clean_encoded = label_encoder.transform(y_clean)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean_encoded, test_size=0.15, random_state=42, stratify=y_clean_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    logging.info("Training RandomForest classifier...")
    clf = RandomForestClassifier(
        n_estimators=150,  # Reduced from 200
        max_depth=12,      # Reduced from 15
        min_samples_leaf=3,  # Increased from 2
        class_weight='balanced',
        random_state=42,
        n_jobs=2  # Limit threads to reduce memory
    )
    clf.fit(X_train_scaled, y_train)

    # Calibrate probabilities
    logging.info("Calibrating probabilities...")
    calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
    calibrated_clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = calibrated_clf.predict(X_test_scaled)
    y_proba = calibrated_clf.predict_proba(X_test_scaled)

    logging.info("\n" + "=" * 60)
    logging.info("CLASSIFICATION REPORT")
    logging.info("=" * 60)
    report = classification_report(y_test, y_pred,
                                   target_names=label_encoder.classes_,
                                   output_dict=True)
    logging.info("\n" + classification_report(y_test, y_pred,
                                              target_names=label_encoder.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"\nConfusion Matrix:\n{cm}")

    # High-confidence accuracy
    max_proba = np.max(y_proba, axis=1)
    for threshold in [0.9, 0.8, 0.7]:
        mask = max_proba >= threshold
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == y_test[mask])
            logging.info(f"Accuracy at {threshold:.0%} confidence: {acc:.1%} ({np.sum(mask)} samples)")

    # Save model
    model_data = {
        'classifier': calibrated_clf,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'classes': label_encoder.classes_.tolist(),
        'feature_names': get_feature_names(),
        'training_stats': {
            'total_samples': len(X),
            'clean_samples': len(X_clean),
            'mislabeled_removed': len(mislabeled_idx),
            'test_accuracy': report['accuracy'],
            'class_counts': dict(Counter(y_clean))
        },
        'cleaning_stats': cleaning_stats,
        'trained_at': datetime.now().isoformat()
    }

    model_path = output_dir / 'instrument_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    logging.info(f"\nModel saved to {model_path}")

    # Feature importance
    logging.info("\nTop 15 important features:")
    feature_names = get_feature_names()
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    for i, idx in enumerate(indices):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        logging.info(f"  {i+1}. {name}: {importances[idx]:.4f}")

    # Cleanup
    del X, y, X_clean, y_clean, X_train, X_test, y_train, y_test
    gc.collect()

    return model_data


# ===================== CLASSIFICATION =====================

def classify_undefined(model_path: Path, undefined_paths: List[str],
                       output_path: Path, workers: int = 8) -> Dict:
    """Classify undefined files using trained model."""

    # Load model
    logging.info(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    clf = model_data['classifier']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    classes = model_data['classes']

    logging.info(f"Model classes: {classes}")

    # Extract features
    pairs = [(path, 'unknown') for path in undefined_paths]
    X, _, paths = extract_features_parallel(pairs, workers=workers,
                                            desc="Extracting features for classification")

    if len(X) == 0:
        logging.error("No features extracted!")
        return {}

    # Scale and predict
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    y_proba = clf.predict_proba(X_scaled)
    max_proba = np.max(y_proba, axis=1)

    # Decode predictions
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Categorize by confidence
    predictions = {
        'high_confidence': [],    # >= 0.85
        'medium_confidence': [],  # >= 0.65
        'low_confidence': []      # < 0.65
    }

    results = []
    for i, (path, label, conf) in enumerate(zip(paths, y_pred_labels, max_proba)):
        result = {
            'path': path,
            'predicted_group': label,
            'confidence': float(conf),
            'all_probabilities': {c: float(p) for c, p in zip(classes, y_proba[i])}
        }
        results.append(result)

        if conf >= CONFIDENCE_HIGH:
            predictions['high_confidence'].append(result)
        elif conf >= CONFIDENCE_MEDIUM:
            predictions['medium_confidence'].append(result)
        else:
            predictions['low_confidence'].append(result)

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("CLASSIFICATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"Total classified: {len(results)}")
    logging.info(f"  High confidence (>={CONFIDENCE_HIGH:.0%}): {len(predictions['high_confidence'])}")
    logging.info(f"  Medium confidence ({CONFIDENCE_MEDIUM:.0%}-{CONFIDENCE_HIGH:.0%}): {len(predictions['medium_confidence'])}")
    logging.info(f"  Low confidence (<{CONFIDENCE_MEDIUM:.0%}): {len(predictions['low_confidence'])}")

    # Predictions by class
    high_by_class = Counter(r['predicted_group'] for r in predictions['high_confidence'])
    logging.info("\nHigh-confidence predictions by class:")
    for cls, count in high_by_class.most_common():
        logging.info(f"  {cls}: {count}")

    # Save results
    output_data = {
        'predictions': results,
        'summary': {
            'total': len(results),
            'high_confidence': len(predictions['high_confidence']),
            'medium_confidence': len(predictions['medium_confidence']),
            'low_confidence': len(predictions['low_confidence']),
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

    # Cleanup
    del X, X_scaled, y_pred, y_proba, results
    gc.collect()

    return output_data


def update_manifest(manifest_path: Path, predictions_path: Path,
                    min_confidence: float = CONFIDENCE_MEDIUM) -> int:
    """Update manifest with predictions above confidence threshold."""

    logging.info(f"Updating manifest with predictions (min confidence: {min_confidence:.0%})...")

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load predictions
    with open(predictions_path) as f:
        pred_data = json.load(f)

    # Update manifest
    updated = 0
    for pred in pred_data['predictions']:
        if pred['confidence'] >= min_confidence:
            path = pred['path']
            group = pred['predicted_group']

            manifest[path] = {
                'group': group,
                'subgroup': 'undefined',  # Could add subgroup classifier later
                'filename': Path(path).name,
                'classifier_confidence': pred['confidence'],
                'labeling_method': 'audio_classifier'
            }
            updated += 1

    # Save updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logging.info(f"Updated {updated} entries in manifest")
    return updated


# ===================== MAIN =====================

def get_memory_usage_mb():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback to /proc on Linux
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024
        except Exception:
            return 0
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Robust multi-class instrument classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--mode', choices=['train', 'classify', 'full'], required=True,
                        help='Mode: train, classify, or full pipeline')
    parser.add_argument('--labeled-dir', type=str,
                        help='Directory with categorized/*.txt files for training')
    parser.add_argument('--undefined', type=str,
                        help='File with undefined audio paths to classify')
    parser.add_argument('--model', type=str,
                        help='Path to trained model (for classify mode)')
    parser.add_argument('--manifest', type=str,
                        help='Manifest to update with predictions')
    parser.add_argument('--output-dir', type=str, default='./classifier_output',
                        help='Output directory')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (reduced for memory safety)')
    parser.add_argument('--no-clean', action='store_true',
                        help='Skip label cleaning during training')
    parser.add_argument('--min-confidence', type=float, default=CONFIDENCE_MEDIUM,
                        help='Minimum confidence to apply predictions')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting with {get_memory_usage_mb():.0f} MB memory usage")

    if args.mode == 'train':
        if not args.labeled_dir:
            parser.error("--labeled-dir required for train mode")
        train_classifier(
            Path(args.labeled_dir),
            output_dir,
            workers=args.workers,
            clean_labels=not args.no_clean
        )

    elif args.mode == 'classify':
        if not args.model or not args.undefined:
            parser.error("--model and --undefined required for classify mode")

        with open(args.undefined) as f:
            undefined_paths = [l.strip() for l in f if l.strip() and not l.startswith('#')]

        predictions_path = output_dir / 'predictions.json'
        classify_undefined(
            Path(args.model),
            undefined_paths,
            predictions_path,
            workers=args.workers
        )

        if args.manifest:
            update_manifest(Path(args.manifest), predictions_path, args.min_confidence)

    elif args.mode == 'full':
        if not args.labeled_dir or not args.undefined:
            parser.error("--labeled-dir and --undefined required for full mode")

        # Train
        logging.info("=" * 60)
        logging.info("PHASE 1: TRAINING")
        logging.info("=" * 60)
        train_classifier(
            Path(args.labeled_dir),
            output_dir,
            workers=args.workers,
            clean_labels=not args.no_clean
        )

        # Cleanup after training
        gc.collect()
        logging.info(f"Memory after training: {get_memory_usage_mb():.0f} MB")

        # Classify
        logging.info("\n" + "=" * 60)
        logging.info("PHASE 2: CLASSIFICATION")
        logging.info("=" * 60)
        with open(args.undefined) as f:
            undefined_paths = [l.strip() for l in f if l.strip() and not l.startswith('#')]

        predictions_path = output_dir / 'predictions.json'
        classify_undefined(
            output_dir / 'instrument_model.pkl',
            undefined_paths,
            predictions_path,
            workers=args.workers
        )

        # Cleanup after classification
        gc.collect()
        logging.info(f"Memory after classification: {get_memory_usage_mb():.0f} MB")

        # Update manifest
        if args.manifest:
            logging.info("\n" + "=" * 60)
            logging.info("PHASE 3: UPDATE MANIFEST")
            logging.info("=" * 60)
            update_manifest(Path(args.manifest), predictions_path, args.min_confidence)


if __name__ == '__main__':
    main()
