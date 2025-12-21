#!/usr/bin/env python3
"""
Train a trumpet vs trombone classifier using spectral features + MFCCs.
"""

import json
import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import subprocess
import tempfile
import os

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """Extract spectral features + MFCCs for classification."""

    def __init__(self, target_sr: int = 22050, n_fft: int = 2048, n_mfcc: int = 13):
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        self.hop_length = 512

        self.bands = {
            'sub_bass': (20, 80),
            'bass': (80, 200),
            'low_mid': (200, 500),
            'mid': (500, 1500),
            'high_mid': (1500, 4000),
            'high': (4000, 8000),
            'ultra_high': (8000, 11025)
        }

    def load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Load audio using sox."""
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                'sox', audio_path,
                '-t', 'raw', '-e', 'float', '-b', '32', '-c', '1',
                '-r', str(self.target_sr), tmp_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return None, 0

            audio = np.fromfile(tmp_path, dtype=np.float32)
            return audio, self.target_sr
        except Exception:
            return None, 0
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def compute_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute MFCCs manually using DCT of log mel spectrogram."""
        # Simple mel filterbank
        n_mels = 40
        fmin, fmax = 80, sr // 2

        # Compute power spectrogram
        hop = self.hop_length
        n_frames = 1 + (len(audio) - self.n_fft) // hop
        if n_frames < 1:
            return np.zeros((self.n_mfcc, 1))

        # STFT
        frames = []
        for i in range(n_frames):
            start = i * hop
            frame = audio[start:start + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            # Apply window
            frame = frame * np.hanning(self.n_fft)
            frames.append(frame)

        frames = np.array(frames)
        spectrum = np.abs(np.fft.rfft(frames, axis=1)) ** 2

        # Mel filterbank
        freqs = np.fft.rfftfreq(self.n_fft, 1/sr)
        mel_low = 2595 * np.log10(1 + fmin / 700)
        mel_high = 2595 * np.log10(1 + fmax / 700)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)

        filterbank = np.zeros((n_mels, len(freqs)))
        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]

            for j, f in enumerate(freqs):
                if left <= f < center:
                    filterbank[i, j] = (f - left) / (center - left)
                elif center <= f < right:
                    filterbank[i, j] = (right - f) / (right - center)

        # Apply filterbank
        mel_spec = np.dot(spectrum, filterbank.T)
        mel_spec = np.maximum(mel_spec, 1e-10)
        log_mel = np.log(mel_spec)

        # DCT to get MFCCs
        from scipy.fftpack import dct
        mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :self.n_mfcc]

        return mfcc.T  # (n_mfcc, n_frames)

    def extract_features(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract feature vector for classification."""
        audio, sr = self.load_audio(audio_path)

        if audio is None or len(audio) < sr * 0.3:
            return None

        # Trim silence
        threshold = np.max(np.abs(audio)) * 0.01
        above = np.abs(audio) > threshold
        if np.any(above):
            indices = np.where(above)[0]
            start = max(0, indices[0] - int(sr * 0.05))
            end = min(len(audio), indices[-1] + int(sr * 0.05))
            audio = audio[start:end]

        if len(audio) < sr * 0.2:
            return None

        features = []

        # 1. MFCCs (mean + std over time)
        mfcc = self.compute_mfcc(audio, sr)
        features.extend(np.mean(mfcc, axis=1))  # 13 features
        features.extend(np.std(mfcc, axis=1))   # 13 features

        # 2. Delta MFCCs
        if mfcc.shape[1] > 2:
            delta_mfcc = np.diff(mfcc, axis=1)
            features.extend(np.mean(delta_mfcc, axis=1))
            features.extend(np.std(delta_mfcc, axis=1))
        else:
            features.extend([0] * 26)

        # 3. Spectral features
        spectrum = rfft(audio[:min(len(audio), self.n_fft * 10)], n=self.n_fft)
        freqs = rfftfreq(self.n_fft, 1/sr)
        magnitude = np.abs(spectrum)

        # Centroid
        if np.sum(magnitude) > 1e-10:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            centroid = 0
        features.append(centroid)

        # Rolloff
        cumsum = np.cumsum(magnitude ** 2)
        total = cumsum[-1]
        if total > 1e-10:
            rolloff_idx = np.searchsorted(cumsum, 0.85 * total)
            rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
        else:
            rolloff = 0
        features.append(rolloff)

        # Band energies
        total_energy = 0
        band_energies = []
        for name, (low, high) in self.bands.items():
            mask = (freqs >= low) & (freqs < high)
            energy = np.mean(magnitude[mask] ** 2) if np.any(mask) else 0
            band_energies.append(energy)
            total_energy += energy

        # Normalize band energies
        if total_energy > 0:
            band_energies = [e / total_energy for e in band_energies]
        features.extend(band_energies)  # 7 features

        # Spectral contrast (std of spectrum in bands)
        for name, (low, high) in self.bands.items():
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

        return np.array(features)


def process_file(args) -> Tuple[Optional[np.ndarray], str, str]:
    """Process single file for parallel extraction."""
    audio_path, label = args
    try:
        extractor = FeatureExtractor()
        features = extractor.extract_features(audio_path)
        return features, label, audio_path
    except Exception as e:
        return None, label, audio_path


def train_classifier(manifest_path: str, output_path: str, workers: int = 8,
                     max_samples: int = None, test_size: float = 0.2):
    """Train and save classifier."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get labeled data
    # Treat None as "not muted" (dry trumpet) - only exclude explicitly muted
    trumpets = [e for e in manifest
                if e.get('sub_group') == 'trumpet' and e.get('is_muted') != True]
    trombones = [e for e in manifest if e.get('sub_group') == 'trombone']

    logging.info(f"Found {len(trumpets)} dry trumpets, {len(trombones)} trombones")

    # Sample if needed
    if max_samples:
        import random
        random.seed(42)
        if len(trumpets) > max_samples:
            trumpets = random.sample(trumpets, max_samples)
        if len(trombones) > max_samples:
            trombones = random.sample(trombones, max_samples)
        logging.info(f"Sampled to {len(trumpets)} trumpets, {len(trombones)} trombones")

    # Prepare tasks
    tasks = []
    for e in trumpets:
        tasks.append((e['audio_path'], 'trumpet'))
    for e in trombones:
        tasks.append((e['audio_path'], 'trombone'))

    logging.info(f"Extracting features from {len(tasks)} files...")

    # Extract features in parallel
    features_list = []
    labels_list = []
    failed = 0

    start_time = datetime.now()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, task): task for task in tasks}

        for i, future in enumerate(as_completed(futures)):
            features, label, path = future.result()
            if features is not None:
                features_list.append(features)
                labels_list.append(label)
            else:
                failed += 1

            if (i + 1) % 100 == 0:
                elapsed = (datetime.now() - start_time).seconds
                logging.info(f"Processed {i+1}/{len(tasks)} ({elapsed}s, {failed} failed)")

    logging.info(f"Feature extraction complete: {len(features_list)} success, {failed} failed")

    if len(features_list) < 50:
        logging.error("Not enough samples for training!")
        return

    # Convert to arrays
    X = np.array(features_list)
    y = np.array(labels_list)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    logging.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    logging.info("Training RandomForest classifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)

    logging.info("\n" + "=" * 60)
    logging.info("CLASSIFICATION REPORT")
    logging.info("=" * 60)
    logging.info("\n" + classification_report(y_test, y_pred))

    logging.info("Confusion Matrix:")
    logging.info(f"\n{confusion_matrix(y_test, y_pred)}")

    # High-confidence accuracy
    max_proba = np.max(y_proba, axis=1)
    high_conf_mask = max_proba > 0.9
    if np.sum(high_conf_mask) > 0:
        high_conf_acc = np.mean(y_pred[high_conf_mask] == y_test[high_conf_mask])
        logging.info(f"\nHigh-confidence (>90%) accuracy: {high_conf_acc:.1%} ({np.sum(high_conf_mask)} samples)")

    # Save model
    model_data = {
        'classifier': clf,
        'scaler': scaler,
        'feature_names': get_feature_names(),
        'classes': clf.classes_.tolist()
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    logging.info(f"\nModel saved to: {output_path}")

    # Feature importance
    logging.info("\nTop 10 important features:")
    feature_names = get_feature_names()
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        logging.info(f"  {i+1}. {name}: {importances[idx]:.4f}")


def get_feature_names() -> List[str]:
    """Get feature names for interpretability."""
    names = []
    # MFCCs
    for i in range(13):
        names.append(f'mfcc_{i}_mean')
    for i in range(13):
        names.append(f'mfcc_{i}_std')
    # Delta MFCCs
    for i in range(13):
        names.append(f'delta_mfcc_{i}_mean')
    for i in range(13):
        names.append(f'delta_mfcc_{i}_std')
    # Spectral
    names.extend(['centroid', 'rolloff'])
    names.extend(['band_sub_bass', 'band_bass', 'band_low_mid', 'band_mid',
                  'band_high_mid', 'band_high', 'band_ultra_high'])
    names.extend(['contrast_sub_bass', 'contrast_bass', 'contrast_low_mid', 'contrast_mid',
                  'contrast_high_mid', 'contrast_high', 'contrast_ultra_high'])
    names.append('flatness')
    return names


def main():
    parser = argparse.ArgumentParser(description='Train trumpet vs trombone classifier')
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--output', type=str, default='./trumpet_trombone_model.pkl')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per class (for quick testing)')
    parser.add_argument('--test-size', type=float, default=0.2)

    args = parser.parse_args()
    train_classifier(args.manifest, args.output, args.workers,
                     args.max_samples, args.test_size)


if __name__ == '__main__':
    main()
