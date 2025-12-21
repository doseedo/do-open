#!/usr/bin/env python3
"""
Trumpet vs Trombone Detector

Distinguishes between trumpet and trombone recordings using:
1. ML classifier (if model provided) - uses MFCC + spectral features
2. Heuristic fallback - spectral analysis when no model available

Key differences:
- Trumpet: Higher fundamental (Bb3-D6, ~233-1175Hz), brighter harmonics, more high-freq energy
- Trombone: Lower fundamental (E2-F5, ~82-698Hz), warmer tone, more low-mid energy
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
from scipy.fftpack import dct

warnings.filterwarnings('ignore')

# Global model cache for multiprocessing
_MODEL_CACHE = {}


class FeatureExtractor:
    """Extract MFCC + spectral features for ML classification."""

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
        """Load audio file using sox."""
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
        """Compute MFCCs using DCT of log mel spectrogram."""
        n_mels = 40
        fmin, fmax = 80, sr // 2

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

        mel_spec = np.dot(spectrum, filterbank.T)
        mel_spec = np.maximum(mel_spec, 1e-10)
        log_mel = np.log(mel_spec)
        mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :self.n_mfcc]

        return mfcc.T  # (n_mfcc, n_frames)

    def extract_ml_features(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract feature vector for ML classification."""
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

        # 1. MFCCs (mean + std)
        mfcc = self.compute_mfcc(audio, sr)
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

        if total_energy > 0:
            band_energies = [e / total_energy for e in band_energies]
        features.extend(band_energies)

        # Spectral contrast
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


class TrumpetTromboneDetector:
    """Spectral analysis-based trumpet vs trombone classifier with ML support."""

    def __init__(self, model_path: str = None, target_sr: int = 22050, n_fft: int = 2048):
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.model = None
        self.scaler = None
        self.feature_extractor = FeatureExtractor(target_sr, n_fft)

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

        self.bands = {
            'sub_bass': (20, 80),
            'bass': (80, 200),
            'low_mid': (200, 500),
            'mid': (500, 1500),
            'high_mid': (1500, 4000),
            'high': (4000, 8000),
            'ultra_high': (8000, 11025)
        }

    def _load_model(self, model_path: str):
        """Load trained ML model."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['classifier']
            self.scaler = model_data['scaler']
            logging.info(f"Loaded ML model from {model_path}")
        except Exception as e:
            logging.warning(f"Failed to load model: {e}")

    def load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Load audio file using sox."""
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
                logging.debug(f"sox failed: {result.stderr}")
                return None, 0

            audio = np.fromfile(tmp_path, dtype=np.float32)
            return audio, self.target_sr
        except Exception as e:
            logging.debug(f"Failed to load {audio_path}: {e}")
            return None, 0
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def trim_silence(self, audio: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """Trim silence from audio."""
        if len(audio) == 0:
            return audio

        rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        if rms < 1e-8:
            return audio

        threshold = np.max(np.abs(audio)) * (10 ** (threshold_db / 20))

        above_thresh = np.abs(audio) > threshold
        if not np.any(above_thresh):
            return audio

        indices = np.where(above_thresh)[0]
        start = max(0, indices[0] - int(self.target_sr * 0.05))
        end = min(len(audio), indices[-1] + int(self.target_sr * 0.05))

        return audio[start:end]

    def get_band_energy(self, spectrum: np.ndarray, freqs: np.ndarray,
                        low: float, high: float) -> float:
        """Get energy in a frequency band."""
        band_mask = (freqs >= low) & (freqs < high)
        if not np.any(band_mask):
            return 0.0
        return float(np.mean(np.abs(spectrum[band_mask]) ** 2))

    def compute_spectral_centroid(self, spectrum: np.ndarray, freqs: np.ndarray) -> float:
        """Compute spectral centroid."""
        magnitude = np.abs(spectrum)
        if np.sum(magnitude) < 1e-10:
            return 0.0
        return float(np.sum(freqs * magnitude) / np.sum(magnitude))

    def compute_spectral_rolloff(self, spectrum: np.ndarray, freqs: np.ndarray,
                                  roll_percent: float = 0.85) -> float:
        """Compute spectral rolloff frequency."""
        magnitude = np.abs(spectrum) ** 2
        total_energy = np.sum(magnitude)
        if total_energy < 1e-10:
            return 0.0

        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.searchsorted(cumsum, roll_percent * total_energy)
        rolloff_idx = min(rolloff_idx, len(freqs) - 1)
        return float(freqs[rolloff_idx])

    def estimate_pitch(self, audio: np.ndarray, sr: int) -> Tuple[float, float, float]:
        """Estimate pitch using autocorrelation on a short segment."""
        max_samples = sr
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        if sr > 8000:
            factor = sr // 8000
            audio_ds = signal.decimate(audio, factor)
            sr_ds = sr // factor
        else:
            audio_ds = audio
            sr_ds = sr

        if len(audio_ds) > 4000:
            audio_ds = audio_ds[:4000]

        correlation = np.correlate(audio_ds, audio_ds, mode='full')
        correlation = correlation[len(correlation)//2:]

        min_period = int(sr_ds / 1000)
        max_period = int(sr_ds / 50)

        if max_period >= len(correlation):
            return 0.0, 0.0, 0.0

        search_region = correlation[min_period:max_period]
        if len(search_region) == 0:
            return 0.0, 0.0, 0.0

        peak_idx = np.argmax(search_region) + min_period
        if correlation[peak_idx] < correlation[0] * 0.1:
            return 0.0, 0.0, 0.0

        pitch = sr_ds / peak_idx
        return pitch, pitch * 0.8, pitch * 1.2

    def extract_features(self, audio_path: str) -> Dict:
        """Extract spectral features for heuristic classification."""
        audio, sr = self.load_audio(audio_path)

        if audio is None or len(audio) < sr * 0.3:
            return {'valid': False, 'duration': 0}

        duration = len(audio) / sr

        audio_trimmed = self.trim_silence(audio)
        if len(audio_trimmed) < sr * 0.2:
            audio_trimmed = audio

        spectrum = rfft(audio_trimmed, n=self.n_fft)
        freqs = rfftfreq(self.n_fft, 1/sr)

        band_energies = {}
        total_energy = 0
        for name, (low, high) in self.bands.items():
            energy = self.get_band_energy(spectrum, freqs, low, high)
            band_energies[name] = energy
            total_energy += energy

        band_ratios = {}
        if total_energy > 0:
            for name in band_energies:
                band_ratios[f'{name}_ratio'] = band_energies[name] / total_energy

        centroid = self.compute_spectral_centroid(spectrum, freqs)
        rolloff = self.compute_spectral_rolloff(spectrum, freqs)
        mean_pitch, min_pitch, max_pitch = self.estimate_pitch(audio_trimmed, sr)

        low_energy = band_energies.get('bass', 0) + band_energies.get('low_mid', 0)
        high_energy = band_energies.get('high_mid', 0) + band_energies.get('high', 0)
        low_high_ratio = low_energy / (high_energy + 1e-10)

        return {
            'valid': True,
            'duration': duration,
            'spectral_centroid': centroid,
            'spectral_rolloff': rolloff,
            'mean_pitch': mean_pitch,
            'min_pitch': min_pitch,
            'max_pitch': max_pitch,
            'low_high_ratio': low_high_ratio,
            **band_energies,
            **band_ratios
        }

    def compute_trumpet_score_heuristic(self, features: Dict) -> Tuple[float, str]:
        """Compute trumpet likelihood using heuristics (fallback when no model)."""
        if not features.get('valid', False):
            return 0.5, 'invalid_audio'

        score = 0.5
        reasons = []

        centroid = features.get('spectral_centroid', 0)
        if centroid > 2200:
            score += 0.20
            reasons.append(f'high_centroid:{centroid:.0f}')
        elif centroid > 1800:
            score += 0.12
        elif centroid < 1300:
            score -= 0.20
            reasons.append(f'low_centroid:{centroid:.0f}')
        elif centroid < 1600:
            score -= 0.10

        mean_pitch = features.get('mean_pitch', 0)
        if mean_pitch > 350:
            score += 0.08
            reasons.append(f'high_pitch:{mean_pitch:.0f}')
        elif 150 < mean_pitch < 250:
            score -= 0.05

        low_high_ratio = features.get('low_high_ratio', 1.0)
        if low_high_ratio < 0.7:
            score += 0.12
            reasons.append('bright_spectrum')
        elif low_high_ratio > 2.5:
            score -= 0.15
            reasons.append('dark_spectrum')
        elif low_high_ratio > 1.8:
            score -= 0.08

        high_ratio = features.get('high_ratio', 0)
        high_mid_ratio = features.get('high_mid_ratio', 0)
        if high_ratio > 0.12 or high_mid_ratio > 0.22:
            score += 0.10
            reasons.append('high_freq_energy')
        elif high_ratio < 0.03 and high_mid_ratio < 0.10:
            score -= 0.08
            reasons.append('low_high_freq')

        bass_ratio = features.get('bass_ratio', 0)
        if bass_ratio > 0.25:
            score -= 0.12
            reasons.append(f'high_bass:{bass_ratio:.2f}')
        elif bass_ratio < 0.06:
            score += 0.08

        rolloff = features.get('spectral_rolloff', 0)
        if rolloff > 4500:
            score += 0.08
        elif rolloff < 2200:
            score -= 0.08

        score = max(0.0, min(1.0, score))
        return score, ', '.join(reasons) if reasons else ''

    def detect_with_ml(self, audio_path: str) -> Tuple[str, float, float, str]:
        """
        Detect using ML model.

        Returns:
            Tuple of (prediction, probability, confidence, reason)
            - probability: 0-1, >0.5 = trumpet
            - confidence: max(proba) - how sure the model is
        """
        features = self.feature_extractor.extract_ml_features(audio_path)

        if features is None:
            return 'uncertain', 0.5, 0.0, 'invalid_audio'

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        proba = self.model.predict_proba(features_scaled)[0]

        # Get class indices
        classes = list(self.model.classes_)
        trumpet_idx = classes.index('trumpet') if 'trumpet' in classes else 0
        trombone_idx = classes.index('trombone') if 'trombone' in classes else 1

        trumpet_proba = proba[trumpet_idx]
        confidence = max(proba)

        if confidence >= 0.9:
            prediction = 'trumpet' if trumpet_proba > 0.5 else 'trombone'
            reason = f'high_conf:{confidence:.2f}'
        elif confidence >= 0.7:
            prediction = 'trumpet' if trumpet_proba > 0.5 else 'trombone'
            reason = f'medium_conf:{confidence:.2f}'
        else:
            prediction = 'uncertain'
            reason = f'low_conf:{confidence:.2f}'

        return prediction, trumpet_proba, confidence, reason

    def detect(self, audio_path: str, threshold: float = 0.5,
               confidence_threshold: float = 0.7) -> Tuple[str, float, Dict, str]:
        """
        Detect if audio is trumpet or trombone.

        If ML model is loaded, uses that with confidence thresholds.
        Otherwise falls back to heuristics.

        Returns:
            Tuple of (prediction, score, features, reason)
            - prediction: 'trumpet', 'trombone', or 'uncertain'
            - score: 0-1 (>0.5 = trumpet)
            - features: dict of extracted features
            - reason: explanation string
        """
        # Extract features for both methods
        features = self.extract_features(audio_path)

        if self.model is not None:
            # Use ML model
            prediction, score, confidence, reason = self.detect_with_ml(audio_path)

            # If low confidence, mark as uncertain
            if confidence < confidence_threshold:
                prediction = 'uncertain'
                reason = f'{reason},low_confidence'

            features['ml_confidence'] = confidence
            return prediction, score, features, reason
        else:
            # Fallback to heuristics
            score, reason = self.compute_trumpet_score_heuristic(features)
            prediction = 'trumpet' if score >= threshold else 'trombone'
            return prediction, score, features, reason


def load_model_for_worker(model_path: str):
    """Load model once per worker process."""
    global _MODEL_CACHE
    if model_path not in _MODEL_CACHE:
        try:
            with open(model_path, 'rb') as f:
                _MODEL_CACHE[model_path] = pickle.load(f)
        except Exception:
            _MODEL_CACHE[model_path] = None
    return _MODEL_CACHE.get(model_path)


def process_file(args) -> Dict:
    """Process a single file (for parallel execution)."""
    audio_path, label, model_path, confidence_threshold = args

    try:
        # Create detector with model if available
        detector = TrumpetTromboneDetector(model_path=model_path)
        prediction, score, features, reason = detector.detect(
            audio_path, confidence_threshold=confidence_threshold
        )

        return {
            'file': audio_path,
            'label': label,
            'prediction': prediction,
            'score': score,
            'correct': prediction == label if prediction != 'uncertain' else None,
            'confident': prediction != 'uncertain',
            'spectral_centroid': features.get('spectral_centroid', 0),
            'mean_pitch': features.get('mean_pitch', 0),
            'low_high_ratio': features.get('low_high_ratio', 0),
            'ml_confidence': features.get('ml_confidence', None),
            'reason': reason
        }
    except Exception as e:
        return {
            'file': audio_path,
            'label': label,
            'prediction': 'error',
            'score': 0.5,
            'correct': False,
            'confident': False,
            'error': str(e)
        }


def evaluate_on_manifest(manifest_path: str, output_path: str,
                         workers: int = 8, sample_size: int = None,
                         model_path: str = None, confidence_threshold: float = 0.7):
    """Evaluate detector on manifest data."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Treat None as "not muted" (dry trumpet) - only exclude explicitly muted
    dry_trumpets = [e for e in manifest
                    if e.get('sub_group') == 'trumpet' and e.get('is_muted') != True]
    trombones = [e for e in manifest if e.get('sub_group') == 'trombone']

    logging.info(f"Found {len(dry_trumpets)} dry trumpets, {len(trombones)} trombones")

    if model_path:
        logging.info(f"Using ML model: {model_path}")
        logging.info(f"Confidence threshold: {confidence_threshold}")
    else:
        logging.info("Using heuristic detection (no model)")

    if sample_size:
        import random
        random.seed(42)
        if len(dry_trumpets) > sample_size:
            dry_trumpets = random.sample(dry_trumpets, sample_size)
        if len(trombones) > sample_size:
            trombones = random.sample(trombones, sample_size)
        logging.info(f"Sampled to {len(dry_trumpets)} trumpets, {len(trombones)} trombones")

    tasks = []
    for e in dry_trumpets:
        tasks.append((e['audio_path'], 'trumpet', model_path, confidence_threshold))
    for e in trombones:
        tasks.append((e['audio_path'], 'trombone', model_path, confidence_threshold))

    logging.info(f"Processing {len(tasks)} files with {workers} workers...")

    results = []
    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, task): task for task in tasks}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)

            if (i + 1) % 100 == 0:
                elapsed = (datetime.now() - start_time).seconds
                logging.info(f"Processed {i+1}/{len(tasks)} ({elapsed}s elapsed)")

    # Analyze results
    trumpet_results = [r for r in results if r['label'] == 'trumpet']
    trombone_results = [r for r in results if r['label'] == 'trombone']

    # Confident predictions only
    trumpet_confident = [r for r in trumpet_results if r.get('confident', True)]
    trombone_confident = [r for r in trombone_results if r.get('confident', True)]

    trumpet_correct = sum(1 for r in trumpet_confident if r.get('correct'))
    trombone_correct = sum(1 for r in trombone_confident if r.get('correct'))

    trumpet_acc = trumpet_correct / len(trumpet_confident) * 100 if trumpet_confident else 0
    trombone_acc = trombone_correct / len(trombone_confident) * 100 if trombone_confident else 0

    total_confident = len(trumpet_confident) + len(trombone_confident)
    total_correct = trumpet_correct + trombone_correct
    total_acc = total_correct / total_confident * 100 if total_confident else 0

    logging.info("=" * 60)
    logging.info("RESULTS SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Trumpet: {trumpet_correct}/{len(trumpet_confident)} confident correct ({trumpet_acc:.1f}%)")
    logging.info(f"Trombone: {trombone_correct}/{len(trombone_confident)} confident correct ({trombone_acc:.1f}%)")
    logging.info(f"Overall confident: {total_correct}/{total_confident} ({total_acc:.1f}%)")

    uncertain_count = sum(1 for r in results if r.get('prediction') == 'uncertain')
    logging.info(f"\nUncertain/low-confidence: {uncertain_count}/{len(results)} ({uncertain_count/len(results)*100:.1f}%)")

    # Show misclassifications
    trumpet_misclass = [r for r in trumpet_confident if r.get('correct') == False]
    trombone_misclass = [r for r in trombone_confident if r.get('correct') == False]

    if trumpet_misclass:
        logging.info(f"\nTrumpets misclassified as trombone ({len(trumpet_misclass)}):")
        for r in sorted(trumpet_misclass, key=lambda x: x['score'])[:10]:
            conf = r.get('ml_confidence', 0)
            logging.info(f"  score={r['score']:.3f} conf={conf:.2f} cent={r['spectral_centroid']:.0f}Hz - {Path(r['file']).name}")

    if trombone_misclass:
        logging.info(f"\nTrombones misclassified as trumpet ({len(trombone_misclass)}):")
        for r in sorted(trombone_misclass, key=lambda x: -x['score'])[:10]:
            conf = r.get('ml_confidence', 0)
            logging.info(f"  score={r['score']:.3f} conf={conf:.2f} cent={r['spectral_centroid']:.0f}Hz - {Path(r['file']).name}")

    # Feature statistics
    logging.info("\n" + "=" * 60)
    logging.info("FEATURE STATISTICS")
    logging.info("=" * 60)

    trumpet_centroids = [r['spectral_centroid'] for r in trumpet_results if r.get('spectral_centroid')]
    trombone_centroids = [r['spectral_centroid'] for r in trombone_results if r.get('spectral_centroid')]

    if trumpet_centroids:
        logging.info(f"Trumpet centroid: mean={np.mean(trumpet_centroids):.0f}Hz, "
                    f"std={np.std(trumpet_centroids):.0f}Hz")
    if trombone_centroids:
        logging.info(f"Trombone centroid: mean={np.mean(trombone_centroids):.0f}Hz, "
                    f"std={np.std(trombone_centroids):.0f}Hz")

    # Save results
    output = {
        'summary': {
            'model_used': model_path is not None,
            'confidence_threshold': confidence_threshold,
            'trumpet_total': len(trumpet_results),
            'trumpet_confident': len(trumpet_confident),
            'trumpet_correct': trumpet_correct,
            'trumpet_accuracy': trumpet_acc,
            'trombone_total': len(trombone_results),
            'trombone_confident': len(trombone_confident),
            'trombone_correct': trombone_correct,
            'trombone_accuracy': trombone_acc,
            'overall_accuracy': total_acc,
            'uncertain_count': uncertain_count
        },
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logging.info(f"\nResults saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description='Trumpet vs Trombone Detector')
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--output', type=str, default='./trumpet_trombone_results.json')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (.pkl)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Confidence threshold for ML predictions (default: 0.7)')

    args = parser.parse_args()
    evaluate_on_manifest(args.manifest, args.output, args.workers, args.sample,
                         args.model, args.confidence)


if __name__ == '__main__':
    main()
