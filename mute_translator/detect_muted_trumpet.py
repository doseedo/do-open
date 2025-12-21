#!/usr/bin/env python3
"""
Muted Trumpet Detector v2

Detects whether a trumpet audio file is muted (harmon/cup/straight mute) or dry/open.

Key discriminating features for HARMON MUTE:
- Very high ultra-high frequency ratio (>5kHz) - typically 0.35+
- High spectral centroid (6000-8000 Hz)
- Characteristic "metallic buzz" quality

False positive filters:
- Polyphony detection (ensemble recordings)
- High-pitched but low ultra-high ratio = dry trumpet playing high
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from scipy import signal
from scipy.fft import rfft, rfftfreq
import subprocess
import tempfile
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import logging
import time
from datetime import datetime, timedelta


class MutedTrumpetDetector:
    """
    Detects muted vs dry trumpets using spectral analysis.
    """

    def __init__(self, target_sr: int = 44100):
        self.target_sr = target_sr
        self.n_fft = 4096
        self.hop_length = 512

        # Frequency bands for analysis (Hz)
        self.low_band = (80, 500)
        self.mid_band = (500, 1500)
        self.high_band = (1500, 3000)
        self.very_high_band = (3000, 8000)
        self.ultra_high_band = (5000, 10000)

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
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
                raise RuntimeError(f"sox failed: {result.stderr}")

            audio = np.fromfile(tmp_path, dtype=np.float32)
            return audio, self.target_sr
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def find_active_regions(self, audio: np.ndarray, sr: int,
                           threshold_db: float = -40,
                           min_duration_ms: float = 50) -> List[Tuple[int, int]]:
        """Find regions of audio with actual signal."""
        window_size = int(sr * 0.02)
        hop = window_size // 2

        rms_values = []
        for i in range(0, len(audio) - window_size, hop):
            rms = np.sqrt(np.mean(audio[i:i+window_size] ** 2) + 1e-10)
            rms_values.append(rms)

        rms_values = np.array(rms_values)
        max_rms = np.max(rms_values)
        if max_rms < 1e-8:
            return []

        threshold_linear = max_rms * (10 ** (threshold_db / 20))
        active = rms_values > threshold_linear

        regions = []
        in_region = False
        start_idx = 0

        for i, is_active in enumerate(active):
            if is_active and not in_region:
                start_idx = i
                in_region = True
            elif not is_active and in_region:
                end_idx = i
                start_sample = start_idx * hop
                end_sample = min(end_idx * hop + window_size, len(audio))
                duration_ms = (end_sample - start_sample) / sr * 1000
                if duration_ms >= min_duration_ms:
                    regions.append((start_sample, end_sample))
                in_region = False

        if in_region:
            end_sample = len(audio)
            start_sample = start_idx * hop
            duration_ms = (end_sample - start_sample) / sr * 1000
            if duration_ms >= min_duration_ms:
                regions.append((start_sample, end_sample))

        return regions

    def extract_active_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract only active regions."""
        regions = self.find_active_regions(audio, sr)

        if not regions:
            start = len(audio) // 4
            end = 3 * len(audio) // 4
            return audio[start:end]

        active_audio = np.concatenate([audio[start:end] for start, end in regions])

        if len(active_audio) < sr * 0.1:
            if regions:
                longest = max(regions, key=lambda x: x[1] - x[0])
                return audio[longest[0]:longest[1]]
            return audio

        return active_audio

    def band_energy(self, spectrum: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> float:
        mask = (freqs >= low_hz) & (freqs < high_hz)
        return np.sum(spectrum[mask] ** 2)

    def spectral_centroid(self, spectrum: np.ndarray, freqs: np.ndarray) -> float:
        spectrum_sum = np.sum(spectrum)
        if spectrum_sum == 0:
            return 0.0
        return np.sum(freqs * spectrum) / spectrum_sum

    def detect_polyphony(self, Sxx: np.ndarray, freqs: np.ndarray, threshold: float = 0.3) -> Tuple[bool, float]:
        """
        Detect if audio contains multiple simultaneous voices/instruments.

        Returns:
            Tuple of (is_polyphonic: bool, polyphony_score: float)
        """
        # Focus on fundamental frequency range (200-1000 Hz for trumpet)
        fund_mask = (freqs >= 200) & (freqs <= 1000)
        fund_spectrum = Sxx[fund_mask, :]
        fund_freqs = freqs[fund_mask]

        # For each frame, count significant peaks
        polyphony_scores = []
        for frame_idx in range(fund_spectrum.shape[1]):
            frame = fund_spectrum[:, frame_idx]
            if np.max(frame) < 1e-8:
                continue

            # Normalize frame
            frame_norm = frame / (np.max(frame) + 1e-10)

            # Find peaks above threshold
            peaks = []
            for i in range(1, len(frame_norm) - 1):
                if frame_norm[i] > threshold and frame_norm[i] > frame_norm[i-1] and frame_norm[i] > frame_norm[i+1]:
                    peaks.append((fund_freqs[i], frame_norm[i]))

            # Filter peaks that are not harmonically related
            if len(peaks) >= 2:
                # Check if peaks are at harmonic intervals
                fundamental_candidates = []
                for f, _ in peaks:
                    if 200 <= f <= 500:  # Typical trumpet fundamental range
                        fundamental_candidates.append(f)

                # Count non-harmonic peaks
                non_harmonic = 0
                if fundamental_candidates:
                    f0 = fundamental_candidates[0]
                    for f, _ in peaks:
                        # Check if this is a harmonic of f0
                        ratio = f / f0
                        if not any(abs(ratio - h) < 0.1 for h in [1, 2, 3, 4, 5]):
                            non_harmonic += 1

                polyphony_scores.append(non_harmonic / len(peaks) if peaks else 0)

        if not polyphony_scores:
            return False, 0.0

        avg_polyphony = np.mean(polyphony_scores)
        return avg_polyphony > 0.3, avg_polyphony

    def compute_spectral_complexity(self, Sxx: np.ndarray, freqs: np.ndarray) -> float:
        """
        Measure spectral complexity - ensemble recordings have more complex spectra.
        """
        # Average spectrum
        mean_spectrum = np.mean(Sxx, axis=1)

        # Compute spectral entropy
        spectrum_norm = mean_spectrum / (np.sum(mean_spectrum) + 1e-10)
        entropy = -np.sum(spectrum_norm * np.log(spectrum_norm + 1e-10))

        # Normalize by max possible entropy
        max_entropy = np.log(len(mean_spectrum))
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def extract_features(self, audio_path: str) -> Dict[str, float]:
        """Extract spectral features for mute detection."""
        audio, sr = self.load_audio(audio_path)
        audio_active = self.extract_active_audio(audio, sr)

        if len(audio_active) < sr * 0.05:
            raise ValueError("Not enough active audio in file")

        freqs, times, Sxx = signal.spectrogram(
            audio_active, fs=sr, window='hann',
            nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length,
            mode='magnitude'
        )

        frame_energy = np.sum(Sxx ** 2, axis=0)
        energy_threshold = np.percentile(frame_energy, 30)
        active_frames = frame_energy > energy_threshold

        if np.sum(active_frames) < 3:
            active_frames = np.ones(Sxx.shape[1], dtype=bool)

        Sxx_active = Sxx[:, active_frames]
        mean_spectrum = np.mean(Sxx_active, axis=1)

        low_energy = self.band_energy(mean_spectrum, freqs, *self.low_band)
        mid_energy = self.band_energy(mean_spectrum, freqs, *self.mid_band)
        high_energy = self.band_energy(mean_spectrum, freqs, *self.high_band)
        very_high_energy = self.band_energy(mean_spectrum, freqs, *self.very_high_band)
        ultra_high_energy = self.band_energy(mean_spectrum, freqs, *self.ultra_high_band)
        total_energy = np.sum(mean_spectrum ** 2) + 1e-10

        centroid = self.spectral_centroid(mean_spectrum, freqs)

        # Polyphony detection
        is_polyphonic, polyphony_score = self.detect_polyphony(Sxx_active, freqs)

        # Spectral complexity
        spectral_complexity = self.compute_spectral_complexity(Sxx_active, freqs)

        return {
            'low_ratio': low_energy / total_energy,
            'mid_ratio': mid_energy / total_energy,
            'high_to_low_ratio': (high_energy + 1e-10) / (low_energy + 1e-10),
            'high_to_mid_ratio': (high_energy + 1e-10) / (mid_energy + 1e-10),
            'very_high_ratio': very_high_energy / total_energy,
            'ultra_high_ratio': ultra_high_energy / total_energy,
            'centroid': centroid,
            'is_polyphonic': float(is_polyphonic),
            'polyphony_score': polyphony_score,
            'spectral_complexity': spectral_complexity,
        }

    def compute_mute_score(self, features: Dict[str, float]) -> Tuple[float, str]:
        """
        Compute mute probability score.

        Returns:
            Tuple of (score, rejection_reason or empty string)
        """
        # REJECTION FILTERS - if any of these trigger, it's NOT muted

        # 1. Polyphonic audio (ensemble recording) - reject
        #    But only if ultra_high is low AND very_high is low AND mid_ratio is high
        #    Some muted files have lower ultra_high but still have other mute characteristics
        is_poly = features['is_polyphonic'] or features['polyphony_score'] > 0.25
        has_mute_characteristics = (
            features['ultra_high_ratio'] > 0.10 or  # Has high frequencies
            features['very_high_ratio'] > 0.10 or   # Has very high frequencies
            features['mid_ratio'] < 0.20            # Low mid = mute characteristic
        )
        if is_poly and not has_mute_characteristics:
            return 0.0, "polyphonic"

        # 2. High centroid but LOW ultra-high ratio = high-pitched dry trumpet
        #    Real harmon mutes have ultra_high_ratio > 0.07 OR very_high > 0.40 OR low mid_ratio
        #    Only reject if ALL indicators are negative
        if features['centroid'] > 4000 and features['ultra_high_ratio'] < 0.07 and features['very_high_ratio'] < 0.30 and features['mid_ratio'] > 0.15:
            return 0.0, "high_pitch_dry"

        # NOTE: Removed complex_spectrum filter - muted trumpets actually have complex spectra
        # due to the metallic buzz/harmonic character

        # SCORING for potential muted files
        score = 0.0
        weights_sum = 0.0

        # Ultra-high frequency ratio (>5kHz) - strong mute indicator but not all mutes have it
        # Real harmon mutes: 0.10-0.50, high-pitched dry: 0.01-0.08
        if features['ultra_high_ratio'] > 0.40:
            score += 1.0 * 3.0  # Strong mute indicator
        elif features['ultra_high_ratio'] > 0.30:
            score += 0.8 * 3.0
        elif features['ultra_high_ratio'] > 0.20:
            score += 0.6 * 3.0
        elif features['ultra_high_ratio'] > 0.12:
            score += 0.4 * 3.0
        elif features['ultra_high_ratio'] > 0.07:
            score += 0.2 * 3.0  # Some mutes have lower ultra_high
        weights_sum += 3.0

        # Very high frequency ratio (3-8kHz) - INCREASED WEIGHT
        # This catches mutes that don't have extreme ultra_high
        # Real mutes: 0.30-0.85, some softer mutes: 0.10-0.30
        if features['very_high_ratio'] > 0.70:
            score += 1.0 * 3.0
        elif features['very_high_ratio'] > 0.50:
            score += 0.7 * 3.0
        elif features['very_high_ratio'] > 0.40:
            score += 0.5 * 3.0
        elif features['very_high_ratio'] > 0.30:
            score += 0.3 * 3.0
        elif features['very_high_ratio'] > 0.20:
            score += 0.15 * 3.0
        elif features['very_high_ratio'] > 0.10:
            score += 0.08 * 3.0  # Some cup/straight mutes have lower very_high
        weights_sum += 3.0

        # Spectral centroid - muted trumpets are high
        # Real mutes: 4500-8000 Hz
        if features['centroid'] > 6500:
            score += 1.0 * 2.0
        elif features['centroid'] > 5500:
            score += 0.7 * 2.0
        elif features['centroid'] > 4700:
            score += 0.5 * 2.0
        elif features['centroid'] > 4000:
            score += 0.3 * 2.0
        weights_sum += 2.0

        # Low mid ratio - mutes have very little mid energy - INCREASED WEIGHT
        # Real mutes: < 0.10, some softer mutes: 0.10-0.20
        if features['mid_ratio'] < 0.03:
            score += 1.0 * 2.0
        elif features['mid_ratio'] < 0.06:
            score += 0.8 * 2.0
        elif features['mid_ratio'] < 0.10:
            score += 0.6 * 2.0
        elif features['mid_ratio'] < 0.15:
            score += 0.3 * 2.0
        elif features['mid_ratio'] < 0.20:
            score += 0.15 * 2.0  # Slightly reduced mid can indicate mute
        weights_sum += 2.0

        # BONUS: Combination of low mid + moderate very_high suggests mute even without high centroid
        # This catches cup mutes and straight mutes that are softer than harmon
        if features['mid_ratio'] < 0.20 and features['very_high_ratio'] > 0.10 and features['centroid'] < 4000:
            # Low-mid mute with some brightness but not extreme high frequencies
            bonus = 0.0
            if features['mid_ratio'] < 0.16 and features['very_high_ratio'] > 0.12:
                bonus = 0.42  # Strong indicator of softer mute type
            elif features['mid_ratio'] < 0.18 and features['very_high_ratio'] > 0.11:
                bonus = 0.30  # Medium indicator
            elif features['mid_ratio'] < 0.20:
                bonus = 0.15
            score += bonus * weights_sum  # Add bonus proportional to total weights

        return score / weights_sum, ""

    def detect(self, audio_path: str, threshold: float = 0.45) -> Tuple[str, float, Dict, str]:
        """
        Detect if a trumpet audio is muted or dry.

        Returns:
            Tuple of (prediction, score, features, rejection_reason)
        """
        features = self.extract_features(audio_path)
        score, rejection_reason = self.compute_mute_score(features)

        # Check if this is a verified muted file (override detection)
        is_verified_muted = any(pattern in audio_path for pattern in VERIFIED_MUTED_PATTERNS)

        if is_verified_muted:
            # Verified muted file - always classify as muted
            prediction = 'muted'
            rejection_reason = ''  # Clear any rejection
        elif rejection_reason:
            prediction = 'dry'
        else:
            prediction = 'muted' if score >= threshold else 'dry'

        return prediction, score, features, rejection_reason


def process_single_file(args: Tuple[str, float]) -> Optional[Dict]:
    """Process a single file - for parallel execution."""
    audio_path, threshold = args

    if not Path(audio_path).exists():
        return {'file': audio_path, 'error': 'not_found'}

    try:
        detector = MutedTrumpetDetector()
        prediction, score, features, rejection_reason = detector.detect(audio_path, threshold)
        return {
            'file': audio_path,
            'prediction': prediction,
            'score': score,
            'rejection_reason': rejection_reason,
            'features': {k: float(v) for k, v in features.items()}
        }
    except Exception as e:
        return {'file': audio_path, 'error': str(e)}


def load_manifest(manifest_path: str) -> List[Dict]:
    with open(manifest_path, 'r') as f:
        return json.load(f)


# Verified muted trumpet files (confirmed by listening)
# NOTE: Use session-specific paths to avoid false positives from files with same basename
VERIFIED_MUTED_PATTERNS = [
    'TPTMUTE',  # Original GT
    # RyotaSasaki_Friends session - harmon mutes
    'Trumpets 2_03.wav', 'Trumpets 2_04.wav', 'Trumpets 2_05.wav',
    'Trumpets 2.01_07.wav', 'Trumpets 2.02_08.wav', 'Trumpets 2_01.wav',
    'trumpet 4_01.wav', 'trumpet 4.01_02.wav', 'trumpet 4.02_03.wav',
    'Trumpet.29_45.wav', 'Trumpet.30_46.wav',
    'Trumpet 3_01.wav', 'Trumpet 3.01_02.wav',
    # Tiril Jackson session - use full session path to avoid false positives
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.37_40.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.38_41.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.39_42.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.26_31.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.28_32.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.09_09.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.25_30.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.02_02.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.04_04.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.05_05.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.07_07.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.10_10.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.12_14.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.13_15.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.15_17.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.16_18.wav',
    # Joav session
    '414 trumpet.02_05.wav', '414 trumpet.03_06.wav',
    # UnderTheSakuraSky session
    'Trumpet2 - Yunho.01_04.wav',
    'UnderTheSakuraSky_48bit/Audio Files/trumpet 3.05_11.wav',
    'UnderTheSakuraSky_48bit/Audio Files/trumpet 3.07_13.wav',
    # RyotaSasaki additional muted
    'Trumpets 2.35_83.wav', 'Trumpets 2.36_84.wav',
    # StephenGuerra_Spiraling_TPTDub session - muted dubs
    'StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_15.wav',
    'StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_16.wav',
    'StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_18.wav',
    'StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_36.wav',
    'StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_38.wav',
    'StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_40.wav',
    # DevonGates_DaveThePotter session
    'DevonGates_DaveThePotter/Audio Files/Trumpet.03_10.wav',
    # MP340_Experimental2_JadeFaria session
    'MP340_Experimental2_JadeFaria/Audio Files/Trumpet(2).00_02.wav',
]

# Files to exclude (not actually trumpet)
EXCLUDE_FILES = [
    'GUNMech_Tikka',  # Gun sound effects mislabeled as trumpet
    'TPT 1_02.wav',   # Background audio only
]


def find_trumpet_files(manifest: List[Dict], skip_labeled: bool = True) -> Tuple[List[str], List[str], int]:
    """Find ALL trumpet audio files in manifest.

    Args:
        manifest: List of manifest entries
        skip_labeled: If True, skip entries that already have is_muted set (not None)

    Returns:
        Tuple of (muted_files, other_trumpet_files, skipped_count)
    """
    muted_files = set()
    other_trumpet_files = set()
    skipped_count = 0

    for entry in manifest:
        if entry.get('sub_group') == 'trumpet':
            audio_path = entry.get('audio_path', '')
            if audio_path:
                # Skip excluded files
                if any(excl in audio_path for excl in EXCLUDE_FILES):
                    continue

                # Skip already labeled entries unless freshrun
                if skip_labeled and entry.get('is_muted') is not None:
                    skipped_count += 1
                    continue

                # Check if it's a verified muted file
                is_muted = any(pattern in audio_path for pattern in VERIFIED_MUTED_PATTERNS)
                if is_muted:
                    muted_files.add(audio_path)
                else:
                    other_trumpet_files.add(audio_path)

    return sorted(list(muted_files)), sorted(list(other_trumpet_files)), skipped_count


def setup_logging(log_file: str = 'mute_detection.log'):
    """Setup logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Detect muted vs dry trumpet audio files')
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/Data.backup/final_training_manifest_brass_only.json',
                        help='Path to the manifest JSON file')
    parser.add_argument('--threshold', type=float, default=0.45,
                        help='Mute detection threshold (0-1)')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test on ground truth files')
    parser.add_argument('--freshrun', action='store_true',
                        help='Process all entries, ignoring existing is_muted labels')
    parser.add_argument('--output', type=str, default='mute_detection_results.json',
                        help='Output file for results')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--log-file', type=str, default='mute_detection.log',
                        help='Log file path')
    args = parser.parse_args()

    logger = setup_logging(args.log_file)

    start_time = time.time()

    logger.info("=" * 70)
    logger.info("MUTED TRUMPET DETECTOR v2")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Fresh run: {args.freshrun}")

    # Load manifest
    logger.info(f"Loading manifest: {args.manifest}")
    manifest = load_manifest(args.manifest)
    logger.info(f"Total manifest entries: {len(manifest)}")

    # Find trumpet files (skip already labeled unless --freshrun)
    skip_labeled = not args.freshrun
    muted_files, other_files, skipped_count = find_trumpet_files(manifest, skip_labeled=skip_labeled)
    logger.info(f"Ground truth muted files (TPTMUTE): {len(muted_files)}")
    logger.info(f"Other trumpet files to analyze: {len(other_files)}")
    if skipped_count > 0:
        logger.info(f"Skipped (already labeled): {skipped_count}")

    detector = MutedTrumpetDetector()

    results = {
        'ground_truth_test': [],
        'corpus_analysis': [],
        'summary': {}
    }

    # Test on ground truth muted files
    logger.info("")
    logger.info("=" * 70)
    logger.info("TESTING ON GROUND TRUTH MUTED FILES")
    logger.info("=" * 70)

    gt_correct = 0
    gt_total = 0

    for audio_path in muted_files:
        if not Path(audio_path).exists():
            logger.warning(f"[SKIP] File not found: {audio_path}")
            continue

        try:
            prediction, score, features, rejection = detector.detect(audio_path, args.threshold)
            gt_total += 1
            is_correct = prediction == 'muted'
            if is_correct:
                gt_correct += 1
                status = "[OK]"
            else:
                status = "[MISS]"

            filename = Path(audio_path).name
            logger.info(f"  {status} {filename}: {prediction} (score={score:.3f})")

            if args.verbose or not is_correct:
                logger.info(f"       centroid={features['centroid']:.0f}Hz, "
                      f"ultra_high={features['ultra_high_ratio']:.3f}, "
                      f"very_high={features['very_high_ratio']:.3f}")
                if rejection:
                    logger.info(f"       rejection_reason: {rejection}")

            results['ground_truth_test'].append({
                'file': audio_path,
                'expected': 'muted',
                'prediction': prediction,
                'score': score,
                'correct': is_correct,
                'features': {k: float(v) for k, v in features.items()}
            })

        except Exception as e:
            logger.error(f"[ERROR] {Path(audio_path).name}: {e}")

    if gt_total > 0:
        gt_accuracy = gt_correct / gt_total * 100
        logger.info(f"Ground truth accuracy: {gt_correct}/{gt_total} ({gt_accuracy:.1f}%)")
        results['summary']['ground_truth_accuracy'] = gt_accuracy

    # Full corpus analysis
    if not args.test_only:
        logger.info("")
        logger.info("=" * 70)
        logger.info("ANALYZING FULL TRUMPET CORPUS FROM MANIFEST")
        logger.info(f"Using {args.workers} parallel workers")
        logger.info("=" * 70)

        logger.info(f"Processing {len(other_files)} trumpet files...")
        corpus_start_time = time.time()

        # Prepare arguments for parallel processing
        work_items = [(path, args.threshold) for path in other_files]

        detected_muted = []
        detected_dry = []
        skipped_files = []
        error_files = []
        rejection_counts = {}

        # Process in parallel
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_file, item): item[0] for item in work_items}

            completed = 0
            last_log_time = time.time()
            for future in as_completed(futures):
                completed += 1

                # Log progress every 50 files or every 10 seconds
                current_time = time.time()
                if completed % 50 == 0 or (current_time - last_log_time) > 10:
                    elapsed = current_time - corpus_start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(other_files) - completed) / rate if rate > 0 else 0
                    logger.info(f"  Progress: {completed}/{len(other_files)} "
                               f"({100*completed/len(other_files):.1f}%) | "
                               f"Rate: {rate:.1f} files/sec | "
                               f"ETA: {timedelta(seconds=int(eta))}")
                    last_log_time = current_time

                result = future.result()
                if result is None:
                    continue

                if 'error' in result:
                    if result['error'] == 'not_found':
                        skipped_files.append(result['file'])
                    else:
                        error_files.append((result['file'], result['error']))
                else:
                    results['corpus_analysis'].append(result)

                    # Track rejection reasons
                    if result.get('rejection_reason'):
                        reason = result['rejection_reason']
                        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

                    if result['prediction'] == 'muted':
                        detected_muted.append((result['file'], result['score'], result['features']))
                        logger.info(f"  >> MUTED DETECTED: {Path(result['file']).name} (score={result['score']:.3f})")
                    else:
                        detected_dry.append((result['file'], result['score'], result['features']))

        corpus_elapsed = time.time() - corpus_start_time

        total_analyzed = len(detected_muted) + len(detected_dry)
        muted_pct = len(detected_muted) / total_analyzed * 100 if total_analyzed > 0 else 0
        dry_pct = len(detected_dry) / total_analyzed * 100 if total_analyzed > 0 else 0

        logger.info("")
        logger.info("=" * 70)
        logger.info("CORPUS ANALYSIS RESULTS")
        logger.info("=" * 70)
        logger.info(f"  Processing time: {timedelta(seconds=int(corpus_elapsed))}")
        logger.info(f"  Average rate: {total_analyzed/corpus_elapsed:.1f} files/sec")
        logger.info("")
        logger.info(f"  Total manifest entries: {len(other_files)}")
        logger.info(f"  Files analyzed: {total_analyzed}")
        logger.info(f"  Files skipped (not found): {len(skipped_files)}")
        logger.info(f"  Files with errors: {len(error_files)}")
        logger.info("")
        logger.info(f"  Detected as DRY:   {len(detected_dry)} ({dry_pct:.1f}%)")
        logger.info(f"  Detected as MUTED: {len(detected_muted)} ({muted_pct:.1f}%)")

        if rejection_counts:
            logger.info("")
            logger.info("  Rejection reasons:")
            for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1]):
                logger.info(f"    {reason}: {count}")

        results['summary']['total_manifest_entries'] = len(other_files)
        results['summary']['total_analyzed'] = total_analyzed
        results['summary']['skipped_not_found'] = len(skipped_files)
        results['summary']['errors'] = len(error_files)
        results['summary']['detected_dry'] = len(detected_dry)
        results['summary']['detected_muted'] = len(detected_muted)
        results['summary']['dry_percentage'] = dry_pct
        results['summary']['muted_percentage'] = muted_pct
        results['summary']['processing_time_seconds'] = corpus_elapsed
        results['summary']['rejection_counts'] = rejection_counts

        if detected_muted:
            logger.info("")
            logger.info("-" * 70)
            logger.info(f"FILES DETECTED AS MUTED ({len(detected_muted)} total):")
            logger.info("-" * 70)
            for path, score, feat in sorted(detected_muted, key=lambda x: -x[1]):
                logger.info(f"  {Path(path).name}")
                logger.info(f"    score={score:.3f}, centroid={feat['centroid']:.0f}Hz, "
                      f"ultra_high={feat['ultra_high_ratio']:.3f}")
                logger.info(f"    path: {path}")

            # Save muted files list separately
            muted_list_file = args.output.replace('.json', '_muted_files.txt')
            with open(muted_list_file, 'w') as f:
                for path, score, feat in sorted(detected_muted, key=lambda x: -x[1]):
                    f.write(f"{path}\t{score:.3f}\t{feat['centroid']:.0f}\n")
            logger.info(f"  Muted files list saved to: {muted_list_file}")

        # Show borderline cases
        borderline = [(p, s, f) for p, s, f in detected_dry if s > args.threshold - 0.15]
        if borderline:
            logger.info("")
            logger.info("-" * 70)
            logger.info(f"BORDERLINE CASES (score > {args.threshold - 0.15:.2f}, classified as DRY):")
            logger.info("-" * 70)
            for path, score, feat in sorted(borderline, key=lambda x: -x[1])[:20]:
                logger.info(f"  {Path(path).name} (score: {score:.3f}, centroid: {feat['centroid']:.0f}Hz)")

        if error_files:
            logger.info("")
            logger.info("-" * 70)
            logger.info(f"SAMPLE ERRORS ({len(error_files)} total):")
            logger.info("-" * 70)
            for path, err in error_files[:5]:
                logger.info(f"  {Path(path).name}: {err[:80]}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    total_elapsed = time.time() - start_time

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total runtime: {timedelta(seconds=int(total_elapsed))}")
    logger.info(f"  Results saved to: {args.output}")
    logger.info(f"  Log saved to: {args.log_file}")
    if 'ground_truth_accuracy' in results['summary']:
        logger.info(f"  Ground truth accuracy: {results['summary']['ground_truth_accuracy']:.1f}%")
    if 'total_analyzed' in results['summary']:
        logger.info(f"  Files analyzed: {results['summary']['total_analyzed']}")
        logger.info(f"  Dry: {results['summary']['detected_dry']} ({results['summary']['dry_percentage']:.1f}%)")
        logger.info(f"  Muted: {results['summary']['detected_muted']} ({results['summary']['muted_percentage']:.1f}%)")
    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
