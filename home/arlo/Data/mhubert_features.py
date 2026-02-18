#!/usr/bin/env python3
"""
mHubert Feature Extractor (mhubert_features.py)

Extracts mHubert acoustic features for speaker embeddings and auxiliary losses.

Features:
- mHuBERT-147 (utter-project/mHuBERT-147) feature extraction
- Global speaker embeddings via temporal pooling
- Frame-level features for auxiliary supervision
- Compatible with existing vocal processing pipeline

Usage:
    python mhubert_features.py --process_all
    python mhubert_features.py --file "path/to/vocal.wav"
"""

import os
import sys
import json
import argparse
import multiprocessing
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from datetime import datetime

# mHubert model
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
    print("✅ transformers available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠ transformers not available. Install with: pip install transformers")

# Configuration
OUTPUT_DIR = Path("/mnt/msdd/mhubert_features")
SAMPLE_RATE = 16000  # mHubert requires 16kHz
DCAE_SR = 44100
DCAE_HOP = 4096
SLOW_HZ = DCAE_SR / DCAE_HOP  # ~10.77 Hz
MHUBERT_FEATURE_DIM = 768  # mHubert-base output dimension


class MHubertFeatureExtractor:
    """Extracts mHubert acoustic features for speaker embeddings and aux losses"""

    def __init__(self, output_dir: Path = OUTPUT_DIR, device: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model loading
        self.mhubert = None
        self._model_loaded = False

    def _load_model_if_needed(self):
        """Load mHubert model lazily"""
        if self._model_loaded:
            return True

        if not TRANSFORMERS_AVAILABLE:
            print("❌ transformers not available")
            return False

        try:
            print("🔧 Loading mHubert model...")
            # Use the correct model: slprl/mhubert-base-25hz or utter-project/mHuBERT-147
            self.mhubert = transformers.HubertModel.from_pretrained("utter-project/mHuBERT-147")
            self.mhubert = self.mhubert.to(self.device)
            self.mhubert.eval()  # Set to evaluation mode
            self._model_loaded = True
            print(f"✅ mHubert loaded on {self.device}")
            return True
        except Exception as e:
            print(f"❌ mHubert loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, float]:
        """
        Load and preprocess audio for mHubert.

        Args:
            audio_path: Path to audio file

        Returns:
            audio_tensor: [1, T] tensor at 16kHz
            original_duration: Duration in seconds
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Calculate original duration
        original_duration = waveform.shape[1] / SAMPLE_RATE

        return waveform, original_duration

    def extract_features(self, audio_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract mHubert features from audio.

        Args:
            audio_tensor: [1, T] waveform at 16kHz

        Returns:
            frame_features: [T_frames, 768] temporal features
            speaker_embed: [768] global speaker embedding (mean pooled)
        """
        if not self._load_model_if_needed():
            raise RuntimeError("mHubert model not available")

        # Move to device
        audio_tensor = audio_tensor.to(self.device)

        # Extract features
        with torch.no_grad():
            outputs = self.mhubert(audio_tensor)
            features = outputs.last_hidden_state  # [B, T, 768]

        # Move back to CPU for saving
        features = features.cpu()

        # Get frame-level features [T_frames, 768]
        frame_features = features.squeeze(0).numpy()

        # Average pool for global speaker embedding [768]
        speaker_embed = features.mean(dim=1).squeeze(0).numpy()

        return frame_features, speaker_embed

    def align_features_to_slow_frames(
        self,
        frame_features: np.ndarray,
        audio_duration: float
    ) -> np.ndarray:
        """
        Align mHubert features to DCAE slow frame rate (~10.77 Hz).

        Args:
            frame_features: [T_mhubert, 768] mHubert features
            audio_duration: Audio duration in seconds

        Returns:
            aligned_features: [T_slow, 768] features aligned to DCAE rate
        """
        T_mhubert = frame_features.shape[0]
        T_slow = int(np.ceil(audio_duration * SLOW_HZ))

        # mHubert frame rate (typically 50 Hz for HuBERT models)
        mhubert_hz = T_mhubert / audio_duration

        # Interpolate to match DCAE slow frame rate
        # Simple nearest neighbor for now (could use linear interpolation)
        aligned_features = np.zeros((T_slow, MHUBERT_FEATURE_DIM), dtype=np.float32)

        for i in range(T_slow):
            # Find corresponding mHubert frame
            slow_time = i / SLOW_HZ
            mhubert_idx = int(slow_time * mhubert_hz)
            mhubert_idx = min(mhubert_idx, T_mhubert - 1)
            aligned_features[i] = frame_features[mhubert_idx]

        return aligned_features

    def process_vocal_file(self, audio_path: str, skip_existing: bool = True) -> Dict[str, Any]:
        """
        Process a vocal file and extract mHubert features.

        Args:
            audio_path: Path to vocal audio file
            skip_existing: If True, skip files that already have mHubert features

        Returns:
            Dictionary with paths to saved features
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            print(f"⚠ File not found: {audio_path}")
            return None

        # Check if already processed
        stem = audio_path.stem
        output_subdir = self.output_dir / stem
        tensor_path = output_subdir / f"{stem}_mhubert_features.pt"
        json_path = output_subdir / f"{stem}_mhubert_metadata.json"

        if skip_existing and tensor_path.exists() and json_path.exists():
            # print(f"  ⏭ Skipping (already processed): {stem}")
            return {
                'mhubert_features': tensor_path,
                'mhubert_metadata': json_path,
                'status': 'skipped'
            }

        try:
            # Preprocess audio
            audio_tensor, duration = self.preprocess_audio(str(audio_path))

            # Extract features
            frame_features, speaker_embed = self.extract_features(audio_tensor)

            # Align to slow frame rate
            aligned_features = self.align_features_to_slow_frames(frame_features, duration)

            # Prepare output directory (reuse paths from skip check)
            output_subdir.mkdir(exist_ok=True)

            # Save features as tensors
            torch.save({
                'frame_features': torch.from_numpy(frame_features).float(),  # [T_mhubert, 768]
                'speaker_embed': torch.from_numpy(speaker_embed).float(),  # [768]
                'aligned_features': torch.from_numpy(aligned_features).float(),  # [T_slow, 768]
                'audio_duration': duration,
                'mhubert_frame_count': frame_features.shape[0],
                'slow_frame_count': aligned_features.shape[0],
            }, tensor_path)

            # Save metadata as JSON
            metadata = {
                'audio_path': str(audio_path),
                'audio_duration': duration,
                'mhubert_frames': int(frame_features.shape[0]),
                'slow_frames': int(aligned_features.shape[0]),
                'speaker_embed_shape': list(speaker_embed.shape),
                'feature_dim': MHUBERT_FEATURE_DIM,
                'processing_metadata': {
                    'processor': 'MHubertFeatureExtractor',
                    'model': 'utter-project/mHuBERT-147',
                    'device': self.device,
                    'timestamp': datetime.now().isoformat()
                }
            }

            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"  ✅ Saved mHubert features: {frame_features.shape[0]} frames, speaker_embed: {speaker_embed.shape}")

            return {
                'mhubert_features': tensor_path,
                'mhubert_metadata': json_path,
                'speaker_embed_shape': speaker_embed.shape,
                'frame_count': frame_features.shape[0],
                'status': 'processed'
            }

        except Exception as e:
            print(f"❌ Error processing {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return None


def process_file_wrapper(args):
    """Wrapper for multiprocessing"""
    audio_path, device_id, skip_existing = args

    # Set device for this worker
    device = f"cuda:{device_id}" if torch.cuda.is_available() and device_id is not None else "cpu"
    processor = MHubertFeatureExtractor(device=device)

    try:
        result = processor.process_vocal_file(audio_path, skip_existing=skip_existing)
        return (audio_path, result)
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return (audio_path, None)


def main():
    parser = argparse.ArgumentParser(description="mHubert feature extractor")
    parser.add_argument('--file', type=str, help='Single audio file to process')
    parser.add_argument('--process_all', action='store_true', help='Process all vocal files')
    parser.add_argument('--vocal_list', type=str,
                       default='/home/arlo/Data/categorized_instrument_paths_subcats_lists/voice/all.txt',
                       help='Path to vocal file list')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='Comma-separated GPU IDs for multi-GPU processing (e.g., "0,1,2")')
    parser.add_argument('--no_skip', action='store_true',
                       help='Reprocess files even if they already have mHubert features')

    args = parser.parse_args()
    skip_existing = not args.no_skip

    # Parse GPU IDs if provided
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]

    if args.file:
        # Process single file
        print(f"Processing: {args.file}")
        processor = MHubertFeatureExtractor(output_dir=Path(args.output_dir), device=args.device)
        result = processor.process_vocal_file(args.file, skip_existing=skip_existing)
        if result:
            if result.get('status') == 'skipped':
                print(f"⏭  Already processed (skipped): {args.file}")
            else:
                print(f"✅ Success: {result}")
        else:
            print(f"❌ Failed to process {args.file}")

    elif args.process_all:
        # Process all files from vocal list
        if not Path(args.vocal_list).exists():
            print(f"❌ Vocal list not found: {args.vocal_list}")
            return

        with open(args.vocal_list, 'r') as f:
            vocal_files = [line.strip() for line in f if line.strip()]

        print(f"Found {len(vocal_files)} vocal files to process")

        # Prepare arguments for multiprocessing
        if gpu_ids and len(gpu_ids) > 0:
            # Distribute files across GPUs
            tasks = []
            for i, f in enumerate(vocal_files):
                gpu_id = gpu_ids[i % len(gpu_ids)]
                tasks.append((f, gpu_id, skip_existing))
        else:
            tasks = [(f, None, skip_existing) for f in vocal_files]

        # Process with multiprocessing
        if args.workers > 1:
            with multiprocessing.Pool(processes=args.workers) as pool:
                results = list(tqdm(
                    pool.imap(process_file_wrapper, tasks),
                    total=len(tasks),
                    desc="Processing vocals"
                ))
        else:
            # Single-threaded processing
            results = []
            for task in tqdm(tasks, desc="Processing vocals"):
                results.append(process_file_wrapper(task))

        # Summary
        successful = sum(1 for _, r in results if r is not None and r.get('status') == 'processed')
        skipped = sum(1 for _, r in results if r is not None and r.get('status') == 'skipped')
        failed = sum(1 for _, r in results if r is None)

        print(f"\n{'='*60}")
        print(f"✅ Processed: {successful}/{len(vocal_files)}")
        print(f"⏭  Skipped (already exists): {skipped}/{len(vocal_files)}")
        print(f"❌ Failed: {failed}/{len(vocal_files)}")
        print(f"{'='*60}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
