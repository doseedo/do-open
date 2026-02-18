#!/usr/bin/env python3
"""
Full Mix Analysis Pipeline V3

Pipeline stages:
1. Mix Detector - Filter to files that are actually mixes (have "mix" in filename or multi-instrument)
2. Demucs Separation - htdemucs_6s separates into: vocals, drums, bass, guitar, piano, other
3. V3 Other Classifier - Runs on "other" stem to detect: brass, strings, winds

Output: Temporal annotations for all 8 instrument categories:
  - From Demucs: vocals, drums, bass, guitar, piano
  - From V3: brass, strings, winds

Usage:
  python3 run_mix_pipeline_v3.py --limit 100
  python3 run_mix_pipeline_v3.py --input-file /path/to/mix.wav  # Single file
"""

import argparse
import json
import subprocess
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import os
import sys
import tempfile
import shutil

# Configuration
MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/master_manifest.json")
V3_MODEL_PATH = Path("/home/arlo/Data/mix_classifier/mix_classifier_v3.pt")
OUTPUT_DIR = Path("/home/arlo/Data/mix_classifier")
CHECKPOINT_DIR = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c"

# Demucs stems
DEMUCS_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]

# Latent params
FRAMES_PER_SEC = 10.77  # ACE-Step latent frame rate
MIN_SAMPLES = int(1 * 48000)
MAX_SAMPLES = int(60 * 48000)
SILENCE_THRESHOLD_DB = -35


class MLPTransform(nn.Module):
    def __init__(self, input_dim=384, output_dim=384, hidden_dims=[1024, 1024]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TargetClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def load_v3_model(model_path):
    """Load the v3 mix classifier for other stem analysis."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    transform = MLPTransform(384, 384, checkpoint['transform_hidden_dims'])
    transform.load_state_dict(checkpoint['transform_state'])
    transform.eval()

    classifier_state = checkpoint['classifier_state']
    hidden_dim = classifier_state['net.0.bias'].shape[0]
    num_classes = len(checkpoint['target_classes'])

    classifier = TargetClassifier(384, hidden_dim, num_classes)
    classifier.load_state_dict(classifier_state)
    classifier.eval()

    return {
        'transform': transform,
        'classifier': classifier,
        'X_mean': checkpoint['X_mean'],
        'X_std': checkpoint['X_std'],
        'target_classes': checkpoint['target_classes']  # ['brass', 'strings', 'winds']
    }


def load_ace_step_model():
    """Load ACE-Step encoder for latent extraction."""
    sys.path.insert(0, '/home/arlo/Data/ACE-Step')
    from acestep.models.dcae.dcae_wrapper import DCAE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dcae = DCAE(
        ckpt_path=CHECKPOINT_DIR,
        device=device,
        dtype=torch.bfloat16
    )
    return dcae, device


def get_mix_files(manifest_path: Path, limit: int = 100) -> list:
    """Get files that are likely mixes from master manifest."""
    print(f"Loading manifest from {manifest_path}...")

    with open(manifest_path) as f:
        data = json.load(f)

    entries = data.get('entries', {})
    mix_files = []

    for path, meta in entries.items():
        # Check if flagged as mix or has mix in filename
        is_mix = meta.get('is_mix', False)
        filename_lower = Path(path).name.lower()
        has_mix_keyword = 'mix' in filename_lower or 'room' in filename_lower

        if (is_mix or has_mix_keyword) and Path(path).exists():
            mix_files.append({
                'path': path,
                'group': meta.get('group', 'undefined'),
                'filename': Path(path).name
            })
            if len(mix_files) >= limit:
                break

    print(f"Found {len(mix_files)} mix files")
    return mix_files


def run_demucs(audio_path: str, output_dir: Path) -> dict:
    """Run Demucs htdemucs_6s separation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "demucs",
        "-n", "htdemucs_6s",
        "-o", str(output_dir),
        audio_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"  Demucs failed: {result.stderr[:200]}")
            return None

        input_name = Path(audio_path).stem
        stems_dir = output_dir / "htdemucs_6s" / input_name

        if not stems_dir.exists():
            return None

        stems = {}
        for stem_name in DEMUCS_STEMS:
            stem_path = stems_dir / f"{stem_name}.wav"
            if stem_path.exists():
                stems[stem_name] = str(stem_path)

        return stems if stems else None

    except subprocess.TimeoutExpired:
        print(f"  Demucs timeout")
        return None
    except Exception as e:
        print(f"  Demucs error: {e}")
        return None


def get_stem_activity(audio_path: str, window_sec: float = 2.0, hop_sec: float = 1.0) -> list:
    """Get temporal activity for a stem based on RMS levels."""
    try:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        total_samples = waveform.shape[-1]
        duration = total_samples / sr
        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)

        segments = []
        pos = 0

        while pos < total_samples:
            end_pos = min(pos + window_samples, total_samples)
            window = waveform[:, pos:end_pos]

            rms = torch.sqrt(torch.mean(window ** 2))
            rms_db = 20 * torch.log10(rms + 1e-10).item()
            is_active = rms_db > SILENCE_THRESHOLD_DB

            start_sec = pos / sr
            end_sec = end_pos / sr

            segments.append({
                'start_sec': round(start_sec, 2),
                'end_sec': round(end_sec, 2),
                'is_active': is_active,
                'rms_db': round(rms_db, 1)
            })

            pos += hop_samples
            if end_pos >= total_samples:
                break

        return segments, duration

    except Exception as e:
        print(f"    Error analyzing {audio_path}: {e}")
        return [], 0


def extract_latent_for_segment(audio_path: str, start_sec: float, end_sec: float,
                                dcae_model, device) -> torch.Tensor:
    """Extract ACE-Step latent for audio segment."""
    try:
        waveform, sr = torchaudio.load(audio_path)

        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment = waveform[:, start_sample:end_sample]

        if segment.shape[-1] < sr:
            return None
        if segment.abs().max() < 1e-6:
            return None

        if segment.shape[0] == 1:
            segment = segment.repeat(2, 1)
        segment = segment / (segment.abs().max() + 1e-8)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            segment = segment.to(device)
            audio_batch = segment.unsqueeze(0).float()
            audio_lengths = torch.tensor([segment.shape[-1]], device=device)
            latents, _ = dcae_model.encode(
                audios=audio_batch,
                audio_lengths=audio_lengths,
                sr=sr
            )

        return latents.float().squeeze(0).cpu()

    except Exception:
        return None


def classify_other_stem_temporal(other_stem_path: str, v3_model, dcae_model, device,
                                  segment_sec: float = 2.0, hop_sec: float = 1.0,
                                  threshold: float = 0.5) -> list:
    """Run V3 classifier on "other" stem to detect brass/strings/winds temporally."""

    transform = v3_model['transform']
    classifier = v3_model['classifier']
    X_mean = v3_model['X_mean']
    X_std = v3_model['X_std']
    target_classes = v3_model['target_classes']

    try:
        waveform, sr = torchaudio.load(other_stem_path)
        duration = waveform.shape[-1] / sr
    except Exception:
        return []

    results = []
    pos_sec = 0

    while pos_sec < duration:
        end_sec = min(pos_sec + segment_sec, duration)
        if end_sec - pos_sec < segment_sec / 2:
            break

        latent = extract_latent_for_segment(other_stem_path, pos_sec, end_sec, dcae_model, device)

        if latent is not None:
            # Extract features (multi-pool)
            pools = []
            for method in ['mean', 'std', 'max']:
                if method == 'mean':
                    pooled = latent.mean(dim=-1)
                elif method == 'std':
                    pooled = latent.std(dim=-1) if latent.shape[-1] > 1 else torch.zeros_like(latent.mean(dim=-1))
                elif method == 'max':
                    pooled = latent.max(dim=-1)[0]
                pools.append(pooled.flatten())
            features = torch.cat(pools)

            features_norm = (features - X_mean) / X_std

            with torch.no_grad():
                transformed = transform(features_norm.unsqueeze(0)).squeeze(0)
                logits = classifier(transformed.unsqueeze(0)).squeeze(0)
                probs = torch.sigmoid(logits)

            active_classes = []
            probabilities = {}
            for i, cls in enumerate(target_classes):
                p = probs[i].item()
                probabilities[cls] = round(p, 3)
                if p >= threshold:
                    active_classes.append(cls)

            results.append({
                'start_sec': round(pos_sec, 2),
                'end_sec': round(end_sec, 2),
                'active_classes': active_classes,
                'probabilities': probabilities
            })

        pos_sec += hop_sec

    return results


def merge_stem_activity(stem_segments: dict, stem_name: str) -> list:
    """Merge consecutive active segments for a stem."""
    segments = stem_segments.get(stem_name, [])

    merged = []
    current_start = None
    current_end = None

    for seg in segments:
        if seg['is_active']:
            if current_start is None:
                current_start = seg['start_sec']
                current_end = seg['end_sec']
            else:
                current_end = seg['end_sec']
        else:
            if current_start is not None:
                merged.append({
                    'class': stem_name,
                    'start_sec': current_start,
                    'end_sec': current_end
                })
                current_start = None
                current_end = None

    if current_start is not None:
        merged.append({
            'class': stem_name,
            'start_sec': current_start,
            'end_sec': current_end
        })

    return merged


def process_mix_file(audio_path: str, v3_model, dcae_model, device, temp_dir: Path) -> dict:
    """Process a single mix file through the full pipeline."""

    result = {
        'path': audio_path,
        'filename': Path(audio_path).name,
        'stems': {},
        'other_v3': [],
        'all_annotations': [],
        'duration': 0
    }

    # Stage 1: Run Demucs separation
    print(f"  Running Demucs...")
    stems = run_demucs(audio_path, temp_dir)

    if not stems:
        print(f"  Demucs failed, skipping")
        return None

    # Stage 2: Analyze each stem for activity
    print(f"  Analyzing stem activity...")
    stem_segments = {}
    all_annotations = []

    for stem_name, stem_path in stems.items():
        if stem_name == 'other':
            continue  # Handle separately with V3

        segments, duration = get_stem_activity(stem_path)
        stem_segments[stem_name] = segments
        result['duration'] = max(result['duration'], duration)

        # Merge into annotation regions
        merged = merge_stem_activity({stem_name: segments}, stem_name)
        all_annotations.extend(merged)

        result['stems'][stem_name] = {
            'path': stem_path,
            'active_segments': len([s for s in segments if s['is_active']]),
            'total_segments': len(segments)
        }

    # Stage 3: Run V3 classifier on "other" stem
    if 'other' in stems:
        print(f"  Running V3 classifier on 'other' stem...")
        other_results = classify_other_stem_temporal(
            stems['other'], v3_model, dcae_model, device,
            segment_sec=2.0, hop_sec=0.5, threshold=0.5
        )
        result['other_v3'] = other_results

        # Merge V3 detections into annotations
        for seg in other_results:
            for cls in seg['active_classes']:
                all_annotations.append({
                    'class': cls,
                    'start_sec': seg['start_sec'],
                    'end_sec': seg['end_sec'],
                    'confidence': seg['probabilities'].get(cls, 0)
                })

        result['stems']['other'] = {
            'path': stems['other'],
            'v3_detections': len([s for s in other_results if s['active_classes']])
        }

    # Sort and dedupe annotations
    all_annotations.sort(key=lambda x: (x['start_sec'], x['class']))
    result['all_annotations'] = all_annotations

    # Get detected instruments summary
    detected = list(set(a['class'] for a in all_annotations))
    result['detected_instruments'] = sorted(detected)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=100, help='Max files to process')
    parser.add_argument('--input-file', type=str, help='Process single file')
    parser.add_argument('--output', type=str, help='Output JSON path')
    parser.add_argument('--keep-stems', action='store_true', help='Keep separated stems')
    args = parser.parse_args()

    if args.output is None:
        args.output = str(OUTPUT_DIR / 'mix_pipeline_v3_results.json')

    # Load models
    print("Loading V3 model...")
    v3_model = load_v3_model(V3_MODEL_PATH)
    print(f"  V3 target classes: {v3_model['target_classes']}")

    print("Loading ACE-Step encoder...")
    dcae_model, device = load_ace_step_model()
    print(f"  Device: {device}")

    # Get files to process
    if args.input_file:
        mix_files = [{'path': args.input_file, 'filename': Path(args.input_file).name}]
    else:
        mix_files = get_mix_files(MANIFEST_PATH, args.limit)

    if not mix_files:
        print("No mix files found!")
        return

    # Process files
    results = []
    detection_counts = defaultdict(int)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i, mix_file in enumerate(mix_files):
            print(f"\n[{i+1}/{len(mix_files)}] Processing: {mix_file['filename']}")

            try:
                result = process_mix_file(
                    mix_file['path'], v3_model, dcae_model, device, temp_path
                )

                if result:
                    results.append(result)
                    for inst in result.get('detected_instruments', []):
                        detection_counts[inst] += 1
                    print(f"  Detected: {result.get('detected_instruments', [])}")

                # Clean up temp stems unless keeping
                if not args.keep_stems:
                    stems_dir = temp_path / "htdemucs_6s"
                    if stems_dir.exists():
                        shutil.rmtree(stems_dir)

            except Exception as e:
                print(f"  Error: {e}")
                continue

    # Save results
    output_data = {
        'pipeline_version': 'v3',
        'stages': [
            'mix_detection',
            'demucs_separation (htdemucs_6s)',
            'v3_other_classifier (brass/strings/winds)'
        ],
        'all_instruments': ['vocals', 'drums', 'bass', 'guitar', 'piano', 'brass', 'strings', 'winds'],
        'total_processed': len(results),
        'detection_counts': dict(detection_counts),
        'generated_at': datetime.now().isoformat(),
        'results': results
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Mix Pipeline V3 Complete")
    print(f"Processed: {len(results)} files")
    print(f"\nDetection counts:")
    for inst in ['vocals', 'drums', 'bass', 'guitar', 'piano', 'brass', 'strings', 'winds']:
        count = detection_counts.get(inst, 0)
        print(f"  {inst}: {count}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
