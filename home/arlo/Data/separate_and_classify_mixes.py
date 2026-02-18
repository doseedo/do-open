#!/usr/bin/env python3
"""
Separate mix files using Demucs and classify the resulting stems.

This script:
1. Selects 500 mix files from the manifest
2. Runs Demucs htdemucs_6s for 6-stem separation
3. Generates ACE-Step latents for each stem
4. Runs the latent classifier on each stem
5. Saves results for UI review

Usage:
    python3 separate_and_classify_mixes.py --limit 500

Output:
    /home/arlo/Data/latent_classifier/separated_stems_predictions.json
"""

import argparse
import json
import subprocess
import torch
import torchaudio
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import shutil
import gc
import os
import sys

# Add paths for imports
sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

# Configuration
MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/combined_manifest.json")
FORMAT_MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/format_manifest.json")
CLASSIFIER_MODEL_PATH = Path("/home/arlo/Data/latent_classifier/model.pt")
OUTPUT_DIR = Path("/home/arlo/Data/latent_classifier")
STEMS_DIR = Path("/tmp/demucs_stems")
LATENTS_DIR = Path("/tmp/stem_latents")
CHECKPOINT_DIR = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c"

# Demucs produces these stems with htdemucs_6s
STEM_NAMES = ["drums", "bass", "vocals", "other", "guitar", "piano"]

# Latent extraction params
LATENT_SHAPE = (8, 16)
MIN_SAMPLES = int(1 * 48000)  # 1 second minimum for stems
MAX_SAMPLES = int(60 * 48000)  # 60 seconds max to avoid OOM
MAX_DEMUCS_DURATION = 120  # Skip files longer than 2 minutes for Demucs

# Temporal analysis params
LATENT_FRAMES_PER_SEC = 44100 / 512  # ~86.13 fps
AUDIO_SAMPLE_RATE = 48000


def get_mix_files(manifest_path: Path, format_manifest_path: Path, limit: int = 500) -> list:
    """Get mix files that have audio available."""
    import random
    random.seed(42)

    print(f"Loading manifest from {manifest_path}...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Collect all candidate mix files (without checking existence yet - too slow on GCS)
    candidates = []
    for path, meta in manifest.items():
        # Check for mix/room in filename (not directory name)
        filename_only = Path(path).name.lower()
        if 'mix' in filename_only or 'room' in filename_only:
            candidates.append({
                'path': path,
                'group': meta.get('group', 'undefined') if isinstance(meta, dict) else 'undefined',
                'filename': Path(path).name
            })

    print(f"Found {len(candidates)} mix/room candidates in manifest")

    # Shuffle and check existence for a larger pool to get enough valid files
    random.shuffle(candidates)

    # Check existence for up to 3x limit to account for missing files
    check_limit = min(len(candidates), limit * 3)
    mix_files = []

    print(f"Checking file existence for up to {check_limit} candidates...")
    for i, candidate in enumerate(candidates[:check_limit]):
        if i % 100 == 0 and i > 0:
            print(f"  Checked {i}/{check_limit}, found {len(mix_files)} valid...")
        if Path(candidate['path']).exists():
            mix_files.append(candidate)
            if len(mix_files) >= limit:
                break

    print(f"Found {len(mix_files)} mix files with audio")
    return mix_files[:limit]


def run_demucs(audio_path: str, output_dir: Path) -> dict:
    """Run Demucs htdemucs_6s on an audio file.

    Returns dict mapping stem name -> stem path, or None if failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use htdemucs_6s for 6-stem separation
    cmd = [
        "demucs",
        "-n", "htdemucs_6s",
        "-o", str(output_dir),
        audio_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per file
        )

        if result.returncode != 0:
            print(f"  Demucs failed: {result.stderr[:200]}")
            return None

        # Find output stems
        input_name = Path(audio_path).stem
        stems_dir = output_dir / "htdemucs_6s" / input_name

        if not stems_dir.exists():
            print(f"  Stems directory not found: {stems_dir}")
            return None

        stems = {}
        for stem_name in STEM_NAMES:
            stem_path = stems_dir / f"{stem_name}.wav"
            if stem_path.exists():
                stems[stem_name] = str(stem_path)
            else:
                print(f"  Missing stem: {stem_name}")

        return stems if stems else None

    except subprocess.TimeoutExpired:
        print(f"  Demucs timeout")
        return None
    except Exception as e:
        print(f"  Demucs error: {e}")
        return None


def is_stem_silent(audio_path: str, threshold_db: float = -40) -> bool:
    """Check if a stem is effectively silent.

    Args:
        audio_path: Path to audio file
        threshold_db: RMS threshold in dB (default -40dB is very quiet)

    Returns True if the stem is silent/near-silent.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)
        # Calculate RMS
        rms = torch.sqrt(torch.mean(waveform ** 2))
        rms_db = 20 * torch.log10(rms + 1e-10)
        return rms_db < threshold_db
    except Exception:
        return True


def extract_latent(audio_path: str, model, device) -> torch.Tensor:
    """Extract ACE-Step latent from audio file.

    Returns latent tensor [8, 16, T] or None if failed.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)

        # Validate
        if waveform.shape[-1] < MIN_SAMPLES:
            return None
        if waveform.shape[-1] > MAX_SAMPLES:
            waveform = waveform[:, :MAX_SAMPLES]
        if waveform.abs().max() < 1e-6:
            return None  # Silent
        # Check RMS level
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms < 1e-4:  # Very quiet
            return None

        # Prepare audio
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        waveform = waveform / (waveform.abs().max() + 1e-8)

        # Encode
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            waveform = waveform.to(device)
            audio_batch = waveform.unsqueeze(0).float()
            audio_lengths = torch.tensor([waveform.shape[-1]], device=device)
            latents, _ = model.encode(
                audios=audio_batch,
                audio_lengths=audio_lengths,
                sr=sr
            )

        latents = latents.float().squeeze(0).cpu()

        # Validate latent shape
        if latents.shape[:2] != LATENT_SHAPE:
            return None

        return latents

    except Exception as e:
        print(f"    Latent extraction failed: {e}")
        return None


def classify_latent(latent: torch.Tensor, model, mean, std, classes, device) -> dict:
    """Classify a latent using the trained classifier.

    Returns dict with predicted_group, confidence, all_probabilities.
    """
    import torch.nn.functional as F

    # Pool latent to feature vector (same as in classifier)
    features = []
    features.append(latent.mean(dim=-1))  # [8, 16]
    features.append(latent.std(dim=-1))   # [8, 16]
    features.append(latent.max(dim=-1)[0])  # [8, 16]
    stacked = torch.stack(features, dim=-1)  # [8, 16, 3]
    feature_vec = stacked.flatten().unsqueeze(0)  # [1, 384]

    # Move to device and normalize
    feature_vec = feature_vec.to(device)
    feature_vec = (feature_vec - mean) / std

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(feature_vec)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = probs.argmax()

    return {
        'predicted_group': classes[pred_idx],
        'confidence': float(probs[pred_idx]),
        'all_probabilities': {c: float(p) for c, p in zip(classes, probs)}
    }


def get_audio_windows_rms(audio_path: str, window_sec: float = 2.0, hop_sec: float = 1.0,
                          silence_threshold_db: float = -35) -> list:
    """Get RMS levels for audio windows to detect silence.

    Returns list of (start_sec, end_sec, is_silent) tuples.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        total_samples = waveform.shape[-1]
        duration = total_samples / sr
        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)

        windows = []
        pos = 0
        while pos < total_samples:
            end_pos = min(pos + window_samples, total_samples)
            window = waveform[:, pos:end_pos]

            # Calculate RMS
            rms = torch.sqrt(torch.mean(window ** 2))
            rms_db = 20 * torch.log10(rms + 1e-10)
            is_silent = rms_db < silence_threshold_db

            start_sec = pos / sr
            end_sec = end_pos / sr
            windows.append((start_sec, end_sec, bool(is_silent)))

            pos += hop_samples
            if end_pos >= total_samples:
                break

        return windows
    except Exception as e:
        print(f"    Error analyzing audio: {e}")
        return []


def extract_latent_segment(audio_path: str, start_sec: float, end_sec: float,
                           model, device) -> torch.Tensor:
    """Extract latent for a specific time segment of audio.

    Returns latent tensor or None if failed.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)

        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment = waveform[:, start_sample:end_sample]

        # Validate
        if segment.shape[-1] < sr:  # Less than 1 second
            return None
        if segment.abs().max() < 1e-6:
            return None

        # Prepare audio
        if segment.shape[0] == 1:
            segment = segment.repeat(2, 1)
        segment = segment / (segment.abs().max() + 1e-8)

        # Encode
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            segment = segment.to(device)
            audio_batch = segment.unsqueeze(0).float()
            audio_lengths = torch.tensor([segment.shape[-1]], device=device)
            latents, _ = model.encode(
                audios=audio_batch,
                audio_lengths=audio_lengths,
                sr=sr
            )

        return latents.float().squeeze(0).cpu()

    except Exception as e:
        return None


def classify_stem_temporal(stem_path: str, stem_name: str, dcae_model, classifier,
                           mean, std, classes, device,
                           window_sec: float = 2.0, hop_sec: float = 1.0,
                           min_confidence: float = 0.5) -> dict:
    """Classify a stem temporally, returning active segments.

    Returns dict with:
        - segments: list of {start, end, instrument, confidence}
        - total_duration: float
        - active_duration: float
        - silent_duration: float
    """
    # Get audio windows with silence detection
    windows = get_audio_windows_rms(stem_path, window_sec, hop_sec)
    if not windows:
        return {'segments': [], 'total_duration': 0, 'active_duration': 0, 'silent_duration': 0}

    total_duration = windows[-1][1] if windows else 0
    segments = []
    active_duration = 0
    silent_duration = 0

    for start_sec, end_sec, is_silent in windows:
        if is_silent:
            silent_duration += (end_sec - start_sec)
            continue

        # Extract latent for this segment
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        latent = extract_latent_segment(stem_path, start_sec, end_sec, dcae_model, device)
        if latent is None:
            silent_duration += (end_sec - start_sec)
            continue

        # Classify
        pred = classify_latent(latent, classifier, mean, std, classes, device)
        del latent

        if pred['confidence'] >= min_confidence:
            segments.append({
                'start': round(start_sec, 2),
                'end': round(end_sec, 2),
                'instrument': pred['predicted_group'],
                'confidence': round(pred['confidence'], 3),
                'stem': stem_name
            })
            active_duration += (end_sec - start_sec)
        else:
            silent_duration += (end_sec - start_sec)

    # Merge adjacent segments with same instrument
    merged = []
    for seg in segments:
        if merged and merged[-1]['instrument'] == seg['instrument'] and \
           abs(merged[-1]['end'] - seg['start']) < hop_sec * 1.5:
            # Extend previous segment
            merged[-1]['end'] = seg['end']
            merged[-1]['confidence'] = max(merged[-1]['confidence'], seg['confidence'])
        else:
            merged.append(seg.copy())

    return {
        'segments': merged,
        'total_duration': round(total_duration, 2),
        'active_duration': round(active_duration, 2),
        'silent_duration': round(silent_duration, 2)
    }


def merge_stem_timelines(stem_results: dict) -> list:
    """Merge temporal results from all stems into a unified timeline.

    Returns list of {start, end, instruments: [{name, confidence, stem}]}
    """
    # Collect all segment boundaries
    all_times = set([0])
    for stem_name, result in stem_results.items():
        for seg in result.get('segments', []):
            all_times.add(seg['start'])
            all_times.add(seg['end'])

    all_times = sorted(all_times)
    if len(all_times) < 2:
        return []

    # For each time slice, find which instruments are active
    timeline = []
    for i in range(len(all_times) - 1):
        start = all_times[i]
        end = all_times[i + 1]

        active_instruments = []
        for stem_name, result in stem_results.items():
            for seg in result.get('segments', []):
                # Check if segment overlaps with this time slice
                if seg['start'] < end and seg['end'] > start:
                    active_instruments.append({
                        'instrument': seg['instrument'],
                        'confidence': seg['confidence'],
                        'stem': stem_name
                    })

        if active_instruments:
            timeline.append({
                'start': round(start, 2),
                'end': round(end, 2),
                'duration': round(end - start, 2),
                'instruments': active_instruments
            })

    # Merge adjacent slices with same instruments
    merged = []
    for entry in timeline:
        inst_key = tuple(sorted((i['instrument'], i['stem']) for i in entry['instruments']))
        if merged and merged[-1]['_key'] == inst_key:
            merged[-1]['end'] = entry['end']
            merged[-1]['duration'] = round(merged[-1]['end'] - merged[-1]['start'], 2)
        else:
            entry['_key'] = inst_key
            merged.append(entry)

    # Remove internal key
    for entry in merged:
        del entry['_key']

    return merged


def main():
    parser = argparse.ArgumentParser(description='Separate mix files and classify stems')
    parser.add_argument('--limit', type=int, default=500, help='Number of mix files to process')
    parser.add_argument('--skip-existing', action='store_true', help='Skip already processed files')
    parser.add_argument('--temporal', action='store_true', help='Run temporal analysis (when are instruments active)')
    parser.add_argument('--window-sec', type=float, default=2.0, help='Window size for temporal analysis')
    parser.add_argument('--hop-sec', type=float, default=1.0, help='Hop size for temporal analysis')
    parser.add_argument('--paths-file', type=str, help='JSON file with list of audio paths to process (overrides mix file detection)')
    parser.add_argument('--output-name', type=str, help='Custom output filename (without .json)')
    parser.add_argument('--min-confidence', type=float, default=0.5, help='Minimum confidence for temporal predictions (default 0.5)')
    args = parser.parse_args()

    print("=" * 60)
    if args.temporal:
        print("MIX FILE SEPARATION AND TEMPORAL ANALYSIS")
    else:
        print("MIX FILE SEPARATION AND CLASSIFICATION")
    print("=" * 60)

    # Setup directories
    STEMS_DIR.mkdir(parents=True, exist_ok=True)
    LATENTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing results
    if args.output_name:
        output_path = OUTPUT_DIR / f"{args.output_name}.json"
    elif args.temporal:
        output_path = OUTPUT_DIR / "separated_stems_temporal.json"
    else:
        output_path = OUTPUT_DIR / "separated_stems_predictions.json"
    existing_results = []
    processed_paths = set()

    if args.skip_existing and output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
            existing_results = data.get('results', [])
            processed_paths = {r['original_path'] for r in existing_results}
            print(f"Found {len(existing_results)} existing results")

    # Get files to process
    if args.paths_file:
        # Load paths from JSON file
        with open(args.paths_file) as f:
            paths = json.load(f)
        mix_files = [{'path': p, 'filename': Path(p).name, 'group': 'unknown'} for p in paths[:args.limit]]
        print(f"Loaded {len(mix_files)} paths from {args.paths_file}")
    else:
        # Get mix files from manifest
        mix_files = get_mix_files(MANIFEST_PATH, FORMAT_MANIFEST_PATH, args.limit)

    # Filter out already processed
    if processed_paths:
        mix_files = [f for f in mix_files if f['path'] not in processed_paths]
        print(f"After filtering processed: {len(mix_files)} files to process")

    if not mix_files:
        print("No files to process!")
        return

    # Load ACE-Step model for latent extraction
    print("\nLoading ACE-Step model for latent extraction...")
    from acestep.pipeline_ace_step import ACEStepPipeline

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR)
    pipeline.load_checkpoint(CHECKPOINT_DIR)
    dcae_model = pipeline.music_dcae.eval().to(device)
    print(f"  ACE-Step loaded on {device}")

    # Load classifier
    print("\nLoading latent classifier...")
    classifier_data = torch.load(CLASSIFIER_MODEL_PATH, map_location='cpu', weights_only=False)

    from latent_instrument_classifier import InstrumentClassifier
    classifier = InstrumentClassifier(
        classifier_data['input_dim'],
        classifier_data['num_classes'],
        classifier_data.get('hidden_dim', 256)
    )
    classifier.load_state_dict(classifier_data['model_state'])
    classifier.to(device)
    classifier.eval()

    mean = classifier_data['mean'].to(device)
    std = classifier_data['std'].to(device)
    classes = classifier_data['label_encoder_classes']
    print(f"  Classifier loaded with {len(classes)} classes")

    # Process files
    results = list(existing_results)
    failed = []

    print(f"\nProcessing {len(mix_files)} files...")

    for i, mix_file in enumerate(mix_files):
        audio_path = mix_file['path']
        print(f"\n[{i+1}/{len(mix_files)}] {Path(audio_path).name}")

        # Check audio duration before running Demucs
        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            if duration > MAX_DEMUCS_DURATION:
                print(f"  Skipping: too long ({duration:.0f}s > {MAX_DEMUCS_DURATION}s)")
                failed.append({'path': audio_path, 'reason': f'too_long_{duration:.0f}s'})
                continue
        except Exception as e:
            print(f"  Skipping: couldn't read audio info: {e}")
            failed.append({'path': audio_path, 'reason': 'audio_info_failed'})
            continue

        # Clear CUDA cache before Demucs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run Demucs
        print(f"  Running Demucs ({duration:.1f}s)...")
        stems = run_demucs(audio_path, STEMS_DIR)

        if not stems:
            failed.append({'path': audio_path, 'reason': 'demucs_failed'})
            continue

        # Process each stem
        stem_results = {}

        if args.temporal:
            # Temporal mode: analyze when each instrument is active
            for stem_name, stem_path in stems.items():
                # Quick silence check on whole stem
                if is_stem_silent(stem_path, threshold_db=-40):
                    stem_results[stem_name] = {
                        'segments': [],
                        'total_duration': 0,
                        'active_duration': 0,
                        'silent_duration': 0,
                        'is_silent': True
                    }
                    print(f"  Stem {stem_name}: fully silent")
                    continue

                print(f"  Temporal analysis: {stem_name}")
                result = classify_stem_temporal(
                    stem_path, stem_name, dcae_model, classifier,
                    mean, std, classes, device,
                    window_sec=args.window_sec, hop_sec=args.hop_sec,
                    min_confidence=args.min_confidence
                )
                result['is_silent'] = len(result['segments']) == 0
                stem_results[stem_name] = result

                if result['segments']:
                    instruments = set(s['instrument'] for s in result['segments'])
                    print(f"    -> {', '.join(instruments)} ({result['active_duration']:.1f}s active)")
                else:
                    print(f"    -> no confident predictions")

            # Merge into unified timeline
            timeline = merge_stem_timelines(stem_results)

            # Collect all detected instruments
            detected = set()
            for entry in timeline:
                for inst in entry['instruments']:
                    detected.add(inst['instrument'])

            result = {
                'original_path': audio_path,
                'original_filename': mix_file['filename'],
                'original_group': mix_file['group'],
                'original_duration': duration,
                'stems': stem_results,
                'timeline': timeline,
                'detected_instruments': sorted(list(detected)),
                'silent_stems': sum(1 for s in stem_results.values() if s.get('is_silent', False)),
                'active_stems': sum(1 for s in stem_results.values() if not s.get('is_silent', False)),
                'processed_at': datetime.now().isoformat()
            }
        else:
            # Non-temporal mode: classify whole stem
            for stem_name, stem_path in stems.items():
                # Check if stem is silent first (fast check before latent extraction)
                if is_stem_silent(stem_path, threshold_db=-35):
                    stem_results[stem_name] = {
                        'predicted_group': 'silent',
                        'confidence': 0.0,
                        'is_silent': True
                    }
                    print(f"  Stem {stem_name}: silent (skipped)")
                    continue

                print(f"  Processing stem: {stem_name}")

                # Clear CUDA cache before each latent extraction
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Extract latent
                latent = extract_latent(stem_path, dcae_model, device)
                if latent is None:
                    stem_results[stem_name] = {
                        'predicted_group': 'silent',
                        'confidence': 0.0,
                        'is_silent': True,
                        'reason': 'latent_extraction_failed'
                    }
                    continue

                # Classify
                pred = classify_latent(latent, classifier, mean, std, classes, device)
                pred['is_silent'] = False
                stem_results[stem_name] = pred
                print(f"    -> {pred['predicted_group']} ({pred['confidence']:.1%}")

                # Free latent tensor
                del latent

            # Build result entry - only count non-silent stems with decent confidence
            detected = set()
            for s in stem_results.values():
                if not s.get('is_silent', False) and s.get('confidence', 0) > 0.5:
                    detected.add(s['predicted_group'])

            result = {
                'original_path': audio_path,
                'original_filename': mix_file['filename'],
                'original_group': mix_file['group'],
                'stems': stem_results,
                'detected_instruments': sorted(list(detected)),
                'silent_stems': sum(1 for s in stem_results.values() if s.get('is_silent', False)),
                'active_stems': sum(1 for s in stem_results.values() if not s.get('is_silent', False)),
                'processed_at': datetime.now().isoformat()
            }

        results.append(result)

        # Clean up stems to save disk space
        input_name = Path(audio_path).stem
        stems_subdir = STEMS_DIR / "htdemucs_6s" / input_name
        if stems_subdir.exists():
            shutil.rmtree(stems_subdir)

        # Periodic save
        if (i + 1) % 10 == 0:
            output_data = {
                'total': len(results),
                'failed': len(failed),
                'results': results,
                'failed_files': failed,
                'generated_at': datetime.now().isoformat()
            }
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"  Saved checkpoint ({len(results)} results)")

        # Clear CUDA cache periodically
        if (i + 1) % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final save
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed: {len(results)}")
    print(f"Failed: {len(failed)}")

    # Analyze results
    all_detected = []
    for r in results:
        all_detected.extend(r.get('detected_instruments', []))

    from collections import Counter
    instrument_counts = Counter(all_detected)
    print("\nInstruments detected across all stems:")
    for inst, count in instrument_counts.most_common():
        print(f"  {inst}: {count}")

    # Save final results
    output_data = {
        'total': len(results),
        'failed': len(failed),
        'instrument_distribution': dict(instrument_counts),
        'results': results,
        'failed_files': failed,
        'generated_at': datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
