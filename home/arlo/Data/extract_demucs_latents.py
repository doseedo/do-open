#!/usr/bin/env python3
"""
Extract ACE-Step latents from Demucs-separated stems for all audio files.

This script:
1. Loads audio files from format_manifest.json
2. Runs Demucs htdemucs_6s for 6-stem separation
3. Extracts ACE-Step latent for each non-silent stem
4. Saves latents to /home/arlo/gcs-bucket/LatentDemucs/ with mirrored structure
5. Deletes temp stem audio after latent is saved

Output structure:
    /home/arlo/gcs-bucket/LatentDemucs/protools/.../SessionName/Audio Files/filename/
        drums.pt, bass.pt, vocals.pt, other.pt, guitar.pt, piano.pt

Usage:
    python3 extract_demucs_latents.py --limit 1000 --skip-existing
    python3 extract_demucs_latents.py --limit 1000 --group guitar  # Only process guitar files
"""

import argparse
import json
import subprocess
import torch
import torchaudio
from pathlib import Path
from datetime import datetime
import shutil
import gc
import os
import sys
import random

# Add paths for imports
sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

# Configuration
FORMAT_MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/format_manifest.json")
COMBINED_MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/combined_manifest.json")
PREDICTIONS_PATH = Path("/home/arlo/Data/latent_classifier/predictions.json")
GCS_BASE = Path("/home/arlo/gcs-bucket")
OUTPUT_BASE = GCS_BASE / "LatentDemucsV2"
AUDIO_OUTPUT_BASE = GCS_BASE / "AudioDemucsV2"  # Save separated audio stems here
STEMS_DIR = Path("/tmp/demucs_stems_batch")
CHECKPOINT_DIR = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c"
PROGRESS_FILE = Path("/home/arlo/Data/demucs_latent_progress.json")

# Demucs stem names
STEM_NAMES = ["drums", "bass", "vocals", "other", "guitar", "piano"]

# Latent extraction params
LATENT_SHAPE = (8, 16)
MIN_SAMPLES = int(0.5 * 48000)  # 0.5 second minimum
MAX_SAMPLES = int(120 * 48000)  # 120 seconds (2 minutes) max
MAX_DEMUCS_DURATION = 180       # Skip files > 3 minutes
SILENCE_THRESHOLD_DB = -70      # Much lower threshold to catch quiet stems


def load_audio_files(format_manifest_path: Path, combined_manifest_path: Path,
                     limit: int = None, group_filter: str = None,
                     skip_existing: bool = False, mix_only: bool = False) -> list:
    """Load audio files from manifest."""
    print(f"Loading format manifest...")
    with open(format_manifest_path) as f:
        format_data = json.load(f)

    print(f"Loading combined manifest for groups...")
    with open(combined_manifest_path) as f:
        combined = json.load(f)

    # Load multi-detected paths if mix_only mode
    multi_paths = set()
    if mix_only and PREDICTIONS_PATH.exists():
        print(f"Loading predictions for multi-detection...")
        with open(PREDICTIONS_PATH) as f:
            predictions = json.load(f)
        for p in predictions.get('predictions', []):
            if p.get('is_multi'):
                multi_paths.add(p.get('path', ''))
        print(f"  Found {len(multi_paths)} multi-detected files")

    # Build set of is_mix paths
    is_mix_paths = set()
    if mix_only:
        for path, meta in combined.items():
            if isinstance(meta, dict) and meta.get('is_mix'):
                is_mix_paths.add(path)
        print(f"  Found {len(is_mix_paths)} is_mix files")

    entries = format_data.get('entries', [])
    print(f"Total entries in format manifest: {len(entries)}")

    # Filter to files with audio (has_latent means original audio exists)
    candidates = []
    for entry in entries:
        rel_path = entry.get('path', '')
        has_latent = entry.get('has_latent')

        # Skip if no original latent (means no audio)
        if has_latent not in [True, 'true', 'True']:
            continue

        # Build absolute path
        abs_path = str(GCS_BASE / rel_path)

        # Get group from combined manifest
        group = 'undefined'
        is_mix = False
        if abs_path in combined:
            meta = combined[abs_path]
            if isinstance(meta, dict):
                group = meta.get('group', 'undefined')
                is_mix = meta.get('is_mix', False)

        # Apply mix_only filter (is_mix OR multi-detected)
        if mix_only:
            is_multi = abs_path in multi_paths
            if not is_mix and not is_multi:
                continue

        # Apply group filter
        if group_filter and group != group_filter:
            continue

        # Check if already processed
        if skip_existing:
            output_dir = OUTPUT_BASE / rel_path.replace('.wav', '').replace('.mp3', '').replace('.flac', '')
            if output_dir.exists() and any(output_dir.glob('*.pt')):
                continue

        candidates.append({
            'path': abs_path,
            'rel_path': rel_path,
            'group': group,
            'filename': Path(rel_path).name,
            'is_mix': is_mix,
            'is_multi': abs_path in multi_paths if mix_only else False
        })

    print(f"Candidates after filtering: {len(candidates)}")

    # Shuffle for variety
    random.seed(42)
    random.shuffle(candidates)

    # Apply limit
    if limit:
        candidates = candidates[:limit]

    return candidates


def run_demucs(audio_path: str, output_dir: Path) -> dict:
    """Run Demucs 6-stem separation on an audio file."""
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
            timeout=600
        )

        if result.returncode != 0:
            print(f"    Demucs failed: {result.stderr[:200]}")
            return None

        # Find output stems
        input_name = Path(audio_path).stem
        stems_dir = output_dir / "htdemucs_6s" / input_name

        if not stems_dir.exists():
            print(f"    Stems dir not found: {stems_dir}")
            return None

        stems = {}
        for stem_name in STEM_NAMES:
            stem_path = stems_dir / f"{stem_name}.wav"
            if stem_path.exists():
                stems[stem_name] = str(stem_path)

        return stems if stems else None

    except subprocess.TimeoutExpired:
        print(f"    Demucs timeout")
        return None
    except Exception as e:
        print(f"    Demucs error: {e}")
        return None


def is_stem_silent(audio_path: str, threshold_db: float = SILENCE_THRESHOLD_DB) -> bool:
    """Check if a stem is silent."""
    try:
        waveform, sr = torchaudio.load(audio_path)
        rms = torch.sqrt(torch.mean(waveform ** 2))
        rms_db = 20 * torch.log10(rms + 1e-10)
        return rms_db < threshold_db
    except:
        return True


def extract_latent(audio_path: str, model, device) -> torch.Tensor:
    """Extract ACE-Step latent from audio file."""
    try:
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[-1] < MIN_SAMPLES:
            return None
        if waveform.shape[-1] > MAX_SAMPLES:
            waveform = waveform[:, :MAX_SAMPLES]
        if waveform.abs().max() < 1e-6:
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

        if latents.shape[:2] != LATENT_SHAPE:
            return None

        return latents

    except Exception as e:
        print(f"      Latent error: {e}")
        return None


def save_progress(processed: list, failed: list, progress_file: Path):
    """Save progress for resume."""
    with open(progress_file, 'w') as f:
        json.dump({
            'processed': processed,
            'failed': failed,
            'updated_at': datetime.now().isoformat()
        }, f)


def main():
    parser = argparse.ArgumentParser(description='Extract Demucs stem latents for all audio')
    parser.add_argument('--limit', type=int, default=None, help='Number of files to process')
    parser.add_argument('--skip-existing', action='store_true', help='Skip already processed files')
    parser.add_argument('--group', type=str, default=None, help='Only process files from this group')
    parser.add_argument('--mix-only', action='store_true',
                        help='Only process mix/room files (is_mix) and multi-detected files (~17k total)')
    parser.add_argument('--silence-threshold', type=float, default=SILENCE_THRESHOLD_DB,
                        help='Silence threshold in dB (default -55)')
    args = parser.parse_args()

    print("=" * 60)
    print("DEMUCS STEM LATENT EXTRACTION")
    print("=" * 60)
    print(f"Latent output: {OUTPUT_BASE}")
    print(f"Audio output:  {AUDIO_OUTPUT_BASE}")
    print(f"Silence threshold: {args.silence_threshold}dB (all stems saved regardless)")
    if args.mix_only:
        print(f"Mode: MIX ONLY (is_mix + multi-detected)")

    # Create output directories
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    AUDIO_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    STEMS_DIR.mkdir(parents=True, exist_ok=True)

    # Load audio files
    audio_files = load_audio_files(
        FORMAT_MANIFEST_PATH,
        COMBINED_MANIFEST_PATH,
        limit=args.limit,
        group_filter=args.group,
        skip_existing=args.skip_existing,
        mix_only=args.mix_only
    )

    if not audio_files:
        print("No files to process!")
        return

    print(f"\nWill process {len(audio_files)} files")

    # Load ACE-Step model
    print("\nLoading ACE-Step model...")
    from acestep.pipeline_ace_step import ACEStepPipeline

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR)
    pipeline.load_checkpoint(CHECKPOINT_DIR)
    dcae_model = pipeline.music_dcae.eval().to(device)
    print(f"  Model loaded on {device}")

    # Process files
    processed = []
    failed = []
    stats = {'total_latents': 0, 'silent_stems': 0}

    print(f"\nProcessing {len(audio_files)} files...")

    for i, audio_file in enumerate(audio_files):
        audio_path = audio_file['path']
        rel_path = audio_file['rel_path']
        filename = audio_file['filename']

        print(f"\n[{i+1}/{len(audio_files)}] {filename}")

        # Check if file exists
        if not Path(audio_path).exists():
            print(f"  File not found, skipping")
            failed.append({'path': audio_path, 'reason': 'not_found'})
            continue

        # Check duration
        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            if duration > MAX_DEMUCS_DURATION:
                print(f"  Too long ({duration:.0f}s > {MAX_DEMUCS_DURATION}s), skipping")
                failed.append({'path': audio_path, 'reason': f'too_long_{duration:.0f}s'})
                continue
        except Exception as e:
            print(f"  Can't read audio info: {e}")
            failed.append({'path': audio_path, 'reason': 'audio_info_failed'})
            continue

        # Clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run Demucs
        print(f"  Running Demucs ({duration:.1f}s)...")
        stems = run_demucs(audio_path, STEMS_DIR)

        if not stems:
            failed.append({'path': audio_path, 'reason': 'demucs_failed'})
            continue

        # Create output directories (mirror structure)
        # rel_path: protools/.../filename.wav -> protools/.../filename/
        rel_dir = rel_path.rsplit('.', 1)[0]  # Remove extension
        output_dir = OUTPUT_BASE / rel_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Also create audio output directory
        audio_output_dir = AUDIO_OUTPUT_BASE / rel_dir
        audio_output_dir.mkdir(parents=True, exist_ok=True)

        # Process ALL stems (don't skip silent ones)
        stem_results = {}
        for stem_name, stem_path in stems.items():
            is_silent = is_stem_silent(stem_path, args.silence_threshold)

            # Save audio stem regardless of silence
            audio_dest = audio_output_dir / f"{stem_name}.wav"
            try:
                shutil.copy2(stem_path, audio_dest)
            except Exception as e:
                print(f"    {stem_name}: audio copy failed - {e}")

            # Clear cache before each extraction
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Extract latent (even for silent stems)
            latent = extract_latent(stem_path, dcae_model, device)

            if latent is None:
                print(f"    {stem_name}: extraction failed")
                stem_results[stem_name] = 'failed'
                continue

            # Save latent
            latent_path = output_dir / f"{stem_name}.pt"
            torch.save(latent, latent_path)

            if is_silent:
                print(f"    {stem_name}: saved ({latent.shape[-1]} frames) [silent]")
                stem_results[stem_name] = 'saved_silent'
                stats['silent_stems'] += 1
            else:
                print(f"    {stem_name}: saved ({latent.shape[-1]} frames)")
                stem_results[stem_name] = 'saved'

            stats['total_latents'] += 1

            # Free memory
            del latent

        # Clean up stems directory for this file (audio already copied)
        input_name = Path(audio_path).stem
        stems_subdir = STEMS_DIR / "htdemucs_6s" / input_name
        if stems_subdir.exists():
            shutil.rmtree(stems_subdir)

        processed.append({
            'path': audio_path,
            'output_dir': str(output_dir),
            'stems': stem_results
        })

        # Save progress periodically
        if (i + 1) % 50 == 0:
            save_progress(processed, failed, PROGRESS_FILE)
            print(f"  Progress saved ({len(processed)} processed, {stats['total_latents']} latents)")

        # Clear CUDA cache periodically
        if (i + 1) % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed: {len(processed)}")
    print(f"Failed: {len(failed)}")
    print(f"Total latents saved: {stats['total_latents']}")
    print(f"Silent stems skipped: {stats['silent_stems']}")
    print(f"Output directory: {OUTPUT_BASE}")

    # Save final progress
    save_progress(processed, failed, PROGRESS_FILE)
    print(f"\nProgress saved to: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
