#!/usr/bin/env python3
"""
Precompute DSP-shifted latents for V4 training - EFFICIENT VERSION.

Uses audio files directly (no decode step) + multiprocessing for sox.

Pipeline:
1. Load segment audio from original audio files (no DCAE decode needed!)
2. Apply DSP pitch shift (sox) in parallel across CPU cores
3. Batch encode shifted audio → latents (GPU)
4. Save shifted latents to disk

Usage:
    python precompute_shifted_latents.py \
        --segments /path/to/trumpet_segments_filtered.json \
        --output_dir /mnt/msdd2/pitchshift_v4_precomputed \
        --shifts -12,-6,-3,-1,0,1,3,6,12 \
        --num_workers 8 \
        --batch_size 16
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import tempfile
import multiprocessing as mp

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# Add paths for DCAE
sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/dø')


def fix_path(path: str) -> str:
    """Fix mount paths."""
    if not path:
        return path
    replacements = [
        ('/mnt/msdd/', '/mnt/msdd2/'),
        ('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/'),
    ]
    for old, new in replacements:
        if old in path:
            path = path.replace(old, new)
    return path


def get_source_hash(latent_path: str, start_frame: int, end_frame: int) -> str:
    """Create a unique hash for a source segment."""
    key = f"{latent_path}:{start_frame}:{end_frame}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# Sample rate constants
DCAE_SR = 44100  # DCAE operates at 44.1kHz
# DCAE: latent_len = audio_samples / sr * 44100 / 512 / 8 = audio_samples / 4096
SAMPLES_PER_FRAME = 4096


def apply_pitch_shift_sox_file(
    audio_path: str,
    output_path: str,
    shift_semitones: int,
    start_time: float,
    duration: float,
    sr: int = DCAE_SR,
) -> bool:
    """
    Apply pitch shift to a segment of an audio file using sox.

    Uses time-based extraction (seconds) to be sample-rate independent.

    Returns True on success, False on failure.
    """
    if shift_semitones == 0:
        # Still need to extract segment
        cmd = [
            'sox', audio_path, output_path,
            'trim', str(start_time), str(duration),
            'rate', '-v', str(sr),
        ]
    else:
        cents = shift_semitones * 100
        cmd = [
            'sox', audio_path, output_path,
            'trim', str(start_time), str(duration),
            'pitch', str(cents),
            'rate', '-v', str(sr),
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False


def process_segment_sox(args: Tuple) -> Optional[Dict]:
    """
    Worker function: extract segment + apply pitch shift with sox.

    This runs in a separate process for parallelism.
    Returns dict with temp audio path and metadata, or None on failure.
    """
    segment, shift, output_dir, temp_dir = args

    audio_path = fix_path(segment.get('audio_path', ''))
    latent_path = segment.get('latent_path', '')
    start_frame = segment['start_frame']
    end_frame = segment['end_frame']

    if not audio_path or not os.path.exists(audio_path):
        return None

    # Convert frames to time (seconds) - sample rate independent
    start_time = start_frame * SAMPLES_PER_FRAME / DCAE_SR
    duration = (end_frame - start_frame) * SAMPLES_PER_FRAME / DCAE_SR

    # Create unique hash and output path
    source_hash = get_source_hash(latent_path, start_frame, end_frame)
    final_latent_path = output_dir / 'shifted_latents' / f"{source_hash}_shift{shift:+d}.pt"

    # Skip if already exists
    if final_latent_path.exists():
        return {
            'skip': True,
            'output_path': str(final_latent_path),
            'shift': shift,
            'source_hash': source_hash,
            'segment': segment,
        }

    # Create temp file for shifted audio
    temp_audio_path = temp_dir / f"{source_hash}_shift{shift:+d}.wav"

    # Apply sox pitch shift
    success = apply_pitch_shift_sox_file(
        audio_path=audio_path,
        output_path=str(temp_audio_path),
        shift_semitones=shift,
        start_time=start_time,
        duration=duration,
    )

    if not success or not temp_audio_path.exists():
        return None

    return {
        'skip': False,
        'temp_audio_path': str(temp_audio_path),
        'final_latent_path': str(final_latent_path),
        'shift': shift,
        'source_hash': source_hash,
        'segment': segment,
    }


class DCEAEncoder:
    """DCAE encoder only (no decoder needed)."""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)
        self.dcae = None

    def load(self):
        if self.dcae is not None:
            return

        from do.pipeline_do import DoTrainComponents

        device_id = 0
        if str(self.device).startswith('cuda:'):
            device_id = int(str(self.device).split(':')[1])

        components = DoTrainComponents(
            checkpoint_dir='/home/arlo/Data/ACE-Step/checkpoints',
            device_id=device_id,
        )
        self.dcae = components.load_dcae()
        print(f"Loaded DCAE encoder on {self.device}")

    @torch.no_grad()
    def encode_batch(self, audio_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Encode a batch of audio tensors to latents.

        Args:
            audio_list: List of [2, T] audio tensors at 44.1kHz

        Returns:
            List of [C, H, T] latent tensors
        """
        self.load()

        if not audio_list:
            return []

        # Find max length for padding
        max_len = max(a.shape[-1] for a in audio_list)

        # Pad and stack into batch
        padded = []
        lengths = []
        for audio in audio_list:
            lengths.append(audio.shape[-1])
            if audio.shape[-1] < max_len:
                pad = torch.zeros(2, max_len - audio.shape[-1])
                audio = torch.cat([audio, pad], dim=-1)
            padded.append(audio)

        batch = torch.stack(padded, dim=0).to(self.device)  # [B, 2, T]

        # Encode
        latents, _ = self.dcae.encode(batch, sr=DCAE_SR)

        # Unpad latents based on original lengths
        results = []
        for i, orig_len in enumerate(lengths):
            # Approximate latent length from audio length
            latent_len = orig_len // SAMPLES_PER_FRAME
            latent = latents[i, :, :, :latent_len].cpu()
            results.append(latent)

        return results


def main():
    parser = argparse.ArgumentParser(description="Precompute DSP-shifted latents (efficient)")

    parser.add_argument('--segments', type=str, required=True,
                        help='Path to segments JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for shifted latents')
    parser.add_argument('--shifts', type=str, default='-12,-6,-3,-1,0,1,3,6,12',
                        help='Comma-separated shift values in semitones')
    parser.add_argument('--max_segments', type=int, default=None,
                        help='Limit number of segments to process')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel sox workers')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for GPU encoding')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'shifted_latents').mkdir(exist_ok=True)

    # Create temp directory for shifted audio
    temp_dir = output_dir / 'temp_audio'
    temp_dir.mkdir(exist_ok=True)

    shifts = [int(s) for s in args.shifts.split(',')]
    print(f"Shifts to compute: {shifts}")

    # Load segments
    print(f"Loading segments from: {args.segments}")
    with open(args.segments) as f:
        data = json.load(f)

    # Collect all segments
    all_segments = []
    for group_id, segments in data['segments_by_group'].items():
        for seg in segments:
            seg['group'] = int(group_id)
            all_segments.append(seg)

    print(f"Total segments: {len(all_segments)}")

    if args.max_segments:
        all_segments = all_segments[:args.max_segments]
        print(f"Limited to {len(all_segments)} segments")

    # Prepare all jobs (segment, shift pairs)
    jobs = []
    for seg in all_segments:
        for shift in shifts:
            jobs.append((seg, shift, output_dir, temp_dir))

    print(f"Total jobs: {len(jobs)} ({len(all_segments)} segments x {len(shifts)} shifts)")

    # Phase 1: Parallel sox pitch shifting (CPU-bound)
    print(f"\nPhase 1: Sox pitch shifting ({args.num_workers} workers)...")

    pending_encodes = []  # Jobs that need GPU encoding
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_segment_sox, job): job for job in jobs}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Sox shifting"):
            result = future.result()
            if result is None:
                failed += 1
            elif result.get('skip'):
                skipped += 1
            else:
                pending_encodes.append(result)

    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Pending encodes: {len(pending_encodes)}")

    if not pending_encodes:
        print("Nothing to encode!")
        return

    # Phase 2: Batch GPU encoding
    print(f"\nPhase 2: DCAE encoding (batch_size={args.batch_size})...")

    encoder = DCEAEncoder(device=args.device)
    encoder.load()

    manifest_entries = []

    # Process in batches
    for batch_start in tqdm(range(0, len(pending_encodes), args.batch_size), desc="Encoding"):
        batch = pending_encodes[batch_start:batch_start + args.batch_size]

        # Load audio for this batch
        audio_list = []
        valid_items = []

        for item in batch:
            try:
                audio, sr = torchaudio.load(item['temp_audio_path'])

                # Resample to 44.1kHz if needed
                if sr != DCAE_SR:
                    audio = torchaudio.functional.resample(audio, sr, DCAE_SR)

                # Ensure stereo
                if audio.shape[0] == 1:
                    audio = audio.repeat(2, 1)
                elif audio.shape[0] > 2:
                    audio = audio[:2]

                audio_list.append(audio)
                valid_items.append(item)
            except Exception as e:
                continue

        if not audio_list:
            continue

        # Encode batch
        try:
            latents = encoder.encode_batch(audio_list)
        except Exception as e:
            print(f"Encoding error: {e}")
            continue

        # Save latents
        for item, latent in zip(valid_items, latents):
            output_path = Path(item['final_latent_path'])
            output_path.parent.mkdir(parents=True, exist_ok=True)

            seg = item['segment']
            torch.save({
                'latent': latent,
                'shift': item['shift'],
                'source_hash': item['source_hash'],
                'source_latent_path': seg.get('latent_path', ''),
                'source_start_frame': seg['start_frame'],
                'source_end_frame': seg['end_frame'],
            }, output_path)

            # Clean up temp audio
            try:
                os.remove(item['temp_audio_path'])
            except:
                pass

    # Build manifest from all saved files
    print("\nBuilding manifest...")

    # Group by source segment
    entries_by_source = {}

    for item in pending_encodes:
        seg = item['segment']
        key = f"{seg.get('latent_path', '')}:{seg['start_frame']}:{seg['end_frame']}"

        if key not in entries_by_source:
            entries_by_source[key] = {
                'source_latent_path': seg.get('latent_path', ''),
                'start_frame': seg['start_frame'],
                'end_frame': seg['end_frame'],
                'group': seg.get('group', 0),
                'median_midi': seg.get('median_midi', 0),
                'shifted_latents': {},
            }

        # Check if the latent file exists
        latent_path = item.get('final_latent_path') or str(
            output_dir / 'shifted_latents' / f"{item['source_hash']}_shift{item['shift']:+d}.pt"
        )
        if os.path.exists(latent_path):
            entries_by_source[key]['shifted_latents'][str(item['shift'])] = latent_path

    # Also add skipped (pre-existing) entries
    for seg in all_segments:
        for shift in shifts:
            key = f"{seg.get('latent_path', '')}:{seg['start_frame']}:{seg['end_frame']}"
            source_hash = get_source_hash(seg.get('latent_path', ''), seg['start_frame'], seg['end_frame'])
            latent_path = output_dir / 'shifted_latents' / f"{source_hash}_shift{shift:+d}.pt"

            if latent_path.exists():
                if key not in entries_by_source:
                    entries_by_source[key] = {
                        'source_latent_path': seg.get('latent_path', ''),
                        'start_frame': seg['start_frame'],
                        'end_frame': seg['end_frame'],
                        'group': seg.get('group', 0),
                        'median_midi': seg.get('median_midi', 0),
                        'shifted_latents': {},
                    }
                entries_by_source[key]['shifted_latents'][str(shift)] = str(latent_path)

    manifest = {
        'source_segments': args.segments,
        'shifts': shifts,
        'entries': list(entries_by_source.values()),
    }

    manifest_path = output_dir / 'shifted_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! Processed {len(manifest['entries'])} segments")
    print(f"Manifest saved to: {manifest_path}")

    # Stats
    total_shifted = sum(len(e['shifted_latents']) for e in manifest['entries'])
    print(f"Total shifted latents: {total_shifted}")

    # Clean up temp directory
    try:
        for f in temp_dir.iterdir():
            f.unlink()
        temp_dir.rmdir()
    except:
        pass


if __name__ == "__main__":
    main()
