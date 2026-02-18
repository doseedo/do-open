#!/usr/bin/env python3
"""
Precompute dry audio segments for existing wet samples.

Reads the existing manifest and extracts matching dry segments from source files,
saving them locally so training doesn't need to load from gcsfuse.
"""

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_single_entry(args):
    """Process a single manifest entry - extract and save dry segment."""
    import torch
    import torchaudio

    entry, dry_dir, sample_rate = args

    file_id = entry['id']
    source_path = entry['source_path']
    segment_start = entry['segment_start']
    segment_length = entry['segment_length']
    needs_padding = entry.get('needs_padding', False)

    dry_path = dry_dir / f"{file_id}.wav"

    # Skip if already exists
    if dry_path.exists():
        return {'id': file_id, 'status': 'skipped', 'dry_path': str(dry_path)}

    try:
        # Load full audio from source
        waveform, sr = torchaudio.load(source_path)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
            # Adjust segment positions for new sample rate
            ratio = sample_rate / sr
            segment_start = int(segment_start * ratio)
            segment_length = int(segment_length * ratio)

        # Extract segment
        if needs_padding:
            segment = waveform
        else:
            end_pos = min(segment_start + segment_length, waveform.size(-1))
            segment = waveform[..., segment_start:end_pos]

        # Ensure correct length
        target_length = 144000  # 3 seconds at 48kHz
        if segment.size(-1) < target_length:
            padding = target_length - segment.size(-1)
            segment = torch.nn.functional.pad(segment, (0, padding))
        elif segment.size(-1) > target_length:
            segment = segment[..., :target_length]

        # Normalize (match wet audio normalization)
        max_val = segment.abs().max()
        if max_val > 0:
            segment = segment / (max_val + 1e-8)
        segment = segment * 0.9

        # Save
        torchaudio.save(str(dry_path), segment, sample_rate)

        return {'id': file_id, 'status': 'success', 'dry_path': str(dry_path)}

    except Exception as e:
        return {'id': file_id, 'status': 'failed', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Precompute dry audio for existing wet samples")

    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default="/mnt/models/inverse_afx_data",
        help="Directory containing manifest.json and wet/ folder",
    )
    parser.add_argument(
        "--num_workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=48000,
        help="Target sample rate",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    dry_dir = data_dir / "dry"

    # Create dry directory
    dry_dir.mkdir(exist_ok=True)

    # Load manifest
    print(f"Loading manifest from {manifest_path}...")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    print(f"Found {len(manifest)} entries")

    # Check how many already exist
    existing = sum(1 for e in manifest if (dry_dir / f"{e['id']}.wav").exists())
    print(f"Already computed: {existing}")
    print(f"Need to compute: {len(manifest) - existing}")

    if existing == len(manifest):
        print("All dry audio already computed!")
        return

    # Prepare args for parallel processing
    process_args = [
        (entry, dry_dir, args.sample_rate)
        for entry in manifest
    ]

    # Process in parallel
    from tqdm import tqdm

    success = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_single_entry, arg): arg[0]['id']
            for arg in process_args
        }

        with tqdm(total=len(futures), desc="Extracting dry audio") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result['status'] == 'success':
                    success += 1
                elif result['status'] == 'skipped':
                    skipped += 1
                else:
                    failed += 1

                pbar.update(1)
                pbar.set_postfix({'success': success, 'skipped': skipped, 'failed': failed})

    print(f"\nDone!")
    print(f"  Success: {success}")
    print(f"  Skipped (already existed): {skipped}")
    print(f"  Failed: {failed}")

    # Update manifest to include dry_path
    print("\nUpdating manifest with dry paths...")
    for entry in manifest:
        entry['dry_path'] = str(dry_dir / f"{entry['id']}.wav")

    # Save updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Updated manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
