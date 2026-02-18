#!/usr/bin/env python3
"""
Generate synthetic training data for Inverse Audio Effects.

Uses dry recordings from the combined manifest and applies random effect chains
to create paired dry/wet training data.

NOTE: Imports are deferred to avoid argparse conflicts with CLAP/FAD packages.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

# NOTE: tqdm, torch, torchaudio, and inverse_afx imports are deferred to after
# argparse to avoid CLAP/FAD argparse hijacking issues


# Groups to include (typically dry recordings)
ALLOWED_GROUPS = {
    'brass',    # trumpet, trombone, french_horn, tuba
    'winds',    # flute, clarinet, oboe, sax, bassoon
    'strings',  # violin, viola, cello
    'voice',    # vocals
    'drums',    # drum recordings
}

# Groups to exclude (often have effects baked in)
EXCLUDED_GROUPS = {
    'guitar',   # Often has amp/effects
    'piano',    # Sometimes has reverb/processing
    'synth',    # Synthetic, not natural dry
    'fx',       # Effects/sound design
    'room',     # Room mics (have natural reverb)
    'click',    # Click tracks
    'undefined', # Unknown sources
}


@dataclass
class GenerationConfig:
    """Configuration for data generation."""
    manifest_path: str = "/home/arlo/gcs-bucket/Manifests/combined_manifest.json"
    output_dir: str = "./generated_data"
    sample_rate: int = 48000  # 48kHz to match most source files and avoid resampling
    segment_length: int = 144000  # 3 seconds at 48kHz
    min_segment_length: int = 48000  # Minimum 1 second at 48kHz
    max_chain_length: int = 4
    effect_types: List[str] = None
    samples_per_file: int = 1  # Number of segments to extract per audio file
    max_files: Optional[int] = None  # Limit number of files to process
    num_workers: int = 16
    allowed_groups: List[str] = None
    seed: int = 42
    balanced: bool = False  # Generate balanced data across magnitude tiers

    def __post_init__(self):
        if self.effect_types is None:
            self.effect_types = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']
        if self.allowed_groups is None:
            self.allowed_groups = list(ALLOWED_GROUPS)


def load_manifest(manifest_path: str) -> Dict[str, Dict]:
    """Load and parse the manifest file."""
    print(f"Loading manifest from {manifest_path}...")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    print(f"Loaded {len(manifest)} entries")
    return manifest


def filter_manifest(
    manifest: Dict[str, Dict],
    allowed_groups: List[str],
) -> Dict[str, Dict]:
    """Filter manifest to only include allowed groups."""
    filtered = {}
    group_counts = {}

    for path, info in manifest.items():
        group = info.get('group', 'undefined')

        if group in allowed_groups:
            filtered[path] = info

            if group not in group_counts:
                group_counts[group] = 0
            group_counts[group] += 1

    print(f"\nFiltered to {len(filtered)} entries from allowed groups:")
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count} files")

    return filtered


def load_audio(
    path: str,
    target_sr: int = 44100,
):
    """Load audio file and resample if needed.

    Returns: (waveform, None) on success, (None, error_string) on failure
    """
    import torch
    import torchaudio
    import os

    # First check if file exists
    if not os.path.exists(path):
        return None, "file_not_found"

    try:
        # Load audio directly (skip info call to reduce overhead)
        waveform, sr = torchaudio.load(path)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Only resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        return waveform, None

    except Exception as e:
        return None, str(e)[:50]


def extract_segment(
    waveform,
    segment_length: int,
    min_length: int,
):
    """Extract a random segment from waveform. Returns (segment, start_sample, needs_padding)."""
    import torch

    length = waveform.size(-1)

    if length < min_length:
        return None, None, None

    if length <= segment_length:
        # Pad if too short
        padding = segment_length - length
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform, 0, True
    else:
        # Random crop
        start = random.randint(0, length - segment_length)
        waveform = waveform[..., start:start + segment_length]
        return waveform, start, False


def normalize_audio(waveform):
    """Normalize audio to prevent clipping."""
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / (max_val + 1e-8)
    return waveform * 0.9  # Leave some headroom


def process_single_file(
    args: Tuple[int, str, Dict, GenerationConfig, str],
) -> Tuple[List[Dict], Optional[str]]:
    """Process a single audio file and generate wet/dry pairs.

    Returns: (results, error_reason) - error_reason is None on success
    """
    # Import here for multiprocessing worker compatibility
    # Clear sys.argv to prevent CLAP argparse from hijacking in worker processes
    import sys
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]  # Keep only script name, remove arguments

    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from inverse_afx.data.synthetic_chain_generator import EffectChainGenerator

    # Restore sys.argv after import
    sys.argv = original_argv

    idx, path, info, config, magnitude_tier = args

    results = []

    # Load audio
    waveform, load_error = load_audio(path, config.sample_rate)

    if waveform is None:
        return results, f"load:{load_error}"

    # Initialize chain generator (per-process to avoid CUDA issues)
    chain_gen = EffectChainGenerator(
        sample_rate=config.sample_rate,
        max_chain_length=config.max_chain_length,
        effect_types=config.effect_types,
        device='cpu',  # Use CPU for data generation
    )

    for seg_idx in range(config.samples_per_file):
        # Extract segment (returns segment, start_sample, needs_padding)
        segment, segment_start, needs_padding = extract_segment(
            waveform,
            config.segment_length,
            config.min_segment_length,
        )

        if segment is None:
            return results, "segment_too_short"

        # Normalize
        segment = normalize_audio(segment)

        # Check for silence
        if segment.abs().max() < 0.01:
            return results, "silent"

        # Generate wet audio
        try:
            segment_3d = segment.unsqueeze(0)  # [1, 1, T]
            wet, chain_spec, _ = chain_gen.generate_sample(
                segment_3d,
                magnitude_tier=magnitude_tier,
            )
            wet = wet.squeeze(0)  # [1, T]

            # Normalize wet audio too
            wet = normalize_audio(wet)

            # Create result entry - store both dry and wet audio
            result = {
                'idx': idx,
                'seg_idx': seg_idx,
                'source_path': path,
                'segment_start': segment_start,
                'segment_length': config.segment_length,
                'needs_padding': needs_padding,
                'group': info.get('group', 'unknown'),
                'subgroup': info.get('subgroup', 'unknown'),
                'dry_audio': segment.numpy(),
                'wet_audio': wet.numpy(),
                'chain_spec': chain_spec.to_list(),
                'chain_length': len(chain_spec),
                'magnitude_tier': magnitude_tier,
            }

            results.append(result)

        except Exception as e:
            return results, f"effect_chain_error: {e}"

    return results, None


def generate_dataset(config: GenerationConfig, resume: bool = False):
    """Generate the full dataset with incremental saving."""
    import os
    import torch
    import torchaudio
    from tqdm import tqdm

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load and filter manifest
    manifest = load_manifest(config.manifest_path)
    manifest = filter_manifest(manifest, config.allowed_groups)

    # Limit files if specified
    file_paths = list(manifest.keys())
    if config.max_files is not None:
        random.shuffle(file_paths)
        file_paths = file_paths[:config.max_files]

    # Resume support: skip already processed files
    already_processed = set()
    start_sample_counter = 0
    if resume:
        output_dir = Path(config.output_dir)
        existing_manifest_path = output_dir / "manifest.json"
        if existing_manifest_path.exists():
            with open(existing_manifest_path, 'r') as f:
                existing_entries = json.load(f)
            already_processed = {e['source_path'] for e in existing_entries}
            start_sample_counter = len(existing_entries)
            print(f"\nResuming: found {len(existing_entries)} existing samples")
            print(f"Skipping {len(already_processed)} already-processed source files")
            file_paths = [p for p in file_paths if p not in already_processed]

    # Skip pre-filtering - handle missing files during processing (faster on GCS)
    print(f"\nWill process {len(file_paths)} files (missing files handled during processing)")

    print(f"\nProcessing {len(file_paths)} files...")

    # Create output directory and subdirs for dry and wet audio
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dry_dir = output_dir / "dry"
    wet_dir = output_dir / "wet"
    dry_dir.mkdir(exist_ok=True)
    wet_dir.mkdir(exist_ok=True)

    # Prepare arguments for parallel processing with magnitude tiers
    MAGNITUDE_TIERS = ['subtle', 'mild', 'moderate', 'strong']

    process_args = []
    for idx, path in enumerate(file_paths):
        # Cycle through tiers for balanced data, or use 'random' for legacy
        if getattr(config, 'balanced', False):
            tier = MAGNITUDE_TIERS[idx % len(MAGNITUDE_TIERS)]
        else:
            tier = 'random'
        process_args.append((idx, path, manifest[path], config, tier))

    # Process files and save incrementally
    # Load existing manifest entries if resuming
    if resume and start_sample_counter > 0:
        with open(output_dir / "manifest.json", 'r') as f:
            manifest_entries = json.load(f)
    else:
        manifest_entries = []
    failed_count = 0
    error_counts = {}  # Track error reasons
    sample_counter = start_sample_counter
    manifest_save_interval = 1000  # Save manifest every N samples
    batch_size = 500  # Process in batches to avoid memory exhaustion

    def save_manifest_checkpoint():
        """Save manifest checkpoint."""
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest_entries, f, indent=2)

    # Process in batches to avoid memory exhaustion
    num_batches = (len(process_args) + batch_size - 1) // batch_size

    with tqdm(total=len(process_args), desc="Generating data") as pbar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(process_args))
            batch_args = process_args[batch_start:batch_end]

            # Process this batch
            with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
                futures = {
                    executor.submit(process_single_file, args): args[0]
                    for args in batch_args
                }

                for future in as_completed(futures):
                    try:
                        results, error_reason = future.result()

                        if error_reason:
                            # Track error reason (truncate for display)
                            err_key = error_reason[:40]
                            error_counts[err_key] = error_counts.get(err_key, 0) + 1
                            failed_count += 1
                        else:
                            # Save each result immediately (both dry and wet audio)
                            for result in results:
                                file_id = f"{sample_counter:06d}"

                                # Save dry audio
                                dry_path = dry_dir / f"{file_id}.wav"
                                dry_tensor = torch.from_numpy(result['dry_audio'])
                                torchaudio.save(str(dry_path), dry_tensor, config.sample_rate)

                                # Save wet audio
                                wet_path = wet_dir / f"{file_id}.wav"
                                wet_tensor = torch.from_numpy(result['wet_audio'])
                                torchaudio.save(str(wet_path), wet_tensor, config.sample_rate)

                                # Add to manifest
                                entry = {
                                    'id': file_id,
                                    'dry_path': str(dry_path),
                                    'wet_path': str(wet_path),
                                    'source_path': result['source_path'],
                                    'group': result['group'],
                                    'subgroup': result['subgroup'],
                                    'chain_spec': result['chain_spec'],
                                    'chain_length': result['chain_length'],
                                    'magnitude_tier': result.get('magnitude_tier', 'random'),
                                }
                                manifest_entries.append(entry)
                                sample_counter += 1

                                # Save manifest checkpoint periodically
                                if sample_counter % manifest_save_interval == 0:
                                    save_manifest_checkpoint()

                    except Exception as e:
                        error_counts['exception'] = error_counts.get('exception', 0) + 1
                        failed_count += 1
                    pbar.update(1)
                    # Show top error reason in progress bar
                    top_err = max(error_counts.items(), key=lambda x: x[1]) if error_counts else ("", 0)
                    pbar.set_postfix({
                        'samples': sample_counter,
                        'failed': failed_count,
                        'top_err': f"{top_err[0]}({top_err[1]})",
                        'batch': f"{batch_idx+1}/{num_batches}"
                    })

            # Executor is closed here, workers are terminated, memory freed

    print(f"\nGenerated {sample_counter} samples from {len(file_paths)} files")
    print(f"Failed to process {failed_count} files")
    if error_counts:
        print("\nError breakdown:")
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {err}: {count}")

    # Save final manifest and config
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest_entries, f, indent=2)

    config_path = output_dir / "generation_config.json"
    config_dict = {
        'sample_rate': config.sample_rate,
        'segment_length': config.segment_length,
        'max_chain_length': config.max_chain_length,
        'effect_types': config.effect_types,
        'allowed_groups': config.allowed_groups,
        'num_samples': sample_counter,
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Print statistics
    print(f"\nDataset saved to {output_dir}")
    print(f"  Total samples: {sample_counter}")

    if manifest_entries:
        chain_lengths = [e['chain_length'] for e in manifest_entries]
        effect_counts = {}
        for e in manifest_entries:
            for effect_type, _ in e['chain_spec']:
                effect_counts[effect_type] = effect_counts.get(effect_type, 0) + 1

        print(f"\nChain length distribution:")
        for length in range(1, config.max_chain_length + 1):
            count = chain_lengths.count(length)
            if count > 0:
                print(f"  {length} effects: {count} ({100*count/len(manifest_entries):.1f}%)")

        print(f"\nEffect usage:")
        for effect, count in sorted(effect_counts.items()):
            print(f"  {effect}: {count}")

    return manifest_entries


def save_dataset(
    results: List[Dict],
    output_dir: Path,
    config: GenerationConfig,
):
    """Save generated dataset to disk."""
    import torch
    import torchaudio
    from tqdm import tqdm

    print(f"\nSaving dataset to {output_dir}...")

    # Create subdirectories
    dry_dir = output_dir / "dry"
    wet_dir = output_dir / "wet"
    dry_dir.mkdir(exist_ok=True)
    wet_dir.mkdir(exist_ok=True)

    # Prepare manifest
    manifest_entries = []

    for i, result in enumerate(tqdm(results, desc="Saving files")):
        file_id = f"{i:06d}"

        # Save dry audio
        dry_path = dry_dir / f"{file_id}.wav"
        dry_tensor = torch.from_numpy(result['dry_audio'])
        torchaudio.save(str(dry_path), dry_tensor, config.sample_rate)

        # Save wet audio
        wet_path = wet_dir / f"{file_id}.wav"
        wet_tensor = torch.from_numpy(result['wet_audio'])
        torchaudio.save(str(wet_path), wet_tensor, config.sample_rate)

        # Manifest entry
        entry = {
            'id': file_id,
            'dry_path': str(dry_path),
            'wet_path': str(wet_path),
            'source_path': result['source_path'],
            'group': result['group'],
            'subgroup': result['subgroup'],
            'chain_spec': result['chain_spec'],
            'chain_length': result['chain_length'],
        }
        manifest_entries.append(entry)

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest_entries, f, indent=2)

    # Save config
    config_path = output_dir / "generation_config.json"
    config_dict = {
        'sample_rate': config.sample_rate,
        'segment_length': config.segment_length,
        'max_chain_length': config.max_chain_length,
        'effect_types': config.effect_types,
        'allowed_groups': config.allowed_groups,
        'num_samples': len(results),
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Print statistics
    print(f"\nDataset saved:")
    print(f"  Manifest: {manifest_path}")
    print(f"  Dry audio: {dry_dir}")
    print(f"  Wet audio: {wet_dir}")
    print(f"  Total samples: {len(results)}")

    # Print chain statistics
    chain_lengths = [r['chain_length'] for r in results]
    effect_counts = {}
    for r in results:
        for effect_type, _ in r['chain_spec']:
            effect_counts[effect_type] = effect_counts.get(effect_type, 0) + 1

    print(f"\nChain length distribution:")
    for length in range(1, config.max_chain_length + 1):
        count = chain_lengths.count(length)
        print(f"  {length} effects: {count} ({100*count/len(results):.1f}%)")

    print(f"\nEffect usage:")
    for effect, count in sorted(effect_counts.items()):
        print(f"  {effect}: {count}")


def main():
    # Parse arguments FIRST before any imports that could trigger CLAP/FAD argparse
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for Inverse AFx"
    )

    parser.add_argument(
        "--manifest", "-m",
        type=str,
        default="/home/arlo/gcs-bucket/Manifests/combined_manifest.json",
        help="Path to source manifest file",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./generated_data",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--max_files", "-n",
        type=int,
        default=None,
        help="Maximum number of source files to process",
    )
    parser.add_argument(
        "--samples_per_file", "-s",
        type=int,
        default=1,
        help="Number of segments to extract per source file",
    )
    parser.add_argument(
        "--max_chain_length", "-c",
        type=int,
        default=4,
        help="Maximum effect chain length",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=144000,
        help="Segment length in samples (default: 144000 = 3s at 48kHz)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=48000,
        help="Target sample rate (default: 48000 to avoid resampling)",
    )
    parser.add_argument(
        "--num_workers", "-w",
        type=int,
        default=16,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["brass", "winds", "strings", "voice", "drums"],
        help="Instrument groups to include",
    )
    parser.add_argument(
        "--effects",
        type=str,
        nargs="+",
        default=["eq", "compressor", "reverb", "distortion", "chorus", "delay"],
        help="Effect types to use",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from existing progress (skip already processed files)",
    )
    parser.add_argument(
        "--balanced", "-b",
        action="store_true",
        help="Generate balanced data across magnitude tiers (subtle/mild/moderate/strong)",
    )

    args = parser.parse_args()

    # Create config (imports are deferred inside each function to avoid CLAP/FAD argparse hijacking)
    config = GenerationConfig(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        max_chain_length=args.max_chain_length,
        effect_types=args.effects,
        samples_per_file=args.samples_per_file,
        max_files=args.max_files,
        num_workers=args.num_workers,
        allowed_groups=args.groups,
        seed=args.seed,
        balanced=args.balanced,
    )

    print("=" * 60)
    print("Inverse AFx Data Generation")
    print("=" * 60)
    print(f"Manifest: {config.manifest_path}")
    print(f"Output: {config.output_dir}")
    print(f"Groups: {', '.join(config.allowed_groups)}")
    print(f"Effects: {', '.join(config.effect_types)}")
    print(f"Max chain length: {config.max_chain_length}")
    print(f"Segment length: {config.segment_length} samples ({config.segment_length/config.sample_rate:.2f}s)")
    print(f"Max files: {config.max_files or 'all'}")
    print(f"Workers: {config.num_workers}")
    print(f"Resume: {args.resume}")
    print(f"Balanced tiers: {args.balanced}")
    print("=" * 60)

    # Generate dataset
    generate_dataset(config, resume=args.resume)

    print("\nDone!")


if __name__ == "__main__":
    main()
