#!/usr/bin/env python3
"""
Add Phoneme Targets to Existing Lyric Processing Outputs

This script processes existing lyrics.py outputs and adds phoneme-level targets
for training a phoneme prediction head. Uses G2P + Whisper word timings for
fast approximation (no MFA needed).

Usage:
    # Process all existing outputs
    python add_phoneme_targets.py --process_all

    # Process specific file
    python add_phoneme_targets.py --file /mnt/msdd/vocal_processing/sample_001/sample_001_ace_step_data.pt

    # Dry run to see what would be processed
    python add_phoneme_targets.py --process_all --dry_run
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchaudio
import librosa
from tqdm import tqdm
from datetime import datetime
import multiprocessing as mp
from functools import partial

# Phoneme processing
try:
    from g2p_en import G2p
    G2P_AVAILABLE = True
    print("✅ G2P available")
except ImportError:
    G2P_AVAILABLE = False
    print("❌ G2P not available. Install with: pip install g2p-en")
    sys.exit(1)

# Configuration
OUTPUT_DIR = Path("/mnt/msdd/vocal_processing")
SAMPLE_RATE = 44100
DCAE_SR = 44100
DCAE_HOP = 4096
SLOW_HZ = DCAE_SR / DCAE_HOP  # ~10.77 Hz

# ARPAbet phoneme set (CMU pronunciation dictionary)
ARPABET_PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
    'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
    'SIL', 'SP'  # Silence and short pause
]

# Create phoneme to index mapping
PHONEME_TO_IDX = {p: i for i, p in enumerate(ARPABET_PHONEMES)}
PHONEME_TO_IDX['<PAD>'] = len(ARPABET_PHONEMES)
PHONEME_TO_IDX['<UNK>'] = len(ARPABET_PHONEMES) + 1
NUM_PHONEMES = len(PHONEME_TO_IDX)

# Reverse mapping for debugging
IDX_TO_PHONEME = {i: p for p, i in PHONEME_TO_IDX.items()}

print(f"📋 Phoneme vocabulary size: {NUM_PHONEMES}")


class PhonemeTargetGenerator:
    """
    Generates phoneme targets from existing Whisper word timings.
    Fast approximation using G2P + interpolation (no MFA needed).
    """

    def __init__(self):
        """Initialize G2P converter."""
        if not G2P_AVAILABLE:
            raise ImportError("G2P is required. Install with: pip install g2p-en")

        self.g2p = G2p()
        print("✅ Phoneme target generator initialized")

    def word_timings_to_phonemes(
        self,
        word_timings: List[Tuple[float, float, str]],
        audio_duration: float
    ) -> Dict[str, Any]:
        """
        Convert word timings to phoneme frames for loss computation.

        Args:
            word_timings: List of (start_time, end_time, word) tuples from Whisper
            audio_duration: Total audio duration in seconds

        Returns:
            Dict containing:
                - phoneme_frames: [T_slow] array of phoneme indices
                - phoneme_boundaries: [T_slow] binary array marking phoneme onsets
                - phoneme_timings: List of (start, end, phoneme) tuples
                - phoneme_confidence: [T_slow] confidence scores (1.0 for interpolated)
        """
        # Calculate frame dimensions at DCAE slow rate
        T_slow = int(np.ceil(audio_duration * SLOW_HZ))

        # Initialize arrays
        phoneme_frames = np.full(T_slow, PHONEME_TO_IDX['<PAD>'], dtype=np.int64)
        phoneme_boundaries = np.zeros(T_slow, dtype=np.float32)
        phoneme_confidence = np.zeros(T_slow, dtype=np.float32)

        all_phoneme_timings = []

        # Add silence at start if needed
        if word_timings and word_timings[0][0] > 0.1:
            start_frames = int(word_timings[0][0] * SLOW_HZ)
            phoneme_frames[:start_frames] = PHONEME_TO_IDX['SIL']
            phoneme_boundaries[0] = 1.0
            phoneme_confidence[:start_frames] = 0.8

        # Process each word
        for word_start, word_end, word_text in word_timings:
            # Clean word text
            word_text = word_text.strip().upper()
            if not word_text:
                continue

            # Convert word to phonemes using G2P
            try:
                phonemes = self.g2p(word_text)
            except Exception as e:
                print(f"⚠ G2P failed for '{word_text}': {e}")
                continue

            # Filter to valid ARPAbet phonemes (remove stress markers, etc.)
            valid_phonemes = []
            for p in phonemes:
                # Remove stress markers (0,1,2)
                p_clean = ''.join(c for c in p if not c.isdigit())
                if p_clean in PHONEME_TO_IDX:
                    valid_phonemes.append(p_clean)
                elif p_clean == ' ':
                    valid_phonemes.append('SP')  # Short pause for spaces

            if not valid_phonemes:
                # Fallback to silence for unknown words
                valid_phonemes = ['SIL']

            # Distribute phonemes evenly across word duration
            word_duration = word_end - word_start
            phoneme_duration = word_duration / len(valid_phonemes)

            for i, phoneme in enumerate(valid_phonemes):
                ph_start = word_start + i * phoneme_duration
                ph_end = word_start + (i + 1) * phoneme_duration

                # Convert to frame indices
                start_frame = int(ph_start * SLOW_HZ)
                end_frame = int(ph_end * SLOW_HZ)

                # Clamp to valid range
                start_frame = max(0, min(start_frame, T_slow - 1))
                end_frame = max(start_frame + 1, min(end_frame, T_slow))

                # Get phoneme ID
                phoneme_id = PHONEME_TO_IDX.get(phoneme, PHONEME_TO_IDX['<UNK>'])

                # Fill frames
                phoneme_frames[start_frame:end_frame] = phoneme_id

                # Mark onset at first frame
                if start_frame < T_slow:
                    phoneme_boundaries[start_frame] = 1.0

                # Set confidence (lower than MFA since this is interpolated)
                phoneme_confidence[start_frame:end_frame] = 0.7

                # Store timing
                all_phoneme_timings.append((ph_start, ph_end, phoneme))

        # Fill remaining silence
        silence_frames = (phoneme_frames == PHONEME_TO_IDX['<PAD>'])
        if silence_frames.any():
            phoneme_frames[silence_frames] = PHONEME_TO_IDX['SIL']
            phoneme_confidence[silence_frames] = 0.5

        return {
            'phoneme_frames': phoneme_frames,
            'phoneme_boundaries': phoneme_boundaries,
            'phoneme_timings': all_phoneme_timings,
            'phoneme_confidence': phoneme_confidence,
            'num_phonemes': len(all_phoneme_timings),
        }

    def process_existing_output(
        self,
        lyrics_json_path: Path,
        tensors_pt_path: Path,
        audio_path: Path,
        dry_run: bool = False
    ) -> bool:
        """
        Process existing lyrics output to add phoneme targets.

        Args:
            lyrics_json_path: Path to *_lyrics_ace_step.json file
            tensors_pt_path: Path to *_tensors.pt file
            audio_path: Path to original audio file
            dry_run: If True, don't save, just report what would be done

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load tensors file
            tensors_data = torch.load(tensors_pt_path, map_location='cpu')

            # Check if already processed
            if 'phoneme_frames' in tensors_data and not dry_run:
                print(f"  ⏭ Already has phoneme data")
                return True

            # Load JSON file for word_timings
            with open(lyrics_json_path) as f:
                json_data = json.load(f)

            # Check for required fields
            word_timings = json_data.get('word_timings')
            if not word_timings:
                print(f"  ⚠ No word_timings found, skipping")
                return False

            if not isinstance(word_timings, list) or len(word_timings) == 0:
                print(f"  ⚠ Empty word_timings, skipping")
                return False

            # Get audio duration
            if not audio_path.exists():
                print(f"  ⚠ Audio file not found: {audio_path}")
                return False

            try:
                audio_info = torchaudio.info(str(audio_path))
                duration = audio_info.num_frames / audio_info.sample_rate
            except Exception:
                # Fallback to librosa
                try:
                    duration = librosa.get_duration(path=str(audio_path))
                except Exception as e:
                    print(f"  ⚠ Could not get audio duration: {e}")
                    return False

            if dry_run:
                print(f"  ✓ Would process: {len(word_timings)} words, {duration:.2f}s")
                return True

            # Generate phoneme targets
            phoneme_data = self.word_timings_to_phonemes(word_timings, duration)

            # Add to tensors_data
            tensors_data['phoneme_frames'] = torch.from_numpy(phoneme_data['phoneme_frames']).long()
            tensors_data['phoneme_boundaries'] = torch.from_numpy(phoneme_data['phoneme_boundaries']).float()
            tensors_data['phoneme_confidence'] = torch.from_numpy(phoneme_data['phoneme_confidence']).float()
            tensors_data['phoneme_timings'] = phoneme_data['phoneme_timings']
            tensors_data['num_phonemes'] = phoneme_data['num_phonemes']
            tensors_data['phoneme_vocab_size'] = NUM_PHONEMES

            # Save updated tensors file
            torch.save(tensors_data, tensors_pt_path)

            # Also save a separate phoneme file for easy access
            phoneme_only_path = tensors_pt_path.parent / f"{tensors_pt_path.stem.replace('_tensors', '')}_phonemes.pt"
            torch.save({
                'phoneme_frames': tensors_data['phoneme_frames'],
                'phoneme_boundaries': tensors_data['phoneme_boundaries'],
                'phoneme_confidence': tensors_data['phoneme_confidence'],
                'phoneme_timings': phoneme_data['phoneme_timings'],
                'num_phonemes': phoneme_data['num_phonemes'],
                'phoneme_vocab_size': NUM_PHONEMES,
                'audio_duration': duration,
            }, phoneme_only_path)

            print(f"  ✅ Added {phoneme_data['num_phonemes']} phonemes")
            return True

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def find_all_outputs(self, output_dir: Path, manifest_path: Path = None) -> List[Tuple[Path, Path, Path]]:
        """
        Find all existing lyrics outputs and their corresponding audio.

        Returns:
            List of (lyrics_json_path, tensors_pt_path, audio_path) tuples
        """
        outputs = []

        if manifest_path and manifest_path.exists():
            # Use manifest to find files (more reliable)
            print(f"  Using manifest: {manifest_path}")
            with open(manifest_path) as f:
                manifest = json.load(f)

            for item in manifest:
                vc_paths = item.get('vocal_conditioning_paths', {})
                lyrics_json = vc_paths.get('lyrics_data')
                lyrics_tensors = vc_paths.get('lyrics_tensors')
                audio_path = item.get('audio_path')

                if lyrics_json and lyrics_tensors and audio_path:
                    lyrics_json = Path(lyrics_json)
                    lyrics_tensors = Path(lyrics_tensors)
                    audio_path = Path(audio_path)

                    if lyrics_json.exists() and lyrics_tensors.exists() and audio_path.exists():
                        outputs.append((lyrics_json, lyrics_tensors, audio_path))
        else:
            # Fallback: scan directory for _lyrics_ace_step.json files
            print(f"  Scanning directory (no manifest provided)")
            json_files = list(output_dir.glob("*/*_lyrics_ace_step.json"))

            for json_file in json_files:
                # Find corresponding tensors file
                stem = json_file.stem.replace('_lyrics_ace_step', '')
                tensors_file = json_file.parent / f"{stem}_tensors.pt"

                # Find corresponding audio file
                audio_path = None
                for ext in ['.wav', '.mp3', '.flac', '.ogg']:
                    candidate = json_file.parent / f"{stem}{ext}"
                    if candidate.exists():
                        audio_path = candidate
                        break

                # Try loading from JSON if no audio found in same dir
                if not audio_path:
                    try:
                        with open(json_file) as f:
                            data = json.load(f)
                            orig_audio = data.get('original_audio_path')
                            if orig_audio and Path(orig_audio).exists():
                                audio_path = Path(orig_audio)
                    except Exception:
                        pass

                if tensors_file.exists() and audio_path:
                    outputs.append((json_file, tensors_file, audio_path))
                else:
                    if not tensors_file.exists():
                        print(f"⚠ No tensors file for {json_file.name}")
                    if not audio_path:
                        print(f"⚠ No audio found for {json_file.name}")

        return outputs


def process_single_file_wrapper(args_tuple):
    """
    Wrapper function for multiprocessing. Each worker creates its own generator.

    Args:
        args_tuple: (lyrics_json_path, tensors_pt_path, audio_path, dry_run)

    Returns:
        (file_name, success: bool)
    """
    lyrics_json_path, tensors_pt_path, audio_path, dry_run = args_tuple

    # Each worker needs its own G2P instance
    generator = PhonemeTargetGenerator()

    try:
        success = generator.process_existing_output(
            lyrics_json_path, tensors_pt_path, audio_path, dry_run
        )
        return (tensors_pt_path.name, success)
    except Exception as e:
        return (tensors_pt_path.name, False)


def main():
    parser = argparse.ArgumentParser(
        description="Add phoneme targets to existing lyric processing outputs"
    )
    parser.add_argument(
        '--process_all',
        action='store_true',
        help='Process all existing outputs in OUTPUT_DIR'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Process specific tensors.pt file'
    )
    parser.add_argument(
        '--audio',
        type=str,
        help='Audio file path (required with --file)'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS_SPKEMB.json',
        help='Path to training manifest JSON (default: vocal_training_manifest_READY_NOGROUPS_SPKEMB.json)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(OUTPUT_DIR),
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Maximum number of files to process (for testing)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8)'
    )

    args = parser.parse_args()

    if not args.process_all and not args.file:
        parser.error("Must specify either --process_all or --file")

    if args.file and not args.audio:
        # Try to find audio automatically
        tensors_path = Path(args.file)
        stem = tensors_path.stem.replace('_tensors', '')
        audio_dir = tensors_path.parent

        audio_path = None
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            candidate = audio_dir / f"{stem}{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if not audio_path:
            parser.error("Could not find audio file automatically. Please specify with --audio")

        args.audio = str(audio_path)

    # Initialize generator
    generator = PhonemeTargetGenerator()

    # Process files
    if args.process_all:
        output_dir = Path(args.output_dir)
        manifest_path = Path(args.manifest) if args.manifest else None

        print(f"\n🔍 Scanning for existing outputs...")

        outputs = generator.find_all_outputs(output_dir, manifest_path)
        print(f"📋 Found {len(outputs)} files to process")

        if args.max_files:
            outputs = outputs[:args.max_files]
            print(f"📋 Limited to {len(outputs)} files for testing")

        if args.dry_run:
            print("\n🔍 DRY RUN - No files will be modified\n")

        # Prepare arguments for multiprocessing
        process_args = [(json_path, tensors_path, audio_path, args.dry_run)
                        for json_path, tensors_path, audio_path in outputs]

        # Process with multiprocessing
        success = 0
        failed = 0
        num_workers = min(args.workers, mp.cpu_count())

        print(f"🚀 Processing with {num_workers} parallel workers...")
        print(f"{'='*60}\n")

        # Use multiprocessing pool with progress bar
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_single_file_wrapper, process_args),
                total=len(process_args),
                desc="Processing files",
                unit="file"
            ))

        # Count results
        for filename, result in results:
            if result:
                success += 1
            else:
                failed += 1

        # Summary
        print(f"\n{'='*60}")
        print(f"✅ Successfully processed: {success}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Total: {len(outputs)}")

        if args.dry_run:
            print("\n💡 Run without --dry_run to actually process files")

    else:
        # Process single file
        tensors_path = Path(args.file)
        audio_path = Path(args.audio)

        # Find corresponding JSON file
        json_path = tensors_path.parent / f"{tensors_path.stem.replace('_tensors', '')}_lyrics_ace_step.json"

        if not json_path.exists():
            print(f"❌ Could not find lyrics JSON: {json_path}")
            return

        print(f"\n📄 Processing: {tensors_path.name}")
        print(f"📝 Lyrics JSON: {json_path.name}")
        print(f"🎵 Audio: {audio_path.name}")

        if args.dry_run:
            print("\n🔍 DRY RUN - No files will be modified\n")

        success = generator.process_existing_output(json_path, tensors_path, audio_path, args.dry_run)

        if success:
            print("\n✅ Successfully processed")
        else:
            print("\n❌ Processing failed")


if __name__ == "__main__":
    main()
