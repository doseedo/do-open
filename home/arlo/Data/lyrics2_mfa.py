#!/usr/bin/env python3
"""
Phoneme-Level Lyric Processor with Montreal Forced Aligner (lyrics2_mfa.py)

Extracts accurate phoneme-level transcription with precise frame alignment.

Requirements:
    pip install g2p-en
    conda install -c conda-forge montreal-forced-aligner
    mfa model download acoustic english_us_arpa
    mfa model download dictionary english_us_arpa

Usage:
    python lyrics2_mfa.py --process_all
    python lyrics2_mfa.py --file "path/to/vocal.wav" --lyrics "path/to/lyrics.txt"
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from datetime import datetime

# Phoneme processing
try:
    from g2p_en import G2p
    G2P_AVAILABLE = True
except ImportError:
    G2P_AVAILABLE = False
    print("❌ G2P not available. Install with: pip install g2p-en")
    sys.exit(1)

# Whisper for lyrics extraction
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠ Whisper not available. Install with: pip install openai-whisper")

# Configuration
OUTPUT_DIR = Path("/mnt/msdd/vocal_processing_phoneme")
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

# MFA models
MFA_ACOUSTIC_MODEL = "english_us_arpa"
MFA_DICTIONARY = "english_us_arpa"


def check_mfa_installed():
    """Check if MFA is installed and models are downloaded"""
    try:
        result = subprocess.run(['mfa', 'version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ MFA not found. Install with: conda install -c conda-forge montreal-forced-aligner")
            return False
        print(f"✅ MFA version: {result.stdout.strip()}")

        # Check if models are downloaded
        model_check = subprocess.run(['mfa', 'model', 'list'], capture_output=True, text=True)
        if MFA_ACOUSTIC_MODEL not in model_check.stdout:
            print(f"⚠ Downloading MFA acoustic model: {MFA_ACOUSTIC_MODEL}")
            subprocess.run(['mfa', 'model', 'download', 'acoustic', MFA_ACOUSTIC_MODEL], check=True)

        if MFA_DICTIONARY not in model_check.stdout:
            print(f"⚠ Downloading MFA dictionary: {MFA_DICTIONARY}")
            subprocess.run(['mfa', 'model', 'download', 'dictionary', MFA_DICTIONARY], check=True)

        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ MFA check failed: {e}")
        return False


def parse_textgrid(textgrid_path: str) -> List[Tuple[float, float, str]]:
    """
    Parse Praat TextGrid file to extract phoneme timings.

    Args:
        textgrid_path: Path to TextGrid file

    Returns:
        List of (start_time, end_time, phoneme) tuples
    """
    phoneme_intervals = []

    try:
        with open(textgrid_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        in_phones_tier = False
        in_interval = False
        current_xmin = None
        current_xmax = None

        for line in lines:
            line = line.strip()

            # Find the phones tier
            if 'name = "phones"' in line:
                in_phones_tier = True
                continue

            # Exit phones tier when we hit another tier
            if in_phones_tier and 'name =' in line and 'phones' not in line:
                in_phones_tier = False
                continue

            if not in_phones_tier:
                continue

            # Parse interval data
            if line.startswith('xmin ='):
                current_xmin = float(line.split('=')[1].strip())
            elif line.startswith('xmax ='):
                current_xmax = float(line.split('=')[1].strip())
            elif line.startswith('text ='):
                phoneme = line.split('=')[1].strip().strip('"').strip()

                if current_xmin is not None and current_xmax is not None:
                    # Skip empty intervals
                    if phoneme and phoneme not in ['', 'sp', 'spn']:
                        # Remove stress markers (0, 1, 2)
                        phoneme_clean = ''.join(c for c in phoneme if not c.isdigit())
                        phoneme_intervals.append((current_xmin, current_xmax, phoneme_clean.upper()))

                current_xmin = None
                current_xmax = None

        return phoneme_intervals

    except Exception as e:
        print(f"⚠ Failed to parse TextGrid: {e}")
        return []


class PhonemeLyricProcessor:
    """Processes lyrics at phoneme level with accurate MFA alignment"""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check MFA
        if not check_mfa_installed():
            print("❌ MFA is required for accurate phoneme alignment")
            sys.exit(1)

        # Initialize G2P
        print("🔧 Loading G2P model...")
        self.g2p = G2p()
        print("✅ G2P model loaded")

        # Whisper for extraction
        self.whisper_model = None
        self._whisper_loaded = False

    def _load_whisper_if_needed(self):
        """Load Whisper model lazily"""
        if not WHISPER_AVAILABLE or self._whisper_loaded:
            return self._whisper_loaded

        try:
            print("🔧 Loading Whisper model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model("base", device=device)
            self._whisper_loaded = True
            print(f"✅ Whisper loaded on {device}")
            return True
        except Exception as e:
            print(f"❌ Whisper loading failed: {e}")
            return False

    def align_with_mfa(
        self,
        audio_path: str,
        text: str,
        audio_duration: float
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float, str]]]:
        """
        Use Montreal Forced Aligner for accurate phoneme timing.

        Args:
            audio_path: Path to audio file
            text: Text transcript
            audio_duration: Total duration in seconds

        Returns:
            phoneme_frames: [T_slow] array of phoneme indices
            phoneme_boundaries: [T_slow] binary array (1 at phoneme onsets)
            phoneme_timings: [(start, end, phoneme), ...] with accurate timing
        """
        T_slow = int(np.ceil(audio_duration * SLOW_HZ))
        phoneme_frames = np.full(T_slow, PHONEME_TO_IDX['<PAD>'], dtype=np.int64)
        phoneme_boundaries = np.zeros(T_slow, dtype=np.float32)

        # Create temporary directory for MFA
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            corpus_dir = temp_path / "corpus"
            output_dir = temp_path / "output"
            corpus_dir.mkdir()
            output_dir.mkdir()

            # Copy audio file
            audio_name = Path(audio_path).stem
            audio_ext = Path(audio_path).suffix
            temp_audio = corpus_dir / f"{audio_name}{audio_ext}"
            shutil.copy2(audio_path, temp_audio)

            # Write transcript
            transcript_path = corpus_dir / f"{audio_name}.txt"
            with open(transcript_path, 'w') as f:
                f.write(text)

            try:
                # Run MFA alignment
                print(f"  🔧 Running MFA alignment...")
                cmd = [
                    'mfa', 'align',
                    str(corpus_dir),
                    MFA_DICTIONARY,
                    MFA_ACOUSTIC_MODEL,
                    str(output_dir),
                    '--clean',
                    '--quiet'
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode != 0:
                    print(f"  ⚠ MFA alignment failed: {result.stderr}")
                    return phoneme_frames, phoneme_boundaries, []

                # Parse TextGrid output
                textgrid_path = output_dir / f"{audio_name}.TextGrid"
                if not textgrid_path.exists():
                    print(f"  ⚠ TextGrid not found: {textgrid_path}")
                    return phoneme_frames, phoneme_boundaries, []

                phoneme_timings = parse_textgrid(str(textgrid_path))

                if not phoneme_timings:
                    print(f"  ⚠ No phoneme timings extracted")
                    return phoneme_frames, phoneme_boundaries, []

                # Convert phoneme timings to frame arrays
                for start_time, end_time, phoneme in phoneme_timings:
                    start_frame = int(start_time * SLOW_HZ)
                    end_frame = int(end_time * SLOW_HZ)

                    # Clamp to valid range
                    start_frame = max(0, min(start_frame, T_slow - 1))
                    end_frame = max(start_frame + 1, min(end_frame, T_slow))

                    # Get phoneme ID
                    phoneme_id = PHONEME_TO_IDX.get(phoneme, PHONEME_TO_IDX['<UNK>'])

                    # Fill frames
                    phoneme_frames[start_frame:end_frame] = phoneme_id

                    # Mark onset
                    phoneme_boundaries[start_frame] = 1.0

                print(f"  ✅ MFA aligned {len(phoneme_timings)} phonemes")
                return phoneme_frames, phoneme_boundaries, phoneme_timings

            except subprocess.TimeoutExpired:
                print(f"  ⚠ MFA alignment timed out")
                return phoneme_frames, phoneme_boundaries, []
            except Exception as e:
                print(f"  ⚠ MFA alignment error: {e}")
                return phoneme_frames, phoneme_boundaries, []

    def extract_lyrics_from_audio(self, audio_path: str) -> Tuple[str, List[Tuple[float, float, str]]]:
        """Extract lyrics and word timings using Whisper"""
        if not self._load_whisper_if_needed():
            return "", []

        try:
            result = self.whisper_model.transcribe(
                audio_path,
                word_timestamps=True,
                language='en'
            )

            word_timings = []
            for segment in result.get('segments', []):
                for word_info in segment.get('words', []):
                    start = word_info.get('start', 0.0)
                    end = word_info.get('end', 0.0)
                    word = word_info.get('word', '').strip()
                    if word:
                        word_timings.append((start, end, word))

            full_text = result.get('text', '')
            return full_text, word_timings

        except Exception as e:
            print(f"⚠ Whisper extraction failed: {e}")
            return "", []

    def process_vocal_file(
        self,
        audio_path: str,
        lyrics_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a vocal file and extract phoneme-level data with MFA.

        Args:
            audio_path: Path to vocal audio file
            lyrics_text: Optional lyrics text (if None, extract with Whisper)

        Returns:
            Dictionary with phoneme data paths
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            print(f"  ❌ Audio file not found: {audio_path}")
            return None

        # Get audio duration
        try:
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
        except Exception as e:
            print(f"  ⚠ Failed to get audio duration: {e}")
            return None

        # Extract lyrics if not provided
        if lyrics_text is None:
            print(f"  🔧 Extracting lyrics from audio...")
            lyrics_text, _ = self.extract_lyrics_from_audio(str(audio_path))

        if not lyrics_text:
            print(f"  ⚠ No lyrics extracted")
            return None

        # Clean text for MFA
        lyrics_clean = lyrics_text.strip()

        # Run MFA alignment
        phoneme_frames, phoneme_boundaries, phoneme_timings = self.align_with_mfa(
            str(audio_path), lyrics_clean, duration
        )

        if not phoneme_timings:
            print(f"  ❌ MFA alignment failed")
            return None

        # Prepare output
        output_data = {
            'text': lyrics_text,
            'phoneme_timings': [(float(s), float(e), p) for s, e, p in phoneme_timings],
            'audio_duration': float(duration),
            'num_phonemes': len(phoneme_timings),
            'processing_metadata': {
                'processor': 'PhonemeLyricProcessor_MFA',
                'mfa_acoustic_model': MFA_ACOUSTIC_MODEL,
                'mfa_dictionary': MFA_DICTIONARY,
                'timestamp': datetime.now().isoformat()
            }
        }

        # Save files
        stem = audio_path.stem
        output_subdir = self.output_dir / stem
        output_subdir.mkdir(exist_ok=True)

        # Save JSON metadata
        json_path = output_subdir / f"{stem}_phonemes_mfa.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Save phoneme tensors
        tensor_path = output_subdir / f"{stem}_phoneme_tensors.pt"
        torch.save({
            'phoneme_ids': torch.from_numpy(phoneme_frames).long(),  # [T_slow]
            'phoneme_boundaries': torch.from_numpy(phoneme_boundaries).float(),  # [T_slow]
            'phoneme_timings': phoneme_timings,  # List of (start, end, phoneme)
        }, tensor_path)

        print(f"  ✅ Saved: {len(phoneme_timings)} phonemes, {len(phoneme_frames)} frames")

        return {
            'phoneme_data': str(json_path),
            'phoneme_tensors': str(tensor_path),
        }


def process_file_wrapper(args):
    """Wrapper for multiprocessing"""
    audio_path, lyrics_text = args
    processor = PhonemeLyricProcessor()

    try:
        print(f"Processing: {audio_path}")
        result = processor.process_vocal_file(audio_path, lyrics_text)
        return (audio_path, result)
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return (audio_path, None)


def main():
    parser = argparse.ArgumentParser(description="Phoneme-level lyric processor with MFA")
    parser.add_argument('--file', type=str, help='Single audio file to process')
    parser.add_argument('--lyrics', type=str, help='Lyrics file (optional)')
    parser.add_argument('--process_all', action='store_true', help='Process all vocal files')
    parser.add_argument('--vocal_list', type=str,
                       default='/home/arlo/Data/categorized_instrument_paths_subcats_lists/voice/all.txt')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (MFA uses CPU)')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR))

    args = parser.parse_args()

    processor = PhonemeLyricProcessor(output_dir=Path(args.output_dir))

    if args.file:
        # Process single file
        lyrics_text = None
        if args.lyrics and Path(args.lyrics).exists():
            with open(args.lyrics, 'r') as f:
                lyrics_text = f.read()

        print(f"Processing: {args.file}")
        result = processor.process_vocal_file(args.file, lyrics_text)
        if result:
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
        print(f"⚠ MFA is CPU-intensive. Parallel processing may be slow.")

        # Process sequentially or with limited parallelism
        successful = 0
        for audio_path in tqdm(vocal_files, desc="Processing vocals"):
            result = processor.process_vocal_file(audio_path, None)
            if result:
                successful += 1

        print(f"\n✅ Processed {successful}/{len(vocal_files)} files successfully")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
