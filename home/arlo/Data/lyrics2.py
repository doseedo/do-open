#!/usr/bin/env python3
"""
Phoneme-Level Lyric Processor (lyrics2.py)

Extracts phoneme-level transcription with frame alignment for training.

Features:
- G2P (Grapheme-to-Phoneme) conversion using g2p_en
- Montreal Forced Aligner integration for phoneme timing
- Frame-aligned phoneme targets for supervision
- Backward compatible with lyrics.py output format

Usage:
    python lyrics2.py --process_all
    python lyrics2.py --file "path/to/vocal.wav" --lyrics "path/to/lyrics.txt"
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

# Phoneme processing
try:
    from g2p_en import G2p
    G2P_AVAILABLE = True
    print("✅ G2P (grapheme-to-phoneme) available")
except ImportError:
    G2P_AVAILABLE = False
    print("⚠ G2P not available. Install with: pip install g2p-en")

# Optional: Montreal Forced Aligner for precise timing
try:
    import montreal_forced_aligner as mfa
    MFA_AVAILABLE = True
    print("✅ Montreal Forced Aligner available")
except ImportError:
    MFA_AVAILABLE = False
    print("⚠ MFA not available (optional). Install for precise phoneme timing.")

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


class PhonemeLyricProcessor:
    """Processes lyrics at phoneme level with frame alignment"""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize G2P
        if G2P_AVAILABLE:
            print("🔧 Loading G2P model...")
            self.g2p = G2p()
            print("✅ G2P model loaded")
        else:
            self.g2p = None
            print("⚠ G2P not available - phoneme extraction disabled")

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

    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phoneme sequence using G2P.

        Args:
            text: Input text (e.g., "hello world")

        Returns:
            List of phonemes (e.g., ['HH', 'AH0', 'L', 'OW1', 'W', 'ER1', 'L', 'D'])
        """
        if not self.g2p:
            return []

        try:
            # G2P returns phonemes with stress markers (0,1,2)
            # E.g., "hello" -> ['HH', 'AH0', 'L', 'OW1']
            phonemes = self.g2p(text)

            # Remove stress markers for consistency (optional)
            # If you want to keep stress: comment out this loop
            cleaned = []
            for p in phonemes:
                # Remove digits (stress markers)
                p_clean = ''.join(c for c in p if not c.isdigit())
                if p_clean and p_clean not in [' ', '']:
                    cleaned.append(p_clean)

            return cleaned
        except Exception as e:
            print(f"⚠ G2P conversion failed for '{text}': {e}")
            return []

    def align_phonemes_with_mfa(
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
        if not MFA_AVAILABLE:
            print("  ⚠ MFA not available, falling back to syllable-level only")
            return None, None, []

        T_slow = int(np.ceil(audio_duration * SLOW_HZ))
        phoneme_frames = np.full(T_slow, PHONEME_TO_IDX['<PAD>'], dtype=np.int64)
        phoneme_boundaries = np.zeros(T_slow, dtype=np.float32)

        try:
            # TODO: MFA integration
            # This requires:
            # 1. Write text to temporary file
            # 2. Run MFA alignment
            # 3. Parse TextGrid output
            # 4. Convert to frame-level arrays

            # For now, return None to indicate MFA is not implemented
            print("  ⚠ MFA integration not yet implemented")
            return None, None, []

        except Exception as e:
            print(f"  ⚠ MFA alignment failed: {e}")
            return None, None, []

    def align_phonemes_to_frames_fallback(
        self,
        word_timings: List[Tuple[float, float, str]],
        audio_duration: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        FALLBACK ONLY: Align at syllable/word level (NOT phoneme level).

        This method does NOT provide accurate phoneme timing.
        Use MFA (align_phonemes_with_mfa) for real phoneme-level supervision.

        Args:
            word_timings: [(start_time, end_time, word), ...]
            audio_duration: Total duration in seconds

        Returns:
            phoneme_frames: [T_slow] array - actually word/syllable level
            phoneme_boundaries: [T_slow] binary array at word onsets
        """
        T_slow = int(np.ceil(audio_duration * SLOW_HZ))
        phoneme_frames = np.full(T_slow, PHONEME_TO_IDX['<PAD>'], dtype=np.int64)
        phoneme_boundaries = np.zeros(T_slow, dtype=np.float32)

        print("  ⚠ Using FALLBACK: Word-level alignment (not true phoneme-level)")
        print("  ⚠ For accurate phoneme supervision, install Montreal Forced Aligner")

        if not word_timings:
            return phoneme_frames, phoneme_boundaries

        # Word-level alignment only (not phoneme-accurate)
        for start_time, end_time, word in word_timings:
            # Get phonemes for this word (but we won't time them accurately)
            word_phonemes = self.text_to_phonemes(word)

            if not word_phonemes:
                continue

            # Calculate frame range
            start_frame = int(start_time * SLOW_HZ)
            end_frame = int(end_time * SLOW_HZ)

            # Clamp to valid range
            start_frame = max(0, min(start_frame, T_slow - 1))
            end_frame = max(start_frame + 1, min(end_frame, T_slow))

            # Mark word onset (this is accurate)
            phoneme_boundaries[start_frame] = 1.0

            # Fill with first phoneme of word (approximation only)
            if word_phonemes:
                first_phoneme = word_phonemes[0]
                phoneme_id = PHONEME_TO_IDX.get(first_phoneme, PHONEME_TO_IDX['<UNK>'])
                for frame in range(start_frame, end_frame):
                    phoneme_frames[frame] = phoneme_id

        return phoneme_frames, phoneme_boundaries

    def extract_lyrics_from_audio(self, audio_path: str) -> Tuple[str, List[Tuple[float, float, str]]]:
        """Extract lyrics and word timings using Whisper"""
        if not self._load_whisper_if_needed():
            return "", []

        try:
            # Load audio
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)

            # Transcribe with word-level timestamps
            result = self.whisper_model.transcribe(
                audio_path,
                word_timestamps=True,
                language='en'  # Adjust as needed
            )

            # Extract word timings
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
        lyrics_text: Optional[str] = None,
        word_timings: Optional[List[Tuple[float, float, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a vocal file and extract phoneme-level data.

        Args:
            audio_path: Path to vocal audio file
            lyrics_text: Optional lyrics text (if None, extract with Whisper)
            word_timings: Optional word timings (if None, extract with Whisper)

        Returns:
            Dictionary with phoneme data
        """
        audio_path = Path(audio_path)

        # Get audio duration
        try:
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
        except Exception as e:
            print(f"⚠ Failed to get audio duration: {e}")
            return None

        # Extract lyrics if not provided
        if lyrics_text is None or word_timings is None:
            print(f"  Extracting lyrics from audio...")
            lyrics_text, word_timings = self.extract_lyrics_from_audio(str(audio_path))

        if not lyrics_text:
            print(f"  ⚠ No lyrics extracted")
            return None

        # Convert to phonemes
        phonemes = self.text_to_phonemes(lyrics_text)

        if not phonemes:
            print(f"  ⚠ No phonemes generated")
            return None

        # Try MFA alignment first (accurate), fall back to word-level
        phoneme_frames, phoneme_boundaries, phoneme_timings = self.align_phonemes_with_mfa(
            str(audio_path), lyrics_text, duration
        )

        # If MFA failed, use word-level fallback (NOT accurate for phonemes)
        if phoneme_frames is None:
            phoneme_frames, phoneme_boundaries = self.align_phonemes_to_frames_fallback(
                word_timings, duration
            )

        # Prepare output
        output_data = {
            'text': lyrics_text,
            'phonemes': phonemes,
            'phoneme_sequence': [ARPABET_PHONEMES[idx] if idx < len(ARPABET_PHONEMES) else '<PAD>'
                                for idx in phoneme_frames],
            'word_timings': word_timings,
            'audio_duration': duration,
            'processing_metadata': {
                'processor': 'PhonemeLyricProcessor',
                'g2p_available': G2P_AVAILABLE,
                'mfa_available': MFA_AVAILABLE,
                'timestamp': datetime.now().isoformat()
            }
        }

        # Save tensors
        stem = audio_path.stem
        output_subdir = self.output_dir / stem
        output_subdir.mkdir(exist_ok=True)

        # Save JSON metadata
        json_path = output_subdir / f"{stem}_phonemes.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Save phoneme tensors
        tensor_path = output_subdir / f"{stem}_phoneme_tensors.pt"
        torch.save({
            'phoneme_ids': torch.from_numpy(phoneme_frames).long(),  # [T_slow]
            'phoneme_boundaries': torch.from_numpy(phoneme_boundaries).float(),  # [T_slow]
            'phoneme_list': phonemes,
        }, tensor_path)

        print(f"  ✅ Saved phoneme data: {len(phonemes)} phonemes, {len(phoneme_frames)} frames")

        return {
            'phoneme_data': json_path,
            'phoneme_tensors': tensor_path,
        }


def process_file_wrapper(args):
    """Wrapper for multiprocessing"""
    audio_path, lyrics_path = args
    processor = PhonemeLyricProcessor()

    try:
        # Load lyrics if provided
        lyrics_text = None
        if lyrics_path and Path(lyrics_path).exists():
            with open(lyrics_path, 'r') as f:
                lyrics_text = f.read()

        result = processor.process_vocal_file(audio_path, lyrics_text)
        return (audio_path, result)
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return (audio_path, None)


def main():
    parser = argparse.ArgumentParser(description="Phoneme-level lyric processor")
    parser.add_argument('--file', type=str, help='Single audio file to process')
    parser.add_argument('--lyrics', type=str, help='Lyrics file (optional, will use Whisper if not provided)')
    parser.add_argument('--process_all', action='store_true', help='Process all vocal files')
    parser.add_argument('--vocal_list', type=str, default='/home/arlo/Data/categorized_instrument_paths_subcats_lists/voice/all.txt')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR))

    args = parser.parse_args()

    processor = PhonemeLyricProcessor(output_dir=Path(args.output_dir))

    if args.file:
        # Process single file
        print(f"Processing: {args.file}")
        result = processor.process_vocal_file(args.file, args.lyrics)
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

        # Prepare arguments for multiprocessing
        tasks = [(f, None) for f in vocal_files]

        # Process with multiprocessing
        with multiprocessing.Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_file_wrapper, tasks),
                total=len(tasks),
                desc="Processing vocals"
            ))

        # Summary
        successful = sum(1 for _, r in results if r is not None)
        print(f"\n✅ Processed {successful}/{len(vocal_files)} files successfully")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
