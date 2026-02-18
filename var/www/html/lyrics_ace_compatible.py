#!/usr/bin/env python3
"""
Vocal Data Preprocessing Script for ACE-Step Style Training
Modified to match ACE-Step's exact text processing approach

This script extracts:
1. Text lyrics with ACE-Step BPE tokenization
2. Word timelines using forced alignment
3. Audio conditioning features (spectrograms, MFCC, etc.)
4. Vocal-specific features (pitch, formants, vocal onset detection)

Usage:
    python lyrics_ace_compatible.py --process_all
    python lyrics_ace_compatible.py --file "path/to/vocal.wav" --lyrics "path/to/lyrics.txt"
    python lyrics_ace_compatible.py --batch_size 8 --workers 4
"""

import os
import json
import argparse
import multiprocessing
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchaudio
import librosa
import pandas as pd
from tqdm import tqdm
import tempfile
import shutil
from datetime import datetime

# Text processing
import re
from dataclasses import dataclass

# ACE-Step lyric processing
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. Install with: pip install openai-whisper")

try:
    import sys
    sys.path.append('/home/arlo/Data/ACE-Step')
    from acestep.models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
    ACE_TOKENIZER_AVAILABLE = True
except ImportError:
    ACE_TOKENIZER_AVAILABLE = False
    print("Warning: ACE-Step tokenizer not available. Check ACE-Step path.")

# Configuration
VOCAL_LIST_FILE = "/home/arlo/Data/categorized_instrument_paths_subcats_lists/voice/all.txt"
OUTPUT_DIR = Path("/home/arlo/Data/vocal_processing")
SAMPLE_RATE = 44100
HOP_LENGTH = 512
FRAME_LENGTH = 2048

@dataclass
class VocalData:
    """Container for all vocal-related data"""
    audio_path: str
    lyrics_text: Optional[str] = None
    lyrics_tokens: Optional[List[int]] = None
    word_times: Optional[List[Tuple[float, float, str]]] = None
    token_times: Optional[List[Tuple[float, float]]] = None
    language: str = "en"
    pitch: Optional[np.ndarray] = None
    formants: Optional[np.ndarray] = None
    spectral_features: Optional[Dict] = None
    onset_frames: Optional[np.ndarray] = None

class VocalProcessor:
    """Main class for processing vocal audio files with ACE-Step compatibility"""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.whisper_model = None
        self.ace_tokenizer = None
        self._init_models()

    def _init_models(self):
        """Initialize ASR models and ACE-Step tokenizer"""
        if WHISPER_AVAILABLE:
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")

        if ACE_TOKENIZER_AVAILABLE:
            print("Loading ACE-Step tokenizer...")
            self.ace_tokenizer = VoiceBpeTokenizer()

    def extract_lyrics_from_audio(self, audio_path: str) -> Tuple[str, List[Tuple[float, float, str]]]:
        """Extract lyrics and word timings from audio using ASR"""
        if not WHISPER_AVAILABLE:
            print("Whisper not available for lyrics extraction")
            return "", []

        try:
            result = self.whisper_model.transcribe(
                audio_path,
                word_timestamps=True,
                verbose=False
            )

            text = result["text"]
            word_times = []

            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    word_times.append((
                        word_info["start"],
                        word_info["end"],
                        word_info["word"].strip()
                    ))

            return text, word_times

        except Exception as e:
            print(f"Error extracting lyrics from {audio_path}: {e}")
            return "", []

    def clean_text_ace_style(self, text: str, language: str = "en") -> str:
        """Clean text using ACE-Step's approach"""
        if not ACE_TOKENIZER_AVAILABLE:
            # Basic cleaning if tokenizer not available
            text = text.lower().strip()
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text)
            return text

        try:
            # Use ACE-Step's preprocessing
            cleaned = self.ace_tokenizer.preprocess_text(text, language)
            return cleaned
        except Exception as e:
            print(f"Error cleaning text: {e}")
            # Fallback basic cleaning
            text = text.lower().strip()
            text = re.sub(r'\s+', ' ', text)
            return text

    def tokenize_lyrics_ace_style(self, text: str, language: str = "en") -> List[int]:
        """Tokenize lyrics using ACE-Step's BPE tokenizer"""
        if not ACE_TOKENIZER_AVAILABLE:
            print("ACE-Step tokenizer not available")
            return []

        try:
            # Use ACE-Step's encoding
            token_ids = self.ace_tokenizer.encode(text, language)
            return token_ids
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return []

    def align_tokens_to_audio(self, tokens: List[int], word_times: List[Tuple[float, float, str]],
                             original_text: str) -> List[Tuple[float, float]]:
        """Align BPE tokens to audio timings"""
        if not ACE_TOKENIZER_AVAILABLE or not tokens or not word_times:
            return []

        try:
            # Decode tokens back to text to understand token boundaries
            decoded_text = self.ace_tokenizer.decode(tokens)

            # Simple approach: distribute tokens across word timings
            # This is a basic implementation - more sophisticated alignment could be added
            total_duration = word_times[-1][1] - word_times[0][0] if word_times else 0
            if total_duration <= 0:
                return []

            token_duration = total_duration / len(tokens)
            start_time = word_times[0][0] if word_times else 0

            token_times = []
            for i in range(len(tokens)):
                token_start = start_time + i * token_duration
                token_end = start_time + (i + 1) * token_duration
                token_times.append((token_start, token_end))

            return token_times

        except Exception as e:
            print(f"Error aligning tokens: {e}")
            return []

    def extract_vocal_features(self, audio_path: str) -> Dict:
        """Extract vocal-specific features from audio"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

            features = {}

            # Fundamental frequency (pitch)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                hop_length=HOP_LENGTH
            )
            features['f0'] = f0
            features['voiced_flag'] = voiced_flag
            features['voiced_probs'] = voiced_probs

            # Spectral features
            features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]

            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
            for i in range(13):
                features[f'mfcc_{i}'] = mfccs[i]

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
            for i in range(12):
                features[f'chroma_{i}'] = chroma[i]

            # RMS energy
            features['rms'] = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]

            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH)
            features['onset_frames'] = onset_frames

            # Convert to time
            features['time_frames'] = librosa.frames_to_time(np.arange(len(features['f0'])), sr=sr, hop_length=HOP_LENGTH)

            return features

        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return {}

    def extract_formants(self, audio_path: str) -> np.ndarray:
        """Extract formant frequencies (basic implementation)"""
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

            # Use LPC to estimate formants
            # This is a simplified approach - proper formant extraction would use more sophisticated methods
            stft = librosa.stft(y, hop_length=HOP_LENGTH, n_fft=FRAME_LENGTH)
            magnitude = np.abs(stft)

            # Find spectral peaks (rough formant estimation)
            formant_freqs = []
            for frame in range(magnitude.shape[1]):
                spectrum = magnitude[:, frame]
                peaks = librosa.util.peak_pick(spectrum, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)

                # Convert bin indices to frequencies
                peak_freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LENGTH)[peaks]

                # Keep formants in typical vocal range (200-4000 Hz)
                vocal_formants = peak_freqs[(peak_freqs >= 200) & (peak_freqs <= 4000)]

                # Take up to 5 formants
                vocal_formants = np.sort(vocal_formants)[:5]

                # Pad with zeros if fewer than 5 formants
                if len(vocal_formants) < 5:
                    vocal_formants = np.pad(vocal_formants, (0, 5 - len(vocal_formants)), 'constant')

                formant_freqs.append(vocal_formants)

            return np.array(formant_freqs)

        except Exception as e:
            print(f"Error extracting formants from {audio_path}: {e}")
            return np.array([])

    def process_vocal_file(self, audio_path: str, lyrics_path: Optional[str] = None, language: str = "en") -> VocalData:
        """Process a single vocal file to extract all features"""
        print(f"Processing: {Path(audio_path).name}")

        vocal_data = VocalData(audio_path=audio_path, language=language)

        # Extract lyrics from audio or load from file
        if lyrics_path and Path(lyrics_path).exists():
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                raw_text = f.read().strip()
            vocal_data.word_times = []  # Would need separate alignment
        else:
            raw_text, vocal_data.word_times = self.extract_lyrics_from_audio(audio_path)

        # Clean text using ACE-Step approach
        if raw_text:
            vocal_data.lyrics_text = self.clean_text_ace_style(raw_text, language)

            # Tokenize using ACE-Step BPE tokenizer
            vocal_data.lyrics_tokens = self.tokenize_lyrics_ace_style(vocal_data.lyrics_text, language)

            # Align tokens to audio
            vocal_data.token_times = self.align_tokens_to_audio(
                vocal_data.lyrics_tokens, vocal_data.word_times, raw_text
            )

        # Extract audio features
        vocal_data.spectral_features = self.extract_vocal_features(audio_path)
        vocal_data.formants = self.extract_formants(audio_path)

        return vocal_data

    def save_vocal_data(self, vocal_data: VocalData, output_path: Path):
        """Save processed vocal data in ACE-Step compatible format"""
        output_path.mkdir(parents=True, exist_ok=True)

        stem = Path(vocal_data.audio_path).stem

        # Save text data in ACE-Step format
        text_data = {
            'audio_path': vocal_data.audio_path,
            'lyrics_text': vocal_data.lyrics_text,
            'lyrics_tokens': vocal_data.lyrics_tokens,
            'token_times': vocal_data.token_times,
            'word_times': vocal_data.word_times,
            'language': vocal_data.language,
            'processing_date': datetime.now().isoformat(),
            'tokenizer_vocab_size': len(self.ace_tokenizer) if self.ace_tokenizer else None
        }

        with open(output_path / f"{stem}_lyrics.json", 'w', encoding='utf-8') as f:
            json.dump(text_data, f, indent=2, ensure_ascii=False)

        # Save features
        if vocal_data.spectral_features:
            # Save each feature array
            for feature_name, feature_data in vocal_data.spectral_features.items():
                if isinstance(feature_data, np.ndarray):
                    np.save(output_path / f"{stem}_{feature_name}.npy", feature_data)

        # Save formants
        if vocal_data.formants is not None and len(vocal_data.formants) > 0:
            np.save(output_path / f"{stem}_formants.npy", vocal_data.formants)

        # Create ACE-Step style timeline file for tokens
        timeline_data = []
        if vocal_data.token_times and vocal_data.lyrics_tokens:
            for i, ((start, end), token_id) in enumerate(zip(vocal_data.token_times, vocal_data.lyrics_tokens)):
                # Decode token for readability
                token_text = ""
                if self.ace_tokenizer:
                    try:
                        token_text = self.ace_tokenizer.decode([token_id])
                    except:
                        token_text = f"<token_{token_id}>"

                timeline_data.append({
                    'start': start,
                    'end': end,
                    'duration': end - start,
                    'token_id': int(token_id),
                    'token_text': token_text,
                    'index': i
                })

        with open(output_path / f"{stem}_timeline.json", 'w', encoding='utf-8') as f:
            json.dump(timeline_data, f, indent=2)

        print(f"Saved: {stem}")

    def create_training_manifest(self, processed_files: List[Path]) -> Path:
        """Create ACE-Step style training manifest"""
        manifest_data = []

        for file_dir in processed_files:
            stem = file_dir.name

            # Check for required files
            lyrics_file = file_dir / f"{stem}_lyrics.json"
            timeline_file = file_dir / f"{stem}_timeline.json"

            if not (lyrics_file.exists() and timeline_file.exists()):
                continue

            # Load data
            with open(lyrics_file, 'r', encoding='utf-8') as f:
                lyrics_data = json.load(f)

            with open(timeline_file, 'r', encoding='utf-8') as f:
                timeline_data = json.load(f)

            # Create manifest entry
            manifest_entry = {
                'audio_path': lyrics_data['audio_path'],
                'lyrics_text': lyrics_data['lyrics_text'],
                'lyrics_tokens': lyrics_data['lyrics_tokens'],
                'timeline_path': str(timeline_file),
                'features_dir': str(file_dir),
                'duration': timeline_data[-1]['end'] if timeline_data else 0,
                'num_tokens': len(timeline_data),
                'language': lyrics_data.get('language', 'en'),
                'instrument': 'voice',
                'group': 'vocal',
                'subgroup': 'lead_vocal',
                'tokenizer_vocab_size': lyrics_data.get('tokenizer_vocab_size')
            }

            manifest_data.append(manifest_entry)

        # Save manifest
        manifest_path = self.output_dir / "vocal_training_manifest_ace.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        return manifest_path

def process_single_file(args_tuple):
    """Process a single file (for multiprocessing)"""
    audio_path, output_dir, lyrics_path, language = args_tuple

    processor = VocalProcessor(output_dir)

    try:
        vocal_data = processor.process_vocal_file(audio_path, lyrics_path, language)

        # Save data
        stem = Path(audio_path).stem
        file_output_dir = output_dir / stem
        processor.save_vocal_data(vocal_data, file_output_dir)

        return file_output_dir

    except Exception as e:
        print(f"Failed to process {audio_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Process vocal files for ACE-Step training (BPE tokenized)")
    parser.add_argument("--file", type=str, help="Process single audio file")
    parser.add_argument("--lyrics", type=str, help="Lyrics file for single audio file")
    parser.add_argument("--language", type=str, default="en", help="Language code (en, es, fr, etc.)")
    parser.add_argument("--process_all", action="store_true", help="Process all files in vocal list")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of files to process in parallel")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of files to process")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.workers is None:
        args.workers = min(args.batch_size, multiprocessing.cpu_count() - 1)

    print(f"Vocal Processing Script (ACE-Step Compatible)")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Language: {args.language}")
    print(f"Models available: Whisper={WHISPER_AVAILABLE}, ACE_Tokenizer={ACE_TOKENIZER_AVAILABLE}")

    if args.file:
        # Process single file
        processor = VocalProcessor(output_dir)
        vocal_data = processor.process_vocal_file(args.file, args.lyrics, args.language)

        stem = Path(args.file).stem
        file_output_dir = output_dir / stem
        processor.save_vocal_data(vocal_data, file_output_dir)

        print(f"Processed: {args.file}")

    elif args.process_all:
        # Process all files from list
        if not Path(VOCAL_LIST_FILE).exists():
            print(f"Vocal list file not found: {VOCAL_LIST_FILE}")
            return

        # Read file list
        with open(VOCAL_LIST_FILE, 'r') as f:
            audio_paths = [line.strip() for line in f if line.strip()]

        if args.max_files:
            audio_paths = audio_paths[:args.max_files]

        print(f"Found {len(audio_paths)} vocal files to process")

        # Prepare arguments for multiprocessing
        process_args = [(path, output_dir, None, args.language) for path in audio_paths]

        # Process files in parallel
        processed_files = []

        if args.workers > 1:
            with multiprocessing.Pool(args.workers) as pool:
                results = list(tqdm(
                    pool.imap(process_single_file, process_args),
                    total=len(process_args),
                    desc="Processing vocals"
                ))
                processed_files = [r for r in results if r is not None]
        else:
            # Sequential processing
            for args_tuple in tqdm(process_args, desc="Processing vocals"):
                result = process_single_file(args_tuple)
                if result:
                    processed_files.append(result)

        print(f"Successfully processed {len(processed_files)}/{len(audio_paths)} files")

        # Create training manifest
        if processed_files:
            processor = VocalProcessor(output_dir)
            manifest_path = processor.create_training_manifest(processed_files)
            print(f"Training manifest saved: {manifest_path}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()