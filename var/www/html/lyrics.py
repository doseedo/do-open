#!/usr/bin/env python3

# Activate ace_step conda environment for ACE-Step compatibility
import subprocess
import sys
import os

# Set up conda environment
os.environ['CONDA_DEFAULT_ENV'] = 'ace_step'
if 'CONDA_PREFIX' not in os.environ:
    # Try to activate ace_step environment
    try:
        conda_path = '/home/arlo/miniconda3/etc/profile.d/conda.sh'
        if os.path.exists(conda_path):
            os.environ['PATH'] = '/home/arlo/miniconda3/envs/ace_step/bin:' + os.environ.get('PATH', '')
            os.environ['CONDA_PREFIX'] = '/home/arlo/miniconda3/envs/ace_step'
            os.environ['CONDA_DEFAULT_ENV'] = 'ace_step'
    except Exception:
        pass  # Continue with base environment if conda setup fails
"""
Lyric Processing Script Compatible with ACE-Step

This script processes lyrics identically to ACE-Step approach:
1. Language detection and segmentation
2. Text normalization (punctuation removal, case normalization)
3. BPE tokenization with language prefixes
4. Structure marker handling ([Verse], [Chorus], etc.)
5. Output format matching ACE-Step expectations

Usage:
    python lyrics.py --process_all
    python lyrics.py --file "path/to/vocal.wav" --lyrics "path/to/lyrics.txt"
    python lyrics.py --batch_size 8 --workers 4
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
import librosa
from tqdm import tqdm
from datetime import datetime

# Add ACE-Step to Python path for imports
sys.path.append('/home/arlo/Data/ACE-Step')

# ACE-Step imports for lyric processing with patches
try:
    # First try to import patched versions
    sys.path.insert(0, '/home/arlo/Data/ace_step_patches')
    from lyric_normalizer_patched import normalize_text
    from zh_num2words_patched import num_to_chinese, chinese_to_num
    print("✅ Using patched ACE-Step lyric normalizer")

    # Try original ACE-Step components
    from acestep.models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
    from acestep.language_segmentation.LangSegment import LangSegment
    ACE_STEP_AVAILABLE = True
    print("✅ ACE-Step components loaded with patches")
except ImportError as e:
    ACE_STEP_AVAILABLE = False
    print(f"⚠ ACE-Step imports not available: {e}")

# Audio processing for extraction
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠ Whisper not available. Install with: pip install openai-whisper")

# Configuration matching ACE-Step
VOCAL_LIST_FILE = "/home/arlo/Data/categorized_instrument_paths_subcats_lists/voice/all.txt"
OUTPUT_DIR = Path("/mnt/msdd/vocal_processing")
SAMPLE_RATE = 44100

# ACE-Step language mapping
SUPPORT_LANGUAGES = {
    "en": 259, "de": 260, "fr": 262, "es": 284, "it": 285, "pt": 286,
    "pl": 294, "tr": 295, "ru": 267, "cs": 293, "nl": 297, "ar": 5022,
    "zh": 5023, "ja": 5412, "hu": 5753, "ko": 6152, "hi": 6680
}

class ACEStepLyricProcessor:
    """Lyric processor that matches ACE-Step implementation exactly"""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all attributes first
        self.tokenizer = None
        self.lang_segment = None
        self.whisper_model = None
        self._whisper_loaded = False
        self._whisper_loading_attempted = False

        # Initialize ACE-Step components
        self._init_components()

    def _init_components(self):
        """Initialize ACE-Step lyric processing components"""
        if not ACE_STEP_AVAILABLE:
            print("⚠ ACE-Step components not available - text processing will be limited")
            return

        try:
            # Initialize language segmentation
            print("🔧 Loading language segmentation model...")
            self.lang_segment = LangSegment()

            # Initialize BPE tokenizer
            print("🔧 Loading ACE-Step lyric tokenizer...")
            # Look for tokenizer in ACE-Step checkpoints or models directory
            tokenizer_path = "/home/arlo/Data/ACE-Step/checkpoints"  # Adjust path as needed
            self.tokenizer = VoiceBpeTokenizer()

        except Exception as e:
            print(f"⚠ Error initializing ACE-Step components: {e}")
            print("   Using fallback text processing")

        # Whisper initialization status
        if not WHISPER_AVAILABLE:
            print("⚠ Whisper not available - audio extraction will be disabled")

    def _load_whisper_if_needed(self):
        """Load Whisper model lazily when first needed"""
        if not WHISPER_AVAILABLE:
            return False

        if self._whisper_loaded and self.whisper_model is not None:
            return True

        if self._whisper_loading_attempted:
            return self.whisper_model is not None

        self._whisper_loading_attempted = True

        try:
            print("🔧 Loading Whisper model (lazy loading)...")

            # Set CUDA device for this worker
            gpu_id = int(os.environ.get('WORKER_GPU_ID', '0'))
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                torch.cuda.set_device(gpu_id)
                device = f"cuda:{gpu_id}"
                print(f"🔧 Setting CUDA device {gpu_id} for Whisper")
            else:
                device = "cpu"
                print(f"🔧 Using CPU for Whisper")

            # Load Whisper model with memory cleanup
            import whisper
            import gc

            # Clear any existing CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.whisper_model = whisper.load_model("base", device=device)
            self._whisper_loaded = True
            print(f"✅ Whisper loaded successfully on {device}")

            # Force garbage collection after loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True

        except Exception as e:
            print(f"❌ Whisper loading failed: {e}")
            # Try CPU fallback
            try:
                print("🔧 Trying CPU fallback for Whisper...")
                self.whisper_model = whisper.load_model("base", device="cpu")
                self._whisper_loaded = True
                print("✅ Whisper loaded on CPU (fallback)")
                return True
            except Exception as e2:
                print(f"❌ CPU fallback also failed: {e2}")
                self.whisper_model = None
                return False

    def extract_lyrics_from_audio(self, audio_path: str) -> Tuple[str, List[Tuple[float, float, str]]]:
        """Extract lyrics and word timings from audio using Whisper"""
        # Try to load Whisper if not already loaded
        if not self._load_whisper_if_needed():
            print("⚠ Whisper not available for lyrics extraction")
            return "", []

        try:
            # Check if audio file exists and is readable
            if not os.path.exists(audio_path):
                print(f"⚠ Audio file not found: {audio_path}")
                return "", []

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
            print(f"❌ Error extracting lyrics from {audio_path}: {e}")
            # Clean up GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "", []

    def detect_and_segment_language(self, text: str) -> List[Dict[str, Any]]:
        """Detect and segment text by language using ACE-Step's LangSegment"""
        if not self.lang_segment:
            # Enhanced fallback: basic language detection
            return self._fallback_language_detection(text)

        try:
            # Use LangSegment to classify languages (correct method name)
            result = self.lang_segment.classify(text)

            # Convert result to our expected format
            if isinstance(result, str):
                # Single language detected
                segments = [{"text": text, "language": result, "confidence": 1.0}]
            elif isinstance(result, dict):
                # Multiple languages or detailed result
                segments = [{"text": text, "language": result.get("language", "en"), "confidence": result.get("confidence", 1.0)}]
            else:
                # Fallback
                segments = [{"text": text, "language": "en", "confidence": 1.0}]

            # Format segments for processing
            formatted_segments = []
            for segment in segments:
                lang_code = segment.get("language", "en")
                # Map to ACE-Step supported languages
                if lang_code not in SUPPORT_LANGUAGES:
                    lang_code = "en"  # Default to English

                formatted_segments.append({
                    "text": segment.get("text", ""),
                    "language": lang_code,
                    "confidence": segment.get("confidence", 1.0)
                })

            return formatted_segments

        except Exception as e:
            print(f"⚠ Language detection error: {e}")
            return [{"text": text, "language": "en", "confidence": 1.0}]

    def normalize_text_ace_step(self, text: str, language: str = "en") -> str:
        """Normalize text using ACE-Step's normalize_text function"""
        if not ACE_STEP_AVAILABLE:
            # Fallback normalization
            return self._fallback_normalize(text)

        try:
            normalized = normalize_text(text, language, strip=True)
            return normalized
        except Exception as e:
            print(f"⚠ ACE-Step normalization error: {e}")
            return self._fallback_normalize(text)

    def _fallback_normalize(self, text: str) -> str:
        """Fallback text normalization when ACE-Step is not available"""
        import re

        # Basic normalization similar to ACE-Step
        text = text.lower()

        # Remove or replace punctuation (similar to ACE-Step)
        text = re.sub(r'[-]', ' ', text)  # Hyphens to spaces
        text = re.sub(r'[,.!?;:"()[\]{}]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = text.strip()

        return text

    def _fallback_language_detection(self, text: str) -> List[Dict[str, Any]]:
        """Basic language detection when ACE-Step is unavailable"""
        import re

        # Simple heuristic-based language detection
        if re.search(r'[\u4e00-\u9fff]', text):  # Chinese characters
            language = "zh"
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):  # Japanese hiragana/katakana
            language = "ja"
        elif re.search(r'[\uac00-\ud7af]', text):  # Korean
            language = "ko"
        elif re.search(r'[\u0600-\u06ff]', text):  # Arabic
            language = "ar"
        elif re.search(r'[\u0400-\u04ff]', text):  # Cyrillic (Russian)
            language = "ru"
        else:
            language = "en"  # Default to English

        return [{"text": text, "language": language, "confidence": 0.8}]

    def _enhanced_fallback_normalize(self, text: str, language: str = "en") -> str:
        """Enhanced text normalization with basic language support"""
        import re

        # Language-specific processing
        if language == "zh":
            # Basic Chinese processing (limited without proper tools)
            text = re.sub(r'[，。！？；：]', '', text)  # Remove Chinese punctuation
        elif language == "ja":
            # Basic Japanese processing
            text = re.sub(r'[、。！？]', '', text)  # Remove Japanese punctuation
        elif language == "ko":
            # Basic Korean processing
            text = re.sub(r'[，。！？]', '', text)  # Remove Korean punctuation
        elif language == "ar":
            # Basic Arabic processing
            text = re.sub(r'[،؟！]', '', text)  # Remove Arabic punctuation
        elif language == "ru":
            # Basic Russian processing
            text = re.sub(r'[，。！？]', '', text)  # Remove Russian punctuation

        # Apply general normalization
        text = self._fallback_normalize(text)

        return text

    def tokenize_lyrics(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Tokenize lyrics using ACE-Step's VoiceBpeTokenizer"""
        if not self.tokenizer:
            # Fallback: simple word tokenization
            words = text.split()
            return {
                "lyric_token_id": list(range(len(words))),  # Dummy token IDs
                "lyric_mask": [1] * len(words),
                "tokens": words,
                "language": language
            }

        try:
            # Use ACE-Step tokenizer (correct method name, with lang param)
            tokenized = self.tokenizer.encode(text, lang=language)

            # Handle different return formats from ACE-Step tokenizer
            if isinstance(tokenized, list):
                # If it returns a list of token IDs directly
                return {
                    "lyric_token_id": tokenized,
                    "lyric_mask": [1] * len(tokenized),
                    "tokens": [str(t) for t in tokenized],
                    "language": language
                }
            elif isinstance(tokenized, dict):
                # If it returns a dictionary
                return {
                    "lyric_token_id": tokenized.get("token_ids", tokenized.get("input_ids", [])),
                    "lyric_mask": tokenized.get("attention_mask", [1] * len(tokenized.get("token_ids", tokenized.get("input_ids", [])))),
                    "tokens": tokenized.get("tokens", []),
                    "language": language
                }
            else:
                # Fallback for unknown format
                print(f"⚠ Unknown tokenizer return format: {type(tokenized)}")
                return {
                    "lyric_token_id": [],
                    "lyric_mask": [],
                    "tokens": [],
                    "language": language
                }

        except Exception as e:
            print(f"⚠ Tokenization error: {e}")
            # Fallback
            words = text.split()
            return {
                "lyric_token_id": list(range(len(words))),
                "lyric_mask": [1] * len(words),
                "tokens": words,
                "language": language
            }

    def process_structure_markers(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Process song structure markers like [Verse], [Chorus], etc."""
        import re

        # Find structure markers
        structure_pattern = r'\[([^\]]+)\]'
        markers = []

        for match in re.finditer(structure_pattern, text):
            markers.append({
                "type": "structure",
                "content": match.group(1),
                "position": match.start(),
                "end_position": match.end()
            })

        # Remove markers from text but keep the information
        clean_text = re.sub(structure_pattern, '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text, markers

    def process_lyrics_text(self, text: str) -> Dict[str, Any]:
        """Process lyrics text through the complete ACE-Step pipeline"""

        # Step 1: Handle structure markers
        clean_text, structure_markers = self.process_structure_markers(text)

        # Step 2: Language detection and segmentation
        language_segments = self.detect_and_segment_language(clean_text)

        processed_segments = []
        all_tokens = []
        all_token_ids = []
        all_masks = []

        for segment in language_segments:
            segment_text = segment["text"]
            language = segment["language"]

            # Step 3: Text normalization
            if ACE_STEP_AVAILABLE:
                normalized_text = self.normalize_text_ace_step(segment_text, language)
            else:
                normalized_text = self._enhanced_fallback_normalize(segment_text, language)

            # Step 4: Tokenization
            tokenized = self.tokenize_lyrics(normalized_text, language)

            processed_segments.append({
                "original_text": segment_text,
                "normalized_text": normalized_text,
                "language": language,
                "confidence": segment["confidence"],
                "tokens": tokenized["tokens"],
                "token_ids": tokenized["lyric_token_id"],
                "mask": tokenized["lyric_mask"]
            })

            # Accumulate tokens
            all_tokens.extend(tokenized["tokens"])
            all_token_ids.extend(tokenized["lyric_token_id"])
            all_masks.extend(tokenized["lyric_mask"])

        return {
            "original_text": text,
            "clean_text": clean_text,
            "structure_markers": structure_markers,
            "language_segments": processed_segments,
            "lyric_token_id": all_token_ids,
            "lyric_mask": all_masks,
            "tokens": all_tokens,
            "primary_language": language_segments[0]["language"] if language_segments else "en"
        }

    def create_ace_step_format(self, lyric_data: Dict[str, Any],
                              word_times: List[Tuple[float, float, str]] = None) -> Dict[str, Any]:
        """Create output in ACE-Step expected format"""

        # Create candidate lyric chunks (ACE-Step format)
        candidate_chunks = []

        for segment in lyric_data["language_segments"]:
            candidate_chunks.append({
                "lyric": segment["normalized_text"],
                "language": segment["language"],
                "tokens": segment["tokens"],
                "confidence": segment["confidence"]
            })

        # Main output format matching ACE-Step and trainer compatibility
        ace_step_format = {
            "lyric_token_idx": torch.tensor(lyric_data["lyric_token_id"], dtype=torch.long),  # Renamed for trainer compatibility
            "lyric_mask": torch.tensor(lyric_data["lyric_mask"], dtype=torch.long),
            "candidate_lyric_chunk": candidate_chunks,
            "original_text": lyric_data["original_text"],
            "primary_language": lyric_data["primary_language"],
            "structure_markers": lyric_data["structure_markers"],
            "processing_metadata": {
                "processor": "ACE-Step-compatible",
                "timestamp": datetime.now().isoformat(),
                "num_segments": len(lyric_data["language_segments"]),
                "total_tokens": len(lyric_data["lyric_token_id"])
            }
        }

        # Add timing information if available
        if word_times:
            ace_step_format["word_timings"] = word_times
            ace_step_format["has_timing"] = True
        else:
            ace_step_format["has_timing"] = False

        return ace_step_format

    def _create_minimal_vocal_entry(self, audio_path: str, lyrics_path: Optional[str] = None) -> Dict[str, Any]:
        """Create minimal entry for vocal files with no clear lyrics (oohs, ahhs, hums, etc.)"""

        # Create basic vocal tokens for non-lyrical vocals
        minimal_tokens = ["[VOCAL]", "[NON_LYRICAL]"]
        minimal_token_ids = [1000, 1001]  # Special IDs for vocal sounds
        minimal_mask = [1, 1]

        # Detect audio duration for timing
        try:
            duration = librosa.get_duration(filename=audio_path)
        except:
            duration = 30.0  # Default duration

        ace_step_format = {
            "lyric_token_idx": torch.tensor(minimal_token_ids, dtype=torch.long),  # Renamed for trainer compatibility
            "lyric_mask": torch.tensor(minimal_mask, dtype=torch.long),
            "candidate_lyric_chunk": [{
                "lyric": "[VOCAL_SOUND]",
                "language": "vocal",
                "tokens": minimal_tokens,
                "confidence": 0.5
            }],
            "original_text": "[NON_LYRICAL_VOCAL]",
            "primary_language": "vocal",
            "structure_markers": [],
            "processing_metadata": {
                "processor": "ACE-Step-compatible",
                "timestamp": datetime.now().isoformat(),
                "num_segments": 1,
                "total_tokens": len(minimal_token_ids),
                "vocal_type": "non_lyrical",
                "duration": duration
            },
            "word_timings": [],
            "has_timing": False,
            "audio_path": audio_path,
            "lyrics_path": lyrics_path
        }

        return ace_step_format

    def detect_vocal_content_type(self, lyrics_text: str) -> str:
        """Detect if lyrics are actual words or just vocal sounds"""
        if not lyrics_text:
            return "silent"

        # Clean and analyze text
        clean_text = lyrics_text.lower().strip()

        # Common vocal sounds that aren't lyrics
        vocal_sounds = [
            'ah', 'oh', 'uh', 'mm', 'hmm', 'la', 'na', 'da', 'hey', 'yeah',
            'ooh', 'aah', 'whoa', 'wow', 'huh', 'shh', 'psst', 'tsk',
            'oooh', 'aaah', 'mmm', 'lalala', 'nanana', 'dadada'
        ]

        # Split into words and check
        words = clean_text.split()
        if len(words) == 0:
            return "silent"

        # If most words are vocal sounds, classify as non-lyrical
        vocal_sound_count = sum(1 for word in words if any(vs in word for vs in vocal_sounds))
        vocal_ratio = vocal_sound_count / len(words)

        if vocal_ratio > 0.7:  # 70% or more are vocal sounds
            return "vocal_sounds"
        elif vocal_ratio > 0.3:  # 30-70% mixed
            return "mixed"
        else:
            return "lyrics"

    def process_vocal_file(self, audio_path: str, lyrics_path: Optional[str] = None) -> Dict[str, Any]:
        """Process a vocal file to extract and process lyrics"""
        print(f"🎵 Processing: {Path(audio_path).name}")

        # Extract or load lyrics
        if lyrics_path and Path(lyrics_path).exists():
            # Load lyrics from file
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                lyrics_text = f.read().strip()
            word_times = []  # No timing info from text file
        else:
            # Extract lyrics from audio
            lyrics_text, word_times = self.extract_lyrics_from_audio(audio_path)

        # Detect vocal content type
        vocal_type = self.detect_vocal_content_type(lyrics_text)

        if vocal_type == "silent" or not lyrics_text:
            print(f"⚠ No lyrics found for {audio_path}")
            # Create minimal entry for vocal files with no clear lyrics
            return self._create_minimal_vocal_entry(audio_path, lyrics_path)
        elif vocal_type == "vocal_sounds":
            print(f"🎵 Detected vocal sounds (oohs/ahhs) in {Path(audio_path).name}")
            # Create enhanced entry for vocal sounds
            vocal_entry = self._create_minimal_vocal_entry(audio_path, lyrics_path)
            vocal_entry["processing_metadata"]["vocal_type"] = "vocal_sounds"
            vocal_entry["original_text"] = lyrics_text
            vocal_entry["candidate_lyric_chunk"][0]["lyric"] = lyrics_text
            return vocal_entry

        # Process lyrics through ACE-Step pipeline
        print(f"🎤 Processing {'mixed content' if vocal_type == 'mixed' else 'lyrics'} for {Path(audio_path).name}")
        lyric_data = self.process_lyrics_text(lyrics_text)

        # Format for ACE-Step compatibility
        ace_step_data = self.create_ace_step_format(lyric_data, word_times)

        # Add file metadata
        ace_step_data["audio_path"] = audio_path
        ace_step_data["lyrics_path"] = lyrics_path

        return ace_step_data

    def cleanup_resources(self):
        """Clean up GPU resources"""
        if hasattr(self, 'whisper_model') and self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
            self._whisper_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc
        gc.collect()

    def save_processed_data(self, processed_data: Dict[str, Any], output_path: Path):
        """Save processed data in ACE-Step compatible format"""
        output_path.mkdir(parents=True, exist_ok=True)

        stem = Path(processed_data["audio_path"]).stem

        # Convert tensors to lists for JSON serialization
        json_data = {}
        for key, value in processed_data.items():
            if torch.is_tensor(value):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value

        # Save main data file with process ID to avoid conflicts
        import os
        pid = os.getpid()
        output_file = output_path / f"{stem}_lyrics_ace_step.json"
        temp_file = output_path / f"{stem}_lyrics_ace_step_{pid}.tmp"

        try:
            # Write to temp file first, then rename (atomic operation)
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            temp_file.rename(output_file)
        except Exception as e:
            print(f"⚠ Error saving {stem}: {e}")
            if temp_file.exists():
                temp_file.unlink()

        # Save tensor data separately for easy loading
        tensor_file = output_path / f"{stem}_tensors.pt"
        tensor_data = {
            "lyric_token_id": processed_data["lyric_token_id"],
            "lyric_mask": processed_data["lyric_mask"]
        }
        torch.save(tensor_data, tensor_file)

        print(f"✅ Saved: {stem}")

    def create_training_manifest(self, processed_files: List[Path]) -> Path:
        """Create training manifest in ACE-Step format"""
        manifest_data = []

        for file_dir in processed_files:
            # Find ACE-Step format files
            ace_step_files = list(file_dir.glob("*_lyrics_ace_step.json"))

            for ace_file in ace_step_files:
                try:
                    with open(ace_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Create manifest entry matching ACE-Step format
                    manifest_entry = {
                        "audio_path": data["audio_path"],
                        "lyrics_data_path": str(ace_file),
                        "tensor_path": str(file_dir / f"{ace_file.stem.replace('_lyrics_ace_step', '_tensors')}.pt"),
                        "primary_language": data["primary_language"],
                        "has_timing": data["has_timing"],
                        "num_tokens": len(data["lyric_token_id"]),
                        "num_segments": data["processing_metadata"]["num_segments"],
                        "instrument": "voice",
                        "group": "vocal",
                        "subgroup": "lead_vocal"
                    }

                    manifest_data.append(manifest_entry)

                except Exception as e:
                    print(f"⚠ Error processing {ace_file}: {e}")

        # Save manifest
        manifest_path = self.output_dir / "ace_step_vocal_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        return manifest_path

# Global processor instance per worker process
_worker_processor = None
_worker_gpu_id = None
_worker_process_id = None

def process_single_file(args_tuple):
    """Process a single file (for multiprocessing)"""
    global _worker_processor, _worker_gpu_id, _worker_process_id
    audio_path, output_dir, lyrics_path, gpu_id = args_tuple

    # Get current process ID to track worker identity
    current_process_id = os.getpid()

    # Initialize processor if this is the first file for this worker process
    if _worker_processor is None or _worker_process_id != current_process_id:
        _worker_process_id = current_process_id
        _worker_gpu_id = gpu_id
        os.environ['WORKER_GPU_ID'] = str(gpu_id)
        _worker_processor = ACEStepLyricProcessor(output_dir)
        print(f"🔧 Worker PID {current_process_id} initialized with GPU {gpu_id}")
    elif _worker_gpu_id != gpu_id:
        # Update GPU assignment for existing processor without recreating
        _worker_gpu_id = gpu_id
        os.environ['WORKER_GPU_ID'] = str(gpu_id)
        print(f"🔧 Worker PID {current_process_id} reassigned to GPU {gpu_id}")

    try:
        processed_data = _worker_processor.process_vocal_file(audio_path, lyrics_path)

        if processed_data is None:
            return None

        # Save data
        stem = Path(audio_path).stem
        file_output_dir = output_dir / stem
        _worker_processor.save_processed_data(processed_data, file_output_dir)

        return file_output_dir

    except Exception as e:
        print(f"❌ Failed to process {audio_path}: {e}")
        return None
    # Don't cleanup resources - keep processor and Whisper model loaded for next file

def create_gpu_distributed_args(audio_paths, output_dir, lyrics_path, num_gpus=4):
    """Create arguments distributed across GPUs"""
    process_args = []

    for i, path in enumerate(audio_paths):
        gpu_id = i % num_gpus  # Distribute across GPUs
        process_args.append((path, output_dir, lyrics_path, gpu_id))

    return process_args

def main():
    parser = argparse.ArgumentParser(description="Process lyrics using ACE-Step compatible pipeline")
    parser.add_argument("--file", type=str, help="Process single audio file")
    parser.add_argument("--lyrics", type=str, help="Lyrics file for single audio file")
    parser.add_argument("--process_all", action="store_true", help="Process all files in vocal list")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of files to process in parallel")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.workers is None:
        args.workers = min(20, multiprocessing.cpu_count())  # Default to 20 workers

    # Validate GPU availability
    if args.num_gpus > 0 and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU-only processing")
        args.num_gpus = 0

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.num_gpus > available_gpus:
        print(f"⚠ Requested {args.num_gpus} GPUs but only {available_gpus} available")
        args.num_gpus = available_gpus

    print(f"🎵 ACE-Step Lyric Processing Script")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚙️ Workers: {args.workers}")
    print(f"🔧 GPUs: {args.num_gpus}")
    print(f"🔧 Components available: ACE-Step={ACE_STEP_AVAILABLE}, Whisper={WHISPER_AVAILABLE}")

    if args.file:
        # Process single file
        processor = ACEStepLyricProcessor(output_dir)
        processed_data = processor.process_vocal_file(args.file, args.lyrics)

        if processed_data:
            stem = Path(args.file).stem
            file_output_dir = output_dir / stem
            processor.save_processed_data(processed_data, file_output_dir)
            print(f"✅ Processed: {args.file}")
        else:
            print(f"❌ Failed to process: {args.file}")

    elif args.process_all:
        # Process all files from list
        if not Path(VOCAL_LIST_FILE).exists():
            print(f"❌ Vocal list file not found: {VOCAL_LIST_FILE}")
            return

        # Read file list
        with open(VOCAL_LIST_FILE, 'r') as f:
            audio_paths = [line.strip() for line in f if line.strip()]

        if args.max_files:
            audio_paths = audio_paths[:args.max_files]

        print(f"🎵 Found {len(audio_paths)} vocal files to process")

        # Prepare arguments for multiprocessing with GPU distribution
        process_args = create_gpu_distributed_args(audio_paths, output_dir, None, args.num_gpus)

        # Process files in parallel
        processed_files = []

        if args.workers > 1:
            # Use spawn method for CUDA compatibility across processes
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(args.workers) as pool:
                # Use imap_unordered for better performance with mixed processing times
                results = list(tqdm(
                    pool.imap_unordered(process_single_file, process_args, chunksize=1),
                    total=len(process_args),
                    desc=f"Processing vocals (GPUs: {args.num_gpus}, Workers: {args.workers})"
                ))
                processed_files = [r for r in results if r is not None]
        else:
            # Sequential processing
            for args_tuple in tqdm(process_args, desc="Processing vocals"):
                result = process_single_file(args_tuple)
                if result:
                    processed_files.append(result)

        print(f"✅ Successfully processed {len(processed_files)}/{len(audio_paths)} files")

        # Create training manifest
        if processed_files:
            processor = ACEStepLyricProcessor(output_dir)
            manifest_path = processor.create_training_manifest(processed_files)
            print(f"📋 Training manifest saved: {manifest_path}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()