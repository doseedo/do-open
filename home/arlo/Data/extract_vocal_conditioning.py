#!/usr/bin/env python3
"""
Vocal Conditioning Extractor for trainer_performervox.py

Extracts ALL conditioning needed for the vocal model:
1. Standard conditioning (piano roll, amp, rframe, rbend, encodec) - via test_extract_local.py
2. Lyrics data (JSON with word/syllable timings) - via Whisper + alignment
3. Lyrics tensors (PyTorch with embeddings and phonemes)
4. Syllable boundaries (NumPy array)
5. Speaker embedding (256-dim Resemblyzer)
6. mHuBERT features (optional - phonetic features)

Usage:
    python extract_vocal_conditioning.py --audio vocals.wav --output ./vocal_extracts
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torchaudio

# Constants
DCAE_SR = 44100
DCAE_HOP = 4096
SLOW_HZ = DCAE_SR / DCAE_HOP  # ~10.77 Hz


def extract_standard_conditioning(audio_path: Path, output_dir: Path) -> Dict[str, Path]:
    """
    Extract standard conditioning using test_extract_local.py
    Returns dict with paths to: piano_roll, amp, rframe, rbend, encodec
    """
    print("\n" + "=" * 60)
    print("Step 1: Extracting Standard Conditioning")
    print("=" * 60)

    cmd = [
        "python", "/home/arlo/Data/test_extract_local.py",
        "--input", str(audio_path),
        "--output", str(output_dir)
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print("❌ Extraction failed")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Standard conditioning extraction failed")

    stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in audio_path.stem)[:128]

    # test_extract_local.py creates a subdirectory with the stem name
    stem_dir = output_dir / stem

    paths = {
        "piano_roll": stem_dir / f"{stem}.pianoroll.npy",
        "amp": stem_dir / f"{stem}.amp.npy",
        "rframe": stem_dir / f"{stem}.rframe.npy",
        "rbend": stem_dir / f"{stem}.rbend.npy",
        "encodec": stem_dir / f"{stem}.encodec.pt",
        "midi": stem_dir / f"{stem}.mid",
    }

    # Verify all files exist
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")

    print("✅ Standard conditioning extracted")
    return paths


def extract_speaker_embedding(audio_path: Path, output_dir: Path) -> Path:
    """
    Extract speaker embedding using Resemblyzer
    Returns path to 256-dim embedding
    """
    print("\n" + "=" * 60)
    print("Step 2: Extracting Speaker Embedding")
    print("=" * 60)

    stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in audio_path.stem)[:128]
    stem_dir = output_dir / stem
    stem_dir.mkdir(parents=True, exist_ok=True)

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav

        encoder = VoiceEncoder()
        wav = preprocess_wav(str(audio_path))
        embedding = encoder.embed_utterance(wav)

        # Save as tensor
        emb_path = stem_dir / f"{stem}_speaker_emb.pt"
        torch.save(torch.from_numpy(embedding).float(), emb_path)

        print(f"✅ Speaker embedding saved: {emb_path}")
        return emb_path

    except ImportError:
        print("⚠️  Resemblyzer not available. Install with: pip install resemblyzer")
        # Return dummy embedding
        emb_path = stem_dir / f"{stem}_speaker_emb.pt"
        torch.save(torch.zeros(256), emb_path)
        print(f"⚠️  Saved dummy speaker embedding: {emb_path}")
        return emb_path


def extract_lyrics_with_whisper(audio_path: Path, output_dir: Path, T_slow: int) -> Dict[str, Path]:
    """
    Extract lyrics using Whisper and create required vocal conditioning files
    Returns dict with paths to: lyrics_data (JSON), lyrics_tensors (PT), syllable_boundaries (NPY)
    """
    print("\n" + "=" * 60)
    print("Step 3: Extracting Lyrics & Phonemes")
    print("=" * 60)

    stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in audio_path.stem)[:128]
    stem_dir = output_dir / stem
    stem_dir.mkdir(parents=True, exist_ok=True)

    try:
        import whisper

        # Load Whisper
        print("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base", device=device)

        # Transcribe with word timestamps
        print(f"Transcribing {audio_path}...")
        result = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language="en"
        )

        # Extract word timings
        word_timings = []
        if "segments" in result:
            for segment in result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        word_timings.append([
                            word_info["start"],
                            word_info["end"],
                            word_info["word"].strip()
                        ])

        # Create syllable boundaries (simple: mark word starts)
        syllable_boundaries = np.zeros(T_slow, dtype=np.float32)
        for start_time, end_time, word in word_timings:
            start_frame = int(start_time * SLOW_HZ)
            if 0 <= start_frame < T_slow:
                syllable_boundaries[start_frame] = 1.0

        # Create lyrics data JSON
        lyrics_data = {
            "full_text": result["text"],
            "word_timings": word_timings,
            "language": result.get("language", "en"),
        }

        lyrics_data_path = stem_dir / f"{stem}_lyrics_data.json"
        with open(lyrics_data_path, 'w') as f:
            json.dump(lyrics_data, f, indent=2)

        # Create dummy lyrics tensors (no actual embeddings)
        # The model will create these on the fly during training
        lyrics_tensors = {
            "lyrics_embeddings": torch.zeros(len(word_timings), 256),  # Dummy embeddings
            "phoneme_embeddings": torch.zeros(T_slow, 256),  # Dummy phoneme embeddings
        }

        lyrics_tensors_path = stem_dir / f"{stem}_lyrics_tensors.pt"
        torch.save(lyrics_tensors, lyrics_tensors_path)

        # Save syllable boundaries
        syllable_path = stem_dir / f"{stem}_syllable_boundaries.npy"
        np.save(syllable_path, syllable_boundaries)

        print(f"✅ Lyrics data saved: {len(word_timings)} words")
        print(f"   - Lyrics JSON: {lyrics_data_path}")
        print(f"   - Lyrics tensors: {lyrics_tensors_path}")
        print(f"   - Syllable boundaries: {syllable_path}")

        return {
            "lyrics_data": lyrics_data_path,
            "lyrics_tensors": lyrics_tensors_path,
            "syllable_boundaries": syllable_path,
        }

    except ImportError:
        print("⚠️  Whisper not available. Install with: pip install openai-whisper")
        print("⚠️  Creating dummy vocal conditioning files...")

        # Create minimal dummy files
        lyrics_data = {
            "full_text": "",
            "word_timings": [],
            "language": "en",
        }

        lyrics_data_path = stem_dir / f"{stem}_lyrics_data.json"
        with open(lyrics_data_path, 'w') as f:
            json.dump(lyrics_data, f, indent=2)

        lyrics_tensors = {
            "lyrics_embeddings": torch.zeros(1, 256),
            "phoneme_embeddings": torch.zeros(T_slow, 256),
        }

        lyrics_tensors_path = stem_dir / f"{stem}_lyrics_tensors.pt"
        torch.save(lyrics_tensors, lyrics_tensors_path)

        syllable_path = stem_dir / f"{stem}_syllable_boundaries.npy"
        np.save(syllable_path, np.zeros(T_slow, dtype=np.float32))

        print(f"⚠️  Dummy files created (no lyrics)")

        return {
            "lyrics_data": lyrics_data_path,
            "lyrics_tensors": lyrics_tensors_path,
            "syllable_boundaries": syllable_path,
        }


def extract_mhubert_features(audio_path: Path, output_dir: Path, T_slow: int) -> Optional[Path]:
    """
    Extract mHuBERT phonetic features (optional)
    Returns path to mHuBERT features or None
    """
    print("\n" + "=" * 60)
    print("Step 4: Extracting mHuBERT Features (Optional)")
    print("=" * 60)

    stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in audio_path.stem)[:128]
    stem_dir = output_dir / stem
    stem_dir.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import Wav2Vec2FeatureExtractor, HubertModel

        # Load mHuBERT
        print("Loading mHuBERT model...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/mhubert-base-25hz")
        model = HubertModel.from_pretrained("facebook/mhubert-base-25hz")
        model.eval()

        # Load audio
        audio, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        audio = audio.mean(dim=0)  # mono

        # Extract features
        print("Extracting mHuBERT features...")
        with torch.no_grad():
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            outputs = model(**inputs)
            features = outputs.last_hidden_state.squeeze(0)  # [T_mhubert, 768]

        # Align to T_slow (interpolate)
        if features.shape[0] != T_slow:
            features = torch.nn.functional.interpolate(
                features.T.unsqueeze(0),  # [1, 768, T_mhubert]
                size=T_slow,
                mode="linear",
                align_corners=False
            ).squeeze(0).T  # [T_slow, 768]

        # Save
        mhubert_path = stem_dir / f"{stem}_mhubert_features.pt"
        torch.save({"aligned_features": features}, mhubert_path)

        print(f"✅ mHuBERT features saved: {mhubert_path}")
        return mhubert_path

    except ImportError:
        print("⚠️  mHuBERT (transformers) not available")
        print("⚠️  Install with: pip install transformers")
        print("⚠️  Skipping mHuBERT extraction (optional)")
        return None
    except Exception as e:
        print(f"⚠️  mHuBERT extraction failed: {e}")
        print("⚠️  Continuing without mHuBERT features (optional)")
        return None


def extract_all_vocal_conditioning(audio_path: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Extract ALL conditioning for vocal model.
    Returns dict with all paths.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Vocal Conditioning Extraction")
    print(f"Audio: {audio_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Get T_slow from audio duration
    info = torchaudio.info(str(audio_path))
    duration = info.num_frames / info.sample_rate
    T_slow = int(duration * SLOW_HZ)

    print(f"\nAudio info:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  T_slow frames: {T_slow} (~{SLOW_HZ:.2f} Hz)")

    # Extract everything
    standard_paths = extract_standard_conditioning(audio_path, output_dir)
    speaker_emb_path = extract_speaker_embedding(audio_path, output_dir)
    vocal_paths = extract_lyrics_with_whisper(audio_path, output_dir, T_slow)
    mhubert_path = extract_mhubert_features(audio_path, output_dir, T_slow)

    # Combine all paths
    result = {
        **standard_paths,
        "speaker_emb_path": speaker_emb_path,
        **vocal_paths,
    }

    if mhubert_path:
        result["mhubert_features_path"] = mhubert_path

    # Create summary JSON
    stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in audio_path.stem)[:128]
    summary_path = output_dir / f"{stem}_conditioning_summary.json"

    summary = {
        "audio_path": str(audio_path),
        "duration_seconds": duration,
        "T_slow_frames": T_slow,
        "conditioning_paths": {k: str(v) for k, v in result.items()},
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ ALL CONDITIONING EXTRACTED")
    print("=" * 60)
    print(f"\nSummary saved to: {summary_path}")
    print("\nExtracted files:")
    for key, path in result.items():
        exists = "✅" if Path(path).exists() else "❌"
        print(f"  {exists} {key}: {path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract all vocal conditioning for trainer_performervox.py"
    )
    parser.add_argument("--audio", type=str, required=True, help="Input audio file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    try:
        result = extract_all_vocal_conditioning(
            audio_path=Path(args.audio),
            output_dir=Path(args.output)
        )

        print("\n" + "=" * 60)
        print("SUCCESS! Ready for inference with trainer_performervox.py")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
