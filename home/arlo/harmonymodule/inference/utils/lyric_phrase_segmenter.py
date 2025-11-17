#!/usr/bin/env python3
"""
Lyric Phrase Segmenter for ACE-Step Detailed Mode

Extracts lyric phrase timing for segmented generation.
Each phrase is processed separately with noise-to-noise architecture.

Usage:
    from lyric_phrase_segmenter import extract_phrase_timings
    phrases = extract_phrase_timings(audio_path, lyrics_text)
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchaudio

# Try to import MFA dependencies
try:
    from g2p_en import G2p
    G2P_AVAILABLE = True
except ImportError:
    G2P_AVAILABLE = False
    print("⚠ G2P not available. Using simple segmentation.")

# Try to import Whisper for auto-alignment
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

SAMPLE_RATE = 44100


def check_mfa_available():
    """Check if MFA is installed"""
    try:
        result = subprocess.run(['mfa', 'version'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def simple_phrase_split(lyrics: str) -> List[str]:
    """
    Simple lyric phrase splitting based on punctuation and line breaks.

    Args:
        lyrics: Full lyrics text

    Returns:
        List of lyric phrases
    """
    # Split by common phrase delimiters
    import re

    # Normalize line breaks
    lyrics = lyrics.replace('\r\n', '\n').replace('\r', '\n')

    # Split by punctuation or double line breaks
    phrases = re.split(r'[.!?,;]\s*|\n\n+', lyrics)

    # Clean up phrases
    phrases = [p.strip() for p in phrases if p.strip()]

    # Further split long phrases (more than 15 words) by line breaks
    final_phrases = []
    for phrase in phrases:
        words = phrase.split()
        if len(words) > 15 and '\n' in phrase:
            # Split by single line breaks
            sub_phrases = [sp.strip() for sp in phrase.split('\n') if sp.strip()]
            final_phrases.extend(sub_phrases)
        else:
            final_phrases.append(phrase)

    return final_phrases


def estimate_phrase_timings(
    audio_path: str,
    phrases: List[str],
    audio_duration: float
) -> List[Dict[str, any]]:
    """
    Estimate phrase timings by distributing evenly across audio duration.

    Args:
        audio_path: Path to audio file
        phrases: List of lyric phrases
        audio_duration: Total audio duration in seconds

    Returns:
        List of dicts with 'phrase', 'start_time', 'end_time'
    """
    if not phrases:
        return []

    # Calculate total "weight" (word count) for each phrase
    phrase_weights = [len(phrase.split()) for phrase in phrases]
    total_weight = sum(phrase_weights)

    # Allocate time proportionally to word count
    phrase_timings = []
    current_time = 0.0

    for i, phrase in enumerate(phrases):
        weight = phrase_weights[i]
        phrase_duration = (weight / total_weight) * audio_duration

        start_time = current_time
        end_time = min(current_time + phrase_duration, audio_duration)

        phrase_timings.append({
            'phrase': phrase,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        })

        current_time = end_time

    return phrase_timings


def align_phrases_with_mfa(
    audio_path: str,
    phrases: List[str],
    audio_duration: float
) -> List[Dict[str, any]]:
    """
    Use Montreal Forced Aligner to get accurate phrase timings.

    Args:
        audio_path: Path to audio file
        phrases: List of lyric phrases
        audio_duration: Total audio duration in seconds

    Returns:
        List of dicts with 'phrase', 'start_time', 'end_time'
    """
    try:
        # Create temporary directory for MFA
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_dir = temp_path / "audio"
            audio_dir.mkdir()

            # Copy audio file
            audio_name = "input.wav"
            import shutil
            shutil.copy(audio_path, audio_dir / audio_name)

            # Create transcript file (all phrases concatenated)
            transcript_path = audio_dir / f"{audio_name.replace('.wav', '.txt')}"
            with open(transcript_path, 'w') as f:
                f.write(' '.join(phrases))

            # Run MFA alignment
            output_dir = temp_path / "output"
            print(f"🔍 Running MFA alignment...")

            result = subprocess.run([
                'mfa', 'align',
                str(audio_dir),
                'english_us_arpa',
                'english_us_arpa',
                str(output_dir),
                '--clean'
            ], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"⚠ MFA alignment failed: {result.stderr}")
                return estimate_phrase_timings(audio_path, phrases, audio_duration)

            # Parse TextGrid output
            textgrid_path = output_dir / audio_name.replace('.wav', '.TextGrid')
            if not textgrid_path.exists():
                print(f"⚠ TextGrid not found, using estimated timings")
                return estimate_phrase_timings(audio_path, phrases, audio_duration)

            # Parse TextGrid to extract word timings
            word_timings = parse_textgrid(textgrid_path)

            # Map words to phrases
            phrase_timings = map_words_to_phrases(phrases, word_timings)

            print(f"✅ MFA alignment completed: {len(phrase_timings)} phrases")
            return phrase_timings

    except Exception as e:
        print(f"⚠ MFA alignment error: {e}")
        print("   Falling back to estimated timings")
        return estimate_phrase_timings(audio_path, phrases, audio_duration)


def parse_textgrid(textgrid_path: Path) -> List[Dict[str, any]]:
    """Parse MFA TextGrid file to extract word timings"""
    word_timings = []

    with open(textgrid_path, 'r') as f:
        lines = f.readlines()

    # Simple TextGrid parser (extract word tier)
    in_word_tier = False
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Find word tier
        if 'name = "words"' in line:
            in_word_tier = True
            i += 1
            continue

        # Exit word tier
        if in_word_tier and line.startswith('item'):
            break

        # Extract intervals
        if in_word_tier and 'xmin' in line:
            xmin = float(line.split('=')[1].strip())
            i += 1
            xmax = float(lines[i].split('=')[1].strip())
            i += 1
            text = lines[i].split('=')[1].strip().strip('"')

            if text and text not in ['', 'sp', 'sil']:
                word_timings.append({
                    'word': text,
                    'start_time': xmin,
                    'end_time': xmax
                })

        i += 1

    return word_timings


def map_words_to_phrases(
    phrases: List[str],
    word_timings: List[Dict[str, any]]
) -> List[Dict[str, any]]:
    """Map word timings to phrase boundaries"""
    phrase_timings = []
    word_index = 0

    for phrase in phrases:
        phrase_words = phrase.lower().split()
        phrase_word_count = len(phrase_words)

        if word_index >= len(word_timings):
            # No more word timings, estimate
            if phrase_timings:
                last_end = phrase_timings[-1]['end_time']
                phrase_timings.append({
                    'phrase': phrase,
                    'start_time': last_end,
                    'end_time': last_end + 3.0,  # Estimate 3 seconds
                    'duration': 3.0
                })
            continue

        # Find start and end times for this phrase
        start_time = word_timings[word_index]['start_time']

        # Advance through words in this phrase
        end_index = min(word_index + phrase_word_count, len(word_timings))
        end_time = word_timings[end_index - 1]['end_time']

        phrase_timings.append({
            'phrase': phrase,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        })

        word_index = end_index

    return phrase_timings


def extract_phrase_timings(
    audio_path: str,
    lyrics: str,
    use_mfa: bool = True
) -> List[Dict[str, any]]:
    """
    Main function to extract lyric phrase timings.

    Args:
        audio_path: Path to audio file
        lyrics: Full lyrics text
        use_mfa: Whether to use MFA for alignment (if available)

    Returns:
        List of phrase timing dicts with:
        - 'phrase': str (lyric phrase text)
        - 'start_time': float (seconds)
        - 'end_time': float (seconds)
        - 'duration': float (seconds)
    """
    # Get audio duration
    waveform, sr = torchaudio.load(audio_path)
    audio_duration = waveform.shape[1] / sr

    print(f"📝 Processing lyrics for {audio_duration:.2f}s audio")

    # Split lyrics into phrases
    phrases = simple_phrase_split(lyrics)
    print(f"   Split into {len(phrases)} phrases")

    # Try MFA alignment if available and requested
    if use_mfa and check_mfa_available() and G2P_AVAILABLE:
        print(f"   Using MFA for accurate alignment...")
        phrase_timings = align_phrases_with_mfa(audio_path, phrases, audio_duration)
    else:
        print(f"   Using estimated timings...")
        phrase_timings = estimate_phrase_timings(audio_path, phrases, audio_duration)

    # Print summary
    print(f"\n✅ Phrase segmentation complete:")
    for i, pt in enumerate(phrase_timings):
        print(f"   [{i+1}] {pt['start_time']:.2f}s - {pt['end_time']:.2f}s: \"{pt['phrase'][:50]}...\"")

    return phrase_timings


if __name__ == "__main__":
    # Test mode
    import argparse
    parser = argparse.ArgumentParser(description="Extract lyric phrase timings")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--lyrics", required=True, help="Path to lyrics text file or lyrics string")
    parser.add_argument("--no-mfa", action="store_true", help="Disable MFA alignment")

    args = parser.parse_args()

    # Load lyrics
    if os.path.exists(args.lyrics):
        with open(args.lyrics, 'r') as f:
            lyrics = f.read()
    else:
        lyrics = args.lyrics

    # Extract phrase timings
    phrase_timings = extract_phrase_timings(
        args.audio,
        lyrics,
        use_mfa=not args.no_mfa
    )

    # Save to JSON
    output_path = args.audio.replace('.wav', '_phrase_timings.json')
    with open(output_path, 'w') as f:
        json.dump(phrase_timings, f, indent=2)

    print(f"\n💾 Saved phrase timings to: {output_path}")
