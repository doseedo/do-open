#!/usr/bin/env python3
"""
eSpeak Singing Synthesizer - MIDI + Lyrics to Singing Voice

Uses eSpeak-NG with pitch shifting and time stretching to create singing voice
from MIDI and lyrics. Fast, local, fully offline, but sounds robotic.

Dependencies:
    - espeak-ng (system): sudo apt install espeak-ng
    - librosa: pip install librosa
    - soundfile: pip install soundfile
    - pretty_midi: pip install pretty_midi

Usage:
    from espeak_singing_synth import ESpeakSinger

    singer = ESpeakSinger()
    singer.synthesize(
        midi_path='melody.mid',
        lyrics='Hello world this is singing',
        output_path='output.wav'
    )
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pretty_midi
from dataclasses import dataclass


@dataclass
class NoteAlignment:
    """MIDI note aligned with lyrics."""
    text: str
    start_time: float
    end_time: float
    pitch_midi: int
    velocity: int

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class ESpeakSinger:
    """
    Singing voice synthesizer using eSpeak-NG with pitch/time manipulation.

    Process:
    1. Load MIDI and extract notes
    2. Align lyrics syllables/words to MIDI notes
    3. Generate speech for each syllable with eSpeak
    4. Pitch-shift to match MIDI pitch
    5. Time-stretch to match note duration
    6. Concatenate into final singing voice
    """

    def __init__(
        self,
        voice: str = 'en-us',
        speech_rate: int = 150,
        base_pitch: int = 60,  # Middle C
        sample_rate: int = 22050
    ):
        """
        Initialize eSpeak singer.

        Args:
            voice: eSpeak voice (e.g., 'en-us', 'en-gb', 'es', 'fr')
            speech_rate: Speech speed (words per minute, 80-450)
            base_pitch: Base MIDI pitch for eSpeak (default: 60 = C4)
            sample_rate: Audio sample rate in Hz
        """
        self.voice = voice
        self.speech_rate = speech_rate
        self.base_pitch = base_pitch
        self.sample_rate = sample_rate

        # Check dependencies
        self._check_dependencies()

    def _check_dependencies(self):
        """Verify required software is installed."""
        # Check eSpeak-NG
        try:
            result = subprocess.run(
                ['espeak-ng', '--version'],
                capture_output=True,
                check=True
            )
            print(f"✅ Found eSpeak-NG: {result.stdout.decode().split()[2]}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "espeak-ng not found. Install with:\n"
                "  Ubuntu/Debian: sudo apt install espeak-ng\n"
                "  macOS: brew install espeak-ng"
            )

        # Check Python libraries
        try:
            import librosa
            print(f"✅ Found librosa: {librosa.__version__}")
        except ImportError:
            raise RuntimeError("librosa not found. Install: pip install librosa")

        try:
            import soundfile
            print(f"✅ Found soundfile: {soundfile.__version__}")
        except ImportError:
            raise RuntimeError("soundfile not found. Install: pip install soundfile")

    def synthesize(
        self,
        midi_path: Optional[str] = None,
        lyrics: Optional[str] = None,
        output_path: str = None,
        audio_path: Optional[str] = None,
        alignment_mode: str = 'syllable',
        add_breathiness: bool = True,
        verbose: bool = True
    ) -> str:
        """
        Synthesize singing voice from MIDI and lyrics.

        Can work in two modes:
        1. Direct: Provide midi_path and lyrics directly
        2. Extract: Provide audio_path to auto-extract MIDI and lyrics

        Args:
            midi_path: Path to MIDI file (or None for auto-extract)
            lyrics: Lyrics text (or None for auto-extract)
            output_path: Output WAV file path
            audio_path: Audio file to extract MIDI and lyrics from
            alignment_mode: 'syllable' or 'phoneme' (syllable recommended)
            add_breathiness: Add slight noise for naturalness
            verbose: Print progress information

        Returns:
            Path to output audio file
        """
        # Auto-extract mode
        if audio_path is not None:
            if verbose:
                print(f"\n{'='*70}")
                print("Auto-extracting MIDI and lyrics from audio")
                print(f"{'='*70}")

            midi_path, lyrics = self._extract_from_audio(audio_path, verbose=verbose)

        # Validate inputs
        if midi_path is None or lyrics is None:
            raise ValueError("Must provide either (midi_path + lyrics) or audio_path")
        if verbose:
            print(f"\n{'='*70}")
            print("eSpeak Singing Synthesizer")
            print(f"{'='*70}")

        # Import here to avoid loading if dependency check fails
        import librosa
        import soundfile as sf

        # Load MIDI and align with lyrics
        alignments = self._align_lyrics_to_midi(midi_path, lyrics, verbose=verbose)

        if not alignments:
            raise ValueError("No notes found in MIDI file for alignment")

        # Get total duration
        midi = pretty_midi.PrettyMIDI(midi_path)
        total_duration = midi.get_end_time()

        if verbose:
            print(f"\nMIDI Info:")
            print(f"  Duration: {total_duration:.2f}s")
            print(f"  Notes: {len(alignments)}")
            print(f"  Lyrics: {lyrics[:60]}{'...' if len(lyrics) > 60 else ''}")

        # Create output audio buffer
        total_samples = int(total_duration * self.sample_rate)
        output_audio = np.zeros(total_samples)

        # Synthesize each aligned segment
        with tempfile.TemporaryDirectory() as tmpdir:
            if verbose:
                print(f"\nSynthesizing segments:")

            for i, align in enumerate(alignments):
                if not align.text.strip():
                    # Skip empty text
                    continue

                # Generate speech segment
                segment_audio = self._synthesize_segment(
                    text=align.text,
                    pitch_midi=align.pitch_midi,
                    duration=align.duration,
                    velocity=align.velocity,
                    tmpdir=tmpdir,
                    segment_id=i
                )

                # Place segment in output buffer
                start_sample = int(align.start_time * self.sample_rate)
                end_sample = start_sample + len(segment_audio)

                # Clip if exceeds total duration
                if end_sample > total_samples:
                    segment_audio = segment_audio[:total_samples - start_sample]
                    end_sample = total_samples

                # Apply velocity as volume
                volume = align.velocity / 127.0
                segment_audio = segment_audio * volume

                # Mix into output (overlap adds)
                output_audio[start_sample:end_sample] += segment_audio

                if verbose:
                    print(f"  [{i+1:3d}/{len(alignments)}] "
                          f"{align.start_time:6.2f}s | "
                          f"MIDI {align.pitch_midi:3d} | "
                          f"'{align.text}'")

        # Add slight breathiness/noise for naturalness
        if add_breathiness:
            breath = np.random.randn(len(output_audio)) * 0.002
            output_audio = output_audio + breath

        # Normalize to prevent clipping
        max_val = np.max(np.abs(output_audio))
        if max_val > 0:
            output_audio = output_audio / (max_val * 1.1)  # Leave headroom

        # Save output
        sf.write(output_path, output_audio, self.sample_rate)

        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ Synthesized singing voice: {output_path}")
            print(f"   Duration: {total_duration:.2f}s")
            print(f"   Sample rate: {self.sample_rate} Hz")
            print(f"{'='*70}\n")

        return output_path

    def _align_lyrics_to_midi(
        self,
        midi_path: str,
        lyrics: str,
        verbose: bool = True
    ) -> List[NoteAlignment]:
        """
        Align lyrics to MIDI notes.

        Simple approach: split lyrics by spaces and distribute across notes.
        For better alignment, could use:
        - Syllable counting (pyphen)
        - Phoneme detection (epitran)
        - Duration-weighted distribution
        """
        # Load MIDI
        midi = pretty_midi.PrettyMIDI(midi_path)

        # Get notes from first instrument
        if len(midi.instruments) == 0:
            raise ValueError("MIDI file contains no instruments")

        notes = sorted(midi.instruments[0].notes, key=lambda n: n.start)

        if not notes:
            raise ValueError("MIDI instrument contains no notes")

        # Split lyrics into words/syllables
        words = lyrics.split()

        if not words:
            raise ValueError("Lyrics are empty")

        # Align words to notes
        alignments = []

        if len(words) <= len(notes):
            # More notes than words: distribute words evenly
            notes_per_word = len(notes) / len(words)

            for i, note in enumerate(notes):
                word_idx = int(i / notes_per_word)
                word_idx = min(word_idx, len(words) - 1)

                alignment = NoteAlignment(
                    text=words[word_idx],
                    start_time=note.start,
                    end_time=note.end,
                    pitch_midi=note.pitch,
                    velocity=note.velocity
                )
                alignments.append(alignment)
        else:
            # More words than notes: multiple words per note
            words_per_note = len(words) / len(notes)

            for i, note in enumerate(notes):
                start_word_idx = int(i * words_per_note)
                end_word_idx = int((i + 1) * words_per_note)
                end_word_idx = min(end_word_idx, len(words))

                # Combine words for this note
                text = ' '.join(words[start_word_idx:end_word_idx])

                alignment = NoteAlignment(
                    text=text,
                    start_time=note.start,
                    end_time=note.end,
                    pitch_midi=note.pitch,
                    velocity=note.velocity
                )
                alignments.append(alignment)

        return alignments

    def _synthesize_segment(
        self,
        text: str,
        pitch_midi: int,
        duration: float,
        velocity: int,
        tmpdir: str,
        segment_id: int
    ) -> np.ndarray:
        """
        Synthesize a single segment (word/syllable) with pitch and time manipulation.

        Args:
            text: Text to speak
            pitch_midi: Target MIDI pitch (0-127)
            duration: Target duration in seconds
            velocity: MIDI velocity (0-127)
            tmpdir: Temporary directory for intermediate files
            segment_id: Unique ID for this segment

        Returns:
            Audio samples as numpy array
        """
        import librosa

        # Generate speech with eSpeak
        wav_path = Path(tmpdir) / f"seg_{segment_id}.wav"

        cmd = [
            'espeak-ng',
            '-v', self.voice,
            '-s', str(self.speech_rate),
            '-w', str(wav_path),
            text
        ]

        subprocess.run(cmd, capture_output=True, check=True)

        # Load generated audio
        audio, sr = librosa.load(wav_path, sr=self.sample_rate)

        # Calculate pitch shift in semitones
        pitch_shift_semitones = pitch_midi - self.base_pitch

        # Apply pitch shift
        if pitch_shift_semitones != 0:
            audio = librosa.effects.pitch_shift(
                audio,
                sr=self.sample_rate,
                n_steps=pitch_shift_semitones
            )

        # Time stretch to match target duration
        current_duration = len(audio) / self.sample_rate

        if current_duration > 0 and duration > 0:
            # stretch_rate > 1 speeds up, < 1 slows down
            stretch_rate = current_duration / duration
            audio = librosa.effects.time_stretch(audio, rate=stretch_rate)

        # Apply fade in/out to avoid clicks
        fade_samples = min(int(0.005 * self.sample_rate), len(audio) // 10)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out

        return audio


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='eSpeak Singing Synthesizer - MIDI + Lyrics to Singing Voice',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python espeak_singing_synth.py \\
      --midi melody.mid \\
      --lyrics "Hello world this is singing" \\
      --output singing.wav

  # With custom voice and rate
  python espeak_singing_synth.py \\
      --midi melody.mid \\
      --lyrics "La la la la la" \\
      --output singing.wav \\
      --voice en-gb \\
      --rate 180

  # Use extracted MIDI
  python espeak_singing_synth.py \\
      --midi /home/arlo/Data/test/vocals_extracted.mid \\
      --lyrics "In a river the color of land" \\
      --output espeak_singing_test.wav
        """
    )

    parser.add_argument('--midi', type=str, required=True,
                       help='Input MIDI file path')
    parser.add_argument('--lyrics', type=str, required=True,
                       help='Lyrics text (words/syllables separated by spaces)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output WAV file path')

    # Optional parameters
    parser.add_argument('--voice', type=str, default='en-us',
                       help='eSpeak voice (default: en-us)')
    parser.add_argument('--rate', type=int, default=150,
                       help='Speech rate in WPM (default: 150)')
    parser.add_argument('--base-pitch', type=int, default=60,
                       help='Base MIDI pitch for eSpeak (default: 60 = C4)')
    parser.add_argument('--no-breathiness', action='store_true',
                       help='Disable breathiness effect')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    # Create singer
    singer = ESpeakSinger(
        voice=args.voice,
        speech_rate=args.rate,
        base_pitch=args.base_pitch
    )

    # Synthesize
    singer.synthesize(
        midi_path=args.midi,
        lyrics=args.lyrics,
        output_path=args.output,
        add_breathiness=not args.no_breathiness,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
