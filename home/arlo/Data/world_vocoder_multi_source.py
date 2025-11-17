#!/usr/bin/env python3
"""
WORLD Vocoder with Multiple Audio Sources

Supports:
1. eSpeak-NG (fast, robotic)
2. FluidSynth soundfonts (instrumental timbres)
3. Custom audio files (your own recordings)

Usage:
    python world_vocoder_multi_source.py \\
        --source espeak \\
        --midi melody.mid \\
        --lyrics "do re mi fa sol" \\
        --output output.wav
"""

import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pretty_midi
import pyworld as pw
import soundfile as sf
import librosa
from typing import Optional


class AudioSourceGenerator:
    """Generate audio syllables from various sources."""

    def __init__(self, source_type: str = 'espeak', **kwargs):
        """
        Initialize audio source generator.

        Args:
            source_type: 'espeak', 'fluidsynth', 'file'
            **kwargs: Source-specific parameters
        """
        self.source_type = source_type
        self.kwargs = kwargs

    def generate_syllable(
        self,
        syllable: str,
        duration: float,
        sample_rate: int = 16000,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate audio for a syllable using configured source.

        Args:
            syllable: Text to synthesize
            duration: Target duration in seconds
            sample_rate: Sample rate
            output_path: Optional output file path

        Returns:
            Audio array
        """
        if self.source_type == 'espeak':
            return self._generate_espeak(syllable, duration, sample_rate, output_path)
        elif self.source_type == 'fluidsynth':
            return self._generate_fluidsynth(syllable, duration, sample_rate, output_path)
        elif self.source_type == 'file':
            return self._load_file(syllable, duration, sample_rate)
        else:
            raise ValueError(f"Unknown source type: {self.source_type}")

    def _generate_espeak(self, syllable, duration, sample_rate, output_path):
        """Generate with eSpeak-NG (original method)."""
        # Calculate WPM
        words_per_second = 1.0 / duration if duration > 0 else 3
        wpm = int(words_per_second * 60)
        wpm = max(80, min(450, wpm))

        # Generate with espeak
        cmd = [
            'espeak-ng',
            '-v', self.kwargs.get('voice', 'en-us'),
            '-s', str(wpm),
            '-p', '50',  # Middle pitch
            '--stdout',
            syllable
        ]

        result = subprocess.run(cmd, capture_output=True, check=True)

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(result.stdout)
            tmp_path = tmp.name

        # Load and time-stretch
        audio, sr = librosa.load(tmp_path, sr=sample_rate)
        current_duration = len(audio) / sample_rate

        if current_duration > 0 and duration > 0:
            stretch_rate = current_duration / duration
            audio = librosa.effects.time_stretch(audio, rate=stretch_rate)

        # Pad or trim to exact length
        target_samples = int(duration * sample_rate)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))

        Path(tmp_path).unlink()

        if output_path:
            sf.write(output_path, audio, sample_rate)

        return audio

    def _generate_fluidsynth(self, syllable, duration, sample_rate, output_path):
        """Generate with FluidSynth soundfont (vowel-like sounds)."""
        soundfont = self.kwargs.get('soundfont', '/home/arlo/Data/soundfonts/vocals1.sf2')

        # Create simple MIDI note at C4
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        note = pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=duration)
        instrument.notes.append(note)
        midi.instruments.append(instrument)

        # Save MIDI to temp file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp_midi:
            midi.write(tmp_midi.name)
            midi_path = tmp_midi.name

        # Render with FluidSynth
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            audio_path = tmp_audio.name

        subprocess.run([
            'fluidsynth', '-ni', '-g', '0.625', '-r', str(sample_rate),
            '-F', audio_path, soundfont, midi_path
        ], check=True, capture_output=True)

        # Load audio
        audio, sr = librosa.load(audio_path, sr=sample_rate)

        # Cleanup
        Path(midi_path).unlink()
        Path(audio_path).unlink()

        if output_path:
            sf.write(output_path, audio, sample_rate)

        return audio

    def _load_file(self, syllable, duration, sample_rate):
        """Load audio from file (syllable is file path)."""
        audio, sr = librosa.load(syllable, sr=sample_rate)

        # Time-stretch to target duration
        current_duration = len(audio) / sample_rate
        if current_duration > 0 and duration > 0:
            stretch_rate = current_duration / duration
            audio = librosa.effects.time_stretch(audio, rate=stretch_rate)

        # Adjust length
        target_samples = int(duration * sample_rate)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))

        return audio


def vocode_with_world(audio, target_freq, sample_rate=16000, frame_period=5.0, formant_boost=1.0):
    """
    Apply WORLD vocoding to set pitch.

    Args:
        audio: Input audio
        target_freq: Target frequency in Hz
        sample_rate: Sample rate
        frame_period: Frame period in ms
        formant_boost: Spectral envelope emphasis (1.0=normal, >1.0=sharper formants)
    """
    audio = audio.astype(np.float64)

    # Decompose
    f0, time_axis = pw.harvest(audio, sample_rate, frame_period=frame_period)
    sp = pw.cheaptrick(audio, f0, time_axis, sample_rate)
    ap = pw.d4c(audio, f0, time_axis, sample_rate)

    # Modify pitch
    modified_f0 = np.full_like(f0, target_freq)

    # Boost formants for clearer syllable articulation
    if formant_boost != 1.0:
        # Apply power scaling to spectral envelope
        # This emphasizes formant peaks (vowel/consonant characteristics)
        sp = np.power(sp, formant_boost)

    # Synthesize
    synthesized = pw.synthesize(modified_f0, sp, ap, sample_rate, frame_period=frame_period)

    return synthesized.astype(np.float32)


def synthesize_with_source(
    midi_path: str,
    lyrics: str,
    output_path: str,
    source_type: str = 'espeak',
    source_kwargs: dict = None,
    sample_rate: int = 16000,
    formant_boost: float = 1.0,
    verbose: bool = True
):
    """
    Main synthesis function with configurable audio source.

    Args:
        midi_path: MIDI file
        lyrics: Space-separated syllables
        output_path: Output WAV file
        source_type: 'espeak', 'fluidsynth', 'file'
        source_kwargs: Source-specific parameters
        sample_rate: Sample rate
        formant_boost: Spectral envelope emphasis (1.0=normal, 1.2-1.5=sharper syllables)
        verbose: Print progress
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"WORLD Vocoder with {source_type.upper()} source")
        print(f"{'='*70}")

    # Initialize source generator
    source_kwargs = source_kwargs or {}
    generator = AudioSourceGenerator(source_type, **source_kwargs)

    # Load MIDI
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted(midi.instruments[0].notes, key=lambda n: n.start)
    syllables = lyrics.split()

    # Align syllables to notes
    alignments = []
    for i, note in enumerate(notes):
        syl = syllables[i] if i < len(syllables) else (syllables[-1] if syllables else "")
        alignments.append((syl, note))

    if verbose:
        print(f"\nAlignments:")
        for i, (syl, note) in enumerate(alignments):
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            print(f"  {i+1}. '{syl}' → {freq:.1f} Hz @ {note.start:.2f}s")

    # Generate and vocode each syllable
    total_duration = midi.get_end_time()
    total_samples = int(total_duration * sample_rate)
    output_audio = np.zeros(total_samples, dtype=np.float32)

    if verbose:
        print(f"\nSynthesizing with {source_type}:")

    for i, (syllable, note) in enumerate(alignments):
        if not syllable.strip():
            continue

        duration = note.end - note.start
        target_freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))

        if verbose:
            print(f"  [{i+1}/{len(alignments)}] '{syllable}' ({duration:.2f}s, {target_freq:.1f} Hz)")

        # Generate syllable
        syl_audio = generator.generate_syllable(syllable, duration, sample_rate)

        # Vocode to target pitch with formant boost
        vocoded = vocode_with_world(syl_audio, target_freq, sample_rate, formant_boost=formant_boost)

        # Place in output buffer
        start_sample = int(note.start * sample_rate)
        end_sample = start_sample + len(vocoded)

        if end_sample > len(output_audio):
            vocoded = vocoded[:len(output_audio) - start_sample]
            end_sample = len(output_audio)

        output_audio[start_sample:end_sample] += vocoded

    # Normalize and save
    max_val = np.max(np.abs(output_audio))
    if max_val > 0:
        output_audio = output_audio / (max_val * 1.1)

    # Upsample to 44.1kHz
    if sample_rate != 44100:
        output_audio = librosa.resample(output_audio, orig_sr=sample_rate, target_sr=44100)
        save_sr = 44100
    else:
        save_sr = sample_rate

    sf.write(output_path, output_audio, save_sr)

    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ Complete: {output_path}")
        print(f"{'='*70}\n")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='WORLD Vocoder with Multiple Audio Sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # eSpeak (default)
  python world_vocoder_multi_source.py --source espeak --midi melody.mid --lyrics "do re mi" --output out.wav

  # FluidSynth soundfont
  python world_vocoder_multi_source.py --source fluidsynth --midi melody.mid --lyrics "ah eh ee" --output out.wav --soundfont /path/to/vocals.sf2
        """
    )

    parser.add_argument('--source', default='espeak',
                       choices=['espeak', 'fluidsynth', 'file'],
                       help='Audio source type')
    parser.add_argument('--midi', required=True, help='Input MIDI file')
    parser.add_argument('--lyrics', required=True, help='Space-separated syllables')
    parser.add_argument('--output', required=True, help='Output WAV file')
    parser.add_argument('--soundfont', help='Soundfont path (for fluidsynth source)')
    parser.add_argument('--formant-boost', type=float, default=1.0,
                       help='Formant emphasis (1.0=normal, 1.2-1.5=sharper syllables)')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Build source kwargs
    source_kwargs = {}
    if args.soundfont:
        source_kwargs['soundfont'] = args.soundfont

    synthesize_with_source(
        midi_path=args.midi,
        lyrics=args.lyrics,
        output_path=args.output,
        source_type=args.source,
        source_kwargs=source_kwargs,
        formant_boost=args.formant_boost,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
