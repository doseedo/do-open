#!/usr/bin/env python3
"""
Vocal Harmonizer - Generate vocal harmonies from input vocals

Pipeline:
1. Extract melody (pitch) from vocal audio using CREPE/librosa
2. Extract lyrics and syllable timing using Whisper
3. Generate harmony MIDI notes using melody_harmonizer logic
4. Pitch-shift original vocal audio to harmony notes using librosa
5. Optionally render through ACE-Step for natural vocal generation

Usage:
    python vocal_harmonizer.py --input vocals.wav --output harmonized/ \
        --num-harmonies 2 --voicing "5-way-closed" --noise-level 0.8
"""

import os
import sys
import json
import argparse
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# Audio processing
import librosa
import soundfile as sf

# MIDI handling
import mido

# Try to import optional dependencies
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. Lyrics extraction disabled.")

try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False
    print("Warning: CREPE not available. Using librosa for pitch detection.")

# Import harmony modules from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from melody_harmonizer import HarmonicAnalyzer, MelodyVoicing
    from chord_progression_generator import ScaleContext
    HARMONIZER_AVAILABLE = True
except ImportError:
    HARMONIZER_AVAILABLE = False
    print("Warning: melody_harmonizer not importable. Using basic harmony logic.")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PitchEvent:
    """A detected pitch event from vocals"""
    start_time: float  # seconds
    end_time: float    # seconds
    pitch_hz: float    # Hz
    midi_note: int     # MIDI note number
    confidence: float  # 0.0 to 1.0
    syllable: str = "" # Associated syllable (if lyrics available)


@dataclass
class HarmonyVoice:
    """A generated harmony voice"""
    voice_index: int      # 0=unison, 1=harmony1, 2=harmony2, etc.
    interval: int         # Semitones from original (-12 to +12)
    events: List[PitchEvent]
    audio: Optional[np.ndarray] = None


# ============================================================================
# PITCH EXTRACTION
# ============================================================================

def extract_pitch_crepe(
    audio_path: str,
    step_size: int = 10,  # ms
    min_confidence: float = 0.5
) -> List[PitchEvent]:
    """
    Extract pitch from audio using CREPE (if available).

    CREPE is more accurate than librosa for vocals.
    """
    if not CREPE_AVAILABLE:
        return extract_pitch_librosa(audio_path, min_confidence)

    print(f"Extracting pitch using CREPE...")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Run CREPE
    time_arr, frequency, confidence, activation = crepe.predict(
        audio, sr,
        step_size=step_size,
        viterbi=True  # Use Viterbi for smoother pitch tracking
    )

    # Convert to pitch events
    events = []
    current_event = None

    for i, (t, f, c) in enumerate(zip(time_arr, frequency, confidence)):
        if c >= min_confidence and f > 50:  # Valid pitch
            midi_note = int(round(librosa.hz_to_midi(f)))

            if current_event is None:
                # Start new event
                current_event = PitchEvent(
                    start_time=t,
                    end_time=t,
                    pitch_hz=f,
                    midi_note=midi_note,
                    confidence=c
                )
            elif abs(midi_note - current_event.midi_note) <= 1:
                # Extend current event (same note or glide)
                current_event.end_time = t
                current_event.pitch_hz = (current_event.pitch_hz + f) / 2
                current_event.confidence = max(current_event.confidence, c)
            else:
                # New note - save current and start new
                if current_event.end_time - current_event.start_time > 0.05:
                    events.append(current_event)
                current_event = PitchEvent(
                    start_time=t,
                    end_time=t,
                    pitch_hz=f,
                    midi_note=midi_note,
                    confidence=c
                )
        else:
            # Low confidence - end current event
            if current_event is not None:
                if current_event.end_time - current_event.start_time > 0.05:
                    events.append(current_event)
                current_event = None

    # Don't forget last event
    if current_event is not None and current_event.end_time - current_event.start_time > 0.05:
        events.append(current_event)

    print(f"  Extracted {len(events)} pitch events")
    return events


def extract_pitch_librosa(
    audio_path: str,
    min_confidence: float = 0.5,
    hop_length: int = 512
) -> List[PitchEvent]:
    """
    Extract pitch from audio using librosa's pyin algorithm.

    This is a fallback if CREPE is not available.
    """
    print(f"Extracting pitch using librosa pyin...")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Extract pitch using pyin (probabilistic YIN)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length
    )

    # Convert to time
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # Convert to pitch events
    events = []
    current_event = None

    for i, (t, f, voiced, prob) in enumerate(zip(times, f0, voiced_flag, voiced_probs)):
        if voiced and not np.isnan(f) and f > 50 and prob >= min_confidence:
            midi_note = int(round(librosa.hz_to_midi(f)))

            if current_event is None:
                current_event = PitchEvent(
                    start_time=t,
                    end_time=t,
                    pitch_hz=f,
                    midi_note=midi_note,
                    confidence=prob
                )
            elif abs(midi_note - current_event.midi_note) <= 1:
                current_event.end_time = t
                current_event.pitch_hz = (current_event.pitch_hz + f) / 2
                current_event.confidence = max(current_event.confidence, prob)
            else:
                if current_event.end_time - current_event.start_time > 0.05:
                    events.append(current_event)
                current_event = PitchEvent(
                    start_time=t,
                    end_time=t,
                    pitch_hz=f,
                    midi_note=midi_note,
                    confidence=prob
                )
        else:
            if current_event is not None:
                if current_event.end_time - current_event.start_time > 0.05:
                    events.append(current_event)
                current_event = None

    if current_event is not None and current_event.end_time - current_event.start_time > 0.05:
        events.append(current_event)

    print(f"  Extracted {len(events)} pitch events")
    return events


# ============================================================================
# LYRICS EXTRACTION
# ============================================================================

def extract_lyrics_with_timing(
    audio_path: str,
    model_size: str = "base"
) -> Tuple[str, List[Dict]]:
    """
    Extract lyrics and word timing from audio using Whisper.

    Returns:
        Tuple of (full_lyrics_text, word_timings)
        word_timings is a list of {"word": str, "start": float, "end": float}
    """
    if not WHISPER_AVAILABLE:
        print("  Whisper not available, skipping lyrics extraction")
        return "", []

    print(f"Extracting lyrics using Whisper ({model_size})...")

    # Load Whisper model
    model = whisper.load_model(model_size)

    # Transcribe with word-level timestamps
    result = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=True
    )

    # Extract word timings
    word_timings = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            word_timings.append({
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"]
            })

    full_lyrics = result.get("text", "").strip()

    print(f"  Extracted lyrics: \"{full_lyrics[:80]}...\"")
    print(f"  Word timings: {len(word_timings)} words")

    return full_lyrics, word_timings


def align_syllables_to_pitches(
    pitch_events: List[PitchEvent],
    word_timings: List[Dict]
) -> List[PitchEvent]:
    """
    Align word/syllable timings to pitch events.

    Assigns the closest word to each pitch event based on timing overlap.
    """
    if not word_timings:
        return pitch_events

    print(f"Aligning {len(word_timings)} words to {len(pitch_events)} pitch events...")

    aligned_events = []
    word_idx = 0

    for event in pitch_events:
        event_center = (event.start_time + event.end_time) / 2

        # Find the word that best overlaps with this pitch event
        best_word = ""
        best_overlap = 0

        for word_info in word_timings:
            # Calculate overlap
            overlap_start = max(event.start_time, word_info["start"])
            overlap_end = min(event.end_time, word_info["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_word = word_info["word"]

        # Also check for closest word by center time
        if not best_word:
            closest_dist = float('inf')
            for word_info in word_timings:
                word_center = (word_info["start"] + word_info["end"]) / 2
                dist = abs(word_center - event_center)
                if dist < closest_dist:
                    closest_dist = dist
                    if closest_dist < 0.5:  # Within 500ms
                        best_word = word_info["word"]

        # Create new event with syllable
        new_event = PitchEvent(
            start_time=event.start_time,
            end_time=event.end_time,
            pitch_hz=event.pitch_hz,
            midi_note=event.midi_note,
            confidence=event.confidence,
            syllable=best_word
        )
        aligned_events.append(new_event)

    syllable_count = sum(1 for e in aligned_events if e.syllable)
    print(f"  Aligned {syllable_count}/{len(aligned_events)} events with syllables")

    return aligned_events


# ============================================================================
# HARMONY GENERATION
# ============================================================================

# Common harmony intervals (in semitones)
HARMONY_PRESETS = {
    "thirds": [-4, -3, 3, 4],          # Major/minor thirds
    "fifths": [-7, 7],                  # Perfect fifths
    "octaves": [-12, 12],               # Octaves
    "close": [-3, -5, 3, 5],            # Close harmony (3rds and 5ths)
    "barbershop": [-4, -7, -12],        # Barbershop quartet style
    "gospel": [-3, -5, 3, 7],           # Gospel style (rich)
    "unison_octave": [0, -12, 12],      # Unison with octave doubling
}


def select_harmony_intervals(
    melody_note: int,
    num_harmonies: int,
    voicing_type: str = "close",
    scale_context: Optional['ScaleContext'] = None
) -> List[int]:
    """
    Select appropriate harmony intervals for a melody note.

    Args:
        melody_note: MIDI note number of melody
        num_harmonies: Number of harmony voices to generate
        voicing_type: Type of voicing ("close", "thirds", "gospel", etc.)
        scale_context: Optional scale context for diatonic selection

    Returns:
        List of intervals in semitones (e.g., [-4, 3] for 2 harmonies)
    """
    if voicing_type in HARMONY_PRESETS:
        available = HARMONY_PRESETS[voicing_type]
    else:
        # Default to close harmony
        available = [-3, -5, -7, 3, 5, 7]

    # Filter to diatonic intervals if scale context provided
    if scale_context and HARMONIZER_AVAILABLE:
        diatonic_intervals = []
        for iv in available:
            harmony_note = melody_note + iv
            if scale_context.is_diatonic(harmony_note):
                diatonic_intervals.append(iv)
        if diatonic_intervals:
            available = diatonic_intervals

    # Select intervals prioritizing variety
    selected = []
    below = [iv for iv in available if iv < 0]
    above = [iv for iv in available if iv > 0]

    # Alternate between below and above for balanced sound
    for i in range(num_harmonies):
        if i % 2 == 0 and below:
            selected.append(below[i // 2 % len(below)])
        elif above:
            selected.append(above[i // 2 % len(above)])
        elif below:
            selected.append(below[i % len(below)])

    return selected[:num_harmonies]


def generate_harmony_voices(
    pitch_events: List[PitchEvent],
    num_harmonies: int = 2,
    voicing_type: str = "close",
    key: str = "C"
) -> List[HarmonyVoice]:
    """
    Generate harmony voices for the detected pitch events.

    Args:
        pitch_events: List of melody pitch events
        num_harmonies: Number of harmony voices (1-4)
        voicing_type: Voicing style preset
        key: Musical key (e.g., "C", "Am", "F#")

    Returns:
        List of HarmonyVoice objects
    """
    print(f"\nGenerating {num_harmonies} harmony voices ({voicing_type} voicing, key={key})...")

    # Parse scale context
    scale_context = None
    if HARMONIZER_AVAILABLE:
        try:
            is_minor = 'm' in key.lower()
            # Parse root note
            root_name = key.replace('m', '').replace('#', '').replace('b', '').upper()
            root_map = {'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71}
            root = root_map.get(root_name, 60)
            if '#' in key:
                root += 1
            elif 'b' in key:
                root -= 1
            scale_type = 'minor' if is_minor else 'major'
            scale_context = ScaleContext(root, scale_type)
            print(f"  Scale context: {key} ({scale_type})")
        except Exception as e:
            print(f"  Could not parse key '{key}': {e}")

    # Generate harmony voices
    harmony_voices = []

    for voice_idx in range(num_harmonies):
        voice_events = []

        for event in pitch_events:
            # Select interval for this voice and note
            intervals = select_harmony_intervals(
                event.midi_note,
                num_harmonies,
                voicing_type,
                scale_context
            )

            if voice_idx < len(intervals):
                interval = intervals[voice_idx]
            else:
                # Fallback to third below
                interval = -3 if voice_idx % 2 == 0 else 3

            # Create harmony event
            harmony_event = PitchEvent(
                start_time=event.start_time,
                end_time=event.end_time,
                pitch_hz=librosa.midi_to_hz(event.midi_note + interval),
                midi_note=event.midi_note + interval,
                confidence=event.confidence,
                syllable=event.syllable
            )
            voice_events.append(harmony_event)

        # Create voice
        voice = HarmonyVoice(
            voice_index=voice_idx + 1,
            interval=intervals[voice_idx] if voice_idx < len(intervals) else -3,
            events=voice_events
        )
        harmony_voices.append(voice)

        print(f"  Voice {voice_idx + 1}: {len(voice_events)} events, interval range: {voice.interval}")

    return harmony_voices


# ============================================================================
# AUDIO PITCH SHIFTING
# ============================================================================

def pitch_shift_audio(
    audio: np.ndarray,
    sr: int,
    events: List[PitchEvent],
    harmony_events: List[PitchEvent]
) -> np.ndarray:
    """
    Pitch shift audio segments to match harmony notes.

    Uses librosa's pitch_shift for each segment based on the
    interval between original and harmony notes.

    Args:
        audio: Original audio array
        sr: Sample rate
        events: Original pitch events
        harmony_events: Target harmony pitch events

    Returns:
        Pitch-shifted audio array
    """
    print(f"Pitch shifting audio ({len(events)} segments)...")

    # Create output buffer
    output = np.zeros_like(audio)

    for orig_event, harm_event in zip(events, harmony_events):
        # Calculate pitch shift in semitones
        shift = harm_event.midi_note - orig_event.midi_note

        if shift == 0:
            # No shift needed
            start_sample = int(orig_event.start_time * sr)
            end_sample = int(orig_event.end_time * sr)
            end_sample = min(end_sample, len(audio))
            output[start_sample:end_sample] = audio[start_sample:end_sample]
        else:
            # Extract segment
            start_sample = int(orig_event.start_time * sr)
            end_sample = int(orig_event.end_time * sr)
            end_sample = min(end_sample, len(audio))
            segment = audio[start_sample:end_sample]

            if len(segment) > sr // 10:  # At least 100ms
                # Pitch shift the segment
                shifted = librosa.effects.pitch_shift(
                    segment,
                    sr=sr,
                    n_steps=shift,
                    bins_per_octave=12
                )

                # Ensure same length
                if len(shifted) != len(segment):
                    shifted = librosa.util.fix_length(shifted, size=len(segment))

                output[start_sample:end_sample] = shifted
            else:
                output[start_sample:end_sample] = segment

    # Normalize
    max_val = np.abs(output).max()
    if max_val > 0:
        output = output / max_val * 0.9

    return output


def create_harmony_audio(
    audio_path: str,
    pitch_events: List[PitchEvent],
    harmony_voices: List[HarmonyVoice],
    output_dir: str
) -> List[str]:
    """
    Create audio files for each harmony voice by pitch-shifting original.

    Args:
        audio_path: Path to original audio
        pitch_events: Original pitch events
        harmony_voices: Generated harmony voices
        output_dir: Output directory

    Returns:
        List of output audio file paths
    """
    print(f"\nCreating harmony audio files...")

    # Load original audio
    audio, sr = librosa.load(audio_path, sr=44100, mono=True)

    output_paths = []

    for voice in harmony_voices:
        # Pitch shift to harmony
        harmony_audio = pitch_shift_audio(audio, sr, pitch_events, voice.events)

        # Save
        output_path = os.path.join(
            output_dir,
            f"harmony_voice_{voice.voice_index}.wav"
        )
        sf.write(output_path, harmony_audio, sr)

        voice.audio = harmony_audio
        output_paths.append(output_path)

        print(f"  Saved: {os.path.basename(output_path)}")

    return output_paths


# ============================================================================
# MIDI EXPORT
# ============================================================================

def export_harmony_midi(
    pitch_events: List[PitchEvent],
    harmony_voices: List[HarmonyVoice],
    output_path: str,
    tempo: int = 120
) -> str:
    """
    Export all voices (original + harmonies) to a MIDI file.

    Args:
        pitch_events: Original melody pitch events
        harmony_voices: Generated harmony voices
        output_path: Output MIDI file path
        tempo: Tempo in BPM

    Returns:
        Path to output MIDI file
    """
    print(f"\nExporting harmony MIDI to {output_path}...")

    mid = mido.MidiFile(ticks_per_beat=480)

    # Set tempo
    tempo_us = mido.bpm2tempo(tempo)

    # Create track for original melody
    melody_track = mido.MidiTrack()
    mid.tracks.append(melody_track)
    melody_track.append(mido.MetaMessage('track_name', name='Lead Vocal'))
    melody_track.append(mido.MetaMessage('set_tempo', tempo=tempo_us))

    # Add melody notes
    current_tick = 0
    for event in sorted(pitch_events, key=lambda e: e.start_time):
        start_tick = int(event.start_time * 480 * tempo / 60)
        end_tick = int(event.end_time * 480 * tempo / 60)
        duration = max(end_tick - start_tick, 10)

        delta = start_tick - current_tick
        melody_track.append(mido.Message(
            'note_on',
            note=event.midi_note,
            velocity=80,
            time=max(0, delta)
        ))
        melody_track.append(mido.Message(
            'note_off',
            note=event.midi_note,
            velocity=0,
            time=duration
        ))
        current_tick = start_tick + duration

    # Create tracks for harmonies
    for voice in harmony_voices:
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage('track_name', name=f'Harmony {voice.voice_index}'))

        current_tick = 0
        for event in sorted(voice.events, key=lambda e: e.start_time):
            start_tick = int(event.start_time * 480 * tempo / 60)
            end_tick = int(event.end_time * 480 * tempo / 60)
            duration = max(end_tick - start_tick, 10)

            delta = start_tick - current_tick
            track.append(mido.Message(
                'note_on',
                note=event.midi_note,
                velocity=70,
                time=max(0, delta)
            ))
            track.append(mido.Message(
                'note_off',
                note=event.midi_note,
                velocity=0,
                time=duration
            ))
            current_tick = start_tick + duration

    mid.save(output_path)
    print(f"  Saved MIDI with {len(mid.tracks)} tracks")

    return output_path


# ============================================================================
# MAIN HARMONIZER FUNCTION
# ============================================================================

def harmonize_vocals(
    input_audio: str,
    output_dir: str,
    num_harmonies: int = 2,
    voicing_type: str = "close",
    key: str = "C",
    noise_level: float = 0.8,
    use_ace_step: bool = False,
    extract_lyrics: bool = True,
    tempo: int = 120
) -> Dict:
    """
    Main function to harmonize vocals.

    Args:
        input_audio: Path to input vocal audio file
        output_dir: Output directory for generated files
        num_harmonies: Number of harmony voices (1-4)
        voicing_type: Voicing style ("close", "thirds", "gospel", etc.)
        key: Musical key (e.g., "C", "Am", "F#")
        noise_level: Noise level for ACE-Step (0.0-1.0)
        use_ace_step: Whether to render through ACE-Step
        extract_lyrics: Whether to extract lyrics with Whisper
        tempo: Tempo in BPM for MIDI export

    Returns:
        Dict with output file paths and metadata
    """
    print(f"\n{'='*80}")
    print(f"VOCAL HARMONIZER")
    print(f"{'='*80}")
    print(f"  Input: {input_audio}")
    print(f"  Output dir: {output_dir}")
    print(f"  Harmonies: {num_harmonies}")
    print(f"  Voicing: {voicing_type}")
    print(f"  Key: {key}")
    print(f"  Noise level: {noise_level}")
    print(f"  ACE-Step: {use_ace_step}")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract pitch from vocals
    print("\n[1/5] Extracting pitch from vocals...")
    if CREPE_AVAILABLE:
        pitch_events = extract_pitch_crepe(input_audio)
    else:
        pitch_events = extract_pitch_librosa(input_audio)

    if not pitch_events:
        raise ValueError("No pitch events detected in audio")

    # Step 2: Extract lyrics (optional)
    lyrics = ""
    word_timings = []
    if extract_lyrics and WHISPER_AVAILABLE:
        print("\n[2/5] Extracting lyrics...")
        lyrics, word_timings = extract_lyrics_with_timing(input_audio)

        # Align syllables to pitch events
        pitch_events = align_syllables_to_pitches(pitch_events, word_timings)
    else:
        print("\n[2/5] Skipping lyrics extraction")

    # Step 3: Generate harmony notes
    print("\n[3/5] Generating harmony notes...")
    harmony_voices = generate_harmony_voices(
        pitch_events,
        num_harmonies=num_harmonies,
        voicing_type=voicing_type,
        key=key
    )

    # Step 4: Create harmony audio by pitch-shifting
    print("\n[4/5] Creating harmony audio...")
    harmony_audio_paths = create_harmony_audio(
        input_audio,
        pitch_events,
        harmony_voices,
        output_dir
    )

    # Step 5: Export MIDI
    print("\n[5/5] Exporting MIDI...")
    midi_path = os.path.join(output_dir, "harmonies.mid")
    export_harmony_midi(pitch_events, harmony_voices, midi_path, tempo)

    # Prepare result
    result = {
        "success": True,
        "input_audio": input_audio,
        "output_dir": output_dir,
        "num_harmonies": num_harmonies,
        "voicing_type": voicing_type,
        "key": key,
        "lyrics": lyrics,
        "harmony_audio_paths": harmony_audio_paths,
        "midi_path": midi_path,
        "num_pitch_events": len(pitch_events),
        "word_count": len(word_timings)
    }

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*80}")
    print(f"HARMONIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Harmony audio files: {len(harmony_audio_paths)}")
    print(f"  MIDI file: {midi_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"{'='*80}\n")

    return result


# ============================================================================
# ACE-STEP INTEGRATION
# ============================================================================

def render_harmony_with_ace_step(
    harmony_midi: str,
    harmony_audio: str,
    output_path: str,
    lyrics: str = "",
    noise_level: float = 0.8,
    prompt: str = "harmonized vocals, choir, backing vocals"
) -> str:
    """
    Render harmony through ACE-Step for more natural vocal generation.

    This uses the pitch-shifted audio as a reference and ACE-Step
    to regenerate it with more natural vocal timbre.

    Args:
        harmony_midi: Path to harmony MIDI file
        harmony_audio: Path to pitch-shifted harmony audio
        output_path: Output path for ACE-Step rendered audio
        lyrics: Lyrics text
        noise_level: Noise level (0.0=pure input, 1.0=pure generation)
        prompt: ACE-Step prompt

    Returns:
        Path to output audio
    """
    import subprocess

    print(f"\nRendering through ACE-Step...")
    print(f"  Input: {harmony_audio}")
    print(f"  Noise level: {noise_level}")
    print(f"  Prompt: {prompt}")

    # Call ACE-Step wrapper
    cmd = [
        'python3',
        '/home/arlo/Data/ace_step_noise_wrapper.py',
        '--prompt', prompt,
        '--ref-audio', harmony_audio,
        '--noise-level', str(noise_level),
        '--output', output_path,
        '--steps', '60',
        '--duration', '30'
    ]

    if lyrics:
        cmd.extend(['--lyrics', lyrics])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode == 0:
        print(f"  ACE-Step render complete: {output_path}")
        return output_path
    else:
        print(f"  ACE-Step failed: {result.stderr}")
        raise RuntimeError(f"ACE-Step render failed: {result.stderr}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Vocal Harmonizer - Generate harmonies from vocals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 2-voice harmony
  python vocal_harmonizer.py --input vocals.wav --output ./harmonized

  # 4-voice gospel style
  python vocal_harmonizer.py --input vocals.wav --output ./harmonized \\
      --num-harmonies 4 --voicing gospel --key Am

  # With ACE-Step rendering
  python vocal_harmonizer.py --input vocals.wav --output ./harmonized \\
      --num-harmonies 2 --use-ace-step --noise-level 0.6
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Input vocal audio file')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--num-harmonies', '-n', type=int, default=2,
                       choices=[1, 2, 3, 4], help='Number of harmony voices (default: 2)')
    parser.add_argument('--voicing', '-v', default='close',
                       choices=list(HARMONY_PRESETS.keys()),
                       help='Voicing style (default: close)')
    parser.add_argument('--key', '-k', default='C', help='Musical key (default: C)')
    parser.add_argument('--noise-level', type=float, default=0.8,
                       help='Noise level for ACE-Step (0.0-1.0, default: 0.8)')
    parser.add_argument('--use-ace-step', action='store_true',
                       help='Render through ACE-Step for natural vocals')
    parser.add_argument('--no-lyrics', action='store_true',
                       help='Skip lyrics extraction')
    parser.add_argument('--tempo', type=int, default=120,
                       help='Tempo in BPM for MIDI export (default: 120)')

    args = parser.parse_args()

    # Run harmonizer
    result = harmonize_vocals(
        input_audio=args.input,
        output_dir=args.output,
        num_harmonies=args.num_harmonies,
        voicing_type=args.voicing,
        key=args.key,
        noise_level=args.noise_level,
        use_ace_step=args.use_ace_step,
        extract_lyrics=not args.no_lyrics,
        tempo=args.tempo
    )

    # Optionally render through ACE-Step
    if args.use_ace_step and result["harmony_audio_paths"]:
        ace_step_outputs = []
        for i, harmony_path in enumerate(result["harmony_audio_paths"]):
            ace_output = os.path.join(
                args.output,
                f"ace_step_harmony_{i+1}.wav"
            )
            try:
                render_harmony_with_ace_step(
                    result["midi_path"],
                    harmony_path,
                    ace_output,
                    lyrics=result.get("lyrics", ""),
                    noise_level=args.noise_level
                )
                ace_step_outputs.append(ace_output)
            except Exception as e:
                print(f"  Warning: ACE-Step render failed for voice {i+1}: {e}")

        result["ace_step_outputs"] = ace_step_outputs

    print(f"\nDone! Output saved to: {args.output}")
