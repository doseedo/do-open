#!/usr/bin/env python3
"""
Arrangement Engine - Auto-Arrange Lead Sheets to Full Arrangements

This module transforms simple lead sheets (melody + chords) into complete
musical arrangements for various ensembles.

Supported Arrangement Styles:
1. BIG BAND - Full jazz big band (saxes, trumpets, trombones, rhythm)
2. STRING QUARTET - Classical chamber music (2 violins, viola, cello)
3. SOLO PIANO - Complete piano arrangement with bass, chords, melody
4. POP BAND - Drums, bass, guitar, keys, vocals
5. JAZZ COMBO - Piano, bass, drums, horn
6. ORCHESTRA - Full symphonic orchestration

Arranging Principles:
- Voice leading optimization
- Register distribution
- Instrumental idioms
- Texture variation
- Dynamic balance

Based on principles from:
- Rimsky-Korsakov: Principles of Orchestration
- Walter Piston: Orchestration
- Duke Ellington: Big band arranging style
- George Russell: Jazz arranging concepts

Enhanced by Agent 5 - Brass Section Arranger:
- Sophisticated brass writing (sustained pads, riffs, shout chorus)
- Call-and-response
- Multiple voicing types
- Dynamic shaping

Author: Agent 8 - Style Transfer & Transformation
Updated: Agent 5 - Brass Section Arranger (2025-01-20)
"""

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import copy

# Import our analyzer and voicing engines
import sys
sys.path.append(str(Path(__file__).parent.parent))
from analysis.midi_analyzer import MidiAnalyzer, NoteEvent, ChordEvent
from transformation.sax_voicing import SaxSoliVoicing, voice_sax_soli

# Import Agent 5's brass arranger
try:
    from transformation.brass_arranger import BrassArranger, ShoutChorusDetector
    BRASS_ARRANGER_AVAILABLE = True
except ImportError:
    BRASS_ARRANGER_AVAILABLE = False
    print("Warning: BrassArranger not available, using legacy brass implementation")


# ==============================================================================
# INSTRUMENT DEFINITIONS
# ==============================================================================

@dataclass
class Instrument:
    """Instrument specification."""
    name: str
    midi_program: int
    range_low: int       # Lowest MIDI note
    range_high: int      # Highest MIDI note
    comfortable_low: int # Comfortable low range
    comfortable_high: int # Comfortable high range
    role: str           # 'melody', 'harmony', 'bass', 'rhythm'

# Standard orchestral instruments
INSTRUMENTS = {
    # Strings
    'violin': Instrument('Violin', 40, 55, 103, 64, 96, 'melody'),
    'viola': Instrument('Viola', 41, 48, 91, 55, 84, 'harmony'),
    'cello': Instrument('Cello', 42, 36, 84, 48, 72, 'harmony'),
    'bass': Instrument('Double Bass', 43, 28, 67, 36, 55, 'bass'),

    # Piano
    'piano': Instrument('Acoustic Piano', 0, 21, 108, 28, 103, 'melody'),

    # Jazz band
    'sax_alto': Instrument('Alto Sax', 65, 49, 81, 55, 76, 'melody'),
    'sax_tenor': Instrument('Tenor Sax', 66, 44, 76, 50, 70, 'melody'),
    'trumpet': Instrument('Trumpet', 56, 55, 82, 60, 77, 'melody'),
    'trombone': Instrument('Trombone', 57, 40, 72, 46, 67, 'harmony'),

    # Rhythm section
    'acoustic_guitar': Instrument('Acoustic Guitar', 24, 40, 88, 48, 76, 'rhythm'),
    'electric_guitar': Instrument('Electric Guitar', 27, 40, 88, 48, 76, 'rhythm'),
    'electric_bass': Instrument('Electric Bass', 33, 28, 55, 31, 52, 'bass'),
    'drums': Instrument('Drums', 0, 35, 81, 35, 81, 'rhythm'),  # Channel 9
}


# ==============================================================================
# ARRANGEMENT TEMPLATES
# ==============================================================================

class BigBandArranger:
    """
    Arrange for big band (4 trumpets, 4 trombones, 5 saxes, rhythm section).

    Follows Duke Ellington and Count Basie arranging principles.
    """

    @staticmethod
    def arrange(melody: List[NoteEvent],
                chords: List[ChordEvent],
                brass_style: str = "riff",
                brass_pattern: str = "basie_riff") -> Dict[str, List[NoteEvent]]:
        """
        Create big band arrangement.

        Enhanced by Agent 5 to support sophisticated brass writing.

        Args:
            melody: Lead melody notes
            chords: Chord progression
            brass_style: "riff", "pad", "stabs" (default: "riff")
            brass_pattern: "basie_riff", "ellington_call", "thad_modern"

        Returns:
            Dictionary mapping instrument names to note lists
        """
        arrangement = {}

        # Lead melody (alto sax or trumpet)
        arrangement['lead'] = BigBandArranger._create_lead(melody)

        # Sax section (harmonize melody in 5-part close voicing)
        arrangement['saxes'] = BigBandArranger._harmonize_saxes(melody, chords)

        # Brass section (enhanced by Agent 5)
        arrangement['brass'] = BigBandArranger._create_brass_figures(
            chords,
            style=brass_style,
            brass_style=brass_pattern
        )

        # Piano comping
        arrangement['piano'] = BigBandArranger._create_piano_comping(chords)

        # Bass walking
        arrangement['bass'] = BigBandArranger._create_walking_bass(chords)

        # Drums (swing pattern)
        arrangement['drums'] = BigBandArranger._create_swing_drums(melody)

        return arrangement

    @staticmethod
    def arrange_with_shout_chorus(melody: List[NoteEvent],
                                  chords: List[ChordEvent],
                                  shout_start_bar: int = 24,
                                  shout_style: str = "basie_unison") -> Dict[str, List[NoteEvent]]:
        """
        Create big band arrangement with shout chorus.

        Enhanced by Agent 5 for authentic big band climactic sections.

        Args:
            melody: Lead melody notes
            chords: Chord progression
            shout_start_bar: Bar number where shout chorus starts (default: 24 for final A in AABA)
            shout_style: "basie_unison", "ellington_harmony", "thad_modern"

        Returns:
            Dictionary mapping instrument names to note lists
        """
        if not BRASS_ARRANGER_AVAILABLE:
            # Fall back to standard arrangement
            return BigBandArranger.arrange(melody, chords)

        arrangement = {}

        # Lead melody
        arrangement['lead'] = BigBandArranger._create_lead(melody)

        # Sax section
        arrangement['saxes'] = BigBandArranger._harmonize_saxes(melody, chords)

        # Separate melody into regular and shout chorus sections
        shout_start_time = shout_start_bar * 4.0  # Assuming 4/4 time
        regular_melody = [n for n in melody if n.start_time < shout_start_time]
        shout_melody = [n for n in melody if n.start_time >= shout_start_time]

        # Regular brass (before shout chorus)
        if regular_melody:
            regular_chords = [c for c in chords if c.start_time < shout_start_time]
            arrangement['brass_regular'] = BigBandArranger._create_brass_figures(
                regular_chords,
                style="riff",
                brass_style="basie_riff"
            )

        # Shout chorus brass
        if shout_melody:
            shout_chords = [c for c in chords if c.start_time >= shout_start_time]
            brass_parts = BrassArranger.create_shout_chorus(
                shout_melody,
                shout_chords,
                intensity=0.9,
                style=shout_style
            )
            # Flatten brass parts into single list
            arrangement['brass_shout'] = []
            for instrument, notes in brass_parts.items():
                arrangement['brass_shout'].extend(notes)

        # Combine regular and shout brass
        arrangement['brass'] = arrangement.get('brass_regular', []) + arrangement.get('brass_shout', [])

        # Piano comping
        arrangement['piano'] = BigBandArranger._create_piano_comping(chords)

        # Bass walking
        arrangement['bass'] = BigBandArranger._create_walking_bass(chords)

        # Drums (swing pattern)
        arrangement['drums'] = BigBandArranger._create_swing_drums(melody)

        return arrangement

    @staticmethod
    def _create_lead(melody: List[NoteEvent]) -> List[NoteEvent]:
        """Create lead melody line."""
        lead = []
        for note in melody:
            new_note = copy.copy(note)
            # Transpose to comfortable range for alto sax
            while new_note.pitch < 60:
                new_note.pitch += 12
            while new_note.pitch > 80:
                new_note.pitch -= 12
            lead.append(new_note)
        return lead

    @staticmethod
    def _harmonize_saxes(melody: List[NoteEvent],
                        chords: List[ChordEvent],
                        voicing_style: str = "drop_2") -> List[NoteEvent]:
        """
        Create professional 5-part sax soli with drop voicings and voice leading optimization.

        NOW USES PROFESSIONAL SAX VOICING ENGINE (Agent 2)
        - Drop-2, drop-3, drop-2-4, spread voicings (not just close!)
        - Voice leading optimization (minimizes voice movement)
        - Register-specific spacing (wider in bass, closer in treble)

        Args:
            melody: Lead melody notes
            chords: Chord progression
            voicing_style: "drop_2" (default), "drop_3", "close", "spread", "drop_2_4"

        Returns:
            List of NoteEvent objects for all 5 sax voices
        """
        # Use the professional sax voicing engine
        sax_parts = voice_sax_soli(
            melody=melody,
            chords=chords,
            style=voicing_style
        )

        # Convert dictionary of parts to flat list of NoteEvents
        sax_notes = []
        track_mapping = {
            'bari': 1,
            'tenor2': 2,
            'tenor1': 3,
            'alto2': 4,
            'alto1': 5
        }

        for voice_name, notes in sax_parts.items():
            track_idx = track_mapping.get(voice_name, 0)
            for note in notes:
                note.track_idx = track_idx
            sax_notes.extend(notes)

        return sax_notes

    @staticmethod
    def _create_brass_figures(chords: List[ChordEvent],
                             style: str = "riff",
                             brass_style: str = "basie_riff") -> List[NoteEvent]:
        """
        Create brass section background figures.

        Enhanced by Agent 5 to support multiple brass writing styles:
        - Sustained pads
        - Brass riffs
        - Stabs (legacy)

        Args:
            chords: Chord progression
            style: "riff", "pad", "stabs" (default: "riff")
            brass_style: "basie_riff", "ellington_call", "thad_modern"

        Returns:
            List of NoteEvent objects for brass section
        """
        if BRASS_ARRANGER_AVAILABLE and style != "stabs":
            # Use Agent 5's enhanced brass arranger
            if style == "pad":
                return BrassArranger.create_sustained_pad(
                    chords,
                    voicing_type="drop_2",
                    dynamic_shape="arch",
                    base_velocity=75
                )
            elif style == "riff":
                # Create riff for first chord (can be extended to all chords)
                if chords:
                    return BrassArranger.create_brass_riff(
                        chords[0],
                        pattern_style=brass_style,
                        bars=min(4, len(chords)),
                        base_velocity=95
                    )
                return []

        # Legacy implementation (stabs)
        brass = []
        for chord in chords:
            # Create brass stab on chord changes
            # Use 4-part voicing (trumpets on top, trombones below)
            voicing = BigBandArranger._create_close_voicing(
                chord.root + 60, chord, 4
            )

            for i, pitch in enumerate(voicing):
                stab = NoteEvent(
                    start_time=chord.start_time,
                    duration=0.25,  # Short stab
                    start_tick=int(chord.start_time * 480),
                    duration_ticks=int(0.25 * 480),
                    pitch=pitch,
                    velocity=100,  # Accent
                    channel=i + 5,
                    track_idx=i + 10
                )
                brass.append(stab)

        return brass

    @staticmethod
    def _create_piano_comping(chords: List[ChordEvent]) -> List[NoteEvent]:
        """Create jazz piano comping (syncopated chords)."""
        piano = []

        for chord in chords:
            # Syncopated rhythm (offbeats)
            offbeat_times = [
                chord.start_time + 0.25,
                chord.start_time + 0.75,
            ]

            for time in offbeat_times:
                if time < chord.start_time + chord.duration:
                    # Rootless voicing (3rd, 7th, 9th, 13th)
                    voicing = BigBandArranger._create_jazz_voicing(chord)

                    for pitch in voicing:
                        comp = NoteEvent(
                            start_time=time,
                            duration=0.2,
                            start_tick=int(time * 480),
                            duration_ticks=int(0.2 * 480),
                            pitch=pitch,
                            velocity=75,
                            channel=0,
                            track_idx=20
                        )
                        piano.append(comp)

        return piano

    @staticmethod
    def _create_walking_bass(chords: List[ChordEvent]) -> List[NoteEvent]:
        """Create walking bass line (quarter notes)."""
        bass = []

        for chord in chords:
            # Walk through chord tones and approach notes
            num_beats = int(chord.duration)
            beat_duration = chord.duration / num_beats if num_beats > 0 else 1.0

            for beat in range(num_beats):
                time = chord.start_time + beat * beat_duration

                # Alternate between root, 5th, and approach notes
                if beat == 0:
                    pitch = chord.root + 36  # Root
                elif beat == 1:
                    pitch = chord.root + 36 + 7  # 5th
                else:
                    pitch = chord.root + 36 + 3  # 3rd or approach

                bass_note = NoteEvent(
                    start_time=time,
                    duration=beat_duration * 0.9,
                    start_tick=int(time * 480),
                    duration_ticks=int(beat_duration * 0.9 * 480),
                    pitch=pitch,
                    velocity=90,
                    channel=1,
                    track_idx=25
                )
                bass.append(bass_note)

        return bass

    @staticmethod
    def _create_swing_drums(melody: List[NoteEvent]) -> List[NoteEvent]:
        """Create swing drum pattern."""
        drums = []

        if not melody:
            return drums

        # Simple swing pattern: ride cymbal, hi-hat on 2&4
        duration = melody[-1].end_time
        beats = int(duration)

        for beat in range(beats):
            time = float(beat)

            # Ride cymbal (swing 8ths)
            for eighth in [0, 0.67]:  # Swing ratio
                ride = NoteEvent(
                    start_time=time + eighth,
                    duration=0.1,
                    start_tick=int((time + eighth) * 480),
                    duration_ticks=int(0.1 * 480),
                    pitch=51,  # Ride cymbal
                    velocity=70,
                    channel=9,  # Drum channel
                    track_idx=30
                )
                drums.append(ride)

            # Hi-hat on 2 and 4
            if beat % 2 == 1:
                hihat = NoteEvent(
                    start_time=time,
                    duration=0.1,
                    start_tick=int(time * 480),
                    duration_ticks=int(0.1 * 480),
                    pitch=42,  # Closed hi-hat
                    velocity=95,
                    channel=9,
                    track_idx=30
                )
                drums.append(hihat)

        return drums

    @staticmethod
    def _find_chord_at_time(time: float, chords: List[ChordEvent]) -> Optional[ChordEvent]:
        """Find chord at specified time."""
        for chord in chords:
            if chord.start_time <= time < chord.start_time + chord.duration:
                return chord
        return None

    @staticmethod
    def _create_close_voicing(melody_pitch: int,
                             chord: ChordEvent,
                             num_voices: int) -> List[int]:
        """Create close voicing below melody pitch."""
        # Build chord tones
        chord_tones = [
            chord.root,
            chord.root + 4 if 'major' in chord.quality else chord.root + 3,
            chord.root + 7,
            chord.root + 10,  # 7th
        ]

        voicing = [melody_pitch]

        # Add voices below in close position
        current_pitch = melody_pitch - 1
        while len(voicing) < num_voices:
            # Find next chord tone below
            for tone in sorted(chord_tones, reverse=True):
                candidate = (current_pitch // 12) * 12 + tone
                if candidate < current_pitch and candidate not in voicing:
                    voicing.append(candidate)
                    current_pitch = candidate
                    break
            else:
                # No chord tone found, use any note below
                voicing.append(current_pitch)
                current_pitch -= 1

        return sorted(voicing)

    @staticmethod
    def _create_jazz_voicing(chord: ChordEvent) -> List[int]:
        """Create rootless jazz piano voicing."""
        root = chord.root + 48  # Mid register

        # Rootless voicing: 3rd, 7th, 9th, 13th
        if 'major' in chord.quality:
            return [root + 4, root + 11, root + 14, root + 21]  # 3, 7, 9, 13
        else:
            return [root + 3, root + 10, root + 14, root + 21]  # m3, m7, 9, 13


class StringQuartetArranger:
    """Arrange for string quartet (2 violins, viola, cello)."""

    @staticmethod
    def arrange(melody: List[NoteEvent],
                chords: List[ChordEvent]) -> Dict[str, List[NoteEvent]]:
        """Create string quartet arrangement."""
        arrangement = {}

        # Violin I: Melody
        arrangement['violin1'] = StringQuartetArranger._adapt_for_violin(melody)

        # Violin II, Viola, Cello: 3-part harmony
        harmony = StringQuartetArranger._create_harmony(melody, chords)
        arrangement['violin2'] = harmony['violin2']
        arrangement['viola'] = harmony['viola']
        arrangement['cello'] = harmony['cello']

        return arrangement

    @staticmethod
    def _adapt_for_violin(melody: List[NoteEvent]) -> List[NoteEvent]:
        """Adapt melody for violin range."""
        violin_melody = []

        for note in melody:
            new_note = copy.copy(note)
            # Violin range: G3 (55) to A7 (105)
            while new_note.pitch < 55:
                new_note.pitch += 12
            while new_note.pitch > 96:  # Comfortable high
                new_note.pitch -= 12
            violin_melody.append(new_note)

        return violin_melody

    @staticmethod
    def _create_harmony(melody: List[NoteEvent],
                       chords: List[ChordEvent]) -> Dict[str, List[NoteEvent]]:
        """Create 3-part harmony for lower strings."""
        harmony = {'violin2': [], 'viola': [], 'cello': []}

        for note in melody:
            # Find chord at this time
            chord = BigBandArranger._find_chord_at_time(note.start_time, chords)

            if chord:
                # Create 4-part voicing (including melody)
                voicing = BigBandArranger._create_close_voicing(note.pitch, chord, 4)

                # Distribute to instruments
                if len(voicing) >= 4:
                    # Violin II: 2nd from top
                    vln2 = copy.copy(note)
                    vln2.pitch = voicing[-2]
                    vln2.velocity = int(note.velocity * 0.9)
                    harmony['violin2'].append(vln2)

                    # Viola: 3rd from top
                    vla = copy.copy(note)
                    vla.pitch = voicing[-3]
                    vla.velocity = int(note.velocity * 0.85)
                    harmony['viola'].append(vla)

                    # Cello: Bass note
                    vc = copy.copy(note)
                    vc.pitch = voicing[0]
                    vc.velocity = int(note.velocity * 0.8)
                    harmony['cello'].append(vc)

        return harmony


class SoloPianoArranger:
    """Arrange for solo piano."""

    @staticmethod
    def arrange(melody: List[NoteEvent],
                chords: List[ChordEvent]) -> Dict[str, List[NoteEvent]]:
        """
        Create solo piano arrangement.

        Right hand: Melody + harmonic fills
        Left hand: Bass + chord voicings
        """
        arrangement = {}

        # Right hand: Melody (top voice)
        arrangement['right_hand_melody'] = melody.copy()

        # Left hand: Bass notes
        arrangement['left_hand_bass'] = SoloPianoArranger._create_bass_line(chords)

        # Left hand: Chord voicings
        arrangement['left_hand_chords'] = SoloPianoArranger._create_chord_voicings(chords)

        return arrangement

    @staticmethod
    def _create_bass_line(chords: List[ChordEvent]) -> List[NoteEvent]:
        """Create bass line (left hand low notes)."""
        bass = []

        for chord in chords:
            # Root note in bass
            bass_note = NoteEvent(
                start_time=chord.start_time,
                duration=chord.duration,
                start_tick=int(chord.start_time * 480),
                duration_ticks=int(chord.duration * 480),
                pitch=chord.root + 36,  # Low bass register
                velocity=80,
                channel=0,
                track_idx=1
            )
            bass.append(bass_note)

        return bass

    @staticmethod
    def _create_chord_voicings(chords: List[ChordEvent]) -> List[NoteEvent]:
        """Create chord voicings for left hand."""
        voicing_notes = []

        for chord in chords:
            # 3-note voicing in mid-low register
            root = chord.root + 48

            if 'major' in chord.quality:
                pitches = [root, root + 4, root + 7]  # Root, 3rd, 5th
            elif 'minor' in chord.quality:
                pitches = [root, root + 3, root + 7]  # Root, m3, 5th
            else:
                pitches = [root, root + 4, root + 7]  # Default

            for pitch in pitches:
                note = NoteEvent(
                    start_time=chord.start_time,
                    duration=chord.duration,
                    start_tick=int(chord.start_time * 480),
                    duration_ticks=int(chord.duration * 480),
                    pitch=pitch,
                    velocity=70,
                    channel=0,
                    track_idx=2
                )
                voicing_notes.append(note)

        return voicing_notes


# ==============================================================================
# MAIN ARRANGEMENT ENGINE
# ==============================================================================

class ArrangementEngine:
    """
    Main arrangement engine.

    Automatically arranges lead sheets into full arrangements.
    """

    ARRANGERS = {
        'big_band': BigBandArranger,
        'string_quartet': StringQuartetArranger,
        'solo_piano': SoloPianoArranger,
    }

    def __init__(self, lead_sheet_midi: str):
        """
        Initialize with lead sheet MIDI.

        Args:
            lead_sheet_midi: Path to lead sheet (melody + chords implied)
        """
        self.input_path = Path(lead_sheet_midi)

        # Analyze input
        print(f"Analyzing lead sheet: {self.input_path.name}")
        self.analyzer = MidiAnalyzer(str(self.input_path))
        self.analysis = self.analyzer.analyze()

        print(f"Analysis: {len(self.analysis.notes)} notes, "
              f"{len(self.analysis.chords)} chords")

    def arrange(self,
                style: str,
                output_path: Optional[str] = None) -> str:
        """
        Create arrangement in specified style.

        Args:
            style: Arrangement style ('big_band', 'string_quartet', 'solo_piano')
            output_path: Output MIDI path

        Returns:
            Path to output MIDI file
        """
        if style not in self.ARRANGERS:
            raise ValueError(f"Unknown style: {style}. "
                           f"Available: {list(self.ARRANGERS.keys())}")

        print(f"\n{'='*80}")
        print(f"ARRANGEMENT: {style.upper()}")
        print(f"{'='*80}\n")

        # Extract melody (highest notes)
        melody = self._extract_melody()

        # Get arranger
        arranger = self.ARRANGERS[style]

        # Create arrangement
        print(f"Creating {style} arrangement...")
        arrangement = arranger.arrange(melody, self.analysis.chords)

        # Write output
        if output_path is None:
            output_path = self.input_path.parent / f"{self.input_path.stem}_{style}.mid"

        self._write_arrangement(arrangement, output_path)

        print(f"\n✅ Arrangement complete!")
        print(f"📁 Output: {output_path}")
        print(f"🎵 Instruments: {', '.join(arrangement.keys())}")
        print(f"{'='*80}\n")

        return str(output_path)

    def _extract_melody(self) -> List[NoteEvent]:
        """Extract melody line (highest notes at each time)."""
        # Simple approach: take highest pitch notes
        melody = []
        seen_times = set()

        sorted_notes = sorted(self.analysis.notes,
                            key=lambda n: (n.start_time, -n.pitch))

        for note in sorted_notes:
            rounded_time = round(note.start_time, 2)
            if rounded_time not in seen_times:
                melody.append(note)
                seen_times.add(rounded_time)

        return sorted(melody, key=lambda n: n.start_time)

    def _write_arrangement(self, arrangement: Dict[str, List[NoteEvent]],
                          output_path: Path):
        """Write arrangement to MIDI file."""
        midi = MidiFile(ticks_per_beat=self.analysis.ticks_per_beat)

        # Create track for each instrument
        for instrument_name, notes in arrangement.items():
            track = MidiTrack()
            midi.tracks.append(track)

            # Track name
            track.append(MetaMessage('track_name', name=instrument_name, time=0))

            # Add tempo (only on first track)
            if len(midi.tracks) == 1:
                if self.analysis.tempo_events:
                    tempo = self.analysis.tempo_events[0].microseconds_per_beat
                else:
                    tempo = 500000  # 120 BPM
                track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

            # Convert notes to messages
            events = []
            for note in notes:
                events.append((note.start_tick, 'note_on', note.pitch,
                             note.velocity, note.channel))
                events.append((note.end_tick, 'note_off', note.pitch,
                             0, note.channel))

            events.sort(key=lambda x: x[0])

            # Write with delta times
            current_time = 0
            for abs_time, msg_type, note, velocity, channel in events:
                delta = max(0, int(abs_time - current_time))
                track.append(Message(msg_type, note=note, velocity=velocity,
                                   channel=channel, time=delta))
                current_time = abs_time

            track.append(MetaMessage('end_of_track', time=0))

        # Save
        midi.save(str(output_path))


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Auto-arrange lead sheets')
    parser.add_argument('input', help='Input lead sheet MIDI file')
    parser.add_argument('style',
                       choices=['big_band', 'string_quartet', 'solo_piano'],
                       help='Arrangement style')
    parser.add_argument('--output', '-o', help='Output MIDI file')

    args = parser.parse_args()

    # Create arrangement
    engine = ArrangementEngine(args.input)
    output = engine.arrange(args.style, args.output)

    print(f"✅ Done! Arrangement saved to: {output}")
