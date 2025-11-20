#!/usr/bin/env python3
"""
Big Band Arrangement Generator
================================

Generates a complete big band arrangement with:
- Rhythm Section: Piano, Bass, Drums
- Sax Section: 2 Alto, 2 Tenor, 1 Baritone (5 saxes)
- Brass Section: 4 Trumpets, 4 Trombones

Based on swing era arranging principles (Duke Ellington, Count Basie)

Usage:
    python generate_big_band.py [output_file.mid] [tempo] [key]

Examples:
    python generate_big_band.py big_band.mid 140 0    # C major, 140 BPM
    python generate_big_band.py swing.mid 120 -3      # Eb major, 120 BPM
    python generate_big_band.py fast_swing.mid 180 2  # D major, 180 BPM
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import random
from typing import List, Dict, Tuple
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

# Import jazz generators
from genres.jazz import (
    JazzGenerator, JazzStyle, JazzForm, SwingFeel,
    JazzNote, JazzChord, JazzProgressions,
    BebopMelodyGenerator, JazzWalkingBass, PianoComping, CompingStyle
)


# =============================================================================
# MIDI INSTRUMENTS (General MIDI Programs)
# =============================================================================

INSTRUMENTS = {
    # Saxophones
    'alto_sax': 65,
    'tenor_sax': 66,
    'baritone_sax': 67,

    # Brass
    'trumpet': 56,
    'trombone': 57,

    # Rhythm section
    'piano': 0,
    'acoustic_bass': 32,
    'drums': 0,  # Channel 9
}


# =============================================================================
# BIG BAND ARRANGEMENT GENERATOR
# =============================================================================

class BigBandGenerator:
    """Complete big band arrangement generator."""

    def __init__(self, tempo: int = 140, key: int = 0):
        """
        Initialize big band generator.

        Args:
            tempo: Tempo in BPM (120-200 typical for swing)
            key: Key as MIDI pitch class (0=C, 1=C#, 2=D, etc.)
        """
        self.tempo = tempo
        self.key = key
        self.ticks_per_beat = 480

        # Initialize jazz components
        self.jazz_gen = JazzGenerator(
            style=JazzStyle.SWING,
            tempo=tempo,
            key=key,
            swing_feel=SwingFeel.MEDIUM_SWING
        )

        self.melody_gen = BebopMelodyGenerator()
        self.bass_gen = JazzWalkingBass(JazzStyle.SWING)
        self.piano_comp = PianoComping(CompingStyle.ROOTLESS)

    def generate_arrangement(self, form: JazzForm = JazzForm.BLUES_12,
                           num_choruses: int = 2) -> Dict[str, List[JazzNote]]:
        """
        Generate complete big band arrangement.

        Args:
            form: Jazz form (BLUES_12, AABA_32, etc.)
            num_choruses: Number of times through the form

        Returns:
            Dictionary mapping instrument/section to list of JazzNote objects
        """
        print(f"\n🎺 Generating Big Band Arrangement...")
        print(f"   Tempo: {self.tempo} BPM")
        print(f"   Key: {self._get_key_name()}")
        print(f"   Form: {form.value}")
        print(f"   Choruses: {num_choruses}")

        # Generate chord progression
        if form == JazzForm.BLUES_12:
            progression = JazzProgressions.jazz_blues(self.key)
        elif form == JazzForm.RHYTHM_CHANGES:
            progression = JazzProgressions.rhythm_changes_A(self.key)
        else:
            # Default to extended ii-V-I
            progression = JazzProgressions.ii_V_I(self.key) * 8

        # Repeat for multiple choruses
        full_progression = progression * num_choruses

        print(f"   Chord changes: {len(full_progression)} chords")

        # Generate all sections
        arrangement = {}

        # 1. MELODY (Lead Alto Sax)
        print("\n   🎷 Generating melody (lead alto sax)...")
        arrangement['melody'] = self._generate_melody(full_progression)

        # 2. SAX SECTION (5 saxes harmonizing melody)
        print("   🎷 Harmonizing sax section (5 parts)...")
        arrangement['sax_section'] = self._harmonize_sax_section(
            arrangement['melody'], full_progression
        )

        # 3. BRASS SECTION (4 trumpets, 4 trombones)
        print("   🎺 Creating brass backgrounds...")
        arrangement['brass_section'] = self._generate_brass_section(full_progression)

        # 4. RHYTHM SECTION
        print("   🎹 Generating rhythm section...")
        arrangement['piano'] = self._generate_piano(full_progression)
        arrangement['bass'] = self._generate_bass(full_progression)
        arrangement['drums'] = self._generate_drums(full_progression)

        print(f"\n✅ Arrangement complete!")
        return arrangement

    def _generate_melody(self, progression: List[JazzChord]) -> List[JazzNote]:
        """Generate bebop melody over chord progression."""
        melody = []
        current_beat = 0.0

        for chord in progression:
            phrase = self.melody_gen.generate_phrase(
                chord,
                length_beats=4,
                density=0.7
            )
            # Adjust timing
            for note in phrase:
                note.start_time += current_beat
            melody.extend(phrase)
            current_beat += 4.0

        return melody

    def _harmonize_sax_section(self, melody: List[JazzNote],
                               progression: List[JazzChord]) -> Dict[str, List[JazzNote]]:
        """
        Harmonize melody for 5-part sax section (drop 2 voicing).

        Sax section:
        - Alto 1 (lead)
        - Alto 2
        - Tenor 1
        - Tenor 2
        - Baritone
        """
        sax_parts = {
            'alto_1': [],
            'alto_2': [],
            'tenor_1': [],
            'tenor_2': [],
            'baritone': []
        }

        for note in melody:
            # Find current chord
            chord = self._find_chord_at_time(note.start_time, progression)
            if not chord:
                continue

            # Create 5-part close voicing (drop 2)
            lead_pitch = note.pitch

            # Sax voicing from top to bottom
            alto_1 = lead_pitch  # Lead melody
            alto_2 = lead_pitch - 3  # Minor/major 3rd below
            tenor_1 = lead_pitch - 7  # 5th below
            tenor_2 = lead_pitch - 10  # Minor 7th below
            baritone = lead_pitch - 12  # Octave below

            # Adjust to comfortable ranges
            voicing = {
                'alto_1': self._constrain_pitch(alto_1, 64, 81),  # E4-A5
                'alto_2': self._constrain_pitch(alto_2, 64, 81),
                'tenor_1': self._constrain_pitch(tenor_1, 55, 76),  # G3-E5
                'tenor_2': self._constrain_pitch(tenor_2, 55, 76),
                'baritone': self._constrain_pitch(baritone, 48, 67)  # C3-G4
            }

            # Create notes for each sax
            for sax_name, pitch in voicing.items():
                sax_note = JazzNote(
                    pitch=pitch,
                    velocity=int(note.velocity * 0.88),  # Slightly softer than lead
                    start_time=note.start_time,
                    duration=note.duration,
                    articulation=note.articulation,
                    channel=0
                )
                sax_parts[sax_name].append(sax_note)

        return sax_parts

    def _generate_brass_section(self, progression: List[JazzChord]) -> Dict[str, List[JazzNote]]:
        """
        Generate brass section backgrounds (punches and sustained pads).

        Brass section:
        - 4 Trumpets (high voicings)
        - 4 Trombones (low voicings)
        """
        brass_parts = {
            'trumpet_1': [],
            'trumpet_2': [],
            'trumpet_3': [],
            'trumpet_4': [],
            'trombone_1': [],
            'trombone_2': [],
            'trombone_3': [],
            'trombone_4': []
        }

        current_beat = 0.0

        for i, chord in enumerate(progression):
            # Brass punches on chord changes (not every chord)
            if i % 2 == 1:  # Every other chord
                # Create brass stab
                start_time = current_beat + 3.0  # On beat 4
                duration = 0.5

                # 4-part brass voicing
                root = chord.root + 60  # Middle C range

                # Trumpet voicing (top 4 notes)
                trumpet_voicing = [
                    root + 12,  # Trumpet 1 (top)
                    root + 7,   # Trumpet 2
                    root + 4,   # Trumpet 3
                    root,       # Trumpet 4
                ]

                # Trombone voicing (bottom 4 notes)
                trombone_voicing = [
                    root - 5,   # Trombone 1
                    root - 8,   # Trombone 2
                    root - 12,  # Trombone 3
                    root - 17   # Trombone 4 (bass trombone)
                ]

                # Add trumpet notes
                for idx, pitch in enumerate(trumpet_voicing):
                    brass_parts[f'trumpet_{idx+1}'].append(JazzNote(
                        pitch=self._constrain_pitch(pitch, 60, 82),  # C4-A#5
                        velocity=100,
                        start_time=start_time,
                        duration=duration,
                        articulation='accent'
                    ))

                # Add trombone notes
                for idx, pitch in enumerate(trombone_voicing):
                    brass_parts[f'trombone_{idx+1}'].append(JazzNote(
                        pitch=self._constrain_pitch(pitch, 40, 72),  # E2-C5
                        velocity=95,
                        start_time=start_time,
                        duration=duration,
                        articulation='accent'
                    ))

            current_beat += 4.0

        return brass_parts

    def _generate_piano(self, progression: List[JazzChord]) -> List[JazzNote]:
        """Generate piano comping (syncopated chords)."""
        piano_notes = []
        current_beat = 0.0

        for chord in progression:
            # Comp on beats 2.5 and 4.5 (syncopated)
            comp_times = [current_beat + 1.5, current_beat + 3.5]

            for comp_time in comp_times:
                # Voice chord (rootless voicing)
                voicing = self.piano_comp.voice_chord(chord, octave=4)

                for pitch in voicing:
                    piano_notes.append(JazzNote(
                        pitch=pitch,
                        velocity=random.randint(70, 85),
                        start_time=comp_time,
                        duration=0.4,
                        articulation='normal'
                    ))

            current_beat += 4.0

        return piano_notes

    def _generate_bass(self, progression: List[JazzChord]) -> List[JazzNote]:
        """Generate walking bass line."""
        return self.bass_gen.generate_line(progression, beats_per_chord=4, style="swing")

    def _generate_drums(self, progression: List[JazzChord]) -> List[JazzNote]:
        """Generate swing drum pattern."""
        drums = []
        total_beats = len(progression) * 4

        for beat in range(total_beats):
            beat_time = float(beat)

            # Ride cymbal (swing 8ths)
            # Upbeats are delayed for swing feel (66.7% of beat)
            for eighth_note in [0.0, 0.667]:
                drums.append(JazzNote(
                    pitch=51,  # Ride cymbal (GM)
                    velocity=65 if eighth_note == 0 else 55,  # Downbeat louder
                    start_time=beat_time + eighth_note,
                    duration=0.1,
                    channel=9  # Drum channel
                ))

            # Hi-hat on beats 2 and 4
            if beat % 4 in [1, 3]:  # Beats 2 and 4
                drums.append(JazzNote(
                    pitch=42,  # Closed hi-hat
                    velocity=95,
                    start_time=beat_time,
                    duration=0.15,
                    channel=9
                ))

            # Kick drum (sparse, on 1 and occasional syncopations)
            if beat % 4 == 0:  # Beat 1
                drums.append(JazzNote(
                    pitch=36,  # Bass drum
                    velocity=80,
                    start_time=beat_time,
                    duration=0.1,
                    channel=9
                ))
            elif beat % 8 == 6 and random.random() < 0.4:  # Occasional syncopation
                drums.append(JazzNote(
                    pitch=36,
                    velocity=70,
                    start_time=beat_time,
                    duration=0.1,
                    channel=9
                ))

        return drums

    def _find_chord_at_time(self, time: float, progression: List[JazzChord]) -> JazzChord:
        """Find the chord at a given time."""
        beat = int(time // 4)
        if beat < len(progression):
            return progression[beat]
        return progression[-1] if progression else None

    def _constrain_pitch(self, pitch: int, min_pitch: int, max_pitch: int) -> int:
        """Constrain pitch to instrument range."""
        while pitch < min_pitch:
            pitch += 12
        while pitch > max_pitch:
            pitch -= 12
        return pitch

    def _get_key_name(self) -> str:
        """Get key name from pitch class."""
        key_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        return key_names[self.key % 12]

    def export_midi(self, arrangement: Dict, output_file: str):
        """Export arrangement to multi-track MIDI file."""
        print(f"\n💾 Exporting to MIDI: {output_file}")

        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)

        # Calculate tempo in microseconds per beat
        tempo_microseconds = int(60_000_000 / self.tempo)

        # Track configuration: (track_name, instrument_program, notes, channel)
        tracks_config = [
            # Melody
            ("Lead Alto Sax", INSTRUMENTS['alto_sax'], arrangement.get('melody', []), 0),

            # Sax section
            ("Alto Sax 1", INSTRUMENTS['alto_sax'],
             arrangement.get('sax_section', {}).get('alto_1', []), 1),
            ("Alto Sax 2", INSTRUMENTS['alto_sax'],
             arrangement.get('sax_section', {}).get('alto_2', []), 2),
            ("Tenor Sax 1", INSTRUMENTS['tenor_sax'],
             arrangement.get('sax_section', {}).get('tenor_1', []), 3),
            ("Tenor Sax 2", INSTRUMENTS['tenor_sax'],
             arrangement.get('sax_section', {}).get('tenor_2', []), 4),
            ("Baritone Sax", INSTRUMENTS['baritone_sax'],
             arrangement.get('sax_section', {}).get('baritone', []), 5),

            # Brass section
            ("Trumpet 1", INSTRUMENTS['trumpet'],
             arrangement.get('brass_section', {}).get('trumpet_1', []), 6),
            ("Trumpet 2", INSTRUMENTS['trumpet'],
             arrangement.get('brass_section', {}).get('trumpet_2', []), 7),
            ("Trumpet 3", INSTRUMENTS['trumpet'],
             arrangement.get('brass_section', {}).get('trumpet_3', []), 8),
            ("Trumpet 4", INSTRUMENTS['trumpet'],
             arrangement.get('brass_section', {}).get('trumpet_4', []), 9),
            ("Trombone 1", INSTRUMENTS['trombone'],
             arrangement.get('brass_section', {}).get('trombone_1', []), 10),
            ("Trombone 2", INSTRUMENTS['trombone'],
             arrangement.get('brass_section', {}).get('trombone_2', []), 11),
            ("Trombone 3", INSTRUMENTS['trombone'],
             arrangement.get('brass_section', {}).get('trombone_3', []), 12),
            ("Trombone 4", INSTRUMENTS['trombone'],
             arrangement.get('brass_section', {}).get('trombone_4', []), 13),

            # Rhythm section
            ("Piano", INSTRUMENTS['piano'], arrangement.get('piano', []), 14),
            ("Bass", INSTRUMENTS['acoustic_bass'], arrangement.get('bass', []), 15),
            ("Drums", INSTRUMENTS['drums'], arrangement.get('drums', []), 9),  # Channel 9
        ]

        for track_name, program, notes, channel in tracks_config:
            track = MidiTrack()
            mid.tracks.append(track)

            # Track metadata
            track.append(MetaMessage('track_name', name=track_name, time=0))
            track.append(MetaMessage('set_tempo', tempo=tempo_microseconds, time=0))

            # Set instrument (except drums)
            if channel != 9:
                track.append(Message('program_change', program=program, channel=channel, time=0))

            # Convert notes to MIDI events
            events = []
            for note in notes:
                note_channel = note.channel if hasattr(note, 'channel') and note.channel is not None else channel

                events.append({
                    'type': 'note_on',
                    'time': note.start_time,
                    'note': note.pitch,
                    'velocity': note.velocity,
                    'channel': note_channel
                })
                events.append({
                    'type': 'note_off',
                    'time': note.start_time + note.duration,
                    'note': note.pitch,
                    'channel': note_channel
                })

            # Sort by time
            events.sort(key=lambda e: e['time'])

            # Convert to delta times and add to track
            last_time = 0.0
            for event in events:
                delta_beats = event['time'] - last_time
                delta_ticks = int(delta_beats * self.ticks_per_beat)

                if event['type'] == 'note_on':
                    track.append(Message('note_on',
                                       note=event['note'],
                                       velocity=event['velocity'],
                                       channel=event['channel'],
                                       time=delta_ticks))
                else:
                    track.append(Message('note_off',
                                       note=event['note'],
                                       velocity=0,
                                       channel=event['channel'],
                                       time=delta_ticks))

                last_time = event['time']

            # End of track
            track.append(MetaMessage('end_of_track', time=0))

            print(f"   ✓ {track_name}: {len(notes)} notes")

        # Save file
        mid.save(output_file)
        print(f"\n✅ MIDI file saved: {output_file}")
        print(f"   Tracks: {len(mid.tracks)}")
        print(f"   Duration: ~{len(arrangement.get('melody', [])) // 4 * 4} beats")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 70)
    print("BIG BAND ARRANGEMENT GENERATOR")
    print("=" * 70)
    print("Duke Ellington / Count Basie Style Swing Arrangement")
    print()

    # Parse arguments
    output_file = sys.argv[1] if len(sys.argv) > 1 else "big_band_arrangement.mid"
    tempo = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    key = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # C major

    # Create generator
    generator = BigBandGenerator(tempo=tempo, key=key)

    # Generate arrangement
    arrangement = generator.generate_arrangement(
        form=JazzForm.BLUES_12,
        num_choruses=2
    )

    # Export to MIDI
    generator.export_midi(arrangement, output_file)

    print("\n" + "=" * 70)
    print("🎵 BIG BAND ARRANGEMENT COMPLETE!")
    print("=" * 70)
    print(f"\nOutput: {output_file}")
    print("\nInstrumentation:")
    print("  Saxophones:")
    print("    • Lead Alto Sax (melody)")
    print("    • 2 Alto Saxes")
    print("    • 2 Tenor Saxes")
    print("    • Baritone Sax")
    print("  Brass:")
    print("    • 4 Trumpets")
    print("    • 4 Trombones")
    print("  Rhythm Section:")
    print("    • Piano (comping)")
    print("    • Acoustic Bass (walking)")
    print("    • Drums (swing pattern)")
    print("\nTotal: 17 tracks")
    print("\nYou can now:")
    print("  • Open in any DAW (Logic, Ableton, FL Studio, GarageBand)")
    print("  • Import into MuseScore, Sibelius, or Finale")
    print("  • Adjust individual instrument levels and panning")
    print("  • Add effects and humanization")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
