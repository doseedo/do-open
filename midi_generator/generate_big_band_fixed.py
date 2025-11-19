#!/usr/bin/env python3
"""
FIXED Big Band Generator - Addresses All Issues
================================================

FIXES:
1. ✅ Melody with rests and phrasing (not continuous)
2. ✅ Monophonic instruments (one note at a time per track)
3. ✅ Proper brass distribution (trombones get notes!)
4. ✅ Varied brass patterns (not repetitive)
5. ✅ Correct voicing (one note per instrument in a chord)

Usage:
    python generate_big_band_fixed.py [name] [tempo] [key]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import random
from typing import List, Dict, Tuple

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: pip3 install mido")
    sys.exit(1)

try:
    from genres.jazz import (
        JazzGenerator, JazzStyle, JazzForm,
        JazzNote, JazzChord, JazzProgressions,
        BebopMelodyGenerator, JazzWalkingBass, PianoComping, CompingStyle
    )
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


class FixedBigBandGenerator:
    """Big band generator with all issues fixed."""

    def __init__(self, tempo: int = 140, key: int = 0):
        self.tempo = tempo
        self.key = key
        self.ticks_per_beat = 480

        self.melody_gen = BebopMelodyGenerator()
        self.bass_gen = JazzWalkingBass(JazzStyle.SWING)
        self.piano_comp = PianoComping(CompingStyle.ROOTLESS)

    def generate(self) -> Dict:
        """Generate complete big band arrangement with all fixes."""

        print("\n" + "=" * 70)
        print("FIXED BIG BAND GENERATOR")
        print("=" * 70)
        print(f"Tempo: {self.tempo} BPM | Key: {self._get_key_name()}")
        print()

        # Generate progression
        progression = JazzProgressions.jazz_blues(self.key)

        arrangement = {
            'melody': [],
            'sax_section': {'alto1': [], 'alto2': [], 'tenor1': [], 'tenor2': [], 'bari': []},
            'brass_section': {
                'trumpet1': [], 'trumpet2': [], 'trumpet3': [], 'trumpet4': [],
                'trombone1': [], 'trombone2': [], 'trombone3': [], 'trombone4': []
            },
            'piano': [],
            'bass': [],
            'drums': []
        }

        # 1. MELODY WITH RESTS AND PHRASING
        print("✓ Generating melody with phrasing...")
        arrangement['melody'] = self._generate_melody_with_rests(progression)
        print(f"  {len(arrangement['melody'])} notes")

        # 2. SAX HARMONY (only when melody plays)
        print("✓ Harmonizing sax section...")
        for note in arrangement['melody']:
            chord_idx = int(note.start_time // 4)
            if chord_idx < len(progression):
                sax_notes = self._create_sax_voicing(note, progression[chord_idx])
                for sax_name, sax_note in sax_notes.items():
                    arrangement['sax_section'][sax_name].append(sax_note)

        total_sax = sum(len(v) for v in arrangement['sax_section'].values())
        print(f"  {total_sax} notes across 5 saxes")

        # 3. BRASS WITH VARIED PATTERNS (MONOPHONIC)
        print("✓ Generating brass (varied, monophonic)...")
        arrangement['brass_section'] = self._generate_varied_brass(progression)

        total_brass = sum(len(v) for v in arrangement['brass_section'].values())
        print(f"  {total_brass} notes across 8 brass")

        # 4. PIANO COMPING
        print("✓ Generating piano...")
        arrangement['piano'] = self._generate_piano(progression)
        print(f"  {len(arrangement['piano'])} notes")

        # 5. WALKING BASS
        print("✓ Generating bass...")
        arrangement['bass'] = self.bass_gen.generate_line(progression, beats_per_chord=4, style="swing")
        print(f"  {len(arrangement['bass'])} notes")

        # 6. DRUMS
        print("✓ Generating drums...")
        arrangement['drums'] = self._generate_drums(len(progression) * 4)
        print(f"  {len(arrangement['drums'])} hits")

        print()
        print("=" * 70)
        print("✅ FIXED ARRANGEMENT COMPLETE")
        print("=" * 70)

        return arrangement

    def _generate_melody_with_rests(self, progression: List[JazzChord]) -> List[JazzNote]:
        """Generate melody with musical phrasing (rests between phrases)."""
        melody = []
        current_beat = 0.0

        for i, chord in enumerate(progression):
            # Add rest every 2-3 bars (musical phrasing)
            if i > 0 and i % 3 == 0:
                current_beat += 2.0  # 2-beat rest
                continue

            # Generate 4-beat phrase
            phrase = self.melody_gen.generate_phrase(chord, length_beats=4, density=0.6)

            # Adjust timing
            for note in phrase:
                note.start_time += current_beat

            melody.extend(phrase)
            current_beat += 4.0

        return melody

    def _create_sax_voicing(self, lead_note: JazzNote, chord: JazzChord) -> Dict[str, JazzNote]:
        """Create 5-part sax voicing (one note per sax)."""

        lead = lead_note.pitch

        # 5-part close voicing
        voicing = {
            'alto1': self._constrain(lead, 64, 81),
            'alto2': self._constrain(lead - 3, 64, 81),
            'tenor1': self._constrain(lead - 7, 55, 76),
            'tenor2': self._constrain(lead - 10, 55, 76),
            'bari': self._constrain(lead - 12, 48, 67)
        }

        # Create JazzNote for each sax
        sax_notes = {}
        for sax, pitch in voicing.items():
            sax_notes[sax] = JazzNote(
                pitch=pitch,
                velocity=int(lead_note.velocity * 0.88),
                start_time=lead_note.start_time,
                duration=lead_note.duration,
                articulation=lead_note.articulation
            )

        return sax_notes

    def _generate_varied_brass(self, progression: List[JazzChord]) -> Dict[str, List[JazzNote]]:
        """
        Generate VARIED brass patterns that are MONOPHONIC.

        Creates different patterns:
        - Staccato hits
        - Sustained pads
        - Call-response

        Ensures ONE NOTE PER INSTRUMENT AT A TIME.
        """
        brass = {
            'trumpet1': [], 'trumpet2': [], 'trumpet3': [], 'trumpet4': [],
            'trombone1': [], 'trombone2': [], 'trombone3': [], 'trombone4': []
        }

        current_beat = 0.0

        for i, chord in enumerate(progression):
            root = chord.root + 60

            # VARIED PATTERNS
            if i % 4 == 0:
                # Pattern 1: Staccato hits on beat 1
                self._add_brass_hit(brass, root, current_beat, 0.3, 100)

            elif i % 4 == 1:
                # Pattern 2: Sustained pad
                self._add_brass_sustain(brass, root, current_beat + 2.0, 1.5, 85)

            elif i % 4 == 2:
                # Pattern 3: Syncopated hits
                self._add_brass_hit(brass, root, current_beat + 2.5, 0.3, 95)
                self._add_brass_hit(brass, root, current_beat + 3.5, 0.3, 90)

            else:
                # Pattern 4: Call-response
                self._add_trumpet_call(brass, root, current_beat + 1.0, 0.5, 90)
                self._add_trombone_response(brass, root, current_beat + 2.0, 0.5, 85)

            current_beat += 4.0

        return brass

    def _add_brass_hit(self, brass: Dict, root: int, time: float, duration: float, velocity: int):
        """Add a brass hit - ONE NOTE PER INSTRUMENT."""

        # Create 4-part voicing (one note per instrument group)
        voicing = [
            root + 12,  # Trumpet 1 (top)
            root + 7,   # Trumpet 2
            root + 4,   # Trumpet 3
            root,       # Trumpet 4
            root - 5,   # Trombone 1
            root - 8,   # Trombone 2
            root - 12,  # Trombone 3
            root - 17   # Trombone 4 (bass trombone)
        ]

        instruments = ['trumpet1', 'trumpet2', 'trumpet3', 'trumpet4',
                      'trombone1', 'trombone2', 'trombone3', 'trombone4']

        for inst, pitch in zip(instruments, voicing):
            # Constrain to instrument range
            if 'trumpet' in inst:
                pitch = self._constrain(pitch, 55, 82)
            else:
                pitch = self._constrain(pitch, 40, 72)

            brass[inst].append(JazzNote(
                pitch=pitch,
                velocity=velocity,
                start_time=time,
                duration=duration,
                articulation='accent'
            ))

    def _add_brass_sustain(self, brass: Dict, root: int, time: float, duration: float, velocity: int):
        """Add sustained brass pad."""

        # Sustained voicing (fewer notes)
        brass['trumpet1'].append(JazzNote(
            pitch=self._constrain(root + 12, 55, 82),
            velocity=velocity,
            start_time=time,
            duration=duration,
            articulation='legato'
        ))

        brass['trombone3'].append(JazzNote(
            pitch=self._constrain(root - 12, 40, 72),
            velocity=velocity - 5,
            start_time=time,
            duration=duration,
            articulation='legato'
        ))

    def _add_trumpet_call(self, brass: Dict, root: int, time: float, duration: float, velocity: int):
        """Trumpets call."""
        for inst in ['trumpet1', 'trumpet2']:
            brass[inst].append(JazzNote(
                pitch=self._constrain(root + 12, 55, 82),
                velocity=velocity,
                start_time=time,
                duration=duration,
                articulation='staccato'
            ))

    def _add_trombone_response(self, brass: Dict, root: int, time: float, duration: float, velocity: int):
        """Trombones respond."""
        for inst in ['trombone1', 'trombone2']:
            brass[inst].append(JazzNote(
                pitch=self._constrain(root, 40, 72),
                velocity=velocity,
                start_time=time,
                duration=duration,
                articulation='staccato'
            ))

    def _generate_piano(self, progression: List[JazzChord]) -> List[JazzNote]:
        """Generate piano comping."""
        piano = []
        current_beat = 0.0

        for chord in progression:
            for offset in [1.5, 3.5]:
                voicing = self.piano_comp.voice_chord(chord, octave=4)
                for pitch in voicing:
                    piano.append(JazzNote(
                        pitch=pitch,
                        velocity=random.randint(70, 85),
                        start_time=current_beat + offset,
                        duration=0.4,
                        articulation='normal'
                    ))
            current_beat += 4.0

        return piano

    def _generate_drums(self, total_beats: int) -> List[JazzNote]:
        """Generate swing drums."""
        drums = []

        for beat in range(total_beats):
            beat_time = float(beat)

            # Ride cymbal
            for eighth in [0.0, 0.667]:
                drums.append(JazzNote(
                    pitch=51,
                    velocity=68 if eighth == 0 else 58,
                    start_time=beat_time + eighth,
                    duration=0.1,
                    channel=9
                ))

            # Hi-hat on 2 & 4
            if beat % 4 in [1, 3]:
                drums.append(JazzNote(
                    pitch=42,
                    velocity=95,
                    start_time=beat_time,
                    duration=0.15,
                    channel=9
                ))

            # Kick
            if beat % 4 == 0:
                drums.append(JazzNote(
                    pitch=36,
                    velocity=85,
                    start_time=beat_time,
                    duration=0.1,
                    channel=9
                ))

        return drums

    def _constrain(self, pitch: int, min_p: int, max_p: int) -> int:
        """Constrain pitch to range."""
        while pitch < min_p:
            pitch += 12
        while pitch > max_p:
            pitch -= 12
        return pitch

    def _get_key_name(self) -> str:
        key_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        return key_names[self.key % 12]

    def export_midi(self, arrangement: Dict, output_file: str):
        """Export to MIDI."""

        print(f"\n💾 Exporting: {output_file}")

        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        tempo_microseconds = int(60_000_000 / self.tempo)

        tracks = [
            ("Lead Melody", 65, arrangement['melody'], 0),
            ("Alto Sax 1", 65, arrangement['sax_section']['alto1'], 1),
            ("Alto Sax 2", 65, arrangement['sax_section']['alto2'], 2),
            ("Tenor Sax 1", 66, arrangement['sax_section']['tenor1'], 3),
            ("Tenor Sax 2", 66, arrangement['sax_section']['tenor2'], 4),
            ("Baritone Sax", 67, arrangement['sax_section']['bari'], 5),
            ("Trumpet 1", 56, arrangement['brass_section']['trumpet1'], 6),
            ("Trumpet 2", 56, arrangement['brass_section']['trumpet2'], 7),
            ("Trumpet 3", 56, arrangement['brass_section']['trumpet3'], 8),
            ("Trumpet 4", 56, arrangement['brass_section']['trumpet4'], 9),
            ("Trombone 1", 57, arrangement['brass_section']['trombone1'], 10),
            ("Trombone 2", 57, arrangement['brass_section']['trombone2'], 11),
            ("Trombone 3", 57, arrangement['brass_section']['trombone3'], 12),
            ("Trombone 4", 57, arrangement['brass_section']['trombone4'], 13),
            ("Piano", 0, arrangement['piano'], 14),
            ("Bass", 32, arrangement['bass'], 15),
            ("Drums", 0, arrangement['drums'], 9),
        ]

        for track_name, program, notes, channel in tracks:
            track = MidiTrack()
            mid.tracks.append(track)

            track.append(MetaMessage('track_name', name=track_name, time=0))
            track.append(MetaMessage('set_tempo', tempo=tempo_microseconds, time=0))

            if channel != 9:
                track.append(Message('program_change', program=program, channel=channel, time=0))

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

            events.sort(key=lambda e: e['time'])

            last_time = 0.0
            for event in events:
                delta_ticks = int((event['time'] - last_time) * self.ticks_per_beat)

                if event['type'] == 'note_on':
                    track.append(Message('note_on', note=event['note'],
                                       velocity=event['velocity'],
                                       channel=event['channel'], time=delta_ticks))
                else:
                    track.append(Message('note_off', note=event['note'],
                                       velocity=0, channel=event['channel'], time=delta_ticks))

                last_time = event['time']

            track.append(MetaMessage('end_of_track', time=0))
            print(f"  ✓ {track_name}: {len(notes)} notes")

        mid.save(output_file)
        print(f"\n✅ Saved: {output_file}")


def main():
    print("\n" + "=" * 70)
    print("FIXED BIG BAND GENERATOR")
    print("All Issues Resolved")
    print("=" * 70)

    name = sys.argv[1] if len(sys.argv) > 1 else "fixed"
    tempo = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    key = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    generator = FixedBigBandGenerator(tempo, key)
    arrangement = generator.generate()
    generator.export_midi(arrangement, f"{name}_fixed.mid")

    print("\n" + "=" * 70)
    print("✅ FIXED ARRANGEMENT COMPLETE!")
    print("=" * 70)
    print("\nFixes Applied:")
    print("  ✅ Melody has rests and musical phrasing")
    print("  ✅ All instruments are monophonic (one note at a time)")
    print("  ✅ Trombones have notes (proper distribution)")
    print("  ✅ Brass patterns are varied (4 different patterns)")
    print("  ✅ Correct voicing (one note per instrument)")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
