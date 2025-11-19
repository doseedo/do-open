#!/usr/bin/env python3
"""
IMPROVED Big Band Generator - Fully Realistic Jazz Arrangement
================================================================

IMPROVEMENTS:
1. ✅ Random jazz chord progression selection (if not specified)
2. ✅ Sax solo melody properly follows chord changes using BebopMelodyGenerator
3. ✅ Swing timing applied to melody, sax section, and piano
4. ✅ Sax voicings use actual chord tones
5. ✅ Piano comping has swing feel and humanization
6. ✅ Melodic phrasing with proper density and articulation

Usage:
    python generate_big_band_improved.py [name] [tempo] [key] [progression]

    progression options: random, jazz_blues, rhythm_changes, ii_V_I, minor_ii_V_i
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
        BebopMelodyGenerator, JazzWalkingBass, PianoComping, CompingStyle,
        SwingTiming, SwingFeel
    )
    from algorithms.rhythm_engine import RhythmNote, HumanizationEngine, TimingStyle
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


class ImprovedBigBandGenerator:
    """Big band generator with realistic jazz feel and proper chord following."""

    def __init__(self, tempo: int = 140, key: int = 0, progression_type: str = "random"):
        self.tempo = tempo
        self.key = key
        self.progression_type = progression_type
        self.ticks_per_beat = 480

        # Jazz modules
        self.melody_gen = BebopMelodyGenerator()
        self.bass_gen = JazzWalkingBass(JazzStyle.SWING)
        self.piano_comp = PianoComping(CompingStyle.ROOTLESS)
        self.swing_timing = SwingTiming()

        # Humanization engine (for piano)
        self.humanizer = HumanizationEngine(ppqn=self.ticks_per_beat)

        # Swing feel
        self.swing_ratio = 0.62  # Medium swing

    def generate(self) -> Dict:
        """Generate complete big band arrangement with all improvements."""

        print("\n" + "=" * 70)
        print("IMPROVED BIG BAND GENERATOR")
        print("=" * 70)
        print(f"Tempo: {self.tempo} BPM | Key: {self._get_key_name()} | Progression: {self.progression_type}")
        print()

        # 1. GENERATE RANDOM PROGRESSION
        print("✓ Generating chord progression...")
        progression = self._get_progression()
        print(f"  Selected: {self.progression_type} ({len(progression)} chords)")

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

        # 2. MELODY THAT FOLLOWS CHORDS (with swing applied)
        print("✓ Generating melody (bebop style, follows chords)...")
        raw_melody = self._generate_melody_following_chords(progression)
        # Apply swing timing to melody
        arrangement['melody'] = SwingTiming.apply_swing(raw_melody, self.swing_ratio, intensity=1.0)
        print(f"  {len(arrangement['melody'])} notes (swung)")

        # 3. SAX HARMONY WITH CHORD TONES
        print("✓ Harmonizing sax section (using chord tones)...")
        for note in arrangement['melody']:
            chord_idx = int(note.start_time // 4)
            if chord_idx < len(progression):
                sax_notes = self._create_sax_voicing_from_chord(note, progression[chord_idx])
                for sax_name, sax_note in sax_notes.items():
                    arrangement['sax_section'][sax_name].append(sax_note)

        # Apply swing to sax section
        for sax_name in arrangement['sax_section']:
            arrangement['sax_section'][sax_name] = SwingTiming.apply_swing(
                arrangement['sax_section'][sax_name], self.swing_ratio, intensity=1.0
            )

        total_sax = sum(len(v) for v in arrangement['sax_section'].values())
        print(f"  {total_sax} notes across 5 saxes (swung)")

        # 4. BRASS WITH VARIED PATTERNS (MONOPHONIC)
        print("✓ Generating brass (varied, monophonic)...")
        raw_brass = self._generate_varied_brass(progression)
        # Apply swing to brass
        for inst in raw_brass:
            raw_brass[inst] = SwingTiming.apply_swing(raw_brass[inst], self.swing_ratio, intensity=1.0)
        arrangement['brass_section'] = raw_brass

        total_brass = sum(len(v) for v in arrangement['brass_section'].values())
        print(f"  {total_brass} notes across 8 brass (swung)")

        # 5. PIANO COMPING (with swing and humanization)
        print("✓ Generating piano (swung, humanized)...")
        arrangement['piano'] = self._generate_swung_piano(progression)
        print(f"  {len(arrangement['piano'])} notes (swung + humanized)")

        # 6. WALKING BASS (swung)
        print("✓ Generating bass...")
        raw_bass = self.bass_gen.generate_line(progression, beats_per_chord=4, style="swing")
        arrangement['bass'] = SwingTiming.apply_swing(raw_bass, self.swing_ratio, intensity=1.0)
        print(f"  {len(arrangement['bass'])} notes (swung)")

        # 7. DRUMS (swing pattern)
        print("✓ Generating drums...")
        arrangement['drums'] = self._generate_swing_drums(len(progression) * 4)
        print(f"  {len(arrangement['drums'])} hits")

        print()
        print("=" * 70)
        print("✅ IMPROVED ARRANGEMENT COMPLETE")
        print("=" * 70)

        return arrangement

    def _get_progression(self) -> List[JazzChord]:
        """Get chord progression (random or specified)."""

        if self.progression_type == "random":
            # Randomly select from available progressions
            options = ["jazz_blues", "rhythm_changes", "ii_V_I", "minor_ii_V_i"]
            selected = random.choice(options)
            self.progression_type = selected
        else:
            selected = self.progression_type

        if selected == "jazz_blues":
            return JazzProgressions.jazz_blues(self.key)
        elif selected == "rhythm_changes":
            return JazzProgressions.rhythm_changes_A(self.key)
        elif selected == "minor_ii_V_i":
            # Repeat minor ii-V-i to make a longer progression
            return JazzProgressions.minor_ii_V_i(self.key) * 4
        else:  # ii_V_I
            # Repeat ii-V-I to make a longer progression
            return JazzProgressions.ii_V_I(self.key) * 4

    def _generate_melody_following_chords(self, progression: List[JazzChord]) -> List[JazzNote]:
        """
        Generate melody that FOLLOWS the chord progression using BebopMelodyGenerator.

        Each chord gets a phrase generated specifically for it.
        """
        melody = []

        for i, chord in enumerate(progression):
            # Add rest every 3-4 chords for phrasing
            if i > 0 and i % 4 == 3:
                # Skip this chord (rest)
                continue

            # Generate phrase for THIS chord
            phrase = self.melody_gen.generate_phrase(
                chord=chord,
                length_beats=4,
                density=0.7  # 70% note density (not too dense, more melodic)
            )

            # Adjust timing to current position
            chord_start = i * 4.0
            for note in phrase:
                note.start_time += chord_start

            melody.extend(phrase)

        return melody

    def _create_sax_voicing_from_chord(self, lead_note: JazzNote, chord: JazzChord) -> Dict[str, JazzNote]:
        """
        Create 5-part sax voicing using actual chord tones.

        Uses PianoComping to get proper chord voicing, then distributes to saxes.
        """
        # Get chord tones from chord (proper jazz voicing)
        chord_tones = self._get_chord_tones(chord, octave=5)

        # Create 5-note voicing (close position)
        # Alto 1 gets the lead (melody note)
        # Others get chord tones below

        lead_pitch = lead_note.pitch

        # Build voicing: lead at top, then 3rd, 5th, 7th, root
        voicing_pitches = [
            lead_pitch,                          # Alto 1 (lead)
            self._constrain(chord_tones[1], 64, 81),  # Alto 2 (3rd)
            self._constrain(chord_tones[2], 55, 76),  # Tenor 1 (5th)
            self._constrain(chord_tones[3], 55, 76),  # Tenor 2 (7th)
            self._constrain(chord_tones[0], 48, 67),  # Bari (root)
        ]

        sax_names = ['alto1', 'alto2', 'tenor1', 'tenor2', 'bari']
        sax_notes = {}

        for sax, pitch in zip(sax_names, voicing_pitches):
            sax_notes[sax] = JazzNote(
                pitch=pitch,
                velocity=int(lead_note.velocity * 0.88),
                start_time=lead_note.start_time,
                duration=lead_note.duration,
                articulation=lead_note.articulation
            )

        return sax_notes

    def _get_chord_tones(self, chord: JazzChord, octave: int = 5) -> List[int]:
        """Get chord tones for a JazzChord."""
        base = 12 * octave + chord.root

        # Determine intervals based on chord quality
        if "maj7" in chord.quality or "maj" in chord.quality:
            third = base + 4   # Major 3rd
            seventh = base + 11  # Major 7th
        elif "min7" in chord.quality or "min" in chord.quality:
            third = base + 3   # Minor 3rd
            seventh = base + 10  # Minor 7th
        elif "dom7" in chord.quality or "dom" in chord.quality:
            third = base + 4   # Major 3rd
            seventh = base + 10  # Minor 7th (dominant)
        elif "min7b5" in chord.quality:
            third = base + 3   # Minor 3rd
            seventh = base + 10  # Minor 7th
        else:
            # Default to dominant
            third = base + 4
            seventh = base + 10

        fifth = base + 7

        return [base, third, fifth, seventh]

    def _generate_varied_brass(self, progression: List[JazzChord]) -> Dict[str, List[JazzNote]]:
        """Generate VARIED brass patterns that are MONOPHONIC."""
        brass = {
            'trumpet1': [], 'trumpet2': [], 'trumpet3': [], 'trumpet4': [],
            'trombone1': [], 'trombone2': [], 'trombone3': [], 'trombone4': []
        }

        current_beat = 0.0

        for i, chord in enumerate(progression):
            root = chord.root + 60

            # VARIED PATTERNS (4 different types)
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

    def _generate_swung_piano(self, progression: List[JazzChord]) -> List[JazzNote]:
        """
        Generate piano comping with SWING FEEL and HUMANIZATION.

        Uses rootless voicings, syncopated rhythm, swing timing, and velocity variation.
        """
        piano = []
        current_beat = 0.0

        for chord in progression:
            # Syncopated comping pattern (not on beat 1)
            # Beat 2.5 and 4 (backbeat-ish)
            for offset in [1.5, 3.5]:
                voicing = self.piano_comp.voice_chord(chord, octave=4)

                # Add each note of the voicing with slight velocity variation
                for pitch in voicing:
                    piano.append(JazzNote(
                        pitch=pitch,
                        velocity=random.randint(70, 85),  # Velocity variation
                        start_time=current_beat + offset,
                        duration=0.4,
                        articulation='normal'
                    ))

            current_beat += 4.0

        # Apply swing timing
        piano = SwingTiming.apply_swing(piano, self.swing_ratio, intensity=1.0)

        # Apply humanization (convert to RhythmNote, humanize, convert back)
        rhythm_notes = self._jazz_to_rhythm_notes(piano)
        rhythm_notes = self.humanizer.humanize_timing(rhythm_notes, style=TimingStyle.HUMAN)
        rhythm_notes = self.humanizer.humanize_velocity(rhythm_notes, variation=0.12)
        piano = self._rhythm_to_jazz_notes(rhythm_notes)

        return piano

    def _jazz_to_rhythm_notes(self, jazz_notes: List[JazzNote]) -> List[RhythmNote]:
        """Convert JazzNote to RhythmNote for humanization."""
        rhythm_notes = []
        for jn in jazz_notes:
            rhythm_notes.append(RhythmNote(
                tick=int(jn.start_time * self.ticks_per_beat),
                duration=int(jn.duration * self.ticks_per_beat),
                velocity=jn.velocity,
                pitch=jn.pitch
            ))
        return rhythm_notes

    def _rhythm_to_jazz_notes(self, rhythm_notes: List[RhythmNote]) -> List[JazzNote]:
        """Convert RhythmNote back to JazzNote."""
        jazz_notes = []
        for rn in rhythm_notes:
            jazz_notes.append(JazzNote(
                pitch=rn.pitch,
                velocity=rn.velocity,
                start_time=rn.tick / self.ticks_per_beat,
                duration=rn.duration / self.ticks_per_beat,
                articulation='normal'
            ))
        return jazz_notes

    def _generate_swing_drums(self, total_beats: int) -> List[JazzNote]:
        """Generate swing drums with ride cymbal pattern."""
        drums = []

        for beat in range(total_beats):
            beat_time = float(beat)

            # Ride cymbal - swing pattern (on-beat and swung off-beat)
            # On-beat
            drums.append(JazzNote(
                pitch=51,  # Ride cymbal
                velocity=68,
                start_time=beat_time,
                duration=0.1,
                channel=9
            ))

            # Off-beat (will be swung by swing timing if applied)
            drums.append(JazzNote(
                pitch=51,
                velocity=58,
                start_time=beat_time + 0.5,  # This will be swung
                duration=0.1,
                channel=9
            ))

            # Hi-hat on 2 & 4 (backbeat)
            if beat % 4 in [1, 3]:
                drums.append(JazzNote(
                    pitch=42,  # Closed hi-hat
                    velocity=95,
                    start_time=beat_time,
                    duration=0.15,
                    channel=9
                ))

            # Kick drum
            if beat % 4 == 0:
                drums.append(JazzNote(
                    pitch=36,
                    velocity=85,
                    start_time=beat_time,
                    duration=0.1,
                    channel=9
                ))

        # Apply swing to drums
        drums = SwingTiming.apply_swing(drums, self.swing_ratio, intensity=1.0)

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
    print("IMPROVED BIG BAND GENERATOR")
    print("Realistic Jazz with Swing Feel")
    print("=" * 70)

    name = sys.argv[1] if len(sys.argv) > 1 else "improved"
    tempo = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    key = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    progression = sys.argv[4] if len(sys.argv) > 4 else "random"

    generator = ImprovedBigBandGenerator(tempo, key, progression)
    arrangement = generator.generate()
    generator.export_midi(arrangement, f"{name}_improved.mid")

    print("\n" + "=" * 70)
    print("✅ IMPROVED ARRANGEMENT COMPLETE!")
    print("=" * 70)
    print("\nImprovements Applied:")
    print("  ✅ Random jazz chord progression selection")
    print("  ✅ Melody follows chord changes (BebopMelodyGenerator)")
    print("  ✅ Swing timing applied to all parts")
    print("  ✅ Sax voicings use actual chord tones")
    print("  ✅ Piano has swing feel and humanization")
    print("  ✅ Realistic melodic phrasing and density")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
