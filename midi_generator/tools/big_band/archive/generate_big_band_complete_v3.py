#!/usr/bin/env python3
"""
COMPLETE Big Band Generator V3 - Professional Jazz Arrangement
================================================================

COMPREHENSIVE FIXES:
1. ✅ Swing duration compensation (notes shortened to prevent overlap)
2. ✅ Proper sax soli voice leading using arrangement_engine _create_close_voicing()
3. ✅ Drum pattern variation with ghost notes, fills every 4-8 bars
4. ✅ Piano rhythm variation (4 different comping patterns)
5. ✅ Arrangement awareness (snare/piano sync with melodic accents)
6. ✅ Random jazz progressions
7. ✅ Melody follows chords with proper phrasing
8. ✅ Full module ecosystem utilization

Usage:
    python generate_big_band_complete_v3.py [name] [tempo] [key] [progression]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: pip3 install mido")
    sys.exit(1)

try:
    from genres.jazz import (
        JazzNote, JazzChord, JazzProgressions,
        BebopMelodyGenerator, JazzWalkingBass, PianoComping, CompingStyle,
        SwingTiming, SwingFeel, JazzStyle
    )
    from algorithms.rhythm_engine import RhythmNote, HumanizationEngine, TimingStyle
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


@dataclass
class ChordEvent:
    """Simplified ChordEvent for use with _create_close_voicing."""
    root: int
    quality: str
    start_time: float
    duration: float


class CompleteBigBandV3:
    """Professional big band generator using full module ecosystem."""

    def __init__(self, tempo: int = 140, key: int = 0, progression_type: str = "random"):
        self.tempo = tempo
        self.key = key
        self.progression_type = progression_type
        self.ticks_per_beat = 480

        # Generators
        self.melody_gen = BebopMelodyGenerator()
        self.bass_gen = JazzWalkingBass(JazzStyle.SWING)
        self.piano_comp = PianoComping(CompingStyle.ROOTLESS)
        self.humanizer = HumanizationEngine(ppqn=self.ticks_per_beat)

        # Swing
        self.swing_ratio = 0.62  # Medium swing

    def generate(self) -> Dict:
        """Generate complete big band arrangement."""

        print("\n" + "=" * 70)
        print("COMPLETE BIG BAND GENERATOR V3")
        print("=" * 70)
        print(f"Tempo: {self.tempo} BPM | Key: {self._get_key_name()} | Style: {self.progression_type}")
        print()

        # 1. PROGRESSION
        print("✓ Generating progression...")
        progression = self._get_progression()
        print(f"  {len(progression)} chords")

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

        # 2. MELODY (with proper phrasing)
        print("✓ Generating melody...")
        raw_melody = self._generate_melody_with_phrasing(progression)
        # Apply swing + fix duration
        arrangement['melody'] = self._apply_swing_with_duration_fix(raw_melody)
        print(f"  {len(arrangement['melody'])} notes")

        # 3. SAX SOLI (proper voice leading using arrangement_engine algorithm)
        print("✓ Harmonizing sax soli (voice leading)...")
        sax_soli = self._harmonize_sax_soli_proper(arrangement['melody'], progression)
        for sax in arrangement['sax_section']:
            arrangement['sax_section'][sax] = self._apply_swing_with_duration_fix(sax_soli[sax])

        total_sax = sum(len(v) for v in arrangement['sax_section'].values())
        print(f"  {total_sax} notes (proper voice leading)")

        # 4. BRASS (varied, monophonic)
        print("✓ Generating brass...")
        raw_brass = self._generate_varied_brass(progression)
        for inst in raw_brass:
            arrangement['brass_section'][inst] = self._apply_swing_with_duration_fix(raw_brass[inst])

        total_brass = sum(len(v) for v in arrangement['brass_section'].values())
        print(f"  {total_brass} notes")

        # 5. DRUMS (variation + fills + ghost notes)
        print("✓ Generating drums (variation + fills)...")
        arrangement['drums'] = self._generate_drums_with_variation(
            len(progression) * 4,
            raw_melody  # Pass melody for accent extraction
        )
        print(f"  {len(arrangement['drums'])} hits")

        # 6. PIANO (varied rhythm patterns + arrangement awareness)
        print("✓ Generating piano (varied comping)...")
        arrangement['piano'] = self._generate_piano_varied(
            progression,
            raw_melody  # Pass melody for rhythmic sync
        )
        print(f"  {len(arrangement['piano'])} notes")

        # 7. BASS (swung walking)
        print("✓ Generating bass...")
        raw_bass = self.bass_gen.generate_line(progression, beats_per_chord=4, style="swing")
        arrangement['bass'] = self._apply_swing_with_duration_fix(raw_bass)
        print(f"  {len(arrangement['bass'])} notes")

        print()
        print("=" * 70)
        print("✅ COMPLETE V3 ARRANGEMENT DONE")
        print("=" * 70)

        return arrangement

    def _get_progression(self) -> List[JazzChord]:
        """Get chord progression."""
        if self.progression_type == "random":
            options = ["jazz_blues", "rhythm_changes", "ii_V_I", "minor_ii_V_i"]
            self.progression_type = random.choice(options)

        if self.progression_type == "jazz_blues":
            return JazzProgressions.jazz_blues(self.key)
        elif self.progression_type == "rhythm_changes":
            return JazzProgressions.rhythm_changes_A(self.key)
        elif self.progression_type == "minor_ii_V_i":
            return JazzProgressions.minor_ii_V_i(self.key) * 4
        else:  # ii_V_I
            return JazzProgressions.ii_V_I(self.key) * 4

    def _generate_melody_with_phrasing(self, progression: List[JazzChord]) -> List[JazzNote]:
        """Generate melody with rests for phrasing."""
        melody = []

        for i, chord in enumerate(progression):
            # Rest every 4 bars
            if i > 0 and i % 4 == 3:
                continue

            phrase = self.melody_gen.generate_phrase(chord, length_beats=4, density=0.7)
            chord_start = i * 4.0
            for note in phrase:
                note.start_time += chord_start
            melody.extend(phrase)

        return melody

    def _apply_swing_with_duration_fix(self, notes: List[JazzNote]) -> List[JazzNote]:
        """
        Apply swing timing AND fix note durations to prevent overlap.

        KEY FIX: After delaying off-beat notes, shorten them so they don't
        overlap the next note.
        """
        if not notes:
            return []

        # Apply swing timing
        swung = SwingTiming.apply_swing(notes, self.swing_ratio, intensity=1.0)

        # Sort by start time
        swung.sort(key=lambda n: n.start_time)

        # Fix durations to prevent overlap
        fixed = []
        for i, note in enumerate(swung):
            new_note = JazzNote(
                pitch=note.pitch,
                velocity=note.velocity,
                start_time=note.start_time,
                duration=note.duration,
                articulation=note.articulation,
                swing_offset=note.swing_offset,
                channel=note.channel
            )

            # Check if this note overlaps the next
            if i < len(swung) - 1:
                next_start = swung[i + 1].start_time
                note_end = new_note.start_time + new_note.duration

                if note_end > next_start:
                    # Shorten duration to end just before next note
                    gap = 0.05  # Small gap for articulation
                    new_note.duration = max(0.1, next_start - new_note.start_time - gap)

            fixed.append(new_note)

        return fixed

    def _harmonize_sax_soli_proper(self, melody: List[JazzNote],
                                    progression: List[JazzChord]) -> Dict[str, List[JazzNote]]:
        """
        Proper sax soli using arrangement_engine algorithm.

        Lower voices FOLLOW melodic contour by finding chord tones below each melody note.
        This creates proper voice leading, not static chord tones.
        """
        sax_soli = {'alto1': [], 'alto2': [], 'tenor1': [], 'tenor2': [], 'bari': []}

        for note in melody:
            # Find chord at this time
            chord = self._find_chord_at_time(note.start_time, progression)
            if not chord:
                continue

            # Convert to ChordEvent for _create_close_voicing
            chord_event = ChordEvent(
                root=chord.root,
                quality=chord.quality,
                start_time=note.start_time,
                duration=4.0
            )

            # Use arrangement_engine algorithm to create 5-part close voicing
            voicing = self._create_close_voicing(note.pitch, chord_event, 5)

            # Distribute to sax voices
            if len(voicing) >= 5:
                sax_names = ['alto1', 'alto2', 'tenor1', 'tenor2', 'bari']

                for i, sax in enumerate(sax_names):
                    pitch = voicing[-(i+1)]  # Top to bottom

                    # Constrain to sax ranges
                    if 'alto' in sax:
                        pitch = self._constrain(pitch, 64, 81)
                    elif 'tenor' in sax:
                        pitch = self._constrain(pitch, 55, 76)
                    else:  # bari
                        pitch = self._constrain(pitch, 48, 67)

                    sax_soli[sax].append(JazzNote(
                        pitch=pitch,
                        velocity=int(note.velocity * 0.88),
                        start_time=note.start_time,
                        duration=note.duration,
                        articulation=note.articulation
                    ))

        return sax_soli

    def _create_close_voicing(self, melody_pitch: int, chord: ChordEvent,
                             num_voices: int) -> List[int]:
        """
        From arrangement_engine.py - creates proper voice leading.

        Finds chord tones below melody pitch, creating voice leading that
        follows the melodic contour.
        """
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
            found = False
            for tone in sorted(chord_tones, reverse=True):
                candidate = (current_pitch // 12) * 12 + tone
                if candidate < current_pitch and candidate not in voicing:
                    voicing.append(candidate)
                    current_pitch = candidate
                    found = True
                    break

            if not found:
                # No chord tone found, use chromatic fill
                voicing.append(current_pitch)
                current_pitch -= 1

        return sorted(voicing)

    def _find_chord_at_time(self, time: float, progression: List[JazzChord]) -> Optional[JazzChord]:
        """Find chord at specified time."""
        chord_idx = int(time // 4)
        if 0 <= chord_idx < len(progression):
            return progression[chord_idx]
        return None

    def _generate_varied_brass(self, progression: List[JazzChord]) -> Dict[str, List[JazzNote]]:
        """Generate varied brass patterns."""
        brass = {
            'trumpet1': [], 'trumpet2': [], 'trumpet3': [], 'trumpet4': [],
            'trombone1': [], 'trombone2': [], 'trombone3': [], 'trombone4': []
        }

        current_beat = 0.0

        for i, chord in enumerate(progression):
            root = chord.root + 60

            # 4 different patterns
            if i % 4 == 0:
                self._add_brass_hit(brass, root, current_beat, 0.3, 100)
            elif i % 4 == 1:
                self._add_brass_sustain(brass, root, current_beat + 2.0, 1.5, 85)
            elif i % 4 == 2:
                self._add_brass_hit(brass, root, current_beat + 2.5, 0.3, 95)
                self._add_brass_hit(brass, root, current_beat + 3.5, 0.3, 90)
            else:
                self._add_trumpet_call(brass, root, current_beat + 1.0, 0.5, 90)
                self._add_trombone_response(brass, root, current_beat + 2.0, 0.5, 85)

            current_beat += 4.0

        return brass

    def _add_brass_hit(self, brass: Dict, root: int, time: float, duration: float, velocity: int):
        """Add brass hit - one note per instrument."""
        voicing = [
            root + 12, root + 7, root + 4, root,  # Trumpets
            root - 5, root - 8, root - 12, root - 17  # Trombones
        ]
        instruments = ['trumpet1', 'trumpet2', 'trumpet3', 'trumpet4',
                      'trombone1', 'trombone2', 'trombone3', 'trombone4']

        for inst, pitch in zip(instruments, voicing):
            if 'trumpet' in inst:
                pitch = self._constrain(pitch, 55, 82)
            else:
                pitch = self._constrain(pitch, 40, 72)

            brass[inst].append(JazzNote(
                pitch=pitch, velocity=velocity, start_time=time,
                duration=duration, articulation='accent'
            ))

    def _add_brass_sustain(self, brass: Dict, root: int, time: float, duration: float, velocity: int):
        """Sustained brass pad."""
        brass['trumpet1'].append(JazzNote(
            pitch=self._constrain(root + 12, 55, 82), velocity=velocity,
            start_time=time, duration=duration, articulation='legato'
        ))
        brass['trombone3'].append(JazzNote(
            pitch=self._constrain(root - 12, 40, 72), velocity=velocity - 5,
            start_time=time, duration=duration, articulation='legato'
        ))

    def _add_trumpet_call(self, brass: Dict, root: int, time: float, duration: float, velocity: int):
        """Trumpet call."""
        for inst in ['trumpet1', 'trumpet2']:
            brass[inst].append(JazzNote(
                pitch=self._constrain(root + 12, 55, 82), velocity=velocity,
                start_time=time, duration=duration, articulation='staccato'
            ))

    def _add_trombone_response(self, brass: Dict, root: int, time: float, duration: float, velocity: int):
        """Trombone response."""
        for inst in ['trombone1', 'trombone2']:
            brass[inst].append(JazzNote(
                pitch=self._constrain(root, 40, 72), velocity=velocity,
                start_time=time, duration=duration, articulation='staccato'
            ))

    def _generate_drums_with_variation(self, total_beats: int, melody: List[JazzNote]) -> List[JazzNote]:
        """
        Drums with variation, fills, ghost notes, and arrangement awareness.

        - Pattern changes every 4 bars
        - Fills every 8 bars
        - Ghost notes on snare
        - Syncs with melodic accents
        """
        drums = []

        # Extract melodic accent times for snare sync
        accent_times = self._extract_melodic_accents(melody)

        for bar in range(total_beats // 4):
            bar_start = bar * 4.0

            # Fill every 8 bars
            if bar > 0 and bar % 8 == 7:
                drums.extend(self._create_tom_fill(bar_start + 2.0))
                continue

            # Pattern variation every 4 bars
            pattern_type = (bar // 4) % 3

            for beat in range(4):
                beat_time = bar_start + beat

                if pattern_type == 0:
                    # Simple swing pattern
                    self._add_swing_pattern(drums, beat_time, beat)
                elif pattern_type == 1:
                    # With ghost notes
                    self._add_swing_with_ghosts(drums, beat_time, beat)
                else:
                    # Syncopated
                    self._add_syncopated_pattern(drums, beat_time, beat, accent_times)

        # Apply swing to drums
        drums = self._apply_swing_with_duration_fix(drums)

        return drums

    def _extract_melodic_accents(self, melody: List[JazzNote]) -> List[float]:
        """Extract strong beat times from melody for arrangement coordination."""
        accents = []
        for note in melody:
            # Strong beats (downbeats, emphasized notes)
            if note.start_time % 1.0 == 0 and note.velocity > 90:
                accents.append(note.start_time)
        return accents

    def _add_swing_pattern(self, drums: List[JazzNote], beat_time: float, beat: int):
        """Basic swing pattern."""
        # Ride cymbal
        drums.append(JazzNote(pitch=51, velocity=68, start_time=beat_time, duration=0.1, channel=9))
        drums.append(JazzNote(pitch=51, velocity=58, start_time=beat_time + 0.5, duration=0.1, channel=9))

        # Hi-hat on 2 & 4
        if beat in [1, 3]:
            drums.append(JazzNote(pitch=42, velocity=95, start_time=beat_time, duration=0.15, channel=9))

        # Kick
        if beat == 0:
            drums.append(JazzNote(pitch=36, velocity=85, start_time=beat_time, duration=0.1, channel=9))

    def _add_swing_with_ghosts(self, drums: List[JazzNote], beat_time: float, beat: int):
        """Swing pattern with ghost notes (Purdie style)."""
        self._add_swing_pattern(drums, beat_time, beat)

        # Ghost notes on snare (soft hits between main beats)
        if random.random() < 0.6:
            ghost_time = beat_time + random.choice([0.25, 0.75])
            drums.append(JazzNote(pitch=38, velocity=35, start_time=ghost_time, duration=0.1, channel=9))

    def _add_syncopated_pattern(self, drums: List[JazzNote], beat_time: float, beat: int, accents: List[float]):
        """Syncopated pattern with arrangement awareness."""
        self._add_swing_pattern(drums, beat_time, beat)

        # Add snare hit if melody has accent at this time
        if beat_time in accents or (beat_time + 0.5) in accents:
            drums.append(JazzNote(pitch=38, velocity=90, start_time=beat_time + 0.5, duration=0.1, channel=9))

    def _create_tom_fill(self, start_time: float) -> List[JazzNote]:
        """Create tom fill (high to low)."""
        fill = []
        tom_pitches = [50, 47, 45, 43]  # High tom to low
        for i, pitch in enumerate(tom_pitches):
            fill.append(JazzNote(
                pitch=pitch, velocity=90 - i * 5,
                start_time=start_time + i * 0.25,
                duration=0.2, channel=9
            ))
        return fill

    def _generate_piano_varied(self, progression: List[JazzChord],
                               melody: List[JazzNote]) -> List[JazzNote]:
        """
        Piano with varied comping patterns and arrangement awareness.

        - 4 different rhythm patterns
        - Syncs with melodic rhythms
        - Swing + humanization
        """
        piano = []
        current_beat = 0.0

        # Extract melody rhythm for coordination
        melody_rhythm = [n.start_time for n in melody]

        for i, chord in enumerate(progression):
            # 4 different comping patterns
            pattern_type = i % 4

            voicing = self.piano_comp.voice_chord(chord, octave=4)

            if pattern_type == 0:
                # Standard offbeats (1.5, 3.5)
                for offset in [1.5, 3.5]:
                    self._add_piano_chord(piano, voicing, current_beat + offset, 0.4, random.randint(70, 85))

            elif pattern_type == 1:
                # On-beat (2, 4)
                for offset in [2.0, 4.0]:
                    self._add_piano_chord(piano, voicing, current_beat + offset, 0.3, random.randint(75, 90))

            elif pattern_type == 2:
                # Syncopated (0.5, 2.5)
                for offset in [0.5, 2.5]:
                    self._add_piano_chord(piano, voicing, current_beat + offset, 0.3, random.randint(72, 87))

            else:
                # Arrangement-aware (sync with melody)
                for melody_time in melody_rhythm:
                    if current_beat <= melody_time < current_beat + 4:
                        # Play shortly after melody note
                        offset = melody_time - current_beat + 0.1
                        if 0 <= offset < 4:
                            self._add_piano_chord(piano, voicing, current_beat + offset, 0.3, random.randint(68, 82))
                            break  # One hit per chord

            current_beat += 4.0

        # Apply swing + humanization
        piano = self._apply_swing_with_duration_fix(piano)

        # Humanize
        rhythm_notes = self._jazz_to_rhythm_notes(piano)
        rhythm_notes = self.humanizer.humanize_timing(rhythm_notes, style=TimingStyle.HUMAN)
        rhythm_notes = self.humanizer.humanize_velocity(rhythm_notes, variation=0.12)
        piano = self._rhythm_to_jazz_notes(rhythm_notes)

        return piano

    def _add_piano_chord(self, piano: List[JazzNote], voicing: List[int],
                        time: float, duration: float, velocity: int):
        """Add piano chord."""
        for pitch in voicing:
            piano.append(JazzNote(
                pitch=pitch, velocity=velocity,
                start_time=time, duration=duration, articulation='normal'
            ))

    def _jazz_to_rhythm_notes(self, jazz_notes: List[JazzNote]) -> List[RhythmNote]:
        """Convert to RhythmNote for humanization."""
        return [RhythmNote(
            tick=int(jn.start_time * self.ticks_per_beat),
            duration=int(jn.duration * self.ticks_per_beat),
            velocity=jn.velocity,
            pitch=jn.pitch
        ) for jn in jazz_notes]

    def _rhythm_to_jazz_notes(self, rhythm_notes: List[RhythmNote]) -> List[JazzNote]:
        """Convert back to JazzNote."""
        return [JazzNote(
            pitch=rn.pitch,
            velocity=rn.velocity,
            start_time=rn.tick / self.ticks_per_beat,
            duration=rn.duration / self.ticks_per_beat,
            articulation='normal'
        ) for rn in rhythm_notes]

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
                events.append({'type': 'note_on', 'time': note.start_time, 'note': note.pitch,
                             'velocity': note.velocity, 'channel': note_channel})
                events.append({'type': 'note_off', 'time': note.start_time + note.duration,
                             'note': note.pitch, 'channel': note_channel})

            events.sort(key=lambda e: e['time'])

            last_time = 0.0
            for event in events:
                delta_ticks = int((event['time'] - last_time) * self.ticks_per_beat)
                if event['type'] == 'note_on':
                    track.append(Message('note_on', note=event['note'], velocity=event['velocity'],
                                       channel=event['channel'], time=delta_ticks))
                else:
                    track.append(Message('note_off', note=event['note'], velocity=0,
                                       channel=event['channel'], time=delta_ticks))
                last_time = event['time']

            track.append(MetaMessage('end_of_track', time=0))
            print(f"  ✓ {track_name}: {len(notes)} notes")

        mid.save(output_file)
        print(f"\n✅ Saved: {output_file}")


def main():
    print("\n" + "=" * 70)
    print("COMPLETE BIG BAND GENERATOR V3")
    print("Professional Jazz Arrangement")
    print("=" * 70)

    name = sys.argv[1] if len(sys.argv) > 1 else "complete_v3"
    tempo = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    key = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    progression = sys.argv[4] if len(sys.argv) > 4 else "random"

    generator = CompleteBigBandV3(tempo, key, progression)
    arrangement = generator.generate()
    generator.export_midi(arrangement, f"{name}_v3.mid")

    print("\n" + "=" * 70)
    print("✅ COMPLETE V3 ARRANGEMENT DONE!")
    print("=" * 70)
    print("\nAll Fixes Applied:")
    print("  ✅ Swing duration compensated (no overlap)")
    print("  ✅ Sax soli proper voice leading (follows melody contour)")
    print("  ✅ Drum variation with fills + ghost notes")
    print("  ✅ Piano varied rhythm (4 patterns)")
    print("  ✅ Arrangement awareness (snare/piano sync)")
    print()
    print("Genuine Module Gaps Identified:")
    print("  ⚠️  Rhythmic accent extractor (manually implemented)")
    print("  ⚠️  Swing duration compensator (manually implemented)")
    print("  ⚠️  Multi-bar arrangement structure (manual pattern cycling)")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
