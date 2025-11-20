#!/usr/bin/env python3
"""
Count Basie Arranger
====================

This module implements Count Basie's arranging style, transforming lead sheets
into authentic Basie-style big band arrangements.

**Basie Arranging Techniques Implemented:**
1. **Riff-based backgrounds** - Short, repeated rhythmic figures
2. **Punchy section hits** - Brass and sax stabs on strong beats
3. **Sparse piano comping** - Minimalist, strategic piano
4. **Powerful rhythm section** - Feathered kick, Freddie Green guitar
5. **Shout chorus** - Climactic final chorus
6. **Button intro/ending** - Short, punchy beginnings and endings
7. **Simple harmony** - Blues-based, mostly 7th chords
8. **Unison/octave voicings** - Simple, powerful section writing

**Usage Example:**
    ```python
    from styles.basie_arranger import BasieArranger
    from styles.basie_profile import BASIE_STYLE

    # Arrange in Basie style
    arrangement = BasieArranger.arrange_in_basie_style(
        melody=melody_notes,
        chords=chord_events,
        style_config=BASIE_STYLE
    )
    ```

**References:**
- "One O'Clock Jump" - Classic Basie arrangement
- "April in Paris" - Famous shout chorus
- "Corner Pocket" - Riff-based swing
- Analysis from ejazzlines.com

Author: Agent 14 - Count Basie Style Analyzer
Date: 2025
License: MIT
"""

import random
import copy
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import from parent directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.midi_analyzer import NoteEvent, ChordEvent
from .basie_profile import BasieStyleConfig, BASIE_STYLE


# ============================================================================
# BASIE RIFF GENERATOR
# ============================================================================

class BasieRiffGenerator:
    """
    Generate authentic Count Basie-style riffs.

    Basie riffs are characterized by:
    - Short (1-2 bars typically)
    - Rhythmically punchy
    - Blues-based
    - Simple melodic content
    - Repetitive and memorable
    """

    @staticmethod
    def generate_basie_riff(
        chord: ChordEvent,
        bars: int = 2,
        riff_style: str = "blues",
        section: str = "brass"  # brass, sax, or both
    ) -> List[NoteEvent]:
        """
        Generate a Basie-style riff.

        Args:
            chord: Chord to build riff over
            bars: Length in bars (typically 1 or 2)
            riff_style: 'blues', 'call_response', 'shout'
            section: 'brass', 'sax', or 'both'

        Returns:
            List of NoteEvent objects forming the riff
        """
        riff_notes = []
        beat_duration = 1.0  # Assuming 4/4

        if riff_style == "blues":
            # Blues riff pattern: syncopated rhythm on blues scale
            rhythm_pattern = [0, 0.5, 1.5, 2, 3, 3.5]  # Offbeat heavy
            pitches = BasieRiffGenerator._get_blues_riff_pitches(chord)

        elif riff_style == "call_response":
            # Call pattern (first bar), response (second bar)
            rhythm_pattern = [0, 1, 2, 3] if bars == 1 else [0, 1, 4, 5, 6]
            pitches = BasieRiffGenerator._get_call_response_pitches(chord)

        elif riff_style == "shout":
            # Shout chorus riff: more notes, higher energy
            rhythm_pattern = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
            pitches = BasieRiffGenerator._get_shout_riff_pitches(chord)

        else:
            # Default simple riff
            rhythm_pattern = [0, 1, 2, 3]
            pitches = [chord.root + 60, chord.root + 64, chord.root + 67]

        # Create notes
        for i, start_beat in enumerate(rhythm_pattern):
            if start_beat >= bars * 4:  # Don't exceed bar count
                break

            pitch = pitches[i % len(pitches)]
            start_time = chord.start_time + start_beat * beat_duration

            # Duration: mostly short (staccato)
            duration = 0.3 if random.random() < 0.7 else 0.6

            note = NoteEvent(
                start_time=start_time,
                duration=duration,
                start_tick=int(start_time * 480),
                duration_ticks=int(duration * 480),
                pitch=pitch,
                velocity=95,  # Punchy
                channel=0,
                track_idx=10
            )
            riff_notes.append(note)

        return riff_notes

    @staticmethod
    def _get_blues_riff_pitches(chord: ChordEvent) -> List[int]:
        """Get pitches for blues-style riff."""
        root = chord.root + 60  # Mid register
        # Blues scale: 1, b3, 4, b5, 5, b7
        return [
            root,
            root + 3,   # b3
            root + 5,   # 4
            root + 6,   # b5
            root + 7,   # 5
            root + 10,  # b7
        ]

    @staticmethod
    def _get_call_response_pitches(chord: ChordEvent) -> List[int]:
        """Get pitches for call-response pattern."""
        root = chord.root + 60
        # Simple ascending/descending pattern
        return [root, root + 4, root + 7, root + 4, root]

    @staticmethod
    def _get_shout_riff_pitches(chord: ChordEvent) -> List[int]:
        """Get pitches for shout chorus riff (higher, more energetic)."""
        root = chord.root + 65  # Higher register
        # More chromatic, energetic
        return [root, root + 2, root + 4, root + 5, root + 7, root + 9]

    @staticmethod
    def generate_section_hit(
        chord: ChordEvent,
        time: float,
        voicing: str = "unison",  # unison, octaves, basic_chord
        duration: float = 0.25
    ) -> List[NoteEvent]:
        """
        Generate a punchy section hit (Basie signature).

        Args:
            chord: Chord for the hit
            time: When to place the hit
            voicing: 'unison', 'octaves', 'basic_chord'
            duration: Length of the hit (short for punch)

        Returns:
            List of NoteEvent objects for the hit
        """
        hit_notes = []

        if voicing == "unison":
            # All instruments play same note
            pitch = chord.root + 60
            pitches = [pitch]

        elif voicing == "octaves":
            # Spread across octaves
            pitch = chord.root + 60
            pitches = [pitch - 12, pitch, pitch + 12]

        elif voicing == "basic_chord":
            # Simple chord voicing
            root = chord.root + 60
            if 'major' in chord.quality:
                pitches = [root, root + 4, root + 7]  # Major triad
            else:
                pitches = [root, root + 3, root + 7]  # Minor triad
        else:
            pitches = [chord.root + 60]

        # Create the hit
        for pitch in pitches:
            note = NoteEvent(
                start_time=time,
                duration=duration,
                start_tick=int(time * 480),
                duration_ticks=int(duration * 480),
                pitch=pitch,
                velocity=105,  # Accented, loud
                channel=0,
                track_idx=15
            )
            hit_notes.append(note)

        return hit_notes


# ============================================================================
# MAIN BASIE ARRANGER
# ============================================================================

class BasieArranger:
    """
    Main arranger for Count Basie style.

    This class applies Basie's arranging techniques to transform
    simple lead sheets into full big band arrangements.
    """

    @staticmethod
    def arrange_in_basie_style(
        melody: List[NoteEvent],
        chords: List[ChordEvent],
        style_config: Optional[BasieStyleConfig] = None
    ) -> Dict[str, List[NoteEvent]]:
        """
        Create complete Basie-style arrangement.

        Args:
            melody: Melody line
            chords: Chord progression
            style_config: Basie style configuration (uses default if None)

        Returns:
            Dictionary mapping section names to note lists:
            - 'lead': Lead melody
            - 'sax_section': Sax section (mostly riffs and hits)
            - 'brass_section': Brass section (mostly riffs and hits)
            - 'piano': Sparse piano comping
            - 'bass': Walking bass
            - 'drums': Swing drums with feathered kick
            - 'guitar': Freddie Green style rhythm guitar
        """
        if style_config is None:
            style_config = BASIE_STYLE

        arrangement = {}

        # Lead melody (alto sax or trumpet)
        arrangement['lead'] = BasieArranger._create_lead_melody(melody)

        # Sax section: riffs and hits (not constant harmony like Ellington)
        arrangement['sax_section'] = BasieArranger._create_sax_riffs_and_hits(
            melody, chords, style_config
        )

        # Brass section: punchy hits and background riffs
        arrangement['brass_section'] = BasieArranger._create_brass_hits_and_riffs(
            chords, style_config
        )

        # Piano: VERY sparse (Basie minimalism)
        arrangement['piano'] = BasieArranger._create_sparse_piano_comping(
            chords, style_config
        )

        # Bass: walking bass
        arrangement['bass'] = BasieArranger._create_walking_bass(chords)

        # Drums: swing with feathered kick
        arrangement['drums'] = BasieArranger._create_basie_drums(
            melody, style_config
        )

        # Guitar: Freddie Green style (if enabled)
        if style_config.freddie_green_guitar:
            arrangement['guitar'] = BasieArranger._create_freddie_green_guitar(
                chords
            )

        return arrangement

    @staticmethod
    def _create_lead_melody(melody: List[NoteEvent]) -> List[NoteEvent]:
        """
        Create lead melody line.

        In Basie arrangements, melody is often played by solo trumpet or
        alto sax, sometimes in unison with section.
        """
        lead = []
        for note in melody:
            new_note = copy.copy(note)
            # Keep in comfortable alto sax/trumpet range
            while new_note.pitch < 60:
                new_note.pitch += 12
            while new_note.pitch > 80:
                new_note.pitch -= 12
            # Slightly higher velocity for lead
            new_note.velocity = min(127, int(new_note.velocity * 1.1))
            lead.append(new_note)
        return lead

    @staticmethod
    def _create_sax_riffs_and_hits(
        melody: List[NoteEvent],
        chords: List[ChordEvent],
        style_config: BasieStyleConfig
    ) -> List[NoteEvent]:
        """
        Create sax section parts: riffs and hits (not constant harmony).

        Basie style: Saxes play riffs and hits, not continuous harmony.
        They often lay out during melody or solo sections.
        """
        sax_notes = []

        # Add section hits at strategic points
        if random.random() < style_config.use_section_hits:
            for i, chord in enumerate(chords):
                # Hit on strong beats (downbeats)
                if i % 4 == 0:  # Every 4 chords
                    hit = BasieRiffGenerator.generate_section_hit(
                        chord,
                        chord.start_time,
                        voicing="octaves",
                        duration=0.3
                    )
                    sax_notes.extend(hit)

        # Add background riffs (sparse)
        if random.random() < style_config.use_riffs:
            for i, chord in enumerate(chords[::2]):  # Every other chord
                if random.random() < style_config.background_density:
                    riff = BasieRiffGenerator.generate_basie_riff(
                        chord,
                        bars=2,
                        riff_style="blues",
                        section="sax"
                    )
                    sax_notes.extend(riff)

        return sax_notes

    @staticmethod
    def _create_brass_hits_and_riffs(
        chords: List[ChordEvent],
        style_config: BasieStyleConfig
    ) -> List[NoteEvent]:
        """
        Create brass section: hits and riffs (Basie signature).

        Brass plays punchy hits and riffs, often in call-response with saxes.
        """
        brass_notes = []

        # More frequent hits for brass (Basie hallmark)
        for i, chord in enumerate(chords):
            # Hits on backbeats and phrase endings
            if i % 2 == 1 or (i + 1) % 8 == 0:  # Offbeats and phrase ends
                if random.random() < style_config.use_section_hits:
                    hit = BasieRiffGenerator.generate_section_hit(
                        chord,
                        chord.start_time,
                        voicing="basic_chord",
                        duration=0.25
                    )
                    brass_notes.extend(hit)

        # Add brass riffs
        for i, chord in enumerate(chords[::4]):  # Every 4th chord
            if random.random() < style_config.use_riffs * 0.7:  # Slightly less than sax
                riff = BasieRiffGenerator.generate_basie_riff(
                    chord,
                    bars=2,
                    riff_style="call_response",
                    section="brass"
                )
                brass_notes.extend(riff)

        return brass_notes

    @staticmethod
    def _create_sparse_piano_comping(
        chords: List[ChordEvent],
        style_config: BasieStyleConfig
    ) -> List[NoteEvent]:
        """
        Create Basie-style sparse piano comping.

        Basie's piano style: MINIMALIST
        - Often plays just one or two notes
        - Strategic placement (not constant)
        - "Comps less, says more"
        - Sometimes silent for long stretches
        """
        piano_notes = []

        for chord in chords:
            # Only comp on some chords (sparse!)
            if random.random() > style_config.piano_density:
                continue  # Lay out (silence)

            # When comping, often just one note or simple voicing
            comp_style = random.choice([
                "one_note",     # Famous Basie "one note" comping
                "two_notes",    # Root and 7th
                "shell",        # Root, 3rd, 7th
            ])

            if comp_style == "one_note":
                # Just play the root or 7th
                pitch = chord.root + 48 + random.choice([0, 10])  # Root or 7th
                pitches = [pitch]
            elif comp_style == "two_notes":
                # Root and 7th
                pitches = [chord.root + 48, chord.root + 58]
            else:
                # Shell voicing: root, 3rd, 7th
                root = chord.root + 48
                third = 4 if 'major' in chord.quality else 3
                pitches = [root, root + third, root + 10]

            # Place on offbeat (Charleston rhythm)
            offbeat_time = chord.start_time + 0.5

            for pitch in pitches:
                note = NoteEvent(
                    start_time=offbeat_time,
                    duration=0.2,  # Short
                    start_tick=int(offbeat_time * 480),
                    duration_ticks=int(0.2 * 480),
                    pitch=pitch,
                    velocity=style_config.base_velocity - 10,  # Softer than horns
                    channel=0,
                    track_idx=20
                )
                piano_notes.append(note)

        return piano_notes

    @staticmethod
    def _create_walking_bass(chords: List[ChordEvent]) -> List[NoteEvent]:
        """
        Create walking bass line (standard for Basie).

        Basie bass: Solid, swinging, foundational.
        """
        bass_notes = []

        for chord in chords:
            num_beats = int(chord.duration)
            if num_beats == 0:
                num_beats = 1

            beat_duration = chord.duration / num_beats

            for beat in range(num_beats):
                time = chord.start_time + beat * beat_duration

                # Walking bass pattern
                if beat == 0:
                    pitch = chord.root + 36  # Root (low)
                elif beat == 1:
                    pitch = chord.root + 36 + 3  # 3rd
                elif beat == 2:
                    pitch = chord.root + 36 + 7  # 5th
                else:
                    # Chromatic approach to next chord's root
                    pitch = chord.root + 36 + 11  # Leading tone

                note = NoteEvent(
                    start_time=time,
                    duration=beat_duration * 0.95,
                    start_tick=int(time * 480),
                    duration_ticks=int(beat_duration * 0.95 * 480),
                    pitch=pitch,
                    velocity=90,
                    channel=1,
                    track_idx=25
                )
                bass_notes.append(note)

        return bass_notes

    @staticmethod
    def _create_basie_drums(
        melody: List[NoteEvent],
        style_config: BasieStyleConfig
    ) -> List[NoteEvent]:
        """
        Create Basie-style drums.

        Characteristics:
        - Swing ride cymbal
        - Hi-hat on 2 & 4
        - FEATHERED BASS DRUM (all four beats, soft) - Signature!
        - Minimal snare (except fills)
        """
        drums = []

        if not melody:
            return drums

        duration = melody[-1].end_time if melody else 32.0
        beats = int(duration)

        for beat in range(beats):
            time = float(beat)

            # Ride cymbal (swing 8ths)
            for eighth in [0, 0.64]:  # Basie swing ratio
                ride = NoteEvent(
                    start_time=time + eighth,
                    duration=0.1,
                    start_tick=int((time + eighth) * 480),
                    duration_ticks=int(0.1 * 480),
                    pitch=51,  # Ride cymbal
                    velocity=75,
                    channel=9,
                    track_idx=30
                )
                drums.append(ride)

            # Hi-hat on 2 and 4 (backbeat)
            if style_config.hi_hat_on_2_and_4 and beat % 2 == 1:
                hihat = NoteEvent(
                    start_time=time,
                    duration=0.1,
                    start_tick=int(time * 480),
                    duration_ticks=int(0.1 * 480),
                    pitch=42,  # Closed hi-hat
                    velocity=100,
                    channel=9,
                    track_idx=30
                )
                drums.append(hihat)

            # FEATHERED BASS DRUM (Basie signature!)
            # All four beats, played softly
            if style_config.feathered_kick:
                kick = NoteEvent(
                    start_time=time,
                    duration=0.1,
                    start_tick=int(time * 480),
                    duration_ticks=int(0.1 * 480),
                    pitch=36,  # Bass drum
                    velocity=45,  # SOFT (feathered)
                    channel=9,
                    track_idx=30
                )
                drums.append(kick)

        return drums

    @staticmethod
    def _create_freddie_green_guitar(
        chords: List[ChordEvent]
    ) -> List[NoteEvent]:
        """
        Create Freddie Green style rhythm guitar.

        Freddie Green: 4-to-the-bar, same velocity, same duration.
        Legendary for providing the rhythmic foundation.
        """
        guitar_notes = []

        for chord in chords:
            num_beats = int(chord.duration)
            if num_beats == 0:
                num_beats = 1

            beat_duration = chord.duration / num_beats

            # Freddie Green voicing: typically 3-note chord on middle strings
            root = chord.root + 52  # Guitar mid register
            if 'major' in chord.quality:
                pitches = [root, root + 4, root + 7]  # Major
            else:
                pitches = [root, root + 3, root + 7]  # Minor

            # 4-to-the-bar
            for beat in range(num_beats):
                time = chord.start_time + beat * beat_duration

                for pitch in pitches:
                    note = NoteEvent(
                        start_time=time,
                        duration=0.2,  # Short, percussive
                        start_tick=int(time * 480),
                        duration_ticks=int(0.2 * 480),
                        pitch=pitch,
                        velocity=70,  # Consistent velocity
                        channel=2,
                        track_idx=22
                    )
                    guitar_notes.append(note)

        return guitar_notes

    @staticmethod
    def create_basie_button_intro(
        first_chord: ChordEvent,
        bars: int = 2
    ) -> List[NoteEvent]:
        """
        Create Basie "button" intro.

        A button is a short, punchy figure (typically 2-4 bars) used as
        intro or ending. Very recognizable Basie signature.
        """
        button_notes = []

        # Simple pattern: hit, rest, hit, rest, hit-hit-hit
        hit_times = [0, 1, 3, 3.5, 3.75]

        for hit_time in hit_times:
            if hit_time >= bars * 4:
                break

            hit = BasieRiffGenerator.generate_section_hit(
                first_chord,
                first_chord.start_time + hit_time,
                voicing="basic_chord",
                duration=0.25
            )
            button_notes.extend(hit)

        return button_notes

    @staticmethod
    def create_basie_shout_chorus(
        melody: List[NoteEvent],
        chords: List[ChordEvent],
        style_config: Optional[BasieStyleConfig] = None
    ) -> Dict[str, List[NoteEvent]]:
        """
        Create Basie-style shout chorus (climactic final chorus).

        Characteristics:
        - Full band in unison or simple harmony
        - Louder dynamics (ff)
        - More section hits
        - Riffs become more intense
        - Piano may play more (but still less than Ellington)
        """
        if style_config is None:
            style_config = BASIE_SHOUT_CHORUS_ONLY

        # Use main arranger but with shout chorus config
        shout = BasieArranger.arrange_in_basie_style(
            melody, chords, style_config
        )

        # Increase velocity for all parts
        for section, notes in shout.items():
            for note in notes:
                note.velocity = min(127, int(note.velocity * 1.15))

        return shout


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compare_basie_vs_ellington(
    arrangement_basie: Dict[str, List[NoteEvent]],
    arrangement_ellington: Dict[str, List[NoteEvent]]
) -> Dict[str, str]:
    """
    Compare Basie vs. Ellington arrangements (for validation).

    Returns:
        Dictionary with comparison metrics
    """
    # Count piano notes
    basie_piano_count = len(arrangement_basie.get('piano', []))
    ellington_piano_count = len(arrangement_ellington.get('piano', []))

    # Count section hits
    basie_hits = len([n for n in arrangement_basie.get('brass_section', [])
                     if n.duration < 0.3])
    ellington_hits = len([n for n in arrangement_ellington.get('brass_section', [])
                         if n.duration < 0.3])

    return {
        "piano_sparseness": f"Basie: {basie_piano_count}, Ellington: {ellington_piano_count}",
        "section_hits": f"Basie: {basie_hits}, Ellington: {ellington_hits}",
        "texture": "Basie: Sparse, Ellington: Dense",
        "harmony": "Basie: Simple, Ellington: Complex",
    }


if __name__ == "__main__":
    # Demo: show Basie arranger capabilities
    print("COUNT BASIE ARRANGER")
    print("=" * 80)
    print()
    print("This module implements Count Basie's arranging style:")
    print()
    print("✓ Riff-based backgrounds")
    print("✓ Punchy section hits")
    print("✓ Sparse piano comping (Basie minimalism)")
    print("✓ Powerful rhythm section (feathered kick, Freddie Green guitar)")
    print("✓ Shout chorus")
    print("✓ Button intro/ending")
    print("✓ Simple, blues-based harmony")
    print()
    print("=" * 80)
    print()
    print("Usage:")
    print("  from styles.basie_arranger import BasieArranger")
    print("  from styles.basie_profile import BASIE_STYLE")
    print()
    print("  arrangement = BasieArranger.arrange_in_basie_style(")
    print("      melody=melody_notes,")
    print("      chords=chord_events,")
    print("      style_config=BASIE_STYLE")
    print("  )")
    print()
    print("=" * 80)
