"""
Dynamic Drum Arranger for Big Band

Form-aware drum arranging with section dynamics, fills, and groove integration.

Features:
- Form-based dynamics (intro, verse, bridge, shout chorus, ending)
- Automatic fill placement at phrase endings
- Groove template integration for authentic feel
- Style-specific arrangements (swing, bebop, Latin)

Research References:
- Buddy Rich "West Side Story" - bebop drum approach with section dynamics
- Louie Bellson - classic swing with varied intensity
- Mel Lewis - modern jazz orchestral with subtle builds

Author: Agent 7 - Drum Pattern & Groove Specialist
Date: 2025
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import from existing modules
from ..midi.midi_constants import GM_DRUM_MAP, PPQN_HIGH_RES
from ..algorithms.rhythm_engine import RhythmNote, GrooveTemplate
from ..algorithms.groove_library import GrooveLibrary, GenreTimingProfiles
from ..generators.form_generator import MusicalForm
from .bigband_drums import BigBandDrumPatterns


# ============================================================================
# Dynamic Drum Arranger
# ============================================================================

class DrumArranger:
    """
    Form-aware drum arranger with dynamic variation.

    Generates drums that respond to musical form:
    - Intro: Soft, sparse
    - A sections: Medium intensity
    - Bridge: Build or contrast
    - Shout chorus: Loud, intense
    - Ending: Soft or sudden cutoff

    Integrates with GrooveTemplateEngine for authentic timing.
    """

    def __init__(self, ppqn: int = PPQN_HIGH_RES):
        """
        Initialize drum arranger.

        Args:
            ppqn: Pulses per quarter note
        """
        self.ppqn = ppqn
        self.patterns = BigBandDrumPatterns()
        self.groove_library = GrooveLibrary(ppqn=ppqn)

    @staticmethod
    def arrange_drums_for_form(
        form: MusicalForm,
        style: str = "swing",  # swing, bebop, latin_afro, latin_bossa
        dynamic_map: Optional[Dict[str, float]] = None,
        ppqn: int = PPQN_HIGH_RES
    ) -> List[RhythmNote]:
        """
        Generate drums with variation per section.

        Args:
            form: MusicalForm object defining structure
            style: Drum style ("swing", "bebop", "latin_afro", "latin_bossa")
            dynamic_map: Optional custom dynamic levels per section
                Example: {"intro": 0.3, "A1": 0.5, "A2": 0.6, "B": 0.7, "A3": 0.9}
            ppqn: Pulses per quarter note

        Returns:
            List of RhythmNote with complete drum arrangement
        """
        # Default dynamic map for AABA form
        if dynamic_map is None:
            dynamic_map = {
                "intro": 0.3,      # Soft, sparse
                "A1": 0.5,         # Medium
                "A2": 0.55,        # Slightly louder
                "B": 0.7,          # Build on bridge
                "A3": 0.9,         # Shout chorus - loud!
                "ending": 0.4      # Soft ending
            }

        all_drums = []
        current_tick = 0

        # Generate intro if present
        if "intro" in dynamic_map:
            intro_drums = DrumArranger._generate_section_drums(
                bars=4,  # Standard 4-bar intro
                style=style,
                intensity=dynamic_map["intro"],
                section_type="intro",
                ppqn=ppqn
            )
            all_drums.extend(DrumArranger._offset_notes(intro_drums, current_tick))
            current_tick += 4 * ppqn * 4  # 4 bars

        # Generate sections based on form
        # TODO: Integrate with actual MusicalForm structure
        # For now, generate standard AABA (8 bars each)
        sections = [
            ("A1", 8, "verse"),
            ("A2", 8, "verse"),
            ("B", 8, "bridge"),
            ("A3", 8, "shout")
        ]

        for section_name, bars, section_type in sections:
            intensity = dynamic_map.get(section_name, 0.5)

            section_drums = DrumArranger._generate_section_drums(
                bars=bars,
                style=style,
                intensity=intensity,
                section_type=section_type,
                ppqn=ppqn
            )

            all_drums.extend(DrumArranger._offset_notes(section_drums, current_tick))
            current_tick += bars * ppqn * 4

        # Generate ending if present
        if "ending" in dynamic_map:
            ending_drums = DrumArranger._generate_section_drums(
                bars=4,
                style=style,
                intensity=dynamic_map["ending"],
                section_type="ending",
                ppqn=ppqn
            )
            all_drums.extend(DrumArranger._offset_notes(ending_drums, current_tick))

        return all_drums

    @staticmethod
    def _generate_section_drums(
        bars: int,
        style: str,
        intensity: float,
        section_type: str,
        ppqn: int
    ) -> List[RhythmNote]:
        """
        Generate drums for a single section.

        Args:
            bars: Number of bars
            style: Drum style
            intensity: Dynamic intensity (0.0-1.0)
            section_type: "intro", "verse", "bridge", "shout", "ending"
            ppqn: Pulses per quarter note

        Returns:
            List of RhythmNote for this section
        """
        drums = []

        if style == "swing":
            # Swing ride cymbal
            ride = BigBandDrumPatterns.swing_ride(ppqn=ppqn, bars=bars, swing_ratio=0.62)
            # Scale velocity by intensity
            ride = DrumArranger._scale_velocity(ride, intensity)
            drums.extend(ride)

            # Hi-hat on 2 & 4
            hihat = BigBandDrumPatterns.hihat_2_4(ppqn=ppqn, bars=bars)
            hihat = DrumArranger._scale_velocity(hihat, intensity)
            drums.extend(hihat)

            # Feathered kick
            kick = BigBandDrumPatterns.feathered_kick(ppqn=ppqn, bars=bars)
            # Kick is always soft, but scale slightly
            kick = DrumArranger._scale_velocity(kick, intensity * 0.7 + 0.3)
            drums.extend(kick)

        elif style == "bebop":
            # Bebop ride
            ride = BigBandDrumPatterns.bebop_ride(ppqn=ppqn, bars=bars)
            ride = DrumArranger._scale_velocity(ride, intensity)
            drums.extend(ride)

            # Hi-hat on 2 & 4
            hihat = BigBandDrumPatterns.hihat_2_4(ppqn=ppqn, bars=bars)
            hihat = DrumArranger._scale_velocity(hihat, intensity)
            drums.extend(hihat)

            # Bebop bombs (syncopated kick)
            kick = BigBandDrumPatterns.bebop_kick(ppqn=ppqn, bars=bars)
            kick = DrumArranger._scale_velocity(kick, intensity)
            drums.extend(kick)

        elif style == "latin_afro":
            # Afro-Cuban cowbell
            cowbell = BigBandDrumPatterns.afro_cuban_bell(ppqn=ppqn, bars=bars)
            cowbell = DrumArranger._scale_velocity(cowbell, intensity)
            drums.extend(cowbell)

            # Samba surdo (bass drum)
            surdo = BigBandDrumPatterns.samba_surdo(ppqn=ppqn, bars=bars)
            surdo = DrumArranger._scale_velocity(surdo, intensity)
            drums.extend(surdo)

            # Hi-hat pattern
            hihat = BigBandDrumPatterns.hihat_all_beats(ppqn=ppqn, bars=bars)
            hihat = DrumArranger._scale_velocity(hihat, intensity * 0.8)
            drums.extend(hihat)

        elif style == "latin_bossa":
            # Bossa nova ride
            ride = BigBandDrumPatterns.bossa_ride(ppqn=ppqn, bars=bars)
            ride = DrumArranger._scale_velocity(ride, intensity * 0.8)  # Bossa is subtle
            drums.extend(ride)

            # Cross-stick pattern
            sixteenth = ppqn // 4
            bar_length = ppqn * 4
            rim_pattern = [4, 10, 14]  # 16th note positions

            for bar in range(bars):
                bar_offset = bar * bar_length
                for pos in rim_pattern:
                    drums.append(RhythmNote(
                        tick=bar_offset + pos * sixteenth,
                        duration=sixteenth,
                        velocity=int(60 * intensity),
                        pitch=GM_DRUM_MAP['SIDE_STICK']
                    ))

            # Subtle kick
            kick_pattern = [0, 6, 12]  # Bossa kick pattern
            for bar in range(bars):
                bar_offset = bar * bar_length
                for pos in kick_pattern:
                    drums.append(RhythmNote(
                        tick=bar_offset + pos * sixteenth,
                        duration=sixteenth * 2,
                        velocity=int(70 * intensity),
                        pitch=GM_DRUM_MAP['BASS_DRUM_1']
                    ))

        # Add section-specific variations
        if section_type == "intro":
            # Sparse intro - remove some notes
            drums = DrumArranger._thin_pattern(drums, keep_ratio=0.6)

        elif section_type == "shout":
            # Shout chorus - add accents
            drums = DrumArranger._add_accents(drums, accent_increase=20)

        return drums

    @staticmethod
    def add_fills_at_phrase_endings(
        drums: List[RhythmNote],
        phrase_length_bars: int = 4,
        fill_length_beats: int = 2,
        intensity: float = 0.7,
        ppqn: int = PPQN_HIGH_RES
    ) -> List[RhythmNote]:
        """
        Insert fills at phrase endings (bar 4, 8, 12, etc.).

        Args:
            drums: Existing drum pattern
            phrase_length_bars: Phrase length (typically 4 or 8 bars)
            fill_length_beats: Fill length in beats (1, 2, or 4)
            intensity: Fill intensity (0.0-1.0)
            ppqn: Pulses per quarter note

        Returns:
            Drum pattern with fills added
        """
        bar_length = ppqn * 4
        phrase_length_ticks = phrase_length_bars * bar_length
        fill_length_ticks = fill_length_beats * ppqn

        # Find phrase ending positions
        max_tick = max(note.tick for note in drums) if drums else 0
        num_phrases = int(max_tick / phrase_length_ticks)

        all_drums = drums.copy()

        for phrase_num in range(1, num_phrases + 1):
            # Fill starts at: (phrase_length - fill_length) ticks before phrase end
            fill_start_tick = phrase_num * phrase_length_ticks - fill_length_ticks

            # Remove conflicting notes in fill area
            all_drums = [
                note for note in all_drums
                if not (fill_start_tick <= note.tick < fill_start_tick + fill_length_ticks)
            ]

            # Generate fill
            fill_type = "snare" if phrase_num % 2 == 1 else "toms"
            fill = BigBandDrumPatterns.generate_fill(
                ppqn=ppqn,
                length_beats=fill_length_beats,
                intensity=intensity,
                target_instrument=fill_type
            )

            # Offset fill to correct position
            fill = DrumArranger._offset_notes(fill, fill_start_tick)
            all_drums.extend(fill)

        # Sort by tick
        return sorted(all_drums, key=lambda n: n.tick)

    @staticmethod
    def apply_groove_template(
        drums: List[RhythmNote],
        template_name: Optional[str] = None,
        genre: str = "jazz_bebop",
        ppqn: int = PPQN_HIGH_RES
    ) -> List[RhythmNote]:
        """
        Apply authentic groove timing from template.

        Uses groove_library.py GrooveTemplate for realistic timing.

        Args:
            drums: Drum pattern to apply groove to
            template_name: Optional specific groove template ("purdie_shuffle", etc.)
            genre: Genre timing profile ("jazz_bebop", "jazz_ballad", "funk", etc.)
            ppqn: Pulses per quarter note

        Returns:
            Drum pattern with groove timing applied
        """
        library = GrooveLibrary(ppqn=ppqn)

        # Get genre timing profile
        profile = library.get_genre_profile(genre)

        if profile is None:
            # No profile found, return drums unchanged
            return drums

        # Apply timing profile to drums
        grooved_drums = []

        for note in drums:
            new_note = RhythmNote(
                tick=note.tick,
                duration=note.duration,
                velocity=note.velocity,
                pitch=note.pitch
            )

            # Apply timing deviation (laid back/rushing)
            # Convert early_late_bias from ms to ticks
            # At 120 BPM: 1 beat = 500ms, 1 tick (960 ppqn) = 0.52ms
            # bias_ticks = (profile.early_late_bias / 0.52)
            timing_offset = int(profile.early_late_bias * 2)  # Approximate conversion

            # Add random jitter based on deviation_std
            import random
            jitter = int(random.gauss(0, profile.deviation_std_ms / 2))

            new_note.tick = max(0, note.tick + timing_offset + jitter)

            # Apply velocity variation
            vel_variation = int(random.gauss(0, note.velocity * profile.velocity_variation / 4))
            new_note.velocity = max(1, min(127, note.velocity + vel_variation))

            grooved_drums.append(new_note)

        return sorted(grooved_drums, key=lambda n: n.tick)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    @staticmethod
    def _scale_velocity(notes: List[RhythmNote], scale: float) -> List[RhythmNote]:
        """Scale velocity of all notes by factor."""
        scaled = []
        for note in notes:
            new_note = RhythmNote(
                tick=note.tick,
                duration=note.duration,
                velocity=int(note.velocity * scale),
                pitch=note.pitch
            )
            new_note.velocity = max(1, min(127, new_note.velocity))
            scaled.append(new_note)
        return scaled

    @staticmethod
    def _offset_notes(notes: List[RhythmNote], offset: int) -> List[RhythmNote]:
        """Offset all notes by tick amount."""
        return [
            RhythmNote(
                tick=note.tick + offset,
                duration=note.duration,
                velocity=note.velocity,
                pitch=note.pitch
            )
            for note in notes
        ]

    @staticmethod
    def _thin_pattern(notes: List[RhythmNote], keep_ratio: float) -> List[RhythmNote]:
        """Remove some notes for sparse sections."""
        import random
        return [note for note in notes if random.random() < keep_ratio]

    @staticmethod
    def _add_accents(notes: List[RhythmNote], accent_increase: int) -> List[RhythmNote]:
        """Add accents to notes for shout chorus."""
        accented = []
        for note in notes:
            new_note = RhythmNote(
                tick=note.tick,
                duration=note.duration,
                velocity=min(127, note.velocity + accent_increase),
                pitch=note.pitch
            )
            accented.append(new_note)
        return accented

    @staticmethod
    def convert_to_note_events(
        rhythm_notes: List[RhythmNote],
        tempo: int = 120
    ) -> List['NoteEvent']:
        """
        Convert RhythmNote list to NoteEvent list for MIDI export.

        Args:
            rhythm_notes: List of RhythmNote
            tempo: Tempo in BPM

        Returns:
            List of NoteEvent objects
        """
        # Import here to avoid circular dependency
        from ..midi.midi_types import NoteEvent

        note_events = []

        for rn in rhythm_notes:
            # Convert ticks to beats (assuming ppqn in rn.tick)
            # This is approximate - should use actual ppqn from context
            ppqn = 960  # TODO: Get from context
            start_time = rn.tick / ppqn
            duration = rn.duration / ppqn

            note_event = NoteEvent(
                start_time=start_time,
                duration=duration,
                start_tick=rn.tick,
                duration_ticks=rn.duration,
                pitch=rn.pitch,
                velocity=rn.velocity,
                channel=9,  # Drum channel
                track_idx=30  # Drum track
            )
            note_events.append(note_event)

        return note_events


# ============================================================================
# Example Usage & Tests
# ============================================================================

if __name__ == "__main__":
    """Example usage of DrumArranger"""

    print("=" * 70)
    print("DRUM ARRANGER - Examples")
    print("=" * 70)

    ppqn = 960
    arranger = DrumArranger(ppqn=ppqn)

    # Example 1: Generate swing drums for AABA form
    print("\n1. Swing Drums for 32-bar AABA:")

    # Create dynamic map
    dynamic_map = {
        "intro": 0.3,
        "A1": 0.5,
        "A2": 0.6,
        "B": 0.7,
        "A3": 0.9,  # Shout chorus
        "ending": 0.4
    }

    # Generate (note: form parameter would be actual MusicalForm object)
    swing_drums = DrumArranger.arrange_drums_for_form(
        form=None,  # TODO: Create actual MusicalForm
        style="swing",
        dynamic_map=dynamic_map,
        ppqn=ppqn
    )
    print(f"   Total notes: {len(swing_drums)}")

    # Example 2: Add fills
    print("\n2. Adding Fills at 4-bar Phrases:")
    drums_with_fills = DrumArranger.add_fills_at_phrase_endings(
        drums=swing_drums,
        phrase_length_bars=4,
        fill_length_beats=2,
        intensity=0.7,
        ppqn=ppqn
    )
    print(f"   Total notes with fills: {len(drums_with_fills)}")
    fill_count = len(drums_with_fills) - len(swing_drums)
    print(f"   Fills added: ~{fill_count} notes")

    # Example 3: Apply groove template
    print("\n3. Applying Jazz Bebop Groove Template:")
    grooved_drums = DrumArranger.apply_groove_template(
        drums=swing_drums[:20],  # Just first 20 notes for demo
        genre="jazz_bebop",
        ppqn=ppqn
    )
    print(f"   Original first tick: {swing_drums[0].tick}")
    print(f"   Grooved first tick: {grooved_drums[0].tick}")
    print(f"   Timing offset applied: {grooved_drums[0].tick - swing_drums[0].tick} ticks")

    # Example 4: Bebop style
    print("\n4. Bebop Drums for AABA:")
    bebop_drums = DrumArranger.arrange_drums_for_form(
        form=None,
        style="bebop",
        dynamic_map=dynamic_map,
        ppqn=ppqn
    )
    print(f"   Total notes: {len(bebop_drums)}")

    # Example 5: Latin Afro-Cuban
    print("\n5. Latin Afro-Cuban Drums:")
    latin_drums = DrumArranger.arrange_drums_for_form(
        form=None,
        style="latin_afro",
        dynamic_map={"A1": 0.8},
        ppqn=ppqn
    )
    print(f"   Total notes: {len(latin_drums)}")

    # Example 6: Bossa Nova
    print("\n6. Bossa Nova Drums:")
    bossa_drums = DrumArranger.arrange_drums_for_form(
        form=None,
        style="latin_bossa",
        dynamic_map={"A1": 0.6},  # Bossa is subtle
        ppqn=ppqn
    )
    print(f"   Total notes: {len(bossa_drums)}")

    print("\n" + "=" * 70)
    print("DrumArranger ready for integration!")
    print("=" * 70)
    print("\nFeatures:")
    print("  • Form-aware dynamics (intro, verse, bridge, shout, ending)")
    print("  • Automatic fill placement at phrase endings")
    print("  • Groove template integration (timing profiles)")
    print("  • Multiple styles (swing, bebop, Latin Afro-Cuban, bossa)")
    print("  • Dynamic scaling per section")
    print("  • Integration with GrooveLibrary")
    print("=" * 70)
