#!/usr/bin/env python3
"""
Improved Melody Harmonizer - Context-aware chord voicings

Harmonizes melodies using specific chord progressions, prioritizing chord tones
and controlling voice spacing for professional-sounding arrangements.
"""

import mido
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import tempfile
import random
from chord_progression_generator import ScaleContext, _parse_chord_root, CHORD_LIBRARY
import math

# ============================================================================
# CHORD ANALYSIS
# ============================================================================

class ChordAnalyzer:
    """Analyzes chords and provides available notes"""

    @staticmethod
    def detect_secondary_dominant(chord_name: str, scale_context: Optional[ScaleContext] = None) -> Optional[str]:
        """
        Detect if chord is a secondary dominant (V7/x) or modal interchange.
        Returns target chord function (e.g., 'V/V', 'V/vi') or None.
        """
        if not scale_context:
            return None

        # Check if it's a dominant 7th chord (has '7' and is major quality)
        is_dominant = '7' in chord_name and 'm7' not in chord_name.lower() and 'maj7' not in chord_name.lower()

        if not is_dominant:
            return None

        root = _parse_chord_root(chord_name)
        root_pc = root % 12
        key_root_pc = scale_context.root_note % 12

        # Calculate interval from key root
        interval = (root_pc - key_root_pc) % 12

        # Map to secondary dominant function
        if scale_context.scale_type == 'major':
            secondary_map = {
                2: 'V/V',      # D7 in C = V7/V
                4: 'V/vi',     # E7 in C = V7/vi
                7: 'V',        # G7 in C = V
                9: 'V/ii',     # A7 in C = V7/ii
                11: 'V/iii',   # B7 in C = V7/iii
            }
        else:  # minor
            secondary_map = {
                2: 'V/iv',     # D7 in Cm = V7/iv
                4: 'V/VI',     # E7 in Cm = V7/bVI
                7: 'V',        # G7 in Cm = V
                9: 'V/VII',    # A7 in Cm = V7/bVII
                10: 'V/III',   # Bb7 in Cm = V7/bIII
            }

        return secondary_map.get(interval, None)

    @staticmethod
    def detect_melodic_minor_mode(chord_name: str, scale_context: Optional[ScaleContext] = None) -> Optional[str]:
        """
        Detect if chord suggests melodic minor modal interchange.
        Returns mode name or None.
        """
        if not scale_context:
            return None

        root = _parse_chord_root(chord_name)
        root_pc = root % 12
        key_root_pc = scale_context.root_note % 12
        interval = (root_pc - key_root_pc) % 12

        # Melodic minor modes by distinctive chord qualities
        # I - Im(maj7)
        if interval == 0 and 'mmaj7' in chord_name.lower():
            return 'melodic_minor_i'

        # IV - Lydian Dominant (X7 with #11)
        if '7' in chord_name and 'm7' not in chord_name.lower() and 'maj7' not in chord_name.lower():
            # Could be Lydian Dominant (mode IV of melodic minor)
            # This would be a dominant 7 chord a perfect 4th above tonic
            if interval == 5:
                return 'lydian_dominant'

        # VII - Altered dominant
        if 'alt' in chord_name.lower():
            return 'altered'

        return None

    @staticmethod
    def get_chord_function(chord_name: str, scale_context: Optional[ScaleContext] = None) -> str:
        """
        Determine the function of a chord within a key.
        Returns: 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', or 'unknown'
        """
        if not scale_context:
            return 'unknown'

        root = _parse_chord_root(chord_name)
        root_pc = root % 12
        key_root_pc = scale_context.root_note % 12

        # Calculate scale degree (0-11 semitones from key root)
        interval = (root_pc - key_root_pc) % 12

        # Map interval to scale degree
        if scale_context.scale_type == 'minor':
            degree_map = {
                0: 'i',      # Root
                2: 'ii',     # 2nd (diminished in minor)
                3: 'bIII',   # b3
                5: 'iv',     # 4th
                7: 'v',      # 5th (or V if major)
                8: 'bVI',    # b6
                10: 'bVII',  # b7
            }
        else:  # major
            degree_map = {
                0: 'I',      # Root
                2: 'ii',     # 2nd
                4: 'iii',    # 3rd
                5: 'IV',     # 4th
                7: 'V',      # 5th
                9: 'vi',     # 6th
                11: 'vii',   # 7th
            }

        return degree_map.get(interval, 'unknown')

    @staticmethod
    def parse_chord(chord_name: str, use_extensions: bool = True, scale_context: Optional[ScaleContext] = None, next_chord_name: Optional[str] = None) -> Dict[str, any]:
        """
        Parse chord name and return chord tones + available extensions.

        Args:
            chord_name: Chord name (e.g., 'C', 'Dm', 'G7')
            use_extensions: If True, calculate and include extensions
            scale_context: Key context for determining proper extensions

        Returns:
            Dict with 'root', 'chord_tones', 'extensions', 'avoid_notes'
        """
        # Get base notes from library
        if chord_name in CHORD_LIBRARY:
            base_notes = CHORD_LIBRARY[chord_name].copy()
        else:
            # Fallback to C major
            base_notes = [60, 64, 67]

        root = base_notes[0]

        # Separate into chord tones (prioritized by order)
        # Triad: root, 3rd, 5th (3 notes)
        # 7th chord: root, 3rd, 5th, 7th (4 notes)
        chord_tones = base_notes[:min(4, len(base_notes))]

        # Extensions from library
        extensions_from_lib = base_notes[4:] if len(base_notes) > 4 else []

        # Get chord function for context-aware extension priority
        chord_function = ChordAnalyzer.get_chord_function(chord_name, scale_context)

        # Check for secondary dominants and melodic minor modes
        secondary_function = ChordAnalyzer.detect_secondary_dominant(chord_name, scale_context)
        melodic_minor_mode = ChordAnalyzer.detect_melodic_minor_mode(chord_name, scale_context)

        # Calculate sequential extensions: 7th → 9th → 11th/♯11 → 13th
        extensions = []
        avoid_extensions = []  # Extensions to avoid based on function
        extension_priority = {}  # Priority order for extensions (lower = higher priority)
        chord_tone_priority = {}  # Priority for chord tones (3rd, 7th boosted on dominants)

        # Define avoid rules and priorities based on chord function (MODAL APPROACH)
        # Priority: 0=highest, 1=high, 2=medium, 3=low, 4=avoid

        if chord_function == 'I':  # Ionian (Major Tonic)
            # Extensions: 9 (high), 13 (high)
            # Avoid: 11 (half-step clash with major 3rd)
            extension_priority[(root + 10) % 12] = 0  # m7
            extension_priority[(root + 11) % 12] = 0  # M7
            extension_priority[(root + 14) % 12] = 1  # 9
            extension_priority[(root + 2) % 12] = 1   # 9 (octave down)
            extension_priority[(root + 21) % 12] = 1  # 13
            extension_priority[(root + 9) % 12] = 1   # 13 (octave down)
            avoid_extensions.append((root + 5) % 12)   # 11 - avoid
            avoid_extensions.append((root + 17) % 12)  # 11 octave up

        elif chord_function == 'ii':  # Dorian (Minor ii)
            # Extensions: 9 (high), 11 (high), 13 (LOW/CONTEXTUAL)
            # 13 is last choice and avoided before V chords
            extension_priority[(root + 10) % 12] = 0  # m7
            extension_priority[(root + 14) % 12] = 1  # 9
            extension_priority[(root + 2) % 12] = 1   # 9 (octave down)
            extension_priority[(root + 17) % 12] = 2  # 11
            extension_priority[(root + 5) % 12] = 2   # 11 (octave down)
            extension_priority[(root + 21) % 12] = 3  # 13 - LAST CHOICE
            extension_priority[(root + 9) % 12] = 3   # 13 (octave down) - LAST CHOICE

            # CONTEXTUAL: Avoid 13 if next chord is V
            if next_chord_name:
                next_function = ChordAnalyzer.get_chord_function(next_chord_name, scale_context)
                if next_function in ['V', 'v']:
                    # Completely avoid 13 before V chord
                    avoid_extensions.append((root + 21) % 12)  # 13
                    avoid_extensions.append((root + 9) % 12)   # 13 (octave down)

        elif chord_function == 'iii':  # Phrygian (Minor iii)
            # Extensions: 11 (high), b13 (medium)
            # Avoid: b9 (too dark/dissonant)
            extension_priority[(root + 10) % 12] = 0  # m7
            extension_priority[(root + 17) % 12] = 1  # 11
            extension_priority[(root + 5) % 12] = 1   # 11 (octave down)
            extension_priority[(root + 20) % 12] = 2  # b13
            extension_priority[(root + 8) % 12] = 2   # b13 (octave down)
            avoid_extensions.append((root + 13) % 12)  # b9 - avoid
            avoid_extensions.append((root + 1) % 12)   # b9 (octave down)

        elif chord_function == 'IV':  # Lydian (Major IV)
            # Extensions: 9 (high), #11 (high), 13 (high)
            # No avoid notes - all extensions sound great
            extension_priority[(root + 10) % 12] = 0  # m7
            extension_priority[(root + 11) % 12] = 0  # M7
            extension_priority[(root + 14) % 12] = 1  # 9
            extension_priority[(root + 2) % 12] = 1   # 9 (octave down)
            extension_priority[(root + 18) % 12] = 1  # #11
            extension_priority[(root + 6) % 12] = 1   # #11 (octave down)
            extension_priority[(root + 21) % 12] = 1  # 13
            extension_priority[(root + 9) % 12] = 1   # 13 (octave down)

        elif chord_function == 'V' or chord_function == 'v' or secondary_function:  # Dominant V
            # CRITICAL: Boost 3rd and 7th priority for voice leading
            root_pc = root % 12
            third_pc = (root_pc + 4) % 12  # Major 3rd
            seventh_pc = (root_pc + 10) % 12  # Minor 7th (dominant)

            # BOOST chord tone priority for 3rd and 7th
            chord_tone_priority[third_pc] = 0   # Highest priority - 3rd defines major quality
            chord_tone_priority[seventh_pc] = 0  # Highest priority - 7th defines dominant function
            chord_tone_priority[root_pc] = 1     # Root is lower priority than 3/7
            chord_tone_priority[(root_pc + 7) % 12] = 2  # 5th is lowest priority

            extension_priority[(root + 10) % 12] = 0  # m7 (dominant 7th)

            # V in MINOR: b9, b13 (Phrygian dominant from harmonic minor)
            # V in MAJOR: natural 9, natural 13 (Mixolydian)
            if scale_context and scale_context.scale_type == 'minor':
                # V in minor - Phrygian dominant (5th mode of harmonic minor)
                extension_priority[(root + 13) % 12] = 1  # b9
                extension_priority[(root + 1) % 12] = 1   # b9 (octave down)
                extension_priority[(root + 20) % 12] = 1  # b13
                extension_priority[(root + 8) % 12] = 1   # b13 (octave down)
            else:
                # V in major - Mixolydian
                extension_priority[(root + 14) % 12] = 1  # 9
                extension_priority[(root + 2) % 12] = 1   # 9 (octave down)
                extension_priority[(root + 21) % 12] = 1  # 13
                extension_priority[(root + 9) % 12] = 1   # 13 (octave down)

            avoid_extensions.append((root + 5) % 12)   # 11 - avoid
            avoid_extensions.append((root + 17) % 12)  # 11 octave up

        elif chord_function == 'vi' or chord_function == 'i':  # Aeolian (Minor vi/i)
            # Extensions: 9 (high), 11 (high)
            # Avoid: b13 (though sometimes used for color)
            extension_priority[(root + 10) % 12] = 0  # m7
            extension_priority[(root + 14) % 12] = 1  # 9
            extension_priority[(root + 2) % 12] = 1   # 9 (octave down)
            extension_priority[(root + 17) % 12] = 1  # 11
            extension_priority[(root + 5) % 12] = 1   # 11 (octave down)
            avoid_extensions.append((root + 20) % 12)  # b13 - avoid
            avoid_extensions.append((root + 8) % 12)   # b13 (octave down)

        elif chord_function == 'vii':  # Locrian (Half-Diminished vii)
            # Extensions: 11 (high), b13 (medium)
            # Avoid: b9 (too dissonant)
            extension_priority[(root + 10) % 12] = 0  # m7
            extension_priority[(root + 17) % 12] = 1  # 11
            extension_priority[(root + 5) % 12] = 1   # 11 (octave down)
            extension_priority[(root + 20) % 12] = 2  # b13
            extension_priority[(root + 8) % 12] = 2   # b13 (octave down)
            avoid_extensions.append((root + 13) % 12)  # b9 - avoid
            avoid_extensions.append((root + 1) % 12)   # b9 (octave down)

        elif chord_function == 'iv':  # Minor iv (in minor keys)
            # Similar to Dorian but more subdominant function
            extension_priority[(root + 10) % 12] = 0  # m7
            extension_priority[(root + 14) % 12] = 1  # 9
            extension_priority[(root + 2) % 12] = 1   # 9 (octave down)
            extension_priority[(root + 17) % 12] = 2  # 11
            extension_priority[(root + 5) % 12] = 2   # 11 (octave down)
            avoid_extensions.append((root + 20) % 12)  # b13 - avoid
            avoid_extensions.append((root + 8) % 12)   # b13 (octave down)

        elif chord_function == 'bIII':  # bIII in minor (Lydian function)
            # Treat like IV - Lydian extensions
            extension_priority[(root + 11) % 12] = 0  # M7
            extension_priority[(root + 14) % 12] = 1  # 9
            extension_priority[(root + 2) % 12] = 1   # 9 (octave down)
            extension_priority[(root + 18) % 12] = 1  # #11
            extension_priority[(root + 6) % 12] = 1   # #11 (octave down)
            extension_priority[(root + 21) % 12] = 1  # 13
            extension_priority[(root + 9) % 12] = 1   # 13 (octave down)

        elif chord_function == 'bVI':  # bVI in minor
            # Lydian-like extensions
            extension_priority[(root + 11) % 12] = 0  # M7
            extension_priority[(root + 14) % 12] = 1  # 9
            extension_priority[(root + 2) % 12] = 1   # 9 (octave down)
            extension_priority[(root + 18) % 12] = 1  # #11
            extension_priority[(root + 21) % 12] = 1  # 13

        elif chord_function == 'bVII':  # bVII in minor (Mixolydian)
            # Dominant-like extensions
            extension_priority[(root + 10) % 12] = 0  # m7
            extension_priority[(root + 14) % 12] = 1  # 9
            extension_priority[(root + 2) % 12] = 1   # 9 (octave down)
            extension_priority[(root + 21) % 12] = 1  # 13
            extension_priority[(root + 9) % 12] = 1   # 13 (octave down)
            avoid_extensions.append((root + 5) % 12)   # 11 - avoid
            avoid_extensions.append((root + 17) % 12)  # 11 octave up

        # MELODIC MINOR MODE EXTENSIONS
        if melodic_minor_mode == 'melodic_minor_i':  # Im(maj7)
            # Extensions: 9, 11, 13 - no avoid notes
            extension_priority[(root + 11) % 12] = 0  # maj7
            extension_priority[(root + 14) % 12] = 1  # 9
            extension_priority[(root + 2) % 12] = 1   # 9 (octave down)
            extension_priority[(root + 17) % 12] = 1  # 11
            extension_priority[(root + 5) % 12] = 1   # 11 (octave down)
            extension_priority[(root + 21) % 12] = 1  # 13
            extension_priority[(root + 9) % 12] = 1   # 13 (octave down)

        elif melodic_minor_mode == 'lydian_dominant':  # IV7 (Lydian Dominant)
            # Extensions: 9, #11, 13 - no avoid notes (overtone scale)
            extension_priority[(root + 10) % 12] = 0  # m7
            extension_priority[(root + 14) % 12] = 1  # 9
            extension_priority[(root + 2) % 12] = 1   # 9 (octave down)
            extension_priority[(root + 18) % 12] = 1  # #11
            extension_priority[(root + 6) % 12] = 1   # #11 (octave down)
            extension_priority[(root + 21) % 12] = 1  # 13
            extension_priority[(root + 9) % 12] = 1   # 13 (octave down)

        elif melodic_minor_mode == 'altered':  # VII7alt (Altered/Super Locrian)
            # Extensions: b9, #9, #11 (b5), b13 (#5) - all alterations
            extension_priority[(root + 10) % 12] = 0  # m7
            extension_priority[(root + 13) % 12] = 1  # b9
            extension_priority[(root + 1) % 12] = 1   # b9 (octave down)
            extension_priority[(root + 15) % 12] = 1  # #9
            extension_priority[(root + 3) % 12] = 1   # #9 (octave down)
            extension_priority[(root + 18) % 12] = 1  # #11/b5
            extension_priority[(root + 6) % 12] = 1   # #11/b5 (octave down)
            extension_priority[(root + 20) % 12] = 1  # b13/#5
            extension_priority[(root + 8) % 12] = 1   # b13/#5 (octave down)

        if use_extensions:
            # Start with library extensions
            extensions.extend(extensions_from_lib)

            # Determine if this chord is diatonic to the key
            is_tonic = False
            is_dominant_function = (chord_function in ['V', 'v'] or secondary_function is not None)

            if scale_context:
                # Check if chord root matches key root
                root_pc = root % 12
                key_root_pc = scale_context.root_note % 12
                is_tonic = (root_pc == key_root_pc)

            # Add 7th if not present (foundational extension)
            # Use maj7 if tonic in major key, or if explicitly maj7
            if is_tonic and scale_context and scale_context.scale_type == 'major':
                seventh = root + 11  # Major 7th for tonic in major key
            elif 'maj7' in chord_name.lower():
                seventh = root + 11  # Major 7th
            else:
                seventh = root + 10  # Minor 7th (dominant 7)

            if seventh not in chord_tones and seventh not in extensions:
                extensions.insert(0, seventh)  # 7th is first priority extension

            # Add 9th (second priority)
            # For dominant chords: depends on key type
            # V in MAJOR: natural 9 (Mixolydian)
            # V in MINOR: b9 (Phrygian dominant from harmonic minor)
            if is_dominant_function:
                if scale_context and scale_context.scale_type == 'minor':
                    # V in minor gets b9 (Phrygian dominant)
                    ninth = root + 13  # b9
                else:
                    # V in major gets natural 9 (Mixolydian)
                    ninth = root + 14  # Major 9th
                if ninth not in extensions:
                    extensions.append(ninth)
            elif scale_context:
                # Calculate diatonic 9th based on key for non-dominant chords
                ninth_candidates = [root + 13, root + 14]  # Minor 9, Major 9
                for ninth in ninth_candidates:
                    if scale_context.is_diatonic(ninth) and ninth not in extensions:
                        extensions.append(ninth)
                        break
            else:
                ninth = root + 14  # Major 9th default
                if ninth not in extensions:
                    extensions.append(ninth)

            # Add 11th or ♯11 (third priority) - must be diatonic to key
            if scale_context:
                # Check for diatonic 11th or ♯11
                eleven_candidates = [root + 17, root + 18]  # Natural 11, ♯11
                for eleven in eleven_candidates:
                    if scale_context.is_diatonic(eleven) and eleven not in extensions:
                        extensions.append(eleven)
                        break
            else:
                # Default behavior without key context
                if 'maj' in chord_name.lower() and '7' in chord_name:
                    sharp_eleven = root + 18
                    if sharp_eleven not in extensions:
                        extensions.append(sharp_eleven)
                else:
                    eleven = root + 17
                    if eleven not in extensions:
                        extensions.append(eleven)

            # Add 13th (fourth priority)
            # For dominant chords: depends on key type
            # V in MAJOR: major 13 (Mixolydian)
            # V in MINOR: b13 (Phrygian dominant from harmonic minor)
            if is_dominant_function:
                if scale_context and scale_context.scale_type == 'minor':
                    # V in minor gets b13 (Phrygian dominant)
                    thirteenth = root + 20  # b13
                else:
                    # V in major gets major 13 (Mixolydian)
                    thirteenth = root + 21  # Major 13th
                thirteenth_pc = thirteenth % 12
                if thirteenth_pc not in avoid_extensions and thirteenth not in extensions:
                    extensions.append(thirteenth)
            elif scale_context:
                thirteenth_candidates = [root + 20, root + 21]  # Minor 13, Major 13
                for thirteenth in thirteenth_candidates:
                    thirteenth_pc = thirteenth % 12
                    # Skip if in avoid list
                    if thirteenth_pc in avoid_extensions:
                        continue
                    if scale_context.is_diatonic(thirteenth):
                        # Check if this is the leading tone (7th scale degree)
                        key_root_pc = scale_context.root_note % 12
                        # Leading tone is 11 semitones above root in major
                        if scale_context.scale_type == 'major' and (thirteenth_pc == (key_root_pc + 11) % 12):
                            continue  # Skip leading tone as 13th
                        if thirteenth not in extensions:
                            extensions.append(thirteenth)
                            break
            else:
                thirteenth = root + 21
                thirteenth_pc = thirteenth % 12
                if thirteenth_pc not in avoid_extensions and thirteenth not in extensions:
                    extensions.append(thirteenth)

        # Avoid notes (notes NOT in the chord and not diatonic)
        avoid_notes = []

        # Add function-based avoid extensions to avoid_notes
        avoid_notes.extend(avoid_extensions)

        # Avoid 4th for major triads (without 7th)
        if ('maj' in chord_name.lower() or ('m' not in chord_name.lower() and '7' not in chord_name.lower())):
            if '7' not in chord_name:
                avoid_pc = (root + 5) % 12
                if avoid_pc not in avoid_notes:
                    avoid_notes.append(avoid_pc)  # Perfect 4th

        # Also avoid non-diatonic notes if we have key context
        # BUT: Don't avoid chord tones even if they're non-diatonic (e.g., A in F7 in Bb minor)
        if scale_context:
            # Get all chord tone pitch classes to protect them
            chord_tone_pcs = set([n % 12 for n in chord_tones])

            for pc in range(12):
                test_note = (root // 12) * 12 + pc
                # Only avoid if: not diatonic AND not a chord tone AND not already in avoid list
                if not scale_context.is_diatonic(test_note):
                    if pc not in avoid_notes and pc not in chord_tone_pcs:
                        avoid_notes.append(pc)

        return {
            'root': root,
            'chord_tones': chord_tones,
            'extensions': extensions,
            'avoid_notes': avoid_notes,
            'all_available': chord_tones + extensions,
            'extension_priority': extension_priority,  # For smart extension ordering
            'chord_tone_priority': chord_tone_priority,  # For V chord 3rd/7th emphasis
            'secondary_function': secondary_function,  # e.g., 'V/V', 'V/vi'
            'melodic_minor_mode': melodic_minor_mode  # e.g., 'lydian_dominant', 'altered'
        }


# ============================================================================
# MELODY ANALYSIS
# ============================================================================

class MelodyAnalyzer:
    """Analyzes a melody MIDI file and extracts note events"""

    @staticmethod
    def extract_melody(midi_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Extract melody notes from MIDI file.

        Returns:
            List of (start_tick, duration_ticks, note, velocity) tuples
        """
        mid = mido.MidiFile(midi_path)

        # Find melody track
        melody_events = []
        for track in mid.tracks:
            note_ons = {}
            time_accum = 0

            for msg in track:
                time_accum += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    note_ons[msg.note] = (time_accum, msg.velocity)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in note_ons:
                        start_time, velocity = note_ons[msg.note]
                        duration = time_accum - start_time
                        melody_events.append((start_time, duration, msg.note, velocity))
                        del note_ons[msg.note]

            if melody_events:
                break

        return sorted(melody_events)


# ============================================================================
# VOICING STRATEGIES (IMPROVED)
# ============================================================================

class ImprovedVoicing:
    """Generates voicings with intelligent spacing and chord-tone priority"""

    @staticmethod
    def get_chord_at_time(tick: int, ticks_per_beat: int, chord_progression: Dict[int, str]) -> str:
        """Determine which chord is active at a given tick"""
        if not chord_progression:
            return 'C'  # Default

        # Find the chord at or before this tick
        active_chord = None
        for chord_tick in sorted(chord_progression.keys()):
            if chord_tick <= tick:
                active_chord = chord_progression[chord_tick]
            else:
                break

        return active_chord if active_chord else 'C'

    @staticmethod
    def select_voicing_notes(
        melody_note: int,
        chord_info: Dict,
        num_voices: int,
        min_interval: int = 3,
        max_interval: int = 7,
        allow_doubling: bool = True,
        rootless: bool = False,
        prev_voicing: Optional[List[int]] = None,
        melody_direction: int = 0,
        voice_direction_angle: float = 90.0,
        voice_direction_strength: float = 0.5,
        voice_leading_strength: float = 0.75,
        max_voicing_range: int = 18
    ) -> List[int]:
        """
        Select notes for voicing with intelligent spacing and voice leading.

        Args:
            melody_note: The melody note (top voice)
            chord_info: Chord analysis from ChordAnalyzer
            num_voices: Number of voices (4 or 5)
            min_interval: Minimum interval between voices (semitones)
            max_interval: Maximum interval between voices (semitones)
            allow_doubling: If False, forces unique pitch classes (requires extensions for 5-way)
            rootless: If True, excludes root from voicing (jazz style)
            prev_voicing: Previous voicing for voice leading
            melody_direction: Direction of melody movement (positive=up, negative=down, 0=first note)
            voice_direction_angle: Angle in degrees (20-160, 90=flat, <90=follows melody, >90=contrary)
            voice_direction_strength: Strength of direction following (0.0-1.0)
            max_voicing_range: Maximum span from lowest to highest note (semitones, default 18)

        Returns:
            List of MIDI notes including melody on top
        """
        # Priority: chord tones first, then extensions
        chord_tones = chord_info['chord_tones']
        extensions = chord_info['extensions']
        avoid_notes = chord_info['avoid_notes']
        root = chord_info['root']
        extension_priority = chord_info.get('extension_priority', {})
        chord_tone_priority = chord_info.get('chord_tone_priority', {})

        # Filter out root if rootless
        if rootless:
            chord_tones = [n for n in chord_tones if (n % 12) != (root % 12)]

        # Get all available notes in all octaves below melody
        available_notes = []
        used_pitch_classes = set()

        # If no doubling, track melody's pitch class as used
        if not allow_doubling:
            used_pitch_classes.add(melody_note % 12)

        # Prioritize chord tones: Use chord_tone_priority if available (for V chords)
        # Otherwise: root/3rd/7th (equal top priority), then 5th, then extensions
        root_pc = root % 12
        for i, note in enumerate(chord_tones):
            note_pc = note % 12

            # Check if we have custom priority (e.g., V chord boosting 3rd/7th)
            if note_pc in chord_tone_priority:
                custom_priority = chord_tone_priority[note_pc]
                # Map to priority type string
                if custom_priority == 0:
                    priority_type = '3rd'  # Highest (3rd and 7th on V)
                elif custom_priority == 1:
                    priority_type = 'root'  # Medium
                else:
                    priority_type = '5th'  # Lower
            else:
                # Default priority assignment
                if note_pc == root_pc:
                    priority_type = 'root'
                elif i == 1:  # 3rd
                    priority_type = '3rd'
                elif i == 3:  # 7th (if present)
                    priority_type = '7th'
                elif i == 2:  # 5th
                    priority_type = '5th'
                else:
                    priority_type = 'chord_tone'

            for octave in range(-3, 1):  # 3 octaves below to current
                candidate = (note % 12) + (octave * 12) + 24  # Start from C1
                while candidate < melody_note:
                    # Check avoid notes and doubling constraints
                    pitch_class = candidate % 12
                    if pitch_class not in avoid_notes:
                        if allow_doubling or pitch_class not in used_pitch_classes:
                            available_notes.append((candidate, priority_type, pitch_class))
                    candidate += 12

        # Add extensions with priority-based classification
        for i, note in enumerate(extensions):
            note_pc = note % 12

            # Use extension_priority if available, otherwise fall back to default classification
            if note_pc in extension_priority:
                priority_value = extension_priority[note_pc]
                # Map priority to type string for compatibility
                if priority_value == 0:
                    ext_type = '7th_ext'  # Highest priority extension
                elif priority_value == 1:
                    ext_type = '9th_ext'  # High priority
                elif priority_value == 2:
                    ext_type = '11th_ext'  # Medium priority
                elif priority_value == 3:
                    ext_type = '13th_ext'  # Low priority (e.g., 13 on Dorian)
                else:
                    ext_type = 'extension'  # Lowest/avoid
            else:
                # Fallback: Determine extension type based on interval from root
                interval_from_root = (note_pc - root_pc) % 12

                # Classify: 7th=10/11, 9th=13/14, 11th=17/18, 13th=20/21
                if interval_from_root in [10, 11]:
                    ext_type = '7th_ext'
                elif interval_from_root in [13, 14, 2, 1]:  # 9th (also wraps to 1/2)
                    ext_type = '9th_ext'
                elif interval_from_root in [17, 18, 5, 6]:  # 11th (also wraps to 5/6)
                    ext_type = '11th_ext'
                elif interval_from_root in [20, 21, 8, 9]:  # 13th (also wraps to 8/9)
                    ext_type = '13th_ext'
                else:
                    ext_type = 'extension'

            for octave in range(-3, 1):
                candidate = (note % 12) + (octave * 12) + 24
                while candidate < melody_note:
                    pitch_class = candidate % 12
                    if pitch_class not in avoid_notes:
                        if allow_doubling or pitch_class not in used_pitch_classes:
                            available_notes.append((candidate, ext_type, pitch_class))
                    candidate += 12

        # Voice leading: prefer notes close to previous voicing positions
        if prev_voicing and melody_direction != 0:
            # Calculate how much voices should move based on melody direction
            # angle < 90: follow melody direction, angle = 90: stay flat, angle > 90: contrary motion
            angle_rad = math.radians(voice_direction_angle)
            # direction_factor: 1.0 = full parallel motion, 0.0 = no motion, -1.0 = contrary motion
            direction_factor = math.cos(angle_rad)  # At 0deg=1, 90deg=0, 180deg=-1

            # Target movement for voices (semitones)
            target_voice_movement = int(melody_direction * direction_factor * voice_direction_strength)

            # Move all previous voices (except melody) by target amount to get target positions
            prev_harmony = prev_voicing[:-1]  # All voices except melody
            target_positions = [v + target_voice_movement for v in prev_harmony]

            # For each available note, calculate minimum distance to any target position
            # Priority: 3rd=7th (0 for V chords with boost), root (0 default), 5th (3), extensions...
            # Use dynamic priority based on chord_tone_priority if available
            priority_order = {
                'root': 0, '3rd': 0, '7th': 0,  # Will be overridden by chord_tone_priority
                '5th': 3,
                '7th_ext': 4,
                '9th_ext': 5,
                '13th_ext': 6,
                '11th_ext': 8,  # 11th lowest priority
                'extension': 7,
                'chord_tone': 9
            }

            # CRITICAL: If we have chord_tone_priority (V chords), override the defaults
            # This ensures 3rd and 7th get highest priority on dominant chords
            if chord_tone_priority:
                # Update priority_order based on actual pitch class priorities
                # Map pitch classes back to priority types for sorting
                for note, priority, pitch_class in available_notes:
                    if pitch_class in chord_tone_priority:
                        custom_prio = chord_tone_priority[pitch_class]
                        # Lower custom_prio = higher priority
                        # Map: 0 -> use as-is, 1 -> demote, 2 -> demote further
                        # This ensures 3rd/7th (priority 0) sort before root (priority 1)
                        if priority in ['3rd', '7th', 'root', '5th']:
                            priority_order[priority] = custom_prio
            available_notes_with_distance = []
            for candidate, priority, pitch_class in available_notes:
                # Find closest target position
                min_distance = min(abs(candidate - target) for target in target_positions)
                priority_value = priority_order.get(priority, 9)

                # Blend distance and priority based on voice_leading_strength
                # voice_leading_strength = 1.0 means pure distance-based
                # voice_leading_strength = 0.0 means pure priority-based
                # voice_leading_strength = 0.75 means 75% distance, 25% priority
                blended_score = (voice_leading_strength * min_distance) + ((1 - voice_leading_strength) * priority_value * 2)

                available_notes_with_distance.append((blended_score, min_distance, priority_value, candidate, priority, pitch_class))

            # Sort by: 1) blended score 2) pitch (higher preferred for tighter voicings)
            available_notes_with_distance.sort(key=lambda x: (x[0], -x[3]))
            available_notes = [(x[3], x[4], x[5]) for x in available_notes_with_distance]
        else:
            # No previous voicing - sort by pitch (highest first) and priority
            # Priority: root=3rd=7th (0), 5th (3), 7th_ext (4), 9th_ext (5), 13th_ext (6), 11th_ext (8)
            priority_order = {
                'root': 0, '3rd': 0, '7th': 0,
                '5th': 3,
                '7th_ext': 4,
                '9th_ext': 5,
                '13th_ext': 6,
                '11th_ext': 8,  # 11th lowest priority
                'extension': 7,
                'chord_tone': 9
            }
            available_notes.sort(key=lambda x: (-x[0], priority_order.get(x[1], 9)))

        # Select notes with good spacing and voice leading
        # CRITICAL: For V chords, reserve slots for 3rd and 7th
        selected = []
        selected_pitch_classes = set([melody_note % 12]) if not allow_doubling else set()
        last_note = melody_note
        required_pitch_classes = set()  # 3rd and 7th for V chords

        # If we have chord_tone_priority (V chords), mark 3rd and 7th as required
        if chord_tone_priority:
            for pitch_class, prio in chord_tone_priority.items():
                if prio == 0:  # Highest priority (3rd and 7th)
                    required_pitch_classes.add(pitch_class)

        # Regular selection, but boost required pitch classes
        for candidate, priority, pitch_class in available_notes:
            # Skip if no doubling and pitch class already used
            if not allow_doubling and pitch_class in selected_pitch_classes:
                continue

            # IMPORTANT: Never use 11th in the bottom 2 voices (bass and second voice)
            is_eleventh = priority == '11th_ext'
            if is_eleventh and len(selected) < 2:
                continue  # Skip 11th for bass and second-lowest voice

            interval = last_note - candidate

            # CRITICAL: No minor 2nds in the bass (first interval from bass)
            # This creates muddy, dissonant voicings
            if len(selected) == 1:  # Adding second voice (interval from bass)
                if interval == 1:  # Minor 2nd in bass
                    continue  # Skip - too dissonant in bass

            # CRITICAL: No minor 2nds from melody (top voice)
            # Avoid clash between melody and highest harmony note
            if len(selected) == 0:  # Adding first voice (closest to melody)
                melody_interval = melody_note - candidate
                if melody_interval == 1:  # Minor 2nd below melody
                    continue  # Skip - clashes with melody

            # Check if this is a required note (3rd/7th on V chords)
            is_required = pitch_class in required_pitch_classes

            # For required notes, relax spacing slightly
            spacing_ok = interval >= min_interval and interval <= max_interval
            if is_required and not spacing_ok:
                # Allow required notes with slightly wider spacing (up to max + 2)
                spacing_ok = interval >= min_interval and interval <= (max_interval + 2)

            # Check spacing constraints
            if spacing_ok:
                # Accept notes based on priority
                # Required (3rd/7th on V) = highest priority
                # Root, 3rd, 7th = always accept
                # 5th = accept (chord tone)
                # Extensions = accept when needed for voice count or voice leading
                is_high_priority = priority in ['root', '3rd', '7th', '7th_ext'] or is_required
                is_fifth = priority == '5th'
                is_extension = priority in ['extension', '9th_ext', '13th_ext', '11th_ext']

                # Accept high priority notes and 5th always
                if is_high_priority or is_fifth:
                    selected.append(candidate)
                    selected_pitch_classes.add(pitch_class)
                    if is_required:
                        required_pitch_classes.remove(pitch_class)  # Mark as fulfilled
                    last_note = candidate
                # Accept extensions only if we need more voices AND all required notes are in
                elif is_extension and len(selected) < (num_voices - 1) and len(required_pitch_classes) == 0:
                    selected.append(candidate)
                    selected_pitch_classes.add(pitch_class)
                    last_note = candidate

                if len(selected) >= (num_voices - 1):  # -1 because melody is on top
                    break

        # If we don't have enough notes, relax spacing constraints
        if len(selected) < (num_voices - 1):
            for candidate, priority, pitch_class in available_notes:
                if candidate in selected:
                    continue
                if not allow_doubling and pitch_class in selected_pitch_classes:
                    continue

                # Still avoid 11th in bottom 2 voices even with relaxed constraints
                is_eleventh = priority == '11th_ext'
                if is_eleventh and len(selected) < 2:
                    continue

                interval = abs(last_note - candidate) if selected else abs(melody_note - candidate)
                if interval >= min_interval:  # At least minimum interval
                    selected.append(candidate)
                    selected_pitch_classes.add(pitch_class)
                    if len(selected) >= (num_voices - 1):
                        break

        # Return sorted with melody on top
        result = sorted(selected[:(num_voices - 1)]) + [melody_note]

        # CRITICAL: Check for minor 2nd from melody after sorting
        if len(result) >= 2:
            melody = result[-1]
            top_harmony = result[-2]
            if melody - top_harmony == 1:  # Minor 2nd below melody
                # Try to find a replacement that avoids m2
                # Prefer chord tones (non-root) over extensions
                found_replacement = False

                # First, check if the m2 note is required (3rd/7th on V)
                top_harmony_pc = top_harmony % 12
                is_required_note = top_harmony_pc in chord_tone_priority and chord_tone_priority.get(top_harmony_pc, 99) == 0

                if not is_required_note:
                    # Not required, so we can replace it
                    # Sort candidates by priority (chord tones first)
                    candidates_sorted = sorted(
                        available_notes,
                        key=lambda x: (0 if x[1] in ['3rd', '7th', 'root', '5th'] else 1, -x[0])
                    )

                    for candidate, priority, pitch_class in candidates_sorted:
                        if candidate in result:
                            continue
                        # Skip if no doubling and pitch class already used
                        if not allow_doubling and pitch_class in [n % 12 for n in result]:
                            continue
                        # Check if this avoids the m2
                        if melody - candidate >= 2:  # At least major 2nd
                            # Check spacing with other voices
                            test_harmony = [n for n in result[:-1] if n != top_harmony] + [candidate]
                            test_result = sorted(test_harmony) + [melody]
                            # Verify all intervals are reasonable
                            intervals_ok = all(
                                test_result[i+1] - test_result[i] >= min_interval
                                for i in range(len(test_result)-1)
                            )
                            if intervals_ok:
                                result = test_result
                                found_replacement = True
                                break

                # If still m2 after replacement attempt, drop to 3-voice
                if not found_replacement and len(result) > 2:
                    # Keep the lower voices, drop the one creating m2
                    result = result[:-2] + [melody]

        # CRITICAL: Check if 11th ended up in bottom 2 voices after sorting
        # If so, try to replace it with a better chord tone
        if len(result) >= 3:  # Only check if we have at least bass + 2nd voice
            root_pc = root % 12
            for i in range(min(2, len(result) - 1)):  # Check bottom 2 voices (not melody)
                note_pc = result[i] % 12
                interval_from_root = (note_pc - root_pc) % 12
                # Check if this is an 11th (interval 5 or 6)
                if interval_from_root in [5, 6]:
                    # Try to find a replacement from available_notes that's NOT an 11th
                    for candidate, priority, pitch_class in available_notes:
                        if candidate in result:
                            continue
                        # Skip if it's also an 11th
                        candidate_interval = (pitch_class - root_pc) % 12
                        if candidate_interval in [5, 6]:
                            continue
                        # Skip if no doubling and already used
                        if not allow_doubling and pitch_class in [n % 12 for n in result]:
                            continue
                        # Replace the 11th with this better note
                        result[i] = candidate
                        # Re-sort
                        result = sorted(result[:-1]) + [result[-1]]
                        break

        # Enforce max voicing range constraint
        # If the span from lowest to highest note exceeds max_voicing_range,
        # shift lower voices up by octave to keep voicing compact
        if len(result) > 1:
            current_range = result[-1] - result[0]  # melody (highest) - bass (lowest)

            # Enforce the 18-semitone limit strictly
            # 18 semitones is plenty of space, so prioritize staying within the limit
            if current_range > max_voicing_range:
                safety_iterations = 0
                max_iterations = len(result) * 2  # Prevent infinite loops

                while current_range > max_voicing_range and safety_iterations < max_iterations:
                    safety_iterations += 1

                    # Try shifting the lowest voice up by octave
                    adjusted = result.copy()
                    adjusted[0] += 12

                    # Re-sort to maintain order (in case shifting creates crossovers)
                    adjusted = sorted(adjusted[:-1]) + [adjusted[-1]]
                    new_range = adjusted[-1] - adjusted[0]

                    # Check for unisons (0 semitones) and minor 2nds from melody
                    has_unison = False
                    has_m2_from_melody = False
                    for j in range(len(adjusted) - 1):
                        interval = adjusted[j+1] - adjusted[j]
                        if interval == 0:
                            has_unison = True
                            break
                        # Check if this creates m2 from melody (top note)
                        if j == len(adjusted) - 2 and interval == 1:
                            has_m2_from_melody = True
                            break

                    # Apply shift as long as:
                    # 1. No unisons created
                    # 2. No m2 from melody created
                    # 3. It reduces the range (gets us closer to the limit)
                    if not has_unison and not has_m2_from_melody and new_range < current_range:
                        result = adjusted
                        current_range = new_range
                    else:
                        # Can't improve further, stop
                        break

        # CRITICAL: Check for minor 2nd from melody AFTER range adjustment
        # If range is still too wide AND we have m2, we need to handle both constraints
        if len(result) >= 2:
            melody = result[-1]
            top_harmony = result[-2]
            current_range = result[-1] - result[0]

            # If we have m2 from melody AND range is still too wide,
            # try shifting the 2nd-lowest voice instead of the lowest
            if melody - top_harmony == 1 and current_range > max_voicing_range:
                if len(result) >= 3:
                    # Try shifting second-lowest voice up
                    adjusted = result.copy()
                    adjusted[1] += 12
                    adjusted = sorted(adjusted[:-1]) + [adjusted[-1]]

                    # Check if this fixes both issues
                    new_range = adjusted[-1] - adjusted[0]
                    new_top_harmony = adjusted[-2]
                    has_m2 = melody - new_top_harmony == 1
                    has_unison = any(adjusted[i+1] - adjusted[i] == 0 for i in range(len(adjusted)-1))

                    if not has_m2 and not has_unison and new_range <= max_voicing_range:
                        result = adjusted
                    else:
                        # Last resort: drop to 4-voice
                        result = result[:-2] + [melody]
                else:
                    # Only 2-3 voices, can't shift, drop to fewer voices
                    result = result[:-2] + [melody] if len(result) > 2 else result
            elif melody - top_harmony == 1:
                # We have m2 but range is OK - just drop the problematic voice
                if len(result) > 2:
                    result = result[:-2] + [melody]

        return result

    @staticmethod
    def four_way_closed(
        melody_note: int,
        chord_info: Dict,
        min_interval: int = 3,
        max_interval: int = 7,
        allow_doubling: bool = True,
        rootless: bool = False,
        prev_voicing: Optional[List[int]] = None,
        melody_direction: int = 0,
        voice_direction_angle: float = 90.0,
        voice_direction_strength: float = 0.5,
        voice_leading_strength: float = 0.75,
        max_voicing_range: int = 18
    ) -> List[int]:
        """4-way closed voicing with intelligent spacing"""
        return ImprovedVoicing.select_voicing_notes(
            melody_note, chord_info, 4,
            min_interval=min_interval,
            max_interval=max_interval,
            allow_doubling=allow_doubling,
            rootless=rootless,
            prev_voicing=prev_voicing,
            melody_direction=melody_direction,
            voice_direction_angle=voice_direction_angle,
            voice_direction_strength=voice_direction_strength,
            voice_leading_strength=voice_leading_strength,
            max_voicing_range=max_voicing_range
        )

    @staticmethod
    def five_way_closed(
        melody_note: int,
        chord_info: Dict,
        min_interval: int = 3,
        max_interval: int = 7,
        allow_doubling: bool = True,
        rootless: bool = False,
        prev_voicing: Optional[List[int]] = None,
        melody_direction: int = 0,
        voice_direction_angle: float = 90.0,
        voice_direction_strength: float = 0.5,
        voice_leading_strength: float = 0.75,
        max_voicing_range: int = 18
    ) -> List[int]:
        """5-way closed voicing with extensions"""
        return ImprovedVoicing.select_voicing_notes(
            melody_note, chord_info, 5,
            min_interval=min_interval,
            max_interval=max_interval,
            allow_doubling=allow_doubling,
            rootless=rootless,
            prev_voicing=prev_voicing,
            melody_direction=melody_direction,
            voice_direction_angle=voice_direction_angle,
            voice_direction_strength=voice_direction_strength,
            voice_leading_strength=voice_leading_strength,
            max_voicing_range=max_voicing_range
        )

    @staticmethod
    def block_chords(
        melody_note: int,
        chord_info: Dict,
        min_interval: int = 3,
        max_interval: int = 7,
        allow_doubling: bool = True,
        rootless: bool = False,
        prev_voicing: Optional[List[int]] = None,
        melody_direction: int = 0,
        voice_direction_angle: float = 90.0,
        voice_direction_strength: float = 0.5,
        voice_leading_strength: float = 0.75,
        max_voicing_range: int = 18
    ) -> List[int]:
        """Block chords with melody doubled"""
        four_way = ImprovedVoicing.four_way_closed(
            melody_note, chord_info, min_interval, max_interval,
            allow_doubling, rootless, prev_voicing, melody_direction,
            voice_direction_angle, voice_direction_strength, voice_leading_strength, max_voicing_range
        )

        # Double melody octave below
        melody_double = melody_note - 12 if melody_note - 12 >= 24 else melody_note

        return sorted(set(four_way + [melody_double]))


# ============================================================================
# APPROACH CHORD GENERATION
# ============================================================================

class ApproachChords:
    """Generate approach chords for voice leading"""

    @staticmethod
    def generate_diatonic_approach(
        current_voicing: List[int],
        target_voicing: List[int],
        melody_note: int,
        scale_context: Optional[ScaleContext]
    ) -> List[int]:
        """
        Generate diatonic approach chord by moving each voice diatonically toward target.
        Each voice moves by step (whole or half step) in the direction of the target note.
        """
        # Major scale intervals
        major_scale_intervals = [0, 2, 4, 5, 7, 9, 11]
        # Natural minor scale intervals
        minor_scale_intervals = [0, 2, 3, 5, 7, 8, 10]

        # Get scale root and type from context
        if scale_context:
            scale_root = scale_context.root_note % 12
            scale_intervals = minor_scale_intervals if scale_context.scale_type == 'minor' else major_scale_intervals
        else:
            scale_root = 0  # C
            scale_intervals = major_scale_intervals

        # Transpose scale intervals to actual pitch classes in the key
        scale_notes = [(scale_root + interval) % 12 for interval in scale_intervals]

        approach_voicing = []

        for i in range(len(current_voicing) - 1):  # Exclude melody
            current_note = current_voicing[i]
            target_note = target_voicing[i] if i < len(target_voicing) - 1 else target_voicing[i]


            # Determine direction and diatonic step
            if target_note > current_note:
                # Try whole step first, fall back to half step
                whole_step = current_note + 2
                half_step = current_note + 1

                # Check if whole step is diatonic (in the scale)
                if whole_step % 12 in scale_notes:
                    approach_note = whole_step
                else:
                    approach_note = half_step

            elif target_note < current_note:
                # Try whole step down first, fall back to half step
                whole_step = current_note - 2
                half_step = current_note - 1

                # Check if whole step is diatonic (in the scale)
                if whole_step % 12 in scale_notes:
                    approach_note = whole_step
                else:
                    approach_note = half_step
            else:
                # Already at target, stay
                approach_note = current_note

            approach_voicing.append(approach_note)

        # Add melody - DON'T sort, maintain voice order
        approach_voicing.append(melody_note)
        return approach_voicing

    @staticmethod
    def generate_chromatic_approach(
        current_voicing: List[int],
        target_voicing: List[int],
        melody_note: int
    ) -> List[int]:
        """
        Generate chromatic approach by moving entire voicing in parallel by half step.
        Maintains exact interval relationships - parallel motion.
        """
        # Determine average direction
        avg_direction = sum(target_voicing[i] - current_voicing[i]
                          for i in range(min(len(current_voicing), len(target_voicing))))

        if avg_direction > 0:
            # Move ALL voices up by half step (parallel motion)
            approach_voicing = [note + 1 for note in current_voicing]
        elif avg_direction < 0:
            # Move ALL voices down by half step (parallel motion)
            approach_voicing = [note - 1 for note in current_voicing]
        else:
            # No change
            approach_voicing = current_voicing.copy()

        # Replace the last note (melody) with the actual melody note from MIDI
        # This keeps the melody following the actual MIDI while harmony moves chromatically
        approach_voicing[-1] = melody_note

        # DON'T sort - maintain parallel motion and voice order
        return approach_voicing

    @staticmethod
    def generate_dominant_approach(
        target_voicing: List[int],
        target_chord_root: int,
        melody_note: int,
        num_voices: int
    ) -> List[int]:
        """
        Generate dominant approach (V of target) creating applied dominant effect.
        For target chord, create its V7 chord.
        """
        # Calculate V of target (perfect 5th above = 7 semitones)
        approach_root = (target_chord_root + 7) % 12

        # Build dominant 7th chord: root, 3rd, 5th, b7
        chord_tones = [
            approach_root,           # Root
            (approach_root + 4) % 12,   # Major 3rd
            (approach_root + 7) % 12,   # Perfect 5th
            (approach_root + 10) % 12,  # Minor 7th
        ]

        # Find octaves for each voice to match target voicing range
        approach_voicing = []
        bass_octave = target_voicing[0] // 12

        for i in range(num_voices - 1):
            # Cycle through chord tones
            pitch_class = chord_tones[i % len(chord_tones)]
            # Find best octave close to target
            target_note = target_voicing[i] if i < len(target_voicing) - 1 else target_voicing[-2]
            octave = target_note // 12

            # Try different octaves to get close to target
            best_note = pitch_class + (octave * 12)
            if abs((pitch_class + (octave - 1) * 12) - target_note) < abs(best_note - target_note):
                best_note = pitch_class + (octave - 1) * 12
            if abs((pitch_class + (octave + 1) * 12) - target_note) < abs(best_note - target_note):
                best_note = pitch_class + (octave + 1) * 12

            approach_voicing.append(best_note)

        approach_voicing.append(melody_note)
        return sorted(approach_voicing[:-1]) + [approach_voicing[-1]]

    @staticmethod
    def generate_diminished_approach(
        target_voicing: List[int],
        target_chord_root: int,
        melody_note: int,
        num_voices: int
    ) -> List[int]:
        """
        Generate diminished 7th approach a half step below target.
        Each voice moves up by half step to chord tone of destination.
        Diminished 7th = root, m3, dim5, dim7 (all minor 3rds apart).
        """
        # Diminished chord half step below target root
        dim_root = (target_chord_root - 1) % 12

        # Build diminished 7th chord (symmetrical, all minor 3rds)
        chord_tones = [
            dim_root,
            (dim_root + 3) % 12,   # Minor 3rd
            (dim_root + 6) % 12,   # Diminished 5th
            (dim_root + 9) % 12,   # Diminished 7th
        ]

        # Find octaves for each voice
        approach_voicing = []

        for i in range(num_voices - 1):
            # Each approach note should be a half step below its target
            target_note = target_voicing[i] if i < len(target_voicing) - 1 else target_voicing[-2]
            approach_note = target_note - 1

            # Adjust to nearest diminished chord tone
            approach_pc = approach_note % 12
            if approach_pc not in chord_tones:
                # Find nearest dim chord tone
                distances = [(abs(approach_pc - ct), ct) for ct in chord_tones]
                _, nearest_ct = min(distances)
                octave = approach_note // 12
                approach_note = nearest_ct + (octave * 12)

            approach_voicing.append(approach_note)

        approach_voicing.append(melody_note)
        return sorted(approach_voicing[:-1]) + [approach_voicing[-1]]


# ============================================================================
# MAIN HARMONIZER
# ============================================================================

def harmonize_melody_improved(
    input_midi_path: str,
    output_midi_path: str,
    chord_progression: Dict[int, str],
    voicing_type: str = '5-way-closed',
    min_interval: int = 3,
    max_interval: int = 7,
    use_extensions: bool = True,
    allow_doubling: bool = True,
    rootless: bool = False,
    voice_direction_angle: float = 90.0,
    voice_direction_strength: float = 0.8,
    voice_leading_strength: float = 0.75,
    voice_tightness: float = 0.5,
    smooth_mode: bool = False,
    multitrack_mode: bool = False,
    approach_mode: Optional[str] = None,
    scale_context: Optional[ScaleContext] = None,
    max_voicing_range: int = 18
) -> str:
    """
    Harmonize melody with specific chord progression.

    Args:
        input_midi_path: Input melody MIDI
        output_midi_path: Output harmonized MIDI
        chord_progression: Dict of {beat: chord_name}, e.g., {0: 'C', 4: 'Dm'}
        voicing_type: '4-way-closed', '5-way-closed', or 'block'
        min_interval: Minimum interval between voices (3=minor 3rd, 4=major 3rd, etc.)
        max_interval: Maximum interval between voices (semitones)
        use_extensions: If True, use 7th, 9th, 11th, 13th extensions
        allow_doubling: If False, forces unique pitch classes (needs extensions for 5-way)
        rootless: If True, omits root from voicing (jazz style)
        voice_direction_angle: Direction angle in degrees (20-160, 90=flat, <90=follow melody)
        voice_direction_strength: Strength of direction following (0.0-1.0)
        voice_leading_strength: Prioritize closer notes for voice leading (0.0-1.0, default 0.75)
        voice_tightness: Control voicing compactness (0.0=loose/wide, 1.0=tight/close, default 0.5)
        smooth_mode: If True, hold harmony notes when voicing doesn't change (legato)
        multitrack_mode: If True, create separate MIDI track for each voice
        approach_mode: Approach chord style: 'diatonic', 'chromatic', 'dominant', 'diminished', or None
        scale_context: Scale/key context for proper extensions
        max_voicing_range: Maximum span from lowest to highest note (semitones, default 18)

    Returns:
        Path to output file
    """
    # Calculate dynamic max_voicing_range based on tightness
    # tightness = 0.0: wide (18 semitones)
    # tightness = 0.5: medium (14 semitones)
    # tightness = 1.0: tight (10 semitones)
    dynamic_max_range = int(18 - (voice_tightness * 8))  # 18 to 10 range

    print(f"\n{'='*80}")
    print(f"🎵 IMPROVED MELODY HARMONIZER")
    print(f"{'='*80}")
    print(f"Input: {input_midi_path}")
    print(f"Voicing: {voicing_type}")
    print(f"Mode: {'SMOOTH (legato)' if smooth_mode else 'RHYTHMIC (follows melody)'}")
    print(f"Output: {'MULTITRACK (separate tracks per voice)' if multitrack_mode else 'SINGLE TRACK'}")
    print(f"Approach: {approach_mode.upper() if approach_mode else 'NONE'}")
    print(f"Key: {scale_context if scale_context else 'Not specified'}")
    print(f"Min interval: {min_interval} semitones, Max: {max_interval}")
    print(f"Max voicing range: {dynamic_max_range} semitones (tightness: {voice_tightness})")
    print(f"Extensions: {use_extensions}, Doubling: {allow_doubling}, Rootless: {rootless}")
    print(f"Voice direction: {voice_direction_angle}° (strength: {voice_direction_strength})")
    print(f"Chord progression: {chord_progression}")

    # Extract melody
    melody_events = MelodyAnalyzer.extract_melody(input_midi_path)
    print(f"Extracted {len(melody_events)} melody notes")

    # Load input MIDI
    input_mid = mido.MidiFile(input_midi_path)
    ticks_per_beat = input_mid.ticks_per_beat

    # Create output MIDI
    if multitrack_mode:
        # Type 1 MIDI file (multitrack)
        output_mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    else:
        # Type 0 MIDI file (single track)
        output_mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        output_track = mido.MidiTrack()
        output_mid.tracks.append(output_track)

        # Copy tempo to single track
        for msg in input_mid.tracks[0]:
            if msg.type in ['set_tempo', 'time_signature']:
                output_track.append(msg.copy())

    # Collect all note events (on and off) with their voicings
    note_events = []
    voice_events = []  # For multitrack mode: list of (tick, msg_type, voice_index, note, velocity)
    prev_voicing = None
    prev_melody_note = None
    active_harmony_notes = {}  # Track harmony notes that are being held in smooth mode: {note: start_tick}

    for i, (start_tick, duration, melody_note, velocity) in enumerate(melody_events):
        # Get active chord at this time
        active_chord = ImprovedVoicing.get_chord_at_time(start_tick, ticks_per_beat, chord_progression)

        # Look ahead to get next chord for contextual analysis
        next_chord = None
        chord_will_change = False
        next_voicing = None
        if i + 1 < len(melody_events):
            next_start_tick = melody_events[i + 1][0]
            next_chord = ImprovedVoicing.get_chord_at_time(next_start_tick, ticks_per_beat, chord_progression)
            chord_will_change = (next_chord != active_chord)

        # Calculate melody direction
        melody_direction = 0
        if prev_melody_note is not None:
            melody_direction = melody_note - prev_melody_note

        # Analyze chord with key context and next chord info
        chord_info = ChordAnalyzer.parse_chord(active_chord, use_extensions=use_extensions, scale_context=scale_context, next_chord_name=next_chord)

        print(f"\n  Tick {start_tick}, Note {melody_note}: Chord = {active_chord}")
        if melody_direction != 0:
            print(f"    Melody direction: {'+' if melody_direction > 0 else ''}{melody_direction} semitones")
        print(f"    Chord tones: {chord_info['chord_tones']}")
        print(f"    Extensions: {chord_info['extensions']}")
        print(f"    Avoid: {[f'{n}' for n in chord_info['avoid_notes']]}")

        # Generate voicing with voice leading
        if voicing_type == '4-way-closed':
            voicing = ImprovedVoicing.four_way_closed(
                melody_note, chord_info, min_interval, max_interval,
                allow_doubling, rootless, prev_voicing, melody_direction,
                voice_direction_angle, voice_direction_strength, voice_leading_strength, dynamic_max_range
            )
        elif voicing_type == '5-way-closed':
            voicing = ImprovedVoicing.five_way_closed(
                melody_note, chord_info, min_interval, max_interval,
                allow_doubling, rootless, prev_voicing, melody_direction,
                voice_direction_angle, voice_direction_strength, voice_leading_strength, dynamic_max_range
            )
        elif voicing_type == 'block':
            voicing = ImprovedVoicing.block_chords(
                melody_note, chord_info, min_interval, max_interval,
                allow_doubling, rootless, prev_voicing, melody_direction,
                voice_direction_angle, voice_direction_strength, voice_leading_strength, dynamic_max_range
            )
        else:
            voicing = [melody_note]

        print(f"    Voicing: {voicing}")

        # Calculate intervals for debugging
        intervals = [voicing[i+1] - voicing[i] for i in range(len(voicing)-1)]
        print(f"    Intervals: {intervals}")

        # Show pitch classes for debugging doubling
        pitch_classes = [n % 12 for n in voicing]
        unique_pcs = len(set(pitch_classes))
        print(f"    Pitch classes: {pitch_classes} (unique: {unique_pcs})")

        # Show voice movement
        if prev_voicing:
            bass_movement = min(voicing[:-1]) - min(prev_voicing[:-1])
            print(f"    Bass movement: {'+' if bass_movement > 0 else ''}{bass_movement} semitones")

        # APPROACH CHORD GENERATION
        # If approach mode is enabled and chord will change on next note, generate approach chord
        use_approach_chord = False
        approach_voicing = None

        if approach_mode and chord_will_change and prev_voicing and i + 1 < len(melody_events):
            # Generate next chord's target voicing to approach toward
            next_melody_note = melody_events[i + 1][2]
            next_chord_info = ChordAnalyzer.parse_chord(next_chord, use_extensions=use_extensions, scale_context=scale_context)

            # Generate target voicing for next chord WITHOUT voice leading influence
            # (We want the "ideal" chord tones, not what voice leading would produce)
            # Set all directional parameters to neutral/zero to get pure chord tones
            if voicing_type == '4-way-closed':
                next_voicing = ImprovedVoicing.four_way_closed(
                    next_melody_note, next_chord_info, min_interval, max_interval,
                    allow_doubling, rootless, None, 0,  # No previous voicing, no melody direction
                    90.0, 0.0, 0.0, dynamic_max_range  # Neutral angle, zero strengths
                )
            elif voicing_type == '5-way-closed':
                next_voicing = ImprovedVoicing.five_way_closed(
                    next_melody_note, next_chord_info, min_interval, max_interval,
                    allow_doubling, rootless, None, 0,  # No previous voicing, no melody direction
                    90.0, 0.0, 0.0, dynamic_max_range  # Neutral angle, zero strengths
                )
            elif voicing_type == 'block':
                next_voicing = ImprovedVoicing.block_chords(
                    next_melody_note, next_chord_info, min_interval, max_interval,
                    allow_doubling, rootless, None, 0,  # No previous voicing, no melody direction
                    90.0, 0.0, 0.0, dynamic_max_range  # Neutral angle, zero strengths
                )
            else:
                next_voicing = voicing  # Fallback

            # Parse target chord root from library
            if next_chord in CHORD_LIBRARY:
                target_chord_root = CHORD_LIBRARY[next_chord][0] % 12
            else:
                target_chord_root = 60 % 12  # Default to C

            # Generate approach chord based on mode
            if approach_mode == 'diatonic':
                approach_voicing = ApproachChords.generate_diatonic_approach(
                    voicing, next_voicing, melody_note, scale_context
                )
                use_approach_chord = True
                print(f"    → DIATONIC APPROACH to {next_chord}: {approach_voicing}")

            elif approach_mode == 'chromatic':
                approach_voicing = ApproachChords.generate_chromatic_approach(
                    voicing, next_voicing, melody_note
                )
                use_approach_chord = True
                print(f"    → CHROMATIC APPROACH to {next_chord}: {approach_voicing}")

            elif approach_mode == 'dominant':
                approach_voicing = ApproachChords.generate_dominant_approach(
                    next_voicing, target_chord_root, melody_note, len(voicing)
                )
                use_approach_chord = True
                print(f"    → DOMINANT APPROACH to {next_chord}: {approach_voicing}")

            elif approach_mode == 'diminished':
                approach_voicing = ApproachChords.generate_diminished_approach(
                    next_voicing, target_chord_root, melody_note, len(voicing)
                )
                use_approach_chord = True
                print(f"    → DIMINISHED APPROACH to {next_chord}: {approach_voicing}")

        # Use approach voicing if generated, otherwise use normal voicing
        final_voicing = approach_voicing if use_approach_chord else voicing

        # In smooth mode, check if harmony voices (not melody) stayed the same
        harmony_changed = True
        harmony_notes = final_voicing[:-1]  # All voices except melody
        melody = final_voicing[-1]

        if smooth_mode and prev_voicing:
            # Compare harmony voices (all except melody/top note)
            prev_harmony = sorted(prev_voicing[:-1])
            curr_harmony = sorted(harmony_notes)
            harmony_changed = prev_harmony != curr_harmony

        # Store for next iteration (store original voicing, not approach)
        prev_voicing = voicing
        prev_melody_note = melody_note

        # Add note events
        end_tick = start_tick + duration

        if smooth_mode:
            if harmony_changed or not active_harmony_notes:
                # Harmony changed or first note: End previous harmony notes and start new ones
                # End all currently active harmony notes
                for note, start_time in active_harmony_notes.items():
                    note_events.append((start_tick, 'note_off', note, 0))

                # Start new harmony notes
                active_harmony_notes = {}
                for voice_idx, note in enumerate(harmony_notes):
                    note_events.append((start_tick, 'note_on', note, velocity))
                    active_harmony_notes[note] = start_tick
                    # For multitrack: track voice index (0=bass, 1=tenor, etc.)
                    if multitrack_mode:
                        voice_events.append((start_tick, 'note_on', voice_idx, note, velocity))

                # Always add melody note events
                note_events.append((start_tick, 'note_on', melody, velocity))
                note_events.append((end_tick, 'note_off', melody, 0))
                # Melody is the last voice (top voice)
                if multitrack_mode:
                    voice_events.append((start_tick, 'note_on', len(harmony_notes), melody, velocity))
                    voice_events.append((end_tick, 'note_off', len(harmony_notes), melody, 0))
            else:
                # Harmony stayed the same: just add melody note events, harmony continues
                note_events.append((start_tick, 'note_on', melody, velocity))
                note_events.append((end_tick, 'note_off', melody, 0))
                # Melody voice continues
                if multitrack_mode:
                    voice_events.append((start_tick, 'note_on', len(harmony_notes), melody, velocity))
                    voice_events.append((end_tick, 'note_off', len(harmony_notes), melody, 0))
                # Harmony notes keep playing from their original start time
        else:
            # RHYTHMIC MODE: Add all note events (use final_voicing which includes approach chords)
            for voice_idx, note in enumerate(final_voicing):
                note_events.append((start_tick, 'note_on', note, velocity))
                note_events.append((end_tick, 'note_off', note, 0))
                # For multitrack: track each voice
                if multitrack_mode:
                    voice_events.append((start_tick, 'note_on', voice_idx, note, velocity))
                    voice_events.append((end_tick, 'note_off', voice_idx, note, 0))

    # In smooth mode, close any still-active harmony notes at the end
    if smooth_mode and active_harmony_notes:
        # End all active harmony notes at the last melody note's end time
        last_end_tick = melody_events[-1][0] + melody_events[-1][1]
        for voice_idx, (note, start_time) in enumerate(active_harmony_notes.items()):
            note_events.append((last_end_tick, 'note_off', note, 0))
            if multitrack_mode:
                voice_events.append((last_end_tick, 'note_off', voice_idx, note, 0))

    # Sort all events by time
    note_events.sort(key=lambda x: (x[0], x[1] == 'note_off'))  # note_on before note_off at same time

    if multitrack_mode:
        # MULTITRACK MODE: Create separate track for each voice part
        # Determine number of voices from voice_events
        voice_indices = sorted(set(voice_idx for _, _, voice_idx, _, _ in voice_events))
        num_voices = len(voice_indices)
        print(f"\nCreating {num_voices} tracks for voice parts")

        # Create a track for tempo/time signature
        tempo_track = mido.MidiTrack()
        output_mid.tracks.append(tempo_track)
        tempo_track.append(mido.MetaMessage('track_name', name='Tempo', time=0))
        for msg in input_mid.tracks[0]:
            if msg.type in ['set_tempo', 'time_signature']:
                tempo_track.append(msg.copy())
        tempo_track.append(mido.MetaMessage('end_of_track', time=0))

        # Create a track for each voice part (bass, tenor, alto, soprano, melody)
        voice_names = ['Bass', 'Tenor', 'Alto', 'Soprano', 'Melody']
        for voice_idx in voice_indices:
            voice_track = mido.MidiTrack()
            output_mid.tracks.append(voice_track)

            # Add track name
            voice_name = voice_names[voice_idx] if voice_idx < len(voice_names) else f'Voice {voice_idx + 1}'
            voice_track.append(mido.MetaMessage('track_name', name=voice_name, time=0))

            # Filter events for this specific voice part
            this_voice_events = [(tick, msg_type, note, vel) for tick, msg_type, v_idx, note, vel in voice_events if v_idx == voice_idx]
            # Sort by time
            this_voice_events.sort(key=lambda x: (x[0], x[1] == 'note_off'))

            # Write events with proper delta times
            current_tick = 0
            for tick, msg_type, note, velocity in this_voice_events:
                delta = tick - current_tick
                if delta < 0:
                    print(f"WARNING: Negative delta {delta} at tick {tick}, setting to 0")
                    delta = 0

                voice_track.append(mido.Message(
                    msg_type,
                    note=note,
                    velocity=velocity,
                    time=delta
                ))
                current_tick = tick

            # End of track
            voice_track.append(mido.MetaMessage('end_of_track', time=0))
    else:
        # SINGLE TRACK MODE: Write all events to one track
        current_tick = 0
        for tick, msg_type, note, velocity in note_events:
            delta = tick - current_tick
            if delta < 0:
                print(f"WARNING: Negative delta {delta} at tick {tick}, setting to 0")
                delta = 0

            output_track.append(mido.Message(
                msg_type,
                note=note,
                velocity=velocity,
                time=delta
            ))
            current_tick = tick

        # End of track
        output_track.append(mido.MetaMessage('end_of_track', time=0))

    # Save
    output_mid.save(output_midi_path)

    print(f"\n✅ Harmonized MIDI saved: {output_midi_path}")
    print(f"{'='*80}\n")

    return output_midi_path


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Improved melody harmonizer')
    parser.add_argument('input', help='Input melody MIDI')
    parser.add_argument('--output', '-o', help='Output MIDI file')
    parser.add_argument('--voicing', '-v',
                       choices=['4-way-closed', '5-way-closed', 'block'],
                       default='5-way-closed',
                       help='Voicing type')
    parser.add_argument('--chords', '-c',
                       help='Chord progression (e.g., "0:C,4:Dm,8:G7")')
    parser.add_argument('--min-interval', type=int, default=3,
                       help='Minimum interval between voices (semitones)')
    parser.add_argument('--max-interval', type=int, default=7,
                       help='Maximum interval between voices (semitones)')
    parser.add_argument('--no-extensions', action='store_true',
                       help='Disable extensions (diatonic mode, chord tones only)')
    parser.add_argument('--no-doubling', action='store_true',
                       help='Disable doubling (forces unique pitch classes, requires extensions)')
    parser.add_argument('--rootless', action='store_true',
                       help='Rootless voicing (omit root from harmony, jazz style)')
    parser.add_argument('--key', '-k', help='Key (e.g., C, Dm, Eb) for proper extension selection')
    parser.add_argument('--voice-angle', type=float, default=90.0,
                       help='Voice leading angle in degrees (20-160, 90=flat, <90=follow melody)')
    parser.add_argument('--voice-strength', type=float, default=0.8,
                       help='Voice leading strength (0.0-1.0)')
    parser.add_argument('--voice-tightness', type=float, default=0.5,
                       help='Voicing tightness (0.0=loose/wide, 1.0=tight/close, default 0.5)')
    parser.add_argument('--smooth', action='store_true',
                       help='Smooth/legato mode: hold harmony notes when voicing stays same')
    parser.add_argument('--multitrack', action='store_true',
                       help='Multitrack mode: create separate MIDI track for each voice')
    parser.add_argument('--approach', choices=['diatonic', 'chromatic', 'dominant', 'diminished'],
                       help='Approach chord style: diatonic, chromatic, dominant, or diminished')
    parser.add_argument('--max-range', type=int, default=18,
                       help='Maximum voicing range from lowest to highest note (semitones)')

    args = parser.parse_args()

    # Parse key/scale context
    scale_context = None
    if args.key:
        is_minor = 'm' in args.key.lower()
        root_note = _parse_chord_root(args.key)
        scale_type = 'minor' if is_minor else 'major'
        scale_context = ScaleContext(root_note, scale_type)

    # Parse chord progression
    chord_progression = {}
    if args.chords:
        for pair in args.chords.split(','):
            beat, chord = pair.split(':')
            chord_progression[int(beat)] = chord.strip()
    else:
        # Default: C major for whole melody
        chord_progression = {0: 'C'}
        # Auto-set key context if not specified
        if not scale_context:
            scale_context = ScaleContext(60, 'major')  # C major

    # Output path
    output_path = args.output
    if not output_path:
        input_path = Path(args.input)
        suffix = args.voicing
        if args.no_extensions:
            suffix += "_diatonic"
        if args.no_doubling:
            suffix += "_nodbl"
        if args.rootless:
            suffix += "_rootless"
        output_path = input_path.parent / f"{input_path.stem}_harmonized_{suffix}.mid"

    # Harmonize
    harmonize_melody_improved(
        args.input,
        str(output_path),
        chord_progression=chord_progression,
        voicing_type=args.voicing,
        min_interval=args.min_interval,
        max_interval=args.max_interval,
        use_extensions=not args.no_extensions,
        allow_doubling=not args.no_doubling,
        rootless=args.rootless,
        voice_direction_angle=args.voice_angle,
        voice_direction_strength=args.voice_strength,
        voice_tightness=args.voice_tightness,
        smooth_mode=args.smooth,
        multitrack_mode=args.multitrack,
        approach_mode=args.approach,
        scale_context=scale_context,
        max_voicing_range=args.max_range
    )

    print(f"✅ Done! Output: {output_path}")
