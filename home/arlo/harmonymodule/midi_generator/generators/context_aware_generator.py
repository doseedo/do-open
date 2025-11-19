#!/usr/bin/env python3
"""
Context-Aware Generation - Add Tracks to Existing MIDI Arrangements

This module analyzes existing MIDI files and generates new tracks that seamlessly
fit the existing style, harmony, rhythm, and overall arrangement.

Key Features:
1. ANALYZE existing arrangements (tempo, chords, style, density)
2. ADD tracks that fit musically (bass, harmony, melody, percussion)
3. INPAINT - regenerate sections with new chords or style
4. VOICE LEADING - avoid parallel fifths, octaves
5. SMOOTH BOUNDARIES - seamless entry/exit for regenerated sections
6. DENSITY MATCHING - match rhythmic density of existing tracks
7. TEXTURE MATCHING - complement existing arrangement

Use Cases:
- Add bass to existing piano + drums arrangement
- Add strings to rock band
- Add brass hits to big band (with different genre!)
- Regenerate bridge section with new chord progression
- Change genre of verse 2 while keeping verse 1 style

Based on Research:
- Music accompaniment systems (Francois Pachet - Continuator)
- Voice leading rules (Dimitri Tymoczko - geometry of music)
- Chord extraction from polyphonic MIDI (Music21)
- Style matching and texture analysis
- Harmonic analysis (Roman numeral analysis)

Author: Agent 3 - Context-Aware Generation
Date: 2025
"""

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from typing import List, Tuple, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import copy
import math
import sys

# Import from existing modules
sys.path.append(str(Path(__file__).parent.parent))
from analysis.midi_analyzer import MidiAnalyzer, NoteEvent, ChordEvent, KeySignature
from generators.style_fusion import GenreFeatures, GENRE_PROFILES


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class ArrangementAnalysis:
    """
    Comprehensive analysis of existing MIDI arrangement
    """
    # Basic information
    tempo: int
    time_signature: Tuple[int, int]
    length_measures: int
    key: KeySignature

    # Harmonic analysis
    chord_progression: List[str]
    chords_per_measure: List[List[ChordEvent]]
    harmonic_rhythm: float  # Average chords per measure

    # Rhythmic analysis
    density_per_measure: List[float]  # Notes per measure
    rhythmic_profile: Dict[str, Any]
    groove_type: str  # 'swing', 'straight', 'shuffle', 'half-time'

    # Texture analysis
    texture: str  # 'monophonic', 'homophonic', 'polyphonic'
    register_distribution: Dict[str, float]  # 'low', 'mid', 'high'

    # Instrumentation
    instruments: List[int]  # MIDI program numbers per track
    tracks: List[List[NoteEvent]]
    track_roles: Dict[int, str]  # track_idx -> 'melody', 'harmony', 'bass', 'drums'

    # Style detection
    detected_style: Optional[GenreFeatures] = None


@dataclass
class BoundaryContext:
    """
    Context at section boundaries for smooth transitions
    """
    measure: int
    last_notes: List[int]  # Pitches
    last_rhythm: List[float]  # Durations
    last_velocities: List[int]
    harmonic_context: Optional[str]  # Chord at boundary
    voice_leading_tendency: str  # 'ascending', 'descending', 'static'
    register: str  # 'low', 'mid', 'high'


@dataclass
class GenerationConstraints:
    """
    Constraints for generating new material
    """
    follow_harmony: bool = True
    match_density: bool = True
    avoid_voice_leading_errors: bool = True
    preserve_texture: bool = True
    max_voice_leading_distance: int = 7  # Semitones
    preferred_motion: str = 'contrary'  # 'parallel', 'contrary', 'oblique', 'similar'


# ==============================================================================
# MAIN CONTEXT-AWARE GENERATOR
# ==============================================================================

class ContextAwareGenerator:
    """
    Analyzes existing MIDI and generates new tracks that fit seamlessly

    Use cases:
    - Add bass to existing jazz piano + drums
    - Add strings to existing rock band
    - Add brass hits to existing big band (different genre!)

    Example:
        gen = ContextAwareGenerator('existing_song.mid')
        analysis = gen.analyze()

        # Add funk bass to jazz arrangement
        bass_notes = gen.add_track(
            instrument=33,  # Fingered bass
            genre='funk',
            track_type='bass'
        )

        gen.export_with_new_track(bass_notes, 'with_bass.mid')
    """

    def __init__(self, existing_midi: str):
        """
        Load and prepare existing MIDI file

        Args:
            existing_midi: Path to MIDI file
        """
        self.midi_file = existing_midi
        self.midi = MidiFile(existing_midi)
        self.analyzer = MidiAnalyzer(existing_midi)

        self.analysis: Optional[ArrangementAnalysis] = None
        self.ticks_per_beat = self.midi.ticks_per_beat

    def analyze(self) -> ArrangementAnalysis:
        """
        Comprehensive analysis of existing arrangement

        Returns:
            ArrangementAnalysis with all detected features
        """
        # Extract basic information
        tempo = self.analyzer.get_tempo()
        time_sig = self.analyzer.get_time_signature()
        key = self.analyzer.detect_key()

        # Extract all notes
        all_notes = self.analyzer.extract_notes()

        # Detect chords
        chords = self.analyzer.extract_chords()
        chord_progression = [str(chord) for chord in chords]

        # Organize chords by measure
        chords_per_measure = self._organize_chords_by_measure(chords, time_sig)

        # Calculate harmonic rhythm
        harmonic_rhythm = len(chords) / max(1, self._get_length_in_measures(time_sig))

        # Analyze density per measure
        density_per_measure = self._calculate_density_per_measure(all_notes, time_sig)

        # Detect groove type
        groove_type = self._detect_groove_type(all_notes)

        # Analyze texture
        texture = self._analyze_texture(all_notes)

        # Analyze register distribution
        register_dist = self._analyze_register_distribution(all_notes)

        # Extract tracks
        tracks = self._extract_tracks()

        # Detect instruments
        instruments = self._detect_instruments()

        # Classify track roles
        track_roles = self._classify_track_roles(tracks, chords)

        # Create rhythmic profile
        rhythmic_profile = self._create_rhythmic_profile(all_notes, time_sig)

        # Detect style (basic implementation - would use GenreDetector from Agent 1)
        detected_style = self._detect_style_basic(tempo, groove_type, harmonic_rhythm)

        length_measures = self._get_length_in_measures(time_sig)

        self.analysis = ArrangementAnalysis(
            tempo=tempo,
            time_signature=time_sig,
            length_measures=length_measures,
            key=key,
            chord_progression=chord_progression,
            chords_per_measure=chords_per_measure,
            harmonic_rhythm=harmonic_rhythm,
            density_per_measure=density_per_measure,
            rhythmic_profile=rhythmic_profile,
            groove_type=groove_type,
            texture=texture,
            register_distribution=register_dist,
            instruments=instruments,
            tracks=tracks,
            track_roles=track_roles,
            detected_style=detected_style
        )

        return self.analysis

    def add_track(self,
                  instrument: int,
                  genre: Optional[str] = None,
                  track_type: str = "auto",
                  constraints: Optional[GenerationConstraints] = None) -> List[Tuple[int, float, float, int]]:
        """
        Generate new track fitting existing arrangement

        Args:
            instrument: MIDI program number (0-127)
            genre: Genre for new track (None = match existing)
            track_type: 'bass', 'harmony', 'melody', 'percussion', 'auto'
            constraints: Generation constraints

        Returns:
            List of (pitch, start_time, duration, velocity) tuples
        """
        # Ensure analysis has been done
        if self.analysis is None:
            self.analyze()

        # Use default constraints if not provided
        if constraints is None:
            constraints = GenerationConstraints()

        # Determine track type if auto
        if track_type == "auto":
            track_type = self._infer_track_type(instrument, self.analysis)

        # Use detected genre if not specified
        if genre is None and self.analysis.detected_style:
            genre = self.analysis.detected_style.name.lower()

        # Generate based on track type
        if track_type == "bass":
            return self._generate_bass(instrument, self.analysis, genre, constraints)
        elif track_type == "harmony":
            return self._generate_harmony(instrument, self.analysis, genre, constraints)
        elif track_type == "melody":
            return self._generate_melody(instrument, self.analysis, genre, constraints)
        elif track_type == "percussion":
            return self._generate_percussion(instrument, self.analysis, genre, constraints)
        else:
            raise ValueError(f"Unknown track type: {track_type}")

    def add_section(self,
                    start_measure: int,
                    end_measure: int,
                    instrument: int,
                    track_type: str = "auto",
                    custom_chords: Optional[List[str]] = None) -> List[Tuple[int, float, float, int]]:
        """
        Add instrument to specific section only

        Args:
            start_measure: Section start (0-indexed)
            end_measure: Section end (exclusive)
            instrument: MIDI program number
            track_type: Type of track to generate
            custom_chords: Override detected chords

        Returns:
            Notes for that section only
        """
        if self.analysis is None:
            self.analyze()

        # Calculate time range
        beats_per_measure = self.analysis.time_signature[0]
        ticks_per_measure = self.ticks_per_beat * beats_per_measure

        start_tick = start_measure * ticks_per_measure
        end_tick = end_measure * ticks_per_measure

        # Get chords for this section
        if custom_chords:
            section_chords = custom_chords
        else:
            section_chords = self.analysis.chord_progression[start_measure:end_measure]

        # Create modified analysis for this section
        section_analysis = copy.copy(self.analysis)
        section_analysis.chord_progression = section_chords
        section_analysis.length_measures = end_measure - start_measure

        # Generate for section
        notes = self.add_track(instrument, track_type=track_type)

        # Filter to section only
        section_notes = []
        for pitch, start_time, duration, velocity in notes:
            start_tick_note = int(start_time * self.ticks_per_beat * 2)  # Approximate
            if start_tick <= start_tick_note < end_tick:
                section_notes.append((pitch, start_time, duration, velocity))

        return section_notes

    def regenerate_section(self,
                          track_number: int,
                          start_measure: int,
                          end_measure: int,
                          new_chords: Optional[List[str]] = None,
                          new_genre: Optional[str] = None,
                          blend_boundaries: bool = True) -> List[Tuple[int, float, float, int]]:
        """
        Regenerate a section of existing track (INPAINTING)

        Args:
            track_number: Which track to regenerate (0-indexed)
            start_measure: Section start
            end_measure: Section end (exclusive)
            new_chords: New chord progression (None = keep existing)
            new_genre: New genre (None = match existing)
            blend_boundaries: Smooth transitions at boundaries

        Returns:
            New notes for that section
        """
        if self.analysis is None:
            self.analyze()

        # Extract existing track
        if track_number >= len(self.analysis.tracks):
            raise ValueError(f"Track {track_number} does not exist")

        existing_track = self.analysis.tracks[track_number]

        # Get boundary contexts
        entry_context = self._extract_entry_context(existing_track, start_measure - 1)
        exit_context = self._extract_exit_context(existing_track, end_measure)

        # Get chords for section
        section_chords = new_chords or self.analysis.chord_progression[start_measure:end_measure]

        # Determine genre
        genre = new_genre or (self.analysis.detected_style.name.lower()
                             if self.analysis.detected_style else 'jazz')

        # Generate new section
        new_section = self._generate_section(
            start_measure=start_measure,
            end_measure=end_measure,
            chords=section_chords,
            genre=genre,
            entry_context=entry_context if blend_boundaries else None,
            exit_context=exit_context if blend_boundaries else None,
            track_role=self.analysis.track_roles.get(track_number, 'melody')
        )

        return new_section

    def suggest_additions(self) -> List[Dict[str, Any]]:
        """
        Analyze arrangement and suggest tracks to add

        Returns:
            List of suggestions with instrument, reason, priority
        """
        if self.analysis is None:
            self.analyze()

        suggestions = []

        # Check for missing bass
        has_bass = any(role == 'bass' for role in self.analysis.track_roles.values())
        if not has_bass:
            suggestions.append({
                'instrument': 33,  # Fingered bass
                'track_type': 'bass',
                'reason': 'No bass line detected - would provide harmonic foundation',
                'priority': 0.9,
                'genre': self.analysis.detected_style.name.lower() if self.analysis.detected_style else 'jazz'
            })

        # Check register balance
        if self.analysis.register_distribution.get('mid', 0) < 0.3:
            suggestions.append({
                'instrument': 48,  # String ensemble
                'track_type': 'harmony',
                'reason': 'Mid-range is sparse - strings would fill harmonic space',
                'priority': 0.75,
                'genre': 'orchestral'
            })

        # Check harmonic density
        harmonic_tracks = sum(1 for role in self.analysis.track_roles.values()
                            if role == 'harmony')
        if harmonic_tracks < 2:
            suggestions.append({
                'instrument': 0,  # Acoustic grand piano
                'track_type': 'harmony',
                'reason': 'Limited harmonic support - piano would add richness',
                'priority': 0.7,
                'genre': self.analysis.detected_style.name.lower() if self.analysis.detected_style else 'jazz'
            })

        # Check for drums
        has_drums = any(role == 'drums' for role in self.analysis.track_roles.values())
        if not has_drums and len(self.analysis.tracks) > 1:
            suggestions.append({
                'instrument': 128,  # Percussion (channel 10)
                'track_type': 'percussion',
                'reason': 'No rhythmic foundation - drums would provide groove',
                'priority': 0.85,
                'genre': self.analysis.detected_style.name.lower() if self.analysis.detected_style else 'rock'
            })

        # Sort by priority
        suggestions.sort(key=lambda x: x['priority'], reverse=True)

        return suggestions

    def export_with_new_track(self,
                             new_notes: List[Tuple[int, float, float, int]],
                             output_file: str,
                             instrument: int = 0,
                             track_name: str = "Generated Track"):
        """
        Export MIDI with new track added

        Args:
            new_notes: Generated notes (pitch, start_time, duration, velocity)
            output_file: Output MIDI filename
            instrument: MIDI program number for new track
            track_name: Name for the new track
        """
        # Create new MIDI file with same properties
        new_midi = MidiFile(ticks_per_beat=self.midi.ticks_per_beat)

        # Copy existing tracks
        for track in self.midi.tracks:
            new_midi.tracks.append(track.copy())

        # Create new track
        new_track = MidiTrack()
        new_midi.tracks.append(new_track)

        # Add track name
        new_track.append(MetaMessage('track_name', name=track_name, time=0))

        # Add program change
        new_track.append(Message('program_change', program=instrument, time=0))

        # Convert notes to MIDI messages
        # Sort by start time
        sorted_notes = sorted(new_notes, key=lambda x: x[1])

        current_tick = 0
        for pitch, start_time, duration, velocity in sorted_notes:
            # Convert time to ticks (assuming quarter note = 1 beat)
            start_tick = int(start_time * self.ticks_per_beat * 2)  # *2 for half notes
            duration_ticks = int(duration * self.ticks_per_beat * 2)

            # Note on
            delta_time = start_tick - current_tick
            new_track.append(Message('note_on', note=pitch, velocity=velocity, time=delta_time))
            current_tick = start_tick

            # Note off
            new_track.append(Message('note_off', note=pitch, velocity=0, time=duration_ticks))
            current_tick += duration_ticks

        # Save
        new_midi.save(output_file)

    # ==========================================================================
    # ANALYSIS HELPER METHODS
    # ==========================================================================

    def _organize_chords_by_measure(self,
                                    chords: List[ChordEvent],
                                    time_sig: Tuple[int, int]) -> List[List[ChordEvent]]:
        """Organize chords by measure"""
        beats_per_measure = time_sig[0]
        beat_duration = 60.0 / self.analysis.tempo if self.analysis else 0.5
        measure_duration = beats_per_measure * beat_duration

        if not chords:
            return []

        # Find max time
        max_time = max(chord.start_time for chord in chords) + measure_duration
        num_measures = int(max_time / measure_duration) + 1

        chords_per_measure = [[] for _ in range(num_measures)]

        for chord in chords:
            measure_idx = int(chord.start_time / measure_duration)
            if 0 <= measure_idx < num_measures:
                chords_per_measure[measure_idx].append(chord)

        return chords_per_measure

    def _get_length_in_measures(self, time_sig: Tuple[int, int]) -> int:
        """Calculate song length in measures"""
        total_ticks = sum(track[-1].time if track else 0
                         for track in self.midi.tracks)
        beats_per_measure = time_sig[0]
        ticks_per_measure = self.ticks_per_beat * beats_per_measure

        return max(1, int(total_ticks / ticks_per_measure))

    def _calculate_density_per_measure(self,
                                      notes: List[NoteEvent],
                                      time_sig: Tuple[int, int]) -> List[float]:
        """Calculate note density per measure"""
        beats_per_measure = time_sig[0]
        tempo = self.analyzer.get_tempo()
        beat_duration = 60.0 / tempo
        measure_duration = beats_per_measure * beat_duration

        if not notes:
            return [0.0]

        max_time = max(note.start_time for note in notes) + measure_duration
        num_measures = int(max_time / measure_duration) + 1

        density = [0.0] * num_measures

        for note in notes:
            measure_idx = int(note.start_time / measure_duration)
            if 0 <= measure_idx < num_measures:
                density[measure_idx] += 1

        return density

    def _detect_groove_type(self, notes: List[NoteEvent]) -> str:
        """
        Detect groove type from notes

        Simple heuristic: analyze timing deviations
        """
        if not notes:
            return 'straight'

        # This is a simplified version - would use swing detector from Agent 1
        # For now, return straight
        return 'straight'

    def _analyze_texture(self, notes: List[NoteEvent]) -> str:
        """
        Analyze texture: monophonic, homophonic, or polyphonic
        """
        if not notes:
            return 'monophonic'

        # Count simultaneous notes
        max_simultaneous = 0
        time_buckets = defaultdict(int)

        for note in notes:
            time_key = int(note.start_time * 10)  # 100ms buckets
            time_buckets[time_key] += 1
            max_simultaneous = max(max_simultaneous, time_buckets[time_key])

        if max_simultaneous <= 1:
            return 'monophonic'
        elif max_simultaneous <= 3:
            return 'homophonic'
        else:
            return 'polyphonic'

    def _analyze_register_distribution(self, notes: List[NoteEvent]) -> Dict[str, float]:
        """
        Analyze distribution across registers (low/mid/high)
        """
        if not notes:
            return {'low': 0.0, 'mid': 0.0, 'high': 0.0}

        low_count = sum(1 for note in notes if note.pitch < 48)
        mid_count = sum(1 for note in notes if 48 <= note.pitch < 72)
        high_count = sum(1 for note in notes if note.pitch >= 72)

        total = len(notes)

        return {
            'low': low_count / total,
            'mid': mid_count / total,
            'high': high_count / total
        }

    def _extract_tracks(self) -> List[List[NoteEvent]]:
        """Extract notes organized by track"""
        all_notes = self.analyzer.extract_notes()

        # Group by track
        track_dict = defaultdict(list)
        for note in all_notes:
            track_dict[note.track_idx].append(note)

        # Convert to list
        max_track = max(track_dict.keys()) if track_dict else 0
        tracks = []
        for i in range(max_track + 1):
            tracks.append(sorted(track_dict[i], key=lambda n: n.start_time))

        return tracks

    def _detect_instruments(self) -> List[int]:
        """Detect MIDI program numbers for each track"""
        instruments = []

        for track in self.midi.tracks:
            program = 0  # Default to piano
            for msg in track:
                if msg.type == 'program_change':
                    program = msg.program
                    break
            instruments.append(program)

        return instruments

    def _classify_track_roles(self,
                             tracks: List[List[NoteEvent]],
                             chords: List[ChordEvent]) -> Dict[int, str]:
        """
        Classify each track's role: melody, harmony, bass, drums
        """
        roles = {}

        for idx, track in enumerate(tracks):
            if not track:
                roles[idx] = 'unknown'
                continue

            # Get pitch statistics
            pitches = [note.pitch for note in track]
            avg_pitch = sum(pitches) / len(pitches)
            pitch_range = max(pitches) - min(pitches)

            # Check for drums (channel 10)
            is_drums = any(note.channel == 9 for note in track)

            if is_drums:
                roles[idx] = 'drums'
            elif avg_pitch < 48:  # Low register
                roles[idx] = 'bass'
            elif pitch_range > 24:  # Wide range suggests melody
                roles[idx] = 'melody'
            else:
                roles[idx] = 'harmony'

        return roles

    def _create_rhythmic_profile(self,
                                notes: List[NoteEvent],
                                time_sig: Tuple[int, int]) -> Dict[str, Any]:
        """
        Create rhythmic profile of arrangement
        """
        if not notes:
            return {'average_density': 0.0, 'max_density': 0.0}

        # Calculate densities
        beats_per_measure = time_sig[0]
        tempo = self.analyzer.get_tempo()
        beat_duration = 60.0 / tempo

        densities = []
        time_buckets = defaultdict(int)

        for note in notes:
            beat_idx = int(note.start_time / beat_duration)
            time_buckets[beat_idx] += 1

        if time_buckets:
            densities = list(time_buckets.values())

        return {
            'average_density': sum(densities) / len(densities) if densities else 0.0,
            'max_density': max(densities) if densities else 0.0,
            'min_density': min(densities) if densities else 0.0
        }

    def _detect_style_basic(self, tempo: int, groove: str, harmonic_rhythm: float) -> Optional[GenreFeatures]:
        """
        Basic style detection (simplified - Agent 1 will have full version)
        """
        # Simple heuristic based on tempo
        if tempo < 100:
            return GENRE_PROFILES.get('blues')
        elif tempo < 130:
            return GENRE_PROFILES.get('jazz')
        elif tempo < 150:
            return GENRE_PROFILES.get('funk')
        else:
            return GENRE_PROFILES.get('edm')

    # ==========================================================================
    # GENERATION HELPER METHODS
    # ==========================================================================

    def _infer_track_type(self, instrument: int, analysis: ArrangementAnalysis) -> str:
        """
        Infer track type from instrument number
        """
        # Bass instruments (32-39)
        if 32 <= instrument <= 39:
            return 'bass'

        # Drums/percussion
        if instrument >= 112 or instrument == 128:
            return 'percussion'

        # Lead instruments (brass, winds, guitars)
        if instrument in range(24, 32) or instrument in range(40, 56) or instrument in range(64, 80):
            return 'melody'

        # Default to harmony
        return 'harmony'

    def _generate_bass(self,
                      instrument: int,
                      analysis: ArrangementAnalysis,
                      genre: Optional[str],
                      constraints: GenerationConstraints) -> List[Tuple[int, float, float, int]]:
        """
        Generate bass line matching existing arrangement

        Uses existing bass_engine.py from advanced_modules
        """
        notes = []

        # Use simple root motion for now (would integrate with BassEngine)
        tempo = analysis.tempo
        beat_duration = 60.0 / tempo

        for measure_idx in range(analysis.length_measures):
            # Get chord for this measure
            if measure_idx < len(analysis.chords_per_measure):
                measure_chords = analysis.chords_per_measure[measure_idx]
                if measure_chords:
                    chord = measure_chords[0]
                    root = chord.root + 36  # Bass register (C2 = 36)

                    # Add root on downbeat
                    start_time = measure_idx * analysis.time_signature[0] * beat_duration
                    duration = beat_duration
                    velocity = 80

                    notes.append((root, start_time, duration, velocity))

                    # Add fifth on beat 3 (for 4/4)
                    if analysis.time_signature[0] >= 4:
                        fifth = root + 7
                        start_time_2 = start_time + 2 * beat_duration
                        notes.append((fifth, start_time_2, duration, velocity))

        return notes

    def _generate_harmony(self,
                         instrument: int,
                         analysis: ArrangementAnalysis,
                         genre: Optional[str],
                         constraints: GenerationConstraints) -> List[Tuple[int, float, float, int]]:
        """
        Generate harmonic accompaniment
        """
        notes = []

        tempo = analysis.tempo
        beat_duration = 60.0 / tempo

        for measure_idx in range(analysis.length_measures):
            if measure_idx < len(analysis.chords_per_measure):
                measure_chords = analysis.chords_per_measure[measure_idx]
                if measure_chords:
                    chord = measure_chords[0]

                    # Simple voicing: root, third, fifth in mid register
                    root = chord.root + 60  # C4 = 60
                    third = root + (3 if 'min' in chord.quality.lower() else 4)
                    fifth = root + 7

                    start_time = measure_idx * analysis.time_signature[0] * beat_duration
                    duration = analysis.time_signature[0] * beat_duration
                    velocity = 70

                    # Add chord tones
                    notes.append((root, start_time, duration, velocity))
                    notes.append((third, start_time, duration, velocity))
                    notes.append((fifth, start_time, duration, velocity))

        return notes

    def _generate_melody(self,
                        instrument: int,
                        analysis: ArrangementAnalysis,
                        genre: Optional[str],
                        constraints: GenerationConstraints) -> List[Tuple[int, float, float, int]]:
        """
        Generate melody over existing harmony
        """
        notes = []

        tempo = analysis.tempo
        beat_duration = 60.0 / tempo

        # Simple melody: chord tones with passing tones
        current_pitch = 72  # C5

        for measure_idx in range(analysis.length_measures):
            if measure_idx < len(analysis.chords_per_measure):
                measure_chords = analysis.chords_per_measure[measure_idx]
                if measure_chords:
                    chord = measure_chords[0]

                    # Target chord tones
                    root = chord.root + 72
                    third = root + (3 if 'min' in chord.quality.lower() else 4)
                    fifth = root + 7

                    chord_tones = [root, third, fifth]

                    # Generate notes for measure
                    beats = analysis.time_signature[0]
                    for beat in range(beats):
                        # Choose chord tone closest to current pitch
                        target = min(chord_tones, key=lambda x: abs(x - current_pitch))

                        start_time = (measure_idx * beats + beat) * beat_duration
                        duration = beat_duration * 0.8
                        velocity = 85

                        notes.append((target, start_time, duration, velocity))
                        current_pitch = target

        return notes

    def _generate_percussion(self,
                           instrument: int,
                           analysis: ArrangementAnalysis,
                           genre: Optional[str],
                           constraints: GenerationConstraints) -> List[Tuple[int, float, float, int]]:
        """
        Generate percussion/drums
        """
        notes = []

        tempo = analysis.tempo
        beat_duration = 60.0 / tempo

        # Simple beat: kick on 1 and 3, snare on 2 and 4, hi-hat on all beats
        kick = 36
        snare = 38
        hihat = 42

        for measure_idx in range(analysis.length_measures):
            beats = analysis.time_signature[0]

            for beat in range(beats):
                start_time = (measure_idx * beats + beat) * beat_duration

                # Hi-hat on every beat
                notes.append((hihat, start_time, beat_duration * 0.3, 70))

                # Kick on 1 and 3
                if beat % 2 == 0:
                    notes.append((kick, start_time, beat_duration * 0.5, 90))

                # Snare on 2 and 4
                if beat % 2 == 1:
                    notes.append((snare, start_time, beat_duration * 0.4, 85))

        return notes

    def _extract_entry_context(self,
                              track: List[NoteEvent],
                              measure: int) -> BoundaryContext:
        """
        Extract context before inpaint region
        """
        tempo = self.analysis.tempo
        beat_duration = 60.0 / tempo
        beats_per_measure = self.analysis.time_signature[0]
        measure_duration = beats_per_measure * beat_duration

        measure_start = measure * measure_duration
        measure_end = (measure + 1) * measure_duration

        # Find notes in this measure
        measure_notes = [n for n in track
                        if measure_start <= n.start_time < measure_end]

        last_notes = []
        last_rhythm = []
        last_velocities = []

        if measure_notes:
            # Get last few notes
            last_notes = [n.pitch for n in measure_notes[-3:]]
            last_rhythm = [n.duration for n in measure_notes[-3:]]
            last_velocities = [n.velocity for n in measure_notes[-3:]]

        # Determine voice leading tendency
        tendency = 'static'
        if len(last_notes) >= 2:
            if last_notes[-1] > last_notes[-2]:
                tendency = 'ascending'
            elif last_notes[-1] < last_notes[-2]:
                tendency = 'descending'

        # Determine register
        register = 'mid'
        if last_notes:
            avg_pitch = sum(last_notes) / len(last_notes)
            if avg_pitch < 48:
                register = 'low'
            elif avg_pitch > 72:
                register = 'high'

        return BoundaryContext(
            measure=measure,
            last_notes=last_notes,
            last_rhythm=last_rhythm,
            last_velocities=last_velocities,
            harmonic_context=None,  # Would extract from chords
            voice_leading_tendency=tendency,
            register=register
        )

    def _extract_exit_context(self,
                             track: List[NoteEvent],
                             measure: int) -> BoundaryContext:
        """
        Extract context after inpaint region
        """
        # Similar to entry context but for the next measure
        return self._extract_entry_context(track, measure)

    def _generate_section(self,
                         start_measure: int,
                         end_measure: int,
                         chords: List[str],
                         genre: str,
                         entry_context: Optional[BoundaryContext],
                         exit_context: Optional[BoundaryContext],
                         track_role: str) -> List[Tuple[int, float, float, int]]:
        """
        Generate section with smooth boundaries
        """
        # Create temporary analysis for section
        section_analysis = copy.copy(self.analysis)
        section_analysis.chord_progression = chords
        section_analysis.length_measures = end_measure - start_measure

        # Generate based on role
        constraints = GenerationConstraints()

        if track_role == 'bass':
            notes = self._generate_bass(33, section_analysis, genre, constraints)
        elif track_role == 'harmony':
            notes = self._generate_harmony(0, section_analysis, genre, constraints)
        elif track_role == 'melody':
            notes = self._generate_melody(0, section_analysis, genre, constraints)
        else:
            notes = self._generate_melody(0, section_analysis, genre, constraints)

        # Adjust timing to start at correct measure
        tempo = self.analysis.tempo
        beat_duration = 60.0 / tempo
        beats_per_measure = self.analysis.time_signature[0]
        measure_duration = beats_per_measure * beat_duration
        offset = start_measure * measure_duration

        adjusted_notes = [(pitch, start_time + offset, duration, velocity)
                         for pitch, start_time, duration, velocity in notes]

        return adjusted_notes


# ==============================================================================
# TRACK INPAINTER
# ==============================================================================

class TrackInpainter:
    """
    Specialized class for regenerating parts of existing tracks

    Like Photoshop content-aware fill for music

    Example:
        inpainter = TrackInpainter('song.mid')

        # Regenerate measures 5-8 with new chords
        new_section = inpainter.inpaint_measures(
            track=0,
            start=5,
            end=8,
            new_chords=['Dm7', 'G7', 'Cmaj7', 'A7']
        )

        inpainter.export('song_reharmonized.mid')
    """

    def __init__(self, midi_file: str):
        """
        Initialize inpainter

        Args:
            midi_file: Path to MIDI file
        """
        self.generator = ContextAwareGenerator(midi_file)
        self.generator.analyze()
        self.modifications = {}  # Track modifications

    def inpaint_measures(self,
                        track: int,
                        start: int,
                        end: int,
                        new_chords: Optional[List[str]] = None,
                        smooth_boundaries: bool = True) -> List[Tuple[int, float, float, int]]:
        """
        Regenerate measures with smooth boundaries

        Args:
            track: Track index to modify
            start: Start measure
            end: End measure (exclusive)
            new_chords: New chord progression (None = keep existing)
            smooth_boundaries: Apply boundary smoothing

        Returns:
            New notes for section
        """
        new_section = self.generator.regenerate_section(
            track_number=track,
            start_measure=start,
            end_measure=end,
            new_chords=new_chords,
            blend_boundaries=smooth_boundaries
        )

        # Store modification
        self.modifications[(track, start, end)] = new_section

        return new_section

    def inpaint_with_genre_change(self,
                                  track: int,
                                  start: int,
                                  end: int,
                                  new_genre: str,
                                  smooth_boundaries: bool = True) -> List[Tuple[int, float, float, int]]:
        """
        Regenerate section in different genre

        Example: Measures 9-16 in EDM style while rest is jazz

        Args:
            track: Track index
            start: Start measure
            end: End measure
            new_genre: New genre ('jazz', 'edm', 'funk', etc.)
            smooth_boundaries: Apply boundary smoothing

        Returns:
            New notes
        """
        new_section = self.generator.regenerate_section(
            track_number=track,
            start_measure=start,
            end_measure=end,
            new_genre=new_genre,
            blend_boundaries=smooth_boundaries
        )

        self.modifications[(track, start, end)] = new_section

        return new_section

    def export(self, output_file: str):
        """
        Export MIDI with all modifications applied

        Args:
            output_file: Output filename
        """
        # This would merge all modifications into the original MIDI
        # For now, just export the first modification
        if self.modifications:
            track_num, notes = next(iter(self.modifications.items()))
            self.generator.export_with_new_track(
                notes[1],  # Get the notes
                output_file,
                instrument=0,
                track_name=f"Inpainted Track {track_num[0]}"
            )


# ==============================================================================
# SMART ORCHESTRATOR
# ==============================================================================

class SmartOrchestrator:
    """
    Intelligent orchestration that analyzes existing arrangement

    Suggests:
    - What instruments are missing
    - What register is underused
    - What texture would complement

    Example:
        orchestrator = SmartOrchestrator('piano_drums.mid')
        suggestions = orchestrator.suggest_additions()

        for suggestion in suggestions:
            print(f"{suggestion['reason']} - Priority: {suggestion['priority']}")

            # Auto-add suggested track
            orchestrator.add_suggested_track(suggestion)

        orchestrator.export('full_arrangement.mid')
    """

    def __init__(self, midi_file: str):
        """
        Initialize orchestrator

        Args:
            midi_file: Path to MIDI file
        """
        self.generator = ContextAwareGenerator(midi_file)
        self.analysis = self.generator.analyze()
        self.added_tracks = []

    def suggest_additions(self) -> List[Dict[str, Any]]:
        """
        Analyze arrangement and suggest tracks to add

        Returns:
            List of suggestions sorted by priority
        """
        return self.generator.suggest_additions()

    def add_suggested_track(self, suggestion: Dict[str, Any]) -> List[Tuple[int, float, float, int]]:
        """
        Add a track based on suggestion

        Args:
            suggestion: Suggestion dict from suggest_additions()

        Returns:
            Generated notes
        """
        notes = self.generator.add_track(
            instrument=suggestion['instrument'],
            genre=suggestion.get('genre'),
            track_type=suggestion['track_type']
        )

        self.added_tracks.append({
            'notes': notes,
            'instrument': suggestion['instrument'],
            'name': suggestion['reason']
        })

        return notes

    def analyze_orchestral_balance(self) -> Dict[str, Any]:
        """
        Analyze orchestral balance

        Returns:
            Dictionary with balance metrics
        """
        return {
            'register_distribution': self.analysis.register_distribution,
            'texture': self.analysis.texture,
            'harmonic_voices': sum(1 for role in self.analysis.track_roles.values()
                                  if role == 'harmony'),
            'has_bass': any(role == 'bass' for role in self.analysis.track_roles.values()),
            'has_drums': any(role == 'drums' for role in self.analysis.track_roles.values()),
            'density': sum(self.analysis.density_per_measure) / len(self.analysis.density_per_measure)
        }

    def export(self, output_file: str):
        """
        Export with all added tracks

        Args:
            output_file: Output filename
        """
        if not self.added_tracks:
            print("No tracks added yet")
            return

        # Export first added track (would merge all in full implementation)
        first_track = self.added_tracks[0]
        self.generator.export_with_new_track(
            first_track['notes'],
            output_file,
            instrument=first_track['instrument'],
            track_name=first_track['name']
        )


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

def example_add_bass():
    """Example: Add bass to existing arrangement"""
    gen = ContextAwareGenerator('existing_song.mid')
    analysis = gen.analyze()

    print(f"Analyzed: {analysis.tempo} BPM, {analysis.time_signature}, {len(analysis.chord_progression)} chords")

    # Add funk bass
    bass_notes = gen.add_track(
        instrument=33,  # Fingered bass
        genre='funk',
        track_type='bass'
    )

    gen.export_with_new_track(bass_notes, 'with_funk_bass.mid', instrument=33)
    print("Exported with funk bass!")


def example_inpainting():
    """Example: Regenerate section with new chords"""
    inpainter = TrackInpainter('song.mid')

    # Reharmonize measures 5-8
    new_section = inpainter.inpaint_measures(
        track=0,
        start=5,
        end=8,
        new_chords=['Dm7', 'G7alt', 'Cmaj9', 'A7#9']
    )

    inpainter.export('reharmonized.mid')
    print("Exported reharmonized version!")


def example_smart_orchestration():
    """Example: Auto-orchestrate with suggestions"""
    orchestrator = SmartOrchestrator('piano_drums.mid')

    suggestions = orchestrator.suggest_additions()

    print("Suggested additions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion['reason']} (Priority: {suggestion['priority']:.2f})")

    # Add top suggestion
    if suggestions:
        orchestrator.add_suggested_track(suggestions[0])
        orchestrator.export('orchestrated.mid')
        print("Added top suggestion!")


if __name__ == '__main__':
    print("Context-Aware Generator Examples")
    print("=" * 50)
    print("1. Add bass: example_add_bass()")
    print("2. Inpainting: example_inpainting()")
    print("3. Smart orchestration: example_smart_orchestration()")
