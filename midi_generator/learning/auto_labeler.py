#!/usr/bin/env python3
"""
Auto-Labeling Pipeline for MIDI Corpus
Agent 03: Metadata & Labeling Manager

Automatically extracts 40 of 50 hierarchical parameters from MIDI files.
The remaining 10 parameters require manual labeling by music experts.

Architecture:
    - Level 1: Global context extractors (6 auto + 2 manual)
    - Level 2: Universal dimension extractors (16 auto + 4 manual)
    - Level 3: Genre-specific extractors (18 auto + 4 manual)

Integration:
    - Uses existing midi_analyzer.py for low-level analysis
    - Uses pattern_extractor.py for pattern recognition
    - Outputs labels compatible with hierarchical_parameters_50.json

Performance Targets:
    - Extraction speed: < 2 seconds per file
    - Accuracy on deterministic params: > 95%
    - Graceful degradation on edge cases

Author: Agent 03 - Metadata & Labeling Manager
License: MIT
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
import numpy as np

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: mido not installed. Install with: pip install mido")
    raise

# Import existing analysis modules
try:
    from midi_generator.analysis.midi_analyzer import (
        MIDIAnalyzer, NoteEvent, ChordEvent, KeySignature, TimeSignature
    )
    MIDI_ANALYZER_AVAILABLE = True
except ImportError:
    print("WARNING: midi_analyzer not available. Using fallback implementations.")
    MIDI_ANALYZER_AVAILABLE = False

try:
    from midi_generator.learning.pattern_extractor import PatternExtractor
    PATTERN_EXTRACTOR_AVAILABLE = True
except ImportError:
    print("WARNING: pattern_extractor not available.")
    PATTERN_EXTRACTOR_AVAILABLE = False


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class HierarchicalLabels:
    """Complete set of labels for a MIDI file."""
    file_id: str
    file_path: str

    # Level 1: Global Context (8 params, 6 auto + 2 manual)
    level1: Dict[str, Any] = field(default_factory=dict)

    # Level 2: Universal Dimensions (20 params, 16 auto + 4 manual)
    level2: Dict[str, Any] = field(default_factory=dict)

    # Level 3: Genre-Specific (22 params, 18 auto + 4 manual)
    level3: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'file_id': self.file_id,
            'file_path': self.file_path,
            'labels': {
                'level1': self.level1,
                'level2': self.level2,
                'level3': self.level3
            },
            'extraction_metadata': self.extraction_metadata
        }


# ==============================================================================
# AUTO-LABELING ENGINE
# ==============================================================================

class AutoLabeler:
    """
    Automatically extracts 40 of 50 hierarchical parameters from MIDI files.
    """

    def __init__(self, parameter_spec_path: Optional[Path] = None):
        """
        Initialize auto-labeler.

        Args:
            parameter_spec_path: Path to hierarchical_parameters_50.json
        """
        if parameter_spec_path is None:
            parameter_spec_path = Path(__file__).parent / 'hierarchical_parameters_50.json'

        # Load parameter specification
        with open(parameter_spec_path, 'r') as f:
            self.param_spec = json.load(f)

        # Initialize analyzer
        if MIDI_ANALYZER_AVAILABLE:
            self.analyzer = MIDIAnalyzer()
        else:
            self.analyzer = None

        # Precompute parameter lists
        self._identify_auto_labelable_params()

    def _identify_auto_labelable_params(self):
        """Identify which parameters can be auto-labeled."""
        self.auto_params = {
            'level1': [],
            'level2': [],
            'level3': []
        }

        for level in ['level1_global', 'level2_universal', 'level3_genre_specific']:
            level_key = level.split('_')[0]  # 'level1', 'level2', 'level3'
            params = self.param_spec['levels'][level]['parameters']

            for param_name, param_def in params.items():
                if param_def.get('auto_labelable', False):
                    self.auto_params[level_key].append(param_name)

    def extract_all(self, midi_path: Path, file_id: Optional[str] = None) -> HierarchicalLabels:
        """
        Extract all auto-labelable parameters from a MIDI file.

        Args:
            midi_path: Path to MIDI file
            file_id: Unique file identifier (optional, will use filename if None)

        Returns:
            HierarchicalLabels with all auto-extracted parameters
        """
        midi_path = Path(midi_path)
        if file_id is None:
            file_id = midi_path.stem

        # Initialize labels
        labels = HierarchicalLabels(
            file_id=file_id,
            file_path=str(midi_path)
        )

        # Load MIDI file
        try:
            midi = MidiFile(str(midi_path))
        except Exception as e:
            labels.extraction_metadata['error'] = str(e)
            labels.extraction_metadata['success'] = False
            return labels

        # Analyze MIDI file (if analyzer available)
        analysis_result = None
        if self.analyzer:
            try:
                analysis_result = self.analyzer.analyze_file(str(midi_path))
            except Exception as e:
                print(f"Warning: MIDI analysis failed for {midi_path}: {e}")

        # Extract Level 1 (Global Context)
        labels.level1 = self._extract_level1(midi, analysis_result)

        # Extract Level 2 (Universal Dimensions)
        labels.level2 = self._extract_level2(midi, analysis_result, labels.level1)

        # Extract Level 3 (Genre-Specific)
        labels.level3 = self._extract_level3(midi, analysis_result, labels.level1)

        # Add extraction metadata
        labels.extraction_metadata.update({
            'success': True,
            'auto_labeled': True,
            'manually_labeled': False,
            'extraction_version': '2.0',
            'analyzer_used': MIDI_ANALYZER_AVAILABLE
        })

        return labels

    # ==========================================================================
    # LEVEL 1: GLOBAL CONTEXT EXTRACTORS
    # ==========================================================================

    def _extract_level1(self, midi: MidiFile, analysis: Any) -> Dict[str, Any]:
        """Extract Level 1 global context parameters."""
        labels = {}

        # genre.primary - requires genre detection (auto-labelable with classifier)
        labels['genre.primary'] = self._extract_genre(midi, analysis)

        # tempo.bpm
        labels['tempo.bpm'] = self._extract_tempo(midi)

        # time_signature
        labels['time_signature'] = self._extract_time_signature(midi)

        # key.tonic and key.mode
        key_sig = self._extract_key(midi, analysis)
        labels['key.tonic'] = key_sig['tonic']
        labels['key.mode'] = key_sig['mode']

        # energy.level - MANUAL (not extracted here)
        # complexity.overall - MANUAL (not extracted here)

        # structure.form
        labels['structure.form'] = self._extract_structure_form(midi, analysis)

        return labels

    def _extract_genre(self, midi: MidiFile, analysis: Any) -> str:
        """
        Extract primary genre using simple heuristics.

        For production, this should use a trained genre classifier.
        """
        # Simple heuristic-based genre detection
        # TODO: Replace with trained ML classifier

        # Count instrument types
        instruments = set()
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'program_change':
                    instruments.add(msg.program)

        # Basic heuristics
        if any(p in range(0, 8) for p in instruments):  # Piano
            if any(p in range(32, 40) for p in instruments):  # Bass
                return 'jazz'  # Piano + bass suggests jazz

        if any(p in range(24, 32) for p in instruments):  # Guitar
            if any(p >= 112 for p in instruments):  # Drums/percussion
                return 'rock'

        if any(p in range(80, 96) for p in instruments):  # Synth lead
            return 'electronic'

        if any(p in range(40, 56) for p in instruments):  # Strings
            return 'classical'

        # Default
        return 'pop'

    def _extract_tempo(self, midi: MidiFile) -> float:
        """Extract tempo in BPM from MIDI file."""
        # Find first tempo message
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    # Convert microseconds per beat to BPM
                    return mido.tempo2bpm(msg.tempo)

        # Default tempo if none found
        return 120.0

    def _extract_time_signature(self, midi: MidiFile) -> str:
        """Extract time signature from MIDI file."""
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'time_signature':
                    return f"{msg.numerator}/{msg.denominator}"

        # Default
        return "4/4"

    def _extract_key(self, midi: MidiFile, analysis: Any) -> Dict[str, str]:
        """Extract key signature (tonic and mode)."""
        if analysis and hasattr(analysis, 'key_signature'):
            key_sig = analysis.key_signature
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            return {
                'tonic': note_names[key_sig.tonic],
                'mode': key_sig.mode
            }

        # Check for key signature meta message
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'key_signature':
                    return {
                        'tonic': msg.key,
                        'mode': 'major' if 'major' in msg.key.lower() else 'minor'
                    }

        # Default
        return {'tonic': 'C', 'mode': 'major'}

    def _extract_structure_form(self, midi: MidiFile, analysis: Any) -> str:
        """
        Extract structural form (simplified).

        For production, implement proper structure detection algorithm.
        """
        # Simplified: estimate based on track count and duration
        total_ticks = max(track[-1].time if track else 0 for track in midi.tracks)
        duration_seconds = mido.tick2second(total_ticks, midi.ticks_per_beat, 500000)  # Assume 120 BPM

        if duration_seconds < 120:
            return 'AABA'
        elif duration_seconds < 240:
            return 'verse_chorus'
        else:
            return 'through_composed'

    # ==========================================================================
    # LEVEL 2: UNIVERSAL DIMENSION EXTRACTORS
    # ==========================================================================

    def _extract_level2(self, midi: MidiFile, analysis: Any, level1: Dict) -> Dict[str, Any]:
        """Extract Level 2 universal dimension parameters."""
        labels = {}

        # Extract all notes first (needed for multiple params)
        notes = self._extract_all_notes(midi)

        # Harmony parameters
        labels['harmony.chord_density'] = self._extract_chord_density(midi, analysis)
        labels['harmony.complexity'] = self._extract_harmony_complexity(midi, analysis)
        labels['harmony.chromaticism'] = self._extract_chromaticism(notes)
        # harmony.tension - MANUAL
        labels['harmony.voicing_spread'] = self._extract_voicing_spread(notes)
        # harmony.progression_predictability - MANUAL

        # Melody parameters
        melody_track_notes = self._extract_melody_notes(notes)
        labels['melody.note_density'] = self._extract_note_density(melody_track_notes, midi)
        labels['melody.range_semitones'] = self._extract_pitch_range(melody_track_notes)
        # melody.contour_smoothness - MANUAL
        labels['melody.rhythmic_complexity'] = self._extract_rhythmic_complexity(melody_track_notes)
        labels['melody.repetition'] = self._extract_motif_repetition(melody_track_notes)

        # Rhythm parameters
        labels['rhythm.subdivision'] = self._extract_subdivision(notes, midi)
        labels['rhythm.syncopation'] = self._extract_syncopation(notes, midi)
        labels['rhythm.groove_consistency'] = self._extract_groove_consistency(notes)
        labels['rhythm.polyrhythm'] = self._extract_polyrhythm(midi)
        labels['rhythm.swing_amount'] = self._extract_swing_amount(notes, midi)

        # Dynamics parameters
        labels['dynamics.overall_level'] = self._extract_overall_dynamics(notes)
        labels['dynamics.range'] = self._extract_dynamics_range(notes)

        # Texture parameters
        labels['texture.polyphony'] = self._extract_polyphony(notes)
        labels['texture.density'] = self._extract_texture_density(notes, midi)

        return labels

    def _extract_all_notes(self, midi: MidiFile) -> List[Dict]:
        """Extract all notes from MIDI file with timing information."""
        notes = []

        for track_idx, track in enumerate(midi.tracks):
            current_time = 0
            active_notes = {}  # pitch -> start_time

            for msg in track:
                current_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = {
                        'start_tick': current_time,
                        'velocity': msg.velocity,
                        'channel': msg.channel
                    }

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        note_info = active_notes.pop(msg.note)
                        notes.append({
                            'pitch': msg.note,
                            'start_tick': note_info['start_tick'],
                            'duration_ticks': current_time - note_info['start_tick'],
                            'velocity': note_info['velocity'],
                            'channel': note_info['channel'],
                            'track_idx': track_idx
                        })

        # Sort by start time
        notes.sort(key=lambda n: n['start_tick'])
        return notes

    def _extract_melody_notes(self, all_notes: List[Dict]) -> List[Dict]:
        """
        Extract melody notes (heuristic: highest pitch track or most active track).
        """
        if not all_notes:
            return []

        # Group notes by track
        track_notes = defaultdict(list)
        for note in all_notes:
            track_notes[note['track_idx']].append(note)

        # Find track with highest average pitch (simple heuristic for melody)
        melody_track = max(track_notes.keys(),
                          key=lambda t: np.mean([n['pitch'] for n in track_notes[t]]))

        return track_notes[melody_track]

    def _extract_chord_density(self, midi: MidiFile, analysis: Any) -> float:
        """Extract average chords per measure."""
        if analysis and hasattr(analysis, 'chords'):
            # Use analyzer's chord detection
            chords = analysis.chords
            time_sig = analysis.time_signature if hasattr(analysis, 'time_signature') else None

            if time_sig and chords:
                beats_per_measure = time_sig.numerator
                total_measures = analysis.duration / (60 / analysis.tempo * beats_per_measure)
                return len(chords) / max(total_measures, 1)

        # Fallback: estimate from simultaneous notes
        notes = self._extract_all_notes(midi)
        if not notes:
            return 0.0

        # Count simultaneous note groups (rough chord approximation)
        chord_count = 0
        i = 0
        while i < len(notes):
            simultaneous = 1
            while i + simultaneous < len(notes) and \
                  notes[i + simultaneous]['start_tick'] == notes[i]['start_tick']:
                simultaneous += 1
            if simultaneous >= 2:  # At least 2 notes = chord
                chord_count += 1
            i += simultaneous

        # Estimate measures (assuming 120 BPM, 4/4)
        duration_ticks = notes[-1]['start_tick'] + notes[-1]['duration_ticks']
        estimated_measures = duration_ticks / (midi.ticks_per_beat * 4)

        return chord_count / max(estimated_measures, 1)

    def _extract_harmony_complexity(self, midi: MidiFile, analysis: Any) -> float:
        """
        Extract harmonic complexity based on chord extensions.

        Returns value in [0, 1].
        """
        # TODO: Implement proper chord analysis
        # For now, estimate from polyphony and chromaticism
        notes = self._extract_all_notes(midi)
        if not notes:
            return 0.0

        # Simple heuristic: higher polyphony + chromaticism = more complex
        polyphony = self._extract_polyphony(notes)
        chromaticism = self._extract_chromaticism(notes)

        # Normalize
        complexity = min((polyphony / 8.0) * 0.5 + chromaticism * 0.5, 1.0)
        return complexity

    def _extract_chromaticism(self, notes: List[Dict]) -> float:
        """
        Extract degree of chromaticism (chromatic notes outside key).

        Returns value in [0, 1].
        """
        if not notes:
            return 0.0

        # Count all pitch classes
        pitch_classes = [n['pitch'] % 12 for n in notes]
        pc_counts = Counter(pitch_classes)

        # A diatonic scale uses 7 of 12 pitch classes
        # Chromatic music uses more pitch classes
        unique_pcs = len(pc_counts)

        # Normalize: 7 = diatonic (0.0), 12 = fully chromatic (1.0)
        chromaticism = (unique_pcs - 7) / 5.0
        return max(0.0, min(chromaticism, 1.0))

    def _extract_voicing_spread(self, notes: List[Dict]) -> float:
        """Extract average pitch range of simultaneous notes (chord voicings)."""
        if not notes:
            return 0.0

        # Find simultaneous note groups
        spreads = []
        i = 0
        while i < len(notes):
            simultaneous_pitches = [notes[i]['pitch']]
            j = i + 1
            while j < len(notes) and notes[j]['start_tick'] == notes[i]['start_tick']:
                simultaneous_pitches.append(notes[j]['pitch'])
                j += 1

            if len(simultaneous_pitches) >= 2:
                spread = max(simultaneous_pitches) - min(simultaneous_pitches)
                spreads.append(spread)

            i = j if j > i else i + 1

        return np.mean(spreads) if spreads else 12.0  # Default to one octave

    def _extract_note_density(self, notes: List[Dict], midi: MidiFile) -> float:
        """Extract notes per beat."""
        if not notes:
            return 0.0

        duration_ticks = notes[-1]['start_tick'] + notes[-1]['duration_ticks'] - notes[0]['start_tick']
        duration_beats = duration_ticks / midi.ticks_per_beat

        return len(notes) / max(duration_beats, 1)

    def _extract_pitch_range(self, notes: List[Dict]) -> int:
        """Extract pitch range in semitones."""
        if not notes:
            return 0

        pitches = [n['pitch'] for n in notes]
        return max(pitches) - min(pitches)

    def _extract_rhythmic_complexity(self, notes: List[Dict]) -> float:
        """
        Extract rhythmic complexity using IOI (inter-onset interval) entropy.

        Returns value in [0, 1].
        """
        if len(notes) < 2:
            return 0.0

        # Calculate inter-onset intervals
        iois = []
        for i in range(len(notes) - 1):
            ioi = notes[i + 1]['start_tick'] - notes[i]['start_tick']
            if ioi > 0:
                iois.append(ioi)

        if not iois:
            return 0.0

        # Calculate entropy of IOI distribution
        ioi_counts = Counter(iois)
        total = len(iois)
        entropy = -sum((count / total) * np.log2(count / total) for count in ioi_counts.values())

        # Normalize (max entropy for 16 different IOIs ≈ 4 bits)
        normalized_entropy = min(entropy / 4.0, 1.0)

        return normalized_entropy

    def _extract_motif_repetition(self, notes: List[Dict]) -> float:
        """
        Extract degree of motif repetition.

        Returns value in [0, 1] where 1 = highly repetitive.
        """
        if len(notes) < 8:
            return 0.0

        # Simple pattern matching: look for repeated pitch sequences
        motif_length = 4
        motifs = []

        for i in range(len(notes) - motif_length + 1):
            motif = tuple(notes[i + j]['pitch'] for j in range(motif_length))
            motifs.append(motif)

        if not motifs:
            return 0.0

        # Calculate repetition ratio
        motif_counts = Counter(motifs)
        total_motifs = len(motifs)
        unique_motifs = len(motif_counts)

        # High repetition = few unique motifs
        repetition = 1.0 - (unique_motifs / total_motifs)

        return max(0.0, min(repetition, 1.0))

    def _extract_subdivision(self, notes: List[Dict], midi: MidiFile) -> str:
        """Extract smallest rhythmic subdivision."""
        if not notes:
            return "quarter"

        # Find smallest duration
        min_duration = min(n['duration_ticks'] for n in notes)

        # Convert to subdivision
        ticks_per_beat = midi.ticks_per_beat

        if min_duration >= ticks_per_beat:
            return "quarter"
        elif min_duration >= ticks_per_beat // 2:
            return "eighth"
        elif min_duration >= ticks_per_beat // 4:
            return "sixteenth"
        elif min_duration >= ticks_per_beat * 2 // 3:  # Triplet
            return "triplet"
        else:
            return "sextuplet"

    def _extract_syncopation(self, notes: List[Dict], midi: MidiFile) -> float:
        """
        Extract syncopation ratio (notes on weak beats / total notes).

        Returns value in [0, 1].
        """
        if not notes:
            return 0.0

        ticks_per_beat = midi.ticks_per_beat

        # Count notes on weak beats (off-beat)
        syncopated = 0
        for note in notes:
            beat_position = (note['start_tick'] % ticks_per_beat) / ticks_per_beat
            # Syncopated if not on strong beat (0.0, 0.5)
            if not (abs(beat_position) < 0.1 or abs(beat_position - 0.5) < 0.1):
                syncopated += 1

        return syncopated / len(notes)

    def _extract_groove_consistency(self, notes: List[Dict]) -> float:
        """
        Extract groove consistency (timing consistency).

        Returns value in [0, 1] where 1 = very consistent.
        """
        if len(notes) < 10:
            return 0.5  # Not enough data

        # Calculate IOI standard deviation
        iois = []
        for i in range(len(notes) - 1):
            ioi = notes[i + 1]['start_tick'] - notes[i]['start_tick']
            if ioi > 0:
                iois.append(ioi)

        if not iois:
            return 0.5

        # Low std dev = high consistency
        std_dev = np.std(iois)
        mean_ioi = np.mean(iois)

        # Coefficient of variation
        cv = std_dev / mean_ioi if mean_ioi > 0 else 1.0

        # Normalize: cv < 0.1 = very consistent, cv > 1.0 = very inconsistent
        consistency = 1.0 - min(cv / 1.0, 1.0)

        return consistency

    def _extract_polyrhythm(self, midi: MidiFile) -> float:
        """
        Detect polyrhythmic elements (different rhythms in different tracks).

        Returns value in [0, 1].
        """
        # TODO: Implement proper polyrhythm detection
        # For now, simple heuristic based on track rhythm diversity

        track_rhythms = []
        for track in midi.tracks:
            iois = []
            current_time = 0
            last_note_time = None

            for msg in track:
                current_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    if last_note_time is not None:
                        iois.append(current_time - last_note_time)
                    last_note_time = current_time

            if iois:
                track_rhythms.append(np.mean(iois))

        if len(track_rhythms) < 2:
            return 0.0

        # High std dev of track rhythms = polyrhythm
        rhythm_diversity = np.std(track_rhythms) / np.mean(track_rhythms) if np.mean(track_rhythms) > 0 else 0

        return min(rhythm_diversity, 1.0)

    def _extract_swing_amount(self, notes: List[Dict], midi: MidiFile) -> float:
        """
        Extract swing amount.

        Returns value in [0, 1] where 0 = straight, 1 = heavy swing.
        """
        if len(notes) < 10:
            return 0.0

        # Analyze timing of consecutive eighth notes
        ticks_per_beat = midi.ticks_per_beat
        eighth_note = ticks_per_beat // 2

        # Look for pairs of notes that should be eighth notes
        swing_ratios = []

        for i in range(len(notes) - 1):
            ioi = notes[i + 1]['start_tick'] - notes[i]['start_tick']
            # Check if this is approximately an eighth note pair
            if 0.8 * eighth_note < ioi < 1.2 * eighth_note:
                # Check the next note
                if i + 2 < len(notes):
                    next_ioi = notes[i + 2]['start_tick'] - notes[i + 1]['start_tick']
                    # Swing ratio: first note longer than second
                    if next_ioi > 0:
                        ratio = ioi / (ioi + next_ioi)
                        swing_ratios.append(ratio)

        if not swing_ratios:
            return 0.0

        # Average swing ratio
        avg_swing = np.mean(swing_ratios)

        # 0.5 = straight, 0.67 = triplet swing, normalize
        if avg_swing < 0.5:
            return 0.0

        swing_amount = (avg_swing - 0.5) / (0.67 - 0.5)
        return min(swing_amount, 1.0)

    def _extract_overall_dynamics(self, notes: List[Dict]) -> float:
        """Extract average velocity."""
        if not notes:
            return 64.0  # MIDI default

        velocities = [n['velocity'] for n in notes]
        return np.mean(velocities)

    def _extract_dynamics_range(self, notes: List[Dict]) -> float:
        """Extract velocity range (standard deviation)."""
        if not notes:
            return 0.0

        velocities = [n['velocity'] for n in notes]
        return np.std(velocities)

    def _extract_polyphony(self, notes: List[Dict]) -> int:
        """Extract maximum number of simultaneous notes."""
        if not notes:
            return 1

        # Create timeline of note on/off events
        events = []
        for note in notes:
            events.append((note['start_tick'], 'on'))
            events.append((note['start_tick'] + note['duration_ticks'], 'off'))

        # Sort by time
        events.sort()

        # Track max simultaneous
        current_polyphony = 0
        max_polyphony = 0

        for time, event_type in events:
            if event_type == 'on':
                current_polyphony += 1
                max_polyphony = max(max_polyphony, current_polyphony)
            else:
                current_polyphony -= 1

        return max_polyphony

    def _extract_texture_density(self, notes: List[Dict], midi: MidiFile) -> float:
        """Extract overall note density (notes per second)."""
        if not notes:
            return 0.0

        duration_ticks = notes[-1]['start_tick'] + notes[-1]['duration_ticks'] - notes[0]['start_tick']
        duration_seconds = mido.tick2second(duration_ticks, midi.ticks_per_beat, 500000)  # Assume 120 BPM

        return len(notes) / max(duration_seconds, 1)

    # ==========================================================================
    # LEVEL 3: GENRE-SPECIFIC EXTRACTORS
    # ==========================================================================

    def _extract_level3(self, midi: MidiFile, analysis: Any, level1: Dict) -> Dict[str, Any]:
        """Extract Level 3 genre-specific parameters."""
        labels = {}

        # Universal parameters (apply to all genres)
        notes = self._extract_all_notes(midi)

        labels['orchestration.instrument_count'] = self._extract_instrument_count(midi)
        labels['orchestration.register_balance'] = self._extract_register_balance(notes)
        labels['articulation.legato_ratio'] = self._extract_legato_ratio(notes)
        labels['structure.section_contrast'] = self._extract_section_contrast(midi)
        labels['structure.repetition_level'] = self._extract_structure_repetition(midi)

        # Genre-specific parameters (only extract for relevant genres)
        genre = level1.get('genre.primary', 'unknown')

        if genre == 'jazz':
            labels.update(self._extract_jazz_params(notes, midi))
        elif genre == 'classical':
            labels.update(self._extract_classical_params(notes, midi))
        elif genre in ['rock', 'metal']:
            labels.update(self._extract_rock_params(notes, midi))
        elif genre == 'electronic':
            labels.update(self._extract_electronic_params(notes, midi))
        elif genre == 'hip_hop':
            labels.update(self._extract_hiphop_params(notes, midi))
        elif genre == 'latin':
            labels.update(self._extract_latin_params(notes, midi))

        return labels

    def _extract_instrument_count(self, midi: MidiFile) -> int:
        """Count distinct instruments."""
        programs = set()
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'program_change':
                    programs.add(msg.program)

        return len(programs) if programs else 1

    def _extract_register_balance(self, notes: List[Dict]) -> float:
        """
        Extract register balance.

        Returns value in [0, 1] where 0 = bass-heavy, 1 = treble-heavy, 0.5 = balanced.
        """
        if not notes:
            return 0.5

        # MIDI note 60 = middle C
        bass_notes = sum(1 for n in notes if n['pitch'] < 60)
        treble_notes = sum(1 for n in notes if n['pitch'] >= 60)

        total = bass_notes + treble_notes
        if total == 0:
            return 0.5

        return treble_notes / total

    def _extract_legato_ratio(self, notes: List[Dict]) -> float:
        """
        Extract legato ratio (note duration / IOI).

        Returns value in [0, 1] where 1 = full legato.
        """
        if len(notes) < 2:
            return 0.5

        legato_scores = []
        for i in range(len(notes) - 1):
            duration = notes[i]['duration_ticks']
            ioi = notes[i + 1]['start_tick'] - notes[i]['start_tick']

            if ioi > 0:
                legato = min(duration / ioi, 1.0)
                legato_scores.append(legato)

        return np.mean(legato_scores) if legato_scores else 0.5

    def _extract_section_contrast(self, midi: MidiFile) -> float:
        """
        Extract section contrast (simplified).

        Returns value in [0, 1].
        """
        # TODO: Implement proper section segmentation and contrast analysis
        return 0.5  # Placeholder

    def _extract_structure_repetition(self, midi: MidiFile) -> float:
        """
        Extract structural repetition level.

        Returns value in [0, 1].
        """
        # TODO: Implement proper repetition analysis at structural level
        return 0.5  # Placeholder

    def _extract_jazz_params(self, notes: List[Dict], midi: MidiFile) -> Dict[str, Any]:
        """Extract jazz-specific parameters."""
        params = {}

        # jazz.swing_feel
        swing_amount = self._extract_swing_amount(notes, midi)
        if swing_amount < 0.2:
            params['jazz.swing_feel'] = 'straight'
        elif swing_amount < 0.4:
            params['jazz.swing_feel'] = 'light'
        elif swing_amount < 0.7:
            params['jazz.swing_feel'] = 'medium'
        else:
            params['jazz.swing_feel'] = 'hard'

        # jazz.walking_bass
        params['jazz.walking_bass'] = self._detect_walking_bass(notes, midi)

        # jazz.improvisation_ratio
        params['jazz.improvisation_ratio'] = self._estimate_improvisation_ratio(notes)

        # jazz.bebop_vocabulary - MANUAL (not extracted here)

        return params

    def _detect_walking_bass(self, notes: List[Dict], midi: MidiFile) -> float:
        """
        Detect walking bass pattern.

        Returns value in [0, 1].
        """
        # Filter bass notes (typically < MIDI 55)
        bass_notes = [n for n in notes if n['pitch'] < 55]

        if len(bass_notes) < 8:
            return 0.0

        # Walking bass characteristics:
        # - Mostly quarter notes
        # - Stepwise or small leaps
        # - Consistent rhythm

        ticks_per_beat = midi.ticks_per_beat

        # Check rhythm consistency (mostly quarter notes)
        quarter_note_count = 0
        for i in range(len(bass_notes) - 1):
            ioi = bass_notes[i + 1]['start_tick'] - bass_notes[i]['start_tick']
            if 0.8 * ticks_per_beat < ioi < 1.2 * ticks_per_beat:
                quarter_note_count += 1

        rhythm_score = quarter_note_count / max(len(bass_notes) - 1, 1)

        # Check melodic motion (stepwise)
        stepwise_count = 0
        for i in range(len(bass_notes) - 1):
            interval = abs(bass_notes[i + 1]['pitch'] - bass_notes[i]['pitch'])
            if interval <= 2:  # Step or repeat
                stepwise_count += 1

        melodic_score = stepwise_count / max(len(bass_notes) - 1, 1)

        # Combine scores
        walking_bass_score = (rhythm_score + melodic_score) / 2.0

        return walking_bass_score

    def _estimate_improvisation_ratio(self, notes: List[Dict]) -> float:
        """
        Estimate improvisation ratio based on pattern unpredictability.

        Returns value in [0, 1].
        """
        # High improvisation = high unpredictability = low repetition
        repetition = self._extract_motif_repetition(notes)
        return 1.0 - repetition

    def _extract_classical_params(self, notes: List[Dict], midi: MidiFile) -> Dict[str, Any]:
        """Extract classical-specific parameters."""
        params = {}

        # classical.counterpoint - MANUAL (not extracted here)

        # classical.development_density
        params['classical.development_density'] = self._estimate_development_density(notes)

        # classical.voice_leading_quality
        params['classical.voice_leading_quality'] = self._analyze_voice_leading(notes)

        return params

    def _estimate_development_density(self, notes: List[Dict]) -> float:
        """
        Estimate thematic development density.

        Returns value in [0, 1].
        """
        # High development = high variation, low exact repetition
        # This is inversely related to repetition
        repetition = self._extract_motif_repetition(notes)

        # But also consider complexity
        complexity = self._extract_rhythmic_complexity(notes)

        development = (1.0 - repetition) * 0.6 + complexity * 0.4

        return development

    def _analyze_voice_leading(self, notes: List[Dict]) -> float:
        """
        Analyze voice leading quality.

        Returns value in [0, 1] where 1 = excellent voice leading.
        """
        # Good voice leading: smooth motion, contrary motion, no parallels

        # Group simultaneous notes (chords)
        chords = []
        i = 0
        while i < len(notes):
            chord_pitches = [notes[i]['pitch']]
            j = i + 1
            while j < len(notes) and notes[j]['start_tick'] == notes[i]['start_tick']:
                chord_pitches.append(notes[j]['pitch'])
                j += 1

            if len(chord_pitches) >= 2:
                chords.append(sorted(chord_pitches))

            i = j if j > i else i + 1

        if len(chords) < 2:
            return 0.5  # Not enough data

        # Analyze voice leading between consecutive chords
        smooth_motions = 0
        total_transitions = 0

        for i in range(len(chords) - 1):
            chord1 = chords[i]
            chord2 = chords[i + 1]

            # Simple heuristic: count small intervals (good voice leading)
            if len(chord1) == len(chord2):
                for v1, v2 in zip(chord1, chord2):
                    interval = abs(v2 - v1)
                    if interval <= 2:  # Step or repeat
                        smooth_motions += 1
                    total_transitions += 1

        if total_transitions == 0:
            return 0.5

        voice_leading_quality = smooth_motions / total_transitions

        return voice_leading_quality

    def _extract_rock_params(self, notes: List[Dict], midi: MidiFile) -> Dict[str, Any]:
        """Extract rock/metal-specific parameters."""
        params = {}

        # rock.power_chord_ratio
        params['rock.power_chord_ratio'] = self._detect_power_chords(notes)

        # rock.riff_repetition - MANUAL (not extracted here)

        # rock.distortion_level
        params['rock.distortion_level'] = self._estimate_distortion_level(notes)

        return params

    def _detect_power_chords(self, notes: List[Dict]) -> float:
        """
        Detect power chord ratio.

        Returns value in [0, 1].
        """
        # Power chord = root + perfect 5th (7 semitones), no 3rd

        # Group simultaneous notes
        chords = []
        i = 0
        while i < len(notes):
            chord_pitches = [notes[i]['pitch']]
            j = i + 1
            while j < len(notes) and notes[j]['start_tick'] == notes[i]['start_tick']:
                chord_pitches.append(notes[j]['pitch'])
                j += 1

            if len(chord_pitches) >= 2:
                chords.append(sorted(chord_pitches))

            i = j if j > i else i + 1

        if not chords:
            return 0.0

        # Check for power chords
        power_chord_count = 0
        for chord in chords:
            # Convert to pitch classes
            pcs = [(p % 12) for p in chord]
            unique_pcs = sorted(set(pcs))

            # Power chord: 2 unique pitch classes, 7 semitones apart
            if len(unique_pcs) == 2:
                interval = (unique_pcs[1] - unique_pcs[0]) % 12
                if interval == 7:  # Perfect 5th
                    power_chord_count += 1

        return power_chord_count / len(chords)

    def _estimate_distortion_level(self, notes: List[Dict]) -> float:
        """
        Estimate distortion/intensity level.

        Returns value in [0, 1].
        """
        # Heuristic: high velocity + staccato articulation = distortion
        if not notes:
            return 0.0

        avg_velocity = np.mean([n['velocity'] for n in notes])
        velocity_score = avg_velocity / 127.0

        # Check articulation (short notes = more aggressive)
        avg_duration = np.mean([n['duration_ticks'] for n in notes])
        # Shorter = more distorted (inverse relationship)
        duration_score = 1.0 - min(avg_duration / 1000.0, 1.0)

        distortion = velocity_score * 0.7 + duration_score * 0.3

        return distortion

    def _extract_electronic_params(self, notes: List[Dict], midi: MidiFile) -> Dict[str, Any]:
        """Extract electronic-specific parameters."""
        params = {}

        # electronic.quantization
        params['electronic.quantization'] = self._measure_quantization(notes, midi)

        # electronic.filter_movement - MANUAL (not extracted here)

        # electronic.arpeggio_density
        params['electronic.arpeggio_density'] = self._detect_arpeggios(notes)

        return params

    def _measure_quantization(self, notes: List[Dict], midi: MidiFile) -> float:
        """
        Measure rhythmic quantization (grid alignment).

        Returns value in [0, 1] where 1 = perfectly quantized.
        """
        if not notes:
            return 0.5

        ticks_per_beat = midi.ticks_per_beat
        grid_size = ticks_per_beat // 4  # 16th note grid

        # Check how many notes align to grid
        on_grid = 0
        for note in notes:
            distance_to_grid = note['start_tick'] % grid_size
            if distance_to_grid < grid_size * 0.1 or distance_to_grid > grid_size * 0.9:
                on_grid += 1

        quantization = on_grid / len(notes)

        return quantization

    def _detect_arpeggios(self, notes: List[Dict]) -> float:
        """
        Detect arpeggio patterns.

        Returns value in [0, 1].
        """
        if len(notes) < 6:
            return 0.0

        # Arpeggio: sequential notes outlining a chord
        # Look for sequences of 3-4 notes with consistent rhythm

        arpeggio_count = 0
        total_sequences = 0

        for i in range(len(notes) - 3):
            # Check for consistent rhythm
            ioi1 = notes[i + 1]['start_tick'] - notes[i]['start_tick']
            ioi2 = notes[i + 2]['start_tick'] - notes[i + 1]['start_tick']
            ioi3 = notes[i + 3]['start_tick'] - notes[i + 2]['start_tick']

            # Consistent rhythm?
            if abs(ioi1 - ioi2) < ioi1 * 0.2 and abs(ioi2 - ioi3) < ioi2 * 0.2:
                # Check if pitches outline a chord (3rds or 4ths)
                pitches = [notes[i + j]['pitch'] for j in range(4)]
                intervals = [pitches[j + 1] - pitches[j] for j in range(3)]

                # Arpeggios typically have intervals of 3-5 semitones
                if all(3 <= abs(iv) <= 5 for iv in intervals):
                    arpeggio_count += 1

                total_sequences += 1

        if total_sequences == 0:
            return 0.0

        return arpeggio_count / total_sequences

    def _extract_hiphop_params(self, notes: List[Dict], midi: MidiFile) -> Dict[str, Any]:
        """Extract hip-hop-specific parameters."""
        params = {}

        # hiphop.sample_based
        params['hiphop.sample_based'] = self._detect_sample_loops(notes)

        # hiphop.boom_bap_feel
        params['hiphop.boom_bap_feel'] = self._detect_boom_bap(notes, midi)

        return params

    def _detect_sample_loops(self, notes: List[Dict]) -> float:
        """
        Detect sample-based composition (repetitive loops).

        Returns value in [0, 1].
        """
        # Sample-based = high exact repetition
        return self._extract_motif_repetition(notes)

    def _detect_boom_bap(self, notes: List[Dict], midi: MidiFile) -> float:
        """
        Detect boom-bap drum pattern characteristics.

        Returns value in [0, 1].
        """
        # Boom-bap characteristics:
        # - Kick on 1 and 3
        # - Snare on 2 and 4
        # - Medium tempo (85-95 BPM)

        # Check tempo
        tempo = self._extract_tempo(midi)
        if 85 <= tempo <= 95:
            tempo_score = 1.0
        elif 80 <= tempo <= 100:
            tempo_score = 0.7
        else:
            tempo_score = 0.3

        # TODO: Analyze drum pattern in detail
        # For now, use tempo as primary indicator

        return tempo_score * 0.7  # Conservative estimate

    def _extract_latin_params(self, notes: List[Dict], midi: MidiFile) -> Dict[str, Any]:
        """Extract Latin-specific parameters."""
        params = {}

        # latin.clave_pattern
        params['latin.clave_pattern'] = self._detect_clave_pattern(notes, midi)

        # latin.montuno_complexity
        params['latin.montuno_complexity'] = self._analyze_montuno(notes)

        return params

    def _detect_clave_pattern(self, notes: List[Dict], midi: MidiFile) -> str:
        """
        Detect clave pattern type.

        Returns categorical value.
        """
        # TODO: Implement proper clave detection
        # This requires analyzing specific rhythmic patterns

        return "none"  # Placeholder

    def _analyze_montuno(self, notes: List[Dict]) -> float:
        """
        Analyze montuno pattern complexity.

        Returns value in [0, 1].
        """
        # TODO: Implement montuno analysis
        # For now, use rhythmic complexity as proxy

        return self._extract_rhythmic_complexity(notes)


# ==============================================================================
# BATCH PROCESSING
# ==============================================================================

def batch_extract_labels(midi_files: List[Path],
                         output_path: Path,
                         parameter_spec_path: Optional[Path] = None) -> List[HierarchicalLabels]:
    """
    Batch extract labels from multiple MIDI files.

    Args:
        midi_files: List of MIDI file paths
        output_path: Path to save labels JSON
        parameter_spec_path: Path to hierarchical_parameters_50.json

    Returns:
        List of HierarchicalLabels
    """
    labeler = AutoLabeler(parameter_spec_path)

    all_labels = []
    for midi_path in midi_files:
        try:
            labels = labeler.extract_all(midi_path)
            all_labels.append(labels)
            print(f"✓ Processed: {midi_path.name}")
        except Exception as e:
            print(f"✗ Failed: {midi_path.name} - {e}")

    # Save to JSON
    output_data = {
        'version': '2.0',
        'total_files': len(all_labels),
        'labels': [labels.to_dict() for labels in all_labels]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved labels for {len(all_labels)} files to {output_path}")

    return all_labels


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python auto_labeler.py <midi_file>")
        sys.exit(1)

    midi_path = Path(sys.argv[1])

    labeler = AutoLabeler()
    labels = labeler.extract_all(midi_path)

    print(json.dumps(labels.to_dict(), indent=2))
