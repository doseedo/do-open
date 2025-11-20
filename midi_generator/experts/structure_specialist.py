#!/usr/bin/env python3
"""
Structure Specialist - Agent 23
================================

Expert module for comprehensive structural analysis of MIDI files, including:
- Form type detection (AABA, sonata, verse-chorus, through-composed, etc.)
- Section boundary detection and segmentation
- Transition analysis and classification
- Climax detection and placement
- Repetition and variation pattern recognition
- Motivic transformation detection

This module is part of the self-expanding inverse music generation system,
providing deep structural feature extraction for parameter prediction.

Research Foundations
--------------------

**Form Analysis:**
- Paulus & Klapuri (2009): "Music Structure Analysis Using a Probabilistic Fitness Measure"
- Peeters (2007): "Sequence Representation of Music Structure Using Higher-Order Similarity Matrix"
- McFee & Ellis (2014): "Analyzing Song Structure with Spectral Clustering"

**Repetition Detection:**
- Müller & Jiang (2012): "A Segment-Based Fitness Measure for Capturing Repetitive Structures"
- Serra et al. (2012): "Unsupervised Detection of Music Boundaries by Time Series Structure Features"

**Climax Detection:**
- Huron (1996): "The Melodic Arch in Western Folksongs"
- Friberg & Battel (2002): "Structural Communication of Musical Expression"

**Motivic Analysis:**
- Lartillot (2004): "Motivic Pattern Recognition in Polyphonic Music"
- Rolland (1999): "Discovering Patterns in Musical Sequences"

Author: Agent 23 - Structure Specialist
Date: 2025
License: MIT
"""

from typing import List, Dict, Tuple, Optional, Set, Any, Union, Callable
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from enum import Enum, auto
from pathlib import Path
import json
import math

try:
    import numpy as np
    from scipy import signal, stats
    from scipy.spatial.distance import euclidean, cosine
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    NUMPY_AVAILABLE = True
    NDArray = np.ndarray
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy/SciPy/scikit-learn not available. Install with: pip install numpy scipy scikit-learn")
    NDArray = Any

try:
    import mido
    from mido import MidiFile
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Warning: mido not available. Install with: pip install mido")


# ==============================================================================
# ENUMERATIONS
# ==============================================================================

class FormType(Enum):
    """Musical form types detected by the structure specialist."""
    AABA = "aaba"                          # 32-bar jazz standard (A-A-B-A)
    ABAB = "abab"                          # Simple verse-chorus
    ABAC = "abac"                          # Variant AABA
    VERSE_CHORUS = "verse_chorus"          # Pop song structure
    VERSE_CHORUS_BRIDGE = "verse_chorus_bridge"  # Extended pop
    TWELVE_BAR_BLUES = "twelve_bar_blues"  # Blues form
    SONATA = "sonata"                      # Sonata allegro form
    RONDO = "rondo"                        # Rondo form (ABACA, ABACABA)
    THEME_VARIATIONS = "theme_variations"  # Theme with variations
    BINARY = "binary"                      # AB form
    TERNARY = "ternary"                    # ABA form
    STROPHIC = "strophic"                  # Repeated A sections
    THROUGH_COMPOSED = "through_composed"  # No repetition, continuous development
    UNKNOWN = "unknown"


class SectionType(Enum):
    """Types of structural sections."""
    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    PRE_CHORUS = "pre_chorus"
    INTERLUDE = "interlude"
    SOLO = "solo"
    OUTRO = "outro"
    CODA = "coda"
    EXPOSITION = "exposition"        # Sonata form
    DEVELOPMENT = "development"      # Sonata form
    RECAPITULATION = "recapitulation"  # Sonata form
    REFRAIN = "refrain"             # Rondo form
    EPISODE = "episode"             # Rondo/Fugue
    VARIATION = "variation"         # Theme and variations
    UNKNOWN = "unknown"


class TransitionType(Enum):
    """Types of transitions between sections."""
    DIRECT = "direct"                # Immediate change
    MODULATION = "modulation"        # Key change
    TURNAROUND = "turnaround"        # Harmonic turnaround
    FILL = "fill"                    # Drum/melodic fill
    BUILDUP = "buildup"              # Crescendo/intensification
    BREAKDOWN = "breakdown"          # Reduction of texture
    RISER = "riser"                  # Upward sweep
    SILENCE = "silence"              # Rest/pause
    COMMON_CHORD = "common_chord"    # Pivot chord modulation
    CHROMATIC = "chromatic"          # Chromatic transition
    SEQUENTIAL = "sequential"        # Sequential progression


class MotivicTransformation(Enum):
    """Types of motivic transformations."""
    EXACT_REPETITION = "exact_repetition"
    TRANSPOSITION = "transposition"
    SEQUENCE = "sequence"
    INVERSION = "inversion"
    RETROGRADE = "retrograde"
    RETROGRADE_INVERSION = "retrograde_inversion"
    AUGMENTATION = "augmentation"
    DIMINUTION = "diminution"
    FRAGMENTATION = "fragmentation"
    EXTENSION = "extension"
    INTERPOLATION = "interpolation"
    RHYTHMIC_SHIFT = "rhythmic_shift"
    INTERVALLIC_EXPANSION = "intervallic_expansion"
    INTERVALLIC_CONTRACTION = "intervallic_contraction"
    VARIATION = "variation"


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class NoteEvent:
    """Represents a MIDI note event."""
    start_time: float      # In seconds or beats
    duration: float
    pitch: int
    velocity: int
    channel: int = 0
    track: int = 0

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    @property
    def pitch_class(self) -> int:
        return self.pitch % 12


@dataclass
class StructuralSection:
    """Represents a detected structural section."""
    start_time: float
    end_time: float
    start_bar: int
    end_bar: int
    section_type: SectionType
    label: str                    # e.g., "A1", "B", "Chorus1"
    notes: List[NoteEvent]

    # Musical characteristics
    key: Optional[int] = None     # Tonic pitch class
    mode: Optional[str] = None    # "major" or "minor"
    tempo: Optional[float] = None
    dynamic_level: float = 0.5    # 0.0-1.0
    density: float = 0.5          # Notes per beat

    # Structural metadata
    similarity_to_previous: float = 0.0
    is_climax: bool = False
    has_modulation: bool = False
    transition_in: Optional['Transition'] = None
    transition_out: Optional['Transition'] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_bars(self) -> int:
        return self.end_bar - self.start_bar


@dataclass
class Transition:
    """Represents a transition between sections."""
    start_time: float
    end_time: float
    from_section: Optional[str] = None
    to_section: Optional[str] = None
    transition_type: TransitionType = TransitionType.DIRECT

    # Characteristics
    has_fill: bool = False
    has_modulation: bool = False
    from_key: Optional[int] = None
    to_key: Optional[int] = None
    intensity_change: float = 0.0  # -1.0 to 1.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class Climax:
    """Represents a detected climax point."""
    time: float
    bar: int
    section: Optional[str] = None

    # Climax characteristics
    pitch_peak: bool = False       # Highest pitch
    dynamic_peak: bool = False     # Loudest point
    density_peak: bool = False     # Most notes
    harmonic_tension: float = 0.0  # Tension level

    # Confidence
    confidence: float = 0.5


@dataclass
class Motif:
    """Represents a musical motif."""
    pitches: List[int]
    intervals: List[int]
    durations: List[float]
    contour: List[int]        # -1, 0, 1 for down, same, up
    start_time: float
    duration: float

    # Metadata
    occurrences: List[float] = field(default_factory=list)  # Start times
    transformations: List[MotivicTransformation] = field(default_factory=list)


@dataclass
class RepetitionGroup:
    """Represents a group of repeated or varied sections."""
    sections: List[StructuralSection]
    similarity_matrix: Optional[NDArray] = None
    average_similarity: float = 0.0
    variation_type: Optional[str] = None  # "exact", "varied", "developed"


@dataclass
class StructuralAnalysis:
    """Complete structural analysis of a MIDI file."""
    form_type: FormType
    sections: List[StructuralSection]
    transitions: List[Transition]
    climaxes: List[Climax]
    motifs: List[Motif]
    repetition_groups: List[RepetitionGroup]

    # Overall characteristics
    total_duration: float = 0.0
    total_bars: int = 0
    primary_key: Optional[int] = None
    primary_mode: Optional[str] = None
    average_tempo: float = 120.0

    # Structural metrics
    repetition_ratio: float = 0.0      # How much is repeated
    development_ratio: float = 0.0     # How much is developed
    contrast_ratio: float = 0.0        # Section diversity
    climax_position: float = 0.618     # Golden ratio by default

    # Confidence scores
    form_confidence: float = 0.5
    segmentation_confidence: float = 0.5


# ==============================================================================
# STRUCTURE SPECIALIST CLASS
# ==============================================================================

class StructureSpecialist:
    """
    Expert system for comprehensive structural analysis of MIDI files.

    This class provides methods for:
    - Form type detection
    - Section boundary detection
    - Transition analysis
    - Climax detection
    - Repetition/variation recognition
    - Motivic transformation detection
    """

    def __init__(self,
                 segment_size_bars: int = 4,
                 similarity_threshold: float = 0.7,
                 min_section_bars: int = 2,
                 max_section_bars: int = 32):
        """
        Initialize the Structure Specialist.

        Args:
            segment_size_bars: Size of segments for initial analysis
            similarity_threshold: Threshold for considering sections similar
            min_section_bars: Minimum section length
            max_section_bars: Maximum section length
        """
        self.segment_size_bars = segment_size_bars
        self.similarity_threshold = similarity_threshold
        self.min_section_bars = min_section_bars
        self.max_section_bars = max_section_bars

    # ==========================================================================
    # MAIN ANALYSIS METHODS
    # ==========================================================================

    def analyze(self, midi_path: str) -> StructuralAnalysis:
        """
        Perform complete structural analysis of a MIDI file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Complete structural analysis
        """
        if not MIDO_AVAILABLE:
            raise RuntimeError("mido library required for MIDI analysis")

        # Load MIDI file
        midi = MidiFile(midi_path)
        notes = self._extract_notes(midi)

        # Extract basic timing information
        total_duration = max([n.end_time for n in notes]) if notes else 0.0
        tempo = self._estimate_tempo(midi)
        beats_per_bar = self._estimate_time_signature(midi)
        total_bars = int(total_duration / (60.0 / tempo * beats_per_bar))

        # Detect sections
        sections = self._detect_sections(notes, tempo, beats_per_bar, total_bars)

        # Detect transitions
        transitions = self._detect_transitions(sections, notes)

        # Detect form type
        form_type, form_confidence = self._detect_form_type(sections)

        # Detect climaxes
        climaxes = self._detect_climaxes(sections, notes)

        # Extract motifs
        motifs = self._extract_motifs(notes, tempo)

        # Analyze repetitions
        repetition_groups = self._analyze_repetitions(sections)

        # Calculate structural metrics
        repetition_ratio = self._calculate_repetition_ratio(sections)
        development_ratio = self._calculate_development_ratio(sections)
        contrast_ratio = self._calculate_contrast_ratio(sections)
        climax_position = self._calculate_climax_position(climaxes, total_duration)

        # Detect primary key
        primary_key, primary_mode = self._detect_primary_key(notes)

        return StructuralAnalysis(
            form_type=form_type,
            sections=sections,
            transitions=transitions,
            climaxes=climaxes,
            motifs=motifs,
            repetition_groups=repetition_groups,
            total_duration=total_duration,
            total_bars=total_bars,
            primary_key=primary_key,
            primary_mode=primary_mode,
            average_tempo=tempo,
            repetition_ratio=repetition_ratio,
            development_ratio=development_ratio,
            contrast_ratio=contrast_ratio,
            climax_position=climax_position,
            form_confidence=form_confidence,
            segmentation_confidence=0.8  # Placeholder
        )

    def extract_structure_parameters(self, analysis: StructuralAnalysis) -> Dict[str, Any]:
        """
        Extract structure-related parameters for XGBoost training.

        Args:
            analysis: Structural analysis result

        Returns:
            Dictionary of parameters
        """
        params = {}

        # Form parameters
        params['structure.form_type'] = analysis.form_type.value
        params['structure.form_confidence'] = analysis.form_confidence
        params['structure.num_sections'] = len(analysis.sections)
        params['structure.total_bars'] = analysis.total_bars

        # Section parameters
        if analysis.sections:
            section_lengths = [s.num_bars for s in analysis.sections]
            params['structure.avg_section_length'] = np.mean(section_lengths)
            params['structure.section_length_variance'] = np.var(section_lengths)
            params['structure.min_section_length'] = min(section_lengths)
            params['structure.max_section_length'] = max(section_lengths)

        # Transition parameters
        params['structure.num_transitions'] = len(analysis.transitions)
        params['structure.has_modulations'] = any(t.has_modulation for t in analysis.transitions)
        params['structure.transition_fill_ratio'] = (
            sum(1 for t in analysis.transitions if t.has_fill) / max(len(analysis.transitions), 1)
        )

        # Climax parameters
        params['structure.num_climaxes'] = len(analysis.climaxes)
        params['structure.climax_position'] = analysis.climax_position
        params['structure.has_golden_ratio_climax'] = abs(analysis.climax_position - 0.618) < 0.1

        # Repetition parameters
        params['structure.repetition_ratio'] = analysis.repetition_ratio
        params['structure.development_ratio'] = analysis.development_ratio
        params['structure.contrast_ratio'] = analysis.contrast_ratio

        # Motif parameters
        params['structure.num_motifs'] = len(analysis.motifs)
        if analysis.motifs:
            motif_occurrences = [len(m.occurrences) for m in analysis.motifs]
            params['structure.avg_motif_occurrences'] = np.mean(motif_occurrences)
            params['structure.max_motif_occurrences'] = max(motif_occurrences)

            # Transformation types
            all_transformations = [t for m in analysis.motifs for t in m.transformations]
            transformation_counts = Counter([t.value for t in all_transformations])
            for trans_type in MotivicTransformation:
                params[f'structure.motif.{trans_type.value}_count'] = transformation_counts.get(trans_type.value, 0)

        # Section type distribution
        section_types = Counter([s.section_type.value for s in analysis.sections])
        for sec_type in SectionType:
            params[f'structure.section.{sec_type.value}_count'] = section_types.get(sec_type.value, 0)

        # Transition type distribution
        transition_types = Counter([t.transition_type.value for t in analysis.transitions])
        for trans_type in TransitionType:
            params[f'structure.transition.{trans_type.value}_count'] = transition_types.get(trans_type.value, 0)

        return params

    # ==========================================================================
    # SECTION DETECTION
    # ==========================================================================

    def _detect_sections(self, notes: List[NoteEvent], tempo: float,
                        beats_per_bar: int, total_bars: int) -> List[StructuralSection]:
        """
        Detect structural sections using novelty detection and similarity analysis.

        Research basis: Paulus & Klapuri (2009), McFee & Ellis (2014)
        """
        if not notes or not NUMPY_AVAILABLE:
            return []

        # Create feature matrix for segments
        segment_duration = (60.0 / tempo) * beats_per_bar * self.segment_size_bars
        num_segments = int(np.ceil(max([n.end_time for n in notes]) / segment_duration))

        if num_segments < 2:
            # Too short, create single section
            return [StructuralSection(
                start_time=0.0,
                end_time=max([n.end_time for n in notes]),
                start_bar=0,
                end_bar=total_bars,
                section_type=SectionType.UNKNOWN,
                label="A",
                notes=notes
            )]

        # Extract features for each segment
        features = []
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            segment_notes = [n for n in notes if start_time <= n.start_time < end_time]

            if segment_notes:
                features.append(self._extract_segment_features(segment_notes))
            else:
                features.append(np.zeros(20))  # Silence

        feature_matrix = np.array(features)

        # Compute self-similarity matrix
        similarity_matrix = self._compute_similarity_matrix(feature_matrix)

        # Detect boundaries using novelty curve
        boundaries = self._detect_boundaries_from_similarity(similarity_matrix, segment_duration)

        # Create sections from boundaries
        sections = []
        for i in range(len(boundaries) - 1):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]
            start_bar = int(start_time / (60.0 / tempo * beats_per_bar))
            end_bar = int(end_time / (60.0 / tempo * beats_per_bar))

            section_notes = [n for n in notes if start_time <= n.start_time < end_time]

            section = StructuralSection(
                start_time=start_time,
                end_time=end_time,
                start_bar=start_bar,
                end_bar=end_bar,
                section_type=SectionType.UNKNOWN,
                label=f"S{i}",
                notes=section_notes,
                dynamic_level=self._calculate_dynamic_level(section_notes),
                density=self._calculate_density(section_notes, end_time - start_time)
            )
            sections.append(section)

        # Label sections based on similarity
        sections = self._label_sections(sections, similarity_matrix)

        # Classify section types
        sections = self._classify_section_types(sections)

        return sections

    def _extract_segment_features(self, notes: List[NoteEvent]) -> NDArray:
        """Extract features from a segment of notes."""
        if not notes or not NUMPY_AVAILABLE:
            return np.zeros(20)

        features = []

        # Pitch features
        pitches = [n.pitch for n in notes]
        features.append(np.mean(pitches) if pitches else 60)
        features.append(np.std(pitches) if len(pitches) > 1 else 0)
        features.append(min(pitches) if pitches else 60)
        features.append(max(pitches) if pitches else 60)

        # Pitch class distribution (12 bins)
        pitch_classes = [n.pitch_class for n in notes]
        pc_dist = np.zeros(12)
        for pc in pitch_classes:
            pc_dist[pc] += 1
        if pc_dist.sum() > 0:
            pc_dist /= pc_dist.sum()
        features.extend(pc_dist.tolist())

        # Velocity features
        velocities = [n.velocity for n in notes]
        features.append(np.mean(velocities) if velocities else 64)
        features.append(np.std(velocities) if len(velocities) > 1 else 0)

        # Density
        features.append(len(notes))

        # Duration features
        durations = [n.duration for n in notes]
        features.append(np.mean(durations) if durations else 0.5)

        return np.array(features[:20])  # Ensure fixed size

    def _compute_similarity_matrix(self, feature_matrix: NDArray) -> NDArray:
        """Compute self-similarity matrix from feature matrix."""
        if not NUMPY_AVAILABLE:
            return np.eye(len(feature_matrix))

        n = len(feature_matrix)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # Cosine similarity
                if np.linalg.norm(feature_matrix[i]) > 0 and np.linalg.norm(feature_matrix[j]) > 0:
                    similarity = 1 - cosine(feature_matrix[i], feature_matrix[j])
                else:
                    similarity = 0
                similarity_matrix[i, j] = max(0, similarity)

        return similarity_matrix

    def _detect_boundaries_from_similarity(self, similarity_matrix: NDArray,
                                          segment_duration: float) -> List[float]:
        """
        Detect section boundaries using novelty curve from similarity matrix.

        Research: Foote (2000) - "Automatic Audio Segmentation Using a Measure of Audio Novelty"
        """
        if not NUMPY_AVAILABLE:
            return [0.0]

        n = len(similarity_matrix)

        # Compute novelty curve using checkerboard kernel
        kernel_size = max(3, n // 10)
        novelty = np.zeros(n)

        for i in range(kernel_size, n - kernel_size):
            # Compare similarity within window to across boundary
            within = similarity_matrix[i-kernel_size:i, i-kernel_size:i].mean()
            across = similarity_matrix[i:i+kernel_size, i-kernel_size:i].mean()
            novelty[i] = within - across

        # Find peaks in novelty curve
        peaks = signal.find_peaks(novelty, distance=2, prominence=0.1)[0]

        # Convert to time boundaries
        boundaries = [0.0]
        for peak in peaks:
            boundaries.append(peak * segment_duration)
        boundaries.append(n * segment_duration)

        return sorted(set(boundaries))

    def _label_sections(self, sections: List[StructuralSection],
                       similarity_matrix: NDArray) -> List[StructuralSection]:
        """Label sections based on similarity (A, A, B, A, etc.)."""
        if not sections or not NUMPY_AVAILABLE:
            return sections

        # Extract features for each section
        section_features = []
        for section in sections:
            if section.notes:
                section_features.append(self._extract_segment_features(section.notes))
            else:
                section_features.append(np.zeros(20))

        section_features = np.array(section_features)

        # Compute section similarity matrix
        n = len(sections)
        sec_similarity = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if np.linalg.norm(section_features[i]) > 0 and np.linalg.norm(section_features[j]) > 0:
                    sec_similarity[i, j] = 1 - cosine(section_features[i], section_features[j])

        # Label sections
        labels = ['A']
        next_label_idx = 1
        label_chars = 'ABCDEFGHIJKLMNOP'

        for i in range(1, n):
            # Find most similar previous section
            max_sim = max(sec_similarity[i, :i])

            if max_sim >= self.similarity_threshold:
                # Similar to previous section
                most_similar = np.argmax(sec_similarity[i, :i])
                labels.append(labels[most_similar])
            else:
                # New section
                if next_label_idx < len(label_chars):
                    labels.append(label_chars[next_label_idx])
                    next_label_idx += 1
                else:
                    labels.append(f"S{next_label_idx}")
                    next_label_idx += 1

        # Update section labels
        for i, section in enumerate(sections):
            section.label = labels[i]
            if i > 0:
                section.similarity_to_previous = sec_similarity[i, i-1]

        return sections

    def _classify_section_types(self, sections: List[StructuralSection]) -> List[StructuralSection]:
        """Classify sections into types (verse, chorus, bridge, etc.)."""
        if not sections:
            return sections

        # First and last sections
        if len(sections) > 0:
            # First section is often intro if short
            if sections[0].num_bars <= 4 and len(sections) > 1:
                sections[0].section_type = SectionType.INTRO

            # Last section is often outro if short
            if len(sections) > 1 and sections[-1].num_bars <= 4:
                sections[-1].section_type = SectionType.OUTRO

        # Middle sections based on labels and characteristics
        for i, section in enumerate(sections):
            if section.section_type != SectionType.UNKNOWN:
                continue  # Already classified

            # Bridge detection: unique section (appears once, different from others)
            label_count = sum(1 for s in sections if s.label == section.label)
            if label_count == 1 and i > 0 and i < len(sections) - 1:
                section.section_type = SectionType.BRIDGE

            # Chorus detection: repeated section with high energy
            elif label_count >= 2 and section.dynamic_level > 0.6:
                section.section_type = SectionType.CHORUS

            # Verse detection: repeated section with moderate energy
            elif label_count >= 2 and section.dynamic_level <= 0.6:
                section.section_type = SectionType.VERSE

            # Default
            else:
                section.section_type = SectionType.UNKNOWN

        return sections

    # ==========================================================================
    # FORM TYPE DETECTION
    # ==========================================================================

    def _detect_form_type(self, sections: List[StructuralSection]) -> Tuple[FormType, float]:
        """
        Detect the overall form type based on section structure.

        Returns:
            (form_type, confidence)
        """
        if not sections:
            return FormType.UNKNOWN, 0.0

        # Get section labels
        labels = [s.label for s in sections]
        label_string = ''.join(labels)

        # Count sections
        num_sections = len(sections)
        unique_labels = len(set(labels))

        # AABA detection (32-bar jazz standard)
        if label_string in ['AABA', 'AAB']:
            if num_sections == 4 or num_sections == 3:
                return FormType.AABA, 0.9

        # ABAC detection
        if len(labels) == 4 and labels[0] == labels[1] and labels[2] != labels[3]:
            if labels[0] != labels[2] and labels[0] != labels[3]:
                return FormType.ABAC, 0.85

        # ABAB detection (simple verse-chorus)
        if label_string in ['ABAB', 'ABABAB']:
            return FormType.ABAB, 0.85

        # ABA (ternary)
        if label_string == 'ABA':
            return FormType.TERNARY, 0.9

        # AB (binary)
        if num_sections == 2:
            return FormType.BINARY, 0.8

        # Verse-chorus-bridge (pop)
        section_types = [s.section_type for s in sections]
        has_verse = SectionType.VERSE in section_types
        has_chorus = SectionType.CHORUS in section_types
        has_bridge = SectionType.BRIDGE in section_types

        if has_verse and has_chorus and has_bridge:
            return FormType.VERSE_CHORUS_BRIDGE, 0.85
        elif has_verse and has_chorus:
            return FormType.VERSE_CHORUS, 0.8

        # Strophic (all same)
        if unique_labels == 1 or (unique_labels == 2 and sections[0].section_type == SectionType.INTRO):
            return FormType.STROPHIC, 0.75

        # Rondo (ABACA or ABACABA pattern)
        if 'A' in labels and labels.count('A') >= 3:
            # Check if A alternates with other sections
            a_positions = [i for i, l in enumerate(labels) if l == 'A']
            if all(a_positions[i+1] - a_positions[i] == 2 for i in range(len(a_positions)-1)):
                return FormType.RONDO, 0.8

        # Theme and variations (A followed by variations)
        if num_sections >= 4 and all(s.similarity_to_previous > 0.5 for s in sections[1:]):
            return FormType.THEME_VARIATIONS, 0.7

        # Through-composed (low similarity between sections)
        if unique_labels == num_sections and num_sections >= 4:
            avg_similarity = np.mean([s.similarity_to_previous for s in sections[1:]])
            if avg_similarity < 0.4:
                return FormType.THROUGH_COMPOSED, 0.75

        # Unknown
        return FormType.UNKNOWN, 0.5

    # ==========================================================================
    # TRANSITION DETECTION
    # ==========================================================================

    def _detect_transitions(self, sections: List[StructuralSection],
                           notes: List[NoteEvent]) -> List[Transition]:
        """Detect and analyze transitions between sections."""
        transitions = []

        for i in range(len(sections) - 1):
            current_section = sections[i]
            next_section = sections[i + 1]

            # Transition window (last beat of current + first beat of next)
            transition_start = current_section.end_time - 1.0
            transition_end = next_section.start_time + 1.0

            transition_notes = [n for n in notes
                              if transition_start <= n.start_time < transition_end]

            # Analyze transition type
            transition_type = self._classify_transition(
                current_section, next_section, transition_notes
            )

            # Check for modulation
            has_modulation = False
            if current_section.key is not None and next_section.key is not None:
                has_modulation = current_section.key != next_section.key

            # Check for fill
            has_fill = self._detect_fill(transition_notes)

            # Intensity change
            intensity_change = next_section.dynamic_level - current_section.dynamic_level

            transition = Transition(
                start_time=current_section.end_time,
                end_time=next_section.start_time,
                from_section=current_section.label,
                to_section=next_section.label,
                transition_type=transition_type,
                has_fill=has_fill,
                has_modulation=has_modulation,
                from_key=current_section.key,
                to_key=next_section.key,
                intensity_change=intensity_change
            )

            transitions.append(transition)

            # Update sections with transitions
            current_section.transition_out = transition
            next_section.transition_in = transition

        return transitions

    def _classify_transition(self, section1: StructuralSection,
                            section2: StructuralSection,
                            transition_notes: List[NoteEvent]) -> TransitionType:
        """Classify the type of transition between sections."""
        # Check for key change
        if section1.key is not None and section2.key is not None:
            if section1.key != section2.key:
                return TransitionType.MODULATION

        # Check for silence
        if not transition_notes:
            return TransitionType.SILENCE

        # Check for buildup (increasing density/dynamics)
        if section2.dynamic_level > section1.dynamic_level + 0.2:
            return TransitionType.BUILDUP

        # Check for breakdown (decreasing density/dynamics)
        if section2.dynamic_level < section1.dynamic_level - 0.2:
            return TransitionType.BREAKDOWN

        # Check for fill pattern
        if self._detect_fill(transition_notes):
            return TransitionType.FILL

        # Default: direct transition
        return TransitionType.DIRECT

    def _detect_fill(self, notes: List[NoteEvent]) -> bool:
        """Detect if notes form a fill pattern (rapid notes, often drums)."""
        if not notes:
            return False

        # Check for high density
        duration = max([n.end_time for n in notes]) - min([n.start_time for n in notes])
        if duration > 0:
            density = len(notes) / duration
            return density > 4.0  # More than 4 notes per second

        return False

    # ==========================================================================
    # CLIMAX DETECTION
    # ==========================================================================

    def _detect_climaxes(self, sections: List[StructuralSection],
                        notes: List[NoteEvent]) -> List[Climax]:
        """
        Detect climax points in the music.

        Research: Huron (1996) - "The Melodic Arch in Western Folksongs"
        """
        climaxes = []

        if not notes or not NUMPY_AVAILABLE:
            return climaxes

        # Analyze pitch peaks
        pitch_climax = self._detect_pitch_climax(notes, sections)
        if pitch_climax:
            climaxes.append(pitch_climax)

        # Analyze dynamic peaks
        dynamic_climax = self._detect_dynamic_climax(notes, sections)
        if dynamic_climax:
            climaxes.append(dynamic_climax)

        # Analyze density peaks
        density_climax = self._detect_density_climax(sections)
        if density_climax:
            climaxes.append(density_climax)

        # Mark sections with climaxes
        for climax in climaxes:
            for section in sections:
                if section.start_time <= climax.time < section.end_time:
                    section.is_climax = True
                    climax.section = section.label

        return climaxes

    def _detect_pitch_climax(self, notes: List[NoteEvent],
                            sections: List[StructuralSection]) -> Optional[Climax]:
        """Detect climax based on highest pitch."""
        if not notes:
            return None

        # Find highest note
        max_pitch_note = max(notes, key=lambda n: n.pitch)

        # Find which section it's in
        for section in sections:
            if section.start_time <= max_pitch_note.start_time < section.end_time:
                return Climax(
                    time=max_pitch_note.start_time,
                    bar=section.start_bar + int((max_pitch_note.start_time - section.start_time) / 2.0),
                    section=section.label,
                    pitch_peak=True,
                    confidence=0.7
                )

        return None

    def _detect_dynamic_climax(self, notes: List[NoteEvent],
                              sections: List[StructuralSection]) -> Optional[Climax]:
        """Detect climax based on highest dynamics."""
        if not notes:
            return None

        # Find loudest note
        max_velocity_note = max(notes, key=lambda n: n.velocity)

        # Find which section it's in
        for section in sections:
            if section.start_time <= max_velocity_note.start_time < section.end_time:
                return Climax(
                    time=max_velocity_note.start_time,
                    bar=section.start_bar,
                    section=section.label,
                    dynamic_peak=True,
                    confidence=0.6
                )

        return None

    def _detect_density_climax(self, sections: List[StructuralSection]) -> Optional[Climax]:
        """Detect climax based on highest note density."""
        if not sections:
            return None

        # Find section with highest density
        max_density_section = max(sections, key=lambda s: s.density)

        if max_density_section.density > 0:
            return Climax(
                time=max_density_section.start_time + max_density_section.duration / 2,
                bar=max_density_section.start_bar + max_density_section.num_bars // 2,
                section=max_density_section.label,
                density_peak=True,
                confidence=0.6
            )

        return None

    # ==========================================================================
    # MOTIF EXTRACTION
    # ==========================================================================

    def _extract_motifs(self, notes: List[NoteEvent], tempo: float) -> List[Motif]:
        """Extract recurring motifs from the melody."""
        motifs = []

        if not notes or not NUMPY_AVAILABLE:
            return motifs

        # Extract melodic line (highest notes)
        melody = self._extract_melody_line(notes)

        if len(melody) < 4:
            return motifs

        # Extract candidate motifs (3-8 notes)
        for motif_length in range(3, 9):
            if len(melody) < motif_length:
                continue

            # Extract all n-grams of this length
            for i in range(len(melody) - motif_length + 1):
                motif_notes = melody[i:i + motif_length]

                # Create motif
                pitches = [n.pitch for n in motif_notes]
                intervals = [pitches[j+1] - pitches[j] for j in range(len(pitches) - 1)]
                durations = [n.duration for n in motif_notes]
                contour = [np.sign(intervals[j]) for j in range(len(intervals))]

                motif = Motif(
                    pitches=pitches,
                    intervals=intervals,
                    durations=durations,
                    contour=contour,
                    start_time=motif_notes[0].start_time,
                    duration=sum(durations)
                )

                # Find occurrences and transformations
                occurrences, transformations = self._find_motif_occurrences(motif, melody)

                if len(occurrences) >= 2:  # At least 2 occurrences
                    motif.occurrences = occurrences
                    motif.transformations = transformations
                    motifs.append(motif)

        # Remove duplicate motifs
        motifs = self._deduplicate_motifs(motifs)

        return motifs

    def _extract_melody_line(self, notes: List[NoteEvent]) -> List[NoteEvent]:
        """Extract the main melody line (highest notes)."""
        if not notes:
            return []

        # Sort by start time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)

        # For each time window, take the highest note
        melody = []
        window_size = 0.5  # 0.5 second window

        current_time = 0.0
        max_time = max([n.end_time for n in notes])

        while current_time < max_time:
            window_notes = [n for n in sorted_notes
                           if current_time <= n.start_time < current_time + window_size]

            if window_notes:
                highest = max(window_notes, key=lambda n: n.pitch)
                melody.append(highest)

            current_time += window_size

        return melody

    def _find_motif_occurrences(self, motif: Motif,
                               melody: List[NoteEvent]) -> Tuple[List[float], List[MotivicTransformation]]:
        """Find occurrences of a motif and its transformations in the melody."""
        occurrences = [motif.start_time]
        transformations = []

        motif_length = len(motif.pitches)

        for i in range(len(melody) - motif_length + 1):
            candidate_notes = melody[i:i + motif_length]
            candidate_pitches = [n.pitch for n in candidate_notes]
            candidate_intervals = [candidate_pitches[j+1] - candidate_pitches[j]
                                  for j in range(len(candidate_pitches) - 1)]

            # Skip if same as original
            if candidate_notes[0].start_time == motif.start_time:
                continue

            # Exact repetition
            if candidate_pitches == motif.pitches:
                occurrences.append(candidate_notes[0].start_time)
                transformations.append(MotivicTransformation.EXACT_REPETITION)

            # Transposition (same intervals)
            elif candidate_intervals == motif.intervals:
                occurrences.append(candidate_notes[0].start_time)
                transformations.append(MotivicTransformation.TRANSPOSITION)

            # Inversion (inverted intervals)
            elif candidate_intervals == [-x for x in motif.intervals]:
                occurrences.append(candidate_notes[0].start_time)
                transformations.append(MotivicTransformation.INVERSION)

            # Retrograde (reversed pitches)
            elif candidate_pitches == list(reversed(motif.pitches)):
                occurrences.append(candidate_notes[0].start_time)
                transformations.append(MotivicTransformation.RETROGRADE)

        return occurrences, transformations

    def _deduplicate_motifs(self, motifs: List[Motif]) -> List[Motif]:
        """Remove duplicate motifs, keeping the most frequent ones."""
        if not motifs:
            return []

        # Sort by number of occurrences (descending)
        sorted_motifs = sorted(motifs, key=lambda m: len(m.occurrences), reverse=True)

        # Keep unique motifs
        unique_motifs = []
        seen_patterns = set()

        for motif in sorted_motifs:
            pattern_key = tuple(motif.intervals)
            if pattern_key not in seen_patterns:
                unique_motifs.append(motif)
                seen_patterns.add(pattern_key)

        return unique_motifs[:20]  # Limit to top 20 motifs

    # ==========================================================================
    # REPETITION ANALYSIS
    # ==========================================================================

    def _analyze_repetitions(self, sections: List[StructuralSection]) -> List[RepetitionGroup]:
        """Analyze repetition and variation patterns across sections."""
        if not sections or not NUMPY_AVAILABLE:
            return []

        repetition_groups = []

        # Group sections by label
        label_groups = defaultdict(list)
        for section in sections:
            label_groups[section.label].append(section)

        # Analyze each group
        for label, group_sections in label_groups.items():
            if len(group_sections) < 2:
                continue  # No repetition

            # Extract features for each section
            features = []
            for section in group_sections:
                if section.notes:
                    features.append(self._extract_segment_features(section.notes))
                else:
                    features.append(np.zeros(20))

            # Compute similarity matrix
            n = len(group_sections)
            similarity_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if np.linalg.norm(features[i]) > 0 and np.linalg.norm(features[j]) > 0:
                        similarity_matrix[i, j] = 1 - cosine(features[i], features[j])

            # Calculate average similarity
            avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])

            # Determine variation type
            if avg_similarity > 0.9:
                variation_type = "exact"
            elif avg_similarity > 0.7:
                variation_type = "varied"
            else:
                variation_type = "developed"

            repetition_group = RepetitionGroup(
                sections=group_sections,
                similarity_matrix=similarity_matrix,
                average_similarity=avg_similarity,
                variation_type=variation_type
            )

            repetition_groups.append(repetition_group)

        return repetition_groups

    # ==========================================================================
    # METRIC CALCULATIONS
    # ==========================================================================

    def _calculate_repetition_ratio(self, sections: List[StructuralSection]) -> float:
        """Calculate what proportion of the piece is repeated material."""
        if not sections:
            return 0.0

        # Count unique vs repeated sections
        label_counts = Counter([s.label for s in sections])
        repeated_sections = sum(1 for count in label_counts.values() if count > 1)

        return repeated_sections / len(sections)

    def _calculate_development_ratio(self, sections: List[StructuralSection]) -> float:
        """Calculate how much development/variation occurs."""
        if len(sections) < 2:
            return 0.0

        # Average similarity to previous sections
        similarities = [s.similarity_to_previous for s in sections[1:]]

        if not similarities:
            return 0.0

        avg_similarity = np.mean(similarities)

        # Development ratio: 1 - similarity (low similarity = high development)
        return 1.0 - avg_similarity

    def _calculate_contrast_ratio(self, sections: List[StructuralSection]) -> float:
        """Calculate the diversity of sections."""
        if not sections:
            return 0.0

        # Number of unique sections / total sections
        unique_labels = len(set([s.label for s in sections]))

        return unique_labels / len(sections)

    def _calculate_climax_position(self, climaxes: List[Climax],
                                   total_duration: float) -> float:
        """Calculate the position of the main climax (0.0-1.0)."""
        if not climaxes or total_duration == 0:
            return 0.618  # Golden ratio default

        # Find the climax with highest confidence
        main_climax = max(climaxes, key=lambda c: c.confidence)

        return main_climax.time / total_duration

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    def _extract_notes(self, midi: MidiFile) -> List[NoteEvent]:
        """Extract all note events from MIDI file."""
        notes = []

        for track_idx, track in enumerate(midi.tracks):
            current_time = 0.0
            active_notes = {}  # pitch -> (start_time, velocity)

            for msg in track:
                current_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = (current_time, msg.velocity, msg.channel)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time, velocity, channel = active_notes[msg.note]
                        duration = current_time - start_time

                        note = NoteEvent(
                            start_time=start_time,
                            duration=duration,
                            pitch=msg.note,
                            velocity=velocity,
                            channel=channel,
                            track=track_idx
                        )
                        notes.append(note)

                        del active_notes[msg.note]

        return sorted(notes, key=lambda n: n.start_time)

    def _estimate_tempo(self, midi: MidiFile) -> float:
        """Estimate tempo from MIDI file."""
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    return mido.tempo2bpm(msg.tempo)

        return 120.0  # Default tempo

    def _estimate_time_signature(self, midi: MidiFile) -> int:
        """Estimate time signature (beats per bar)."""
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'time_signature':
                    return msg.numerator

        return 4  # Default 4/4

    def _detect_primary_key(self, notes: List[NoteEvent]) -> Tuple[Optional[int], Optional[str]]:
        """Detect the primary key using Krumhansl-Schmuckler algorithm."""
        if not notes or not NUMPY_AVAILABLE:
            return None, None

        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Count pitch classes
        pitch_class_counts = np.zeros(12)
        for note in notes:
            pitch_class_counts[note.pitch_class] += note.duration * note.velocity

        # Normalize
        if pitch_class_counts.sum() > 0:
            pitch_class_counts /= pitch_class_counts.sum()

        # Correlate with all keys
        best_correlation = -1
        best_key = 0
        best_mode = "major"

        for tonic in range(12):
            # Rotate profiles to match tonic
            major_rotated = np.roll(major_profile, tonic)
            minor_rotated = np.roll(minor_profile, tonic)

            # Compute correlations
            major_corr = np.corrcoef(pitch_class_counts, major_rotated)[0, 1]
            minor_corr = np.corrcoef(pitch_class_counts, minor_rotated)[0, 1]

            if major_corr > best_correlation:
                best_correlation = major_corr
                best_key = tonic
                best_mode = "major"

            if minor_corr > best_correlation:
                best_correlation = minor_corr
                best_key = tonic
                best_mode = "minor"

        return best_key, best_mode

    def _calculate_dynamic_level(self, notes: List[NoteEvent]) -> float:
        """Calculate average dynamic level (0.0-1.0) from notes."""
        if not notes:
            return 0.5

        velocities = [n.velocity for n in notes]
        avg_velocity = np.mean(velocities)

        return avg_velocity / 127.0

    def _calculate_density(self, notes: List[NoteEvent], duration: float) -> float:
        """Calculate note density (notes per second)."""
        if duration == 0:
            return 0.0

        return len(notes) / duration


# ==============================================================================
# ADVANCED ANALYSIS METHODS
# ==============================================================================

class AdvancedStructureAnalyzer:
    """
    Advanced structural analysis methods including:
    - Genre-specific form detection
    - Harmonic progression analysis
    - Texture analysis
    - Statistical methods
    - Deep pattern recognition
    """

    def __init__(self, specialist: 'StructureSpecialist'):
        self.specialist = specialist

    # ==========================================================================
    # HARMONIC STRUCTURE ANALYSIS
    # ==========================================================================

    def analyze_harmonic_structure(self, notes: List[NoteEvent],
                                   sections: List[StructuralSection]) -> Dict[str, Any]:
        """
        Analyze harmonic structure and progressions.

        Returns:
            Dictionary with harmonic analysis results
        """
        results = {
            'chord_progressions': [],
            'modulations': [],
            'cadences': [],
            'harmonic_rhythm': 0.0,
            'tonal_stability': 0.0
        }

        if not notes or not NUMPY_AVAILABLE:
            return results

        # Analyze each section for chord progressions
        for section in sections:
            if section.notes:
                chords = self._detect_chords_in_section(section.notes)
                results['chord_progressions'].append({
                    'section': section.label,
                    'chords': chords
                })

        # Detect modulations between sections
        for i in range(len(sections) - 1):
            if sections[i].key != sections[i+1].key and sections[i].key is not None:
                results['modulations'].append({
                    'from_section': sections[i].label,
                    'to_section': sections[i+1].label,
                    'from_key': sections[i].key,
                    'to_key': sections[i+1].key,
                    'interval': (sections[i+1].key - sections[i].key) % 12
                })

        # Calculate harmonic rhythm (chord changes per measure)
        results['harmonic_rhythm'] = self._calculate_harmonic_rhythm(notes)

        # Calculate tonal stability
        results['tonal_stability'] = self._calculate_tonal_stability(notes)

        return results

    def _detect_chords_in_section(self, notes: List[NoteEvent]) -> List[Dict[str, Any]]:
        """Detect chords in a section using vertical slice analysis."""
        if not notes:
            return []

        chords = []
        window_size = 0.5  # Half-second windows

        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)

        current_time = 0.0
        max_time = max([n.end_time for n in sorted_notes])

        while current_time < max_time:
            # Get notes active in this window
            active_notes = [
                n for n in sorted_notes
                if n.start_time <= current_time < n.end_time
            ]

            if active_notes:
                # Extract pitch classes
                pitch_classes = list(set([n.pitch_class for n in active_notes]))
                pitch_classes.sort()

                # Detect chord type
                chord_quality = self._identify_chord_quality(pitch_classes)

                chords.append({
                    'time': current_time,
                    'pitch_classes': pitch_classes,
                    'quality': chord_quality,
                    'bass': min([n.pitch for n in active_notes])
                })

            current_time += window_size

        return chords

    def _identify_chord_quality(self, pitch_classes: List[int]) -> str:
        """Identify chord quality from pitch classes."""
        if len(pitch_classes) < 2:
            return "unknown"

        # Normalize to root = 0
        intervals = sorted([(pc - pitch_classes[0]) % 12 for pc in pitch_classes])

        # Major triads
        if intervals == [0, 4, 7]:
            return "major"
        elif intervals == [0, 3, 7]:
            return "minor"
        elif intervals == [0, 3, 6]:
            return "diminished"
        elif intervals == [0, 4, 8]:
            return "augmented"

        # Seventh chords
        elif intervals == [0, 4, 7, 10]:
            return "dominant_7"
        elif intervals == [0, 4, 7, 11]:
            return "major_7"
        elif intervals == [0, 3, 7, 10]:
            return "minor_7"
        elif intervals == [0, 3, 6, 10]:
            return "half_diminished"
        elif intervals == [0, 3, 6, 9]:
            return "diminished_7"

        # Extended chords
        elif len(intervals) >= 5:
            return "extended"

        return "other"

    def _calculate_harmonic_rhythm(self, notes: List[NoteEvent]) -> float:
        """Calculate harmonic rhythm (chord changes per measure)."""
        if not notes:
            return 0.0

        # Simplified: detect chord changes by looking at pitch class sets
        window_size = 0.5
        max_time = max([n.end_time for n in notes])
        num_windows = int(max_time / window_size)

        changes = 0
        prev_pcs = set()

        for i in range(num_windows):
            window_start = i * window_size
            window_end = (i + 1) * window_size

            active_notes = [n for n in notes
                           if n.start_time <= window_start < n.end_time]

            current_pcs = set([n.pitch_class for n in active_notes])

            if current_pcs != prev_pcs and len(current_pcs) > 0:
                changes += 1

            prev_pcs = current_pcs

        # Assume 4/4, 2 seconds per measure at 120 BPM
        measures = max_time / 2.0
        return changes / max(measures, 1)

    def _calculate_tonal_stability(self, notes: List[NoteEvent]) -> float:
        """
        Calculate tonal stability (how stable the key is).

        Returns value 0.0-1.0, where 1.0 is very stable.
        """
        if not notes or not NUMPY_AVAILABLE:
            return 0.5

        # Count pitch class distribution
        pc_counts = np.zeros(12)
        for note in notes:
            pc_counts[note.pitch_class] += note.duration

        if pc_counts.sum() == 0:
            return 0.5

        # Normalize
        pc_dist = pc_counts / pc_counts.sum()

        # Calculate entropy (low entropy = high stability)
        entropy = -np.sum([p * np.log2(p + 1e-10) for p in pc_dist if p > 0])
        max_entropy = np.log2(12)  # Maximum for uniform distribution

        # Convert to stability (inverse of normalized entropy)
        stability = 1.0 - (entropy / max_entropy)

        return stability

    # ==========================================================================
    # TEXTURE ANALYSIS
    # ==========================================================================

    def analyze_texture(self, sections: List[StructuralSection]) -> Dict[str, Any]:
        """
        Analyze textural changes across sections.

        Texture includes:
        - Polyphony (number of voices)
        - Register (pitch range)
        - Density (notes per second)
        - Homogeneity (how similar voices are)
        """
        results = {
            'section_textures': [],
            'texture_changes': [],
            'average_polyphony': 0.0,
            'texture_variety': 0.0
        }

        if not sections:
            return results

        polyphony_values = []

        for section in sections:
            texture = self._analyze_section_texture(section)
            results['section_textures'].append({
                'section': section.label,
                **texture
            })
            polyphony_values.append(texture['polyphony'])

        # Calculate texture changes
        for i in range(len(results['section_textures']) - 1):
            current = results['section_textures'][i]
            next_tex = results['section_textures'][i+1]

            # Calculate texture change magnitude
            change_magnitude = abs(current['polyphony'] - next_tex['polyphony'])
            change_magnitude += abs(current['register_span'] - next_tex['register_span'])
            change_magnitude += abs(current['density'] - next_tex['density'])

            results['texture_changes'].append({
                'from_section': current['section'],
                'to_section': next_tex['section'],
                'magnitude': change_magnitude
            })

        # Average polyphony
        if polyphony_values:
            results['average_polyphony'] = np.mean(polyphony_values)
            results['texture_variety'] = np.std(polyphony_values)

        return results

    def _analyze_section_texture(self, section: StructuralSection) -> Dict[str, Any]:
        """Analyze texture of a single section."""
        if not section.notes:
            return {
                'polyphony': 0,
                'register_span': 0,
                'density': 0.0,
                'homogeneity': 0.0
            }

        # Calculate average polyphony (simultaneous notes)
        polyphony = self._calculate_polyphony(section.notes)

        # Calculate register span
        pitches = [n.pitch for n in section.notes]
        register_span = max(pitches) - min(pitches) if pitches else 0

        # Density already calculated
        density = section.density

        # Homogeneity (rhythmic similarity)
        homogeneity = self._calculate_rhythmic_homogeneity(section.notes)

        return {
            'polyphony': polyphony,
            'register_span': register_span,
            'density': density,
            'homogeneity': homogeneity
        }

    def _calculate_polyphony(self, notes: List[NoteEvent]) -> float:
        """Calculate average number of simultaneous notes."""
        if not notes:
            return 0.0

        # Sample at regular intervals
        sample_interval = 0.1  # Sample every 0.1 seconds
        max_time = max([n.end_time for n in notes])
        num_samples = int(max_time / sample_interval)

        polyphony_samples = []

        for i in range(num_samples):
            sample_time = i * sample_interval

            # Count active notes at this time
            active = sum(1 for n in notes if n.start_time <= sample_time < n.end_time)
            polyphony_samples.append(active)

        return np.mean(polyphony_samples) if polyphony_samples else 0.0

    def _calculate_rhythmic_homogeneity(self, notes: List[NoteEvent]) -> float:
        """
        Calculate rhythmic homogeneity (0.0 = diverse, 1.0 = uniform).
        """
        if len(notes) < 2:
            return 1.0

        # Get inter-onset intervals
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        iois = [sorted_notes[i+1].start_time - sorted_notes[i].start_time
                for i in range(len(sorted_notes) - 1)]

        if not iois:
            return 1.0

        # Calculate coefficient of variation
        mean_ioi = np.mean(iois)
        std_ioi = np.std(iois)

        if mean_ioi == 0:
            return 1.0

        cv = std_ioi / mean_ioi

        # Convert to homogeneity (inverse of CV, normalized)
        homogeneity = 1.0 / (1.0 + cv)

        return homogeneity

    # ==========================================================================
    # GENRE-SPECIFIC ANALYSIS
    # ==========================================================================

    def detect_genre_specific_forms(self, analysis: StructuralAnalysis) -> Dict[str, Any]:
        """
        Detect genre-specific structural patterns.

        Includes:
        - Blues forms (12-bar blues)
        - Jazz forms (AABA, rhythm changes)
        - Classical forms (sonata, rondo, fugue)
        - Pop forms (verse-chorus-bridge)
        """
        results = {
            'blues_score': 0.0,
            'jazz_standard_score': 0.0,
            'sonata_score': 0.0,
            'pop_score': 0.0,
            'likely_genre': 'unknown'
        }

        # Check for 12-bar blues patterns
        results['blues_score'] = self._check_blues_form(analysis)

        # Check for jazz standard (AABA)
        results['jazz_standard_score'] = self._check_jazz_standard_form(analysis)

        # Check for sonata form
        results['sonata_score'] = self._check_sonata_form(analysis)

        # Check for pop structure
        results['pop_score'] = self._check_pop_form(analysis)

        # Determine likely genre
        scores = {
            'blues': results['blues_score'],
            'jazz': results['jazz_standard_score'],
            'classical': results['sonata_score'],
            'pop': results['pop_score']
        }

        results['likely_genre'] = max(scores, key=scores.get)

        return results

    def _check_blues_form(self, analysis: StructuralAnalysis) -> float:
        """Check if structure matches 12-bar blues."""
        score = 0.0

        # Check for 12-bar sections
        twelve_bar_sections = sum(1 for s in analysis.sections if s.num_bars == 12)
        if twelve_bar_sections >= 2:
            score += 0.5

        # Check for repetitive structure
        if analysis.repetition_ratio > 0.7:
            score += 0.3

        # Check for simple form (AAA or AAAA)
        labels = [s.label for s in analysis.sections]
        if len(set(labels)) <= 2:
            score += 0.2

        return min(score, 1.0)

    def _check_jazz_standard_form(self, analysis: StructuralAnalysis) -> float:
        """Check if structure matches jazz standard (AABA)."""
        score = 0.0

        # Check form type
        if analysis.form_type == FormType.AABA:
            score += 0.6

        # Check for 32 bars total (4 x 8-bar sections)
        if analysis.total_bars == 32:
            score += 0.2

        # Check for 8-bar sections
        eight_bar_sections = sum(1 for s in analysis.sections if s.num_bars == 8)
        if eight_bar_sections == 4:
            score += 0.2

        return min(score, 1.0)

    def _check_sonata_form(self, analysis: StructuralAnalysis) -> float:
        """Check if structure matches sonata form."""
        score = 0.0

        # Check form type
        if analysis.form_type == FormType.SONATA:
            score += 0.5

        # Check for three main sections (exposition, development, recapitulation)
        section_types = [s.section_type for s in analysis.sections]
        if (SectionType.EXPOSITION in section_types and
            SectionType.DEVELOPMENT in section_types and
            SectionType.RECAPITULATION in section_types):
            score += 0.3

        # Check for modulations (common in sonata form)
        if len(analysis.transitions) > 0 and any(t.has_modulation for t in analysis.transitions):
            score += 0.2

        return min(score, 1.0)

    def _check_pop_form(self, analysis: StructuralAnalysis) -> float:
        """Check if structure matches pop song form."""
        score = 0.0

        # Check form type
        if analysis.form_type in [FormType.VERSE_CHORUS, FormType.VERSE_CHORUS_BRIDGE]:
            score += 0.6

        # Check for verse and chorus sections
        section_types = [s.section_type for s in analysis.sections]
        if SectionType.VERSE in section_types and SectionType.CHORUS in section_types:
            score += 0.3

        # Check for bridge
        if SectionType.BRIDGE in section_types:
            score += 0.1

        return min(score, 1.0)

    # ==========================================================================
    # STATISTICAL ANALYSIS
    # ==========================================================================

    def compute_structural_statistics(self, analysis: StructuralAnalysis) -> Dict[str, Any]:
        """
        Compute comprehensive statistical measures of structure.
        """
        stats = {}

        if not analysis.sections:
            return stats

        # Section length statistics
        section_lengths = [s.num_bars for s in analysis.sections]
        stats['section_length_mean'] = np.mean(section_lengths)
        stats['section_length_std'] = np.std(section_lengths)
        stats['section_length_min'] = min(section_lengths)
        stats['section_length_max'] = max(section_lengths)
        stats['section_length_median'] = np.median(section_lengths)

        # Dynamic level statistics
        dynamic_levels = [s.dynamic_level for s in analysis.sections]
        stats['dynamic_mean'] = np.mean(dynamic_levels)
        stats['dynamic_std'] = np.std(dynamic_levels)
        stats['dynamic_range'] = max(dynamic_levels) - min(dynamic_levels)

        # Density statistics
        densities = [s.density for s in analysis.sections]
        stats['density_mean'] = np.mean(densities)
        stats['density_std'] = np.std(densities)
        stats['density_range'] = max(densities) - min(densities)

        # Similarity statistics
        similarities = [s.similarity_to_previous for s in analysis.sections[1:]]
        if similarities:
            stats['similarity_mean'] = np.mean(similarities)
            stats['similarity_std'] = np.std(similarities)

        # Transition statistics
        if analysis.transitions:
            transition_types = [t.transition_type.value for t in analysis.transitions]
            stats['transition_diversity'] = len(set(transition_types)) / len(transition_types)

            intensity_changes = [abs(t.intensity_change) for t in analysis.transitions]
            stats['avg_intensity_change'] = np.mean(intensity_changes)

        # Motif statistics
        if analysis.motifs:
            motif_lengths = [len(m.pitches) for m in analysis.motifs]
            stats['motif_length_mean'] = np.mean(motif_lengths)

            motif_occurrences = [len(m.occurrences) for m in analysis.motifs]
            stats['motif_occurrences_mean'] = np.mean(motif_occurrences)

        # Climax statistics
        if analysis.climaxes:
            climax_positions = [c.time / analysis.total_duration for c in analysis.climaxes]
            stats['climax_position_mean'] = np.mean(climax_positions)
            stats['climax_position_std'] = np.std(climax_positions)

        return stats

    # ==========================================================================
    # PATTERN RECOGNITION
    # ==========================================================================

    def detect_advanced_patterns(self, analysis: StructuralAnalysis) -> Dict[str, Any]:
        """
        Detect advanced structural patterns:
        - Arc structure (intensity builds to climax, then falls)
        - Symmetry (palindromic structures)
        - Fractal patterns (self-similarity at multiple scales)
        - Golden ratio proportions
        """
        patterns = {
            'has_arc_structure': False,
            'arc_score': 0.0,
            'has_symmetry': False,
            'symmetry_score': 0.0,
            'has_golden_ratio': False,
            'golden_ratio_score': 0.0,
            'has_fractal_pattern': False,
            'fractal_score': 0.0
        }

        # Detect arc structure
        arc_score = self._detect_arc_structure(analysis)
        patterns['arc_score'] = arc_score
        patterns['has_arc_structure'] = arc_score > 0.6

        # Detect symmetry
        symmetry_score = self._detect_symmetry(analysis)
        patterns['symmetry_score'] = symmetry_score
        patterns['has_symmetry'] = symmetry_score > 0.6

        # Detect golden ratio
        golden_score = self._detect_golden_ratio(analysis)
        patterns['golden_ratio_score'] = golden_score
        patterns['has_golden_ratio'] = golden_score > 0.7

        # Detect fractal patterns
        fractal_score = self._detect_fractal_patterns(analysis)
        patterns['fractal_score'] = fractal_score
        patterns['has_fractal_pattern'] = fractal_score > 0.5

        return patterns

    def _detect_arc_structure(self, analysis: StructuralAnalysis) -> float:
        """
        Detect arc structure (intensity builds to climax, then falls).

        Research: Huron (1996) - "The Melodic Arch in Western Folksongs"
        """
        if len(analysis.sections) < 3:
            return 0.0

        # Get dynamic levels across sections
        dynamics = [s.dynamic_level for s in analysis.sections]

        # Find peak position
        peak_idx = dynamics.index(max(dynamics))
        peak_position = peak_idx / len(dynamics)

        # Check if peak is in middle (0.3 - 0.7)
        if not (0.3 <= peak_position <= 0.7):
            return 0.0

        # Check if dynamics increase before peak and decrease after
        before_peak = dynamics[:peak_idx+1]
        after_peak = dynamics[peak_idx:]

        # Calculate slopes
        if len(before_peak) > 1:
            before_increasing = sum(before_peak[i+1] > before_peak[i]
                                  for i in range(len(before_peak)-1))
            before_score = before_increasing / (len(before_peak) - 1)
        else:
            before_score = 0.5

        if len(after_peak) > 1:
            after_decreasing = sum(after_peak[i+1] < after_peak[i]
                                 for i in range(len(after_peak)-1))
            after_score = after_decreasing / (len(after_peak) - 1)
        else:
            after_score = 0.5

        # Combined score
        arc_score = (before_score + after_score) / 2.0

        return arc_score

    def _detect_symmetry(self, analysis: StructuralAnalysis) -> float:
        """Detect palindromic/symmetric structure (ABA, ABCBA, etc.)."""
        if len(analysis.sections) < 3:
            return 0.0

        labels = [s.label for s in analysis.sections]

        # Check if labels are palindromic
        if labels == list(reversed(labels)):
            return 1.0

        # Check for partial symmetry
        n = len(labels)
        matches = sum(1 for i in range(n // 2) if labels[i] == labels[-(i+1)])
        symmetry_score = matches / (n // 2)

        return symmetry_score

    def _detect_golden_ratio(self, analysis: StructuralAnalysis) -> float:
        """
        Detect golden ratio proportions (0.618).

        Many classical pieces place climax at golden ratio point.
        """
        golden = 0.618
        tolerance = 0.1

        score = 0.0

        # Check climax position
        if abs(analysis.climax_position - golden) < tolerance:
            score += 0.5

        # Check section proportions
        if len(analysis.sections) >= 2:
            for i in range(len(analysis.sections) - 1):
                section1_duration = analysis.sections[i].duration
                section2_duration = analysis.sections[i+1].duration
                total = section1_duration + section2_duration

                if total > 0:
                    ratio = section1_duration / total
                    if abs(ratio - golden) < tolerance or abs(ratio - (1-golden)) < tolerance:
                        score += 0.3

        return min(score, 1.0)

    def _detect_fractal_patterns(self, analysis: StructuralAnalysis) -> float:
        """
        Detect fractal-like patterns (self-similarity at multiple scales).
        """
        if len(analysis.sections) < 4:
            return 0.0

        # Check if section structure repeats within itself
        # This is a simplified check for self-similarity

        labels = [s.label for s in analysis.sections]

        # Look for repeated patterns at different scales
        pattern_scores = []

        for pattern_length in range(2, len(labels) // 2 + 1):
            # Extract patterns of this length
            patterns = []
            for i in range(len(labels) - pattern_length + 1):
                pattern = tuple(labels[i:i+pattern_length])
                patterns.append(pattern)

            # Count unique vs total
            if patterns:
                unique_ratio = len(set(patterns)) / len(patterns)
                # Low unique ratio = high repetition = higher fractal score
                pattern_scores.append(1.0 - unique_ratio)

        if pattern_scores:
            return np.mean(pattern_scores)

        return 0.0


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def analyze_midi_structure(midi_path: str, **kwargs) -> StructuralAnalysis:
    """
    Convenience function to analyze MIDI file structure.

    Args:
        midi_path: Path to MIDI file
        **kwargs: Additional arguments for StructureSpecialist

    Returns:
        Structural analysis result
    """
    specialist = StructureSpecialist(**kwargs)
    return specialist.analyze(midi_path)


def extract_structure_features(midi_path: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to extract structure parameters.

    Args:
        midi_path: Path to MIDI file
        **kwargs: Additional arguments for StructureSpecialist

    Returns:
        Dictionary of structure parameters
    """
    specialist = StructureSpecialist(**kwargs)
    analysis = specialist.analyze(midi_path)
    return specialist.extract_structure_parameters(analysis)


def print_structure_report(analysis: StructuralAnalysis) -> None:
    """
    Print a human-readable structure analysis report.

    Args:
        analysis: Structural analysis result
    """
    print("=" * 80)
    print("STRUCTURAL ANALYSIS REPORT")
    print("=" * 80)
    print()

    print(f"Form Type: {analysis.form_type.value} (confidence: {analysis.form_confidence:.2f})")
    print(f"Total Duration: {analysis.total_duration:.1f}s ({analysis.total_bars} bars)")
    print(f"Primary Key: {analysis.primary_key} {analysis.primary_mode}")
    print(f"Average Tempo: {analysis.average_tempo:.0f} BPM")
    print()

    print("SECTIONS:")
    print("-" * 80)
    for section in analysis.sections:
        print(f"  {section.label} ({section.section_type.value}): "
              f"bars {section.start_bar}-{section.end_bar} "
              f"(dynamic: {section.dynamic_level:.2f}, density: {section.density:.2f})")
    print()

    print("TRANSITIONS:")
    print("-" * 80)
    for trans in analysis.transitions:
        print(f"  {trans.from_section} → {trans.to_section}: {trans.transition_type.value}")
        if trans.has_modulation:
            print(f"    Modulation: {trans.from_key} → {trans.to_key}")
    print()

    print("CLIMAXES:")
    print("-" * 80)
    for climax in analysis.climaxes:
        types = []
        if climax.pitch_peak:
            types.append("pitch")
        if climax.dynamic_peak:
            types.append("dynamic")
        if climax.density_peak:
            types.append("density")
        print(f"  Bar {climax.bar} ({climax.section}): {', '.join(types)} "
              f"(confidence: {climax.confidence:.2f})")
    print()

    print("MOTIFS:")
    print("-" * 80)
    for i, motif in enumerate(analysis.motifs[:5], 1):
        print(f"  Motif {i}: {len(motif.occurrences)} occurrences")
        print(f"    Intervals: {motif.intervals}")
        if motif.transformations:
            trans_counts = Counter([t.value for t in motif.transformations])
            print(f"    Transformations: {dict(trans_counts)}")
    print()

    print("STRUCTURAL METRICS:")
    print("-" * 80)
    print(f"  Repetition Ratio: {analysis.repetition_ratio:.2f}")
    print(f"  Development Ratio: {analysis.development_ratio:.2f}")
    print(f"  Contrast Ratio: {analysis.contrast_ratio:.2f}")
    print(f"  Climax Position: {analysis.climax_position:.2f}")
    print()

    print("=" * 80)


def export_analysis_to_json(analysis: StructuralAnalysis, output_path: str) -> None:
    """
    Export structural analysis to JSON file.

    Args:
        analysis: Structural analysis result
        output_path: Output file path
    """
    data = {
        'form_type': analysis.form_type.value,
        'form_confidence': analysis.form_confidence,
        'total_duration': analysis.total_duration,
        'total_bars': analysis.total_bars,
        'primary_key': analysis.primary_key,
        'primary_mode': analysis.primary_mode,
        'average_tempo': analysis.average_tempo,
        'sections': [
            {
                'label': s.label,
                'type': s.section_type.value,
                'start_time': s.start_time,
                'end_time': s.end_time,
                'start_bar': s.start_bar,
                'end_bar': s.end_bar,
                'dynamic_level': s.dynamic_level,
                'density': s.density,
                'key': s.key,
                'mode': s.mode,
                'is_climax': s.is_climax,
                'similarity_to_previous': s.similarity_to_previous
            }
            for s in analysis.sections
        ],
        'transitions': [
            {
                'from_section': t.from_section,
                'to_section': t.to_section,
                'type': t.transition_type.value,
                'start_time': t.start_time,
                'end_time': t.end_time,
                'has_fill': t.has_fill,
                'has_modulation': t.has_modulation,
                'from_key': t.from_key,
                'to_key': t.to_key,
                'intensity_change': t.intensity_change
            }
            for t in analysis.transitions
        ],
        'climaxes': [
            {
                'time': c.time,
                'bar': c.bar,
                'section': c.section,
                'pitch_peak': c.pitch_peak,
                'dynamic_peak': c.dynamic_peak,
                'density_peak': c.density_peak,
                'harmonic_tension': c.harmonic_tension,
                'confidence': c.confidence
            }
            for c in analysis.climaxes
        ],
        'motifs': [
            {
                'pitches': m.pitches,
                'intervals': m.intervals,
                'durations': m.durations,
                'contour': m.contour,
                'start_time': m.start_time,
                'duration': m.duration,
                'num_occurrences': len(m.occurrences),
                'transformations': [t.value for t in m.transformations]
            }
            for m in analysis.motifs[:20]  # Limit to top 20
        ],
        'metrics': {
            'repetition_ratio': analysis.repetition_ratio,
            'development_ratio': analysis.development_ratio,
            'contrast_ratio': analysis.contrast_ratio,
            'climax_position': analysis.climax_position
        }
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_analysis_from_json(input_path: str) -> Dict[str, Any]:
    """
    Load structural analysis from JSON file.

    Args:
        input_path: Input file path

    Returns:
        Dictionary with analysis data
    """
    with open(input_path, 'r') as f:
        return json.load(f)


# ==============================================================================
# INTEGRATION WITH XGBOOST SYSTEM
# ==============================================================================

class StructureParameterExtractor:
    """
    Extracts structure parameters for the XGBoost training system.

    This class provides comprehensive parameter extraction tailored for
    the self-expanding inverse music generation system.
    """

    def __init__(self):
        self.specialist = StructureSpecialist()
        self.advanced_analyzer = None

    def extract_all_parameters(self, midi_path: str) -> Dict[str, Any]:
        """
        Extract all structure-related parameters from MIDI file.

        This method provides 100+ structure parameters for XGBoost training.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Dictionary of all structure parameters
        """
        # Basic structural analysis
        analysis = self.specialist.analyze(midi_path)
        params = self.specialist.extract_structure_parameters(analysis)

        # Advanced analysis
        self.advanced_analyzer = AdvancedStructureAnalyzer(self.specialist)

        # Add harmonic structure parameters
        harmonic_analysis = self.advanced_analyzer.analyze_harmonic_structure(
            self.specialist._extract_notes(MidiFile(midi_path)),
            analysis.sections
        )
        params.update(self._flatten_harmonic_params(harmonic_analysis))

        # Add texture parameters
        texture_analysis = self.advanced_analyzer.analyze_texture(analysis.sections)
        params.update(self._flatten_texture_params(texture_analysis))

        # Add genre-specific scores
        genre_analysis = self.advanced_analyzer.detect_genre_specific_forms(analysis)
        params.update({f'structure.genre.{k}': v for k, v in genre_analysis.items()})

        # Add statistical parameters
        stats = self.advanced_analyzer.compute_structural_statistics(analysis)
        params.update({f'structure.stats.{k}': v for k, v in stats.items()})

        # Add advanced patterns
        patterns = self.advanced_analyzer.detect_advanced_patterns(analysis)
        params.update({f'structure.pattern.{k}': v for k, v in patterns.items()})

        return params

    def _flatten_harmonic_params(self, harmonic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten harmonic analysis into parameters."""
        params = {}

        params['structure.harmonic.harmonic_rhythm'] = harmonic_analysis['harmonic_rhythm']
        params['structure.harmonic.tonal_stability'] = harmonic_analysis['tonal_stability']
        params['structure.harmonic.num_modulations'] = len(harmonic_analysis['modulations'])

        # Chord quality distribution
        if harmonic_analysis['chord_progressions']:
            all_chords = []
            for prog in harmonic_analysis['chord_progressions']:
                all_chords.extend([c['quality'] for c in prog['chords']])

            quality_counts = Counter(all_chords)
            total = len(all_chords)

            for quality in ['major', 'minor', 'dominant_7', 'major_7', 'minor_7',
                          'diminished', 'augmented', 'extended']:
                params[f'structure.harmonic.{quality}_ratio'] = quality_counts.get(quality, 0) / max(total, 1)

        return params

    def _flatten_texture_params(self, texture_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten texture analysis into parameters."""
        params = {}

        params['structure.texture.average_polyphony'] = texture_analysis['average_polyphony']
        params['structure.texture.texture_variety'] = texture_analysis['texture_variety']
        params['structure.texture.num_texture_changes'] = len(texture_analysis['texture_changes'])

        # Average texture change magnitude
        if texture_analysis['texture_changes']:
            avg_change = np.mean([t['magnitude'] for t in texture_analysis['texture_changes']])
            params['structure.texture.avg_change_magnitude'] = avg_change

        return params

    def generate_training_data(self, midi_files: List[str],
                              output_file: str = "structure_training_data.json") -> None:
        """
        Generate training data for XGBoost from multiple MIDI files.

        Args:
            midi_files: List of MIDI file paths
            output_file: Output JSON file for training data
        """
        training_data = []

        for i, midi_path in enumerate(midi_files):
            try:
                print(f"Processing {i+1}/{len(midi_files)}: {midi_path}")

                params = self.extract_all_parameters(midi_path)

                training_data.append({
                    'file': midi_path,
                    'parameters': params
                })

            except Exception as e:
                print(f"Error processing {midi_path}: {e}")
                continue

        # Save training data
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"\nGenerated training data for {len(training_data)} files")
        print(f"Saved to: {output_file}")


# ==============================================================================
# VISUALIZATION HELPERS
# ==============================================================================

def generate_structure_ascii_diagram(analysis: StructuralAnalysis) -> str:
    """
    Generate ASCII art diagram of musical structure.

    Args:
        analysis: Structural analysis result

    Returns:
        ASCII art string representation
    """
    diagram = []
    diagram.append("=" * 80)
    diagram.append("STRUCTURE DIAGRAM")
    diagram.append("=" * 80)
    diagram.append("")

    # Timeline
    timeline_width = 70
    total_duration = analysis.total_duration

    # Section timeline
    diagram.append("Sections:")
    timeline = [' '] * timeline_width

    for section in analysis.sections:
        start_pos = int((section.start_time / total_duration) * timeline_width)
        end_pos = int((section.end_time / total_duration) * timeline_width)

        for i in range(start_pos, min(end_pos, timeline_width)):
            timeline[i] = section.label[0]  # First letter

    diagram.append('|' + ''.join(timeline) + '|')

    # Labels
    label_line = [' '] * timeline_width
    for section in analysis.sections:
        start_pos = int((section.start_time / total_duration) * timeline_width)
        if start_pos < timeline_width - len(section.label):
            for i, char in enumerate(section.label):
                if start_pos + i < timeline_width:
                    label_line[start_pos + i] = char

    diagram.append(' ' + ''.join(label_line))
    diagram.append("")

    # Dynamic contour
    diagram.append("Dynamic Contour:")
    height = 10

    for h in range(height, 0, -1):
        line = ['|']
        for section in analysis.sections:
            section_width = int((section.duration / total_duration) * timeline_width)
            level = int(section.dynamic_level * height)

            char = '█' if level >= h else ' '
            line.extend([char] * max(1, section_width))

        diagram.append(''.join(line[:timeline_width+1]))

    diagram.append('|' + '-' * timeline_width + '|')
    diagram.append("")

    # Climaxes
    if analysis.climaxes:
        diagram.append("Climaxes:")
        climax_line = [' '] * timeline_width

        for climax in analysis.climaxes:
            pos = int((climax.time / total_duration) * timeline_width)
            if pos < timeline_width:
                climax_line[pos] = '*'

        diagram.append(' ' + ''.join(climax_line))
        diagram.append("")

    diagram.append("=" * 80)

    return '\n'.join(diagram)


def print_parameter_summary(params: Dict[str, Any]) -> None:
    """
    Print a summary of extracted parameters organized by category.

    Args:
        params: Parameter dictionary
    """
    print("=" * 80)
    print("STRUCTURE PARAMETERS SUMMARY")
    print("=" * 80)
    print()

    # Group parameters by category
    categories = defaultdict(list)

    for key, value in sorted(params.items()):
        parts = key.split('.')
        if len(parts) >= 2:
            category = parts[1]
            categories[category].append((key, value))

    # Print by category
    for category, params_list in sorted(categories.items()):
        print(f"{category.upper()}:")
        print("-" * 40)

        for key, value in params_list[:10]:  # Show first 10 per category
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

        if len(params_list) > 10:
            print(f"  ... and {len(params_list) - 10} more")

        print()

    print(f"Total parameters: {sum(len(p) for p in categories.values())}")
    print("=" * 80)


# ==============================================================================
# COMPARISON AND VALIDATION
# ==============================================================================

def compare_structures(analysis1: StructuralAnalysis,
                      analysis2: StructuralAnalysis) -> Dict[str, Any]:
    """
    Compare two structural analyses and compute similarity metrics.

    Useful for:
    - Comparing original vs reconstructed MIDI
    - Finding similar pieces
    - Validation of generation quality

    Args:
        analysis1: First structural analysis
        analysis2: Second structural analysis

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'form_match': analysis1.form_type == analysis2.form_type,
        'num_sections_diff': abs(len(analysis1.sections) - len(analysis2.sections)),
        'duration_diff': abs(analysis1.total_duration - analysis2.total_duration),
        'repetition_ratio_diff': abs(analysis1.repetition_ratio - analysis2.repetition_ratio),
        'development_ratio_diff': abs(analysis1.development_ratio - analysis2.development_ratio),
        'climax_position_diff': abs(analysis1.climax_position - analysis2.climax_position),
        'overall_similarity': 0.0
    }

    # Calculate overall similarity score
    scores = []

    # Form type match
    scores.append(1.0 if comparison['form_match'] else 0.0)

    # Section count similarity
    max_sections = max(len(analysis1.sections), len(analysis2.sections))
    if max_sections > 0:
        section_similarity = 1.0 - (comparison['num_sections_diff'] / max_sections)
        scores.append(section_similarity)

    # Duration similarity
    max_duration = max(analysis1.total_duration, analysis2.total_duration)
    if max_duration > 0:
        duration_similarity = 1.0 - min(1.0, comparison['duration_diff'] / max_duration)
        scores.append(duration_similarity)

    # Metric similarities
    scores.append(1.0 - comparison['repetition_ratio_diff'])
    scores.append(1.0 - comparison['development_ratio_diff'])
    scores.append(1.0 - comparison['climax_position_diff'])

    comparison['overall_similarity'] = np.mean(scores)

    return comparison


def validate_reconstruction(original_midi: str,
                           reconstructed_midi: str,
                           threshold: float = 0.7) -> Dict[str, Any]:
    """
    Validate a reconstructed MIDI against the original.

    Args:
        original_midi: Path to original MIDI
        reconstructed_midi: Path to reconstructed MIDI
        threshold: Similarity threshold for validation (0.0-1.0)

    Returns:
        Validation results with pass/fail
    """
    # Analyze both files
    original_analysis = analyze_midi_structure(original_midi)
    reconstructed_analysis = analyze_midi_structure(reconstructed_midi)

    # Compare structures
    comparison = compare_structures(original_analysis, reconstructed_analysis)

    # Validation results
    results = {
        'passed': comparison['overall_similarity'] >= threshold,
        'similarity_score': comparison['overall_similarity'],
        'threshold': threshold,
        'comparison': comparison,
        'original_form': original_analysis.form_type.value,
        'reconstructed_form': reconstructed_analysis.form_type.value,
        'issues': []
    }

    # Identify specific issues
    if not comparison['form_match']:
        results['issues'].append(f"Form mismatch: {original_analysis.form_type.value} vs {reconstructed_analysis.form_type.value}")

    if comparison['num_sections_diff'] > 2:
        results['issues'].append(f"Significant section count difference: {comparison['num_sections_diff']}")

    if comparison['climax_position_diff'] > 0.2:
        results['issues'].append(f"Climax position mismatch: {comparison['climax_position_diff']:.2f}")

    return results


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python structure_specialist.py <midi_file>")
        sys.exit(1)

    midi_path = sys.argv[1]

    # Analyze structure
    print(f"Analyzing: {midi_path}")
    print()

    analysis = analyze_midi_structure(midi_path)
    print_structure_report(analysis)

    # Extract parameters
    print("\nEXTRACTED PARAMETERS:")
    print("-" * 80)
    params = extract_structure_features(midi_path)
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}")
