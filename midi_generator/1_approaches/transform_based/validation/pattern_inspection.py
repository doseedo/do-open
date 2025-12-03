"""
Step 5: Pattern Inspection (Musical Meaningfulness)
===================================================

Examines whether the discovered canonical patterns are musically coherent.

Flow:
1. Take the top 20 most-used canonical patterns
2. For each pattern:
   - Convert pitch classes to note names (0=C, 1=C#, 2=D, etc.)
   - Compute intervals between consecutive notes
   - Check: is this a recognizable musical structure?

Questions to answer:
- Is it a chord arpeggio? (C-E-G = major triad)
- Is it a scale fragment? (C-D-E-F-G = major scale)
- Is it a recognizable melodic motif?
- Or is it arbitrary/random-looking?

Also check:
- Pattern lengths: are they musically meaningful durations?
- Transform relationships: when pattern A = T_7(pattern B), does that make musical sense?
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from pathlib import Path
import numpy as np
import json
from collections import Counter


# Musical constants
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Common chord types by interval set (from root)
CHORD_TYPES = {
    frozenset([0, 4, 7]): 'major triad',
    frozenset([0, 3, 7]): 'minor triad',
    frozenset([0, 4, 7, 11]): 'major 7th',
    frozenset([0, 3, 7, 10]): 'minor 7th',
    frozenset([0, 4, 7, 10]): 'dominant 7th',
    frozenset([0, 3, 6]): 'diminished triad',
    frozenset([0, 4, 8]): 'augmented triad',
    frozenset([0, 5, 7]): 'sus4',
    frozenset([0, 2, 7]): 'sus2',
    frozenset([0, 3, 6, 9]): 'diminished 7th',
    frozenset([0, 3, 6, 10]): 'half-diminished',
}

# Scale patterns (intervals from root)
SCALE_PATTERNS = {
    (2, 2, 1, 2, 2, 2, 1): 'major scale',
    (2, 1, 2, 2, 1, 2, 2): 'natural minor',
    (2, 1, 2, 2, 2, 2, 1): 'melodic minor (asc)',
    (2, 1, 2, 2, 1, 3, 1): 'harmonic minor',
    (1, 2, 2, 2, 1, 2, 2): 'phrygian',
    (2, 2, 2, 1, 2, 2, 1): 'lydian',
    (2, 2, 1, 2, 2, 1, 2): 'mixolydian',
    (1, 2, 2, 1, 2, 2, 2): 'locrian',
    (2, 2, 1, 2, 1, 3, 1): 'dorian #4',
    (3, 2, 1, 1, 3, 2): 'blues scale',
    (2, 2, 3, 2, 3): 'pentatonic major',
    (3, 2, 2, 3, 2): 'pentatonic minor',
}

# Transform musical meanings
TRANSFORM_MEANINGS = {
    0: 'identity (exact repeat)',
    1: 'up minor 2nd',
    2: 'up major 2nd',
    3: 'up minor 3rd',
    4: 'up major 3rd',
    5: 'up perfect 4th',
    6: 'up tritone',
    7: 'up perfect 5th',
    8: 'up minor 6th',
    9: 'up major 6th',
    10: 'up minor 7th',
    11: 'up major 7th',
    12: 'inversion (I_0)',
    13: 'inversion (I_1)',
    14: 'inversion (I_2)',
    15: 'inversion (I_3)',
    16: 'inversion (I_4)',
    17: 'inversion (I_5)',
    18: 'inversion (I_6)',
    19: 'inversion (I_7)',
    20: 'inversion (I_8)',
    21: 'inversion (I_9)',
    22: 'inversion (I_10)',
    23: 'inversion (I_11)',
}


@dataclass
class PatternAnalysis:
    """Analysis of a single canonical pattern."""
    pattern_id: int
    pitch_classes: List[int]
    note_names: List[str]
    intervals: List[int]
    usage_count: int

    # Musical interpretations
    is_chord_arpeggio: bool = False
    chord_type: Optional[str] = None
    is_scale_fragment: bool = False
    scale_type: Optional[str] = None
    is_melodic_motif: bool = False
    motif_description: Optional[str] = None

    # Quality metrics
    length: int = 0
    unique_pitch_classes: int = 0
    interval_variety: int = 0
    has_stepwise_motion: bool = False
    has_leaps: bool = False

    # Overall assessment
    musical_coherence: str = 'unknown'  # 'high', 'medium', 'low', 'unknown'
    interpretation: str = ''

    def to_dict(self) -> Dict:
        return {
            'pattern_id': self.pattern_id,
            'pitch_classes': self.pitch_classes,
            'note_names': self.note_names,
            'intervals': self.intervals,
            'usage_count': self.usage_count,
            'chord_type': self.chord_type,
            'scale_type': self.scale_type,
            'motif_description': self.motif_description,
            'musical_coherence': self.musical_coherence,
            'interpretation': self.interpretation,
        }


@dataclass
class TransformAnalysis:
    """Analysis of transform relationships."""
    transform_id: int
    musical_meaning: str
    usage_count: int
    is_musically_common: bool


@dataclass
class InspectionResult:
    """Complete pattern inspection result."""
    n_patterns_analyzed: int
    pattern_analyses: List[PatternAnalysis]
    transform_analyses: List[TransformAnalysis]

    # Aggregate statistics
    coherent_patterns: int
    incoherent_patterns: int
    chord_arpeggios: int
    scale_fragments: int
    melodic_motifs: int

    # Length distribution
    length_distribution: Dict[int, int]

    # Transform distribution
    transform_distribution: Dict[int, int]
    musically_common_transforms: int

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "PATTERN INSPECTION RESULTS",
            "=" * 70,
            f"Patterns analyzed: {self.n_patterns_analyzed}",
            "",
            "Musical Coherence:",
            f"  High coherence: {self.coherent_patterns}",
            f"  Low coherence:  {self.incoherent_patterns}",
            "",
            "Pattern Types:",
            f"  Chord arpeggios:  {self.chord_arpeggios}",
            f"  Scale fragments:  {self.scale_fragments}",
            f"  Melodic motifs:   {self.melodic_motifs}",
            "",
            "Length Distribution:",
        ]

        for length, count in sorted(self.length_distribution.items()):
            lines.append(f"  {length} notes: {count} patterns")

        lines.extend([
            "",
            "Transform Usage (top 5):",
        ])

        sorted_transforms = sorted(self.transform_distribution.items(), key=lambda x: -x[1])[:5]
        for tid, count in sorted_transforms:
            meaning = TRANSFORM_MEANINGS.get(tid, f'unknown ({tid})')
            lines.append(f"  T{tid} ({meaning}): {count} uses")

        lines.extend([
            "",
            f"Musically common transforms (5ths, 4ths, octaves): {self.musically_common_transforms}",
        ])

        return '\n'.join(lines)

    def pattern_table(self) -> str:
        """Generate a table of top patterns with interpretations."""
        lines = [
            "",
            "TOP 20 PATTERNS",
            "-" * 90,
            f"{'ID':>4} | {'Usage':>6} | {'Notes':<30} | {'Type':<15} | {'Interpretation'}",
            "-" * 90,
        ]

        for pa in self.pattern_analyses[:20]:
            notes = ' '.join(pa.note_names[:8])
            if len(pa.note_names) > 8:
                notes += '...'
            ptype = pa.chord_type or pa.scale_type or pa.motif_description or '-'
            lines.append(
                f"{pa.pattern_id:>4} | {pa.usage_count:>6} | {notes:<30} | {ptype:<15} | {pa.interpretation}"
            )

        lines.append("-" * 90)
        return '\n'.join(lines)


class PatternInspector:
    """
    Inspects canonical patterns for musical meaningfulness.
    """

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load checkpoint data."""
        ckpt = np.load(self.checkpoint_path, allow_pickle=True)

        self.canonicals = json.loads(str(ckpt['canonical_patterns_json'][0]))
        self.tokens = json.loads(str(ckpt['encoding_tokens_json'][0]))

        # Count transform usage
        self.transform_counts = Counter()
        for track in self.tokens:
            for token in track:
                if token.get('type') == 'TRANSFORM':
                    self.transform_counts[token.get('transform_id', 0)] += 1

    def inspect(
        self,
        n_patterns: int = 20,
        verbose: bool = True,
    ) -> InspectionResult:
        """
        Inspect top patterns for musical meaningfulness.

        Args:
            n_patterns: Number of top patterns to analyze
            verbose: Print progress

        Returns:
            InspectionResult
        """
        if verbose:
            print(f"\n{'='*60}")
            print("PATTERN INSPECTION")
            print(f"{'='*60}")
            print(f"Total canonicals: {len(self.canonicals)}")
            print(f"Analyzing top {n_patterns} by usage")
            print()

        # Sort by usage count
        sorted_canonicals = sorted(
            enumerate(self.canonicals),
            key=lambda x: x[1].get('usage_count', 0),
            reverse=True
        )

        pattern_analyses = []
        length_dist = Counter()

        for idx, canon in sorted_canonicals[:n_patterns]:
            analysis = self._analyze_pattern(idx, canon)
            pattern_analyses.append(analysis)
            length_dist[analysis.length] += 1

        # Analyze transforms
        transform_analyses = self._analyze_transforms()

        # Compute aggregates
        coherent = sum(1 for pa in pattern_analyses if pa.musical_coherence == 'high')
        incoherent = sum(1 for pa in pattern_analyses if pa.musical_coherence == 'low')
        chord_arps = sum(1 for pa in pattern_analyses if pa.is_chord_arpeggio)
        scale_frags = sum(1 for pa in pattern_analyses if pa.is_scale_fragment)
        melodic = sum(1 for pa in pattern_analyses if pa.is_melodic_motif)

        # Common transforms: 5ths (7), 4ths (5), octaves (0 for identity)
        common_transform_ids = {0, 5, 7, 12}  # identity, 4th, 5th, I_0
        common_count = sum(
            count for tid, count in self.transform_counts.items()
            if tid in common_transform_ids
        )

        result = InspectionResult(
            n_patterns_analyzed=len(pattern_analyses),
            pattern_analyses=pattern_analyses,
            transform_analyses=transform_analyses,
            coherent_patterns=coherent,
            incoherent_patterns=incoherent,
            chord_arpeggios=chord_arps,
            scale_fragments=scale_frags,
            melodic_motifs=melodic,
            length_distribution=dict(length_dist),
            transform_distribution=dict(self.transform_counts),
            musically_common_transforms=common_count,
        )

        if verbose:
            print(result.summary())
            print(result.pattern_table())

        return result

    def _analyze_pattern(self, pattern_id: int, canon: Dict) -> PatternAnalysis:
        """Analyze a single pattern."""
        pc = canon.get('pitch_classes', [])
        usage = canon.get('usage_count', 0)

        # Convert to note names
        note_names = [NOTE_NAMES[p % 12] for p in pc]

        # Compute intervals (semitones between consecutive notes)
        intervals = []
        for i in range(1, len(pc)):
            interval = (pc[i] - pc[i-1]) % 12
            # Normalize to smallest interval
            if interval > 6:
                interval = interval - 12
            intervals.append(interval)

        analysis = PatternAnalysis(
            pattern_id=pattern_id,
            pitch_classes=pc,
            note_names=note_names,
            intervals=intervals,
            usage_count=usage,
            length=len(pc),
            unique_pitch_classes=len(set(pc)),
        )

        # Check for chord arpeggio
        pc_set = frozenset(p % 12 for p in pc)
        # Normalize to root = 0
        root = min(pc_set)
        normalized_set = frozenset((p - root) % 12 for p in pc_set)

        for chord_intervals, chord_name in CHORD_TYPES.items():
            if normalized_set == chord_intervals:
                analysis.is_chord_arpeggio = True
                analysis.chord_type = chord_name
                analysis.musical_coherence = 'high'
                analysis.interpretation = f"{NOTE_NAMES[root]} {chord_name} arpeggio"
                break

        # Check for scale fragment
        if not analysis.is_chord_arpeggio and len(intervals) >= 3:
            interval_tuple = tuple(intervals)
            for scale_intervals, scale_name in SCALE_PATTERNS.items():
                # Check if our intervals are a subset of scale intervals
                if self._is_scale_fragment(interval_tuple, scale_intervals):
                    analysis.is_scale_fragment = True
                    analysis.scale_type = scale_name
                    analysis.musical_coherence = 'high'
                    analysis.interpretation = f"{scale_name} fragment"
                    break

        # Check for melodic characteristics
        if intervals:
            analysis.has_stepwise_motion = any(abs(i) <= 2 for i in intervals)
            analysis.has_leaps = any(abs(i) >= 5 for i in intervals)
            analysis.interval_variety = len(set(intervals))

            # Simple melodic motif detection
            if not analysis.is_chord_arpeggio and not analysis.is_scale_fragment:
                if analysis.has_stepwise_motion and analysis.interval_variety <= 3:
                    analysis.is_melodic_motif = True
                    analysis.motif_description = 'stepwise motif'
                    analysis.musical_coherence = 'medium'
                    analysis.interpretation = 'stepwise melodic motif'
                elif len(pc) <= 4 and analysis.unique_pitch_classes <= 3:
                    analysis.is_melodic_motif = True
                    analysis.motif_description = 'short motif'
                    analysis.musical_coherence = 'medium'
                    analysis.interpretation = 'short melodic cell'

        # Fallback assessment
        if analysis.musical_coherence == 'unknown':
            if analysis.unique_pitch_classes <= 4 and len(pc) <= 6:
                analysis.musical_coherence = 'medium'
                analysis.interpretation = 'compact note group'
            else:
                analysis.musical_coherence = 'low'
                analysis.interpretation = 'arbitrary sequence'

        return analysis

    def _is_scale_fragment(self, intervals: Tuple, scale_intervals: Tuple) -> bool:
        """Check if intervals are a contiguous fragment of scale intervals."""
        if len(intervals) > len(scale_intervals):
            return False

        # Extend scale intervals to cover 2 octaves
        extended = scale_intervals * 2

        for start in range(len(extended) - len(intervals) + 1):
            if extended[start:start + len(intervals)] == intervals:
                return True
        return False

    def _analyze_transforms(self) -> List[TransformAnalysis]:
        """Analyze transform usage patterns."""
        analyses = []

        # Common musical transforms
        musically_common = {
            0,   # identity
            5,   # up 4th
            7,   # up 5th
            12,  # inversion
        }

        for tid, count in sorted(self.transform_counts.items(), key=lambda x: -x[1]):
            analyses.append(TransformAnalysis(
                transform_id=tid,
                musical_meaning=TRANSFORM_MEANINGS.get(tid, f'T{tid}'),
                usage_count=count,
                is_musically_common=tid in musically_common,
            ))

        return analyses


def inspect_patterns(
    checkpoint_path: str,
    n_patterns: int = 20,
    verbose: bool = True,
) -> InspectionResult:
    """
    Convenience function to inspect patterns.

    Args:
        checkpoint_path: Path to checkpoint file
        n_patterns: Number of top patterns to analyze
        verbose: Print results

    Returns:
        InspectionResult
    """
    inspector = PatternInspector(checkpoint_path)
    return inspector.inspect(n_patterns, verbose)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pattern_inspection.py <checkpoint_path> [n_patterns]")
        sys.exit(1)

    checkpoint = sys.argv[1]
    n_patterns = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    result = inspect_patterns(checkpoint, n_patterns)
