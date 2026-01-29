"""
Step 4: Generalization Evaluation
=================================

Tests whether the vocabulary learned from training can encode unseen files.

Flow:
1. Load training-only checkpoint (grammar, canonicals, transforms)
2. For each test file:
   - Try to encode using existing vocabulary
   - Track: how many patterns matched existing canonicals?
   - Track: how many patterns required new canonicals (OOV)?
   - Track: reconstruction accuracy after encoding

Metrics:
- Coverage: % of test patterns that match training canonicals (via identity or D24 transform)
- OOV rate: % of test patterns with no match (out-of-vocabulary)
- Reconstruction: Can you still decode accurately using only training vocabulary + OOV handling?

What good looks like:
- Coverage > 80%: vocabulary generalizes well
- Coverage 50-80%: some generalization, genre-dependent
- Coverage < 50%: overfitting to training corpus
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from pathlib import Path
import numpy as np
import json


@dataclass
class FileGeneralizationResult:
    """Generalization result for a single test file."""
    file_path: str
    piece_id: str
    n_patterns: int              # Total patterns extracted from file
    n_matched: int               # Patterns matching training vocabulary
    n_oov: int                   # Out-of-vocabulary patterns
    coverage: float              # n_matched / n_patterns
    match_details: List[Dict]    # Details of each match


@dataclass
class GeneralizationResult:
    """Complete generalization evaluation result."""
    n_test_files: int
    n_train_canonicals: int
    file_results: List[FileGeneralizationResult]

    # Aggregate metrics
    total_patterns: int
    total_matched: int
    total_oov: int
    overall_coverage: float
    oov_rate: float

    # Pattern-level stats
    match_type_counts: Dict[str, int]  # identity, transposition, inversion
    most_used_canonicals: List[Tuple[int, int]]  # (canonical_id, match_count)
    never_used_canonicals: List[int]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "GENERALIZATION EVALUATION RESULTS",
            "=" * 60,
            f"Test files: {self.n_test_files}",
            f"Training canonicals: {self.n_train_canonicals}",
            "",
            "Pattern Coverage:",
            f"  Total patterns in test: {self.total_patterns}",
            f"  Matched by training:    {self.total_matched} ({self.overall_coverage:.1%})",
            f"  Out-of-vocabulary:      {self.total_oov} ({self.oov_rate:.1%})",
            "",
            "Match Types:",
        ]

        for mtype, count in sorted(self.match_type_counts.items(), key=lambda x: -x[1]):
            pct = count / max(1, self.total_matched)
            lines.append(f"  {mtype}: {count} ({pct:.1%})")

        lines.extend([
            "",
            f"Canonicals used: {len(self.most_used_canonicals)} / {self.n_train_canonicals}",
            f"Canonicals never used: {len(self.never_used_canonicals)}",
        ])

        if self.most_used_canonicals[:5]:
            lines.append("\nTop 5 most-used canonicals:")
            for cid, count in self.most_used_canonicals[:5]:
                lines.append(f"  Canon_{cid}: {count} matches")

        lines.extend([
            "",
            "Interpretation:",
        ])

        if self.overall_coverage >= 0.8:
            lines.append("  EXCELLENT: Vocabulary generalizes well to unseen data")
        elif self.overall_coverage >= 0.5:
            lines.append("  MODERATE: Some generalization, may be genre-dependent")
        else:
            lines.append("  POOR: Vocabulary may be overfitting to training corpus")

        return '\n'.join(lines)


def apply_d24_transform(pitch_classes: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply D24 transform to pitch classes."""
    pitch_classes = np.asarray(pitch_classes)
    if transform_id < 12:
        return (pitch_classes + transform_id) % 12
    else:
        n = transform_id - 12
        return (n - pitch_classes) % 12


def find_transform(source: np.ndarray, target: np.ndarray) -> Optional[int]:
    """Find D24 transform that maps source to target."""
    if len(source) != len(target):
        return None
    if len(source) == 0:
        return 0

    source = np.asarray(source)
    target = np.asarray(target)

    for t in range(24):
        transformed = apply_d24_transform(source, t)
        if np.array_equal(transformed, target):
            return t
    return None


class GeneralizationEvaluator:
    """
    Evaluates vocabulary generalization on held-out test files.

    Key question: Can the patterns discovered on training files
    also represent patterns in unseen test files?
    """

    def __init__(self, training_checkpoint_path: str):
        """
        Args:
            training_checkpoint_path: Path to checkpoint built from training set only
        """
        self.checkpoint_path = training_checkpoint_path
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load training checkpoint."""
        ckpt = np.load(self.checkpoint_path, allow_pickle=True)

        # Load canonical patterns
        self.canonicals = json.loads(str(ckpt['canonical_patterns_json'][0]))

        # Build lookup structures for fast matching
        self._build_pattern_index()

        # Load grammar stats for pattern extraction
        self.grammar_rules = json.loads(str(ckpt['grammar_rules_json'][0]))
        self.grammar_stats = json.loads(str(ckpt['grammar_stats_json'][0]))

    def _build_pattern_index(self):
        """Build index for fast pattern matching."""
        # Index by pattern length for faster matching
        self.patterns_by_length: Dict[int, List[Tuple[int, np.ndarray]]] = {}

        for i, canon in enumerate(self.canonicals):
            pc = np.array(canon['pitch_classes'])
            length = len(pc)
            if length not in self.patterns_by_length:
                self.patterns_by_length[length] = []
            self.patterns_by_length[length].append((i, pc))

        # Also index by normalized form (sorted pitch class set)
        self.patterns_by_pcset: Dict[tuple, List[int]] = {}
        for i, canon in enumerate(self.canonicals):
            pc = canon['pitch_classes']
            pcset = tuple(sorted(set(pc)))
            if pcset not in self.patterns_by_pcset:
                self.patterns_by_pcset[pcset] = []
            self.patterns_by_pcset[pcset].append(i)

    def evaluate(
        self,
        test_files: List[str],
        verbose: bool = True,
    ) -> GeneralizationResult:
        """
        Evaluate generalization on test files.

        Args:
            test_files: List of paths to test MIDI files
            verbose: Print progress

        Returns:
            GeneralizationResult
        """
        if verbose:
            print(f"\n{'='*60}")
            print("GENERALIZATION EVALUATION")
            print(f"{'='*60}")
            print(f"Training canonicals: {len(self.canonicals)}")
            print(f"Test files: {len(test_files)}")
            print()

        file_results = []
        total_patterns = 0
        total_matched = 0
        total_oov = 0
        match_type_counts = {'identity': 0, 'transposition': 0, 'inversion': 0}
        canonical_usage = {i: 0 for i in range(len(self.canonicals))}

        for file_idx, test_file in enumerate(test_files):
            if verbose and file_idx % 10 == 0:
                print(f"  Processing {file_idx + 1}/{len(test_files)}...")

            result = self._evaluate_file(test_file)
            file_results.append(result)

            total_patterns += result.n_patterns
            total_matched += result.n_matched
            total_oov += result.n_oov

            # Update match type counts and canonical usage
            for match in result.match_details:
                if match['matched']:
                    mtype = match.get('match_type', 'identity')
                    match_type_counts[mtype] = match_type_counts.get(mtype, 0) + 1
                    canonical_usage[match['canonical_id']] += 1

        # Find most/never used canonicals
        used_canonicals = [(cid, count) for cid, count in canonical_usage.items() if count > 0]
        used_canonicals.sort(key=lambda x: -x[1])
        never_used = [cid for cid, count in canonical_usage.items() if count == 0]

        overall_coverage = total_matched / max(1, total_patterns)
        oov_rate = total_oov / max(1, total_patterns)

        result = GeneralizationResult(
            n_test_files=len(test_files),
            n_train_canonicals=len(self.canonicals),
            file_results=file_results,
            total_patterns=total_patterns,
            total_matched=total_matched,
            total_oov=total_oov,
            overall_coverage=overall_coverage,
            oov_rate=oov_rate,
            match_type_counts=match_type_counts,
            most_used_canonicals=used_canonicals,
            never_used_canonicals=never_used,
        )

        if verbose:
            print(result.summary())

        return result

    def _evaluate_file(self, file_path: str) -> FileGeneralizationResult:
        """Evaluate a single test file."""
        piece_id = Path(file_path).stem

        try:
            # Extract patterns from test file
            patterns = self._extract_patterns_from_file(file_path)

            if not patterns:
                return FileGeneralizationResult(
                    file_path=file_path,
                    piece_id=piece_id,
                    n_patterns=0,
                    n_matched=0,
                    n_oov=0,
                    coverage=0.0,
                    match_details=[],
                )

            # Try to match each pattern against training canonicals
            match_details = []
            n_matched = 0
            n_oov = 0

            for pattern in patterns:
                match = self._find_matching_canonical(pattern)
                match_details.append(match)

                if match['matched']:
                    n_matched += 1
                else:
                    n_oov += 1

            coverage = n_matched / len(patterns) if patterns else 0.0

            return FileGeneralizationResult(
                file_path=file_path,
                piece_id=piece_id,
                n_patterns=len(patterns),
                n_matched=n_matched,
                n_oov=n_oov,
                coverage=coverage,
                match_details=match_details,
            )

        except Exception as e:
            return FileGeneralizationResult(
                file_path=file_path,
                piece_id=piece_id,
                n_patterns=0,
                n_matched=0,
                n_oov=0,
                coverage=0.0,
                match_details=[{'error': str(e)}],
            )

    def _extract_patterns_from_file(self, file_path: str) -> List[np.ndarray]:
        """
        Extract pitch-class patterns from a MIDI file.

        Uses same extraction logic as the training pipeline.
        """
        import mido

        midi = mido.MidiFile(file_path)
        ticks_per_beat = midi.ticks_per_beat
        ticks_per_16th = ticks_per_beat // 4

        # Extract notes per track
        notes_by_track = {}

        for track_idx, track in enumerate(midi.tracks):
            current_time = 0
            active_notes = {}

            for msg in track:
                current_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[(msg.channel, msg.note)] = (current_time, msg.velocity)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (msg.channel, msg.note)
                    if key in active_notes:
                        onset, vel = active_notes[key]
                        if track_idx not in notes_by_track:
                            notes_by_track[track_idx] = []
                        notes_by_track[track_idx].append({
                            'pitch': msg.note,
                            'onset': onset,
                            'duration': current_time - onset,
                            'velocity': vel,
                        })
                        del active_notes[key]

        # Convert to pitch-class sequences
        # Split into patterns (e.g., by time gaps or fixed windows)
        patterns = []

        for track_idx, notes in notes_by_track.items():
            if not notes:
                continue

            # Sort by onset
            notes = sorted(notes, key=lambda n: n['onset'])

            # Extract patterns using sliding window or gap detection
            pattern_notes = []
            last_onset = -1

            for note in notes:
                # If gap is large, start new pattern
                if last_onset >= 0 and (note['onset'] - last_onset) > ticks_per_beat * 2:
                    if len(pattern_notes) >= 2:
                        pc = np.array([n['pitch'] % 12 for n in pattern_notes])
                        patterns.append(pc)
                    pattern_notes = []

                pattern_notes.append(note)
                last_onset = note['onset']

            # Don't forget the last pattern
            if len(pattern_notes) >= 2:
                pc = np.array([n['pitch'] % 12 for n in pattern_notes])
                patterns.append(pc)

        return patterns

    def _find_matching_canonical(self, pattern: np.ndarray) -> Dict:
        """
        Find a training canonical that matches this pattern via D24 transform.

        Returns dict with:
        - matched: bool
        - canonical_id: int (if matched)
        - transform_id: int (if matched)
        - match_type: 'identity' | 'transposition' | 'inversion'
        """
        length = len(pattern)

        # Only check canonicals of same length
        candidates = self.patterns_by_length.get(length, [])

        for canonical_id, canonical_pc in candidates:
            transform = find_transform(canonical_pc, pattern)
            if transform is not None:
                # Determine match type
                if transform == 0:
                    match_type = 'identity'
                elif transform < 12:
                    match_type = 'transposition'
                else:
                    match_type = 'inversion'

                return {
                    'matched': True,
                    'canonical_id': canonical_id,
                    'transform_id': transform,
                    'match_type': match_type,
                    'pattern_length': length,
                }

        return {
            'matched': False,
            'canonical_id': -1,
            'transform_id': -1,
            'match_type': None,
            'pattern_length': length,
        }


def evaluate_generalization(
    training_checkpoint: str,
    test_files: List[str],
    verbose: bool = True,
) -> GeneralizationResult:
    """
    Convenience function to evaluate generalization.

    Args:
        training_checkpoint: Path to training-only checkpoint
        test_files: List of test file paths
        verbose: Print progress

    Returns:
        GeneralizationResult
    """
    evaluator = GeneralizationEvaluator(training_checkpoint)
    return evaluator.evaluate(test_files, verbose)


if __name__ == '__main__':
    import sys
    import glob

    if len(sys.argv) < 3:
        print("Usage: python generalization.py <training_checkpoint> <test_corpus_path>")
        sys.exit(1)

    checkpoint = sys.argv[1]
    test_path = sys.argv[2]

    # Find test files
    test_files = sorted(glob.glob(str(Path(test_path) / "*.mid")))
    test_files += sorted(glob.glob(str(Path(test_path) / "*.midi")))

    if not test_files:
        print(f"No MIDI files found in {test_path}")
        sys.exit(1)

    result = evaluate_generalization(checkpoint, test_files)
    print(result.summary())
