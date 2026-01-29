"""
Step 2: Round-Trip Test
=======================

Verifies that encoding and decoding are consistent.

Flow:
1. Load original MIDI file
2. Extract notes (pitch, onset, duration, velocity)
3. Run through encoder -> get transform-relative encoding
4. Run through decoder -> get reconstructed notes
5. Compare original notes vs reconstructed notes

Metrics:
- Same number of notes
- Same pitch (exact)
- Same onset time (within quantization tolerance)
- Same duration (within tolerance)
- Same velocity (within quantization bins)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import json
import glob

from .decoder import ValidationDecoder, DecodedNote, DecodingResult


@dataclass
class NoteComparison:
    """Result of comparing a single note."""
    original: Tuple[int, int, int, int]  # pitch, onset, duration, velocity
    reconstructed: Optional[Tuple[int, int, int, int]]
    pitch_match: bool = False
    onset_match: bool = False
    duration_match: bool = False
    velocity_match: bool = False

    @property
    def full_match(self) -> bool:
        return self.pitch_match and self.onset_match and self.duration_match and self.velocity_match


@dataclass
class FileResult:
    """Round-trip result for a single file."""
    file_path: str
    piece_id: str
    n_original_notes: int
    n_reconstructed_notes: int
    n_matched: int
    pitch_accuracy: float
    onset_accuracy: float
    duration_accuracy: float
    velocity_accuracy: float
    overall_accuracy: float
    error_message: Optional[str] = None

    def __repr__(self):
        return (f"FileResult({self.piece_id}: {self.n_original_notes} notes, "
                f"{self.n_reconstructed_notes} reconstructed, "
                f"{self.overall_accuracy:.1%} match)")


@dataclass
class RoundTripResult:
    """Complete round-trip test result."""
    n_files_tested: int
    n_files_passed: int
    n_files_failed: int
    file_results: List[FileResult]
    aggregate_stats: Dict[str, float]
    failure_modes: Dict[str, int]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "ROUND-TRIP TEST RESULTS",
            "=" * 60,
            f"Files tested: {self.n_files_tested}",
            f"Files passed: {self.n_files_passed} ({self.n_files_passed/max(1,self.n_files_tested):.1%})",
            f"Files failed: {self.n_files_failed}",
            "",
            "Aggregate metrics:",
            f"  Pitch accuracy:    {self.aggregate_stats.get('pitch_accuracy', 0):.1%}",
            f"  Onset accuracy:    {self.aggregate_stats.get('onset_accuracy', 0):.1%}",
            f"  Duration accuracy: {self.aggregate_stats.get('duration_accuracy', 0):.1%}",
            f"  Velocity accuracy: {self.aggregate_stats.get('velocity_accuracy', 0):.1%}",
            f"  Overall accuracy:  {self.aggregate_stats.get('overall_accuracy', 0):.1%}",
        ]

        if self.failure_modes:
            lines.extend(["", "Failure modes:"])
            for mode, count in sorted(self.failure_modes.items(), key=lambda x: -x[1]):
                lines.append(f"  {mode}: {count}")

        return '\n'.join(lines)


class RoundTripTest:
    """
    Tests round-trip consistency of encoding/decoding.

    This test verifies that:
    1. All notes from the original are present in reconstruction
    2. Pitches match exactly (accounting for D24 transforms)
    3. Timing matches within quantization tolerance
    """

    def __init__(
        self,
        onset_tolerance_ticks: int = 60,    # ~1/8 of 16th note at 480 tpb
        duration_tolerance_ticks: int = 60,
        velocity_tolerance: int = 8,        # Within 8 velocity units
        pitch_tolerance: int = 0,           # Pitch must match exactly
    ):
        self.onset_tolerance = onset_tolerance_ticks
        self.duration_tolerance = duration_tolerance_ticks
        self.velocity_tolerance = velocity_tolerance
        self.pitch_tolerance = pitch_tolerance

        self.decoder = ValidationDecoder()

    def test_file(
        self,
        midi_path: str,
        encoded_data: Dict,
        canonical_note_data: Optional[Dict[int, np.ndarray]] = None,
    ) -> FileResult:
        """
        Test round-trip for a single MIDI file.

        Args:
            midi_path: Path to original MIDI file
            encoded_data: Dict with 'canonicals' and 'tokens' from encoding
            canonical_note_data: Optional full note data for canonicals

        Returns:
            FileResult with comparison metrics
        """
        piece_id = Path(midi_path).stem

        try:
            # Load original notes
            original_notes = self._extract_notes_from_midi(midi_path)

            if not original_notes:
                return FileResult(
                    file_path=midi_path,
                    piece_id=piece_id,
                    n_original_notes=0,
                    n_reconstructed_notes=0,
                    n_matched=0,
                    pitch_accuracy=0.0,
                    onset_accuracy=0.0,
                    duration_accuracy=0.0,
                    velocity_accuracy=0.0,
                    overall_accuracy=0.0,
                    error_message="No notes extracted from MIDI",
                )

            # Decode the encoding
            decoding = self.decoder.decode(
                canonicals=encoded_data.get('canonicals', []),
                tokens_by_track=encoded_data.get('tokens', []),
                canonical_note_data=canonical_note_data,
            )

            # Compare notes
            return self._compare_notes(
                original_notes=original_notes,
                reconstructed_notes=decoding.notes,
                midi_path=midi_path,
                piece_id=piece_id,
            )

        except Exception as e:
            return FileResult(
                file_path=midi_path,
                piece_id=piece_id,
                n_original_notes=0,
                n_reconstructed_notes=0,
                n_matched=0,
                pitch_accuracy=0.0,
                onset_accuracy=0.0,
                duration_accuracy=0.0,
                velocity_accuracy=0.0,
                overall_accuracy=0.0,
                error_message=str(e),
            )

    def test_from_checkpoint(
        self,
        checkpoint_path: str,
        corpus_path: str,
        max_files: Optional[int] = None,
    ) -> RoundTripResult:
        """
        Run round-trip test using checkpoint and original corpus.

        NOTE: This is a LIMITED test because the checkpoint doesn't store
        per-piece encodings - it stores a combined encoding. We can only
        verify that decoding produces valid notes, not exact reconstruction.

        For true round-trip testing, we need to re-encode each file individually.

        Args:
            checkpoint_path: Path to checkpoint .npz file
            corpus_path: Path to original MIDI corpus
            max_files: Maximum files to test (None = all)

        Returns:
            RoundTripResult
        """
        # Load checkpoint
        ckpt = np.load(checkpoint_path, allow_pickle=True)

        canonicals = json.loads(str(ckpt['canonical_patterns_json'][0]))
        tokens = json.loads(str(ckpt['encoding_tokens_json'][0]))

        # Find MIDI files
        midi_files = sorted(glob.glob(str(Path(corpus_path) / "*.mid")))
        midi_files += sorted(glob.glob(str(Path(corpus_path) / "*.midi")))

        if max_files:
            midi_files = midi_files[:max_files]

        # Since checkpoint doesn't have per-file encodings,
        # we'll test what we can: decode the checkpoint and report statistics
        print(f"Testing {len(midi_files)} files against checkpoint encoding...")
        print("NOTE: Full round-trip requires re-encoding each file individually.")

        # Decode the checkpoint
        decoding = self.decoder.decode(
            canonicals=canonicals,
            tokens_by_track=tokens,
        )

        # Report basic statistics
        file_results = []
        failure_modes = {}

        for midi_path in midi_files:
            piece_id = Path(midi_path).stem
            try:
                original_notes = self._extract_notes_from_midi(midi_path)
                result = FileResult(
                    file_path=midi_path,
                    piece_id=piece_id,
                    n_original_notes=len(original_notes),
                    n_reconstructed_notes=0,  # Can't match without per-file encoding
                    n_matched=0,
                    pitch_accuracy=0.0,
                    onset_accuracy=0.0,
                    duration_accuracy=0.0,
                    velocity_accuracy=0.0,
                    overall_accuracy=0.0,
                    error_message="Per-file encoding not available in checkpoint",
                )
            except Exception as e:
                result = FileResult(
                    file_path=midi_path,
                    piece_id=piece_id,
                    n_original_notes=0,
                    n_reconstructed_notes=0,
                    n_matched=0,
                    pitch_accuracy=0.0,
                    onset_accuracy=0.0,
                    duration_accuracy=0.0,
                    velocity_accuracy=0.0,
                    overall_accuracy=0.0,
                    error_message=str(e),
                )
                failure_modes['load_error'] = failure_modes.get('load_error', 0) + 1

            file_results.append(result)

        # Report overall statistics from decoding
        print(f"\nCheckpoint decoding produced {decoding.stats['n_notes']} notes")
        print(f"from {decoding.stats['n_patterns']} patterns")
        print(f"across {decoding.stats['n_tracks']} tracks")

        return RoundTripResult(
            n_files_tested=len(midi_files),
            n_files_passed=0,  # Can't determine without full round-trip
            n_files_failed=len(midi_files),
            file_results=file_results,
            aggregate_stats={
                'pitch_accuracy': 0.0,
                'onset_accuracy': 0.0,
                'duration_accuracy': 0.0,
                'velocity_accuracy': 0.0,
                'overall_accuracy': 0.0,
                'decoded_notes': decoding.stats['n_notes'],
                'decoded_patterns': decoding.stats['n_patterns'],
            },
            failure_modes={'no_per_file_encoding': len(midi_files)},
        )

    def _extract_notes_from_midi(self, midi_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Extract notes from MIDI file.

        Returns list of (pitch, onset, duration, velocity) tuples.
        """
        import mido

        midi = mido.MidiFile(midi_path)
        ticks_per_beat = midi.ticks_per_beat

        notes = []
        active_notes = {}  # (track, channel, pitch) -> (onset, velocity)

        for track_idx, track in enumerate(midi.tracks):
            current_time = 0
            for msg in track:
                current_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    key = (track_idx, msg.channel, msg.note)
                    active_notes[key] = (current_time, msg.velocity)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (track_idx, msg.channel, msg.note)
                    if key in active_notes:
                        onset, velocity = active_notes[key]
                        duration = current_time - onset
                        notes.append((msg.note, onset, duration, velocity))
                        del active_notes[key]

        # Handle any still-active notes
        for key, (onset, velocity) in active_notes.items():
            track_idx, channel, pitch = key
            notes.append((pitch, onset, ticks_per_beat * 4, velocity))  # Default 1 bar duration

        return sorted(notes, key=lambda x: (x[1], x[0]))  # Sort by onset, then pitch

    def _compare_notes(
        self,
        original_notes: List[Tuple[int, int, int, int]],
        reconstructed_notes: List[DecodedNote],
        midi_path: str,
        piece_id: str,
    ) -> FileResult:
        """Compare original and reconstructed notes."""
        n_original = len(original_notes)
        n_reconstructed = len(reconstructed_notes)

        if n_original == 0:
            return FileResult(
                file_path=midi_path,
                piece_id=piece_id,
                n_original_notes=0,
                n_reconstructed_notes=n_reconstructed,
                n_matched=0,
                pitch_accuracy=0.0,
                onset_accuracy=0.0,
                duration_accuracy=0.0,
                velocity_accuracy=0.0,
                overall_accuracy=0.0,
            )

        # Convert reconstructed to tuples for comparison
        reconstructed_tuples = [n.to_tuple() for n in reconstructed_notes]

        # Match notes using a greedy algorithm
        # For each original note, find closest matching reconstructed note
        matched_original = set()
        matched_reconstructed = set()

        pitch_matches = 0
        onset_matches = 0
        duration_matches = 0
        velocity_matches = 0

        for i, orig in enumerate(original_notes):
            orig_pitch, orig_onset, orig_dur, orig_vel = orig

            best_match = None
            best_score = float('inf')

            for j, recon in enumerate(reconstructed_tuples):
                if j in matched_reconstructed:
                    continue

                recon_pitch, recon_onset, recon_dur, recon_vel = recon

                # Must match pitch exactly
                if abs(recon_pitch - orig_pitch) > self.pitch_tolerance:
                    continue

                # Score based on onset proximity
                onset_diff = abs(recon_onset - orig_onset)
                if onset_diff > self.onset_tolerance * 10:  # Loose initial filter
                    continue

                if onset_diff < best_score:
                    best_score = onset_diff
                    best_match = j

            if best_match is not None:
                matched_original.add(i)
                matched_reconstructed.add(best_match)

                recon = reconstructed_tuples[best_match]
                recon_pitch, recon_onset, recon_dur, recon_vel = recon

                # Check individual matches
                if abs(recon_pitch - orig_pitch) <= self.pitch_tolerance:
                    pitch_matches += 1
                if abs(recon_onset - orig_onset) <= self.onset_tolerance:
                    onset_matches += 1
                if abs(recon_dur - orig_dur) <= self.duration_tolerance:
                    duration_matches += 1
                if abs(recon_vel - orig_vel) <= self.velocity_tolerance:
                    velocity_matches += 1

        n_matched = len(matched_original)

        return FileResult(
            file_path=midi_path,
            piece_id=piece_id,
            n_original_notes=n_original,
            n_reconstructed_notes=n_reconstructed,
            n_matched=n_matched,
            pitch_accuracy=pitch_matches / max(1, n_original),
            onset_accuracy=onset_matches / max(1, n_original),
            duration_accuracy=duration_matches / max(1, n_original),
            velocity_accuracy=velocity_matches / max(1, n_original),
            overall_accuracy=n_matched / max(1, n_original),
        )

    def _count_token_types(self, tokens_by_track: List[List[Dict]]) -> Dict[str, int]:
        """Count token types."""
        counts = {}
        for track in tokens_by_track:
            for token in track:
                t = token.get('type', 'UNKNOWN')
                counts[t] = counts.get(t, 0) + 1
        return counts


def run_round_trip_test(
    checkpoint_path: str,
    corpus_path: str,
    max_files: Optional[int] = 10,
) -> RoundTripResult:
    """
    Convenience function to run round-trip test.

    Args:
        checkpoint_path: Path to checkpoint file
        corpus_path: Path to MIDI corpus
        max_files: Max files to test (default 10 for quick sanity check)

    Returns:
        RoundTripResult
    """
    tester = RoundTripTest()
    return tester.test_from_checkpoint(checkpoint_path, corpus_path, max_files)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python round_trip.py <checkpoint_path> <corpus_path> [max_files]")
        sys.exit(1)

    checkpoint = sys.argv[1]
    corpus = sys.argv[2]
    max_files = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    result = run_round_trip_test(checkpoint, corpus, max_files)
    print(result.summary())
