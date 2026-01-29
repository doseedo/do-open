"""
Live Round-Trip Validation
==========================

Re-encodes MIDI files using existing grammar from checkpoint,
then decodes back to MIDI for comparison.

This is Option B: encode at test time rather than storing per-file encodings.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import mido


@dataclass
class Note:
    """A MIDI note."""
    pitch: int
    onset: int  # ticks
    duration: int  # ticks
    velocity: int
    track: int = 0
    channel: int = 0


@dataclass
class RoundTripResult:
    """Result of round-trip test."""
    original_notes: List[Note]
    reconstructed_notes: List[Note]

    # Metrics
    pitch_accuracy: float
    onset_accuracy: float
    duration_accuracy: float
    overall_match_rate: float

    # Encoding info
    n_patterns_used: int
    n_canonicals_matched: int
    encoding_tokens: List[Dict]


def load_midi_notes(midi_path: str) -> Tuple[List[Note], int]:
    """
    Load notes from MIDI file.

    Returns:
        (list of Notes, ticks_per_beat)
    """
    midi = mido.MidiFile(midi_path)
    notes = []

    for track_idx, track in enumerate(midi.tracks):
        current_time = 0
        active = {}  # (channel, pitch) -> (onset, velocity)

        for msg in track:
            current_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                key = (msg.channel, msg.note)
                active[key] = (current_time, msg.velocity)

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    onset, velocity = active[key]
                    duration = current_time - onset
                    notes.append(Note(
                        pitch=msg.note,
                        onset=onset,
                        duration=duration,
                        velocity=velocity,
                        track=track_idx,
                        channel=msg.channel,
                    ))
                    del active[key]

        # Close any still-active notes
        for (channel, pitch), (onset, velocity) in active.items():
            notes.append(Note(
                pitch=pitch,
                onset=onset,
                duration=midi.ticks_per_beat * 4,  # Default to 1 bar
                velocity=velocity,
                track=track_idx,
                channel=channel,
            ))

    return sorted(notes, key=lambda n: (n.onset, n.pitch)), midi.ticks_per_beat


def extract_pitch_class_sequence(notes: List[Note]) -> np.ndarray:
    """Extract pitch class sequence from notes."""
    return np.array([n.pitch % 12 for n in notes], dtype=np.int8)


def find_pattern_matches(
    sequence: np.ndarray,
    canonicals: List[Dict],
) -> List[Tuple[int, int, int, int]]:
    """
    Find where canonical patterns match in the sequence.

    Uses D24 transform matching.

    Returns:
        List of (start_pos, end_pos, canonical_id, transform_id)
    """
    matches = []

    # Sort canonicals by length (longer first for greedy matching)
    indexed_canonicals = [(i, np.array(c['pitch_classes'], dtype=np.int8))
                          for i, c in enumerate(canonicals)]
    indexed_canonicals.sort(key=lambda x: -len(x[1]))

    pos = 0
    while pos < len(sequence):
        best_match = None
        best_length = 0

        for canon_id, canon_pc in indexed_canonicals:
            length = len(canon_pc)
            if pos + length > len(sequence):
                continue

            target = sequence[pos:pos + length]

            # Try all 24 D24 transforms
            for transform_id in range(24):
                if transform_id < 12:
                    transformed = (canon_pc + transform_id) % 12
                else:
                    n = transform_id - 12
                    transformed = (n - canon_pc) % 12

                if np.array_equal(transformed, target):
                    if length > best_length:
                        best_match = (pos, pos + length, canon_id, transform_id)
                        best_length = length
                    break

        if best_match:
            matches.append(best_match)
            pos = best_match[1]  # Move past the match
        else:
            pos += 1  # No match, advance by 1

    return matches


def encode_with_grammar(
    notes: List[Note],
    canonicals: List[Dict],
) -> Tuple[List[Dict], List[Note]]:
    """
    Encode notes using existing canonical patterns.

    Returns:
        (encoding_tokens, covered_notes)
    """
    if not notes:
        return [], []

    # Extract pitch classes
    pc_sequence = extract_pitch_class_sequence(notes)

    # Find pattern matches
    matches = find_pattern_matches(pc_sequence, canonicals)

    # Build encoding tokens
    tokens = []
    covered_note_indices = set()
    introduced = {}  # canonical_id -> token_idx

    for start, end, canon_id, transform_id in matches:
        covered_note_indices.update(range(start, end))

        if canon_id not in introduced:
            # First occurrence
            tokens.append({
                'type': 'INTRO',
                'pattern_idx': canon_id,
                'transform_id': 0,
                'note_range': (start, end),
            })
            introduced[canon_id] = len(tokens) - 1
        elif transform_id == 0:
            # Exact repeat
            tokens.append({
                'type': 'REPEAT',
                'pattern_idx': introduced[canon_id],
                'note_range': (start, end),
            })
        else:
            # Transform
            tokens.append({
                'type': 'TRANSFORM',
                'pattern_idx': introduced[canon_id],
                'transform_id': transform_id,
                'note_range': (start, end),
            })

    # Get covered notes
    covered_notes = [notes[i] for i in sorted(covered_note_indices)]

    return tokens, covered_notes


def decode_tokens(
    tokens: List[Dict],
    canonicals: List[Dict],
    base_octave: int = 4,
    default_duration: int = 120,
    default_velocity: int = 100,
) -> List[Note]:
    """
    Decode encoding tokens back to notes.
    """
    notes = []
    current_time = 0
    emitted = {}  # token_idx -> pitch_classes

    for token_idx, token in enumerate(tokens):
        token_type = token.get('type', '')
        pattern_idx = token.get('pattern_idx', -1)
        transform_id = token.get('transform_id', 0)

        midi_pitches = None  # Use full MIDI pitches when available

        if token_type == 'INTRO':
            if 0 <= pattern_idx < len(canonicals):
                pattern = canonicals[pattern_idx]
                # Prefer canonical_pitches (full MIDI) over pitch_classes
                if 'canonical_pitches' in pattern and pattern['canonical_pitches']:
                    midi_pitches = np.array(pattern['canonical_pitches'], dtype=np.int32)
                else:
                    # Fallback to pitch_classes with base octave
                    pitch_classes = np.array(pattern['pitch_classes'], dtype=np.int8)
                    midi_pitches = pitch_classes + (base_octave + 1) * 12
                emitted[token_idx] = midi_pitches.copy()

        elif token_type == 'REPEAT':
            if pattern_idx in emitted:
                midi_pitches = emitted[pattern_idx].copy()
                emitted[token_idx] = midi_pitches

        elif token_type == 'TRANSFORM':
            if pattern_idx in emitted:
                source = emitted[pattern_idx]
                if transform_id < 12:
                    # Transposition: add semitones
                    midi_pitches = source + transform_id
                else:
                    # Inversion: reflect around axis
                    n = transform_id - 12
                    # Inversion around axis n: new_pitch = 2*n - old_pitch (mod 12 for pitch class)
                    # For MIDI pitches, we need to preserve octave info
                    pitch_classes = source % 12
                    octaves = source // 12
                    new_pc = (2 * n - pitch_classes) % 12
                    midi_pitches = new_pc + octaves * 12
                emitted[token_idx] = midi_pitches.copy()

        if midi_pitches is not None:
            for i, pitch in enumerate(midi_pitches):
                notes.append(Note(
                    pitch=int(pitch),
                    onset=current_time + i * default_duration,
                    duration=default_duration,
                    velocity=default_velocity,
                ))
            current_time += len(midi_pitches) * default_duration

    return notes


def compare_notes(
    original: List[Note],
    reconstructed: List[Note],
    pitch_tolerance: int = 0,
    onset_tolerance: int = 60,
    duration_tolerance: int = 120,
) -> Dict[str, float]:
    """
    Compare original and reconstructed notes.
    """
    if not original:
        return {
            'pitch_accuracy': 0.0,
            'onset_accuracy': 0.0,
            'duration_accuracy': 0.0,
            'overall_match_rate': 0.0,
        }

    # Only compare pitch classes (since we don't preserve octave info)
    orig_pc = [n.pitch % 12 for n in original]
    recon_pc = [n.pitch % 12 for n in reconstructed]

    # Count pitch class matches
    matches = 0
    min_len = min(len(orig_pc), len(recon_pc))
    for i in range(min_len):
        if orig_pc[i] == recon_pc[i]:
            matches += 1

    pitch_accuracy = matches / len(orig_pc) if orig_pc else 0.0

    # Overall match rate is sequence coverage
    coverage = min_len / len(original) if original else 0.0

    return {
        'pitch_accuracy': pitch_accuracy,
        'onset_accuracy': 0.0,  # Not comparing timing (no timing info in pc-only encoding)
        'duration_accuracy': 0.0,
        'overall_match_rate': coverage * pitch_accuracy,
        'n_original': len(original),
        'n_reconstructed': len(reconstructed),
        'n_matched_pc': matches,
    }


def run_live_round_trip(
    midi_path: str,
    checkpoint_path: str,
) -> RoundTripResult:
    """
    Run live round-trip test on a MIDI file.

    Args:
        midi_path: Path to MIDI file
        checkpoint_path: Path to checkpoint with grammar

    Returns:
        RoundTripResult
    """
    # Load checkpoint
    ckpt = np.load(checkpoint_path, allow_pickle=True)
    canonicals = json.loads(str(ckpt['canonical_patterns_json'][0]))

    # Load original MIDI
    original_notes, tpb = load_midi_notes(midi_path)

    # Encode using existing grammar
    tokens, covered_notes = encode_with_grammar(original_notes, canonicals)

    # Decode back
    reconstructed_notes = decode_tokens(tokens, canonicals)

    # Compare
    metrics = compare_notes(original_notes, reconstructed_notes)

    return RoundTripResult(
        original_notes=original_notes,
        reconstructed_notes=reconstructed_notes,
        pitch_accuracy=metrics['pitch_accuracy'],
        onset_accuracy=metrics['onset_accuracy'],
        duration_accuracy=metrics['duration_accuracy'],
        overall_match_rate=metrics['overall_match_rate'],
        n_patterns_used=len(tokens),
        n_canonicals_matched=len(set(t.get('pattern_idx') for t in tokens if t.get('type') == 'INTRO')),
        encoding_tokens=tokens,
    )


def save_reconstructed_midi(
    notes: List[Note],
    output_path: str,
    ticks_per_beat: int = 480,
):
    """Save reconstructed notes to MIDI file."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Build events
    events = []
    for note in notes:
        events.append((note.onset, 'on', note.pitch, note.velocity))
        events.append((note.onset + note.duration, 'off', note.pitch, 0))

    events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))

    # Write as delta times
    current_time = 0
    for time, etype, pitch, vel in events:
        delta = time - current_time
        if etype == 'on':
            track.append(mido.Message('note_on', note=pitch, velocity=vel, time=delta))
        else:
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=delta))
        current_time = time

    mid.save(output_path)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python live_round_trip.py <midi_path> <checkpoint_path> [output_midi]")
        sys.exit(1)

    midi_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    result = run_live_round_trip(midi_path, checkpoint_path)

    print(f"\n=== Live Round-Trip Results ===")
    print(f"Original notes: {len(result.original_notes)}")
    print(f"Reconstructed notes: {len(result.reconstructed_notes)}")
    print(f"Patterns used: {result.n_patterns_used}")
    print(f"Canonicals matched: {result.n_canonicals_matched}")
    print(f"\nMetrics:")
    print(f"  Pitch accuracy: {result.pitch_accuracy:.1%}")
    print(f"  Overall match rate: {result.overall_match_rate:.1%}")

    if output_path:
        save_reconstructed_midi(result.reconstructed_notes, output_path)
        print(f"\nReconstructed MIDI saved to: {output_path}")
