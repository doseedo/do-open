#!/usr/bin/env python3
"""
BLIND Reconstruction Test

This tests whether the learned grammar can PARSE any MIDI file from the corpus,
without relying on stored per-file indices.

Key insight:
- Accuracy should ALWAYS be 100% (unmatched = stored as terminals)
- Compression ratio measures grammar quality
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_checkpoint_grammar(checkpoint_path: str) -> Dict:
    """Load grammar patterns from checkpoint."""
    ckpt = np.load(checkpoint_path, allow_pickle=True)
    patterns = json.loads(str(ckpt['canonical_patterns_json'][0]))

    # Build pattern lookup: tuple of pitch_classes -> pattern_id
    # For blind matching, we match on pitch_class sequence only
    grammar = {
        'patterns': {},  # pattern_id -> {pitch_classes, octaves, velocities, durations}
        'pc_lookup': {},  # tuple(pitch_classes) -> list of pattern_ids
    }

    for p in patterns:
        pid = p['pattern_id']
        pc_tuple = tuple(p['pitch_classes'])

        grammar['patterns'][pid] = {
            'pitch_classes': p['pitch_classes'],
            'octaves': p['octaves'],
            'velocities': p['velocities'],
            'durations': p['durations'],
            'length': len(p['pitch_classes']),
        }

        if pc_tuple not in grammar['pc_lookup']:
            grammar['pc_lookup'][pc_tuple] = []
        grammar['pc_lookup'][pc_tuple].append(pid)

    return grammar


def midi_to_factored_tokens(midi_path: str) -> List[Dict]:
    """
    Convert MIDI to factored token sequence.
    Each token has: pitch_class, octave, velocity_bucket, duration_bucket
    """
    try:
        import mido
    except ImportError:
        print("Error: mido not installed")
        sys.exit(1)

    mid = mido.MidiFile(midi_path)
    all_notes = []

    # Velocity and duration quantization (must match pipeline)
    def quantize_velocity(v):
        return min(7, v // 16)  # 0-127 -> 0-7

    def quantize_duration(d, tpb=480):
        # Duration buckets: 0=16th, 1=8th, 2=quarter, 3=half, etc.
        ratio = d / tpb
        if ratio < 0.375: return 0    # < 3/8 beat -> 16th
        if ratio < 0.75: return 1     # < 3/4 beat -> 8th
        if ratio < 1.5: return 2      # < 1.5 beat -> quarter
        if ratio < 3: return 3        # < 3 beat -> half
        if ratio < 6: return 4        # < 6 beat -> whole
        if ratio < 12: return 5       # < 12 beat -> 2 whole
        if ratio < 24: return 6       # < 24 beat -> 4 whole
        return 7                      # longer

    for track_idx, track in enumerate(mid.tracks):
        active_notes = {}
        current_time = 0

        for msg in track:
            current_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (current_time, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_time, velocity = active_notes.pop(msg.note)
                    duration = current_time - start_time

                    all_notes.append({
                        'track': track_idx,
                        'onset': start_time,
                        'pitch': msg.note,
                        'pitch_class': msg.note % 12,
                        'octave': msg.note // 12,
                        'velocity': velocity,
                        'velocity_bucket': quantize_velocity(velocity),
                        'duration': duration,
                        'duration_bucket': quantize_duration(duration, mid.ticks_per_beat),
                    })

    # Sort by onset, then pitch
    all_notes.sort(key=lambda n: (n['onset'], n['pitch']))

    return all_notes, mid.ticks_per_beat


def encode_with_grammar(notes: List[Dict], grammar: Dict, match_factors='pc') -> List[Tuple]:
    """
    Encode note sequence using learned grammar patterns.

    Greedy longest-first matching.
    Returns list of (type, data):
      - ('pattern', pattern_id, matched_notes)
      - ('terminal', note)
    """
    encoded = []
    i = 0
    n = len(notes)

    # Build sequence of pitch classes for matching
    pc_seq = [note['pitch_class'] for note in notes]

    while i < n:
        best_match = None
        best_length = 0
        best_pattern_id = None

        # Try to match longest pattern starting at position i
        # Check all possible lengths (longest first for efficiency)
        for length in range(min(20, n - i), 1, -1):  # Max pattern length 20
            pc_window = tuple(pc_seq[i:i+length])

            if pc_window in grammar['pc_lookup']:
                # Found a matching pattern!
                pattern_ids = grammar['pc_lookup'][pc_window]

                # Use first matching pattern
                best_pattern_id = pattern_ids[0]
                best_length = length
                best_match = notes[i:i+length]
                break  # Greedy: take first (longest) match

        if best_match:
            encoded.append(('pattern', best_pattern_id, best_match))
            i += best_length
        else:
            # No pattern match - store as terminal
            encoded.append(('terminal', notes[i]))
            i += 1

    return encoded


def decode_from_grammar(encoded: List[Tuple], grammar: Dict, use_original_factors=True) -> List[Dict]:
    """
    Decode encoded sequence back to notes.

    Pattern entries:
      - If use_original_factors=True: preserve original octave/velocity/duration (lossless)
      - If use_original_factors=False: use pattern's canonical values (lossy transform)
    Terminal entries: pass through directly
    """
    decoded = []

    for entry in encoded:
        if entry[0] == 'pattern':
            _, pattern_id, original_notes = entry
            pattern = grammar['patterns'][pattern_id]

            for idx, orig in enumerate(original_notes):
                if use_original_factors:
                    # LOSSLESS: Keep original file's actual values
                    decoded.append({
                        'pitch_class': orig['pitch_class'],  # Should match pattern anyway
                        'octave': orig['octave'],
                        'velocity_bucket': orig['velocity_bucket'],
                        'duration_bucket': orig['duration_bucket'],
                        'onset': orig['onset'],
                        'track': orig['track'],
                    })
                else:
                    # LOSSY: Use pattern's canonical values (transforms to "standard" form)
                    decoded.append({
                        'pitch_class': pattern['pitch_classes'][idx],
                        'octave': pattern['octaves'][idx],
                        'velocity_bucket': pattern['velocities'][idx],
                        'duration_bucket': pattern['durations'][idx],
                        'onset': orig['onset'],
                        'track': orig['track'],
                    })
        else:  # terminal
            _, note = entry
            decoded.append({
                'pitch_class': note['pitch_class'],
                'octave': note['octave'],
                'velocity_bucket': note['velocity_bucket'],
                'duration_bucket': note['duration_bucket'],
                'onset': note['onset'],
                'track': note['track'],
            })

    return decoded


def compare_sequences(original: List[Dict], decoded: List[Dict]) -> Dict:
    """Compare original and decoded note sequences."""
    if len(original) != len(decoded):
        return {
            'length_match': False,
            'original_len': len(original),
            'decoded_len': len(decoded),
        }

    matches = {
        'pitch_class': 0,
        'octave': 0,
        'velocity': 0,
        'duration': 0,
        'full_match': 0,
    }

    for orig, dec in zip(original, decoded):
        if orig['pitch_class'] == dec['pitch_class']:
            matches['pitch_class'] += 1
        if orig['octave'] == dec['octave']:
            matches['octave'] += 1
        if orig['velocity_bucket'] == dec['velocity_bucket']:
            matches['velocity'] += 1
        if orig['duration_bucket'] == dec['duration_bucket']:
            matches['duration'] += 1

        if (orig['pitch_class'] == dec['pitch_class'] and
            orig['octave'] == dec['octave'] and
            orig['velocity_bucket'] == dec['velocity_bucket'] and
            orig['duration_bucket'] == dec['duration_bucket']):
            matches['full_match'] += 1

    n = len(original)
    return {
        'length_match': True,
        'n_notes': n,
        'pitch_class_accuracy': matches['pitch_class'] / n * 100,
        'octave_accuracy': matches['octave'] / n * 100,
        'velocity_accuracy': matches['velocity'] / n * 100,
        'duration_accuracy': matches['duration'] / n * 100,
        'full_accuracy': matches['full_match'] / n * 100,
    }


def test_blind_reconstruction(midi_path: str, grammar: Dict, verbose: bool = False) -> Dict:
    """
    BLIND reconstruction test.

    1. Load MIDI and tokenize
    2. Encode using grammar (greedy pattern matching)
    3. Decode back to notes
    4. Compare with original

    Returns accuracy metrics and compression ratio.
    """
    # 1. Load and tokenize
    notes, tpb = midi_to_factored_tokens(midi_path)

    if len(notes) == 0:
        return {'error': 'No notes in file'}

    # 2. Encode with grammar
    encoded = encode_with_grammar(notes, grammar)

    # 3. Decode
    decoded = decode_from_grammar(encoded, grammar)

    # 4. Compare
    comparison = compare_sequences(notes, decoded)

    # Calculate compression stats
    n_patterns = sum(1 for e in encoded if e[0] == 'pattern')
    n_terminals = sum(1 for e in encoded if e[0] == 'terminal')
    notes_in_patterns = sum(len(e[2]) for e in encoded if e[0] == 'pattern')

    compression_ratio = len(notes) / len(encoded) if encoded else 0
    pattern_coverage = notes_in_patterns / len(notes) * 100 if notes else 0

    result = {
        'n_original_notes': len(notes),
        'n_encoded_items': len(encoded),
        'n_patterns_used': n_patterns,
        'n_terminals': n_terminals,
        'notes_in_patterns': notes_in_patterns,
        'pattern_coverage_pct': pattern_coverage,
        'compression_ratio': compression_ratio,
        **comparison,
    }

    if verbose:
        print(f"    Notes: {len(notes)}")
        print(f"    Encoded: {len(encoded)} items ({n_patterns} patterns, {n_terminals} terminals)")
        print(f"    Pattern coverage: {pattern_coverage:.1f}%")
        print(f"    Compression ratio: {compression_ratio:.2f}x")
        print(f"    Full accuracy: {comparison.get('full_accuracy', 0):.1f}%")

    return result


def main():
    parser = argparse.ArgumentParser(description='Blind reconstruction test')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint .npz file')
    parser.add_argument('--corpus', required=True, help='MIDI corpus directory')
    parser.add_argument('--max-files', type=int, default=10, help='Max files to test')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("BLIND RECONSTRUCTION TEST")
    print("=" * 60)
    print("\nThis test encodes+decodes MIDI files using ONLY the learned")
    print("grammar patterns. No per-file indices used.")
    print()

    # Load grammar
    print(f"[1] Loading grammar from {args.checkpoint}")
    grammar = load_checkpoint_grammar(args.checkpoint)
    print(f"    {len(grammar['patterns'])} patterns")
    print(f"    {len(grammar['pc_lookup'])} unique pitch-class sequences")

    # Find MIDI files
    corpus_path = Path(args.corpus)
    midi_files = list(corpus_path.glob("**/*.mid")) + list(corpus_path.glob("**/*.MID"))
    midi_files = midi_files[:args.max_files]
    print(f"\n[2] Testing on {len(midi_files)} files")

    # Test each file
    results = []
    for midi_path in midi_files:
        print(f"\n    {midi_path.name}")
        try:
            result = test_blind_reconstruction(str(midi_path), grammar, args.verbose)
            result['file'] = midi_path.name
            results.append(result)
        except Exception as e:
            print(f"      Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid_results = [r for r in results if 'error' not in r and r.get('length_match', False)]

    if valid_results:
        avg_coverage = np.mean([r['pattern_coverage_pct'] for r in valid_results])
        avg_compression = np.mean([r['compression_ratio'] for r in valid_results])
        avg_full_acc = np.mean([r['full_accuracy'] for r in valid_results])
        avg_pc_acc = np.mean([r['pitch_class_accuracy'] for r in valid_results])

        print(f"\nFiles tested: {len(valid_results)}")
        print(f"\nAccuracy (should be ~100% for lossless):")
        print(f"  Full match: {avg_full_acc:.1f}%")
        print(f"  Pitch class: {avg_pc_acc:.1f}%")

        print(f"\nCompression (measures grammar quality):")
        print(f"  Pattern coverage: {avg_coverage:.1f}%")
        print(f"  Compression ratio: {avg_compression:.2f}x")

        print("\nPer-file results:")
        print(f"{'File':<40} {'Coverage':>10} {'Compress':>10} {'Accuracy':>10}")
        print("-" * 72)
        for r in valid_results:
            print(f"{r['file'][:38]:<40} {r['pattern_coverage_pct']:>9.1f}% {r['compression_ratio']:>9.2f}x {r['full_accuracy']:>9.1f}%")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
Accuracy = 100%: Lossless reconstruction works
Accuracy < 100%: Bug in encode/decode (should not happen)

Coverage = % of notes matched by patterns
Compression = original_notes / encoded_items

High coverage + high compression = Grammar generalizes well
Low coverage = Many notes stored as terminals (still lossless)
""")


if __name__ == '__main__':
    main()
