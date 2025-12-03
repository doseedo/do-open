#!/usr/bin/env python3
"""
Test reconstruction from checkpoint to verify data integrity.

This script:
1. Loads original MIDI files
2. Loads checkpoint patterns
3. Attempts to reconstruct notes from patterns + occurrences
4. Compares with original to measure accuracy
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load checkpoint and parse JSON fields."""
    ckpt = np.load(checkpoint_path, allow_pickle=True)

    return {
        'patterns': json.loads(str(ckpt['canonical_patterns_json'][0])),
        'grammar_rules': json.loads(str(ckpt['grammar_rules_json'][0])),
        'transforms': json.loads(str(ckpt['transform_vocabulary_json'][0])),
        'n_files': int(ckpt['n_files'][0]),
        'n_tracks': int(ckpt['n_tracks'][0]),
        'n_notes': int(ckpt['n_notes'][0]),
    }


def load_original_midi(midi_path: str) -> Dict[int, List[Dict]]:
    """Load original MIDI file and extract notes per track."""
    try:
        import mido
    except ImportError:
        print("Error: mido not installed. Run: pip install mido")
        sys.exit(1)

    mid = mido.MidiFile(midi_path)
    tracks = {}

    for track_idx, track in enumerate(mid.tracks):
        notes = []
        active_notes = {}  # pitch -> (start_time, velocity)
        current_time = 0

        for msg in track:
            current_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (current_time, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_time, velocity = active_notes.pop(msg.note)
                    notes.append({
                        'pitch': msg.note,
                        'pitch_class': msg.note % 12,
                        'octave': msg.note // 12,
                        'onset': start_time,
                        'duration': current_time - start_time,
                        'velocity': velocity,
                    })

        if notes:
            # Sort by onset time
            notes.sort(key=lambda n: (n['onset'], n['pitch']))
            tracks[track_idx] = notes

    return tracks, mid.ticks_per_beat


def reconstruct_from_patterns(patterns: List[Dict], piece_id: str) -> Dict[int, List[Dict]]:
    """Reconstruct notes from patterns and their occurrences."""
    tracks = defaultdict(list)

    for pattern in patterns:
        p_len = len(pattern['pitch_classes'])

        for occ in pattern['occurrences']:
            if occ['piece_id'] == piece_id:
                track_id = occ['track_id']
                onset_time = occ['onset_time']
                position = occ['position']

                # Reconstruct each note in the pattern
                # Use rhythm_ioi for inter-onset intervals within pattern
                rhythm_ioi = pattern.get('rhythm_ioi', [1] * (p_len - 1))

                current_time = onset_time
                for i in range(p_len):
                    note = {
                        'position': position + i,
                        'pitch_class': pattern['pitch_classes'][i],
                        'octave': pattern['octaves'][i],
                        'velocity_bucket': pattern['velocities'][i],
                        'duration_bucket': pattern['durations'][i],
                        'onset': current_time,
                    }
                    tracks[track_id].append(note)

                    # Advance time by IOI (if not last note)
                    if i < len(rhythm_ioi):
                        # rhythm_ioi is relative, need to convert to ticks
                        # For now, use a default tick step (240 = 16th note at 480 tpb)
                        current_time += rhythm_ioi[i] * 240  # TODO: use actual ticks

    # Sort by position and deduplicate
    for track_id in tracks:
        # Sort by position
        tracks[track_id].sort(key=lambda n: n['position'])

        # Deduplicate (same position = same note)
        seen_positions = set()
        unique_notes = []
        for note in tracks[track_id]:
            if note['position'] not in seen_positions:
                seen_positions.add(note['position'])
                unique_notes.append(note)
        tracks[track_id] = unique_notes

    return dict(tracks)


def compare_tracks(original: Dict[int, List[Dict]],
                   reconstructed: Dict[int, List[Dict]]) -> Dict:
    """Compare original and reconstructed tracks."""
    results = {
        'total_original_notes': 0,
        'total_reconstructed_notes': 0,
        'matched_notes': 0,
        'pitch_class_matches': 0,
        'full_pitch_matches': 0,
        'tracks': {}
    }

    for track_id, orig_notes in original.items():
        recon_notes = reconstructed.get(track_id, [])

        orig_count = len(orig_notes)
        recon_count = len(recon_notes)

        results['total_original_notes'] += orig_count
        results['total_reconstructed_notes'] += recon_count

        # Compare by position index
        matched = 0
        pc_matched = 0
        pitch_matched = 0

        # Build lookup for reconstructed by position
        recon_by_pos = {n['position']: n for n in recon_notes}

        for i, orig_note in enumerate(orig_notes):
            if i in recon_by_pos:
                matched += 1
                recon_note = recon_by_pos[i]

                if orig_note['pitch_class'] == recon_note['pitch_class']:
                    pc_matched += 1

                    if orig_note['octave'] == recon_note['octave']:
                        pitch_matched += 1

        results['matched_notes'] += matched
        results['pitch_class_matches'] += pc_matched
        results['full_pitch_matches'] += pitch_matched

        coverage = matched / orig_count * 100 if orig_count > 0 else 0
        pc_accuracy = pc_matched / matched * 100 if matched > 0 else 0
        pitch_accuracy = pitch_matched / matched * 100 if matched > 0 else 0

        results['tracks'][track_id] = {
            'original_notes': orig_count,
            'reconstructed_notes': recon_count,
            'matched_notes': matched,
            'coverage_pct': coverage,
            'pitch_class_accuracy_pct': pc_accuracy,
            'full_pitch_accuracy_pct': pitch_accuracy,
        }

    # Overall metrics
    if results['total_original_notes'] > 0:
        results['overall_coverage_pct'] = results['matched_notes'] / results['total_original_notes'] * 100
    if results['matched_notes'] > 0:
        results['overall_pc_accuracy_pct'] = results['pitch_class_matches'] / results['matched_notes'] * 100
        results['overall_pitch_accuracy_pct'] = results['full_pitch_matches'] / results['matched_notes'] * 100

    return results


def main():
    parser = argparse.ArgumentParser(description='Test reconstruction from checkpoint')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .npz file')
    parser.add_argument('--corpus', required=True, help='Path to MIDI corpus directory')
    parser.add_argument('--piece', help='Specific piece name to test (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print("=" * 60)
    print("RECONSTRUCTION TEST")
    print("=" * 60)

    # Load checkpoint
    print(f"\n[1] Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)
    print(f"    Patterns: {len(ckpt['patterns'])}")
    print(f"    Files: {ckpt['n_files']}, Tracks: {ckpt['n_tracks']}, Notes: {ckpt['n_notes']}")

    # Get unique piece IDs from patterns
    piece_ids = set()
    for p in ckpt['patterns']:
        for occ in p['occurrences']:
            piece_ids.add(occ['piece_id'])
    print(f"    Unique pieces in checkpoint: {len(piece_ids)}")

    # Find corresponding MIDI files
    corpus_path = Path(args.corpus)
    midi_files = list(corpus_path.glob("**/*.mid")) + list(corpus_path.glob("**/*.MID"))
    print(f"\n[2] Found {len(midi_files)} MIDI files in corpus")

    # Match piece IDs to files
    piece_to_file = {}
    for midi_file in midi_files:
        # Try to match by name (piece_id is typically filename without extension)
        stem = midi_file.stem
        if stem in piece_ids:
            piece_to_file[stem] = midi_file
        else:
            # Try with underscores replaced
            stem_clean = stem.replace(' ', '_').replace("'", "'")
            if stem_clean in piece_ids:
                piece_to_file[stem_clean] = midi_file

    print(f"    Matched {len(piece_to_file)} pieces to files")

    if args.piece:
        if args.piece not in piece_to_file:
            print(f"    Error: Piece '{args.piece}' not found")
            print(f"    Available: {list(piece_to_file.keys())[:5]}...")
            return
        pieces_to_test = [args.piece]
    else:
        pieces_to_test = list(piece_to_file.keys())[:5]  # Test first 5

    # Test reconstruction
    print(f"\n[3] Testing reconstruction on {len(pieces_to_test)} pieces...")

    all_results = []
    for piece_id in pieces_to_test:
        midi_path = piece_to_file[piece_id]
        print(f"\n    Piece: {piece_id}")

        # Load original
        original_tracks, tpb = load_original_midi(str(midi_path))
        print(f"      Original: {sum(len(t) for t in original_tracks.values())} notes across {len(original_tracks)} tracks")

        # Reconstruct from patterns
        reconstructed = reconstruct_from_patterns(ckpt['patterns'], piece_id)
        print(f"      Reconstructed: {sum(len(t) for t in reconstructed.values())} notes across {len(reconstructed)} tracks")

        # Compare
        results = compare_tracks(original_tracks, reconstructed)
        results['piece_id'] = piece_id
        all_results.append(results)

        print(f"      Coverage: {results.get('overall_coverage_pct', 0):.1f}%")
        print(f"      Pitch class accuracy: {results.get('overall_pc_accuracy_pct', 0):.1f}%")
        print(f"      Full pitch accuracy: {results.get('overall_pitch_accuracy_pct', 0):.1f}%")

        if args.verbose:
            print("      Per-track breakdown:")
            for track_id, track_stats in sorted(results['tracks'].items()):
                print(f"        Track {track_id}: {track_stats['coverage_pct']:.1f}% coverage, "
                      f"{track_stats['pitch_class_accuracy_pct']:.1f}% PC accuracy")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_orig = sum(r['total_original_notes'] for r in all_results)
    total_matched = sum(r['matched_notes'] for r in all_results)
    total_pc_match = sum(r['pitch_class_matches'] for r in all_results)
    total_pitch_match = sum(r['full_pitch_matches'] for r in all_results)

    if total_orig > 0:
        print(f"Total original notes: {total_orig}")
        print(f"Total matched notes: {total_matched}")
        print(f"Overall coverage: {total_matched/total_orig*100:.1f}%")
    if total_matched > 0:
        print(f"Pitch class accuracy (of matched): {total_pc_match/total_matched*100:.1f}%")
        print(f"Full pitch accuracy (of matched): {total_pitch_match/total_matched*100:.1f}%")

    print("\nGAPS IDENTIFIED:")
    print("1. Notes not covered by any pattern")
    print("2. Timing information (rhythm_ioi interpretation)")
    print("3. Velocity/duration quantization loss")
    print("4. MIDI metadata (tempo, time signature)")


if __name__ == '__main__':
    main()
