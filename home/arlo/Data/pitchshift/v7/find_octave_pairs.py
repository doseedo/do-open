#!/usr/bin/env python3
"""
Find EXACT octave equivalent pairs in trumpet data.

Looks for SEGMENTS (not just whole files) where:
1. One segment is exactly 12 semitones (1 octave) above/below another
2. At least 80% of pitch contour and timing matches
3. Timing is relative to first note of segment

Can find pairs:
- Between different audio files
- Within the SAME audio file (player repeats phrase an octave up/down)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to MIDI, handling zeros."""
    result = np.zeros_like(hz, dtype=float)
    valid = hz > 20
    if valid.any():
        result[valid] = 69 + 12 * np.log2(hz[valid] / 440.0)
    return result


@dataclass
class Note:
    midi: float
    start_frame: int
    end_frame: int
    duration_frames: int

    @property
    def rounded_midi(self) -> int:
        return int(round(self.midi))


@dataclass
class Segment:
    """A segment of notes from an audio file."""
    audio_path: str
    latent_path: str
    f0_path: str
    notes: List[Note]
    start_frame: int  # Start frame in original audio
    end_frame: int    # End frame in original audio
    median_midi: float

    def get_relative_notes(self) -> List[Dict]:
        """Get notes with timing relative to segment start."""
        if not self.notes:
            return []
        first_start = self.notes[0].start_frame
        return [
            {
                'midi': n.midi,
                'rel_start': n.start_frame - first_start,
                'duration': n.duration_frames,
            }
            for n in self.notes
        ]


def extract_notes(f0: np.ndarray, min_note_frames: int = 5) -> List[Note]:
    """Extract note events from f0 contour."""
    midi = hz_to_midi(f0)
    voiced = midi > 0

    notes = []
    in_note = False
    note_start = 0
    note_pitches = []

    for i, (is_voiced, pitch) in enumerate(zip(voiced, midi)):
        if is_voiced and not in_note:
            in_note = True
            note_start = i
            note_pitches = [pitch]
        elif is_voiced and in_note:
            note_pitches.append(pitch)
        elif not is_voiced and in_note:
            if len(note_pitches) >= min_note_frames:
                notes.append(Note(
                    midi=float(np.median(note_pitches)),
                    start_frame=note_start,
                    end_frame=i,
                    duration_frames=i - note_start,
                ))
            in_note = False
            note_pitches = []

    if in_note and len(note_pitches) >= min_note_frames:
        notes.append(Note(
            midi=float(np.median(note_pitches)),
            start_frame=note_start,
            end_frame=len(midi),
            duration_frames=len(midi) - note_start,
        ))

    return notes


def extract_segments(
    notes: List[Note],
    audio_path: str,
    latent_path: str,
    f0_path: str,
    min_notes: int = 3,
    max_notes: int = 20,
    slide_step: int = 1,
) -> List[Segment]:
    """
    Extract overlapping segments of notes from a recording.

    Uses sliding window to find all possible segments.
    """
    segments = []

    for seg_len in range(min_notes, min(max_notes + 1, len(notes) + 1)):
        for start_idx in range(0, len(notes) - seg_len + 1, slide_step):
            seg_notes = notes[start_idx:start_idx + seg_len]

            # Calculate median MIDI for this segment
            midi_values = [n.midi for n in seg_notes]
            median_midi = float(np.median(midi_values))

            segments.append(Segment(
                audio_path=audio_path,
                latent_path=latent_path,
                f0_path=f0_path,
                notes=seg_notes,
                start_frame=seg_notes[0].start_frame,
                end_frame=seg_notes[-1].end_frame,
                median_midi=median_midi,
            ))

    return segments


def compare_segments(
    seg1: Segment,
    seg2: Segment,
    pitch_tolerance: float = 1.5,  # semitones
    timing_tolerance_ratio: float = 0.2,  # fraction of note duration
    octave_offset: int = 12,
) -> Tuple[float, int]:
    """
    Compare two segments, checking if seg2 is an octave transposition of seg1.

    Returns (match_ratio, matched_notes)
    """
    notes1 = seg1.get_relative_notes()
    notes2 = seg2.get_relative_notes()

    if len(notes1) != len(notes2):
        return 0.0, 0

    matched = 0

    for n1, n2 in zip(notes1, notes2):
        # Check pitch (with octave offset)
        pitch_diff = abs((n1['midi'] + octave_offset) - n2['midi'])
        if pitch_diff > pitch_tolerance:
            continue

        # Check relative timing (allow some flexibility based on duration)
        avg_duration = (n1['duration'] + n2['duration']) / 2
        timing_tolerance = max(5, avg_duration * timing_tolerance_ratio)
        timing_diff = abs(n1['rel_start'] - n2['rel_start'])
        if timing_diff > timing_tolerance:
            continue

        # Check duration similarity
        duration_ratio = min(n1['duration'], n2['duration']) / max(n1['duration'], n2['duration'])
        if duration_ratio < 0.5:
            continue

        matched += 1

    match_ratio = matched / len(notes1) if notes1 else 0.0
    return match_ratio, matched


def load_and_process_entry(entry: Dict) -> Optional[Dict]:
    """Load f0 and extract notes/segments for an entry."""
    cond = entry.get('conditioning_paths', {})
    f0_path = fix_path(cond.get('f0', ''))
    audio_path = fix_path(entry.get('audio_path', ''))
    latent_path = fix_path(entry.get('latent_path', ''))

    if not f0_path or not os.path.exists(f0_path):
        return None

    try:
        f0 = np.load(f0_path)
        f0 = np.nan_to_num(f0, nan=0.0)
        notes = extract_notes(f0)

        if len(notes) < 3:
            return None

        return {
            'audio_path': audio_path,
            'latent_path': latent_path,
            'f0_path': f0_path,
            'notes': notes,
            'is_muted': entry.get('is_muted', False),
        }
    except Exception as e:
        return None


def find_octave_pairs(
    manifest_path: str,
    output_path: str,
    min_match_ratio: float = 0.8,
    min_notes: int = 4,
    max_notes: int = 12,
    instrument: str = 'trumpet',
    max_entries: int = None,
    workers: int = 8,
    check_same_file: bool = True,
):
    """Find octave equivalent segment pairs in manifest."""

    print(f"Loading manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Filter to instrument and non-muted
    entries = [
        e for e in manifest
        if e.get('sub_group') == instrument and not e.get('is_muted', False)
    ]

    if max_entries:
        entries = entries[:max_entries]

    print(f"Processing {len(entries)} {instrument} entries...")

    # Load all entries
    processed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load_and_process_entry, e): e for e in entries}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading f0"):
            result = future.result()
            if result:
                processed.append(result)

    print(f"Successfully loaded {len(processed)} entries")

    # Extract segments and group by note count and pitch
    print(f"Extracting segments (notes: {min_notes}-{max_notes})...")

    # Group segments by (num_notes, approximate_pitch_bin)
    segment_groups = defaultdict(list)

    for entry in tqdm(processed, desc="Extracting segments"):
        segments = extract_segments(
            entry['notes'],
            entry['audio_path'],
            entry['latent_path'],
            entry['f0_path'],
            min_notes=min_notes,
            max_notes=max_notes,
            slide_step=2,  # Slide by 2 notes to reduce redundancy
        )

        for seg in segments:
            # Group by (num_notes, pitch_bin) for efficient matching
            key = (len(seg.notes), int(round(seg.median_midi)))
            segment_groups[key].append(seg)

    total_segments = sum(len(v) for v in segment_groups.values())
    print(f"Extracted {total_segments} segments in {len(segment_groups)} groups")

    # Find pairs
    pairs = []
    checked = 0

    # For each segment group, find matching groups an octave apart
    keys = list(segment_groups.keys())

    for num_notes, pitch_low in tqdm(keys, desc="Finding pairs"):
        pitch_high = pitch_low + 12  # Octave up
        key_high = (num_notes, pitch_high)

        if key_high not in segment_groups:
            continue

        low_segments = segment_groups[(num_notes, pitch_low)]
        high_segments = segment_groups[key_high]

        for seg_low in low_segments:
            for seg_high in high_segments:
                # Skip if same file and segments overlap (unless check_same_file)
                same_file = seg_low.audio_path == seg_high.audio_path
                if same_file:
                    if not check_same_file:
                        continue
                    # Check if segments overlap in time
                    if (seg_low.start_frame < seg_high.end_frame and
                        seg_high.start_frame < seg_low.end_frame):
                        continue

                checked += 1

                match_ratio, matched = compare_segments(
                    seg_low, seg_high,
                    pitch_tolerance=1.5,
                    timing_tolerance_ratio=0.25,
                    octave_offset=12,
                )

                if match_ratio >= min_match_ratio:
                    pairs.append({
                        'low': {
                            'audio_path': seg_low.audio_path,
                            'latent_path': seg_low.latent_path,
                            'f0_path': seg_low.f0_path,
                            'median_midi': seg_low.median_midi,
                            'start_frame': seg_low.start_frame,
                            'end_frame': seg_low.end_frame,
                            'num_notes': len(seg_low.notes),
                        },
                        'high': {
                            'audio_path': seg_high.audio_path,
                            'latent_path': seg_high.latent_path,
                            'f0_path': seg_high.f0_path,
                            'median_midi': seg_high.median_midi,
                            'start_frame': seg_high.start_frame,
                            'end_frame': seg_high.end_frame,
                            'num_notes': len(seg_high.notes),
                        },
                        'match_ratio': match_ratio,
                        'matched_notes': matched,
                        'total_notes': len(seg_low.notes),
                        'pitch_diff': seg_high.median_midi - seg_low.median_midi,
                        'same_file': same_file,
                    })

    # Deduplicate pairs (same audio pair, overlapping segments)
    print(f"Found {len(pairs)} raw pairs, deduplicating...")

    # Sort by match ratio, then dedupe
    pairs.sort(key=lambda x: x['match_ratio'], reverse=True)

    seen = set()
    unique_pairs = []
    for pair in pairs:
        # Key: (low_file, high_file, low_start//50, high_start//50)
        key = (
            pair['low']['audio_path'],
            pair['high']['audio_path'],
            pair['low']['start_frame'] // 50,
            pair['high']['start_frame'] // 50,
        )
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    pairs = unique_pairs

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Checked {checked} potential pairs")
    print(f"Found {len(pairs)} unique pairs with >= {min_match_ratio*100:.0f}% match")

    same_file_pairs = [p for p in pairs if p['same_file']]
    diff_file_pairs = [p for p in pairs if not p['same_file']]
    print(f"  Same file: {len(same_file_pairs)}")
    print(f"  Different files: {len(diff_file_pairs)}")

    if pairs:
        print(f"\nTop 20 pairs:")
        for i, pair in enumerate(pairs[:20]):
            same_str = " [SAME FILE]" if pair['same_file'] else ""
            print(f"  {i+1}. Match: {pair['match_ratio']*100:.1f}% ({pair['matched_notes']}/{pair['total_notes']} notes){same_str}")
            print(f"      Low:  MIDI {pair['low']['median_midi']:.1f}, frames {pair['low']['start_frame']}-{pair['low']['end_frame']}")
            print(f"            {os.path.basename(pair['low']['audio_path'])}")
            print(f"      High: MIDI {pair['high']['median_midi']:.1f}, frames {pair['high']['start_frame']}-{pair['high']['end_frame']}")
            print(f"            {os.path.basename(pair['high']['audio_path'])}")

    # Save results
    output = {
        'min_match_ratio': min_match_ratio,
        'min_notes': min_notes,
        'max_notes': max_notes,
        'total_processed': len(processed),
        'total_segments': total_segments,
        'pairs_found': len(pairs),
        'same_file_pairs': len(same_file_pairs),
        'diff_file_pairs': len(diff_file_pairs),
        'pairs': pairs,
    }

    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Find octave equivalent segment pairs")
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json')
    parser.add_argument('--output', type=str,
                        default='/home/arlo/Data/pitchshift/v7/octave_pairs.json')
    parser.add_argument('--min_match', type=float, default=0.8,
                        help='Minimum match ratio (0-1)')
    parser.add_argument('--min_notes', type=int, default=4,
                        help='Minimum notes in a segment')
    parser.add_argument('--max_notes', type=int, default=12,
                        help='Maximum notes in a segment')
    parser.add_argument('--instrument', type=str, default='trumpet')
    parser.add_argument('--max_entries', type=int, default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--no_same_file', action='store_true',
                        help='Disable same-file pair detection')

    args = parser.parse_args()

    find_octave_pairs(
        manifest_path=args.manifest,
        output_path=args.output,
        min_match_ratio=args.min_match,
        min_notes=args.min_notes,
        max_notes=args.max_notes,
        instrument=args.instrument,
        max_entries=args.max_entries,
        workers=args.workers,
        check_same_file=not args.no_same_file,
    )


if __name__ == "__main__":
    main()
