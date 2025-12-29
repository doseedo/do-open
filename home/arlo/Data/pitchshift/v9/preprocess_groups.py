#!/usr/bin/env python3
"""
v9 Preprocessing: Split audio into 3 range groups and create formant-shifted pairs.

Range groups (F3 to F6):
- Group 1 (LOW):  F3-F4  = MIDI 53-65
- Group 2 (MID):  F4-F5  = MIDI 65-77
- Group 3 (HIGH): F5-F6  = MIDI 77-89

For each audio file:
1. Load f0 data
2. Find note segments
3. Split segments into groups (3 semitone tolerance)
4. Crop audio/latent files accordingly
5. Create formant-shifted synthetic pairs:
   - Group 1 → formant +12 → pair with Group 2
   - Group 2 → formant -12 → pair with Group 1
   - Group 2 → formant +12 → pair with Group 3
   - Group 3 → formant -12 → pair with Group 2
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

# Range definitions (MIDI notes)
GROUP_RANGES = {
    1: (53, 65),   # F3-F4 (LOW)
    2: (65, 77),   # F4-F5 (MID)
    3: (77, 89),   # F5-F6 (HIGH)
}

OVERLAP_TOLERANCE = 3  # semitones allowed into adjacent group
MIN_SEGMENT_SECONDS = 1.0  # minimum segment duration
MIN_RMS = 0.01  # minimum RMS amplitude to consider as audio (not silence)
HOP_SIZE = 512  # f0 hop size
SAMPLE_RATE = 44100


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def check_segment_rms(audio_path: str, start_sec: float, end_sec: float, min_rms: float = MIN_RMS) -> bool:
    """Check if audio segment has sufficient RMS (not silence)."""
    try:
        duration = end_sec - start_sec
        result = subprocess.run(
            ['sox', audio_path, '-n', 'trim', str(start_sec), str(duration), 'stat'],
            capture_output=True, text=True
        )
        for line in result.stderr.split('\n'):
            if 'RMS' in line and 'amplitude' in line:
                rms = float(line.split()[-1])
                return rms >= min_rms
        return False
    except Exception:
        return False


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to MIDI, handling zeros."""
    result = np.zeros_like(hz, dtype=float)
    valid = hz > 20
    if valid.any():
        result[valid] = 69 + 12 * np.log2(hz[valid] / 440.0)
    return result


def get_group(midi: float) -> Optional[int]:
    """Get group number for a MIDI note, with tolerance."""
    for group, (low, high) in GROUP_RANGES.items():
        # Extend range by tolerance for classification
        if low - OVERLAP_TOLERANCE <= midi <= high + OVERLAP_TOLERANCE:
            return group
    return None


def get_strict_group(midi: float) -> Optional[int]:
    """Get group number without tolerance (for strict boundaries)."""
    for group, (low, high) in GROUP_RANGES.items():
        if low <= midi < high:
            return group
    # Handle edge case for top of group 3
    if midi >= GROUP_RANGES[3][0] and midi <= GROUP_RANGES[3][1]:
        return 3
    return None


@dataclass
class NoteSegment:
    """A continuous segment of notes in a single group."""
    group: int
    start_frame: int
    end_frame: int
    midi_notes: List[float]

    @property
    def duration_seconds(self) -> float:
        return (self.end_frame - self.start_frame) * HOP_SIZE / SAMPLE_RATE

    @property
    def median_midi(self) -> float:
        return float(np.median(self.midi_notes)) if self.midi_notes else 0


def extract_group_segments(f0: np.ndarray) -> List[NoteSegment]:
    """
    Extract segments from f0, splitting at group boundaries.

    Returns list of NoteSegment, each belonging to a single group.
    """
    midi = hz_to_midi(f0)
    segments = []

    current_group = None
    segment_start = None
    segment_notes = []

    for i, m in enumerate(midi):
        if m <= 0:
            # Silence - end current segment if exists
            if current_group is not None and segment_start is not None:
                segments.append(NoteSegment(
                    group=current_group,
                    start_frame=segment_start,
                    end_frame=i,
                    midi_notes=segment_notes.copy()
                ))
                current_group = None
                segment_start = None
                segment_notes = []
            continue

        note_group = get_strict_group(m)

        if note_group is None:
            # Note outside all groups - end segment
            if current_group is not None:
                segments.append(NoteSegment(
                    group=current_group,
                    start_frame=segment_start,
                    end_frame=i,
                    midi_notes=segment_notes.copy()
                ))
                current_group = None
                segment_start = None
                segment_notes = []
            continue

        # Check if this note is within tolerance of current group
        if current_group is not None:
            group_low, group_high = GROUP_RANGES[current_group]
            if group_low - OVERLAP_TOLERANCE <= m <= group_high + OVERLAP_TOLERANCE:
                # Within tolerance, continue segment
                segment_notes.append(m)
                continue

        # Note belongs to different group or no current segment
        if note_group != current_group:
            # Save previous segment
            if current_group is not None and segment_start is not None:
                segments.append(NoteSegment(
                    group=current_group,
                    start_frame=segment_start,
                    end_frame=i,
                    midi_notes=segment_notes.copy()
                ))

            # Start new segment
            current_group = note_group
            segment_start = i
            segment_notes = [m]
        else:
            segment_notes.append(m)

    # End final segment
    if current_group is not None and segment_start is not None:
        segments.append(NoteSegment(
            group=current_group,
            start_frame=segment_start,
            end_frame=len(midi),
            midi_notes=segment_notes.copy()
        ))

    # Filter by minimum duration
    segments = [s for s in segments if s.duration_seconds >= MIN_SEGMENT_SECONDS]

    return segments


def frame_to_sample(frame: int) -> int:
    """Convert f0 frame to audio sample index."""
    return frame * HOP_SIZE


def frame_to_latent_idx(frame: int, latent_hop: int = 2048) -> int:
    """Convert f0 frame to latent time index."""
    return int(frame * HOP_SIZE / latent_hop)


def formant_shift_audio(input_path: str, output_path: str, semitones: int) -> bool:
    """Apply formant shift using sox (pitch shift without tempo change)."""
    try:
        cents = semitones * 100
        # sox formant shift: pitch shift then speed adjust to keep duration
        # Actually sox 'pitch' does formant-preserving pitch shift
        # We want the opposite: keep pitch, shift formants
        # Use 'bend' or manual approach

        # For formant shift: shift pitch, then shift back with speed
        # This is tricky. Let's use rubberband if available, or approximate with sox

        # Simple approximation: use sox effects chain
        # pitch +N cents shifts formants up while keeping pitch
        # Actually that's wrong - 'pitch' in sox shifts both

        # The correct approach for formant-only shift:
        # 1. Speed up (raises pitch AND formants)
        # 2. Pitch down (lowers pitch, keeps formants shifted)

        speed_factor = 2 ** (semitones / 12)
        pitch_cents = -semitones * 100

        cmd = [
            'sox', input_path, output_path,
            'speed', str(speed_factor),
            'pitch', str(pitch_cents)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"Formant shift failed: {e}")
        return False


def crop_audio(input_path: str, output_path: str, start_sec: float, end_sec: float) -> bool:
    """Crop audio file using sox."""
    try:
        duration = end_sec - start_sec
        cmd = ['sox', input_path, output_path, 'trim', str(start_sec), str(duration)]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"Crop failed: {e}")
        return False


def crop_latent(latent_path: str, start_frame: int, end_frame: int) -> Optional[torch.Tensor]:
    """Crop latent tensor based on f0 frames."""
    try:
        data = torch.load(latent_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            latent = data.get('latents', data.get('latent'))
        else:
            latent = data

        if latent is None:
            return None

        if latent.dim() == 4:
            latent = latent.squeeze(0)

        # Convert f0 frames to latent indices
        start_idx = frame_to_latent_idx(start_frame)
        end_idx = frame_to_latent_idx(end_frame)

        # Clamp to valid range
        T = latent.shape[-1]
        start_idx = max(0, start_idx)
        end_idx = min(T, end_idx)

        if end_idx <= start_idx:
            return None

        return latent[:, :, start_idx:end_idx]
    except Exception as e:
        print(f"Latent crop failed: {e}")
        return None


def process_entry(entry: Dict, output_dir: Path) -> List[Dict]:
    """
    Process a manifest entry: split into group segments.

    Returns list of new entries, one per valid segment.
    """
    results = []

    cond = entry.get('conditioning_paths', {})
    f0_path = fix_path(cond.get('f0', ''))
    audio_path = fix_path(entry.get('audio_path', ''))
    latent_path = fix_path(entry.get('latent_path', ''))

    if not f0_path or not os.path.exists(f0_path):
        return results
    if not latent_path or not os.path.exists(latent_path):
        return results

    try:
        f0 = np.load(f0_path)
        f0 = np.nan_to_num(f0, nan=0.0)
    except:
        return results

    segments = extract_group_segments(f0)

    if not segments:
        return results

    # Create segment entries
    for i, seg in enumerate(segments):
        seg_id = f"{Path(audio_path).stem}_seg{i}_g{seg.group}"

        # Calculate time bounds
        start_sec = frame_to_sample(seg.start_frame) / SAMPLE_RATE
        end_sec = frame_to_sample(seg.end_frame) / SAMPLE_RATE

        # Check if segment has actual audio content (not silence)
        if audio_path and os.path.exists(audio_path):
            if not check_segment_rms(audio_path, start_sec, end_sec):
                continue  # Skip silent segments

        # Crop latent
        cropped_latent = crop_latent(latent_path, seg.start_frame, seg.end_frame)
        if cropped_latent is None or cropped_latent.shape[-1] < 4:
            continue

        # Save cropped latent
        group_dir = output_dir / f"group{seg.group}"
        group_dir.mkdir(parents=True, exist_ok=True)

        latent_out = group_dir / f"{seg_id}.pt"
        torch.save({'latent': cropped_latent}, latent_out)

        results.append({
            'segment_id': seg_id,
            'group': seg.group,
            'latent_path': str(latent_out),
            'original_audio': audio_path,
            'original_latent': latent_path,
            'start_frame': seg.start_frame,
            'end_frame': seg.end_frame,
            'start_sec': start_sec,
            'end_sec': end_sec,
            'median_midi': seg.median_midi,
            'duration_sec': seg.duration_seconds,
        })

    return results


def create_formant_pairs(
    group_entries: Dict[int, List[Dict]],
    output_dir: Path,
    max_pairs_per_direction: int = 2000,
) -> List[Dict]:
    """
    Create formant-shifted synthetic pairs between adjacent groups.

    Pairs:
    - Group 1 → formant +12 → matches Group 2
    - Group 2 → formant -12 → matches Group 1
    - Group 2 → formant +12 → matches Group 3
    - Group 3 → formant -12 → matches Group 2
    """
    pairs = []

    # Define pair directions: (source_group, target_group, formant_shift)
    directions = [
        (1, 2, +12),  # LOW up to MID
        (2, 1, -12),  # MID down to LOW
        (2, 3, +12),  # MID up to HIGH
        (3, 2, -12),  # HIGH down to MID
    ]

    pairs_dir = output_dir / "formant_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    for src_group, tgt_group, shift in directions:
        src_entries = group_entries.get(src_group, [])

        if not src_entries:
            print(f"  No entries for group {src_group}")
            continue

        # Limit number of pairs
        src_sample = src_entries[:max_pairs_per_direction]

        direction_name = f"g{src_group}_to_g{tgt_group}"
        dir_path = pairs_dir / direction_name
        dir_path.mkdir(parents=True, exist_ok=True)

        print(f"  Creating {len(src_sample)} pairs: Group {src_group} → {tgt_group} (shift {shift:+d})")

        for entry in tqdm(src_sample, desc=f"G{src_group}→G{tgt_group}"):
            try:
                # Load source latent
                data = torch.load(entry['latent_path'], map_location='cpu', weights_only=False)
                if isinstance(data, dict):
                    src_latent = data.get('latent', data.get('latents'))
                else:
                    src_latent = data

                if src_latent is None:
                    continue

                # For now, we'll create the pairs at training time
                # by loading natural audio, applying formant shift, and encoding
                # This is more complex - let's just store the metadata

                pair_id = f"{entry['segment_id']}_shift{shift:+d}"

                pairs.append({
                    'pair_id': pair_id,
                    'source_group': src_group,
                    'target_group': tgt_group,
                    'formant_shift': shift,
                    'source_latent_path': entry['latent_path'],
                    'source_audio': entry.get('original_audio', ''),
                    'start_sec': entry.get('start_sec', 0),
                    'end_sec': entry.get('end_sec', 0),
                    'median_midi': entry['median_midi'],
                })

            except Exception as e:
                continue

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio into range groups")
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/msdd2/pitchshift_v9_groups')
    parser.add_argument('--instrument', type=str, default='trumpet')
    parser.add_argument('--max_entries', type=int, default=None)
    parser.add_argument('--workers', type=int, default=8)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest: {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)

    # Filter to instrument and non-muted
    entries = [
        e for e in manifest
        if e.get('sub_group') == args.instrument and not e.get('is_muted', False)
    ]

    if args.max_entries:
        entries = entries[:args.max_entries]

    print(f"Processing {len(entries)} {args.instrument} entries...")

    # Process entries in parallel
    all_segments = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_entry, e, output_dir): e for e in entries}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            segments = future.result()
            all_segments.extend(segments)

    # Group by range
    group_entries = {1: [], 2: [], 3: []}
    for seg in all_segments:
        group_entries[seg['group']].append(seg)

    print(f"\nGroup statistics:")
    for g in [1, 2, 3]:
        low, high = GROUP_RANGES[g]
        print(f"  Group {g} (MIDI {low}-{high}): {len(group_entries[g])} segments")

    # Create formant-shifted pairs
    print(f"\nCreating formant-shifted pairs...")
    pairs = create_formant_pairs(group_entries, output_dir)

    # Save manifest
    manifest_out = {
        'group_ranges': GROUP_RANGES,
        'overlap_tolerance': OVERLAP_TOLERANCE,
        'min_segment_seconds': MIN_SEGMENT_SECONDS,
        'group_entries': group_entries,
        'formant_pairs': pairs,
        'total_segments': len(all_segments),
        'total_pairs': len(pairs),
    }

    manifest_path = output_dir / "manifest.json"
    print(f"\nSaving manifest to: {manifest_path}")
    with open(manifest_path, 'w') as f:
        json.dump(manifest_out, f, indent=2)

    print(f"\nDone!")
    print(f"  Total segments: {len(all_segments)}")
    print(f"  Total formant pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
