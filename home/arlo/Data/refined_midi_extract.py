#!/usr/bin/env python3
"""
Refined MIDI Extraction - Instrument-Specific Transcription

Uses multiple data sources to improve MIDI transcription accuracy:
- F0 contour (torchcrepe) - ground truth for monophonic content
- Amplitude envelope - note boundaries and velocity
- Voicing mask (rframe) - voiced/unvoiced detection
- Chroma - harmonic content validation
- MultiF0 - polyphonic pitch salience (where available)
- BasicPitch MIDI - baseline for polyphonic content

Strategies by instrument:
- MONOPHONIC (voice, winds, brass, strings solo, bass): F0 -> MIDI directly
- POLYPHONIC (guitar, piano, synth): BasicPitch + overtone filtering with F0
- DRUMS/PERCUSSION: Onset detection only (no pitch)
- FX/ROOM: Skip or use BasicPitch as-is

Usage:
    python refined_midi_extract.py --test              # Test on samples
    python refined_midi_extract.py --evaluate          # Compare vs BasicPitch
    python refined_midi_extract.py --process FILE      # Process single file
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Lazy imports for speed
pretty_midi = None
def get_pretty_midi():
    global pretty_midi
    if pretty_midi is None:
        import pretty_midi as pm
        pretty_midi = pm
    return pretty_midi


@dataclass
class Config:
    """Configuration for MIDI extraction."""
    frame_rate: float = 10.77  # Hz (44100 / 4096)

    # F0 -> MIDI parameters (monophonic)
    min_note_frames: int = 3
    pitch_stability_cents: float = 50.0
    confidence_thresh: float = 0.5
    amp_thresh: float = 0.02

    # Overtone filtering (polyphonic)
    harmonic_tolerance_cents: float = 40.0
    max_harmonic: int = 5

    # Note merging
    merge_gap_sec: float = 0.05
    min_note_duration_sec: float = 0.03


# Instrument-specific configs (tuned from testing)
INSTRUMENT_CONFIGS = {
    # Stable instruments - use tighter settings
    'strings': Config(pitch_stability_cents=50, min_note_frames=3, merge_gap_sec=0.05),
    'winds': Config(pitch_stability_cents=50, min_note_frames=3, merge_gap_sec=0.05),
    'brass': Config(pitch_stability_cents=50, min_note_frames=3, merge_gap_sec=0.05),
    'bass': Config(pitch_stability_cents=50, min_note_frames=3, merge_gap_sec=0.05),

    # Voice - needs more tolerance for vibrato
    'voice': Config(pitch_stability_cents=100, min_note_frames=5, merge_gap_sec=0.1),

    # Polyphonic - use BasicPitch filtering
    'guitar': Config(harmonic_tolerance_cents=50, max_harmonic=5),
    'piano': Config(harmonic_tolerance_cents=50, max_harmonic=5),
    'synth': Config(harmonic_tolerance_cents=50, max_harmonic=5),
    'organ': Config(harmonic_tolerance_cents=50, max_harmonic=5),
    'plucked': Config(harmonic_tolerance_cents=50, max_harmonic=5),
    'mallets': Config(harmonic_tolerance_cents=50, max_harmonic=5),
}


def get_config_for_instrument(group: str, subgroup: str) -> Config:
    """Get tuned config for instrument type."""
    if group in INSTRUMENT_CONFIGS:
        return INSTRUMENT_CONFIGS[group]
    return Config()  # Default


# Instrument classification
MONOPHONIC_GROUPS = {'voice', 'winds', 'brass'}
MONOPHONIC_SUBGROUPS = {'violin', 'viola', 'cello', 'flute', 'clarinet', 'oboe',
                         'sax', 'trumpet', 'trombone', 'french_horn', 'tuba'}
POLYPHONIC_GROUPS = {'guitar', 'piano', 'synth', 'organ', 'plucked', 'mallets'}
BASS_GROUPS = {'bass'}
SKIP_GROUPS = {'drums', 'percussion', 'fx', 'room'}


def is_monophonic(group: str, subgroup: str) -> bool:
    """Check if instrument is primarily monophonic."""
    if group in MONOPHONIC_GROUPS:
        return True
    if subgroup in MONOPHONIC_SUBGROUPS:
        return True
    return False


def is_bass(group: str, subgroup: str) -> bool:
    """Check if instrument is bass (mostly monophonic, low range)."""
    return group in BASS_GROUPS or 'bass' in subgroup.lower()


def should_skip(group: str, subgroup: str) -> bool:
    """Check if we should skip MIDI extraction for this instrument."""
    return group in SKIP_GROUPS


def hz_to_midi(hz: float) -> float:
    """Convert Hz to MIDI pitch."""
    if hz <= 0:
        return 0
    return 12 * np.log2(hz / 440.0) + 69


def midi_to_hz(midi: float) -> float:
    """Convert MIDI pitch to Hz."""
    return 440.0 * (2 ** ((midi - 69) / 12))


def f0_to_midi_notes(
    f0: np.ndarray,
    confidence: np.ndarray,
    amp: np.ndarray,
    config: Config
) -> List[Dict]:
    """
    Convert F0 contour to MIDI notes using segmentation.
    For monophonic instruments.
    """
    notes = []
    frame_rate = config.frame_rate

    # Voiced mask
    voiced = (confidence > config.confidence_thresh) & (amp > config.amp_thresh) & (f0 > 0)

    # Convert F0 to continuous MIDI pitch
    midi_pitch = np.zeros_like(f0)
    valid_f0 = f0 > 0
    midi_pitch[valid_f0] = 12 * np.log2(f0[valid_f0] / 440.0) + 69

    # State machine for note detection
    in_note = False
    note_start = 0
    note_pitches = []
    note_amps = []

    for i in range(len(f0)):
        if voiced[i] and not in_note:
            # Start new note
            in_note = True
            note_start = i
            note_pitches = [midi_pitch[i]]
            note_amps = [amp[i]]

        elif voiced[i] and in_note:
            # Check if pitch changed significantly (new note)
            current_pitch = midi_pitch[i]
            # Use recent median (last 5-10 frames)
            recent_pitches = note_pitches[-10:] if len(note_pitches) >= 10 else note_pitches
            median_pitch = np.median(recent_pitches)
            cents_diff = abs(current_pitch - median_pitch) * 100

            if cents_diff > config.pitch_stability_cents:
                # End current note, start new one
                if len(note_pitches) >= config.min_note_frames:
                    notes.append({
                        'start': note_start / frame_rate,
                        'end': i / frame_rate,
                        'pitch': int(round(np.median(note_pitches))),
                        'velocity': int(np.clip(max(note_amps) * 127, 1, 127))
                    })
                note_start = i
                note_pitches = [current_pitch]
                note_amps = [amp[i]]
            else:
                note_pitches.append(current_pitch)
                note_amps.append(amp[i])

        elif not voiced[i] and in_note:
            # End note
            if len(note_pitches) >= config.min_note_frames:
                notes.append({
                    'start': note_start / frame_rate,
                    'end': i / frame_rate,
                    'pitch': int(round(np.median(note_pitches))),
                    'velocity': int(np.clip(max(note_amps) * 127, 1, 127))
                })
            in_note = False
            note_pitches = []
            note_amps = []

    # Handle final note
    if in_note and len(note_pitches) >= config.min_note_frames:
        notes.append({
            'start': note_start / frame_rate,
            'end': len(f0) / frame_rate,
            'pitch': int(round(np.median(note_pitches))),
            'velocity': int(np.clip(max(note_amps) * 127, 1, 127))
        })

    return notes


def filter_overtones_with_f0(
    bp_notes: List[Dict],
    f0: np.ndarray,
    confidence: np.ndarray,
    config: Config
) -> List[Dict]:
    """
    Filter BasicPitch notes that are likely overtones using F0 as validator.
    For polyphonic instruments.
    """
    filtered = []
    frame_rate = config.frame_rate

    for note in bp_notes:
        start_frame = int(note['start'] * frame_rate)
        end_frame = min(int(note['end'] * frame_rate), len(f0))

        if start_frame >= end_frame or start_frame >= len(f0):
            continue

        note_hz = midi_to_hz(note['pitch'])

        # Get F0 during this note
        f0_segment = f0[start_frame:end_frame]
        conf_segment = confidence[start_frame:end_frame]

        voiced = conf_segment > config.confidence_thresh
        if voiced.sum() < 3:
            # No confident F0 - could be percussive/noise, keep with lower confidence
            note['confidence'] = 0.5
            filtered.append(note)
            continue

        voiced_f0 = f0_segment[voiced]
        median_f0 = np.median(voiced_f0[voiced_f0 > 0]) if (voiced_f0 > 0).sum() > 0 else 0

        if median_f0 <= 0:
            note['confidence'] = 0.5
            filtered.append(note)
            continue

        # Check if note IS the fundamental (or close to it)
        cents_from_f0 = 1200 * np.log2(note_hz / median_f0)
        if abs(cents_from_f0) < config.harmonic_tolerance_cents:
            # Matches F0 - high confidence
            note['confidence'] = 0.95
            filtered.append(note)
            continue

        # Check if note is a harmonic of F0 (overtone - reject)
        is_overtone = False
        for harmonic in range(2, config.max_harmonic + 1):
            harmonic_hz = median_f0 * harmonic
            cents_from_harmonic = 1200 * np.log2(note_hz / harmonic_hz)
            if abs(cents_from_harmonic) < config.harmonic_tolerance_cents:
                # This is an overtone - skip it
                is_overtone = True
                break

        if not is_overtone:
            # Not F0 and not an overtone - could be a chord note
            # Check if it's harmonically related (3rd, 5th, etc.)
            note['confidence'] = 0.7
            filtered.append(note)

    return filtered


def validate_with_chroma(
    notes: List[Dict],
    chroma: np.ndarray,
    config: Config
) -> List[Dict]:
    """
    Validate notes against chroma features.
    Remove notes that don't have chroma support.
    """
    if chroma is None or len(chroma) == 0:
        return notes

    # Chroma is [12, T] - pitch classes C, C#, D, ..., B
    frame_rate = config.frame_rate
    validated = []

    for note in notes:
        start_frame = int(note['start'] * frame_rate)
        end_frame = min(int(note['end'] * frame_rate), chroma.shape[1])

        if start_frame >= end_frame or start_frame >= chroma.shape[1]:
            continue

        # Get pitch class (0-11)
        pitch_class = note['pitch'] % 12

        # Get chroma energy for this pitch class during the note
        chroma_segment = chroma[pitch_class, start_frame:end_frame]
        chroma_energy = chroma_segment.mean()

        # Also check if this pitch class is the strongest
        all_chroma = chroma[:, start_frame:end_frame].mean(axis=1)
        is_strong = chroma_energy >= np.percentile(all_chroma, 50)

        if chroma_energy > 0.1 or is_strong:
            validated.append(note)
        # Else: note doesn't have chroma support, skip it

    return validated


def merge_short_notes(notes: List[Dict], config: Config) -> List[Dict]:
    """Merge consecutive notes with same pitch that are close together."""
    if len(notes) == 0:
        return notes

    # Sort by start time
    notes = sorted(notes, key=lambda n: n['start'])

    merged = [notes[0].copy()]

    for note in notes[1:]:
        prev = merged[-1]
        gap = note['start'] - prev['end']

        if note['pitch'] == prev['pitch'] and gap < config.merge_gap_sec:
            # Merge with previous
            prev['end'] = note['end']
            prev['velocity'] = max(prev['velocity'], note['velocity'])
        else:
            merged.append(note.copy())

    # Filter very short notes
    min_dur = config.min_note_duration_sec
    merged = [n for n in merged if n['end'] - n['start'] >= min_dur]

    return merged


def notes_to_midi(notes: List[Dict], output_path: Path) -> None:
    """Convert note list to MIDI file."""
    pm = get_pretty_midi()
    midi = pm.PrettyMIDI()
    instrument = pm.Instrument(program=0)

    for note in notes:
        midi_note = pm.Note(
            velocity=note.get('velocity', 100),
            pitch=note['pitch'],
            start=note['start'],
            end=note['end']
        )
        instrument.notes.append(midi_note)

    midi.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))


def midi_to_notes(midi_path: Path) -> List[Dict]:
    """Load MIDI file and convert to note list."""
    pm = get_pretty_midi()
    midi = pm.PrettyMIDI(str(midi_path))

    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'velocity': note.velocity
            })

    return sorted(notes, key=lambda n: n['start'])


class RefinedMIDIExtractor:
    """
    Instrument-aware MIDI extraction using multiple data sources.
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.gcs_bucket = Path('/home/arlo/gcs-bucket')

        # Load manifests
        self.format_manifest = None
        self.combined_manifest = None
        self._load_manifests()

    def _load_manifests(self):
        """Load format and combined manifests."""
        manifest_dir = self.gcs_bucket / 'Manifests'

        format_path = manifest_dir / 'format_manifest.json'
        if format_path.exists():
            with open(format_path) as f:
                self.format_manifest = json.load(f)

        combined_path = manifest_dir / 'combined_manifest.json'
        if combined_path.exists():
            with open(combined_path) as f:
                self.combined_manifest = json.load(f)

    def get_instrument_info(self, audio_path: str) -> Tuple[str, str]:
        """Get instrument group and subgroup for audio file."""
        if self.combined_manifest is None:
            return 'undefined', 'undefined'

        # Try different path formats
        info = self.combined_manifest.get(audio_path)
        if not info:
            full_path = str(self.gcs_bucket / audio_path)
            info = self.combined_manifest.get(full_path)

        if info:
            return info.get('group', 'undefined'), info.get('subgroup', 'undefined')

        return 'undefined', 'undefined'

    def get_data_paths(self, audio_path: str) -> Dict[str, Optional[Path]]:
        """Get paths to all data files for an audio file."""
        # Parse audio path: protools/DATE/New|Prev/SESSION/Audio Files/FILE.wav
        parts = Path(audio_path).parts

        # Find the base structure
        source = parts[0]  # protools or protoolsA
        stem = Path(audio_path).stem

        paths = {
            'audio': self.gcs_bucket / audio_path,
            'f0': None,
            'amp': None,
            'rframe': None,
            'chroma': None,
            'multif0': None,
            'basicpitch': None,
        }

        # Conditioning files (same structure as audio)
        cond_base = self.gcs_bucket / 'Conditioning' / audio_path
        cond_dir = cond_base.parent
        paths['f0'] = cond_dir / f'{stem}.f0.npy'
        paths['amp'] = cond_dir / f'{stem}.amp.npy'
        paths['rframe'] = cond_dir / f'{stem}.rframe.npy'

        # BasicPitch MIDI
        bp_base = self.gcs_bucket / 'BasicPitch' / audio_path
        paths['basicpitch'] = bp_base.parent / f'{stem}.mid'

        # Chroma (different structure - by session name, no date)
        # Try to find matching chroma file
        try:
            # Extract session name from path
            if len(parts) >= 4:
                session_name = parts[3]  # e.g., "2025.02.21_Sueñero_VoxRec"
                chroma_dir = self.gcs_bucket / 'Chroma' / session_name
                chroma_file = chroma_dir / f'{stem}.chroma.npy'
                if chroma_file.exists():
                    paths['chroma'] = chroma_file
        except:
            pass

        # MultiF0 (date organized like audio)
        multif0_base = self.gcs_bucket / 'MultiF0' / audio_path
        multif0_file = multif0_base.parent / f'{stem}.npy'
        if multif0_file.exists():
            paths['multif0'] = multif0_file

        return paths

    def load_data(self, paths: Dict[str, Optional[Path]]) -> Dict[str, Optional[np.ndarray]]:
        """Load all available data arrays."""
        data = {}

        for key in ['f0', 'amp', 'rframe', 'chroma', 'multif0']:
            path = paths.get(key)
            if path and path.exists():
                try:
                    data[key] = np.load(path)
                except Exception as e:
                    print(f"  Warning: Failed to load {key}: {e}")
                    data[key] = None
            else:
                data[key] = None

        return data

    def extract_monophonic(
        self,
        f0: np.ndarray,
        rframe: np.ndarray,
        amp: np.ndarray,
        chroma: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """Extract MIDI for monophonic instruments using F0."""
        # Use rframe as confidence (voicing)
        confidence = rframe if rframe is not None else np.ones_like(f0)
        if amp is None:
            amp = np.ones_like(f0)

        # Convert F0 to notes
        notes = f0_to_midi_notes(f0, confidence, amp, self.config)

        # Validate with chroma if available
        if chroma is not None:
            notes = validate_with_chroma(notes, chroma, self.config)

        # Merge fragmented notes
        notes = merge_short_notes(notes, self.config)

        return notes

    def extract_polyphonic(
        self,
        bp_notes: List[Dict],
        f0: np.ndarray,
        rframe: np.ndarray,
        amp: Optional[np.ndarray] = None,
        chroma: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """Extract MIDI for polyphonic instruments using BasicPitch + F0 filtering."""
        confidence = rframe if rframe is not None else np.ones_like(f0)

        # Filter overtones using F0
        notes = filter_overtones_with_f0(bp_notes, f0, confidence, self.config)

        # Validate with chroma if available
        if chroma is not None:
            notes = validate_with_chroma(notes, chroma, self.config)

        # Merge fragmented notes
        notes = merge_short_notes(notes, self.config)

        return notes

    def extract(self, audio_path: str, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Extract refined MIDI for an audio file.

        Returns dict with:
            - notes: List of note dicts
            - strategy: 'monophonic', 'polyphonic', 'bass', or 'skipped'
            - group, subgroup: Instrument classification
            - stats: Note count, duration, etc.
        """
        result = {
            'audio_path': audio_path,
            'notes': [],
            'strategy': None,
            'group': None,
            'subgroup': None,
            'stats': {},
            'error': None
        }

        # Get instrument info
        group, subgroup = self.get_instrument_info(audio_path)
        result['group'] = group
        result['subgroup'] = subgroup

        # Determine strategy
        if should_skip(group, subgroup):
            result['strategy'] = 'skipped'
            return result

        # Get data paths
        paths = self.get_data_paths(audio_path)

        # Check required data exists
        if not paths['f0'] or not paths['f0'].exists():
            result['error'] = 'F0 data not found'
            return result

        # Load data
        data = self.load_data(paths)
        f0 = data['f0']
        amp = data['amp']
        rframe = data['rframe']
        chroma = data['chroma']

        if f0 is None:
            result['error'] = 'Failed to load F0'
            return result

        try:
            if is_monophonic(group, subgroup) or is_bass(group, subgroup):
                # Use F0-based extraction
                result['strategy'] = 'monophonic' if is_monophonic(group, subgroup) else 'bass'
                result['notes'] = self.extract_monophonic(f0, rframe, amp, chroma)

            else:
                # Polyphonic - use BasicPitch + filtering
                result['strategy'] = 'polyphonic'

                # Load BasicPitch MIDI
                if paths['basicpitch'] and paths['basicpitch'].exists():
                    bp_notes = midi_to_notes(paths['basicpitch'])
                    result['notes'] = self.extract_polyphonic(bp_notes, f0, rframe, amp, chroma)
                else:
                    # Fall back to F0-based if no BasicPitch
                    result['strategy'] = 'monophonic_fallback'
                    result['notes'] = self.extract_monophonic(f0, rframe, amp, chroma)

            # Compute stats
            notes = result['notes']
            if notes:
                result['stats'] = {
                    'note_count': len(notes),
                    'total_duration': sum(n['end'] - n['start'] for n in notes),
                    'pitch_range': (min(n['pitch'] for n in notes), max(n['pitch'] for n in notes)),
                    'avg_note_duration': np.mean([n['end'] - n['start'] for n in notes])
                }

            # Save if output path provided
            if output_path and notes:
                notes_to_midi(notes, output_path)
                result['output_path'] = str(output_path)

        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()

        return result


def compare_midi_files(
    refined_notes: List[Dict],
    baseline_notes: List[Dict]
) -> Dict[str, Any]:
    """Compare refined MIDI against BasicPitch baseline."""
    stats = {
        'refined_count': len(refined_notes),
        'baseline_count': len(baseline_notes),
        'removed_notes': 0,
        'added_notes': 0,
    }

    # Count potential overtones removed
    baseline_pitches = set(n['pitch'] for n in baseline_notes)
    refined_pitches = set(n['pitch'] for n in refined_notes)

    # Notes that were removed (potential overtones)
    removed = baseline_pitches - refined_pitches
    added = refined_pitches - baseline_pitches

    stats['removed_pitches'] = list(removed)
    stats['added_pitches'] = list(added)
    stats['reduction_pct'] = 100 * (1 - len(refined_notes) / max(len(baseline_notes), 1))

    return stats


def test_samples():
    """Test refined extraction on sample files from each instrument group."""
    print("=" * 60)
    print("Testing Refined MIDI Extraction")
    print("=" * 60)

    extractor = RefinedMIDIExtractor()

    # Sample files by group (from earlier exploration)
    samples = {
        'voice': 'protools/2025-03-30/New/25.02.07_JmilliganFkawase_Movies_Rec 01/Audio Files/LDVOX WET_01.wav',
        'winds': 'protools/2025-03-29/New/2025.02.21_Sueñero_VoxRec/Audio Files/Alto_bip_1.wav',
        'brass': 'protools/2025-03-31/New/Grapevine_PRE Drums/Audio Files/TB_01.wav',
        'strings': 'protools/2025-03-31/New/Dead Weight_POST DRUMS/Audio Files/cello 1_01.wav',
        'guitar': 'protools/2025-03-29/New/2025.02.21_Sueñero_VoxRec/Audio Files/Classical Guitar_bip_1.wav',
        'piano': 'protools/2025-03-29/New/2025.02.21_Sueñero_VoxRec/Audio Files/Piano_bip_1.wav',
        'bass': 'protools/2025-03-29/New/2025.02.21_Sueñero_VoxRec/Audio Files/Bass Guitar_bip_1.wav',
    }

    results = {}

    for group, audio_path in samples.items():
        print(f"\n--- {group.upper()} ---")
        print(f"File: {Path(audio_path).name}")

        # Extract refined MIDI
        result = extractor.extract(audio_path)

        if result['error']:
            print(f"  ERROR: {result['error']}")
            continue

        print(f"  Strategy: {result['strategy']}")
        print(f"  Notes extracted: {result['stats'].get('note_count', 0)}")

        if result['stats'].get('pitch_range'):
            print(f"  Pitch range: {result['stats']['pitch_range']}")

        # Compare with BasicPitch baseline
        paths = extractor.get_data_paths(audio_path)
        if paths['basicpitch'] and paths['basicpitch'].exists():
            bp_notes = midi_to_notes(paths['basicpitch'])
            comparison = compare_midi_files(result['notes'], bp_notes)

            print(f"  BasicPitch notes: {comparison['baseline_count']}")
            print(f"  Refined notes: {comparison['refined_count']}")
            print(f"  Reduction: {comparison['reduction_pct']:.1f}%")

            result['comparison'] = comparison

        results[group] = result

    return results


def evaluate_accuracy():
    """
    Evaluate refined extraction across multiple files per instrument group.
    Compare against BasicPitch baseline.
    """
    print("=" * 60)
    print("Evaluating Refined MIDI Extraction Accuracy")
    print("=" * 60)

    extractor = RefinedMIDIExtractor()

    # Get sample files from manifest
    if extractor.format_manifest is None:
        print("ERROR: Format manifest not found")
        return

    # Find entries with all 3 formats
    all_three = [
        e for e in extractor.format_manifest['entries']
        if e.get('has_latent') is True
        and e.get('has_conditioning') is True
        and e.get('has_midi') is True
    ]

    # Group by instrument
    by_group = defaultdict(list)
    for entry in all_three:
        group, subgroup = extractor.get_instrument_info(entry['path'])
        if group != 'undefined' and not should_skip(group, subgroup):
            by_group[group].append(entry['path'])

    # Evaluate up to 20 files per group
    MAX_PER_GROUP = 20
    results = defaultdict(list)

    for group, paths in sorted(by_group.items()):
        print(f"\n=== {group.upper()} ({len(paths)} files available) ===")

        sample_paths = paths[:MAX_PER_GROUP]

        for audio_path in sample_paths:
            result = extractor.extract(audio_path)

            if result['error']:
                continue

            # Compare with BasicPitch
            data_paths = extractor.get_data_paths(audio_path)
            if data_paths['basicpitch'] and data_paths['basicpitch'].exists():
                bp_notes = midi_to_notes(data_paths['basicpitch'])
                comparison = compare_midi_files(result['notes'], bp_notes)
                result['comparison'] = comparison
                results[group].append(result)

        # Summarize group
        group_results = results[group]
        if group_results:
            avg_reduction = np.mean([
                r['comparison']['reduction_pct']
                for r in group_results if 'comparison' in r
            ])
            avg_notes = np.mean([r['stats']['note_count'] for r in group_results])

            print(f"  Files processed: {len(group_results)}")
            print(f"  Avg notes: {avg_notes:.1f}")
            print(f"  Avg reduction vs BasicPitch: {avg_reduction:.1f}%")

    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    all_results = [r for group_results in results.values() for r in group_results]

    if all_results:
        total_reduction = np.mean([
            r['comparison']['reduction_pct']
            for r in all_results if 'comparison' in r
        ])

        by_strategy = defaultdict(list)
        for r in all_results:
            by_strategy[r['strategy']].append(r)

        print(f"\nTotal files evaluated: {len(all_results)}")
        print(f"Average note reduction: {total_reduction:.1f}%")
        print(f"\nBy strategy:")
        for strategy, strategy_results in by_strategy.items():
            avg_notes = np.mean([r['stats']['note_count'] for r in strategy_results])
            print(f"  {strategy}: {len(strategy_results)} files, avg {avg_notes:.1f} notes")

    return results


def main():
    parser = argparse.ArgumentParser(description="Refined MIDI Extraction")
    parser.add_argument("--test", action="store_true", help="Test on sample files")
    parser.add_argument("--evaluate", action="store_true", help="Full evaluation")
    parser.add_argument("--process", type=str, help="Process single file")
    parser.add_argument("--output", type=str, help="Output MIDI path")

    args = parser.parse_args()

    if args.test:
        test_samples()
    elif args.evaluate:
        evaluate_accuracy()
    elif args.process:
        extractor = RefinedMIDIExtractor()
        output_path = Path(args.output) if args.output else None
        result = extractor.extract(args.process, output_path)
        print(json.dumps(result, indent=2, default=str))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
