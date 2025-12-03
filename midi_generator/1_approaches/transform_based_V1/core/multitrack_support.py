"""
Multitrack MIDI Enhancement
============================

CRITICAL ENHANCEMENT: Add full multitrack/orchestration support.

Current Status (GAPS FOUND):
✅ Track index preserved in extract_notes_from_midi()
❌ Instrument type NOT extracted (program_change messages)
❌ Channel info NOT extracted
❌ Track names NOT extracted
❌ Transforms operate on ALL tracks uniformly (no instrument awareness)

This module fixes these gaps for proper orchestration learning.

For multitrack corpus with:
- 20 tracks per file
- 10 different instrument types (piano, drums, bass, strings, etc.)

We need to:
1. Extract instrument types from program_change messages
2. Preserve channel and track metadata
3. Create instrument-aware transforms
4. Learn orchestration patterns (which instruments play together)
5. Learn instrument-specific patterns (piano voicings vs string spacing)

Author: Agent 8 - Multitrack Enhancement
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import mido
from dataclasses import dataclass


# ============================================================================
# Instrument Definitions (General MIDI)
# ============================================================================

GM_INSTRUMENT_FAMILIES = {
    # Piano (0-7)
    'piano': list(range(0, 8)),

    # Chromatic Percussion (8-15)
    'chromatic_percussion': list(range(8, 16)),

    # Organ (16-23)
    'organ': list(range(16, 24)),

    # Guitar (24-31)
    'guitar': list(range(24, 32)),

    # Bass (32-39)
    'bass': list(range(32, 40)),

    # Strings (40-47)
    'strings': list(range(40, 48)),

    # Ensemble (48-55)
    'ensemble': list(range(48, 56)),

    # Brass (56-63)
    'brass': list(range(56, 64)),

    # Reed (64-71)
    'reed': list(range(64, 72)),

    # Pipe (72-79)
    'pipe': list(range(72, 80)),

    # Synth Lead (80-87)
    'synth_lead': list(range(80, 88)),

    # Synth Pad (88-95)
    'synth_pad': list(range(88, 96)),

    # Synth Effects (96-103)
    'synth_effects': list(range(96, 104)),

    # Ethnic (104-111)
    'ethnic': list(range(104, 112)),

    # Percussive (112-119)
    'percussive': list(range(112, 120)),

    # Sound Effects (120-127)
    'sound_effects': list(range(120, 128)),

    # Drums (channel 10)
    'drums': [128]  # Special marker for drum channel
}


def get_instrument_family(program: int, channel: int = 0) -> str:
    """Get instrument family from MIDI program number"""
    # Channel 10 (9 in 0-indexed) is drums
    if channel == 9:
        return 'drums'

    for family, programs in GM_INSTRUMENT_FAMILIES.items():
        if program in programs:
            return family

    return 'unknown'


# ============================================================================
# Enhanced Note Extraction (WITH INSTRUMENTS)
# ============================================================================

@dataclass
class TrackInfo:
    """Metadata for a MIDI track"""
    track_idx: int
    track_name: str
    instrument_program: int  # MIDI program number (0-127)
    instrument_family: str  # 'piano', 'strings', 'drums', etc.
    channel: int
    note_count: int


def extract_track_info(midi: mido.MidiFile) -> List[TrackInfo]:
    """
    Extract metadata for each track.

    Args:
        midi: MIDI file

    Returns:
        List of TrackInfo for each track
    """
    track_infos = []

    for track_idx, track in enumerate(midi.tracks):
        # Extract track name
        track_name = f"Track {track_idx}"
        for msg in track:
            if msg.type == 'track_name':
                track_name = msg.name
                break

        # Extract instrument program (first program_change message)
        instrument_program = 0  # Default: Acoustic Grand Piano
        channel = 0
        for msg in track:
            if msg.type == 'program_change':
                instrument_program = msg.program
                channel = msg.channel
                break
            elif hasattr(msg, 'channel'):
                channel = msg.channel

        # Get instrument family
        instrument_family = get_instrument_family(instrument_program, channel)

        # Count notes in this track
        note_count = sum(
            1 for msg in track
            if msg.type == 'note_on' and msg.velocity > 0
        )

        track_infos.append(TrackInfo(
            track_idx=track_idx,
            track_name=track_name,
            instrument_program=instrument_program,
            instrument_family=instrument_family,
            channel=channel,
            note_count=note_count
        ))

    return track_infos


def extract_notes_with_instruments(midi: mido.MidiFile) -> List[Dict[str, Any]]:
    """
    Extract notes WITH instrument information.

    Enhanced version of extract_notes_from_midi() that includes:
    - Instrument type (program number)
    - Instrument family ('piano', 'strings', etc.)
    - Channel
    - Track name

    Args:
        midi: MIDI file

    Returns:
        List of note dictionaries with:
        - pitch: MIDI note number (0-127)
        - velocity: Note velocity (0-127)
        - start_time: Onset time in seconds
        - duration: Note duration in seconds
        - track: Track index
        - channel: MIDI channel (0-15)
        - instrument_program: MIDI program (0-127)
        - instrument_family: Family name ('piano', 'strings', etc.)
        - track_name: Track name string
    """
    # First, get track metadata
    track_infos = extract_track_info(midi)

    # Extract notes (similar to original)
    notes = []
    current_time = [0.0] * len(midi.tracks)
    tempo = 500000  # Default tempo (120 BPM)
    active_notes = {}  # (track, channel, pitch) -> start_info

    for track_idx, track in enumerate(midi.tracks):
        track_info = track_infos[track_idx]

        for msg in track:
            # Update time
            current_time[track_idx] += mido.tick2second(
                msg.time, midi.ticks_per_beat, tempo
            )

            # Update tempo
            if msg.type == 'set_tempo':
                tempo = msg.tempo

            # Note on
            elif msg.type == 'note_on' and msg.velocity > 0:
                key = (track_idx, msg.channel, msg.note)
                active_notes[key] = {
                    'start_time': current_time[track_idx],
                    'velocity': msg.velocity,
                    'channel': msg.channel
                }

            # Note off
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (track_idx, msg.channel, msg.note)
                if key in active_notes:
                    note_info = active_notes[key]
                    notes.append({
                        # Basic note info
                        'pitch': msg.note,
                        'velocity': note_info['velocity'],
                        'start_time': note_info['start_time'],
                        'duration': current_time[track_idx] - note_info['start_time'],

                        # Track info
                        'track': track_idx,
                        'channel': note_info['channel'],

                        # Instrument info (NEW!)
                        'instrument_program': track_info.instrument_program,
                        'instrument_family': track_info.instrument_family,
                        'track_name': track_info.track_name,
                    })
                    del active_notes[key]

    return notes


def notes_to_midi_multitrack(
    notes: List[Dict[str, Any]],
    ticks_per_beat: int = 480,
    tempo: int = 500000,
    preserve_instruments: bool = True
) -> mido.MidiFile:
    """
    Convert notes back to MIDI WITH instrument information.

    Enhanced version that preserves:
    - Track structure
    - Instrument programs
    - Track names
    - Channels

    Args:
        notes: List of note dictionaries (from extract_notes_with_instruments)
        ticks_per_beat: MIDI ticks per beat
        tempo: Tempo in microseconds per beat
        preserve_instruments: If True, preserve instrument assignments

    Returns:
        MIDI file with proper multitrack structure
    """
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    # Group notes by track
    tracks_notes = defaultdict(list)
    track_infos = {}  # track_idx -> (program, channel, name)

    for note in notes:
        track_idx = note.get('track', 0)
        tracks_notes[track_idx].append(note)

        # Store track info
        if track_idx not in track_infos and preserve_instruments:
            track_infos[track_idx] = (
                note.get('instrument_program', 0),
                note.get('channel', 0),
                note.get('track_name', f'Track {track_idx}')
            )

    # Create tracks
    for track_idx in sorted(tracks_notes.keys()):
        track = mido.MidiTrack()
        midi.tracks.append(track)

        # Add track name
        if preserve_instruments and track_idx in track_infos:
            _, _, track_name = track_infos[track_idx]
            track.append(mido.MetaMessage('track_name', name=track_name, time=0))

        # Add tempo (only in first track)
        if track_idx == 0:
            track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

        # Add program change (instrument)
        if preserve_instruments and track_idx in track_infos:
            program, channel, _ = track_infos[track_idx]
            track.append(mido.Message(
                'program_change',
                program=program,
                channel=channel,
                time=0
            ))

        # Sort notes by start time
        track_notes = sorted(tracks_notes[track_idx], key=lambda n: n['start_time'])

        # Create note on/off events
        events = []
        for note in track_notes:
            channel = note.get('channel', 0)

            events.append({
                'time': note['start_time'],
                'type': 'note_on',
                'note': note['pitch'],
                'velocity': note['velocity'],
                'channel': channel
            })
            events.append({
                'time': note['start_time'] + note['duration'],
                'type': 'note_off',
                'note': note['pitch'],
                'velocity': 0,
                'channel': channel
            })

        # Sort all events by time
        events.sort(key=lambda e: e['time'])

        # Convert to MIDI messages with delta times
        current_time = 0.0
        for event in events:
            delta_time = event['time'] - current_time
            delta_ticks = mido.second2tick(delta_time, ticks_per_beat, tempo)

            if event['type'] == 'note_on':
                track.append(mido.Message(
                    'note_on',
                    note=event['note'],
                    velocity=event['velocity'],
                    channel=event['channel'],
                    time=int(delta_ticks)
                ))
            else:
                track.append(mido.Message(
                    'note_off',
                    note=event['note'],
                    velocity=0,
                    channel=event['channel'],
                    time=int(delta_ticks)
                ))

            current_time = event['time']

        # End of track
        track.append(mido.MetaMessage('end_of_track', time=0))

    return midi


# ============================================================================
# Orchestration Analysis
# ============================================================================

def analyze_orchestration(notes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze orchestration patterns in multitrack MIDI.

    Returns:
        Dict with orchestration statistics:
        - instrument_counts: Notes per instrument family
        - instrument_density: Temporal density per instrument
        - instrument_ranges: Pitch range per instrument
        - simultaneous_instruments: Which instruments play together
        - instrument_roles: Which instruments are melody/harmony/bass
    """
    # Group notes by instrument
    by_instrument = defaultdict(list)
    for note in notes:
        family = note.get('instrument_family', 'unknown')
        by_instrument[family].append(note)

    # Count notes per instrument
    instrument_counts = {
        family: len(notes_list)
        for family, notes_list in by_instrument.items()
    }

    # Calculate density (notes per second)
    instrument_density = {}
    for family, notes_list in by_instrument.items():
        if notes_list:
            total_time = max(n['start_time'] + n['duration'] for n in notes_list)
            density = len(notes_list) / total_time if total_time > 0 else 0
            instrument_density[family] = density

    # Calculate pitch ranges
    instrument_ranges = {}
    for family, notes_list in by_instrument.items():
        if notes_list:
            pitches = [n['pitch'] for n in notes_list]
            instrument_ranges[family] = {
                'min': min(pitches),
                'max': max(pitches),
                'mean': np.mean(pitches)
            }

    # Find simultaneous instruments (which play at same time)
    # Sample at 0.1s intervals
    if notes:
        max_time = max(n['start_time'] + n['duration'] for n in notes)
        time_points = np.arange(0, max_time, 0.1)

        simultaneous_sets = []
        for t in time_points:
            active = set()
            for note in notes:
                if note['start_time'] <= t < note['start_time'] + note['duration']:
                    active.add(note.get('instrument_family', 'unknown'))
            if len(active) > 1:
                simultaneous_sets.append(frozenset(active))

        # Count frequency of simultaneous instrument combinations
        from collections import Counter
        simultaneous_counts = Counter(simultaneous_sets)
        simultaneous_instruments = dict(simultaneous_counts.most_common(10))
    else:
        simultaneous_instruments = {}

    # Identify roles (melody, harmony, bass)
    instrument_roles = {}
    for family, notes_list in by_instrument.items():
        if not notes_list:
            continue

        pitches = [n['pitch'] for n in notes_list]
        mean_pitch = np.mean(pitches)
        pitch_std = np.std(pitches)

        # Heuristics:
        # - Bass: low pitch, low variance
        # - Melody: mid-high pitch, high variance
        # - Harmony: mid pitch, low variance

        if mean_pitch < 48:  # Below C3
            role = 'bass'
        elif pitch_std > 10:  # High variance
            role = 'melody'
        else:
            role = 'harmony'

        instrument_roles[family] = role

    return {
        'instrument_counts': instrument_counts,
        'instrument_density': instrument_density,
        'instrument_ranges': instrument_ranges,
        'simultaneous_instruments': {
            str(sorted(list(s))): count
            for s, count in simultaneous_instruments.items()
        },
        'instrument_roles': instrument_roles
    }


# ============================================================================
# Instrument-Specific Filtering
# ============================================================================

def filter_notes_by_instrument(
    notes: List[Dict[str, Any]],
    instrument_family: str
) -> List[Dict[str, Any]]:
    """Filter notes to only specified instrument family"""
    return [
        n for n in notes
        if n.get('instrument_family') == instrument_family
    ]


def filter_notes_by_role(
    notes: List[Dict[str, Any]],
    role: str  # 'melody', 'harmony', 'bass'
) -> List[Dict[str, Any]]:
    """
    Filter notes by musical role.

    Heuristic role assignment:
    - melody: Highest notes, most variation
    - harmony: Middle notes, chord tones
    - bass: Lowest notes, less variation
    """
    if not notes:
        return []

    # Group by track
    by_track = defaultdict(list)
    for note in notes:
        by_track[note['track']].append(note)

    # Assign role to each track
    track_roles = {}
    for track_idx, track_notes in by_track.items():
        pitches = [n['pitch'] for n in track_notes]
        mean_pitch = np.mean(pitches)
        pitch_std = np.std(pitches)

        if mean_pitch < 48:
            track_roles[track_idx] = 'bass'
        elif pitch_std > 10:
            track_roles[track_idx] = 'melody'
        else:
            track_roles[track_idx] = 'harmony'

    # Filter by role
    return [
        n for n in notes
        if track_roles.get(n['track']) == role
    ]
