#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Arrangement Pattern Transfer (Base Triads -> Reference Arrangements)
- Properly handles chord voicings, not just single notes
- Supports different rhythmic patterns (block chords, repeated chords, arpeggios)
- Matches chord progressions intelligently using harmonic analysis
- Works with render.py chord generation logic

Author: Claude (Sonnet 4)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import mido
import re

# ---------- PATHS ----------
BASE_DIR = Path("/home/user/app/chord_progressions")
REF_ARP8_DIR = Path("/home/user/app/ReferenceMidi/Arp8th")
REF_ARP16_DIR = Path("/home/user/app/ReferenceMidi/Arp16th")
REF_CHORDS_DIR = Path("/home/user/app/ReferenceMidi/Chords")
REF_BASS_DIR = Path("/home/user/app/ReferenceMidi/Bass")
OUTPUT_DIR = Path("/home/user/app/Outputs")

# ---------- CHORD MAPPING FROM RENDER.PY ----------
C_MAJOR_SCALE = {
    'I': [60, 64, 67],      # C major (C-E-G)
    'ii': [62, 65, 69],     # D minor (D-F-A)
    'ii-': [62, 65, 69],    # D minor (D-F-A)
    'II': [62, 66, 69],     # D major (D-F#-A)
    'iii': [64, 67, 71],    # E minor (E-G-B)
    'iii-': [64, 67, 71],   # E minor (E-G-B)
    'IV': [65, 69, 72],     # F major (F-A-C)
    'iv': [65, 68, 72],     # F minor (F-Ab-C) - borrowed from parallel minor
    'iv-': [65, 68, 72],    # F minor (F-Ab-C) - borrowed from parallel minor
    'V': [67, 71, 74],      # G major (G-B-D)
    'V7': [67, 71, 74, 77], # G dominant 7 (G-B-D-F)
    'v-': [67, 70, 74],     # G minor (G-Bb-D)
    'vi': [69, 72, 76],     # A minor (A-C-E)
    'vi-': [69, 72, 76],    # A minor (A-C-E)
    'vii': [71, 74, 77],    # B diminished (B-D-F)
    'vii-': [71, 74, 77],   # B diminished (B-D-F)
    'i': [60, 63, 67],      # C minor (C-Eb-G)
    'i-': [60, 63, 67],     # C minor (C-Eb-G)
    'bII': [61, 65, 68],    # Db major (Db-F-Ab)
    'bIII': [63, 67, 70],   # Eb major (Eb-G-Bb)
    'biii-': [63, 66, 70],  # Eb minor (Eb-Gb-Bb)
    'III': [64, 68, 71],    # E major (E-G#-B)
    'v': [67, 70, 74],      # G minor (G-Bb-D)
    'v-': [67, 70, 74],     # G minor (G-Bb-D)
    'VI': [69, 73, 76],     # A major (A-C#-E)
    'VII': [71, 75, 78],    # B major (B-D#-F#)
    'bVI': [68, 72, 75],    # Ab major (Ab-C-Eb)
    'bVII': [70, 74, 77],   # Bb major (Bb-D-F)
    'i-7': [60, 63, 67, 70], # C minor 7 (C-Eb-G-Bb)
    'V7/VI': [64, 68, 71, 74], # E dominant 7 (E-G#-B-D) - V7 of vi

    # 9th chord extensions (Root-3rd-5th-7th-9th)
    'I9': [60, 64, 67, 71, 74],    # C major 9 (C-E-G-B-D) - maj7
    'ii9': [62, 65, 69, 72, 76],   # D minor 9 (D-F-A-C-E) - min7
    'ii-9': [62, 65, 69, 72, 76],  # D minor 9 (D-F-A-C-E) - min7
    'II9': [62, 66, 69, 73, 76],   # D major 9 (D-F#-A-C#-E) - maj7
    'iii9': [64, 67, 71, 74, 77],  # E minor 9 (E-G-B-D-F#) - min7
    'iii-9': [64, 67, 71, 74, 77], # E minor 9 (E-G-B-D-F#) - min7
    'IV9': [65, 69, 72, 76, 79],   # F major 9 (F-A-C-E-G) - maj7
    'iv9': [65, 68, 72, 75, 79],   # F minor 9 (F-Ab-C-Eb-G) - min7
    'iv-9': [65, 68, 72, 75, 79],  # F minor 9 (F-Ab-C-Eb-G) - min7
    'V9': [67, 71, 74, 77, 69],    # G dominant 9 (G-B-D-F-A) - dom7
    'V79': [67, 71, 74, 77, 69],   # G dominant 9 (G-B-D-F-A) - dom7
    'v-9': [67, 70, 74, 77, 69],   # G minor 9 (G-Bb-D-F-A) - min7
    'vi9': [69, 72, 76, 79, 71],   # A minor 9 (A-C-E-G-B) - min7
    'vi-9': [69, 72, 76, 79, 71],  # A minor 9 (A-C-E-G-B) - min7
    'vii9': [71, 74, 77, 81, 73],  # B dim 9 (B-D-F-Ab-C#) - dim7
    'vii-9': [71, 74, 77, 81, 73], # B dim 9 (B-D-F-Ab-C#) - dim7
    'bVI9': [68, 72, 75, 79, 70],  # Ab major 9 (Ab-C-Eb-G-Bb) - maj7
    'bVII9': [70, 74, 77, 81, 72], # Bb major 9 (Bb-D-F-A-C) - maj7
    'V7/VI9': [64, 68, 71, 74, 78] # E dominant 9 (E-G#-B-D-F#) - dom7
}

# C Minor scale chord mappings for minor key progressions
C_MINOR_SCALE = {
    'i': [60, 63, 67],      # C minor (C-Eb-G)
    'i-': [60, 63, 67],     # C minor (C-Eb-G)
    'i-7': [60, 63, 67, 70], # C minor 7 (C-Eb-G-Bb)
    'i-9': [60, 63, 67, 70, 74], # C minor 9 (C-Eb-G-Bb-D)
    'ii': [62, 65, 69],     # D diminished (D-F-Ab) - actually D half-dim in natural minor
    'ii-': [62, 65, 68],    # D diminished (D-F-Ab)
    'ii-7': [62, 65, 68, 72], # D half-diminished 7 (D-F-Ab-C)
    'ii-9': [62, 65, 68, 72, 76], # D half-diminished 9 (D-F-Ab-C-E)
    'bII': [61, 65, 68],    # Db major (Db-F-Ab)
    'bIII': [63, 67, 70],   # Eb major (Eb-G-Bb)
    'biii-': [63, 66, 70],  # Eb minor (Eb-Gb-Bb)
    'biii-7': [63, 66, 70, 74], # Eb minor 7 (Eb-Gb-Bb-D)
    'biii-9': [63, 66, 70, 74, 77], # Eb minor 9 (Eb-Gb-Bb-D-F)
    'III': [64, 68, 71],    # E major (E-G#-B)
    'III7': [64, 68, 71, 75], # E major 7 (E-G#-B-D#)
    'III9': [64, 68, 71, 75, 78], # E major 9 (E-G#-B-D#-F#)
    'iv': [65, 68, 72],     # F minor (F-Ab-C)
    'iv-': [65, 68, 72],    # F minor (F-Ab-C)
    'IV': [65, 69, 72],     # F major (F-A-C)
    'v': [67, 70, 74],      # G minor (G-Bb-D)
    'v-': [67, 70, 74],     # G minor (G-Bb-D)
    'V': [67, 71, 74],      # G major (G-B-D)
    'V7': [67, 71, 74, 77], # G dominant 7 (G-B-D-F)
    'vi': [68, 72, 75],     # Ab major (Ab-C-Eb) - bVI in minor
    'vi-': [68, 72, 75],    # A minor (A-C-Eb) - rare in minor key
    'vi-7': [68, 72, 75, 79], # A minor 7 (A-C-Eb-G) - rare in minor key
    'vi-9': [68, 72, 75, 79, 70], # A minor 9 (A-C-Eb-G-Bb) - rare in minor key
    'bVI': [68, 72, 75],    # Ab major (Ab-C-Eb)
    'bVI7': [68, 72, 75, 79], # Ab major 7 (Ab-C-Eb-G)
    'VI': [69, 73, 76],     # A major (A-C#-E)
    'vii': [70, 74, 77],    # Bb major (Bb-D-F) - bVII in minor
    'bVII': [70, 74, 77],   # Bb major (Bb-D-F)
    'bVII7': [70, 74, 77, 81], # Bb major 7 (Bb-D-F-A)
    'bVII9': [70, 74, 77, 81, 72], # Bb major 9 (Bb-D-F-A-C)
    'VII': [71, 75, 78],    # B major (B-D#-F#)
    'iv-7': [65, 68, 72, 75], # F minor 7 (F-Ab-C-Eb)
    'iv-9': [65, 68, 72, 75, 79], # F minor 9 (F-Ab-C-Eb-G)
    'v-7': [67, 70, 74, 77], # G minor 7 (G-Bb-D-F)
    'v-9': [67, 70, 74, 77, 69], # G minor 9 (G-Bb-D-F-A)
    'bIII7': [63, 67, 70, 74], # Eb major 7 (Eb-G-Bb-D)
    'bIII9': [63, 67, 70, 74, 77], # Eb major 9 (Eb-G-Bb-D-F)
    'bVI9': [68, 72, 75, 79, 70], # Ab major 9 (Ab-C-Eb-G-Bb)
}

# Map MIDI note sets back to chord symbols for pattern recognition
NOTES_TO_CHORD = {}
for chord, notes in C_MAJOR_SCALE.items():
    frozen_notes = frozenset(notes)
    NOTES_TO_CHORD[frozen_notes] = chord

# Add minor scale mappings (they may overlap with major, but that's OK)
for chord, notes in C_MINOR_SCALE.items():
    frozen_notes = frozenset(notes)
    NOTES_TO_CHORD[frozen_notes] = chord

# ---------- MIDI UTILS ----------
def get_first_tempo(mid: mido.MidiFile) -> int:
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo":
                return msg.tempo
    return 500000  # default 120bpm

def ticks_to_beats(ticks: int, ppq: int) -> float:
    return ticks / ppq

def beats_to_ticks(beats: float, ppq: int) -> int:
    return int(round(beats * ppq))

def collect_note_events(mid: mido.MidiFile) -> List[Dict[str, Any]]:
    """Return list of note events with absolute ticks."""
    abs_msgs: List[Tuple[int, mido.Message]] = []
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            abs_msgs.append((t, msg))
    abs_msgs.sort(key=lambda x: x[0])

    active: Dict[Tuple[int, int], Tuple[int, int]] = {}
    events: List[Dict[str, Any]] = []
    for at, msg in abs_msgs:
        if msg.type == "note_on" and msg.velocity > 0:
            ch = getattr(msg, "channel", 0)
            active[(ch, msg.note)] = (at, msg.velocity)
        elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
            ch = getattr(msg, "channel", 0)
            key = (ch, msg.note)
            if key in active:
                st, vel = active.pop(key)
                events.append({"start": st, "end": at, "pitch": msg.note, "velocity": vel, "channel": ch})
    return events

def group_simultaneous_notes(events: List[Dict[str, Any]], tolerance_beats: float = 0.1) -> List[Dict[str, Any]]:
    """Group notes that start at nearly the same time into chord events."""
    if not events:
        return []

    # Sort by start time
    events = sorted(events, key=lambda e: e["start"])

    grouped = []
    current_group = [events[0]]

    for i in range(1, len(events)):
        curr_event = events[i]
        prev_start_beats = current_group[0]["start"] / 480  # Assume 480 PPQ
        curr_start_beats = curr_event["start"] / 480

        if abs(curr_start_beats - prev_start_beats) <= tolerance_beats:
            current_group.append(curr_event)
        else:
            # Process current group
            if len(current_group) == 1:
                grouped.append(current_group[0])
            else:
                # Create chord event
                chord_event = {
                    "start": current_group[0]["start"],
                    "end": max(e["end"] for e in current_group),
                    "pitches": [e["pitch"] for e in current_group],
                    "velocity": int(sum(e["velocity"] for e in current_group) / len(current_group)),
                    "channel": current_group[0]["channel"],
                    "is_chord": True
                }
                grouped.append(chord_event)
            current_group = [curr_event]

    # Process final group
    if len(current_group) == 1:
        grouped.append(current_group[0])
    else:
        chord_event = {
            "start": current_group[0]["start"],
            "end": max(e["end"] for e in current_group),
            "pitches": [e["pitch"] for e in current_group],
            "velocity": int(sum(e["velocity"] for e in current_group) / len(current_group)),
            "channel": current_group[0]["channel"],
            "is_chord": True
        }
        grouped.append(chord_event)

    return grouped

def split_by_bars(events: List[Dict[str, Any]], ppq: int, beats_per_bar: float = 4.0) -> Dict[int, List[Dict[str, Any]]]:
    by_bar: Dict[int, List[Dict[str, Any]]] = {}
    for e in events:
        bar = int(ticks_to_beats(e["start"], ppq) // beats_per_bar) + 1
        by_bar.setdefault(bar, []).append(e)
    return by_bar

# ---------- CHORD RECOGNITION ----------
def identify_chord_from_notes(notes: List[int]) -> Optional[str]:
    """Identify chord symbol from MIDI notes."""
    note_set = frozenset(notes)
    return NOTES_TO_CHORD.get(note_set)

def detect_chord_quality_from_notes(notes: List[int]) -> Tuple[int, str]:
    """Detect chord quality (major/minor/dim) from actual MIDI notes."""
    if len(notes) < 3:
        # Not enough notes to determine quality, assume major
        return notes[0] if notes else 60, "major"

    # Sort notes and get unique note classes
    unique_notes = sorted(set(note % 12 for note in notes))

    # Use lowest note as root
    root = min(notes) % 12

    # Calculate intervals from root
    intervals = []
    for note_class in unique_notes:
        if note_class != root:
            interval = (note_class - root) % 12
            intervals.append(interval)

    # Determine quality based on intervals
    if 3 in intervals:  # Minor third
        if 6 in intervals:  # Diminished fifth
            quality = "diminished"
        else:
            quality = "minor"
    elif 4 in intervals:  # Major third
        quality = "major"
    else:
        # Ambiguous or other - default to major
        quality = "major"

    return min(notes), quality

def generate_chord_from_root_and_quality(root_midi: int, quality: str, extensions: List[str] = None) -> List[int]:
    """Generate chord notes from root, quality, and optional extensions."""
    root_class = root_midi % 12
    octave = root_midi // 12

    if extensions is None:
        extensions = []

    # Base triad intervals
    if quality == "major":
        intervals = [0, 4, 7]  # Root, major 3rd, perfect 5th
    elif quality == "minor":
        intervals = [0, 3, 7]  # Root, minor 3rd, perfect 5th
    elif quality == "diminished":
        intervals = [0, 3, 6]  # Root, minor 3rd, diminished 5th
    else:
        # Default to major
        intervals = [0, 4, 7]

    # Add extensions
    for ext in extensions:
        if ext == "7":
            if quality == "major":
                intervals.append(11)  # Major 7th
            else:
                intervals.append(10)  # Minor/dominant 7th
        elif ext == "9":
            # Add 7th first if not already present
            if "7" not in extensions:
                if quality == "major":
                    intervals.append(11)  # Major 7th
                else:
                    intervals.append(10)  # Minor/dominant 7th
            intervals.append(14)  # 9th (2nd octave)

    # Generate MIDI notes in the same octave as root
    chord_notes = []
    for interval in intervals:
        note_class = (root_class + interval) % 12
        # Calculate which octave this note should be in
        octave_offset = interval // 12
        note_midi = (octave + octave_offset) * 12 + note_class
        # If the note would be lower than root, move it up an octave
        if note_midi < root_midi:
            note_midi += 12
        chord_notes.append(note_midi)

    return sorted(chord_notes)

def detect_chord_extensions_from_pattern_name(pattern_name: str) -> List[str]:
    """Detect required chord extensions from pattern name."""
    extensions = []
    pattern_upper = pattern_name.upper()

    if "9" in pattern_upper:
        extensions.append("9")
    elif "7" in pattern_upper:
        extensions.append("7")

    return extensions

def get_chord_root_midi(chord_symbol: str) -> int:
    """Extract just the MIDI root note from a chord symbol."""
    # Map chord symbols to their root MIDI notes (in C)
    root_map = {
        "i": 60, "i-": 60, "I": 60,
        "ii": 62, "ii-": 62, "II": 62,
        "iii": 64, "iii-": 64, "III": 64,
        "iv": 65, "iv-": 65, "IV": 65,
        "v": 67, "v-": 67, "V": 67,
        "vi": 69, "vi-": 69, "VI": 69,
        "vii": 71, "vii-": 71, "VII": 71,
        "bII": 61, "bIII": 63, "bVI": 68, "bVII": 70,
        # Extended chords (7th and 9th) - same root as base chord
        "i-7": 60, "i-9": 60, "ii-7": 62, "ii-9": 62,
        "iv-7": 65, "iv-9": 65, "v-7": 67, "v-9": 67
    }

    # First try direct lookup (preserves extensions)
    if chord_symbol in root_map:
        return root_map[chord_symbol]

    # If not found, clean the chord symbol for lookup
    clean_symbol = chord_symbol.replace("7", "").replace("9", "")
    return root_map.get(clean_symbol, 60)  # Default to C if not found

def get_chord_root_and_quality(chord_symbol: str, chord_scale: dict = None) -> Tuple[int, str]:
    """Get root note and quality from chord symbol."""
    if chord_scale is None:
        chord_scale = C_MAJOR_SCALE

    # First try to find the chord symbol directly (for extended chords like i-7, i-9)
    chord_info = chord_scale.get(chord_symbol)

    # If not found, try both major and minor scales
    if chord_info is None:
        chord_info = C_MINOR_SCALE.get(chord_symbol)

    # If still not found, fallback to default
    if chord_info is None:
        chord_info = [60, 64, 67]  # Default C major triad

    root = chord_info[0]

    # Determine quality from chord symbol - be precise about case sensitivity
    if "9" in chord_symbol:
        quality = "9"  # 9th chord (Root-3rd-5th-7th-9th)
    elif "7" in chord_symbol:
        quality = "7"  # 7th chord
    elif chord_symbol in ['ii', 'iii', 'vi', 'vii', 'ii-', 'iii-', 'vi-', 'vii-', 'iv', 'iv-', 'v-', 'i', 'i-', 'v', 'biii-']:
        quality = "min"
    elif len(chord_info) >= 5:
        quality = "9"  # 9th chord (Root-3rd-5th-7th-9th)
    elif len(chord_info) >= 4:
        quality = "7"  # 7th chord
    else:
        quality = "maj"

    return root, quality

# ---------- PATTERN ANALYSIS ----------
class PatternEvent:
    def __init__(self, time_beats: float, chord_tones: List[str], midi_notes: List[int],
                 velocity: int, duration_beats: float, is_chord: bool = False,
                 reference_quality: str = "major"):
        self.time_beats = time_beats
        self.chord_tones = chord_tones  # ['R', '3', '5'] etc
        self.midi_notes = midi_notes    # Original MIDI notes from reference
        self.velocity = velocity
        self.duration_beats = duration_beats
        self.is_chord = is_chord
        self.reference_quality = reference_quality  # Detected quality from reference MIDI

def analyze_base_progression(base_mid: mido.MidiFile) -> Dict[int, str]:
    """Analyze base progression to identify chord per bar."""
    ppq = base_mid.ticks_per_beat
    evs = collect_note_events(base_mid)
    grouped_evs = group_simultaneous_notes(evs)
    bars = split_by_bars(grouped_evs, ppq)

    progression: Dict[int, str] = {}
    for bar, events in bars.items():
        if not events:
            continue

        # Find the chord event (should be one per bar in our base progressions)
        chord_event = None
        for event in events:
            if event.get("is_chord", False):
                chord_event = event
                break

        if chord_event:
            notes = chord_event["pitches"]
            chord_symbol = identify_chord_from_notes(notes)
            if chord_symbol:
                progression[bar] = chord_symbol

    return progression

def chord_tone_label(note: int, root: int, quality: str, chord_symbol: str) -> str:
    """Label a note as a chord tone relative to root with proper restrictions."""
    interval = (note - root) % 12

    # Basic chord tones available to all chords
    if interval == 0: return "R"
    elif interval == 4 and quality in ["maj", "7", "9"]: return "3"
    elif interval == 3 and quality in ["min", "7", "9"]: return "b3"  # Fixed: allow b3 for 7th and 9th chords
    elif interval == 7: return "5"
    elif interval == 6: return "b5"

    # 7th available for appropriate chords (including minor chords)
    elif interval == 10: return "b7"
    elif interval == 11: return "M7"

    # 9th available for extended chords (including minor chords)
    elif interval == 2 and (quality == "9" or chord_symbol.endswith("9") or chord_symbol.endswith("-9")): return "9"

    else:
        # Find closest allowed chord tone based on chord type
        if quality == "9" or chord_symbol.endswith("9") or chord_symbol.endswith("-9"):
            candidates = [0, 2, 3, 4, 7, 10, 11]  # R, 9, b3, 3, 5, b7, M7
        elif quality == "7" or chord_symbol.endswith("7") or chord_symbol.endswith("-7"):
            candidates = [0, 3, 4, 7, 10, 11]  # R, b3, 3, 5, b7, M7
        elif "/" in chord_symbol:  # Secondary dominants
            candidates = [0, 4, 7, 10]  # R, 3, 5, b7
        elif chord_symbol.endswith("-"):  # Minor chords - allow 7ths and 9ths
            # Check if it's an extended minor chord
            if "9" in chord_symbol:
                candidates = [0, 2, 3, 7, 10]  # R, 9, b3, 5, b7
            elif "7" in chord_symbol:
                candidates = [0, 3, 7, 10]  # R, b3, 5, b7
            else:
                candidates = [0, 3, 7]  # R, b3, 5 (basic triad)
        else:  # Triads and other chords
            candidates = [0, 3, 4, 7]  # R, b3, 3, 5

        closest = min(candidates, key=lambda x: min(abs(interval - x), abs(interval - x + 12), abs(interval - x - 12)))
        if closest == 0: return "R"
        elif closest == 2: return "9"
        elif closest == 3: return "b3"
        elif closest == 4: return "3"
        elif closest == 7: return "5"
        elif closest == 10: return "b7"
        elif closest == 11: return "M7"
    return "R"  # fallback

def analyze_reference_pattern(ref_mid: mido.MidiFile, base_progression: Dict[int, str], chord_scale: dict = None) -> Dict[int, List[PatternEvent]]:
    """Analyze reference MIDI to extract abstract patterns."""
    if chord_scale is None:
        chord_scale = C_MAJOR_SCALE
    ppq = ref_mid.ticks_per_beat
    evs = collect_note_events(ref_mid)
    grouped_evs = group_simultaneous_notes(evs)
    bars = split_by_bars(grouped_evs, ppq)

    patterns: Dict[int, List[PatternEvent]] = {}

    # Handle different bar ranges - some files start at bar 5-8
    available_bars = sorted(bars.keys())
    if not available_bars:
        return patterns

    # Map reference bars to base progression bars (1-4)
    if available_bars[0] >= 5:
        # This is a "repeat" pattern (bars 5-8 -> map to 1-4)
        bar_mapping = {bar: ((bar - 5) % 4) + 1 for bar in available_bars}
    else:
        # Normal pattern (bars 1-4)
        bar_mapping = {bar: bar for bar in available_bars}

    for ref_bar, events in bars.items():
        target_bar = bar_mapping.get(ref_bar)
        if target_bar not in base_progression or not events:
            continue

        # Analyze the actual MIDI content to determine chord quality
        chord_notes = []
        for event in events:
            if event.get("is_chord", False):
                chord_notes.extend(event["pitches"])
            else:
                pitch = event.get("pitch")
                if pitch is not None:
                    chord_notes.append(pitch)

        # Detect actual chord quality from the reference MIDI
        if chord_notes:
            implicit_root_midi, detected_quality = detect_chord_quality_from_notes(chord_notes)
            implicit_root = implicit_root_midi % 12
        else:
            # Fallback to using the progression-based approach
            chord_symbol = base_progression[target_bar]
            implicit_root_midi, detected_quality = get_chord_root_and_quality(chord_symbol, chord_scale)
            implicit_root = implicit_root_midi % 12

        bar_patterns = []
        bar_start_beats = (ref_bar - 1) * 4.0

        for event in sorted(events, key=lambda x: x["start"]):
            start_beats = ticks_to_beats(event["start"], ppq)
            dur_beats = ticks_to_beats(event["end"] - event["start"], ppq)
            time_in_bar = start_beats - bar_start_beats

            if event.get("is_chord", False):
                # Handle chord events
                notes = event["pitches"]
                chord_tones = []

                for note in notes:
                    tone = chord_tone_label(note, implicit_root_midi, "maj", "")  # Use implicit root, assume major for now
                    chord_tones.append(tone)

                pattern_event = PatternEvent(
                    time_beats=time_in_bar,
                    chord_tones=chord_tones,
                    midi_notes=notes,  # Store original MIDI notes
                    velocity=event["velocity"],
                    duration_beats=dur_beats,
                    is_chord=True,
                    reference_quality=detected_quality
                )
                bar_patterns.append(pattern_event)
            else:
                # Handle single note events
                note = event["pitch"]
                tone = chord_tone_label(note, implicit_root_midi, detected_quality, "")

                pattern_event = PatternEvent(
                    time_beats=time_in_bar,
                    chord_tones=[tone],
                    midi_notes=[note],  # Store original MIDI note
                    velocity=event["velocity"],
                    duration_beats=dur_beats,
                    is_chord=False,
                    reference_quality=detected_quality
                )
                bar_patterns.append(pattern_event)

        patterns[target_bar] = bar_patterns

    return patterns

# ---------- SYNTHESIS ----------
def transpose_reference_note(ref_note: int, ref_root: int, target_root: int, target_chord: str = "", chord_scale: dict = None) -> int:
    """Transpose a reference note to match target chord while preserving octave relationship."""
    # Calculate the interval from reference root
    interval = ref_note - ref_root

    # Apply same interval to target root
    transposed_note = target_root + interval

    # Global fix for B natural -> Bb in any minor chord context
    if target_chord and target_chord.endswith("-") and (transposed_note % 12) == 11:
        transposed_note -= 1  # Convert B natural to Bb

    return transposed_note

def get_bass_root_note(chord_symbol: str) -> int:
    """Get the bass root note for a chord, typically in the bass range (C2-C4)."""
    root, _ = get_chord_root_and_quality(chord_symbol)

    # Map to bass range (C2 = 36, C4 = 60)
    # Start with C3 (48) as default bass register
    bass_root = root + 48

    # Ensure we're in a reasonable bass range (36-60)
    while bass_root < 36:
        bass_root += 12
    while bass_root > 60:
        bass_root -= 12

    return bass_root

def get_bass_chord_tones(chord_symbol: str, bass_root: int) -> List[int]:
    """Get available chord tones for bass playing in appropriate range."""
    _, quality = get_chord_root_and_quality(chord_symbol)

    # Available chord tones for bass (typically root, 3rd, 5th, and sometimes 7th)
    tones = []

    # Root
    tones.append(bass_root)

    # Third
    if quality in ["maj", "7", "9"]:
        tones.append(bass_root + 4)  # Major third
    elif quality in ["min", "9"]:
        tones.append(bass_root + 3)  # Minor third

    # Fifth
    tones.append(bass_root + 7)

    # Seventh (for extended chords)
    if "7" in chord_symbol or "9" in chord_symbol:
        if quality in ["7"]:
            tones.append(bass_root + 10)  # Dominant 7th
        else:
            tones.append(bass_root + 11)  # Major 7th

    # Keep notes in bass range (filter out notes too high)
    tones = [note for note in tones if note <= 60]

    return tones

def get_chromatic_approach_notes(target_note: int, direction: str = "below") -> List[int]:
    """Get chromatic approach notes leading to a target note."""
    approaches = []

    if direction == "below":
        # Chromatic walkup (typically 1-2 semitones below)
        approaches.extend([target_note - 2, target_note - 1])
    elif direction == "above":
        # Chromatic walkdown (typically 1-2 semitones above)
        approaches.extend([target_note + 1, target_note + 2])
    elif direction == "both":
        # Both directions
        approaches.extend([target_note - 2, target_note - 1, target_note + 1, target_note + 2])

    # Filter to keep in bass range
    approaches = [note for note in approaches if 36 <= note <= 60]

    return approaches

def realize_chord_tone(tone: str, root: int, ref_note: int, target_chord: str = "", chord_scale: dict = None) -> int:
    """Convert chord tone label back to MIDI note, adapting to target chord quality."""
    tone_map = {
        "R": 0, "3": 4, "b3": 3, "5": 7, "b5": 6,
        "b7": 10, "M7": 11, "9": 2, "b9": 1
    }


    # Adapt chord tone based on target chord quality
    target_root, target_quality = get_chord_root_and_quality(target_chord, chord_scale)

    # Convert chord tones based on target chord quality and context
    # Check if the reference note matches the target root
    target_note_class = target_root % 12
    ref_note_class = ref_note % 12

    # Allow 9ths on minor chords - removed restriction
    if tone == "M7" and target_chord.startswith("V"):
        # All V chords (V, V7, V9, V7/VI) should use dominant 7th (b7), not major 7th (M7)
        adapted_tone = "b7"
    elif tone == "b7" and target_quality == "maj" and not target_chord.startswith("V"):
        # Minor 7th in reference becomes major 7th in major chord (but never on V chords)
        adapted_tone = "M7"
    elif tone == "M7" and target_quality == "maj":
        # Major 7th in reference stays major 7th in major chord (diatonic)
        adapted_tone = "M7"
    elif tone == "M7" and target_quality == "min" and target_chord.endswith("-"):
        # Only convert M7 to b7 for explicitly minor chords (ending with -)
        # Major chords in minor keys should keep their diatonic 7ths
        adapted_tone = "b7"
    elif tone == "b3" and target_quality == "maj":
        # Minor 3rd in reference becomes major 3rd in major chord
        adapted_tone = "3"
    elif tone == "3" and (target_quality == "min" or (target_quality in ["7", "9"] and ("-" in target_chord or target_chord in ["i", "ii", "iii", "vi", "vii", "iv", "v", "i-", "ii-", "iii-", "iv-", "v-", "vi-", "vii-"]))):
        # Major 3rd in reference becomes minor 3rd in minor chord (including minor 7th and 9th chords)
        adapted_tone = "b3"
    elif ref_note_class == target_note_class:
        # The reference note is the same as the target chord root - make it root
        adapted_tone = "R"
    else:
        adapted_tone = tone
    if adapted_tone not in tone_map:
        adapted_tone = "R"  # fallback

    base_interval = tone_map[adapted_tone]
    target_note = root + base_interval

    if adapted_tone == "9":  # 9th is an octave higher
        target_note += 12

    # Special fix for B natural -> Bb in minor chords
    if target_quality == "min" and (target_note % 12) == 11:  # B natural
        # Convert B natural to Bb (subtract 1 semitone)
        target_note -= 1

    # Find the octave that puts the note closest to the reference note
    best_note = target_note
    best_distance = abs(target_note - ref_note)

    # Try different octaves
    for octave_offset in [-2, -1, 1, 2]:
        candidate = target_note + (octave_offset * 12)
        if 0 <= candidate <= 127:  # Valid MIDI range
            distance = abs(candidate - ref_note)
            if distance < best_distance:
                best_distance = distance
                best_note = candidate


    return best_note

def constrain_octave_leap(new_note: int, prev_note: int, max_leap: int = 7) -> int:
    """Constrain octave to avoid leaps larger than max_leap semitones."""
    if prev_note is None:
        return new_note

    leap = abs(new_note - prev_note)
    if leap <= max_leap:
        return new_note

    # Try different octaves to minimize leap
    best_note = new_note
    best_leap = leap

    # Try octaves up and down
    for octave_shift in [-1, 1, -2, 2]:
        candidate = new_note + (octave_shift * 12)
        candidate_leap = abs(candidate - prev_note)
        if candidate_leap < best_leap and candidate_leap <= max_leap:
            best_note = candidate
            best_leap = candidate_leap

    return best_note

def write_arrangement(base_progression: Dict[int, str],
                      patterns: Dict[int, List[PatternEvent]],
                      out_path: Path,
                      tempo: int = 500000,
                      channel: int = 0,
                      reference_progression: Dict[int, str] = None,
                      use_correct_voicings: bool = True,
                      pattern_name: str = "",
                      folder_type: str = "",
                      mode: str = "major"):
    """Write arrangement MIDI file."""
    # Select the appropriate chord scale based on mode
    chord_scale = C_MINOR_SCALE if mode == "minor" else C_MAJOR_SCALE

    ppq = 480

    out = mido.MidiFile(ticks_per_beat=ppq)
    tr = mido.MidiTrack()
    out.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    tr.append(mido.MetaMessage("time_signature", numerator=4, denominator=4,
                               clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

    curr_abs = 0
    def emit(abs_ticks: int, msg: mido.Message):
        nonlocal curr_abs
        delta = abs_ticks - curr_abs
        msg.time = max(0, delta)
        tr.append(msg)
        curr_abs = abs_ticks

    max_bar = max(patterns.keys()) if patterns else 0
    last_note = None  # Track last note for octave smoothing
    current_bar = None  # Track current bar for leap detection

    for bar in range(1, max_bar + 1):
        if bar not in base_progression or bar not in patterns:
            continue

        chord_symbol = base_progression[bar]

        bar_start_beats = (bar - 1) * 4.0

        for pattern_event in patterns[bar]:
            start_beats = bar_start_beats + pattern_event.time_beats
            start_ticks = beats_to_ticks(start_beats, ppq)
            dur_ticks = beats_to_ticks(pattern_event.duration_beats, ppq)

            if pattern_event.is_chord:
                # Detect required chord extensions from pattern name
                required_extensions = detect_chord_extensions_from_pattern_name(pattern_name)

                # Use chord-quality-aware mapping from the chord scale
                notes = []
                # Get the target chord notes from the chord scale (this contains the correct quality)
                if chord_symbol in chord_scale and not required_extensions:
                    # Use basic chord scale mapping for triads
                    target_chord_notes = chord_scale[chord_symbol]
                    # Use the target chord notes, extending to higher octaves if needed
                    base_octave_notes = target_chord_notes[:]

                    # If the pattern has more notes than the basic chord, extend to higher octaves
                    while len(notes) < len(pattern_event.midi_notes):
                        for note in base_octave_notes:
                            if len(notes) >= len(pattern_event.midi_notes):
                                break
                            # Add notes in higher octaves to match pattern length
                            octave_offset = (len(notes) // len(base_octave_notes)) * 12
                            notes.append(note + octave_offset)
                else:
                    # Generate chord with extensions
                    target_root_midi = get_chord_root_midi(chord_symbol)

                    # Determine target chord quality based on chord symbol
                    if '-' in chord_symbol or chord_symbol.lower().startswith('i'):
                        target_quality = 'minor'
                    elif 'dim' in chord_symbol.lower():
                        target_quality = 'diminished'
                    else:
                        target_quality = 'major'

                    # Generate target chord with correct quality and extensions
                    target_chord_notes = generate_chord_from_root_and_quality(target_root_midi, target_quality, required_extensions)

                    # Match the pattern length by extending chord
                    notes = target_chord_notes[:]
                    while len(notes) < len(pattern_event.midi_notes):
                        for base_note in target_chord_notes:
                            if len(notes) >= len(pattern_event.midi_notes):
                                break
                            octave_offset = ((len(notes) - len(target_chord_notes)) // len(target_chord_notes) + 1) * 12
                            notes.append(base_note + octave_offset)

                # Apply octave smoothing for bass patterns
                if folder_type == "Bass" and last_note is not None:
                    notes = [constrain_octave_leap(note, last_note, max_leap=12) for note in notes]

                # For non-Bass patterns, apply octave smoothing
                elif last_note is not None:
                    notes = [constrain_octave_leap(note, last_note, max_leap=24) for note in notes]

                # Emit MIDI events for the chord notes - all note_on messages first
                for note in notes:
                    emit(start_ticks, mido.Message("note_on", channel=channel, note=note, velocity=pattern_event.velocity))

                # Then all note_off messages at the same end time
                for note in notes:
                    emit(start_ticks + dur_ticks, mido.Message("note_off", channel=channel, note=note, velocity=0))

                # Update last note for octave smoothing
                if notes:
                    last_note = notes[-1]

            else:
                # Handle single note events - map to target chord tones
                if len(pattern_event.midi_notes) > 0:
                    ref_note = pattern_event.midi_notes[0]

                    # Get reference chord info for mapping
                    if reference_progression and bar in reference_progression:
                        ref_chord_symbol = reference_progression[bar]
                    else:
                        ref_chord_symbol = base_progression[bar]
                    ref_root_midi = get_chord_root_midi(ref_chord_symbol)

                    # Calculate which chord tone this was in the reference
                    interval = (ref_note - ref_root_midi) % 12
                    octave_offset = ref_note - ref_root_midi - interval

                    # Check if this pattern requires chord extensions
                    required_extensions = detect_chord_extensions_from_pattern_name(pattern_name)

                    # Map to target chord tone - use chord scale only for basic triads
                    if chord_symbol in chord_scale and not required_extensions:
                        target_chord_notes = chord_scale[chord_symbol]

                        # Map reference intervals to target chord tones
                        if interval == 0:  # Root
                            target_note = target_chord_notes[0] + octave_offset
                        elif interval == 3 or interval == 4:  # 3rd (minor or major)
                            if len(target_chord_notes) > 1:
                                target_note = target_chord_notes[1] + octave_offset
                            else:
                                target_note = target_chord_notes[0] + octave_offset
                        elif interval == 7:  # 5th
                            if len(target_chord_notes) > 2:
                                target_note = target_chord_notes[2] + octave_offset
                            else:
                                target_note = target_chord_notes[0] + octave_offset
                        else:
                            # For other intervals, use proportional mapping
                            chord_tone_index = min(interval // 3, len(target_chord_notes) - 1)
                            target_note = target_chord_notes[chord_tone_index] + octave_offset
                    else:
                        # Fallback: generate target chord with correct quality based on symbol
                        target_root_midi = get_chord_root_midi(chord_symbol)

                        # Determine target chord quality based on chord symbol
                        if '-' in chord_symbol or chord_symbol.lower().startswith('i'):
                            target_quality = 'minor'
                        elif 'dim' in chord_symbol.lower():
                            target_quality = 'diminished'
                        else:
                            target_quality = 'major'

                        # Generate target chord with correct quality and extensions
                        required_extensions = detect_chord_extensions_from_pattern_name(pattern_name)
                        target_chord_notes = generate_chord_from_root_and_quality(target_root_midi, target_quality, required_extensions)

                        # Map to appropriate chord tone
                        if interval == 0:  # Root
                            target_note = target_chord_notes[0] + octave_offset
                        elif interval == 3 or interval == 4:  # 3rd (minor or major)
                            if len(target_chord_notes) > 1:
                                target_note = target_chord_notes[1] + octave_offset
                            else:
                                target_note = target_chord_notes[0] + octave_offset
                        elif interval == 7:  # 5th
                            if len(target_chord_notes) > 2:
                                target_note = target_chord_notes[2] + octave_offset
                            else:
                                target_note = target_chord_notes[0] + octave_offset
                        elif interval == 10:  # Minor 7th
                            if len(target_chord_notes) > 3:
                                target_note = target_chord_notes[3] + octave_offset
                            else:
                                # Add minor 7th to the chord
                                target_note = target_root_midi + 10 + octave_offset
                        elif interval == 11:  # Major 7th
                            if len(target_chord_notes) > 3:
                                target_note = target_chord_notes[3] + octave_offset
                            else:
                                # Add major 7th to the chord
                                target_note = target_root_midi + 11 + octave_offset
                        elif interval == 14 or interval == 2:  # 9th (could be 2nd or 14th)
                            if len(target_chord_notes) > 4:
                                target_note = target_chord_notes[4] + octave_offset
                            else:
                                # Add 9th to the chord (2nd + octave)
                                ninth_note = target_root_midi + 2 + octave_offset
                                # If it's lower than the reference, add an octave
                                if ninth_note < ref_note - 6:  # Allow some flexibility
                                    ninth_note += 12
                                target_note = ninth_note
                        else:
                            # For other intervals, map proportionally
                            chord_tone_index = min(interval // 3, len(target_chord_notes) - 1)
                            target_note = target_chord_notes[chord_tone_index] + octave_offset
                else:
                    # Fallback to target root
                    target_note = get_chord_root_midi(chord_symbol)

                # Apply octave smoothing
                if last_note is not None:
                    target_note = constrain_octave_leap(target_note, last_note, max_leap=12)

                # Emit MIDI events
                emit(start_ticks, mido.Message("note_on", channel=channel, note=target_note, velocity=pattern_event.velocity))
                emit(start_ticks + dur_ticks, mido.Message("note_off", channel=channel, note=target_note, velocity=0))

                # Update last note for octave smoothing
                last_note = target_note

    out.save(out_path)

# ---------- DISCOVERY ----------
def discover_reference_files() -> Dict[str, Tuple[Path, str]]:
    """Find all reference .mid files with their source folder."""
    refs: Dict[str, Tuple[Path, str]] = {}
    for p in REF_ARP8_DIR.glob("*.mid"):
        refs[p.stem] = (p, "Arp8th")
    for p in REF_ARP16_DIR.glob("*.mid"):
        refs[p.stem] = (p, "Arp16th")
    for p in REF_CHORDS_DIR.glob("*.mid"):
        refs[p.stem] = (p, "Chords")
    for p in REF_BASS_DIR.glob("*.mid"):
        refs[p.stem] = (p, "Bass")
    print(f"Found {len(refs)} reference files: {list(refs.keys())}")
    return refs

# ---------- MAIN ----------
def parse_progs_txt(progs_path: Path) -> Dict[int, Dict[int, str]]:
    """Parse progs.txt to extract chord progressions."""
    progressions = {}

    with open(progs_path, 'r') as f:
        lines = f.readlines()

    prog_num = 1
    for line in lines:
        line = line.strip()
        if line and '||:' in line and ':||' in line:
            # Extract content between ||: and :||
            match = re.search(r'\|\|:\s*(.*?)\s*:\|\|', line)
            if match:
                content = match.group(1)
                # Split by | and clean up
                chords = [chord.strip() for chord in content.split('|') if chord.strip()]

                # Create bar-to-chord mapping
                progression = {}
                for i, chord in enumerate(chords, 1):
                    # Handle repeat symbol '%'
                    if chord == '%':
                        if i > 1:
                            progression[i] = progression[i-1]  # Repeat previous
                        else:
                            progression[i] = 'I'  # Default
                    else:
                        progression[i] = chord

                progressions[prog_num] = progression
                prog_num += 1

    return progressions

def detect_mode_from_midi(midi_file_path: Path) -> str:
    """Detect if a MIDI file is in major or minor mode by analyzing chord content."""
    try:
        mid = mido.MidiFile(midi_file_path)

        # Collect all simultaneous note events
        all_chords = []
        for track in mid.tracks:
            time = 0
            events = []

            for msg in track:
                time += msg.time
                if hasattr(msg, 'note'):
                    if msg.type == 'note_on' and msg.velocity > 0:
                        events.append((time, 'on', msg.note))
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        events.append((time, 'off', msg.note))

            # Group simultaneous events
            current_time = None
            current_notes = set()

            for event_time, event_type, note in events:
                if current_time is None or event_time != current_time:
                    # Process previous chord if it exists
                    if current_notes and len(current_notes) >= 3:
                        all_chords.append(sorted(list(current_notes)))
                    current_time = event_time
                    current_notes = set()

                if event_type == 'on':
                    current_notes.add(note)
                else:
                    current_notes.discard(note)

            # Don't forget the last chord
            if current_notes and len(current_notes) >= 3:
                all_chords.append(sorted(list(current_notes)))

        # Analyze chord quality indicators - prioritize first chord (tonic)
        minor_indicators = 0
        major_indicators = 0

        for i, chord_notes in enumerate(all_chords[:4]):  # Analyze first few chords
            if len(chord_notes) >= 3:
                # Check for minor third (3 semitones) and major third (4 semitones) from root
                root = chord_notes[0]
                intervals = [(note - root) % 12 for note in chord_notes[1:]]

                # Weight the first chord more heavily (it's likely the tonic)
                weight = 3 if i == 0 else 1

                if 3 in intervals:  # Minor third present
                    minor_indicators += weight
                if 4 in intervals:  # Major third present
                    major_indicators += weight

        # If more minor thirds than major thirds, it's likely minor
        if minor_indicators > major_indicators:
            print(f"Detected MINOR mode in {midi_file_path.name} (minor:{minor_indicators} vs major:{major_indicators})")
            return "minor"
        else:
            print(f"Detected MAJOR mode in {midi_file_path.name} (minor:{minor_indicators} vs major:{major_indicators})")
            return "major"

    except Exception as e:
        print(f"Error detecting mode from {midi_file_path}: {e}")
        return "major"  # Default fallback

def main():
    import sys

    # Allow command line argument for mode or specific progression
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ["major", "minor"]:
            # Mode specified
            mode = arg
            progs_file = "progs.txt" if mode == "major" else "progsmin.txt"
            input_dir = Path("/home/user/app/chord_progressions") / mode
        elif arg.startswith("progression_"):
            # Specific progression specified, determine mode from additional arg
            mode = sys.argv[2] if len(sys.argv) > 2 else "major"
            progs_file = "progs.txt" if mode == "major" else "progsmin.txt"
            input_dir = Path("/home/user/app/chord_progressions") / mode
        else:
            # Legacy: directory path specified
            input_dir = Path(arg)
            if "major" in str(input_dir):
                mode = "major"
                progs_file = "progs.txt"
            elif "minor" in str(input_dir):
                mode = "minor"
                progs_file = "progsmin.txt"
            else:
                mode = "major"  # default
                progs_file = "progs.txt"

        output_dir = OUTPUT_DIR / mode
    else:
        input_dir = BASE_DIR
        output_dir = OUTPUT_DIR
        mode = "major"
        progs_file = "progs.txt"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse progs.txt for correct chord progressions
    progs_path = Path("/home/user/app") / progs_file
    if not progs_path.exists():
        raise FileNotFoundError(f"Missing {progs_file}: {progs_path}")

    all_progressions = parse_progs_txt(progs_path)

    # Check if a specific progression was requested
    specific_progression = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("progression_"):
        specific_progression = sys.argv[1]
    elif len(sys.argv) > 2 and sys.argv[2].startswith("progression_"):
        specific_progression = sys.argv[2]

    # Use the specific progression as base, or default to progression 1
    if specific_progression:
        prog_num = int(specific_progression.split('_')[1])
        if prog_num in all_progressions:
            base_progression = all_progressions[prog_num]
        else:
            print(f"Warning: {specific_progression} not found, using progression 1")
            base_progression = all_progressions[1]
    else:
        base_progression = all_progressions[1]

    print(f"Base progression: {base_progression}")

    # Learn patterns from reference files
    if specific_progression or mode == "minor":
        # When a specific progression is requested OR when processing minor progressions,
        # learn patterns from progression-specific MIDI files instead of ReferenceMidi
        if specific_progression:
            prog_num = int(specific_progression.split('_')[1])
            specific_prog_path = input_dir / f"progression_{prog_num:02d}.mid"
            if not specific_prog_path.exists():
                print(f"Error: {specific_prog_path} not found")
                return

            print(f"Learning patterns from {specific_prog_path}")
            ref_mid = mido.MidiFile(specific_prog_path)

            # Create learned patterns based on this single progression
            learned: Dict[str, Tuple[Dict[int, List[PatternEvent]], str]] = {}
            patterns = analyze_reference_pattern(ref_mid, base_progression)
            if patterns:
                # Create entries for each type of arrangement
                learned["RTriad"] = (patterns, "Chords")
                learned["ArpTriad"] = (patterns, "Arp8th")
                print(f"Learned patterns from {specific_progression}: bars {sorted(patterns.keys())}")
            else:
                print(f"Error: no valid patterns found in {specific_progression}")
                return
        else:
            # For minor mode without specific progression, only generate basic patterns
            # We'll learn patterns from each progression individually in the loop below
            learned: Dict[str, Tuple[Dict[int, List[PatternEvent]], str]] = {}
    else:
        # Use the traditional reference files for major progressions when no specific progression is requested
        ref_map = discover_reference_files()
        if not ref_map:
            print("No reference files found in Arp8th/ or Chords/.")
            return

        learned: Dict[str, Tuple[Dict[int, List[PatternEvent]], str]] = {}
        for name, (path, folder_type) in ref_map.items():
            ref_mid = mido.MidiFile(path)
            patterns = analyze_reference_pattern(ref_mid, base_progression)
            if patterns:
                learned[name] = (patterns, folder_type)
                print(f"Learned {name}: bars {sorted(patterns.keys())}")
            else:
                print(f"SKIPPED {name}: no valid patterns found")

    # Apply learned patterns to every base progression (or just the specific one)
    for base_path in sorted(input_dir.glob("progression_*.mid")):
        # Skip if a specific progression was requested and this isn't it
        if specific_progression and base_path.stem != specific_progression:
            continue

        # Extract progression number from filename
        match = re.search(r'progression_(\d+)', base_path.stem)
        if not match:
            continue

        prog_num = int(match.group(1))
        if prog_num not in all_progressions:
            print(f"Warning: No progression found for {base_path.stem}")
            continue

        # Get the correct progression from progs.txt
        target_progression = all_progressions[prog_num]

        base_mid = mido.MidiFile(base_path)
        base_tag = base_path.stem
        progression_dir = output_dir / base_tag
        progression_dir.mkdir(parents=True, exist_ok=True)

        tempo = get_first_tempo(base_mid)

        # For minor progressions without specific progression, learn patterns from ReferenceMidi files
        # but apply them to the progression-specific harmonic content
        current_learned = learned
        if mode == "minor" and not specific_progression:
            print(f"Learning patterns for {base_tag} using ReferenceMidi structures with minor harmonic content")
            # Get all reference files to learn their unique patterns
            ref_map = discover_reference_files()
            current_learned = {}

            # Define the reference progression that ReferenceMidi files were created with
            reference_progression = {1: 'I', 2: 'vi', 3: 'IV', 4: 'V'}

            for name, (ref_path, folder_type) in ref_map.items():
                # Learn the rhythmic/melodic pattern from the reference file
                ref_mid = mido.MidiFile(ref_path)
                # Analyze the reference pattern structure using the reference progression
                # This gets the timing, rhythm, and structure
                ref_patterns = analyze_reference_pattern(ref_mid, reference_progression)

                if ref_patterns:
                    current_learned[name] = (ref_patterns, folder_type)

            print(f"Learned {len(current_learned)} patterns with ReferenceMidi structures for {base_tag}")

            if not current_learned:
                print(f"Error: no valid patterns found for {base_tag}")
                continue

        for label, (pattern, folder_type) in current_learned.items():
            # Create organized subfolder structure
            type_dir = progression_dir / folder_type
            type_dir.mkdir(parents=True, exist_ok=True)

            out_path = type_dir / f"{label}.mid"
            # For minor progressions, use the reference progression as base for proper mapping
            if mode == "minor" and not specific_progression:
                reference_progression = {1: 'I', 2: 'vi', 3: 'IV', 4: 'V'}
                write_arrangement(target_progression, pattern, out_path, tempo, 0, reference_progression, True, label, folder_type, mode)
            else:
                write_arrangement(target_progression, pattern, out_path, tempo, 0, target_progression, True, label, folder_type, mode)
            print(f"→ {out_path}")

if __name__ == "__main__":
    main()