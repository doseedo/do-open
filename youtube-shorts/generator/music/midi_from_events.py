"""Convert physics collision events to MIDI notes."""

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

# Scale intervals (semitones from root)
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
}

NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_key(key_str):
    """Parse 'C major' into (root_midi, scale_intervals)."""
    parts = key_str.strip().split()
    note_name = parts[0].upper()
    scale_name = parts[1].lower() if len(parts) > 1 else "major"

    root_pc = NOTE_NAMES.get(note_name, 0)
    scale = SCALES.get(scale_name, SCALES["major"])
    return root_pc, scale


def snap_to_scale(pitch, root_pc, scale):
    """Snap a MIDI pitch to the nearest note in the given scale."""
    octave = pitch // 12
    pc = pitch % 12
    relative = (pc - root_pc) % 12

    # Find closest scale degree
    best = min(scale, key=lambda s: min(abs(relative - s), 12 - abs(relative - s)))
    snapped_pc = (root_pc + best) % 12
    result = octave * 12 + snapped_pc

    # Pick the closest octave
    if abs(result - pitch) > abs(result + 12 - pitch):
        result += 12
    elif abs(result - pitch) > abs(result - 12 - pitch):
        result -= 12
    return max(0, min(127, result))


def y_to_pitch(y, screen_height, pitch_range=(48, 84), root_pc=0, scale=None):
    """Map Y-position to a scale-quantized MIDI pitch.
    Top of screen = high pitch, bottom = low pitch.
    """
    normalized = 1.0 - (y / screen_height)  # 0=bottom, 1=top
    normalized = max(0.0, min(1.0, normalized))
    raw_pitch = pitch_range[0] + normalized * (pitch_range[1] - pitch_range[0])
    if scale:
        return snap_to_scale(int(raw_pitch), root_pc, scale)
    return int(raw_pitch)


def x_to_pitch(x, screen_width, pitch_range=(48, 84), root_pc=0, scale=None):
    """Map X-position to a scale-quantized MIDI pitch.
    Left = low pitch, right = high pitch.
    """
    normalized = x / screen_width
    normalized = max(0.0, min(1.0, normalized))
    raw_pitch = pitch_range[0] + normalized * (pitch_range[1] - pitch_range[0])
    if scale:
        return snap_to_scale(int(raw_pitch), root_pc, scale)
    return int(raw_pitch)


def force_to_velocity(force, max_force=5000.0):
    """Map collision force to MIDI velocity (30-127)."""
    return int(min(127, max(30, (force / max_force) * 127)))


def deduplicate_events(events, min_interval_sec=0.05):
    """Remove events that are too close together in time."""
    if not events:
        return []
    sorted_events = sorted(events, key=lambda e: e["time_sec"])
    result = [sorted_events[0]]
    for ev in sorted_events[1:]:
        if ev["time_sec"] - result[-1]["time_sec"] >= min_interval_sec:
            result.append(ev)
    return result


def quantize_events(events, tempo, quantize_to="16th"):
    """Snap event times to nearest musical grid position."""
    beat_sec = 60.0 / tempo
    grid_map = {
        "8th": beat_sec / 2,
        "16th": beat_sec / 4,
        "32nd": beat_sec / 8,
    }
    grid = grid_map.get(quantize_to, beat_sec / 4)

    for ev in events:
        t = ev["time_sec"]
        ev["time_sec"] = round(t / grid) * grid
    return events


def events_to_midi(events, output_path, key="C major", pitch_range=(48, 84),
                   screen_height=1920, screen_width=1080,
                   tempo=120, note_duration_sec=0.15,
                   pitch_mapping="y_position", max_force=5000.0,
                   program=13):
    """Convert collision events to a MIDI file.

    Args:
        events: List of dicts with keys: time_sec, x, y, force
        output_path: Where to write the MIDI file
        key: Musical key string like 'C major'
        pitch_range: (min_pitch, max_pitch) MIDI note range
        screen_height/width: For coordinate mapping
        tempo: BPM for quantization
        note_duration_sec: How long each note rings
        pitch_mapping: 'y_position' or 'x_position'
        max_force: Force value that maps to velocity 127
        program: GM program number (13=xylophone, 12=marimba, 11=vibraphone)

    Returns:
        Path to the written MIDI file
    """
    root_pc, scale = parse_key(key)

    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo)))
    track.append(Message("program_change", program=program, channel=0, time=0))

    # Sort events by time
    sorted_events = sorted(events, key=lambda e: e["time_sec"])
    if not sorted_events:
        mid.save(output_path)
        return output_path

    note_dur_ticks = int(note_duration_sec * (tempo / 60.0) * 480)

    # Build note-on/note-off pairs, then sort by absolute tick
    midi_events = []
    for ev in sorted_events:
        abs_tick = int(ev["time_sec"] * (tempo / 60.0) * 480)

        if pitch_mapping == "x_position":
            pitch = x_to_pitch(ev["x"], screen_width, pitch_range, root_pc, scale)
        else:
            pitch = y_to_pitch(ev["y"], screen_height, pitch_range, root_pc, scale)

        velocity = force_to_velocity(ev["force"], max_force)

        midi_events.append((abs_tick, "note_on", pitch, velocity))
        midi_events.append((abs_tick + note_dur_ticks, "note_off", pitch, 0))

    # Sort by tick, convert to delta times
    midi_events.sort(key=lambda e: e[0])
    prev_tick = 0
    for tick, msg_type, pitch, vel in midi_events:
        delta = max(0, tick - prev_tick)
        if msg_type == "note_on":
            track.append(Message("note_on", note=pitch, velocity=vel, time=delta, channel=0))
        else:
            track.append(Message("note_off", note=pitch, velocity=0, time=delta, channel=0))
        prev_tick = tick

    mid.save(output_path)
    return output_path


def video_events_to_midi(events, output_path, key="C major", tempo=120,
                         max_force=5000.0, default_program=13):
    """Convert video grid events to a multi-instrument MIDI file.

    Each event can have its own 'instrument', 'duration_sec', and pre-computed
    pitch (from grid position). Events are grouped by instrument and written
    to separate MIDI channels.

    Args:
        events: List of dicts with: time_sec, x, y, force, and optionally
                instrument, duration_sec, grid_row, grid_col
        output_path: Where to write the MIDI file
        key: Musical key for scale quantization
        tempo: BPM
        max_force: Force value mapping to velocity 127
        default_program: Default GM program if instrument not specified

    Returns:
        Path to written MIDI file, dict mapping instrument→channel
    """
    from collections import defaultdict

    root_pc, scale = parse_key(key)

    # Group events by instrument
    inst_groups = defaultdict(list)
    for ev in events:
        inst = ev.get("instrument", "xylophone")
        inst_groups[inst].append(ev)

    mid = MidiFile(ticks_per_beat=480)
    ticks_per_sec = 480 * (tempo / 60.0)
    inst_channels = {}
    channel_idx = 0

    for inst_name, inst_events in inst_groups.items():
        if channel_idx >= 15:
            break  # Max 16 MIDI channels (skip 9=drums)
        ch = channel_idx if channel_idx < 9 else channel_idx + 1
        inst_channels[inst_name] = ch
        channel_idx += 1

        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo)))

        # Import GM_PROGRAMS from engine to get program number
        try:
            from .engine import GM_PROGRAMS
            program = GM_PROGRAMS.get(inst_name, default_program)
        except ImportError:
            program = default_program

        track.append(Message("program_change", program=program, channel=ch, time=0))

        # Sort events by time
        sorted_ev = sorted(inst_events, key=lambda e: e["time_sec"])

        # Build note events
        midi_events = []
        for ev in sorted_ev:
            abs_tick = int(ev["time_sec"] * ticks_per_sec)

            # Use grid-based pitch if available, otherwise use y_position
            if "grid_col" in ev and "grid_row" in ev:
                # Pitch already computed by the scene — use x/y for mapping
                pitch = y_to_pitch(ev["y"], 1920, (48, 84), root_pc, scale)
            else:
                pitch = y_to_pitch(ev["y"], 1920, (48, 84), root_pc, scale)

            # Snap to scale
            pitch = snap_to_scale(pitch, root_pc, scale)

            velocity = force_to_velocity(ev["force"], max_force)

            # Per-event duration
            dur_sec = ev.get("duration_sec", 0.15)
            dur_ticks = max(1, int(dur_sec * ticks_per_sec))

            midi_events.append((abs_tick, "note_on", pitch, velocity, ch))
            midi_events.append((abs_tick + dur_ticks, "note_off", pitch, 0, ch))

        # Sort and write delta times
        midi_events.sort(key=lambda e: e[0])
        prev_tick = 0
        for tick, msg_type, pitch, vel, ch in midi_events:
            delta = max(0, tick - prev_tick)
            if msg_type == "note_on":
                track.append(Message("note_on", note=pitch, velocity=vel,
                                     time=delta, channel=ch))
            else:
                track.append(Message("note_off", note=pitch, velocity=0,
                                     time=delta, channel=ch))
            prev_tick = tick

    mid.save(output_path)
    return output_path, inst_channels
