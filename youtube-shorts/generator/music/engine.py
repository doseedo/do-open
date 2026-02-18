"""MusicEngine — orchestrates MIDI generation and audio rendering for animations."""

import os
import sys
import random
import subprocess
import tempfile

# Add paths for external music tools
sys.path.insert(0, "/home/arlo/Data")
sys.path.insert(0, "/home/arlo/do-repo/harmonymodule")

from .. import config
from .midi_from_events import (
    events_to_midi, video_events_to_midi,
    deduplicate_events, quantize_events,
    parse_key, y_to_pitch, force_to_velocity,
)
from .mixer import mix_stems, pad_or_trim, add_reverb

# GM program numbers for common percussion/melodic instruments
GM_PROGRAMS = {
    "celesta": 8,
    "glockenspiel": 9,
    "music_box": 10,
    "vibraphone": 11,
    "marimba": 12,
    "xylophone": 13,
    "tubular_bells": 14,
    "acoustic_piano": 0,
    "electric_piano": 4,
    "acoustic_guitar": 24,
    "electric_guitar": 27,
    "violin": 40,
    "cello": 42,
    "flute": 73,
    "clarinet": 71,
    "trumpet": 56,
    "trombone": 57,
    "electric_bass": 33,
    "sax": 65,
}

# Map instrument names to dedicated soundfont paths (higher quality)
DEDICATED_SOUNDFONTS = {
    "acoustic_piano": "/home/arlo/Data/soundfonts/Piano.sf2",
    "electric_piano": "/home/arlo/Data/soundfonts/Electric Piano.sf2",
    "violin": "/home/arlo/Data/soundfonts/violin.sf2",
    "viola": "/home/arlo/Data/soundfonts/viola.sf2",
    "cello": "/home/arlo/Data/soundfonts/cello.sf2",
    "flute": "/home/arlo/Data/soundfonts/flute.sf2",
    "clarinet": "/home/arlo/Data/soundfonts/clarinet.sf2",
    "trumpet": "/home/arlo/Data/soundfonts/trumpet.sf2",
    "trombone": "/home/arlo/Data/soundfonts/trombone.sf2",
    "sax": "/home/arlo/Data/soundfonts/sax.sf2",
    "acoustic_guitar": "/home/arlo/Data/soundfonts/acoustic guitar.sf2",
    "electric_guitar": "/home/arlo/Data/soundfonts/electric guitar.sf2",
    "electric_bass": "/home/arlo/Data/soundfonts/electric bass.sf2",
}


def _render_midi_fluidsynth(midi_path, output_path, instrument="xylophone"):
    """Render MIDI to WAV using fluidsynth.

    Uses dedicated soundfont if available for the instrument,
    otherwise uses FluidR3_GM with the appropriate GM program number.
    """
    if instrument in DEDICATED_SOUNDFONTS:
        soundfont = DEDICATED_SOUNDFONTS[instrument]
    else:
        soundfont = config.DEFAULT_SOUNDFONT

    cmd = [
        "fluidsynth", "-ni",
        "-g", "1.0",
        "-r", str(config.AUDIO_SAMPLE_RATE),
        "-F", output_path,
        soundfont,
        midi_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"fluidsynth failed: {result.stderr.decode()[:500]}")
    return output_path


class MusicEngine:
    """Generates and renders music for animation scenes."""

    def __init__(self, temp_dir=None, seed=None):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="yt_music_")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def _temp_path(self, name):
        return os.path.join(self.temp_dir, name)

    # -----------------------------------------------------------------
    # Mode 1: Animation drives music (collision events → audio)
    # -----------------------------------------------------------------

    def generate_from_events(self, events, key="C major", instrument="xylophone",
                             tempo=120, pitch_range=(48, 84),
                             screen_height=1920, screen_width=1080,
                             pitch_mapping="y_position",
                             note_duration_sec=0.15, max_force=5000.0):
        """Convert collision events to rendered audio.

        Args:
            events: List of dicts with {time_sec, x, y, force}
            key, instrument, tempo, etc.: Music parameters

        Returns:
            Path to WAV file, or None if no events
        """
        if not events:
            return None

        # Deduplicate and quantize
        processed = deduplicate_events(events, min_interval_sec=0.03)
        processed = quantize_events(processed, tempo, quantize_to="16th")

        # Limit to prevent overwhelming MIDI (cap at ~500 notes)
        if len(processed) > 500:
            # Keep events with highest force
            processed.sort(key=lambda e: e["force"], reverse=True)
            processed = processed[:500]

        # Get GM program number
        program = GM_PROGRAMS.get(instrument, 13)

        # Convert to MIDI
        midi_path = self._temp_path("events.mid")
        events_to_midi(
            processed, midi_path,
            key=key, pitch_range=pitch_range,
            screen_height=screen_height, screen_width=screen_width,
            tempo=tempo, note_duration_sec=note_duration_sec,
            pitch_mapping=pitch_mapping, max_force=max_force,
            program=program,
        )

        # Render to audio
        raw_wav = self._temp_path("events_raw.wav")
        _render_midi_fluidsynth(midi_path, raw_wav, instrument)

        # Add light reverb
        output_wav = self._temp_path("events_final.wav")
        add_reverb(raw_wav, output_wav, reverberance=25, room_scale=60)

        return output_wav

    # -----------------------------------------------------------------
    # Mode 1b: Video grid events → multi-instrument audio
    # -----------------------------------------------------------------

    def generate_from_video_events(self, events, key="C major", tempo=120,
                                    max_force=5000.0):
        """Convert video grid events (with per-event instruments) to audio.

        Groups events by instrument, renders each to a separate MIDI track,
        then renders per-instrument WAV stems and mixes them.

        Returns:
            Path to WAV file, or None if no events
        """
        if not events:
            return None

        # Deduplicate (per-instrument grouping is preserved)
        processed = deduplicate_events(events, min_interval_sec=0.02)
        processed = quantize_events(processed, tempo, quantize_to="16th")

        # Cap total events
        if len(processed) > 800:
            processed.sort(key=lambda e: e["force"], reverse=True)
            processed = processed[:800]

        # Find unique instruments
        instruments = set(ev.get("instrument", "xylophone") for ev in processed)

        if len(instruments) <= 1:
            # Single instrument — use standard pipeline
            inst = list(instruments)[0] if instruments else "xylophone"
            program = GM_PROGRAMS.get(inst, 13)
            midi_path = self._temp_path("video_events.mid")
            events_to_midi(
                processed, midi_path,
                key=key, pitch_range=(48, 84),
                screen_height=1920, screen_width=1080,
                tempo=tempo, note_duration_sec=0.15,
                pitch_mapping="y_position", max_force=max_force,
                program=program,
            )
            raw_wav = self._temp_path("video_events_raw.wav")
            _render_midi_fluidsynth(midi_path, raw_wav, inst)
            output_wav = self._temp_path("video_events_final.wav")
            add_reverb(raw_wav, output_wav, reverberance=20, room_scale=50)
            return output_wav

        # Multi-instrument: render each instrument separately, then mix
        midi_path = self._temp_path("video_multi.mid")
        video_events_to_midi(
            processed, midi_path,
            key=key, tempo=tempo, max_force=max_force,
        )

        # Group events by instrument to render separate stems
        from collections import defaultdict
        inst_events = defaultdict(list)
        for ev in processed:
            inst_events[ev.get("instrument", "xylophone")].append(ev)

        stems = []
        for i, (inst_name, inst_evs) in enumerate(inst_events.items()):
            stem_midi = self._temp_path(f"stem_{i}_{inst_name}.mid")
            program = GM_PROGRAMS.get(inst_name, 13)
            events_to_midi(
                inst_evs, stem_midi,
                key=key, pitch_range=(48, 84),
                screen_height=1920, screen_width=1080,
                tempo=tempo,
                note_duration_sec=inst_evs[0].get("duration_sec", 0.15),
                pitch_mapping="y_position",
                max_force=max_force, program=program,
            )
            stem_wav = self._temp_path(f"stem_{i}_{inst_name}.wav")
            _render_midi_fluidsynth(stem_midi, stem_wav, inst_name)
            if os.path.exists(stem_wav) and os.path.getsize(stem_wav) > 100:
                stems.append(stem_wav)

        if not stems:
            return None

        # Mix all stems
        mixed = self._temp_path("video_mixed.wav")
        gains = [0.7 / len(stems)] * len(stems)
        mix_stems(stems, mixed, gains=gains)

        output_wav = self._temp_path("video_events_final.wav")
        add_reverb(mixed, output_wav, reverberance=20, room_scale=50)
        return output_wav

    # -----------------------------------------------------------------
    # Mode 2: Music drives animation (generate note schedule)
    # -----------------------------------------------------------------

    def generate_note_schedule(self, duration_sec, key="C major", tempo=120,
                               num_notes=30, pitch_range=(48, 84)):
        """Generate a note schedule for scenes to consume.

        Returns:
            List of {time_sec, pitch, velocity, duration_sec}
        """
        try:
            from melody_generator import MelodyGenerator
        except ImportError:
            # Fallback: generate simple random schedule
            return self._fallback_note_schedule(duration_sec, key, tempo,
                                                 num_notes, pitch_range)

        # Build a simple chord progression
        chord_map = self._make_chord_progression(key, duration_sec, tempo)

        gen = MelodyGenerator(key=key, tempo=tempo, seed=self.seed)
        ticks_per_beat = 480
        beats = duration_sec * (tempo / 60.0)
        num_bars = max(2, int(beats / 4))

        melody = gen.generate_melody(
            chord_progression=chord_map,
            num_bars=num_bars,
            min_pitch=pitch_range[0],
            max_pitch=pitch_range[1],
            creativity=0.5,
            rhythmic_density="moderate",
        )

        # Convert ticks to absolute seconds
        schedule = []
        ticks_per_sec = ticks_per_beat * (tempo / 60.0)
        for tick, dur_ticks, pitch in melody:
            time_sec = tick / ticks_per_sec
            dur_sec = dur_ticks / ticks_per_sec
            if time_sec < duration_sec:
                schedule.append({
                    "time_sec": time_sec,
                    "pitch": pitch,
                    "velocity": random.randint(70, 110),
                    "duration_sec": dur_sec,
                })

        # Limit number of notes if needed
        if len(schedule) > num_notes:
            # Evenly space out notes
            step = len(schedule) / num_notes
            schedule = [schedule[int(i * step)] for i in range(num_notes)]

        return schedule

    def render_note_schedule_to_audio(self, schedule, instrument="xylophone",
                                       key="C major", tempo=120):
        """Render a note schedule to audio WAV.

        Args:
            schedule: List of {time_sec, pitch, velocity, duration_sec}

        Returns:
            Path to WAV file
        """
        import mido
        from mido import MidiFile, MidiTrack, Message, MetaMessage

        program = GM_PROGRAMS.get(instrument, 13)
        mid = MidiFile(ticks_per_beat=480)
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo)))
        track.append(Message("program_change", program=program, channel=0, time=0))

        ticks_per_sec = 480 * (tempo / 60.0)
        midi_events = []
        for note in schedule:
            on_tick = int(note["time_sec"] * ticks_per_sec)
            dur_ticks = max(1, int(note["duration_sec"] * ticks_per_sec))
            midi_events.append((on_tick, "note_on", note["pitch"], note["velocity"]))
            midi_events.append((on_tick + dur_ticks, "note_off", note["pitch"], 0))

        midi_events.sort(key=lambda e: e[0])
        prev_tick = 0
        for tick, msg_type, pitch, vel in midi_events:
            delta = max(0, tick - prev_tick)
            if msg_type == "note_on":
                track.append(Message("note_on", note=pitch, velocity=vel,
                                     time=delta, channel=0))
            else:
                track.append(Message("note_off", note=pitch, velocity=0,
                                     time=delta, channel=0))
            prev_tick = tick

        midi_path = self._temp_path("schedule.mid")
        mid.save(midi_path)

        raw_wav = self._temp_path("schedule_raw.wav")
        _render_midi_fluidsynth(midi_path, raw_wav, instrument)

        output_wav = self._temp_path("schedule_final.wav")
        add_reverb(raw_wav, output_wav, reverberance=20, room_scale=50)

        return output_wav

    # -----------------------------------------------------------------
    # Mode 3: Parallel sync (independent background music)
    # -----------------------------------------------------------------

    def generate_background(self, duration_sec, key="C major", tempo=120,
                             instrument="acoustic_piano"):
        """Generate background music independently of animation.

        Returns:
            Path to WAV file
        """
        stems = []

        # Generate chord progression
        chord_map = self._make_chord_progression(key, duration_sec, tempo)
        chord_wav = self._generate_chord_audio(chord_map, key, tempo, instrument)
        if chord_wav:
            stems.append(chord_wav)

        # Generate melody
        melody_wav = self._generate_melody_audio(chord_map, key, tempo, instrument,
                                                  duration_sec)
        if melody_wav:
            stems.append(melody_wav)

        if not stems:
            return None

        # Mix stems
        mixed = self._temp_path("background_mixed.wav")
        gains = [0.4, 0.6] if len(stems) == 2 else [0.8]
        mix_stems(stems, mixed, gains=gains)

        # Pad or trim to exact duration
        output = self._temp_path("background_final.wav")
        pad_or_trim(mixed, duration_sec, output)

        return output

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _make_chord_progression(self, key, duration_sec, tempo):
        """Generate a simple chord progression dict (beat_index → chord_name)."""
        parts = key.strip().split()
        root = parts[0]
        scale_type = parts[1] if len(parts) > 1 else "major"

        if scale_type == "minor":
            progressions = [
                {0: f"{root}m", 4: f"{root}m", 8: f"{_rel_major(root)}", 12: f"{_dom_of(root)}"},
                {0: f"{root}m7", 4: f"{_sub_of(root, 'minor', 3)}", 8: f"{_sub_of(root, 'minor', 6)}", 12: f"{_dom_of(root)}7"},
                {0: f"{root}m", 4: f"{_sub_of(root, 'minor', 6)}", 8: f"{_sub_of(root, 'minor', 3)}", 12: f"{_dom_of(root)}"},
            ]
        else:
            progressions = [
                {0: f"{root}", 4: f"{root}", 8: f"{_sub_of(root, 'major', 4)}", 12: f"{_sub_of(root, 'major', 5)}"},
                {0: f"{root}maj7", 4: f"{_sub_of(root, 'major', 6)}m7", 8: f"{_sub_of(root, 'major', 4)}maj7", 12: f"{_sub_of(root, 'major', 5)}7"},
                {0: f"{root}", 4: f"{_sub_of(root, 'major', 5)}", 8: f"{_sub_of(root, 'major', 6)}m", 12: f"{_sub_of(root, 'major', 4)}"},
            ]

        # Pick one and repeat to fill duration
        base = random.choice(progressions)
        beats_total = int(duration_sec * (tempo / 60.0))
        chord_map = {}
        cycle_len = 16  # 4 bars of 4/4
        for beat in range(0, beats_total, cycle_len):
            for offset, chord in base.items():
                chord_map[beat + offset] = chord

        return chord_map

    def _generate_chord_audio(self, chord_map, key, tempo, instrument):
        """Render chord progression to audio."""
        try:
            from chord_progression_generator import generate_chord_progression_midi
        except ImportError:
            return None

        midi_path = self._temp_path("chords.mid")
        voicings = ["close", "drop2", "shell", "spread"]
        rhythms = ["whole", "half", "quarter"]
        generate_chord_progression_midi(
            chord_beat_map=chord_map,
            bpm=tempo,
            voicing=random.choice(voicings),
            rhythm=random.choice(rhythms),
            output_path=midi_path,
        )

        wav_path = self._temp_path("chords.wav")
        _render_midi_fluidsynth(midi_path, wav_path, instrument)
        return wav_path

    def _generate_melody_audio(self, chord_map, key, tempo, instrument, duration_sec):
        """Generate a melody over chords and render to audio."""
        try:
            from melody_generator import MelodyGenerator
        except ImportError:
            return None

        gen = MelodyGenerator(key=key, tempo=tempo, seed=self.seed)
        beats = duration_sec * (tempo / 60.0)
        num_bars = max(2, int(beats / 4))

        melody = gen.generate_melody(
            chord_progression=chord_map,
            num_bars=num_bars,
            min_pitch=60,
            max_pitch=84,
            creativity=0.6,
            rhythmic_density="moderate",
        )

        midi_path = self._temp_path("melody.mid")
        gen.save_midi(melody, midi_path, velocity=80)

        wav_path = self._temp_path("melody.wav")
        # Use a contrasting instrument for melody
        melody_instruments = ["xylophone", "vibraphone", "marimba", "flute", "celesta"]
        melody_inst = random.choice(melody_instruments)
        _render_midi_fluidsynth(midi_path, wav_path, melody_inst)
        return wav_path

    def _fallback_note_schedule(self, duration_sec, key, tempo, num_notes, pitch_range):
        """Simple random note schedule when MelodyGenerator isn't available."""
        root_pc, scale = parse_key(key)
        beat_sec = 60.0 / tempo
        interval = duration_sec / max(1, num_notes)

        schedule = []
        for i in range(num_notes):
            t = i * interval + random.uniform(-0.02, 0.02)
            pitch = random.randint(pitch_range[0], pitch_range[1])
            # Snap to scale
            from .midi_from_events import snap_to_scale
            pitch = snap_to_scale(pitch, root_pc, scale)
            schedule.append({
                "time_sec": max(0, t),
                "pitch": pitch,
                "velocity": random.randint(60, 100),
                "duration_sec": beat_sec * 0.5,
            })
        return schedule


# --- Utility functions for chord progression building ---

_NOTE_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Also support flats
_FLAT_MAP = {"Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B"}


def _note_index(name):
    if name in _FLAT_MAP:
        name = _FLAT_MAP[name]
    return _NOTE_ORDER.index(name) if name in _NOTE_ORDER else 0


def _note_from_index(idx):
    return _NOTE_ORDER[idx % 12]


def _sub_of(root, scale_type, degree):
    """Get the scale degree note name."""
    root_idx = _note_index(root)
    if scale_type == "major":
        intervals = [0, 2, 4, 5, 7, 9, 11]
    else:
        intervals = [0, 2, 3, 5, 7, 8, 10]
    return _note_from_index(root_idx + intervals[degree - 1])


def _dom_of(root):
    """Get the dominant (5th) of a root."""
    return _note_from_index(_note_index(root) + 7)


def _rel_major(root):
    """Get the relative major of a minor key root."""
    return _note_from_index(_note_index(root) + 3)
