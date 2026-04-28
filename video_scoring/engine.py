"""Film scoring engine — feature → progression → MIDI.

Ported from Do-Dev/home/arlo/Data/film_scoring_engine.py with:
  * Video analysis decoupled (now lives in analyzer.py — this module only
    consumes pre-computed VideoFeatures, so the engine has no cv2 / scenedetect
    dependency and is unit-testable in isolation).
  * generate_score actually emits MIDI notes (the original had a `# TODO`
    where note events should have been written, producing empty MIDI files).
  * ostinato_pattern bug fixed (the source dict literal had duplicate `0`/`2`
    keys in the suspense pattern that silently overwrote each other).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mido

# ---------------------------------------------------------------------------
# Enums + dataclasses
# ---------------------------------------------------------------------------


class TensionLevel(Enum):
    VERY_LOW = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4
    CLIMAX = 5


class MoodCategory(Enum):
    WARM_BRIGHT = "warm_bright"
    WARM_DARK = "warm_dark"
    COOL_BRIGHT = "cool_bright"
    COOL_DARK = "cool_dark"
    SATURATED = "saturated"
    DESATURATED = "desaturated"
    HIGH_CONTRAST = "high_contrast"
    LOW_CONTRAST = "low_contrast"


class ScoringSyncType(Enum):
    MICKEY_MOUSE = "mickey_mouse"
    UNDERSCORING = "underscoring"
    SOURCE_MUSIC = "source_music"
    HIT_POINTS = "hit_points"
    TENSION_ARC = "tension_arc"
    OSTINATO = "ostinato"


@dataclass
class SMPTETimecode:
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    frames: int = 0
    framerate: float = 24.0

    def to_seconds(self) -> float:
        return (
            self.hours * 3600
            + self.minutes * 60
            + self.seconds
            + self.frames / self.framerate
        )

    @classmethod
    def from_seconds(cls, total_seconds: float, framerate: float = 24.0) -> "SMPTETimecode":
        hours = int(total_seconds // 3600)
        rem = total_seconds % 3600
        minutes = int(rem // 60)
        seconds = int(rem % 60)
        frames = int((rem % 1) * framerate)
        return cls(hours, minutes, seconds, frames, framerate)

    def __str__(self) -> str:
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}:{self.frames:02d}"


@dataclass
class VideoFeatures:
    start_time: float
    end_time: float
    duration: float

    avg_brightness: float = 0.5
    avg_saturation: float = 0.5
    avg_hue: float = 0.0
    contrast_level: float = 0.5
    motion_intensity: float = 0.0
    cut_density: float = 0.0

    mood: MoodCategory = MoodCategory.COOL_BRIGHT
    visual_tension: float = 0.5

    has_dialogue: bool = False
    dialogue_density: float = 0.0
    audio_intensity: float = 0.5

    is_scene_start: bool = False
    scene_id: int = 0

    # VLM additions (free-text descriptions sourced from Moondream)
    objects: List[str] = field(default_factory=list)
    detected_text: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class HitPoint:
    timecode: SMPTETimecode
    time_seconds: float
    description: str
    tension_level: TensionLevel
    sync_type: ScoringSyncType
    musical_event: str = "chord_change"


@dataclass
class Leitmotif:
    name: str
    chord_progression: Dict[int, str]
    melody_contour: List[int] = field(default_factory=list)
    harmonic_character: str = "major"
    tempo_range: Tuple[int, int] = (100, 140)

    can_invert: bool = True
    can_retrograde: bool = True
    can_augment: bool = True
    can_diminish: bool = True
    can_transpose: bool = True

    heroic_variation: Optional[str] = None
    tragic_variation: Optional[str] = None
    mysterious_variation: Optional[str] = None


@dataclass
class TensionArc:
    timestamps: List[float]
    tension_values: List[float]

    def get_tension_at(self, t: float) -> float:
        if not self.timestamps:
            return 0.5
        if t <= self.timestamps[0]:
            return self.tension_values[0]
        if t >= self.timestamps[-1]:
            return self.tension_values[-1]
        for i in range(len(self.timestamps) - 1):
            if self.timestamps[i] <= t <= self.timestamps[i + 1]:
                span = self.timestamps[i + 1] - self.timestamps[i]
                if span <= 0:
                    return self.tension_values[i]
                u = (t - self.timestamps[i]) / span
                return self.tension_values[i] * (1 - u) + self.tension_values[i + 1] * u
        return 0.5


# ---------------------------------------------------------------------------
# Music-theory helpers
# ---------------------------------------------------------------------------

_NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}
_PC_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _parse_chord(symbol: str) -> Tuple[int, str]:
    """Return (root_pc, quality) from a chord symbol like 'C', 'Cm7', 'F#dim'."""
    if not symbol:
        return 0, ""
    if len(symbol) > 1 and symbol[1] in {"#", "b", "♯", "♭"}:
        root = symbol[:2].replace("♯", "#").replace("♭", "b")
        quality = symbol[2:]
    else:
        root = symbol[0]
        quality = symbol[1:]
    return _NOTE_TO_PC.get(root, 0), quality


def _intervals_for_quality(quality: str) -> List[int]:
    """Map a chord quality string to semitone offsets above the root."""
    q = quality.strip()
    if q == "" or q == "maj":
        return [0, 4, 7]
    if q == "m":
        return [0, 3, 7]
    if q == "dim":
        return [0, 3, 6]
    if q == "aug":
        return [0, 4, 8]
    if q == "maj7":
        return [0, 4, 7, 11]
    if q == "m7":
        return [0, 3, 7, 10]
    if q == "7":
        return [0, 4, 7, 10]
    if q == "7b9":
        return [0, 4, 7, 10, 13]
    if q == "m9":
        return [0, 3, 7, 10, 14]
    if q == "9":
        return [0, 4, 7, 10, 14]
    return [0, 4, 7]


def _chord_to_midi_notes(symbol: str, octave: int = 4) -> List[int]:
    root_pc, quality = _parse_chord(symbol)
    base = 12 * (octave + 1) + root_pc  # MIDI C4 = 60
    return [base + i for i in _intervals_for_quality(quality)]


# ---------------------------------------------------------------------------
# Compositional techniques
# ---------------------------------------------------------------------------


class FilmScoringTechniques:
    """Static helpers — tension/mood mappings, progression morphing, ostinati."""

    @staticmethod
    def chromatic_voice_leading(start_chord: str, end_chord: str, steps: int = 4) -> List[str]:
        start_root, _ = _parse_chord(start_chord)
        end_root, _ = _parse_chord(end_chord)

        distance = (end_root - start_root) % 12
        if distance > 6:
            distance -= 12

        step_size = distance / max(steps, 1)
        out = []
        for i in range(steps + 1):
            pc = int(start_root + step_size * i) % 12
            out.append(f"{_PC_TO_NOTE[pc]}m")
        return out

    @staticmethod
    def ostinato_pattern(root_note: str = "C", pattern_type: str = "suspense") -> Dict[int, str]:
        if pattern_type == "action":
            return {0: root_note, 1: root_note, 2: root_note, 3: root_note}
        if pattern_type == "mystery":
            return {0: f"{root_note}dim", 2: f"{root_note}dim"}
        # default: suspense — four-bar minor pedal alternating with m7
        return {
            0: f"{root_note}m",
            2: f"{root_note}m7",
            4: f"{root_note}m",
            6: f"{root_note}m7",
        }

    @staticmethod
    def tension_to_chord_complexity(tension: float) -> str:
        if tension < 0.2:
            return "maj"
        if tension < 0.4:
            return "maj7"
        if tension < 0.6:
            return "m7"
        if tension < 0.8:
            return "7"
        if tension < 0.9:
            return "7b9"
        return "dim"

    @staticmethod
    def mood_to_scale_context(mood: MoodCategory) -> str:
        return {
            MoodCategory.WARM_BRIGHT: "major",
            MoodCategory.WARM_DARK: "minor",
            MoodCategory.COOL_BRIGHT: "major",
            MoodCategory.COOL_DARK: "harmonic_minor",
            MoodCategory.SATURATED: "major",
            MoodCategory.DESATURATED: "minor",
            MoodCategory.HIGH_CONTRAST: "harmonic_minor",
            MoodCategory.LOW_CONTRAST: "major",
        }.get(mood, "minor")

    @staticmethod
    def morph_progression(
        original: Dict[int, str],
        target_mood: MoodCategory,
        tension: float,
    ) -> Dict[int, str]:
        target_quality = FilmScoringTechniques.tension_to_chord_complexity(tension)
        dark_moods = {
            MoodCategory.WARM_DARK,
            MoodCategory.COOL_DARK,
            MoodCategory.DESATURATED,
        }
        out: Dict[int, str] = {}
        for beat, chord in original.items():
            root_pc, _ = _parse_chord(chord)
            root_name = _PC_TO_NOTE[root_pc]
            if target_mood in dark_moods:
                out[beat] = f"{root_name}m7" if tension > 0.5 else f"{root_name}m"
            else:
                out[beat] = f"{root_name}{target_quality}"
        return out


# ---------------------------------------------------------------------------
# Leitmotifs
# ---------------------------------------------------------------------------


class LeitmotifEngine:
    def __init__(self) -> None:
        self.motifs: Dict[str, Leitmotif] = {}

    def register_motif(self, motif: Leitmotif) -> None:
        self.motifs[motif.name] = motif

    def get_variation(
        self,
        motif_name: str,
        tension: float,
        tempo_factor: float = 1.0,
        transpose_semitones: int = 0,
    ) -> Dict[int, str]:
        if motif_name not in self.motifs:
            return {0: "C"}
        motif = self.motifs[motif_name]
        prog = dict(motif.chord_progression)

        if tension > 0.7 and motif.can_augment:
            prog = {int(b * 2): c for b, c in prog.items()}
        elif tension < 0.3 and motif.can_diminish:
            prog = {int(b * 0.5): c for b, c in prog.items()}

        if transpose_semitones != 0 and motif.can_transpose:
            transposed: Dict[int, str] = {}
            for beat, chord in prog.items():
                pc, q = _parse_chord(chord)
                transposed[beat] = f"{_PC_TO_NOTE[(pc + transpose_semitones) % 12]}{q}"
            prog = transposed
        return prog


# ---------------------------------------------------------------------------
# Engine — VideoFeatures → MIDI
# ---------------------------------------------------------------------------


class FilmScoringEngine:
    """Generate MIDI from a list of `VideoFeatures` (one per scene).

    The engine is video-agnostic: it never opens the video file. Run
    `analyzer.VLMVideoAnalyzer` (or any other producer of `VideoFeatures`) and
    pass the list to `generate_score`.
    """

    def __init__(self, bpm: int = 120, framerate: float = 24.0) -> None:
        self.bpm = bpm
        self.framerate = framerate
        self.techniques = FilmScoringTechniques()
        self.leitmotif_engine = LeitmotifEngine()

    @staticmethod
    def tension_arc_from_features(
        features: List[VideoFeatures],
        smoothing: float = 0.3,
    ) -> TensionArc:
        if not features:
            return TensionArc(timestamps=[0.0], tension_values=[0.5])
        ts = [f.start_time for f in features]
        vs = [f.visual_tension for f in features]
        if smoothing > 0 and len(vs) > 2:
            try:
                import numpy as np

                window = max(1, int(len(vs) * smoothing))
                vs = np.convolve(vs, np.ones(window) / window, mode="same").tolist()
            except ImportError:
                pass
        return TensionArc(timestamps=ts, tension_values=vs)

    def _progression_for(
        self,
        features: VideoFeatures,
        base_progression: Optional[Dict[int, str]],
    ) -> Dict[int, str]:
        if base_progression:
            return self.techniques.morph_progression(
                base_progression, features.mood, features.visual_tension
            )

        scale = self.techniques.mood_to_scale_context(features.mood)
        quality = self.techniques.tension_to_chord_complexity(features.visual_tension)
        # Default I–IV–V–I in the chosen modality, coloured by tension.
        if scale in {"minor", "harmonic_minor"}:
            return {0: f"Cm", 4: f"Fm", 8: f"G{quality}", 12: f"Cm"}
        return {0: f"C{quality}", 4: f"F{quality}", 8: f"G{quality}", 12: f"C{quality}"}

    def generate_score(
        self,
        features: List[VideoFeatures],
        base_progression: Optional[Dict[int, str]] = None,
        scoring_approach: ScoringSyncType = ScoringSyncType.TENSION_ARC,
        output_path: Optional[str] = None,
    ) -> str:
        """Render scenes → one MIDI file. Returns the output path.

        Each scene gets its own track. Within a track, the chord progression
        repeats to fill the scene's duration. Notes are written sustained for
        a beat each (quarter-note rhythm) — the engine is a starting point;
        downstream consumers can rearrange via the chord_progression_generator
        module if you want voicings, inversions, etc.
        """
        ticks_per_beat = 480
        seconds_per_beat = 60.0 / max(self.bpm, 1)
        beats_per_chord = 1  # one beat per chord-symbol

        midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        meta = mido.MidiTrack()
        midi.tracks.append(meta)
        meta.append(mido.MetaMessage("set_tempo", tempo=int(60_000_000 / self.bpm), time=0))

        for idx, feat in enumerate(features):
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.append(mido.MetaMessage(
                "track_name",
                name=f"scene_{idx:02d}_{feat.mood.value}_t{feat.visual_tension:.2f}",
                time=0,
            ))

            prog = self._progression_for(feat, base_progression)
            if not prog:
                continue

            chord_seq = [prog[k] for k in sorted(prog.keys())]
            n_beats = max(1, int(round(feat.duration / seconds_per_beat)))
            velocity = int(40 + 60 * max(0.0, min(1.0, feat.visual_tension)))

            for beat_idx in range(n_beats):
                chord = chord_seq[beat_idx % len(chord_seq)]
                notes = _chord_to_midi_notes(chord, octave=4)
                for n in notes:
                    track.append(mido.Message("note_on", note=n, velocity=velocity, time=0))
                # Hold for `beats_per_chord` beats — dt only on the *first*
                # note_off so the chord rings together.
                first = True
                for n in notes:
                    track.append(mido.Message(
                        "note_off",
                        note=n,
                        velocity=0,
                        time=ticks_per_beat * beats_per_chord if first else 0,
                    ))
                    first = False

        if output_path is None:
            output_path = str(Path(tempfile.gettempdir()) / "film_score.mid")
        midi.save(output_path)
        return output_path
