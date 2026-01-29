"""Timing assignment for generated patterns."""

from typing import Dict, List
from .orchestrator import Track, TrackSegment


# Rhythm bucket to ticks mapping (at 480 ticks per beat)
BUCKET_TO_TICKS = {
    0: 30,     # 32nd note
    1: 40,
    2: 60,     # 16th note triplet
    3: 80,
    4: 120,    # 16th note
    5: 160,    # 8th note triplet
    6: 180,
    7: 200,
    8: 240,    # 8th note
    9: 280,
    10: 320,   # quarter note triplet
    11: 360,
    12: 480,   # quarter note
    13: 640,
    14: 720,
    15: 960,   # half note
}


class TimingAssigner:
    """Assign timing to track segments."""

    def __init__(self, ticks_per_beat: int = 480):
        self.ticks_per_beat = ticks_per_beat
        self.tempo = 120

    def bucket_to_ioi(self, rhythm_bucket: int) -> int:
        """Convert rhythm bucket to inter-onset interval in ticks."""
        if rhythm_bucket in BUCKET_TO_TICKS:
            return BUCKET_TO_TICKS[rhythm_bucket]

        if rhythm_bucket < 0:
            return BUCKET_TO_TICKS[0]
        if rhythm_bucket > 15:
            return BUCKET_TO_TICKS[15]

        lower = max(k for k in BUCKET_TO_TICKS.keys() if k <= rhythm_bucket)
        upper = min(k for k in BUCKET_TO_TICKS.keys() if k >= rhythm_bucket)
        ratio = (rhythm_bucket - lower) / (upper - lower) if upper != lower else 0
        return int(BUCKET_TO_TICKS[lower] + ratio * (BUCKET_TO_TICKS[upper] - BUCKET_TO_TICKS[lower]))

    def assign_timing(self, track: Track) -> Track:
        """Assign onset times and durations to track segments.

        Segments are placed sequentially based on their rhythm bucket.
        """
        current_time = 0

        for segment in track.segments:
            segment.onset_time = current_time

            n_notes = len(segment.pitch_intervals) + 1
            ioi = self.bucket_to_ioi(segment.rhythm_bucket)
            segment.duration = ioi * n_notes

            current_time += segment.duration

        return track

    def assign_timing_to_tracks(self, tracks: Dict[str, Track]) -> Dict[str, Track]:
        """Assign timing to all tracks, aligning them together."""
        for name, track in tracks.items():
            self.assign_timing(track)
        return tracks

    def stretch_to_bars(self, track: Track, n_bars: int, beats_per_bar: int = 4) -> Track:
        """Stretch/compress track timing to fit exactly n_bars."""
        total_ticks = n_bars * beats_per_bar * self.ticks_per_beat

        if not track.segments:
            return track

        current_duration = sum(s.duration for s in track.segments)
        if current_duration == 0:
            return track

        ratio = total_ticks / current_duration

        current_time = 0
        for segment in track.segments:
            segment.onset_time = int(current_time)
            segment.duration = int(segment.duration * ratio)
            current_time += segment.duration

        return track

    def quantize_to_grid(
        self,
        track: Track,
        grid_ticks: int = 240,  # 8th note grid
        strength: float = 1.0
    ) -> Track:
        """Quantize onset times to a grid.

        Args:
            track: Track to quantize
            grid_ticks: Grid size in ticks (240 = 8th note at 480 TPB)
            strength: 0.0 = no quantization, 1.0 = snap to grid
        """
        for segment in track.segments:
            raw_time = segment.onset_time
            quantized = round(raw_time / grid_ticks) * grid_ticks
            segment.onset_time = int(raw_time + strength * (quantized - raw_time))
        return track
