"""Main ArrangementGenerator class for generating multitrack MIDI arrangements."""

import os
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from .sampler import PatternSampler, TransformSampler, Pattern
from .orchestrator import Orchestrator, Track, TrackSegment, INSTRUMENT_TO_GM, BIG_BAND_INSTRUMENTS
from .timing import TimingAssigner
from .midi_output import MIDIWriter


class ArrangementGenerator:
    """Generate multitrack arrangements from v53 checkpoint."""

    def __init__(self, checkpoint_path: str):
        """Initialize generator from checkpoint.

        Args:
            checkpoint_path: Path to .npz checkpoint file
        """
        self.checkpoint_path = checkpoint_path
        checkpoint_dir = os.path.dirname(checkpoint_path)
        base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]

        cp = np.load(checkpoint_path, allow_pickle=True)

        patterns_file = str(cp['patterns_json_file'].item())
        if not os.path.isabs(patterns_file):
            patterns_file = os.path.join(checkpoint_dir, patterns_file)

        transforms_file = str(cp['transforms_json_file'].item())
        if not os.path.isabs(transforms_file):
            transforms_file = os.path.join(checkpoint_dir, transforms_file)

        meta_file = str(cp['meta_patterns_json_file'].item())
        if not os.path.isabs(meta_file):
            meta_file = os.path.join(checkpoint_dir, meta_file)

        self.pattern_sampler = PatternSampler(patterns_file)
        self.transform_sampler = TransformSampler(transforms_file, meta_file)
        self.orchestrator = Orchestrator(meta_file, self.transform_sampler)
        self.timing = TimingAssigner()
        self.midi_writer = MIDIWriter()

    def generate(
        self,
        seed_pattern_id: Optional[int] = None,
        n_patterns: int = 16,
        instruments: Optional[List[str]] = None,
        base_pitch: int = 60,
        use_meta_patterns: bool = True,
        variation: float = 0.0,
    ) -> Dict[str, Track]:
        """Generate a multitrack arrangement.

        Args:
            seed_pattern_id: Starting pattern ID (random if None)
            n_patterns: Number of patterns to chain horizontally
            instruments: List of instruments (default: big band section)
            base_pitch: Base MIDI pitch (60 = middle C)
            use_meta_patterns: Use meta-patterns for transform sequences
            variation: 0.0 = exact rules, 1.0 = random variations

        Returns:
            Dict mapping instrument name to Track
        """
        if instruments is None:
            instruments = ['Trumpet', 'Trombone', 'Alto Sax', 'Tenor Sax', 'Piano']

        if seed_pattern_id is not None:
            seed_pattern = self.pattern_sampler.get_pattern(seed_pattern_id)
            if seed_pattern is None:
                seed_pattern = self.pattern_sampler.sample()
        else:
            seed_pattern = self.pattern_sampler.sample()

        transform_sequence = self.transform_sampler.sample_transform_sequence(
            n_patterns, use_meta=use_meta_patterns
        )

        lead_track = self._develop_horizontal(seed_pattern, transform_sequence)
        lead_track.instrument = instruments[0]
        lead_track.gm_program = INSTRUMENT_TO_GM.get(instruments[0], 0)

        self.timing.assign_timing(lead_track)

        tracks = self.orchestrator.orchestrate(
            lead_track,
            instruments[1:],
            variation=variation
        )

        for name, track in tracks.items():
            self.timing.assign_timing(track)

        return tracks

    def _develop_horizontal(
        self,
        seed_pattern: Pattern,
        transform_sequence: List[str]
    ) -> Track:
        """Develop melody horizontally by chaining patterns with transforms."""
        segments = []
        current_pattern = seed_pattern
        current_pitch_offset = 0

        segment = TrackSegment(
            pattern_id=current_pattern.id,
            pitch_intervals=current_pattern.pitch_intervals.copy(),
            pitch_offset=current_pitch_offset,
            rhythm_bucket=current_pattern.rhythm_bucket,
            velocity_bucket=current_pattern.velocity_bucket,
        )
        segments.append(segment)

        for transform in transform_sequence:
            new_intervals = self.transform_sampler.apply_transform_to_intervals(
                current_pattern.pitch_intervals, transform
            )
            pitch_delta, is_inverted, is_retrograde = self.transform_sampler.get_transform_delta(transform)

            current_pitch_offset = (current_pitch_offset + pitch_delta) % 12

            next_pattern = self._find_matching_pattern(new_intervals)
            if next_pattern is None:
                next_pattern = current_pattern

            segment = TrackSegment(
                pattern_id=next_pattern.id,
                pitch_intervals=new_intervals,
                pitch_offset=current_pitch_offset,
                rhythm_bucket=next_pattern.rhythm_bucket,
                velocity_bucket=next_pattern.velocity_bucket,
            )
            segments.append(segment)

            current_pattern = next_pattern

        return Track(
            instrument='Lead',
            gm_program=0,
            segments=segments,
        )

    def _find_matching_pattern(self, intervals: List[int]) -> Optional[Pattern]:
        """Find a pattern matching the given intervals."""
        for pattern in self.pattern_sampler.patterns.values():
            if pattern.pitch_intervals == intervals:
                return pattern
        return None

    def save_midi(
        self,
        tracks: Dict[str, Track],
        output_path: str,
        base_pitch: int = 60
    ) -> str:
        """Save tracks to MIDI file."""
        return self.midi_writer.save(tracks, output_path, base_pitch)

    def generate_and_save(
        self,
        output_path: str,
        seed_pattern_id: Optional[int] = None,
        n_patterns: int = 16,
        instruments: Optional[List[str]] = None,
        base_pitch: int = 60,
        use_meta_patterns: bool = True,
        variation: float = 0.0,
    ) -> str:
        """Generate arrangement and save to MIDI file."""
        tracks = self.generate(
            seed_pattern_id=seed_pattern_id,
            n_patterns=n_patterns,
            instruments=instruments,
            base_pitch=base_pitch,
            use_meta_patterns=use_meta_patterns,
            variation=variation,
        )
        return self.save_midi(tracks, output_path, base_pitch)


class RemixGenerator(ArrangementGenerator):
    """Generate variations of existing arrangements."""

    def remix(
        self,
        source_tracks: Dict[str, Track],
        variation_level: float = 0.5
    ) -> Dict[str, Track]:
        """Generate variations of source tracks.

        Args:
            source_tracks: Original tracks
            variation_level: 0.0 = keep original, 1.0 = fully random

        Returns:
            Remixed tracks
        """
        import random
        remixed = {}

        for name, track in source_tracks.items():
            new_segments = []

            for segment in track.segments:
                r = random.random()

                if r < (1 - variation_level):
                    new_segments.append(segment)
                elif r < (1 - variation_level / 2):
                    transform = self.transform_sampler.sample_transform()
                    new_intervals = self.transform_sampler.apply_transform_to_intervals(
                        segment.pitch_intervals, transform
                    )
                    delta, _, _ = self.transform_sampler.get_transform_delta(transform)
                    new_segment = TrackSegment(
                        pattern_id=segment.pattern_id,
                        pitch_intervals=new_intervals,
                        pitch_offset=(segment.pitch_offset + delta) % 12,
                        rhythm_bucket=segment.rhythm_bucket,
                        velocity_bucket=segment.velocity_bucket,
                        onset_time=segment.onset_time,
                        duration=segment.duration,
                    )
                    new_segments.append(new_segment)
                else:
                    similar = self.pattern_sampler.get_similar_patterns(
                        Pattern(
                            id=segment.pattern_id,
                            pitch_intervals=segment.pitch_intervals,
                            rhythm_bucket=segment.rhythm_bucket,
                            velocity_bucket=segment.velocity_bucket,
                            count=1,
                            is_hierarchical=False,
                            contour=[0, 8, 3],
                        ),
                        n=1
                    )
                    if similar:
                        sub = similar[0]
                        new_segment = TrackSegment(
                            pattern_id=sub.id,
                            pitch_intervals=sub.pitch_intervals,
                            pitch_offset=segment.pitch_offset,
                            rhythm_bucket=sub.rhythm_bucket,
                            velocity_bucket=sub.velocity_bucket,
                            onset_time=segment.onset_time,
                            duration=segment.duration,
                        )
                        new_segments.append(new_segment)
                    else:
                        new_segments.append(segment)

            remixed[name] = Track(
                instrument=track.instrument,
                gm_program=track.gm_program,
                segments=new_segments,
                is_drum=track.is_drum,
            )

        return remixed
