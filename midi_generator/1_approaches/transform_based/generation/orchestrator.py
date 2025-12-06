"""Multi-track orchestration using orchestration rules and TrackDerive relationships."""

import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .sampler import TransformSampler


# GM Program to instrument name mapping
GM_PROGRAMS = {
    0: 'Acoustic Grand Piano', 1: 'Bright Acoustic Piano',
    24: 'Acoustic Guitar (nylon)', 25: 'Acoustic Guitar (steel)',
    26: 'Electric Guitar (jazz)', 27: 'Electric Guitar (clean)',
    32: 'Acoustic Bass', 33: 'Electric Bass (finger)',
    40: 'Violin', 41: 'Viola', 42: 'Cello', 43: 'Contrabass',
    56: 'Trumpet', 57: 'Trombone', 58: 'Tuba', 59: 'Muted Trumpet',
    60: 'French Horn', 61: 'Brass Section',
    64: 'Soprano Sax', 65: 'Alto Sax', 66: 'Tenor Sax', 67: 'Baritone Sax',
    68: 'Oboe', 69: 'English Horn', 70: 'Bassoon', 71: 'Clarinet',
    72: 'Piccolo', 73: 'Flute', 74: 'Recorder',
}

# Instrument name to GM program (reverse lookup)
INSTRUMENT_TO_GM = {v: k for k, v in GM_PROGRAMS.items()}

# Common big band instruments
BIG_BAND_INSTRUMENTS = [
    'Trumpet', 'Trombone', 'Alto Sax', 'Tenor Sax', 'Baritone Sax',
    'Piano', 'Acoustic Bass', 'Drums'
]

# Octave offset by instrument register (semitones relative to middle C = 60)
# This helps differentiate tracks even when they have identity transforms
INSTRUMENT_OCTAVE_OFFSET = {
    'Piccolo': 12, 'Flute': 0, 'Clarinet': 0, 'Oboe': 0,
    'Alto Sax': 0, 'Soprano Sax': 0,
    'Trumpet': 0, 'Muted Trumpet': 0,
    'Tenor Sax': -12, 'French Horn': -12,
    'Trombone': -12, 'Bassoon': -12,
    'Baritone Sax': -12,
    'Tuba': -24, 'Contrabass': -24, 'Acoustic Bass': -24, 'Electric Bass (finger)': -24,
    'Piano': 0, 'Acoustic Grand Piano': 0,
}


@dataclass
class OrchestrationRule:
    """A rule for deriving one instrument from another."""
    leader: str
    follower: str
    transform: str
    confidence: float = 1.0
    count: int = 0


@dataclass
class TrackSegment:
    """A segment of a track (one pattern instance)."""
    pattern_id: int
    pitch_intervals: List[int]
    pitch_offset: int  # Semitones from C (pitch class)
    rhythm_bucket: int
    velocity_bucket: int
    onset_time: int = 0
    duration: int = 0


@dataclass
class Track:
    """A track containing multiple segments."""
    instrument: str
    gm_program: int
    segments: List[TrackSegment] = field(default_factory=list)
    is_drum: bool = False
    octave_offset: int = 0  # Semitones to offset the entire track (for register)


class Orchestrator:
    """Derive multiple tracks using orchestration rules."""

    def __init__(self, meta_patterns_json_path: str, transform_sampler: TransformSampler):
        self.transform_sampler = transform_sampler

        with open(meta_patterns_json_path, 'r') as f:
            meta_data = json.load(f)

        self.orchestration_rules: Dict[Tuple[str, str], OrchestrationRule] = {}
        raw_rules = meta_data.get('orchestration_rules', [])

        for rule in raw_rules:
            # Support both old format (leader/follower) and new format (source_name/target_name)
            leader = rule.get('source_name', rule.get('leader', 'Piano'))
            follower = rule.get('target_name', rule.get('follower', 'Piano'))
            transform = rule.get('transform', 'identity')
            confidence = rule.get('confidence', 1.0)
            count = rule.get('frequency', rule.get('count', 0))

            key = (leader, follower)
            self.orchestration_rules[key] = OrchestrationRule(
                leader=leader,
                follower=follower,
                transform=transform,
                confidence=confidence,
                count=count,
            )

        self._build_leader_followers()

    def _build_leader_followers(self):
        """Build lookup of which instruments can follow which leaders."""
        self.leader_to_followers: Dict[str, List[str]] = {}
        self.follower_to_leaders: Dict[str, List[str]] = {}

        for (leader, follower), rule in self.orchestration_rules.items():
            if leader not in self.leader_to_followers:
                self.leader_to_followers[leader] = []
            self.leader_to_followers[leader].append(follower)

            if follower not in self.follower_to_leaders:
                self.follower_to_leaders[follower] = []
            self.follower_to_leaders[follower].append(leader)

    def get_rule(self, leader: str, follower: str) -> Optional[OrchestrationRule]:
        """Get the orchestration rule for a leader-follower pair."""
        return self.orchestration_rules.get((leader, follower))

    def derive_track(self, lead_track: Track, follower_instrument: str) -> Track:
        """Derive a follower track from a leader track.

        Uses orchestration rules if available, otherwise uses identity transform.
        Also applies register-appropriate octave offsets for differentiation.
        """
        rule = self.get_rule(lead_track.instrument, follower_instrument)

        if rule:
            transform = rule.transform
        else:
            transform = 'identity'

        # Calculate octave offset based on instrument registers
        leader_octave = INSTRUMENT_OCTAVE_OFFSET.get(lead_track.instrument, 0)
        follower_octave = INSTRUMENT_OCTAVE_OFFSET.get(follower_instrument, 0)
        octave_offset = follower_octave - leader_octave

        derived_segments = []
        for segment in lead_track.segments:
            new_intervals = self.transform_sampler.apply_transform_to_intervals(
                segment.pitch_intervals, transform
            )
            pitch_delta, _, _ = self.transform_sampler.get_transform_delta(transform)

            # Apply both transform pitch delta and octave offset
            total_offset = pitch_delta + octave_offset

            derived_segment = TrackSegment(
                pattern_id=segment.pattern_id,
                pitch_intervals=new_intervals,
                pitch_offset=(segment.pitch_offset + total_offset) % 12,
                rhythm_bucket=segment.rhythm_bucket,
                velocity_bucket=segment.velocity_bucket,
                onset_time=segment.onset_time,
                duration=segment.duration,
            )
            derived_segments.append(derived_segment)

        gm = INSTRUMENT_TO_GM.get(follower_instrument, 0)

        return Track(
            instrument=follower_instrument,
            gm_program=gm,
            segments=derived_segments,
            is_drum=False,
            octave_offset=octave_offset,  # Store for MIDI output
        )

    def orchestrate(
        self,
        lead_track: Track,
        follower_instruments: List[str],
        variation: float = 0.0
    ) -> Dict[str, Track]:
        """Generate multiple tracks from a lead track.

        Args:
            lead_track: The leader track
            follower_instruments: List of instruments to derive
            variation: 0.0 = use exact rules, 1.0 = random transforms

        Returns:
            Dict mapping instrument name to Track
        """
        tracks = {lead_track.instrument: lead_track}

        for follower in follower_instruments:
            if follower == lead_track.instrument:
                continue

            derived = self.derive_track(lead_track, follower)

            if variation > 0 and random.random() < variation:
                for seg in derived.segments:
                    extra_transform = self.transform_sampler.sample_transform()
                    seg.pitch_intervals = self.transform_sampler.apply_transform_to_intervals(
                        seg.pitch_intervals, extra_transform
                    )
                    delta, _, _ = self.transform_sampler.get_transform_delta(extra_transform)
                    seg.pitch_offset = (seg.pitch_offset + delta) % 12

            tracks[follower] = derived

        return tracks

    def suggest_followers(self, leader: str, n: int = 4) -> List[str]:
        """Suggest follower instruments based on orchestration rules."""
        followers = self.leader_to_followers.get(leader, [])

        if len(followers) >= n:
            return random.sample(followers, n)

        all_instruments = list(INSTRUMENT_TO_GM.keys())
        remaining = [i for i in all_instruments if i not in followers and i != leader]
        additional = random.sample(remaining, min(n - len(followers), len(remaining)))

        return followers + additional
