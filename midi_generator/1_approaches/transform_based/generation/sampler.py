"""Pattern and Transform sampling for generation."""

import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Pattern:
    """A pattern from the grammar."""
    id: int
    pitch_intervals: List[int]
    rhythm_bucket: int
    velocity_bucket: int
    count: int
    is_hierarchical: bool
    contour: List[int]


class PatternSampler:
    """Sample patterns from the grammar vocabulary."""

    def __init__(self, patterns_json_path: str):
        with open(patterns_json_path, 'r') as f:
            raw_patterns = json.load(f)

        self.patterns: Dict[int, Pattern] = {}
        self.pattern_ids: List[int] = []
        self.pattern_weights: List[float] = []

        for pid_str, data in raw_patterns.items():
            pid = int(pid_str)
            pattern = Pattern(
                id=pid,
                pitch_intervals=data.get('pitch_intervals', []),
                rhythm_bucket=data.get('rhythm_bucket', 8),
                velocity_bucket=data.get('velocity_bucket', 3),
                count=data.get('count', 1),
                is_hierarchical=data.get('is_hierarchical', False),
                contour=data.get('contour', [0, 8, 3]),
            )
            self.patterns[pid] = pattern
            self.pattern_ids.append(pid)
            self.pattern_weights.append(float(pattern.count))

        total_weight = sum(self.pattern_weights)
        self.pattern_weights = [w / total_weight for w in self.pattern_weights]

    def sample(self, weighted: bool = True) -> Pattern:
        """Sample a pattern, optionally weighted by corpus frequency."""
        if weighted:
            pid = random.choices(self.pattern_ids, weights=self.pattern_weights, k=1)[0]
        else:
            pid = random.choice(self.pattern_ids)
        return self.patterns[pid]

    def sample_n(self, n: int, weighted: bool = True) -> List[Pattern]:
        """Sample n patterns."""
        return [self.sample(weighted=weighted) for _ in range(n)]

    def get_pattern(self, pid: int) -> Optional[Pattern]:
        """Get a specific pattern by ID."""
        return self.patterns.get(pid)

    def get_similar_patterns(self, pattern: Pattern, n: int = 5) -> List[Pattern]:
        """Find patterns with similar contour."""
        target_len = len(pattern.pitch_intervals)
        similar = []

        for p in self.patterns.values():
            if p.id == pattern.id:
                continue
            if len(p.pitch_intervals) == target_len:
                similar.append(p)

        if not similar:
            return self.sample_n(n)

        random.shuffle(similar)
        return similar[:n]


class TransformSampler:
    """Sample transforms and meta-patterns."""

    def __init__(self, transforms_json_path: str, meta_patterns_json_path: str):
        with open(transforms_json_path, 'r') as f:
            transforms_data = json.load(f)

        self.vocabulary: List[str] = transforms_data.get('vocabulary', [])
        self.stats: Dict[str, dict] = transforms_data.get('stats', {})

        self.transform_weights: List[float] = []
        for t in self.vocabulary:
            freq = self.stats.get(t, {}).get('frequency', 1)
            self.transform_weights.append(float(freq))

        total = sum(self.transform_weights)
        if total > 0:
            self.transform_weights = [w / total for w in self.transform_weights]
        else:
            self.transform_weights = [1.0 / len(self.vocabulary)] * len(self.vocabulary)

        with open(meta_patterns_json_path, 'r') as f:
            meta_data = json.load(f)

        self.meta_rules: Dict[str, dict] = meta_data.get('rules', {})
        self.orchestration_rules: List[dict] = meta_data.get('orchestration_rules', [])

        self._build_meta_sequences()

    def _build_meta_sequences(self):
        """Convert meta-pattern transform_ids to transform names."""
        self.meta_sequences: List[List[str]] = []

        for rule_name, rule_data in self.meta_rules.items():
            transform_ids = rule_data.get('transform_ids', [])
            sequence = []
            for tid in transform_ids:
                if 0 <= tid < len(self.vocabulary):
                    sequence.append(self.vocabulary[tid])
                else:
                    sequence.append('identity')
            if sequence:
                self.meta_sequences.append(sequence)

    def sample_transform(self, weighted: bool = True) -> str:
        """Sample a single transform."""
        if weighted and self.transform_weights:
            return random.choices(self.vocabulary, weights=self.transform_weights, k=1)[0]
        return random.choice(self.vocabulary)

    def sample_transform_sequence(self, length: int, use_meta: bool = True) -> List[str]:
        """Generate a transform sequence.

        If use_meta=True, chains meta-patterns together.
        Otherwise samples individual transforms.
        """
        if not use_meta or not self.meta_sequences:
            return [self.sample_transform() for _ in range(length)]

        sequence = []
        while len(sequence) < length:
            meta = random.choice(self.meta_sequences)
            sequence.extend(meta)

        return sequence[:length]

    def get_transform_delta(self, transform: str) -> Tuple[int, bool, bool]:
        """Parse transform into (transposition, inversion, retrograde).

        Returns:
            (pitch_delta, is_inverted, is_retrograde)
        """
        pitch_delta = 0
        is_inverted = False
        is_retrograde = False

        parts = transform.replace('\u2218', ',').split(',')

        for part in parts:
            part = part.strip()
            if part == 'identity':
                continue
            elif part == 'R':
                is_retrograde = not is_retrograde
            elif part.startswith('I'):
                is_inverted = not is_inverted
                if len(part) > 1:
                    try:
                        pitch_delta += int(part[1:])
                    except ValueError:
                        pass
            elif part.startswith('T'):
                if len(part) > 1:
                    try:
                        pitch_delta += int(part[1:])
                    except ValueError:
                        pass

        return (pitch_delta % 12, is_inverted, is_retrograde)

    def apply_transform_to_intervals(self, intervals: List[int], transform: str) -> List[int]:
        """Apply a transform to a list of pitch intervals."""
        pitch_delta, is_inverted, is_retrograde = self.get_transform_delta(transform)

        result = intervals.copy()

        if is_inverted:
            result = [-i for i in result]

        if is_retrograde:
            result = result[::-1]

        return result
