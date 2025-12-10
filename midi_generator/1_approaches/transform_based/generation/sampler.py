"""Pattern and Transform sampling for generation."""

import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field


# Instrument name to GM program mapping
INSTRUMENT_TO_GM = {
    'Acoustic Grand Piano': 0, 'Piano': 0,
    'Bright Acoustic Piano': 1,
    'Acoustic Guitar (nylon)': 24, 'Acoustic Guitar (steel)': 25,
    'Electric Guitar (jazz)': 26, 'Electric Guitar (clean)': 27,
    'Acoustic Bass': 32, 'Electric Bass (finger)': 33,
    'Violin': 40, 'Viola': 41, 'Cello': 42, 'Contrabass': 43,
    'Trumpet': 56, 'Trombone': 57, 'Tuba': 58, 'Muted Trumpet': 59,
    'French Horn': 60, 'Brass Section': 61,
    'Soprano Sax': 64, 'Alto Sax': 65, 'Tenor Sax': 66, 'Baritone Sax': 67,
    'Oboe': 68, 'English Horn': 69, 'Bassoon': 70, 'Clarinet': 71,
    'Piccolo': 72, 'Flute': 73, 'Recorder': 74,
}

# Instrument families for fallback sampling
INSTRUMENT_FAMILIES = {
    'brass': [56, 57, 58, 59, 60, 61],  # Trumpet, Trombone, Tuba, etc.
    'sax': [64, 65, 66, 67],  # Soprano, Alto, Tenor, Baritone Sax
    'woodwind': [68, 69, 70, 71, 72, 73, 74],  # Oboe, Clarinet, Flute, etc.
    'bass': [32, 33, 43],  # Acoustic Bass, Electric Bass, Contrabass
    'piano': [0, 1, 2, 3, 4, 5, 6, 7],  # Piano family
    'strings': [40, 41, 42, 43],  # Violin, Viola, Cello, Contrabass
}


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
    gm_programs: Set[int] = field(default_factory=set)  # Which instruments play this


class PatternSampler:
    """Sample patterns from the grammar vocabulary."""

    def __init__(self, patterns_json_path: str):
        with open(patterns_json_path, 'r') as f:
            raw_patterns = json.load(f)

        self.patterns: Dict[int, Pattern] = {}
        self.pattern_ids: List[int] = []
        self.pattern_weights: List[float] = []

        # Index patterns by GM program
        self.gm_to_patterns: Dict[int, List[int]] = defaultdict(list)
        self.gm_to_weights: Dict[int, List[float]] = defaultdict(list)

        for pid_str, data in raw_patterns.items():
            pid = int(pid_str)

            # Collect GM programs from occurrences
            gm_programs = set()
            gm_occurrence_counts = defaultdict(int)
            for occ in data.get('occurrences', []):
                gm = occ.get('gm_program', 0)
                gm_programs.add(gm)
                gm_occurrence_counts[gm] += 1

            pattern = Pattern(
                id=pid,
                pitch_intervals=data.get('pitch_intervals', []),
                rhythm_bucket=data.get('rhythm_bucket', 8),
                velocity_bucket=data.get('velocity_bucket', 3),
                count=data.get('count', 1),
                is_hierarchical=data.get('is_hierarchical', False),
                contour=data.get('contour', [0, 8, 3]),
                gm_programs=gm_programs,
            )
            self.patterns[pid] = pattern
            self.pattern_ids.append(pid)
            self.pattern_weights.append(float(pattern.count))

            # Index by GM program with per-instrument occurrence counts as weights
            for gm, count in gm_occurrence_counts.items():
                self.gm_to_patterns[gm].append(pid)
                self.gm_to_weights[gm].append(float(count))

        total_weight = sum(self.pattern_weights)
        self.pattern_weights = [w / total_weight for w in self.pattern_weights]

        # Normalize per-GM weights
        for gm in self.gm_to_weights:
            total = sum(self.gm_to_weights[gm])
            if total > 0:
                self.gm_to_weights[gm] = [w / total for w in self.gm_to_weights[gm]]

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

    def sample_for_instrument(self, instrument: str, weighted: bool = True) -> Optional[Pattern]:
        """Sample a pattern appropriate for a specific instrument.

        Args:
            instrument: Instrument name (e.g., 'Trumpet', 'Acoustic Bass')
            weighted: Weight by corpus frequency for this instrument

        Returns:
            Pattern that this instrument plays in the corpus, or None if not found
        """
        gm = INSTRUMENT_TO_GM.get(instrument)
        if gm is None:
            return self.sample(weighted=weighted)

        # Try exact GM program first
        if gm in self.gm_to_patterns and self.gm_to_patterns[gm]:
            pattern_ids = self.gm_to_patterns[gm]
            weights = self.gm_to_weights[gm] if weighted else None
            if weighted and weights:
                pid = random.choices(pattern_ids, weights=weights, k=1)[0]
            else:
                pid = random.choice(pattern_ids)
            return self.patterns[pid]

        # Fall back to instrument family
        family = self._get_instrument_family(gm)
        if family:
            combined_patterns = []
            combined_weights = []
            for family_gm in INSTRUMENT_FAMILIES[family]:
                if family_gm in self.gm_to_patterns:
                    combined_patterns.extend(self.gm_to_patterns[family_gm])
                    combined_weights.extend(self.gm_to_weights[family_gm])

            if combined_patterns:
                if weighted and combined_weights:
                    total = sum(combined_weights)
                    normalized = [w / total for w in combined_weights]
                    pid = random.choices(combined_patterns, weights=normalized, k=1)[0]
                else:
                    pid = random.choice(combined_patterns)
                return self.patterns[pid]

        # Last resort: sample any pattern
        return self.sample(weighted=weighted)

    def _get_instrument_family(self, gm: int) -> Optional[str]:
        """Get the instrument family for a GM program."""
        for family, programs in INSTRUMENT_FAMILIES.items():
            if gm in programs:
                return family
        return None

    def sample_n_for_instrument(self, instrument: str, n: int, weighted: bool = True) -> List[Pattern]:
        """Sample n patterns appropriate for a specific instrument."""
        return [self.sample_for_instrument(instrument, weighted=weighted) for _ in range(n)]

    def get_patterns_for_instrument(self, instrument: str) -> List[Pattern]:
        """Get all patterns played by a specific instrument."""
        gm = INSTRUMENT_TO_GM.get(instrument)
        if gm is None or gm not in self.gm_to_patterns:
            return []
        return [self.patterns[pid] for pid in self.gm_to_patterns[gm]]


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
