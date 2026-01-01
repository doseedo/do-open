"""
Grammar-Based MIDI Generator using discovered structure.

NO TRAINING - uses track derives and patterns directly.

Algorithm:
1. Choose a leader instrument (alto/tenor sax, trumpet, trombone)
2. Sample patterns for the leader track from corpus frequency
3. Derive follower tracks using instrument pair transforms
4. Convert all patterns to MIDI using pattern lookup

Structure used:
- track_derives.json: (src_inst, src_pattern) -> (tgt_inst, tgt_pattern, transform)
- patterns.json: pattern_name -> {pitch_intervals, gm_program, ...}

KEY: Uses first_pitch from pattern occurrences for absolute pitch positioning,
ensuring instruments play in their natural registers.
"""

import json
import random
import re
import orjson
from pathlib import Path
from collections import defaultdict
import pretty_midi

# GM instrument names
GM_NAMES = {
    0: 'piano', 24: 'nylon_guitar', 25: 'steel_guitar', 27: 'clean_guitar',
    32: 'acoustic_bass', 33: 'electric_bass', 56: 'trumpet', 57: 'trombone',
    60: 'french_horn', 65: 'alto_sax', 66: 'tenor_sax', 67: 'baritone_sax',
    71: 'clarinet', 72: 'flute', 73: 'piccolo', 128: 'drums'
}

# Typical pitch ranges for instruments (MIDI pitch)
INSTRUMENT_RANGES = {
    0: (36, 96),    # piano - wide range
    32: (28, 55),   # acoustic bass - E1 to G3
    33: (28, 55),   # electric bass - E1 to G3
    56: (55, 82),   # trumpet - G3 to Bb5
    57: (40, 72),   # trombone - E2 to C5
    60: (34, 77),   # french horn - Bb1 to F5
    65: (49, 80),   # alto sax - Db3 to Ab5
    66: (44, 75),   # tenor sax - Ab2 to Eb5
    67: (36, 68),   # baritone sax - Db2 to Ab4
}


class GrammarGenerator:
    """Generate MIDI using discovered grammar rules (no ML)."""

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.patterns = None
        self.derives = None

        # Build indexes
        self.pattern_by_gm = defaultdict(list)  # gm -> [pattern_names]
        self.derive_index = {}  # (src_pattern, src_inst, tgt_inst) -> (tgt_pattern, transform)
        self.inst_pair_transforms = defaultdict(list)  # (src_inst, tgt_inst) -> [(transform, count)]

        # Mapping from numeric indices to pattern string keys
        # (derives use numeric indices, patterns use string keys)
        self.idx_to_pattern_key = {}  # int index -> string key like "GM57_151"

        # Leader instruments (most frequently source in derives)
        self.leader_instruments = [65, 66, 56, 57, 33, 0]  # sax, brass, bass, piano

        # Accompaniment instruments (added independently, not derived)
        self.accompaniment_instruments = [0, 32]  # piano, acoustic bass (double bass)

        # Co-occurrence index: PIECE-AWARE
        # leader_pattern -> [{piece, piano: [(pat, offset)], bass: [...]}]
        # Using same-piece data preserves harmonic context
        self.co_occurrence = {}

        # Global co-occurrence (fallback when piece-aware has no data)
        # Aggregated across all pieces - still 78% consonant
        self.global_co_occurrence = {}

        self.fallback_piano = []  # Most common piano patterns (last resort)
        self.fallback_bass = []   # Most common bass patterns (last resort)

    def load(self):
        """Load patterns and derives."""
        print("Loading patterns...")
        with open(self.base_path / 'checkpoint_v55_pure_contour_1000files_patterns.json', 'rb') as f:
            self.patterns = orjson.loads(f.read())
        print(f"  Loaded {len(self.patterns)} patterns")

        # Build index-to-key mapping (derives use numeric indices, patterns use string keys)
        # The order of keys in the patterns dict matches the original discovery order
        pattern_keys = list(self.patterns.keys())
        self.idx_to_pattern_key = {i: key for i, key in enumerate(pattern_keys)}
        print(f"  Built index-to-key mapping for {len(self.idx_to_pattern_key)} patterns")

        print("Loading track derives...")
        with open(self.base_path / 'checkpoint_v55_pure_contour_1000files_track_derives.json', 'r') as f:
            derives_data = json.load(f)
        self.derives = derives_data['derives']
        print(f"  Loaded {len(self.derives)} derives")

        # Load OVERLAP-BASED co-occurrence (captures sustained notes, not just exact onset)
        print("Loading co-occurrence index...")
        co_occur_path = self.base_path / 'co_occurrence_active.json'
        global_co_occur_path = self.base_path / 'co_occurrence_active_global.json'
        fallback_path = self.base_path / 'accompaniment_fallback.json'

        if co_occur_path.exists():
            with open(co_occur_path, 'r') as f:
                self.co_occurrence = json.load(f)
            print(f"  Loaded overlap-based co-occurrence for {len(self.co_occurrence)} leader patterns")
        else:
            print("  WARNING: co_occurrence_active.json not found")

        # Load global co-occurrence as fallback
        if global_co_occur_path.exists():
            with open(global_co_occur_path, 'r') as f:
                self.global_co_occurrence = json.load(f)
            print(f"  Loaded global co-occurrence for {len(self.global_co_occurrence)} leader patterns")
        else:
            print("  WARNING: co_occurrence_active_global.json not found")

        if fallback_path.exists():
            with open(fallback_path, 'r') as f:
                fallback_data = json.load(f)
            self.fallback_piano = [p[0] for p in fallback_data.get('piano', [])]
            self.fallback_bass = [p[0] for p in fallback_data.get('bass', [])]
            print(f"  Loaded {len(self.fallback_piano)} fallback piano, {len(self.fallback_bass)} fallback bass")
        else:
            print("  WARNING: accompaniment_fallback.json not found")

        self._build_indexes()

    def _build_indexes(self):
        """Build lookup indexes for generation."""
        # Index patterns by GM program
        for name, data in self.patterns.items():
            gm = data.get('gm_program', 0)
            # Only include patterns with actual intervals
            intervals = data.get('pitch_intervals', [])
            if len(intervals) >= 1:  # At least 2 notes
                self.pattern_by_gm[gm].append(name)

        print(f"Patterns by instrument:")
        for gm in sorted(self.pattern_by_gm.keys()):
            count = len(self.pattern_by_gm[gm])
            name = GM_NAMES.get(gm, f'GM{gm}')
            if count > 100:
                print(f"  {name}: {count}")

        # Index derives for lookup
        # CRITICAL: derives use numeric indices, patterns use string keys
        # We must convert numeric IDs to string keys using self.idx_to_pattern_key
        derive_counts = defaultdict(lambda: defaultdict(int))
        converted_count = 0
        skipped_count = 0

        for d in self.derives:
            src_idx = d['source_pattern_id']  # numeric index
            src_inst = d['source_instrument']
            tgt_inst = d['target_instrument']
            tgt_idx = d['target_pattern_id']  # numeric index
            transform = d['transform']

            # Convert numeric indices to string keys
            src_pattern = self.idx_to_pattern_key.get(src_idx)
            tgt_pattern = self.idx_to_pattern_key.get(tgt_idx)

            if src_pattern is None or tgt_pattern is None:
                skipped_count += 1
                continue

            converted_count += 1

            # Store the derive with STRING keys
            key = (src_pattern, src_inst, tgt_inst)
            if key not in self.derive_index:
                self.derive_index[key] = (tgt_pattern, transform)

            # Count transforms per instrument pair
            transform_key = json.dumps(transform, sort_keys=True)
            derive_counts[(src_inst, tgt_inst)][transform_key] += 1

        print(f"  Converted {converted_count} derives to string keys, skipped {skipped_count}")

        # Build instrument pair transform distribution
        for (src_inst, tgt_inst), transforms in derive_counts.items():
            sorted_transforms = sorted(transforms.items(), key=lambda x: -x[1])
            self.inst_pair_transforms[(src_inst, tgt_inst)] = sorted_transforms

        print(f"\nDerive index entries: {len(self.derive_index)}")
        print(f"Instrument pair transforms: {len(self.inst_pair_transforms)}")

    def _sample_pattern(self, gm, prefer_with_cooccurrence=True):
        """Sample a pattern for an instrument, weighted by corpus frequency.

        Prefers patterns that have co-occurrence data when generating leader tracks,
        since those are the ones that appeared with accompaniment in the corpus.

        Args:
            gm: GM program number
            prefer_with_cooccurrence: If True, prefer patterns with co-occurrence data

        Returns:
            Pattern name or None
        """
        patterns = self.pattern_by_gm.get(gm, [])
        if not patterns:
            return None

        # Filter to patterns with co-occurrence data (for leaders)
        if prefer_with_cooccurrence and self.co_occurrence:
            patterns_with_co = [p for p in patterns if p in self.co_occurrence]
            if patterns_with_co:
                patterns = patterns_with_co

        # Weight by corpus frequency
        weights = []
        for p in patterns:
            if p in self.patterns:
                weights.append(self.patterns[p].get('count', 1))
            else:
                weights.append(1)

        # Weighted random selection
        total = sum(weights)
        if total > 0:
            r = random.random() * total
            cumulative = 0
            for pattern, weight in zip(patterns, weights):
                cumulative += weight
                if r <= cumulative:
                    return pattern

        return random.choice(patterns)

    def _get_first_pitch_from_occurrences(self, pattern_name, gm):
        """Get a first_pitch from real occurrences of this pattern.

        This uses actual pitch data from the corpus to position patterns
        in the correct register for each instrument.
        """
        if pattern_name not in self.patterns:
            return None

        pattern_data = self.patterns[pattern_name]
        occurrences = pattern_data.get('occurrences', [])

        # Filter occurrences for this instrument
        relevant_occs = [
            occ for occ in occurrences
            if occ.get('gm_program') == gm and 'first_pitch' in occ
        ]

        if relevant_occs:
            # Return a random first_pitch from real occurrences
            occ = random.choice(relevant_occs)
            return occ['first_pitch']

        # Fall back to any occurrence with this pattern
        any_occs = [occ for occ in occurrences if 'first_pitch' in occ]
        if any_occs:
            return random.choice(any_occs)['first_pitch']

        return None

    def _get_default_pitch_for_instrument(self, gm):
        """Get a sensible default starting pitch for an instrument."""
        if gm in INSTRUMENT_RANGES:
            low, high = INSTRUMENT_RANGES[gm]
            return (low + high) // 2  # Middle of range
        return 60  # Middle C as fallback

    def _get_intervals(self, pattern_name):
        """Get pitch intervals for a pattern."""
        if pattern_name not in self.patterns:
            return []
        return self.patterns[pattern_name].get('pitch_intervals', [])

    def _parse_transform(self, transform):
        """Parse transform and return (modified_intervals_func, transposition).

        Returns:
            transposition: int - semitones to add to base pitch
        """
        if not transform:
            return 0

        primitives = transform.get('primitives', '')
        if isinstance(primitives, list):
            primitives = str(primitives)

        transposition = 0

        # Parse transposition from T<n> patterns
        # Format: "(T3,)" or "(I1,)" or "(R, T5)" etc.
        t_match = re.search(r'T(\d+)', primitives)
        if t_match:
            transposition = int(t_match.group(1))

        return transposition

    def _apply_transform(self, intervals, transform):
        """Apply a transform to pitch intervals (not transposition)."""
        if not transform or not intervals:
            return intervals

        primitives = transform.get('primitives', '')
        if isinstance(primitives, list):
            primitives = str(primitives)

        result = list(intervals)

        # Parse and apply transforms that affect interval sequence
        # Format: "(T3,)" or "(I1,)" or "(R, T5)" etc.
        if 'R' in primitives:
            result = list(reversed(result))

        if 'I' in primitives:
            # Inversion - negate intervals
            result = [-i for i in result]

        # Note: T (transpose) doesn't change intervals, only base pitch
        # That's handled in _parse_transform

        return result

    def _get_dominant_transform(self, src_inst, tgt_inst):
        """Get the most common transform for an instrument pair."""
        transforms = self.inst_pair_transforms.get((src_inst, tgt_inst), [])
        if not transforms:
            return None

        # Return the most common transform
        transform_str, count = transforms[0]
        return json.loads(transform_str)

    def _derive_pattern(self, src_pattern, src_inst, tgt_inst):
        """Derive a pattern for target instrument from source pattern."""
        intervals, _ = self._derive_pattern_with_transform(src_pattern, src_inst, tgt_inst)
        return intervals

    def _derive_pattern_with_transform(self, src_pattern, src_inst, tgt_inst):
        """Derive a pattern for target instrument, returning both intervals and transform."""
        # Try exact lookup first
        key = (src_pattern, src_inst, tgt_inst)
        if key in self.derive_index:
            tgt_pattern, transform = self.derive_index[key]
            # Look up the target pattern intervals
            tgt_intervals = self._get_intervals(tgt_pattern)
            if tgt_intervals:
                return tgt_intervals, transform

        # Fall back to applying dominant transform
        src_intervals = self._get_intervals(src_pattern)
        if not src_intervals:
            return None, None

        transform = self._get_dominant_transform(src_inst, tgt_inst)
        if transform:
            return self._apply_transform(src_intervals, transform), transform

        # No derive available - return None
        return None, None

    def generate(self, n_patterns=16, leader_gm=None, follower_gms=None, tempo=120):
        """
        Generate a piece using the grammar.

        Args:
            n_patterns: Number of patterns for leader track
            leader_gm: GM program for leader (default: random from leaders)
            follower_gms: List of GM programs for followers (default: auto-derive)
            tempo: BPM

        Returns:
            dict with 'events' and 'midi'
        """
        # Choose leader instrument
        if leader_gm is None:
            leader_gm = random.choice(self.leader_instruments)

        print(f"Leader: {GM_NAMES.get(leader_gm, f'GM{leader_gm}')}")

        # Generate leader track
        leader_events = []
        current_beat = 0

        for i in range(n_patterns):
            pattern_name = self._sample_pattern(leader_gm)
            if pattern_name is None:
                continue

            intervals = self._get_intervals(pattern_name)
            if not intervals:
                continue

            # Get first_pitch from real occurrences (KEY FIX!)
            first_pitch = self._get_first_pitch_from_occurrences(pattern_name, leader_gm)
            if first_pitch is None:
                first_pitch = self._get_default_pitch_for_instrument(leader_gm)

            leader_events.append({
                'beat': current_beat,
                'gm': leader_gm,
                'pattern': pattern_name,
                'intervals': intervals,
                'first_pitch': first_pitch  # Use actual pitch, not offset!
            })

            # Advance time based on pattern length
            current_beat += len(intervals) + 1

        print(f"Leader patterns: {len(leader_events)}")

        # Determine follower instruments
        if follower_gms is None:
            # Find instruments that commonly follow the leader
            follower_gms = []
            for (src, tgt), transforms in self.inst_pair_transforms.items():
                if src == leader_gm and tgt != leader_gm:
                    total_count = sum(c for _, c in transforms)
                    if total_count >= 50:  # Minimum derives
                        follower_gms.append((tgt, total_count))

            # Sort by count and take top 3
            follower_gms = [gm for gm, _ in sorted(follower_gms, key=lambda x: -x[1])[:3]]

        print(f"Followers: {[GM_NAMES.get(gm, f'GM{gm}') for gm in follower_gms]}")

        # Generate follower tracks by deriving from leader
        all_events = leader_events.copy()

        for follower_gm in follower_gms:
            derived_count = 0
            for leader_evt in leader_events:
                # Try to derive this pattern for the follower
                derived_intervals, transform = self._derive_pattern_with_transform(
                    leader_evt['pattern'],
                    leader_gm,
                    follower_gm
                )

                if derived_intervals:
                    # Parse transposition from transform (e.g., T3 = +3 semitones)
                    transposition = self._parse_transform(transform)

                    # Calculate follower pitch: start from leader pitch and apply transposition
                    leader_pitch = leader_evt['first_pitch']
                    follower_pitch = leader_pitch + transposition

                    # Ensure pitch is in valid range for this instrument
                    if follower_gm in INSTRUMENT_RANGES:
                        low, high = INSTRUMENT_RANGES[follower_gm]
                        # Adjust octave if needed to fit instrument range
                        while follower_pitch < low:
                            follower_pitch += 12
                        while follower_pitch > high:
                            follower_pitch -= 12

                    all_events.append({
                        'beat': leader_evt['beat'],
                        'gm': follower_gm,
                        'intervals': derived_intervals,
                        'first_pitch': follower_pitch
                    })
                    derived_count += 1

            print(f"  {GM_NAMES.get(follower_gm, f'GM{follower_gm}')}: {derived_count} derived")

        # Add accompaniment (piano/bass) if not already present
        present_gms = {evt['gm'] for evt in all_events}
        accompaniment_events = self._generate_accompaniment(
            leader_events, present_gms, current_beat
        )
        all_events.extend(accompaniment_events)

        # Convert to MIDI
        midi = self._events_to_midi(all_events, tempo)

        return {
            'events': all_events,
            'midi': midi,
            'leader_gm': leader_gm,
            'follower_gms': follower_gms,
            'n_patterns': len(leader_events)
        }

    def _build_form_from_occurrences(self, gm):
        """Build form structure from pattern occurrence statistics.

        Uses the fact that patterns with multiple occurrences in real pieces
        indicate thematic elements that should recur. Returns a form template
        like AABA or verse-chorus patterns.

        Args:
            gm: GM program to analyze patterns for

        Returns:
            List of (pattern_name, relative_position) for form structure
        """
        # Find patterns with multiple occurrences (indicates thematic importance)
        recurring_patterns = []
        for pattern_name in self.pattern_by_gm.get(gm, []):
            if pattern_name not in self.patterns:
                continue
            occs = self.patterns[pattern_name].get('occurrences', [])
            if len(occs) >= 2:  # Pattern appears at least twice
                # Calculate average recurrence distance
                occ_times = sorted([o.get('onset_time', 0) for o in occs if o.get('gm_program') == gm])
                if len(occ_times) >= 2:
                    # Pattern recurs - good for form
                    recurring_patterns.append({
                        'pattern': pattern_name,
                        'count': len(occ_times),
                        'intervals': self._get_intervals(pattern_name)
                    })

        # Sort by occurrence count (most recurring = most thematic)
        recurring_patterns.sort(key=lambda x: -x['count'])

        # Build AABA-style form from most recurring patterns
        form_template = []
        if len(recurring_patterns) >= 2:
            # A = most common, B = second most common
            pattern_a = recurring_patterns[0]['pattern']
            pattern_b = recurring_patterns[1]['pattern'] if len(recurring_patterns) > 1 else pattern_a

            # Classic AABA form with variations
            form_template = [
                ('A', pattern_a),
                ('A', pattern_a),
                ('B', pattern_b),
                ('A', pattern_a),
            ]
        elif len(recurring_patterns) >= 1:
            # Just repeat the main theme
            pattern_a = recurring_patterns[0]['pattern']
            form_template = [
                ('A', pattern_a),
                ('A', pattern_a),
                ('A', pattern_a),
            ]

        return form_template, recurring_patterns

    def generate_with_form(self, n_sections=4, leader_gm=None, follower_gms=None, tempo=120):
        """Generate a piece with explicit form structure (AABA, verse-chorus, etc).

        Uses pattern occurrence statistics to identify thematic patterns and
        creates structured repetition.

        Args:
            n_sections: Number of form sections (e.g., 4 for AABA)
            leader_gm: GM program for leader
            follower_gms: List of follower GM programs
            tempo: BPM

        Returns:
            dict with events and midi
        """
        # Choose leader instrument
        if leader_gm is None:
            leader_gm = random.choice(self.leader_instruments)

        print(f"Leader: {GM_NAMES.get(leader_gm, f'GM{leader_gm}')}")

        # Build form from occurrence statistics
        form_template, recurring = self._build_form_from_occurrences(leader_gm)

        if not form_template:
            print("No form template available, using random generation")
            return self.generate(n_patterns=16, leader_gm=leader_gm,
                               follower_gms=follower_gms, tempo=tempo)

        print(f"Form template: {[f[0] for f in form_template]}")
        print(f"Found {len(recurring)} recurring patterns for form")

        # Generate events following the form
        leader_events = []
        current_beat = 0
        patterns_per_section = 4  # Patterns per form section

        for section_idx, (section_label, main_pattern) in enumerate(form_template[:n_sections]):
            print(f"  Section {section_label} ({section_idx + 1}/{n_sections})")

            # Start with the thematic pattern
            intervals = self._get_intervals(main_pattern)
            if intervals:
                # Get first_pitch from real occurrences
                first_pitch = self._get_first_pitch_from_occurrences(main_pattern, leader_gm)
                if first_pitch is None:
                    first_pitch = self._get_default_pitch_for_instrument(leader_gm)

                leader_events.append({
                    'beat': current_beat,
                    'gm': leader_gm,
                    'pattern': main_pattern,
                    'intervals': intervals,
                    'first_pitch': first_pitch,
                    'section': section_label
                })
                current_beat += len(intervals) + 1

            # Fill rest of section with related patterns
            for _ in range(patterns_per_section - 1):
                # Mix thematic and random patterns
                if random.random() < 0.5 and recurring:
                    # Use another recurring pattern
                    fill_pattern = random.choice(recurring[:min(5, len(recurring))])['pattern']
                else:
                    # Random fill
                    fill_pattern = self._sample_pattern(leader_gm)

                if fill_pattern:
                    fill_intervals = self._get_intervals(fill_pattern)
                    if fill_intervals:
                        # Get first_pitch from real occurrences
                        first_pitch = self._get_first_pitch_from_occurrences(fill_pattern, leader_gm)
                        if first_pitch is None:
                            first_pitch = self._get_default_pitch_for_instrument(leader_gm)

                        leader_events.append({
                            'beat': current_beat,
                            'gm': leader_gm,
                            'pattern': fill_pattern,
                            'intervals': fill_intervals,
                            'first_pitch': first_pitch,
                            'section': section_label
                        })
                        current_beat += len(fill_intervals) + 1

        print(f"Leader patterns: {len(leader_events)}")

        # Determine follower instruments (same as regular generate)
        if follower_gms is None:
            follower_gms = []
            for (src, tgt), transforms in self.inst_pair_transforms.items():
                if src == leader_gm and tgt != leader_gm:
                    total_count = sum(c for _, c in transforms)
                    if total_count >= 50:
                        follower_gms.append((tgt, total_count))
            follower_gms = [gm for gm, _ in sorted(follower_gms, key=lambda x: -x[1])[:3]]

        print(f"Followers: {[GM_NAMES.get(gm, f'GM{gm}') for gm in follower_gms]}")

        # Generate follower tracks by deriving from leader
        all_events = leader_events.copy()

        for follower_gm in follower_gms:
            derived_count = 0
            for leader_evt in leader_events:
                derived_intervals, transform = self._derive_pattern_with_transform(
                    leader_evt['pattern'],
                    leader_gm,
                    follower_gm
                )

                if derived_intervals:
                    # Parse transposition from transform (e.g., T3 = +3 semitones)
                    transposition = self._parse_transform(transform)

                    # Calculate follower pitch: start from leader pitch and apply transposition
                    leader_pitch = leader_evt['first_pitch']
                    follower_pitch = leader_pitch + transposition

                    # Ensure pitch is in valid range for this instrument
                    if follower_gm in INSTRUMENT_RANGES:
                        low, high = INSTRUMENT_RANGES[follower_gm]
                        while follower_pitch < low:
                            follower_pitch += 12
                        while follower_pitch > high:
                            follower_pitch -= 12

                    all_events.append({
                        'beat': leader_evt['beat'],
                        'gm': follower_gm,
                        'intervals': derived_intervals,
                        'first_pitch': follower_pitch,
                        'section': leader_evt.get('section')
                    })
                    derived_count += 1

            print(f"  {GM_NAMES.get(follower_gm, f'GM{follower_gm}')}: {derived_count} derived")

        # Add accompaniment
        present_gms = {evt['gm'] for evt in all_events}
        accompaniment_events = self._generate_accompaniment(
            leader_events, present_gms, current_beat
        )
        all_events.extend(accompaniment_events)

        # Convert to MIDI
        midi = self._events_to_midi(all_events, tempo)

        return {
            'events': all_events,
            'midi': midi,
            'leader_gm': leader_gm,
            'follower_gms': follower_gms,
            'n_patterns': len(leader_events),
            'form': [f[0] for f in form_template[:n_sections]]
        }

    def _select_accompaniment_pattern(self, leader_pattern, leader_pitch, acc_type, piece_hint=None):
        """Select accompaniment pattern using PIECE-AWARE co-occurrence data.

        Uses patterns from the SAME PIECE as the leader pattern to preserve
        the harmonic context. Falls back to global co-occurrence (still 78%
        consonant) when piece-aware data is unavailable.

        Args:
            leader_pattern: The leader pattern name (e.g., "GM57_151")
            leader_pitch: The pitch the leader is playing at
            acc_type: "piano" or "bass"
            piece_hint: Optional piece_id to prefer (for consistency within generation)

        Returns:
            (pattern_name, first_pitch, piece_id) or (None, None, None)
        """
        # Try piece-aware co-occurrence lookup first
        if leader_pattern in self.co_occurrence:
            pieces_data = self.co_occurrence[leader_pattern]

            if pieces_data:
                # If we have a piece hint, try to use the same piece for consistency
                selected_piece = None
                if piece_hint:
                    for pd in pieces_data:
                        if pd['piece'] == piece_hint and pd.get(acc_type):
                            selected_piece = pd
                            break

                # Otherwise, pick a random piece that has this accompaniment type
                if not selected_piece:
                    valid_pieces = [pd for pd in pieces_data if pd.get(acc_type)]
                    if valid_pieces:
                        selected_piece = random.choice(valid_pieces)

                if selected_piece:
                    acc_entries = selected_piece[acc_type]
                    if acc_entries:
                        pattern, offset = random.choice(acc_entries)
                        acc_pitch = leader_pitch + offset
                        return pattern, acc_pitch, selected_piece['piece']

        # Fallback to GLOBAL co-occurrence (still 78% consonant, just not piece-specific)
        if leader_pattern in self.global_co_occurrence:
            global_data = self.global_co_occurrence[leader_pattern].get(acc_type, [])
            if global_data:
                # global_data is list of [pattern, offset, count]
                # Weight by frequency
                weights = [entry[2] for entry in global_data]
                total = sum(weights)
                if total > 0:
                    r = random.random() * total
                    cumulative = 0
                    for entry in global_data:
                        pattern, offset, count = entry
                        cumulative += count
                        if r <= cumulative:
                            acc_pitch = leader_pitch + offset
                            return pattern, acc_pitch, None  # No specific piece

        # Last resort: most common patterns (no pitch offset)
        if acc_type == "piano" and self.fallback_piano:
            pattern = random.choice(self.fallback_piano[:20])
            return pattern, None, None
        elif acc_type == "bass" and self.fallback_bass:
            pattern = random.choice(self.fallback_bass[:20])
            return pattern, None, None

        return None, None, None

    def _generate_accompaniment(self, leader_events, present_gms, total_beats):
        """Generate piano/bass accompaniment patterns alongside leader melody.

        Uses PIECE-AWARE co-occurrence to select patterns that:
        1. Actually accompanied the leader patterns in the SAME PIECE
        2. Play at the correct pitch relative to the leader
        3. Maintain harmonic context from the original piece

        Args:
            leader_events: List of leader track events with beat positions
            present_gms: Set of GM programs already in the arrangement
            total_beats: Total duration in beats

        Returns:
            List of accompaniment events
        """
        accompaniment_events = []

        # Map GM programs to accompaniment types
        acc_type_map = {0: "piano", 32: "bass", 33: "bass"}

        for acc_gm in self.accompaniment_instruments:
            if acc_gm in present_gms:
                continue  # Skip if already present via derives

            acc_type = acc_type_map.get(acc_gm)
            if not acc_type:
                continue

            acc_count = 0
            co_occur_hits = 0
            current_piece = None  # Track piece for consistency

            # Generate accompaniment aligned with leader beats
            for leader_evt in leader_events:
                leader_pattern = leader_evt.get('pattern')
                leader_pitch = leader_evt.get('first_pitch', 60)

                # Select pattern using PIECE-AWARE co-occurrence
                pattern_name, first_pitch, piece_id = self._select_accompaniment_pattern(
                    leader_pattern, leader_pitch, acc_type, piece_hint=current_piece
                )

                # Update piece hint for next selection (try to stay in same piece)
                if piece_id:
                    current_piece = piece_id

                if pattern_name is None:
                    # Last resort: random from all patterns for this instrument
                    patterns = self.pattern_by_gm.get(acc_gm, [])
                    if patterns:
                        pattern_name = random.choice(patterns)
                        first_pitch = None

                if pattern_name is None:
                    continue

                intervals = self._get_intervals(pattern_name)
                if not intervals:
                    continue

                # Track co-occurrence hits for debugging
                if leader_pattern in self.co_occurrence:
                    co_occur_hits += 1

                # Use co-occurrence pitch if available, otherwise fall back
                if first_pitch is None:
                    first_pitch = self._get_first_pitch_from_occurrences(pattern_name, acc_gm)
                if first_pitch is None:
                    first_pitch = self._get_default_pitch_for_instrument(acc_gm)

                # Clamp to valid MIDI range and instrument range
                if acc_gm in INSTRUMENT_RANGES:
                    low, high = INSTRUMENT_RANGES[acc_gm]
                    while first_pitch < low:
                        first_pitch += 12
                    while first_pitch > high:
                        first_pitch -= 12

                accompaniment_events.append({
                    'beat': leader_evt['beat'],
                    'gm': acc_gm,
                    'pattern': pattern_name,
                    'intervals': intervals,
                    'first_pitch': first_pitch
                })
                acc_count += 1

            if acc_count > 0:
                co_occur_pct = 100 * co_occur_hits / acc_count if acc_count else 0
                print(f"  {GM_NAMES.get(acc_gm, f'GM{acc_gm}')}: {acc_count} accompaniment ({co_occur_pct:.0f}% co-occur)")

        return accompaniment_events

    def _get_rhythm_info(self, pattern_name):
        """Get rhythm and duration ratios for a pattern."""
        if pattern_name not in self.patterns:
            return None, None
        data = self.patterns[pattern_name]
        rhythm_ratios = data.get('rhythm_ratios', [1.0] * len(data.get('pitch_intervals', [])))
        duration_ratios = data.get('duration_ratios', [1.0] * (len(data.get('pitch_intervals', [])) + 1))
        return rhythm_ratios, duration_ratios

    def _events_to_midi(self, events, tempo=120):
        """Convert events to MIDI file with proper rhythm using pretty_midi."""
        if not events:
            return None

        # Create PrettyMIDI object
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        # Seconds per beat at this tempo
        sec_per_beat = 60.0 / tempo

        # Base timing unit (quarter note = 1 beat)
        base_ioi = 0.5  # Base inter-onset interval (eighth note in beats)
        base_dur = 0.4  # Base note duration in beats

        # Group events by instrument
        events_by_gm = defaultdict(list)
        for event in events:
            events_by_gm[event['gm']].append(event)

        # Create one instrument per GM program
        for gm, gm_events in events_by_gm.items():
            # Handle drums (channel 9) vs melodic instruments
            if gm == 128:
                instrument = pretty_midi.Instrument(program=0, is_drum=True, name='drums')
            else:
                instrument = pretty_midi.Instrument(program=gm, is_drum=False, name=GM_NAMES.get(gm, f'gm{gm}'))

            for event in gm_events:
                intervals = event['intervals']
                beat = event['beat']
                pattern_name = event.get('pattern')

                # Get rhythm info
                rhythm_ratios, duration_ratios = None, None
                if pattern_name:
                    rhythm_ratios, duration_ratios = self._get_rhythm_info(pattern_name)

                # Default if no rhythm info
                n_notes = len(intervals) + 1
                if rhythm_ratios is None:
                    rhythm_ratios = [1.0] * len(intervals)
                if duration_ratios is None:
                    duration_ratios = [1.0] * n_notes

                # Use first_pitch from occurrence data (KEY FIX!)
                # This places notes in the correct register for each instrument
                base_pitch = event.get('first_pitch')
                if base_pitch is None:
                    # Fallback for old-style events with pitch_offset
                    pitch_offset = event.get('pitch_offset', 0)
                    base_pitch = 60 + pitch_offset

                velocity = 90

                # First note
                current_pitch = base_pitch
                current_time = float(beat) * sec_per_beat  # Convert beats to seconds

                # Duration for first note
                dur = base_dur * duration_ratios[0] if duration_ratios else base_dur
                dur = max(0.1, min(dur, 2.0))  # Clamp duration in beats
                dur_sec = dur * sec_per_beat  # Convert to seconds

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=int(current_pitch),
                    start=current_time,
                    end=current_time + dur_sec
                )
                instrument.notes.append(note)

                # Subsequent notes from intervals with rhythm
                for i, interval in enumerate(intervals):
                    # Inter-onset interval from rhythm_ratios
                    ioi = base_ioi
                    if i < len(rhythm_ratios):
                        ratio = rhythm_ratios[i]
                        if ratio == 0.0:
                            # Simultaneous note (chord)
                            ioi = 0.0
                        else:
                            ioi = base_ioi * ratio
                            ioi = max(0.01, min(ioi, 4.0))  # Clamp IOI

                    current_time += ioi * sec_per_beat  # Convert to seconds
                    current_pitch += interval
                    current_pitch = max(21, min(108, current_pitch))  # Keep in range

                    # Duration for this note
                    dur = base_dur
                    if i + 1 < len(duration_ratios):
                        dur = base_dur * duration_ratios[i + 1]
                    dur = max(0.1, min(dur, 2.0))  # Clamp duration
                    dur_sec = dur * sec_per_beat

                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=int(current_pitch),
                        start=current_time,
                        end=current_time + dur_sec
                    )
                    instrument.notes.append(note)

            pm.instruments.append(instrument)

        return pm

    def save_midi(self, midi, output_path):
        """Save MIDI file."""
        midi.write(str(output_path))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Grammar-based MIDI generator')
    parser.add_argument('--output-dir', type=str, default='outputs/grammar_generation/')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--patterns', type=int, default=16)
    parser.add_argument('--tempo', type=int, default=120)
    parser.add_argument('--leader', type=int, default=None, help='Leader GM program')
    parser.add_argument('--with-form', action='store_true', help='Generate with AABA form structure')
    parser.add_argument('--sections', type=int, default=4, help='Number of form sections (with --with-form)')
    args = parser.parse_args()

    base_path = Path('/home/arlo/do-repo/midi_generator/1_approaches/transform_based')
    output_dir = base_path / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    gen = GrammarGenerator(base_path)
    gen.load()

    # Generate samples
    mode = "with form" if args.with_form else "random"
    print(f"\n=== Generating {args.num_samples} samples ({mode}) ===\n")

    for i in range(args.num_samples):
        print(f"\n--- Sample {i+1}/{args.num_samples} ---")

        if args.with_form:
            result = gen.generate_with_form(
                n_sections=args.sections,
                leader_gm=args.leader,
                tempo=args.tempo
            )
        else:
            result = gen.generate(
                n_patterns=args.patterns,
                leader_gm=args.leader,
                tempo=args.tempo
            )

        if result['midi']:
            output_path = output_dir / f'grammar_sample_{i+1}.mid'
            gen.save_midi(result['midi'], output_path)
            print(f"Saved to {output_path}")

            # Save metadata
            meta_path = output_dir / f'grammar_sample_{i+1}_meta.json'
            meta = {
                'leader_gm': result['leader_gm'],
                'leader_name': GM_NAMES.get(result['leader_gm'], f'GM{result["leader_gm"]}'),
                'follower_gms': result['follower_gms'],
                'follower_names': [GM_NAMES.get(gm, f'GM{gm}') for gm in result['follower_gms']],
                'n_patterns': result['n_patterns'],
                'n_events': len(result['events']),
                'tempo': args.tempo
            }
            if 'form' in result:
                meta['form'] = result['form']
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

    print(f"\nGeneration complete. Output saved to {output_dir}")


if __name__ == '__main__':
    main()
