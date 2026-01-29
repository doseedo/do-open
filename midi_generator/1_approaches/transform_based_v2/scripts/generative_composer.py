#!/usr/bin/env python3
"""
GENERATIVE COMPOSER - True Pattern-Based Generation with Sequential Awareness

The Problem:
- Previous "hybrid_generator" just re-renders one piece section
- No combination of patterns from different sources
- No awareness of sequential context (what comes before/after)

The Solution:
1. Build SEQUENTIAL CO-OCCURRENCE: For each track, what patterns follow what?
2. Build SUBSTITUTION CANDIDATES: Patterns with same pitch_intervals can substitute
3. CONSTRAINED GENERATION: Only substitute if it makes sense in sequence

This is TRULY GENERATIVE because:
- Patterns can come from ANY piece in the corpus
- But constrained by what actually co-occurred sequentially
- Result: Novel combinations that still sound coherent

Usage:
    python scripts/generative_composer.py --template "Caravan" --variation 0.3 -o output.mid
    python scripts/generative_composer.py --template "score - 2025-08-07T204344.144" --bars 16 --variation 0.5 -o output.mid
"""

import orjson
import json
import random
import argparse
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np


# Quantization grid - standard musical divisions at 480 ticks/beat
STRICT_GRID = [120, 240, 480, 720, 960, 1920]

GM_NAMES = {
    0: 'Piano', 32: 'Acoustic Bass', 33: 'Electric Bass',
    56: 'Trumpet', 57: 'Trombone', 60: 'French Horn',
    65: 'Alto Sax', 66: 'Tenor Sax', 67: 'Baritone Sax',
    71: 'Clarinet', 73: 'Flute', 128: 'Drums'
}


def quantize_tau(tau: int, grid: List[int] = STRICT_GRID) -> int:
    """Quantize tau to nearest grid value."""
    if tau <= 0:
        return 480
    return min(grid, key=lambda g: abs(g - tau))


class GenerativeComposer:
    """
    True generative composition with sequential awareness.

    Key data structures:
    - seq_bigrams[gm][(pattern_A, pattern_B)] = count
      "How often does pattern_B follow pattern_A in track gm?"

    - substitutes[pattern_A] = [pattern_B, pattern_C, ...]
      "Patterns with same pitch_intervals that can substitute for A"

    - patterns: The raw pattern data
    """

    def __init__(self, patterns_path: str, verbose: bool = True):
        self.verbose = verbose
        self.patterns_path = patterns_path
        self.patterns = None

        # Core indices
        self.piece_tracks = defaultdict(lambda: defaultdict(list))  # piece -> gm -> [events]
        self.pattern_by_gm = defaultdict(list)  # gm -> [pattern_ids]

        # Sequential co-occurrence (built per GM program)
        self.seq_bigrams = defaultdict(Counter)  # gm -> Counter[(prev_pid, next_pid)]
        self.seq_prev = defaultdict(lambda: defaultdict(Counter))  # gm -> pid -> Counter[prev_pids]
        self.seq_next = defaultdict(lambda: defaultdict(Counter))  # gm -> pid -> Counter[next_pids]

        # Vertical co-occurrence (what patterns play together across instruments)
        self.vertical_cooccur = {}  # pattern_id -> {'piano': [[pid, offset], ...], 'bass': [...]}

        # Substitution candidates (patterns with same contour OR intervals)
        self.substitutes = defaultdict(list)  # pattern_id -> [substitute_ids]
        self.contour_to_patterns = defaultdict(list)  # contour_string -> [pattern_ids]
        self.intervals_to_patterns = defaultdict(list)  # tuple(intervals) -> [pattern_ids]

        self._load()
        self._build_indices()
        self._build_sequential_cooccurrence()
        self._build_substitutes()
        self._load_vertical_cooccurrence()

    def _load(self):
        """Load patterns from checkpoint."""
        if self.verbose:
            print(f"Loading patterns from {self.patterns_path}...")

        with open(self.patterns_path, 'rb') as f:
            self.patterns = orjson.loads(f.read())

        if self.verbose:
            print(f"  Loaded {len(self.patterns)} patterns")

    def _build_indices(self):
        """Build basic indices: piece_tracks and pattern_by_gm."""
        if self.verbose:
            print("Building basic indices...")

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            self.pattern_by_gm[gm].append(pid)

            for occ in p.get('occurrences', []):
                piece_id = occ.get('piece_id', 'unknown')
                tau = occ.get('tau_offset', 480)

                if tau <= 0:
                    continue

                self.piece_tracks[piece_id][gm].append({
                    'pattern_id': pid,
                    'gm': occ.get('gm_program', gm),
                    'onset': occ.get('onset_time', 0),
                    'pitch': occ.get('first_pitch', 60),
                    'tau': tau,
                    'intervals': p.get('pitch_intervals', []),
                    'duration_ratios': p.get('duration_ratios', []),
                    'contour': p.get('contour', ''),
                })

        if self.verbose:
            print(f"  Indexed {len(self.piece_tracks)} pieces")
            print(f"  Indexed {len(self.pattern_by_gm)} instruments")

    def _build_sequential_cooccurrence(self):
        """
        Build sequential bigrams: what patterns follow what?

        For each piece, for each track (GM program):
            Sort events by onset time
            Record consecutive pairs as bigrams
        """
        if self.verbose:
            print("Building sequential co-occurrence...")

        total_bigrams = 0

        for piece_id, tracks in self.piece_tracks.items():
            for gm, events in tracks.items():
                # Sort by onset time
                sorted_events = sorted(events, key=lambda x: x['onset'])

                # Record bigrams (consecutive pairs)
                for i in range(len(sorted_events) - 1):
                    prev_pid = sorted_events[i]['pattern_id']
                    next_pid = sorted_events[i + 1]['pattern_id']

                    # Record in both directions
                    self.seq_bigrams[gm][(prev_pid, next_pid)] += 1
                    self.seq_prev[gm][next_pid][prev_pid] += 1
                    self.seq_next[gm][prev_pid][next_pid] += 1
                    total_bigrams += 1

        if self.verbose:
            print(f"  Recorded {total_bigrams} sequential bigrams")
            # Show some stats
            for gm in sorted(self.seq_bigrams.keys())[:5]:
                unique = len(self.seq_bigrams[gm])
                name = GM_NAMES.get(gm, f'GM{gm}')
                print(f"    {name}: {unique} unique bigrams")

    def _build_substitutes(self, max_pattern_len: int = 16):
        """
        Build substitution candidates using CONTOUR matching.

        Contour captures melodic shape (up/down/same) without exact intervals.
        This allows more flexible substitution while preserving musical shape.

        Also uses exact interval matching for short patterns (≤4 intervals).
        """
        if self.verbose:
            print("Building substitution index...")

        # Group patterns by contour (for contour-based substitution)
        for pid, p in self.patterns.items():
            contour = p.get('contour', [])
            gm = p.get('gm_program', 0)
            intervals = p.get('pitch_intervals', [])

            # Skip hierarchical/very long patterns
            if len(intervals) > max_pattern_len:
                continue

            # Use (gm, tuple(contour)) as key so substitutes are same instrument
            if contour:
                key = (gm, tuple(contour))
                self.contour_to_patterns[key].append(pid)

            # Also group by exact intervals for short patterns
            if len(intervals) <= 4:
                intervals_key = (gm, tuple(intervals))
                self.intervals_to_patterns[intervals_key].append(pid)

        # Build substitutes list using CONTOUR matching
        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            contour = p.get('contour', [])
            intervals = p.get('pitch_intervals', [])

            if len(intervals) > max_pattern_len:
                continue

            subs = set()

            # Contour-based substitutes
            if contour:
                key = (gm, tuple(contour))
                for other_pid in self.contour_to_patterns.get(key, []):
                    if other_pid != pid:
                        subs.add(other_pid)

            # Exact interval substitutes (for short patterns)
            if len(intervals) <= 4:
                intervals_key = (gm, tuple(intervals))
                for other_pid in self.intervals_to_patterns.get(intervals_key, []):
                    if other_pid != pid:
                        subs.add(other_pid)

            self.substitutes[pid] = list(subs)

        if self.verbose:
            has_subs = sum(1 for v in self.substitutes.values() if v)
            avg_subs = np.mean([len(v) for v in self.substitutes.values() if v]) if has_subs else 0
            print(f"  {has_subs} patterns have substitutes (contour-based)")
            print(f"  Average {avg_subs:.1f} substitutes per pattern")

    def _load_vertical_cooccurrence(self):
        """Load vertical co-occurrence data (what patterns play together)."""
        import os
        cooccur_path = os.path.join(
            os.path.dirname(self.patterns_path),
            'co_occurrence_active.json'
        )

        if not os.path.exists(cooccur_path):
            if self.verbose:
                print("  No vertical co-occurrence file found")
            return

        if self.verbose:
            print("Loading vertical co-occurrence...")

        with open(cooccur_path, 'rb') as f:
            raw = orjson.loads(f.read())

        # Flatten: pid -> {piano: set(pids), bass: set(pids)}
        for pid, pieces in raw.items():
            self.vertical_cooccur[pid] = {
                'piano': set(),
                'bass': set(),
            }
            for piece_data in pieces:
                for p_entry in piece_data.get('piano', []):
                    self.vertical_cooccur[pid]['piano'].add(p_entry[0])  # [pid, offset] -> pid
                for b_entry in piece_data.get('bass', []):
                    self.vertical_cooccur[pid]['bass'].add(b_entry[0])

        if self.verbose:
            print(f"  Loaded co-occurrence for {len(self.vertical_cooccur)} patterns")

    def list_pieces(self, min_events: int = 50) -> List[Tuple[str, int]]:
        """List available pieces by event count."""
        pieces = []
        for pid, tracks in self.piece_tracks.items():
            total = sum(len(events) for events in tracks.values())
            if total >= min_events:
                pieces.append((pid, total))
        return sorted(pieces, key=lambda x: -x[1])

    def find_piece(self, query: str) -> Optional[str]:
        """Find piece by partial name match."""
        query_lower = query.lower()
        for piece_id in self.piece_tracks:
            if query_lower in piece_id.lower():
                return piece_id
        return None

    def get_template(
        self,
        piece_id: str,
        start_bar: int = 0,
        num_bars: int = 16,
        ticks_per_beat: int = 480,
        deduplicate: bool = True,
    ) -> Dict[int, List[dict]]:
        """
        Extract a section from a piece as a template.

        Same as before, but now we'll use this as a STARTING POINT
        for generative substitution.
        """
        if piece_id not in self.piece_tracks:
            raise ValueError(f"Piece not found: {piece_id}")

        start_tick = start_bar * 4 * ticks_per_beat
        end_tick = (start_bar + num_bars) * 4 * ticks_per_beat

        template = defaultdict(list)

        for gm, events in self.piece_tracks[piece_id].items():
            section_events = [
                e for e in events
                if start_tick <= e['onset'] < end_tick
            ]

            # Deduplicate hierarchical patterns (keep smallest at each onset)
            if deduplicate and section_events:
                by_onset = defaultdict(list)
                for e in section_events:
                    by_onset[e['onset']].append(e)

                section_events = []
                for onset, patterns in by_onset.items():
                    smallest = min(patterns, key=lambda p: len(p.get('intervals', [])))
                    section_events.append(smallest)

            # Normalize onset times to start at 0
            for e in section_events:
                template[gm].append({
                    **e,
                    'onset': e['onset'] - start_tick,
                })

            template[gm] = sorted(template[gm], key=lambda x: x['onset'])

        if self.verbose:
            total = sum(len(v) for v in template.values())
            print(f"\nTemplate: {piece_id} bars {start_bar}-{start_bar+num_bars}")
            print(f"  Total events: {total}")

        return dict(template)

    def can_substitute(
        self,
        gm: int,
        original_pid: str,
        substitute_pid: str,
        prev_pid: Optional[str],
        next_pid: Optional[str],
        require_both: bool = False,
        relax_context: bool = True,
    ) -> bool:
        """
        Check if substitute_pid can replace original_pid in sequence.

        Checks:
        1. Same GM program (already guaranteed by substitutes list)
        2. If prev_pid exists: does prev_pid -> substitute_pid occur in corpus?
        3. If next_pid exists: does substitute_pid -> next_pid occur in corpus?

        Args:
            gm: GM program number
            original_pid: Pattern being replaced
            substitute_pid: Potential replacement
            prev_pid: What comes before in template (None if first)
            next_pid: What comes after in template (None if last)
            require_both: If True, both prev and next must validate
            relax_context: If True, also allow substitution if the substitute
                          pattern appears ANYWHERE after a pattern with same
                          contour as prev (more flexible)
        """
        checks_passed = 0
        checks_required = 0

        # Check: does prev_pid -> substitute_pid occur?
        if prev_pid is not None:
            checks_required += 1
            # Direct bigram check
            if self.seq_bigrams[gm][(prev_pid, substitute_pid)] > 0:
                checks_passed += 1
            elif relax_context:
                # Relaxed check: does ANY pattern with same contour as prev
                # lead to substitute_pid?
                prev_contour = self.patterns.get(prev_pid, {}).get('contour', [])
                if prev_contour:
                    for other_prev in self.contour_to_patterns.get((gm, tuple(prev_contour)), []):
                        if self.seq_bigrams[gm][(other_prev, substitute_pid)] > 0:
                            checks_passed += 1
                            break

        # Check: does substitute_pid -> next_pid occur?
        if next_pid is not None:
            checks_required += 1
            # Direct bigram check
            if self.seq_bigrams[gm][(substitute_pid, next_pid)] > 0:
                checks_passed += 1
            elif relax_context:
                # Relaxed check: does substitute_pid lead to ANY pattern
                # with same contour as next?
                next_contour = self.patterns.get(next_pid, {}).get('contour', [])
                if next_contour:
                    for other_next in self.contour_to_patterns.get((gm, tuple(next_contour)), []):
                        if self.seq_bigrams[gm][(substitute_pid, other_next)] > 0:
                            checks_passed += 1
                            break

        if require_both:
            return checks_passed == checks_required
        else:
            # Accept if at least one check passes (or no checks needed)
            return checks_passed > 0 or checks_required == 0

    def generate(
        self,
        template: Dict[int, List[dict]],
        variation: float = 0.3,
        require_both_context: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[int, List[dict]]:
        """
        Generate a new piece by substituting patterns in the template.

        STRUCTURAL AWARENESS:
        - Patterns that repeat 2+ times are "structural" (e.g., bass riff)
        - Structural patterns get ONE substitute applied to ALL occurrences
        - This preserves musical structure (the riff stays consistent)

        Args:
            template: Template from get_template()
            variation: Probability of attempting substitution (0.0-1.0)
            require_both_context: If True, substitutions must validate
                                  against BOTH prev and next patterns
            seed: Random seed for reproducibility

        Returns:
            New piece with substituted patterns
        """
        if seed is not None:
            random.seed(seed)

        result = defaultdict(list)

        stats = {
            'total_events': 0,
            'substitution_attempts': 0,
            'substitution_success': 0,
            'structural_patterns': 0,
            'unique_sources': set(),
        }

        for gm, events in template.items():
            # PHASE 1: Identify structural patterns (repeat 2+ times)
            pattern_counts = Counter(e['pattern_id'] for e in events)
            structural_pids = {pid for pid, count in pattern_counts.items() if count >= 2}
            stats['structural_patterns'] += len(structural_pids)

            # PHASE 2: Pre-decide substitutes for structural patterns
            # (same substitute for all occurrences)
            structural_subs = {}  # pid -> substitute_pid (or None to keep original)

            for pid in structural_pids:
                if random.random() < variation and pid in self.substitutes:
                    # Find a valid substitute (check context from first occurrence)
                    first_idx = next(i for i, e in enumerate(events) if e['pattern_id'] == pid)
                    prev_pid = events[first_idx - 1]['pattern_id'] if first_idx > 0 else None
                    next_pid = events[first_idx + 1]['pattern_id'] if first_idx < len(events) - 1 else None

                    candidates = [
                        sub_pid for sub_pid in self.substitutes[pid]
                        if self.can_substitute(gm, pid, sub_pid, prev_pid, next_pid,
                                               require_both=require_both_context)
                    ]

                    if candidates:
                        weights = [self.patterns.get(c, {}).get('count', 1) for c in candidates]
                        structural_subs[pid] = random.choices(candidates, weights=weights, k=1)[0]

            # PHASE 3: Apply substitutions
            for i, event in enumerate(events):
                stats['total_events'] += 1
                original_pid = event['pattern_id']

                # Check if this is a structural pattern with pre-decided substitute
                if original_pid in structural_subs:
                    chosen = structural_subs[original_pid]
                    stats['substitution_success'] += 1

                    sub_pattern = self.patterns.get(chosen, {})
                    sub_occs = sub_pattern.get('occurrences', [])

                    if sub_occs:
                        sub_occ = random.choice(sub_occs)
                        stats['unique_sources'].add(sub_occ.get('piece_id', 'unknown'))

                        result[gm].append({
                            'pattern_id': chosen,
                            'gm': gm,
                            'onset': event['onset'],
                            'pitch': sub_occ.get('first_pitch', event['pitch']),
                            'tau': sub_occ.get('tau_offset', event['tau']),
                            'intervals': sub_pattern.get('pitch_intervals', []),
                            'rhythm_ratios': sub_pattern.get('rhythm_ratios', []),
                            'duration_ratios': sub_pattern.get('duration_ratios', []),
                            'source_piece': sub_occ.get('piece_id', 'unknown'),
                        })
                        continue

                # Non-structural or kept structural: individual substitution logic
                elif original_pid not in structural_pids:
                    prev_pid = events[i - 1]['pattern_id'] if i > 0 else None
                    next_pid = events[i + 1]['pattern_id'] if i < len(events) - 1 else None

                    if random.random() < variation and original_pid in self.substitutes:
                        stats['substitution_attempts'] += 1

                        candidates = [
                            sub_pid for sub_pid in self.substitutes[original_pid]
                            if self.can_substitute(gm, original_pid, sub_pid, prev_pid, next_pid,
                                                   require_both=require_both_context)
                        ]

                        if candidates:
                            weights = [self.patterns.get(c, {}).get('count', 1) for c in candidates]
                            chosen = random.choices(candidates, weights=weights, k=1)[0]
                            stats['substitution_success'] += 1

                            sub_pattern = self.patterns.get(chosen, {})
                            sub_occs = sub_pattern.get('occurrences', [])

                            if sub_occs:
                                sub_occ = random.choice(sub_occs)
                                stats['unique_sources'].add(sub_occ.get('piece_id', 'unknown'))

                                result[gm].append({
                                    'pattern_id': chosen,
                                    'gm': gm,
                                    'onset': event['onset'],
                                    'pitch': sub_occ.get('first_pitch', event['pitch']),
                                    'tau': sub_occ.get('tau_offset', event['tau']),
                                    'intervals': sub_pattern.get('pitch_intervals', []),
                                    'rhythm_ratios': sub_pattern.get('rhythm_ratios', []),
                                    'duration_ratios': sub_pattern.get('duration_ratios', []),
                                    'source_piece': sub_occ.get('piece_id', 'unknown'),
                                })
                                continue

                # Keep original (no substitution or substitution failed)
                result[gm].append({
                    **event,
                    'source_piece': 'template',
                })

        if self.verbose:
            print(f"\nGeneration stats:")
            print(f"  Total events: {stats['total_events']}")
            print(f"  Structural patterns (repeat 2+): {stats['structural_patterns']}")
            print(f"  Substitution attempts: {stats['substitution_attempts']}")
            print(f"  Successful substitutions: {stats['substitution_success']}")
            print(f"  Unique source pieces: {len(stats['unique_sources'])}")
            if stats['unique_sources']:
                for src in list(stats['unique_sources'])[:5]:
                    print(f"    - {src[:50]}...")

        return dict(result)

    def expand_to_notes(
        self,
        events: Dict[int, List[dict]],
        quantize: bool = True,
    ) -> Dict[int, List[dict]]:
        """
        Expand pattern events to individual notes.

        Uses rhythm_ratios to determine time between notes:
        - rhythm_ratio = 0.0 → simultaneous (chord)
        - rhythm_ratio = 1.0 → one tau apart
        - rhythm_ratio = 2.0 → two tau apart
        """
        notes_by_gm = defaultdict(list)

        for gm, event_list in events.items():
            for event in event_list:
                onset = event['onset']
                pitch = event['pitch']
                intervals = event.get('intervals', [])
                rhythm_ratios = event.get('rhythm_ratios', [])
                tau = event.get('tau', 480)

                if quantize:
                    tau = quantize_tau(tau)
                    onset = (onset // 120) * 120

                # First note
                notes_by_gm[gm].append({
                    'pitch': pitch,
                    'onset': onset,
                    'duration': tau,
                    'velocity': 80,
                })

                # Subsequent notes using rhythm_ratios for timing
                current_onset = onset
                current_pitch = pitch

                for i, interval in enumerate(intervals):
                    # Get rhythm_ratio for this transition
                    if i < len(rhythm_ratios):
                        ratio = rhythm_ratios[i]
                    else:
                        ratio = 1.0  # Default to one tau

                    # ratio=0 means simultaneous, ratio=1 means one tau
                    time_delta = int(tau * ratio)
                    current_onset += time_delta
                    current_pitch += interval

                    if 0 <= current_pitch <= 127:
                        notes_by_gm[gm].append({
                            'pitch': current_pitch,
                            'onset': current_onset,
                            'duration': tau,
                            'velocity': 80,
                        })

        # Sort by onset
        for gm in notes_by_gm:
            notes_by_gm[gm] = sorted(notes_by_gm[gm], key=lambda x: x['onset'])

        return dict(notes_by_gm)

    def _is_chord_pattern(self, pid: str) -> bool:
        """Check if pattern has simultaneous notes (rhythm_ratio = 0)."""
        p = self.patterns.get(pid, {})
        rhythm_ratios = p.get('rhythm_ratios', [])
        return any(r == 0.0 for r in rhythm_ratios)

    def _is_walking_bass(self, pid: str) -> bool:
        """Check if pattern has steady rhythm (walking bass style)."""
        p = self.patterns.get(pid, {})
        rhythm_ratios = p.get('rhythm_ratios', [])
        if not rhythm_ratios or len(rhythm_ratios) < 2:
            return False
        # Walking bass = steady rhythm (ratios near 1.0)
        near_one = sum(1 for r in rhythm_ratios if 0.7 < r < 1.3)
        return near_one >= len(rhythm_ratios) * 0.6

    def generate_free(
        self,
        instruments: List[int] = None,
        num_bars: int = 16,
        ticks_per_beat: int = 480,
        seed: Optional[int] = None,
        drum_repeat: int = 4,   # Repeat drum pattern every N bars
        source_piece: str = None,  # Restrict to patterns from this piece for harmonic coherence
    ) -> Dict[int, List[dict]]:
        """
        Generate from scratch without a template.

        Uses sequential co-occurrence to chain patterns together.
        No template needed - fully free generation.

        Special handling:
        - Drums: Repeat patterns for consistency (real drums loop)
        - Piano: Use ACTUAL chord patterns (rhythm_ratio=0 = simultaneous notes)
        - Bass: Use walking bass patterns (steady rhythm_ratios)

        Args:
            instruments: List of GM programs to include (default: [0, 32, 128])
            num_bars: Length in bars
            ticks_per_beat: Timing resolution
            seed: Random seed
            drum_repeat: Repeat drum pattern every N bars
            source_piece: If specified, only use patterns from this piece (ensures harmonic coherence)

        Returns:
            Generated events by GM program
        """
        if seed is not None:
            random.seed(seed)

        if instruments is None:
            instruments = [0, 32, 128]  # Piano, Bass, Drums

        # If source_piece specified, find patterns from that piece only
        piece_patterns = None
        if source_piece:
            piece_id = self.find_piece(source_piece)
            if piece_id and piece_id in self.piece_tracks:
                piece_patterns = set()
                for gm, events in self.piece_tracks[piece_id].items():
                    for e in events:
                        piece_patterns.add(e['pattern_id'])
                if self.verbose:
                    print(f"Restricting to {len(piece_patterns)} patterns from {piece_id}")

        total_ticks = num_bars * 4 * ticks_per_beat
        result = defaultdict(list)

        stats = {
            'total_patterns': 0,
            'unique_sources': set(),
            'piano_chords': 0,
            'bass_walking': 0,
        }

        # HARMONIC DEPENDENCY ORDER: drums -> bass -> piano -> others
        # This ensures harmonic coherence through vertical co-occurrence
        def instrument_order(gm):
            if gm == 128:  # Drums - independent
                return 0
            elif gm in (32, 33, 34, 35):  # Bass family
                return 1
            elif gm == 0:  # Piano - depends on bass
                return 2
            else:  # Other instruments
                return 3

        sorted_instruments = sorted(instruments, key=instrument_order)
        generated_bass_patterns = set()  # Track bass patterns for piano co-occurrence

        for gm in sorted_instruments:
            # Get all patterns for this instrument
            available = self.pattern_by_gm.get(gm, [])
            if not available:
                continue

            # Filter to leaf patterns only (short)
            available = [
                pid for pid in available
                if len(self.patterns[pid].get('pitch_intervals', [])) <= 16
            ]

            if not available:
                continue

            # PIANO: Use chord patterns that CO-OCCURRED with bass (harmonic coherence)
            if gm == 0:
                chord_patterns = set(pid for pid in available if self._is_chord_pattern(pid))
                if not chord_patterns:
                    chord_patterns = set(available)  # Fallback
                stats['piano_chords'] = len(chord_patterns)

                # Find piano patterns that co-occurred with generated bass patterns
                # Also check what piano patterns co-occurred WITH each other
                harmonically_compatible = set()
                for bass_pid in generated_bass_patterns:
                    cooccur = self.vertical_cooccur.get(bass_pid, {})
                    for piano_pid in cooccur.get('piano', []):
                        if piano_pid in chord_patterns:
                            harmonically_compatible.add(piano_pid)

                # If no bass co-occurrence, use chord patterns with high counts
                if not harmonically_compatible:
                    # Prefer patterns that appear in vertical co-occurrence (proven harmony)
                    for pid in chord_patterns:
                        if pid in self.vertical_cooccur:
                            harmonically_compatible.add(pid)
                    stats['piano_in_cooccur'] = len(harmonically_compatible)

                # Use compatible patterns if we have enough, otherwise fallback
                if len(harmonically_compatible) >= 10:
                    chord_patterns = harmonically_compatible
                    stats['harmonic_compatible'] = len(harmonically_compatible)

                chord_patterns_list = list(chord_patterns)
                current_pid = random.choice(chord_patterns_list)
                current_onset = 0

                while current_onset < total_ticks:
                    p = self.patterns.get(current_pid, {})
                    occs = p.get('occurrences', [])
                    if not occs:
                        break

                    occ = random.choice(occs)
                    tau = quantize_tau(occ.get('tau_offset', 480))
                    intervals = p.get('pitch_intervals', [])
                    pattern_duration = tau * (len(intervals) + 1)

                    result[gm].append({
                        'pattern_id': current_pid,
                        'gm': gm,
                        'onset': current_onset,
                        'pitch': occ.get('first_pitch', 60),
                        'tau': tau,
                        'intervals': intervals,
                        'duration_ratios': p.get('duration_ratios', []),
                        'rhythm_ratios': p.get('rhythm_ratios', []),
                        'source_piece': occ.get('piece_id', 'unknown'),
                    })

                    stats['total_patterns'] += 1
                    stats['unique_sources'].add(occ.get('piece_id', 'unknown'))
                    current_onset += pattern_duration

                    # Next pattern - prefer chord patterns that co-occurred with bass
                    next_candidates = self.seq_next[gm].get(current_pid, Counter())
                    if next_candidates:
                        # Filter to harmonically compatible chord patterns
                        chord_next = [(c, w) for c, w in next_candidates.items()
                                      if c in chord_patterns]
                        if chord_next:
                            candidates, weights = zip(*chord_next)
                            current_pid = random.choices(candidates, weights=weights, k=1)[0]
                        else:
                            current_pid = random.choice(chord_patterns_list)
                    else:
                        current_pid = random.choice(chord_patterns_list)

            # BASS: Use walking bass patterns (steady rhythm)
            elif gm in [32, 33]:
                walking_patterns = [pid for pid in available if self._is_walking_bass(pid)]
                if not walking_patterns:
                    walking_patterns = available  # Fallback
                stats['bass_walking'] = len(walking_patterns)

                current_pid = random.choice(walking_patterns)
                current_onset = 0

                while current_onset < total_ticks:
                    p = self.patterns.get(current_pid, {})
                    occs = p.get('occurrences', [])
                    if not occs:
                        break

                    occ = random.choice(occs)
                    tau = quantize_tau(occ.get('tau_offset', 480))
                    intervals = p.get('pitch_intervals', [])
                    pattern_duration = tau * (len(intervals) + 1)

                    result[gm].append({
                        'pattern_id': current_pid,
                        'gm': gm,
                        'onset': current_onset,
                        'pitch': occ.get('first_pitch', 40),  # Bass range
                        'tau': tau,
                        'intervals': intervals,
                        'duration_ratios': p.get('duration_ratios', []),
                        'rhythm_ratios': p.get('rhythm_ratios', []),
                        'source_piece': occ.get('piece_id', 'unknown'),
                    })

                    # Track for vertical co-occurrence with piano
                    generated_bass_patterns.add(current_pid)

                    stats['total_patterns'] += 1
                    stats['unique_sources'].add(occ.get('piece_id', 'unknown'))
                    current_onset += pattern_duration

                    # Next pattern - prefer walking patterns
                    next_candidates = self.seq_next[gm].get(current_pid, Counter())
                    if next_candidates:
                        walking_next = [(c, w) for c, w in next_candidates.items()
                                        if c in walking_patterns]
                        if walking_next:
                            candidates, weights = zip(*walking_next)
                            current_pid = random.choices(candidates, weights=weights, k=1)[0]
                        else:
                            current_pid = random.choice(walking_patterns)
                    else:
                        current_pid = random.choice(walking_patterns)

            # DRUMS: Select REAL groove patterns (diverse drums) and repeat
            elif gm == 128:
                # Filter to REAL grooves: diverse intervals (not just repeated single drum)
                groove_patterns = []
                for pid in available:
                    p = self.patterns[pid]
                    intervals = p.get('pitch_intervals', [])
                    if len(intervals) < 4:  # Too short
                        continue
                    unique_intervals = len(set(intervals))
                    if unique_intervals < 3:  # Must have diverse drums
                        continue
                    # Must include some larger movement (kick/snare jumps)
                    if not any(abs(i) >= 4 for i in intervals):
                        continue
                    if p.get('count', 0) >= 100:  # Some occurrence threshold
                        groove_patterns.append(pid)

                if not groove_patterns:
                    groove_patterns = available
                stats['drum_grooves'] = len(groove_patterns)

                # Pick ONE solid groove pattern and use it for the whole cycle
                # Weight by count (higher = more proven)
                weights = [self.patterns[pid].get('count', 1) for pid in groove_patterns]
                main_groove = random.choices(groove_patterns, weights=weights, k=1)[0]

                cycle_ticks = drum_repeat * 4 * ticks_per_beat
                drum_cycle = []
                current_onset = 0

                # Fill the cycle with the main groove pattern
                while current_onset < cycle_ticks:
                    p = self.patterns.get(main_groove, {})
                    occs = p.get('occurrences', [])
                    if not occs:
                        break

                    occ = random.choice(occs)
                    tau = quantize_tau(occ.get('tau_offset', 480))
                    intervals = p.get('pitch_intervals', [])
                    rhythm_ratios = p.get('rhythm_ratios', [])
                    pattern_duration = tau * (len(intervals) + 1)

                    drum_cycle.append({
                        'pattern_id': main_groove,
                        'gm': gm,
                        'onset': current_onset,
                        'pitch': occ.get('first_pitch', 42),  # Keep drum pitch as-is
                        'tau': tau,
                        'intervals': intervals,
                        'rhythm_ratios': rhythm_ratios,
                        'duration_ratios': p.get('duration_ratios', []),
                        'source_piece': occ.get('piece_id', 'unknown'),
                    })

                    stats['total_patterns'] += 1
                    stats['unique_sources'].add(occ.get('piece_id', 'unknown'))
                    current_onset += pattern_duration

                # Repeat the cycle
                for repeat in range(num_bars // drum_repeat + 1):
                    offset = repeat * cycle_ticks
                    for event in drum_cycle:
                        new_onset = event['onset'] + offset
                        if new_onset < total_ticks:
                            result[gm].append({**event, 'onset': new_onset})

            # PIANO: Layer multiple voices for chords
            elif gm == 0:
                for voice in range(piano_voices):
                    # Different starting pitch range per voice
                    pitch_offset = (voice - 1) * 12  # -12, 0, +12 for 3 voices

                    current_pid = random.choice(available)
                    current_onset = 0

                    while current_onset < total_ticks:
                        p = self.patterns.get(current_pid, {})
                        occs = p.get('occurrences', [])
                        if not occs:
                            break

                        occ = random.choice(occs)
                        tau = quantize_tau(occ.get('tau_offset', 480))
                        intervals = p.get('pitch_intervals', [])
                        pattern_duration = tau * (len(intervals) + 1)

                        # Adjust pitch for voice layering
                        base_pitch = occ.get('first_pitch', 60) + pitch_offset
                        base_pitch = max(36, min(96, base_pitch))  # Keep in range

                        result[gm].append({
                            'pattern_id': current_pid,
                            'gm': gm,
                            'onset': current_onset,
                            'pitch': base_pitch,
                            'tau': tau,
                            'intervals': intervals,
                            'duration_ratios': p.get('duration_ratios', []),
                            'source_piece': occ.get('piece_id', 'unknown'),
                        })

                        stats['total_patterns'] += 1
                        stats['unique_sources'].add(occ.get('piece_id', 'unknown'))
                        current_onset += pattern_duration

                        # Next pattern
                        next_candidates = self.seq_next[gm].get(current_pid, Counter())
                        if next_candidates:
                            candidates = list(next_candidates.keys())
                            weights = [next_candidates[c] for c in candidates]
                            current_pid = random.choices(candidates, weights=weights, k=1)[0]
                        else:
                            current_pid = random.choice(available)

            # OTHER INSTRUMENTS: Normal generation
            else:
                current_pid = random.choice(available)
                current_onset = 0

                while current_onset < total_ticks:
                    p = self.patterns.get(current_pid, {})
                    occs = p.get('occurrences', [])
                    if not occs:
                        break

                    occ = random.choice(occs)
                    tau = quantize_tau(occ.get('tau_offset', 480))
                    intervals = p.get('pitch_intervals', [])
                    pattern_duration = tau * (len(intervals) + 1)

                    result[gm].append({
                        'pattern_id': current_pid,
                        'gm': gm,
                        'onset': current_onset,
                        'pitch': occ.get('first_pitch', 60),
                        'tau': tau,
                        'intervals': intervals,
                        'duration_ratios': p.get('duration_ratios', []),
                        'source_piece': occ.get('piece_id', 'unknown'),
                    })

                    stats['total_patterns'] += 1
                    stats['unique_sources'].add(occ.get('piece_id', 'unknown'))
                    current_onset += pattern_duration

                    # Next pattern
                    next_candidates = self.seq_next[gm].get(current_pid, Counter())
                    if next_candidates:
                        candidates = list(next_candidates.keys())
                        weights = [next_candidates[c] for c in candidates]
                        current_pid = random.choices(candidates, weights=weights, k=1)[0]
                    else:
                        current_pid = random.choice(available)

        if self.verbose:
            print(f"\nFree generation stats:")
            print(f"  Instruments (ordered): {sorted_instruments}")
            print(f"  Piano chord patterns available: {stats.get('piano_chords', 0)}")
            if 'harmonic_compatible' in stats:
                print(f"  Harmonically compatible with bass: {stats['harmonic_compatible']}")
            if 'piano_in_cooccur' in stats:
                print(f"  Piano patterns in co-occur data: {stats['piano_in_cooccur']}")
            print(f"  Bass walking patterns available: {stats.get('bass_walking', 0)}")
            print(f"  Drum groove patterns (count>=500): {stats.get('drum_grooves', 0)}")
            print(f"  Drum repeat: every {drum_repeat} bars")
            print(f"  Total patterns used: {stats['total_patterns']}")
            print(f"  Unique source pieces: {len(stats['unique_sources'])}")
            for src in list(stats['unique_sources'])[:5]:
                print(f"    - {src[:50]}...")

        return dict(result)

    def to_midi(
        self,
        notes: Dict[int, List[dict]],
        output_path: str,
        ticks_per_beat: int = 480,
        tempo_bpm: int = 120,
    ):
        """Convert notes to MIDI file."""
        mid = MidiFile(ticks_per_beat=ticks_per_beat)

        for gm, note_list in sorted(notes.items()):
            track = MidiTrack()
            mid.tracks.append(track)

            name = GM_NAMES.get(gm, f'GM{gm}')
            track.append(MetaMessage('track_name', name=name, time=0))

            if gm != 128:
                track.append(Message('program_change', program=gm, time=0, channel=0))

            # Build events list
            events = []
            for note in note_list:
                events.append({
                    'type': 'note_on',
                    'pitch': note['pitch'],
                    'time': note['onset'],
                    'velocity': note['velocity'],
                })
                events.append({
                    'type': 'note_off',
                    'pitch': note['pitch'],
                    'time': note['onset'] + note['duration'],
                    'velocity': 0,
                })

            events = sorted(events, key=lambda x: (x['time'], x['type'] == 'note_off'))

            current_time = 0
            channel = 9 if gm == 128 else 0

            for event in events:
                delta = event['time'] - current_time
                current_time = event['time']

                track.append(Message(
                    event['type'],
                    note=event['pitch'],
                    velocity=event['velocity'],
                    time=delta,
                    channel=channel,
                ))

        # Add tempo to first track
        if mid.tracks:
            tempo = mido.bpm2tempo(tempo_bpm)
            mid.tracks[0].insert(0, MetaMessage('set_tempo', tempo=tempo, time=0))

        mid.save(output_path)

        if self.verbose:
            total_notes = sum(len(v) for v in notes.values())
            print(f"\nSaved to {output_path}")
            print(f"  Total notes: {total_notes}")
            print(f"  Tracks: {len(notes)}")


def main():
    parser = argparse.ArgumentParser(description='Generative Composer')
    parser.add_argument('--template', '-t', type=str, default=None,
                        help='Template piece (partial name match)')
    parser.add_argument('--start-bar', type=int, default=0,
                        help='Starting bar')
    parser.add_argument('--bars', '-b', type=int, default=16,
                        help='Number of bars')
    parser.add_argument('--variation', '-v', type=float, default=0.3,
                        help='Variation amount 0.0-1.0')
    parser.add_argument('--require-both', action='store_true',
                        help='Require both prev/next context for substitution')
    parser.add_argument('--output', '-o', type=str, default='generated.mid',
                        help='Output MIDI file')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoint_v55_pure_contour_1000files_patterns.json',
                        help='Patterns checkpoint file')
    parser.add_argument('--list-pieces', action='store_true',
                        help='List available pieces and exit')
    parser.add_argument('--free', action='store_true',
                        help='Free generation mode (no template)')
    parser.add_argument('--instruments', type=str, default='0,32,128',
                        help='Comma-separated GM programs for free generation')

    args = parser.parse_args()

    # Find checkpoint
    base_dir = Path(__file__).parent.parent
    checkpoint_path = base_dir / args.checkpoint

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    # Initialize
    composer = GenerativeComposer(str(checkpoint_path))

    if args.list_pieces:
        print("\nAvailable pieces (top 20):")
        for name, count in composer.list_pieces()[:20]:
            print(f"  {count:5d} events: {name}")
        return

    # Free generation mode
    if args.free:
        instruments = [int(x) for x in args.instruments.split(',')]
        print(f"Free generation mode with instruments: {instruments}")

        generated = composer.generate_free(
            instruments=instruments,
            num_bars=args.bars,
            seed=args.seed,
        )

        # Expand to notes
        notes = composer.expand_to_notes(generated)

        # Save
        output_path = base_dir / 'outputs' / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composer.to_midi(notes, str(output_path))
        return

    if not args.template:
        print("Error: --template is required for template-based generation")
        print("Use --free for template-free generation")
        parser.print_help()
        return

    # Find template piece
    piece_id = composer.find_piece(args.template)
    if not piece_id:
        print(f"Piece not found: {args.template}")
        print("Use --list-pieces to see available pieces")
        return

    print(f"Using template: {piece_id}")

    # Get template
    template = composer.get_template(
        piece_id,
        start_bar=args.start_bar,
        num_bars=args.bars,
    )

    # Generate
    generated = composer.generate(
        template,
        variation=args.variation,
        require_both_context=args.require_both,
        seed=args.seed,
    )

    # Expand to notes
    notes = composer.expand_to_notes(generated)

    # Save
    output_path = base_dir / 'outputs' / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composer.to_midi(notes, str(output_path))


if __name__ == '__main__':
    main()
