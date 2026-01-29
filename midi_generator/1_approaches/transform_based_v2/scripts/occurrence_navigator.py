#!/usr/bin/env python3
"""
Occurrence Navigator - Generate by navigating through REAL occurrences.

The key insight: Instead of synthesizing notes weighted by statistics,
we NAVIGATE through actual occurrences from the corpus:

1. Use interval PPM* to pick melodic direction
2. Find occurrences where that interval actually occurred
3. Pull REAL co-occurrences from the SAME piece/beat
4. Emit with REAL rhythm from the occurrence

This is fundamentally different from the synthetic approach:
- Synthetic: Generate note, weight by statistics
- Navigator: Find occurrence matching criteria, emit with real context

Benefits:
- Real timing relationships (rhythm_ratios preserved)
- Real vertical coordination (from same piece/beat)
- Real piece context binds tracks together
"""

import orjson
import random
import pickle
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


class OccurrenceNavigator:
    """Generate by navigating through real corpus occurrences."""

    def __init__(
        self,
        patterns: dict,
        ppm_model_path: str,
        verbose: bool = True
    ):
        self.patterns = patterns
        self.verbose = verbose

        # Load interval PPM model
        with open(ppm_model_path, 'rb') as f:
            self.ppm_model = pickle.load(f)

        if verbose:
            print(f"Loaded interval PPM: vocab_size={self.ppm_model['vocab_size']}, "
                  f"max_order={self.ppm_model['max_order']}")

        self._build_indices()

    def _build_indices(self):
        """Build indices for occurrence lookup."""
        if self.verbose:
            print("Building occurrence indices...")

        # Index 1: (gm, first_interval) -> [(pid, occ_index), ...]
        # Used to find occurrences with a specific melodic direction
        self.gm_interval_index = defaultdict(list)

        # Index 2: (piece, beat) -> [(gm, pid, occ_idx), ...]
        # Used to find co-occurrences at the same time
        self.beat_occs = defaultdict(list)

        # Index 3: gm -> [(pid, occ_idx), ...]
        # Used for initial seeding
        self.gm_occs = defaultdict(list)

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            intervals = p.get('pitch_intervals', [])
            first_interval = intervals[0] if intervals else 0

            for occ_idx, occ in enumerate(p.get('occurrences', [])):
                # Index by (gm, first_interval)
                self.gm_interval_index[(gm, first_interval)].append((pid, occ_idx))

                # Index by (piece, beat) - use half-beat granularity for better matching
                piece = occ.get('piece_id', 'unknown')
                onset = occ.get('onset_time', 0)
                half_beat = onset // 240  # Half-beat (240 ticks) for finer matching
                self.beat_occs[(piece, half_beat)].append((gm, pid, occ_idx))

                # Index by gm
                self.gm_occs[gm].append((pid, occ_idx))

        if self.verbose:
            print(f"  (gm, interval) index: {len(self.gm_interval_index)} entries")
            print(f"  (piece, half_beat) index: {len(self.beat_occs)} entries")
            print(f"  gm index: {len(self.gm_occs)} GMs")

            # Count multi-instrument half-beats
            multi_inst = sum(1 for k, v in self.beat_occs.items()
                             if len(set(gm for gm, _, _ in v)) >= 2)
            print(f"  Multi-instrument half-beats: {multi_inst}/{len(self.beat_occs)} "
                  f"({100*multi_inst/len(self.beat_occs):.1f}%)")

    def sample_interval_ppm(
        self,
        gm: int,
        interval_history: List[int],
        escape_probability: float = 0.1
    ) -> int:
        """Sample next interval using PPM* with backoff."""
        ppm_counts = self.ppm_model['ppm_counts']
        max_order = self.ppm_model['max_order']
        interval_to_id = self.ppm_model['interval_to_id']
        id_to_interval = self.ppm_model['id_to_interval']

        if gm not in ppm_counts:
            return random.choice(list(interval_to_id.keys()))

        gm_counts = ppm_counts[gm]
        history_ids = [interval_to_id.get(iv, 0) for iv in interval_history if iv in interval_to_id]

        for order in range(min(max_order, len(history_ids)), -1, -1):
            if order > 0 and random.random() < escape_probability:
                continue

            context = tuple(history_ids[-order:]) if order > 0 else ()

            if order not in gm_counts or context not in gm_counts[order]:
                continue

            transitions = gm_counts[order][context]
            if not transitions:
                continue

            ids = list(transitions.keys())
            counts = list(transitions.values())
            total = sum(counts)
            weights = [c / total for c in counts]

            selected_id = random.choices(ids, weights=weights)[0]
            return id_to_interval.get(selected_id, 0)

        return random.choice([-2, -1, 0, 1, 2])

    def find_occurrence_with_interval(
        self,
        gm: int,
        target_interval: int,
        tolerance: int = 2
    ) -> Optional[Tuple[str, int, dict]]:
        """Find an occurrence that starts with approximately the target interval.

        Args:
            gm: GM program
            target_interval: Target first interval (from PPM*)
            tolerance: How much interval mismatch to allow

        Returns:
            (pid, occ_idx, pattern) or None
        """
        # Try exact match first
        candidates = self.gm_interval_index.get((gm, target_interval), [])

        # If no exact match, try within tolerance
        if not candidates:
            for delta in range(1, tolerance + 1):
                for offset in [delta, -delta]:
                    candidates = self.gm_interval_index.get((gm, target_interval + offset), [])
                    if candidates:
                        break
                if candidates:
                    break

        if not candidates:
            # Fallback to any occurrence for this GM
            candidates = self.gm_occs.get(gm, [])

        if not candidates:
            return None

        pid, occ_idx = random.choice(candidates)
        return pid, occ_idx, self.patterns[pid]

    def get_cooccurrences(
        self,
        piece_id: str,
        half_beat: int,
        target_gms: Set[int],
        exclude_gm: int = None,
        window: int = 2
    ) -> List[Tuple[int, str, int, dict]]:
        """Get co-occurrences from the same piece/beat with window search.

        Args:
            piece_id: Piece to look in
            half_beat: Half-beat number (240 ticks)
            target_gms: GMs we want to find co-occurrences for
            exclude_gm: GM to exclude (the lead we already have)
            window: Search window in half-beats (default 2 = +/- 1 beat)

        Returns:
            List of (gm, pid, occ_idx, pattern)
        """
        results = []
        found_gms = set()

        # Search window centered on target half-beat
        # Prefer exact match, then adjacent
        for offset in range(window + 1):
            for delta in ([0] if offset == 0 else [offset, -offset]):
                key = (piece_id, half_beat + delta)
                if key not in self.beat_occs:
                    continue

                for gm, pid, occ_idx in self.beat_occs[key]:
                    if gm in target_gms and gm != exclude_gm and gm not in found_gms:
                        results.append((gm, pid, occ_idx, self.patterns[pid]))
                        found_gms.add(gm)

                # Stop early if we found all target GMs
                if found_gms >= target_gms - {exclude_gm}:
                    return results

        return results

    def emit_occurrence(
        self,
        pid: str,
        occ_idx: int,
        pattern: dict,
        start_time: int,
        output: Dict[int, List[dict]]
    ) -> Tuple[int, int]:
        """Emit notes from an occurrence with real rhythm.

        Args:
            pid: Pattern ID
            occ_idx: Occurrence index
            pattern: Pattern dict
            start_time: Output start time (ticks)
            output: Output dict to append to

        Returns:
            (end_time, n_notes_emitted)
        """
        gm = pattern.get('gm_program', 0)
        occ = pattern['occurrences'][occ_idx]

        first_pitch = occ.get('first_pitch', 60)
        intervals = pattern.get('pitch_intervals', [])
        rhythm_ratios = occ.get('rhythm_ratios', pattern.get('rhythm_ratios', []))

        # Build note sequence
        pitches = [first_pitch]
        for iv in intervals:
            pitches.append(pitches[-1] + iv)

        # Calculate IOIs (inter-onset intervals) from rhythm_ratios
        # rhythm_ratios are SUCCESSIVE IOI RATIOS: each IOI is prev_IOI * ratio
        # tau_offset is the initial IOI
        tau = occ.get('tau_offset', 240)  # Default to eighth note if missing

        iois = []
        if rhythm_ratios:
            # First IOI is tau_offset
            current_ioi = tau
            iois.append(current_ioi)

            # Each subsequent IOI = previous_IOI * ratio
            for r in rhythm_ratios:
                # Clamp ratio to reasonable range to avoid extreme values
                r = max(0.1, min(10.0, r))
                current_ioi = int(current_ioi * r)
                # Clamp IOI to reasonable range (1/32 note to 2 bars)
                current_ioi = max(60, min(3840, current_ioi))
                iois.append(current_ioi)
        else:
            # No rhythm data - use constant eighth notes
            iois = [240] * len(pitches)

        # Emit notes
        current_time = start_time
        for i, pitch in enumerate(pitches):
            ioi = iois[i] if i < len(iois) else 240

            # Note duration is slightly less than IOI for articulation
            dur = int(ioi * 0.9)

            output[gm].append({
                'pitch': max(24, min(108, pitch)),
                'onset': current_time,
                'duration': max(60, dur),  # Minimum duration
                'velocity': 80,
            })

            # Advance by IOI (except for last note)
            if i < len(pitches) - 1:
                current_time += ioi

        # Return end time (after last note's duration)
        final_ioi = iois[-1] if iois else 240
        return current_time + final_ioi, len(pitches)

    def generate(
        self,
        n_beats: int,
        instruments: List[int],
        lead_gm: int = None
    ) -> Dict[int, List[dict]]:
        """Generate by navigating through real occurrences.

        Architecture:
            for beat in range(n_beats):
                # Use interval PPM* to pick melodic direction for lead
                target_interval = interval_ppm.sample(lead_history)

                # Find occurrence where this interval actually occurred
                lead_occ = find_occurrence_with_interval(lead_gm, target_interval)

                # Pull REAL co-occurrences from THAT piece/beat
                co_occs = get_cooccurrences(lead_occ.piece_id, lead_occ.beat)

                # Emit with REAL rhythm from occurrences
                emit(lead_occ)
                for co in co_occs:
                    emit(co)
        """
        if lead_gm is None:
            lead_gm = instruments[0]

        output = {gm: [] for gm in instruments}
        other_gms = set(instruments) - {lead_gm}

        # State
        interval_history = {gm: [] for gm in instruments}
        current_time = {gm: 0 for gm in instruments}

        # Track which pieces we've drawn from (for variety)
        used_pieces = Counter()

        # Hit rate tracking
        stats = {
            'total_lookups': 0,
            'cooc_found': 0,  # Found real co-occurrences
            'cooc_partial': 0,  # Found some but not all
            'cooc_miss': 0,  # Had to generate independently
        }

        beat = 0
        while beat < n_beats:
            # 1. Use interval PPM* to pick melodic direction for lead
            target_interval = self.sample_interval_ppm(lead_gm, interval_history[lead_gm])

            # 2. Find occurrence where this interval actually occurred
            result = self.find_occurrence_with_interval(lead_gm, target_interval)
            if result is None:
                beat += 1
                continue

            pid, occ_idx, pattern = result
            occ = pattern['occurrences'][occ_idx]
            piece_id = occ.get('piece_id', 'unknown')
            occ_half_beat = occ.get('onset_time', 0) // 240  # Half-beat for matching

            # Prefer variety in pieces (but not too aggressively - coherence matters)
            if used_pieces[piece_id] > 10 and random.random() < 0.5:
                # Try to find alternative
                for _ in range(3):
                    result2 = self.find_occurrence_with_interval(lead_gm, target_interval)
                    if result2:
                        pid2, occ_idx2, pattern2 = result2
                        occ2 = pattern2['occurrences'][occ_idx2]
                        piece_id2 = occ2.get('piece_id', 'unknown')
                        if used_pieces[piece_id2] < used_pieces[piece_id]:
                            pid, occ_idx, pattern, occ = pid2, occ_idx2, pattern2, occ2
                            piece_id = piece_id2
                            occ_half_beat = occ.get('onset_time', 0) // 240
                            break

            used_pieces[piece_id] += 1

            # 3. Emit lead occurrence
            lead_start = current_time[lead_gm]
            end_time, n_notes = self.emit_occurrence(pid, occ_idx, pattern, lead_start, output)

            # Update lead state
            intervals = pattern.get('pitch_intervals', [])
            interval_history[lead_gm].extend(intervals)
            if len(interval_history[lead_gm]) > 10:
                interval_history[lead_gm] = interval_history[lead_gm][-10:]
            current_time[lead_gm] = end_time

            # 4. Pull REAL co-occurrences from that piece/half-beat (with window search)
            co_occs = self.get_cooccurrences(piece_id, occ_half_beat, other_gms, exclude_gm=lead_gm)
            stats['total_lookups'] += 1

            found_gms = set(gm for gm, _, _, _ in co_occs)
            if found_gms == other_gms:
                stats['cooc_found'] += 1
            elif found_gms:
                stats['cooc_partial'] += 1
            else:
                stats['cooc_miss'] += 1

            if co_occs:
                # Emit co-occurrences with their real timing
                for co_gm, co_pid, co_occ_idx, co_pattern in co_occs:
                    co_start = lead_start  # Align with lead
                    co_end, co_notes = self.emit_occurrence(
                        co_pid, co_occ_idx, co_pattern, co_start, output
                    )

                    # Update state
                    co_intervals = co_pattern.get('pitch_intervals', [])
                    interval_history[co_gm].extend(co_intervals)
                    if len(interval_history[co_gm]) > 10:
                        interval_history[co_gm] = interval_history[co_gm][-10:]
                    current_time[co_gm] = max(current_time[co_gm], co_end)
            else:
                # No real co-occurrence found - generate independently using PPM*
                for gm in other_gms:
                    target_iv = self.sample_interval_ppm(gm, interval_history[gm])
                    result = self.find_occurrence_with_interval(gm, target_iv)
                    if result:
                        ind_pid, ind_occ_idx, ind_pattern = result
                        ind_start = lead_start
                        ind_end, _ = self.emit_occurrence(
                            ind_pid, ind_occ_idx, ind_pattern, ind_start, output
                        )
                        ind_intervals = ind_pattern.get('pitch_intervals', [])
                        interval_history[gm].extend(ind_intervals)
                        if len(interval_history[gm]) > 10:
                            interval_history[gm] = interval_history[gm][-10:]
                        current_time[gm] = max(current_time[gm], ind_end)

            beat += n_notes  # Advance by number of notes emitted

        # Store stats for analysis
        self.last_stats = stats
        return output

    def generate_to_midi(
        self,
        n_beats: int,
        instruments: List[int],
        output_path: str,
        bpm: int = 120,
        lead_gm: int = None
    ):
        """Generate and save to MIDI."""
        print(f"\nGenerating {n_beats} beats for {instruments}...")
        print(f"  Method: Occurrence navigation (real co-occurrences)")
        print(f"  Lead: GM {lead_gm or instruments[0]}")

        note_output = self.generate(n_beats, instruments, lead_gm)

        # Convert to MIDI
        mid = MidiFile(ticks_per_beat=480, type=1)

        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
        tempo_track.append(MetaMessage('end_of_track', time=0))

        channel_map = {}
        next_channel = 0

        for gm, note_list in sorted(note_output.items()):
            if not note_list:
                continue

            if gm not in channel_map:
                channel_map[gm] = next_channel
                next_channel += 1
                if next_channel == 9:
                    next_channel = 10

            channel = channel_map[gm]
            track = MidiTrack()
            mid.tracks.append(track)

            track.append(Message('program_change', program=gm % 128, channel=channel, time=0))

            events = []
            for n in note_list:
                events.append((n['onset'], 'on', n['pitch'], n['velocity']))
                events.append((n['onset'] + n['duration'], 'off', n['pitch'], 0))

            events.sort(key=lambda x: (x[0], x[1] == 'on'))

            last_time = 0
            for event_time, event_type, pitch, vel in events:
                delta = event_time - last_time
                if event_type == 'on':
                    track.append(Message('note_on', note=pitch, velocity=vel, channel=channel, time=delta))
                else:
                    track.append(Message('note_off', note=pitch, velocity=0, channel=channel, time=delta))
                last_time = event_time

            track.append(MetaMessage('end_of_track', time=0))

        mid.save(output_path)

        total_notes = sum(len(n) for n in note_output.values())
        print(f"\nSaved to: {output_path}")
        print(f"  Tracks: {len(mid.tracks)}")
        print(f"  Notes: {total_notes}")

        # Print co-occurrence hit rate stats
        if hasattr(self, 'last_stats') and self.last_stats:
            stats = self.last_stats
            total = stats['total_lookups']
            if total > 0:
                print(f"\n=== CO-OCCURRENCE HIT RATE ===")
                print(f"  Total lookups: {total}")
                print(f"  Full hit (all GMs found):   {stats['cooc_found']:3d} ({100*stats['cooc_found']/total:.1f}%)")
                print(f"  Partial hit (some GMs):     {stats['cooc_partial']:3d} ({100*stats['cooc_partial']/total:.1f}%)")
                print(f"  Miss (fallback to PPM):     {stats['cooc_miss']:3d} ({100*stats['cooc_miss']/total:.1f}%)")
                hit_rate = (stats['cooc_found'] + stats['cooc_partial']) / total
                print(f"  Combined hit rate: {100*hit_rate:.1f}%")

        self._analyze_output(note_output, instruments)

        return note_output

    def _analyze_output(self, note_output: Dict[int, List[dict]], instruments: List[int]):
        """Analyze output quality."""
        print("\n=== ANALYSIS ===")

        consonant_intervals = {0, 3, 4, 5, 7, 8, 9}

        # Horizontal: interval statistics
        print("\nHorizontal (per-track):")
        for gm in instruments:
            notes = note_output[gm]
            if len(notes) < 2:
                continue

            intervals = [notes[i+1]['pitch'] - notes[i]['pitch'] for i in range(len(notes)-1)]
            step_count = sum(1 for iv in intervals if abs(iv) <= 2)
            step_ratio = step_count / len(intervals) if intervals else 0

            print(f"  GM {gm}: {len(notes)} notes, step_ratio={step_ratio:.2f}")

        # Vertical: harmony at sync points
        print("\nVertical (sync points):")

        sync_points = set()
        for gm, notes in note_output.items():
            for n in notes:
                if (n['onset'] // 480) % 2 == 0:
                    sync_points.add(n['onset'] // 480 * 480)

        all_intervals = []
        for sync_time in sorted(sync_points)[:50]:
            pitches_at_sync = {}
            for gm, notes in note_output.items():
                for n in notes:
                    if n['onset'] <= sync_time < n['onset'] + n['duration']:
                        pitches_at_sync[gm] = n['pitch']
                        break

            if len(pitches_at_sync) >= 2:
                pcs = sorted(set(p % 12 for p in pitches_at_sync.values()))
                for i, pc1 in enumerate(pcs):
                    for pc2 in pcs[i+1:]:
                        interval = (pc2 - pc1) % 12
                        all_intervals.append(interval)

        if all_intervals:
            consonant_count = sum(1 for iv in all_intervals if iv in consonant_intervals)
            consonance = consonant_count / len(all_intervals)
            print(f"  Consonance: {100*consonance:.1f}%")


def main():
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python occurrence_navigator.py <checkpoint.npz> [-o output.mid] [--beats N]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_path = '/tmp/navigator.mid'
    n_beats = 128
    instruments = [65, 66, 67]

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '-o' and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--beats' and i + 1 < len(sys.argv):
            n_beats = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--instruments':
            instruments = []
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith('-'):
                instruments.append(int(sys.argv[i]))
                i += 1
        else:
            i += 1

    # Load patterns
    print(f"Loading checkpoint: {checkpoint_path}")
    data = np.load(checkpoint_path, allow_pickle=True)
    patterns_file = str(data['patterns_json_file'][0])

    base_dir = os.path.dirname(checkpoint_path)
    json_path = os.path.join(base_dir, patterns_file) if base_dir else patterns_file

    print(f"Loading patterns from: {json_path}")
    with open(json_path, 'rb') as f:
        patterns = orjson.loads(f.read())

    print(f"Loaded {len(patterns)} patterns")

    # Find PPM model
    ppm_path = os.path.join(base_dir, 'ppm_interval_model.pkl')
    if not os.path.exists(ppm_path):
        print(f"Error: PPM model not found at {ppm_path}")
        sys.exit(1)

    # Create navigator and generate
    navigator = OccurrenceNavigator(patterns, ppm_path)
    navigator.generate_to_midi(n_beats, instruments, output_path)


if __name__ == '__main__':
    main()
