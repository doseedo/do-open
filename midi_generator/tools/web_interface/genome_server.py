#!/usr/bin/env python3
"""
Genome Graph Server
==================

REST API for genome graph visualization and editing.

Endpoints:
    GET /                       - Serve web interface
    GET /api/stats              - Graph statistics (includes v43 data)
    GET /api/cytoscape          - Full graph in Cytoscape format
    GET /api/cytoscape/<piece>  - Piece subgraph in Cytoscape format
    GET /api/pattern/<id>       - Pattern details
    GET /api/edges/<id>         - Edges from pattern
    GET /api/pieces             - List all pieces
    GET /api/transforms         - Pitch transform vocabulary (T, I, R)
    GET /api/playback/<piece>   - Reconstruct MIDI events for playback
    POST /api/factor/<edge_id>  - Factor a compound edge
    POST /api/entangle          - Entangle edges into compound
    POST /api/clone             - Clone subgraph with transform
    POST /api/apply             - Apply transform to pattern

    v43+ Endpoints:
    GET /api/multi_factor       - τ, v, d transform vocabulary
    GET /api/track_derives      - Cross-track arrangement derivations
    GET /api/feature_importance - MDL-discovered useful features

Usage:
    python genome_server.py checkpoint_v43_1000files.npz [--port 8080]
"""

import json
import sys
import os
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
from collections import defaultdict

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '1_approaches', 'transform_based'))

from grammar.genome_graph import GenomeGraph, convert_v24_to_genome_graph
from grammar.music_dag import MusicDAG, build_dag_from_checkpoint, compress_dag_mdl, evaluate


class GenomeGraphHandler(BaseHTTPRequestHandler):
    """HTTP handler for genome graph API."""

    graph: GenomeGraph = None
    dag: MusicDAG = None  # Expression DAG (optional)
    edits: dict = {}  # edge_id -> modified_transform (persisted edits)
    pattern_edits: dict = {}  # pattern_id -> modified pattern data (pitch_classes, octaves, etc.)
    v24_rules: dict = None  # grammar_rules from v24 checkpoint (for playback reconstruction)
    track_info: list = None  # track_info from checkpoint (for is_drum detection)
    # v43 additions
    multi_factor: dict = None  # τ, v, d transform vocabulary
    track_derives: list = None  # Cross-track arrangement derivations
    feature_importance: dict = None  # MDL-discovered useful features
    checkpoint_stats: dict = None  # Stats from checkpoint (for /api/stats)

    def send_json(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_html(self, html, status=200):
        """Send HTML response."""
        self.send_response(status)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def reconstruct_piece_events(self, piece_id: str) -> dict:
        """Reconstruct MIDI events from genome graph occurrences for a piece.

        Uses the occurrences dict which stores all pattern instances with their
        absolute onset times. This is more accurate than edge-based reconstruction
        which only captures consecutive pairs of patterns.

        Returns events sorted by time.
        """
        from grammar.genome_graph import apply_pitch_transform
        from collections import Counter

        # Ticks per beat (standard MIDI)
        tpb = 480

        # Estimate original tempo from tau_offset distribution
        # Most pieces use eighth notes (240 ticks) or quarter notes (480 ticks)
        # at their native tempo. We look at common IOI values to estimate.
        estimated_bpm = 120  # Default fallback

        if self.v24_rules:
            # Collect tau_offsets for this piece
            tau_values = []
            for rid, rule in self.v24_rules.items():
                for occ in rule.get('occurrences', []):
                    if occ.get('piece_id') == piece_id:
                        tau = occ.get('tau_offset')
                        if tau and tau > 0:
                            tau_values.append(tau)

            # Tempo estimation is unreliable since tick values don't encode tempo.
            # Just use 120 BPM as default and let user adjust via the tempo slider.
            # The slider range is 40-300 BPM to cover slow ballads to fast bebop.

        bpm = estimated_bpm
        # Don't convert to seconds on server - return ticks and let client handle tempo
        # Client will use: seconds = ticks * 60 / (bpm * tpb)
        ticks_to_sec = 1.0  # Just return ticks as-is

        # Quantization conversion:
        # - Duration bucket (0-7) -> ticks: bucket * 120 + 60 (middle of 120-tick bucket)
        # - Velocity bucket (1-7) -> MIDI (0-127): bucket * 16 + 8
        def dur_bucket_to_ticks(bucket):
            return bucket * 120 + 60  # Use middle of bucket

        def vel_bucket_to_midi(bucket):
            return min(127, bucket * 16 + 8)

        def rhythm_bucket_to_ratio(bucket: int) -> float:
            """Convert v53 rhythm bucket (0-15) to IOI ratio.

            Buckets 0-7 represent subdivision ratios (< 1.0)
            Buckets 8-15 represent multiplication ratios (>= 1.0)
            Bucket 7 and 8 are both ~1.0 (equal rhythm).
            """
            # Inverse of compute_rhythm_bucket from grammar/v4/repair_pure_contour.py
            ratios = [
                0.125,   # 0: very short subdivision
                0.2375,  # 1: short subdivision
                0.355,   # 2
                0.5025,  # 3: ~half
                0.6475,  # 4
                0.7625,  # 5
                0.875,   # 6
                1.0,     # 7: equal
                1.0,     # 8: equal
                1.5,     # 9
                2.125,   # 10: ~double
                3.0,     # 11
                4.25,    # 12
                6.0,     # 13
                7.75,    # 14
                10.0,    # 15: very long
            ]
            return ratios[min(bucket, 15)]

        def emit_pattern_events(pattern, time_ticks):
            """Generate note events for a pattern at a given time."""
            events = []
            if not pattern:
                return events

            base_octave = 5  # octave 5 gives middle C area (MIDI 60-71)
            for i, (pc, dur_bucket) in enumerate(zip(pattern.pitch_classes, pattern.durations)):
                oct = pattern.octaves[i] if i < len(pattern.octaves) else base_octave
                vel_bucket = pattern.velocities[i] if i < len(pattern.velocities) else 5
                midi_pitch = pc + (oct + 1) * 12
                dur_ticks = dur_bucket_to_ticks(dur_bucket)
                midi_vel = vel_bucket_to_midi(vel_bucket)

                events.append({
                    'time': time_ticks * ticks_to_sec,
                    'pitch': midi_pitch,
                    'velocity': midi_vel,
                    'duration': dur_ticks * ticks_to_sec,
                    'pattern_id': pattern.id if hasattr(pattern, 'id') else 0
                })
            return events

        # Collect all occurrences for this piece, grouped by track
        # Tuple: (pattern_id, onset_time, rule_data, occ_data)
        # - occ_data contains v33 timing offsets (tau_offset, duration_offset, v_offset)
        tracks_occs = defaultdict(list)

        # Check if we have v24_rules with full occurrences (better for playback)
        if self.v24_rules:
            # Use v24/v33 rules - these have all occurrences embedded
            for rid, rule in self.v24_rules.items():
                for occ in rule.get('occurrences', []):
                    if occ.get('piece_id') == piece_id:
                        track_id = occ.get('track_id', 0)
                        onset_time = occ.get('onset_time', 0)
                        # Store rule data AND occurrence data (for v33 timing)
                        tracks_occs[track_id].append((rid, onset_time, rule, occ))
        else:
            # Fall back to graph.occurrences
            for pid, occ_list in self.graph.occurrences.items():
                for occ in occ_list:
                    if occ.get('piece_id') == piece_id:
                        track_id = occ.get('track_id', 0)
                        onset_time = occ.get('onset_time', 0)
                        tracks_occs[track_id].append((pid, onset_time, None, None))

        if not tracks_occs:
            # Fall back to checking edges for piece_id
            edge_ids = self.graph._edges_by_piece.get(piece_id, [])
            if not edge_ids:
                return {'error': f'Piece not found: {piece_id}', 'tracks': [], 'duration': 0}

        # Filter overlapping occurrences for hierarchical patterns
        # When pattern R21 contains R13, both will have occurrences at overlapping positions.
        # We must keep only the top-level (largest) pattern at each position to avoid duplicates.
        def filter_overlapping_occurrences(occ_list):
            """Filter to keep only non-overlapping occurrences.

            For each position range, keep the largest pattern (most notes).
            Uses 'position' (note index) to identify overlaps.
            """
            if not occ_list:
                return occ_list

            # Build list with position info: (rid, onset_time, rule, occ, position, length)
            items_with_pos = []
            for item in occ_list:
                rid, onset_time, rule, occ = item
                position = occ.get('position', 0) if occ else 0
                # Get pattern length from rule data
                length = 2  # default for atomic patterns
                if rule:
                    # Use canonical_pitches or pitch_classes length
                    cp = rule.get('canonical_pitches', [])
                    pc = rule.get('pitch_classes', [])
                    length = len(cp) if cp else len(pc) if pc else 2
                items_with_pos.append((rid, onset_time, rule, occ, position, length))

            # Sort by position, then by length descending (larger patterns first)
            items_with_pos.sort(key=lambda x: (x[4], -x[5]))

            # Greedy selection: keep patterns that don't overlap with already-selected ones
            selected = []
            covered_positions = set()

            for rid, onset_time, rule, occ, position, length in items_with_pos:
                # Check if any position in this pattern's range is already covered
                pattern_range = set(range(position, position + length))
                if pattern_range & covered_positions:
                    # This pattern overlaps with an already-selected one - skip it
                    continue

                # No overlap - select this pattern
                selected.append((rid, onset_time, rule, occ))
                covered_positions.update(pattern_range)

            return selected

        # Apply overlap filtering to each track
        for track_id in tracks_occs:
            tracks_occs[track_id] = filter_overlapping_occurrences(tracks_occs[track_id])

        def emit_v24_rule_events(rule_data, time_ticks, occ_data=None, rule_id=None):
            """Generate note events from v24/v33/v37 rule data.

            v33/v37 format includes per-note timing data:
            - rhythm_ratios: IOI ratios relative to first IOI
            - duration_ratios: duration ratios relative to first duration
            - velocity_ratios: velocity ratios relative to first velocity

            Occurrence data (occ_data) provides the offsets:
            - tau_offset: first IOI in ticks
            - duration_offset: first duration in ticks
            - v_offset: first velocity BUCKET (0-7) in v37 format, or MIDI (0-127) in v33

            If there are pattern edits saved for this rule_id, they will override
            the original rule_data values.
            """
            events = []

            # Check for pattern edits - these override original data
            edited_data = None
            if rule_id is not None:
                edited_data = GenomeGraphHandler.pattern_edits.get(str(rule_id))

            # Use edited data if available, otherwise use original
            if edited_data:
                pitch_classes = edited_data.get('pitch_classes', rule_data.get('pitch_classes', []))
            else:
                pitch_classes = rule_data.get('pitch_classes', [])

            # v41+ transformational approach: canonical_pitches + octave_transform
            # v50+ contour-normalized: also adds pitch_offset (0-11) for full transposition
            # v40 and earlier: octaves array (per-pattern or per-occurrence)
            if edited_data and 'canonical_pitches' in edited_data:
                canonical_pitches = edited_data.get('canonical_pitches', [])
            else:
                canonical_pitches = rule_data.get('canonical_pitches', [])

            # v50+/v53 contour-normalized: use first_pitch + pitch_intervals
            # v53 (pure contour) stores first_pitch as absolute MIDI pitch (0-127)
            pitch_intervals = rule_data.get('pitch_intervals', [])
            first_pitch = occ_data.get('first_pitch') if occ_data and 'first_pitch' in occ_data else None

            octave_transform = 0
            pitch_offset = 0  # v50+: pitch class offset (0-11) for contour-normalized patterns
            if occ_data and 'octave_transform' in occ_data:
                octave_transform = occ_data.get('octave_transform', 0)
            if occ_data and 'pitch_offset' in occ_data:
                pitch_offset = occ_data.get('pitch_offset', 0)

            # Fallback for older checkpoints: use octaves array
            if edited_data and 'octaves' in edited_data:
                octaves = edited_data.get('octaves', [])
            else:
                octaves = rule_data.get('octaves', [])

            # Check for v33/v37 per-note ratio data
            if edited_data and 'rhythm_ratios' in edited_data:
                rhythm_ratios = edited_data.get('rhythm_ratios', [])
            else:
                rhythm_ratios = rule_data.get('rhythm_ratios', [])

            if edited_data and 'duration_ratios' in edited_data:
                duration_ratios = edited_data.get('duration_ratios', [])
            else:
                duration_ratios = rule_data.get('duration_ratios', [])

            if edited_data and 'velocity_ratios' in edited_data:
                velocity_ratios = edited_data.get('velocity_ratios', [])
            else:
                velocity_ratios = rule_data.get('velocity_ratios', [])

            # v53 pure contour: check if pattern uses contour-based representation
            is_pure_contour = rule_data.get('is_pure_contour', False)
            rhythm_bucket = rule_data.get('rhythm_bucket', 7)  # Default to 1.0 ratio (bucket 7)

            # Get offsets from occurrence data (v33/v37) or use defaults
            # IMPORTANT: Use 'or' to handle None values (not just missing keys)
            tau_offset = 480  # Default IOI
            duration_offset = 480  # Default duration
            v_offset_raw = 80  # Default velocity (MIDI)

            if occ_data:
                tau_offset = occ_data.get('tau_offset') or 480  # Handle None
                duration_offset = occ_data.get('duration_offset') or 480  # Handle None
                v_offset_raw = occ_data.get('v_offset') or 80  # Handle None

            # Sanity bounds - prevent extreme values from corrupted data
            tau_offset = max(1, min(tau_offset, 48000))  # 1 tick to 100 quarter notes
            duration_offset = max(1, min(duration_offset, 48000))

            # v37 format stores v_offset as a bucket (0-7), not MIDI velocity
            # Detect by checking if value is small (bucket) vs large (MIDI)
            if v_offset_raw <= 10:
                # It's a velocity bucket - convert to MIDI velocity
                base_velocity = vel_bucket_to_midi(v_offset_raw)
            else:
                # It's already MIDI velocity (v33 format or default)
                base_velocity = v_offset_raw

            # Fall back to old bucket-based data if no ratios
            velocities = rule_data.get('velocities', [])
            durations = rule_data.get('durations', [])

            base_octave = 5  # octave 5 gives middle C area (MIDI 60-71)
            current_time = time_ticks

            for i, pc in enumerate(pitch_classes):
                # v53 pure contour: first_pitch + pitch_intervals takes priority
                # Each occurrence has its own first_pitch (absolute MIDI 0-127)
                if first_pitch is not None and pitch_intervals and i <= len(pitch_intervals):
                    # v50+/v53 contour: use first_pitch + intervals (works for first_pitch=0 too)
                    if i == 0:
                        midi_pitch = first_pitch
                    else:
                        # Compute pitch from first_pitch + sum of intervals up to this note
                        midi_pitch = first_pitch + sum(pitch_intervals[:i])
                elif canonical_pitches and i < len(canonical_pitches):
                    # v41+/v50 transformational approach: canonical_pitches + octave_transform + pitch_offset
                    # v50 contour-normalized: pitch_offset is the pitch class offset (0-11)
                    # The full transposition is octave_transform (multiple of 12) + pitch_offset (0-11)
                    midi_pitch = canonical_pitches[i] + octave_transform + pitch_offset
                else:
                    # Fallback for older checkpoints: use octaves array
                    # octave is stored as pitch // 12, so midi_pitch = pc + oct * 12
                    oct = octaves[i] if i < len(octaves) else base_octave
                    midi_pitch = pc + oct * 12

                # Compute velocity
                if velocity_ratios and i < len(velocity_ratios):
                    # v33/v37 format: use ratio * base_velocity
                    midi_vel = min(127, max(1, int(velocity_ratios[i] * base_velocity)))
                elif i < len(velocities):
                    # Fall back to bucket
                    midi_vel = vel_bucket_to_midi(velocities[i])
                else:
                    midi_vel = base_velocity

                # Compute duration
                if duration_ratios and i < len(duration_ratios):
                    # v33/v37 format: use ratio * offset
                    # Cap ratio to prevent extremely long notes from corrupted data
                    ratio = min(duration_ratios[i], 50.0)  # Max 50x the base duration
                    dur_ticks = int(ratio * duration_offset)
                elif i < len(durations):
                    # Fall back to bucket
                    dur_ticks = dur_bucket_to_ticks(durations[i])
                else:
                    dur_ticks = duration_offset
                # Sanity bounds on final duration (1 tick to 20 quarter notes)
                dur_ticks = max(1, min(dur_ticks, 9600))

                events.append({
                    'time': current_time * ticks_to_sec,
                    'pitch': midi_pitch,
                    'velocity': midi_vel,
                    'duration': dur_ticks * ticks_to_sec,
                    'pattern_id': 0
                })

                # Advance time using rhythm ratios (for next note)
                if rhythm_ratios and i < len(rhythm_ratios):
                    # v33/v37 format: use ratio * tau_offset
                    ioi = int(rhythm_ratios[i] * tau_offset)
                    current_time += ioi
                elif is_pure_contour and i < len(pitch_classes) - 1:
                    # v53 pure contour: use rhythm_bucket for uniform IOI advancement
                    # Rhythm bucket encodes the ratio between consecutive notes
                    rhythm_ratio = rhythm_bucket_to_ratio(rhythm_bucket)
                    ioi = int(tau_offset * rhythm_ratio)
                    current_time += ioi

            return events

        tracks = []
        max_time = 0

        for track_id, occ_list in tracks_occs.items():
            events = []

            for item in occ_list:
                pid, onset_time, rule_data, occ_data = item
                if rule_data:
                    # Use v24/v33 rule data with occurrence-level timing offsets
                    # Pass rule_id (pid) so edits can be looked up
                    events.extend(emit_v24_rule_events(rule_data, onset_time, occ_data, rule_id=pid))
                else:
                    # Use graph pattern
                    pattern = self.graph.patterns.get(pid)
                    events.extend(emit_pattern_events(pattern, onset_time))

            # Sort events by time
            events.sort(key=lambda e: e['time'])

            # Track max time
            if events:
                last = events[-1]
                track_end = last['time'] + last['duration']
                max_time = max(max_time, track_end)

            # Determine if drum track - prefer stored metadata from checkpoint
            is_drum = False

            # Method 1: Check occurrence data (v53+ stores is_drum per occurrence)
            # This is the most accurate method since it comes from MIDI channel 9
            for item in occ_list:
                pid, onset_time, rule_data, occ_data = item
                if occ_data and occ_data.get('is_drum', False):
                    is_drum = True
                    break  # Any occurrence marked as drum means the track is drum

            # Method 2: Fall back to track_info (for older checkpoints)
            if not is_drum and self.track_info:
                # Look up is_drum from stored track_info
                # track_id here is the global track index from the grammar (matches index in track_info)
                if track_id < len(self.track_info):
                    is_drum = self.track_info[track_id].get('is_drum', False)

            # NOTE: Pitch-based drum detection heuristic is DISABLED.
            # Without channel information (channel 9 = drums in MIDI), we cannot reliably
            # distinguish drums from bass, baritone vocals, or other low-pitched instruments.
            # Bass notes (e.g., D1 = MIDI 38) collide with GM drum pitches (snare = 38).
            # The occurrence/track_info approach above is the only reliable method.

            tracks.append({
                'track_id': track_id,
                'is_drum': is_drum,
                'events': events
            })

        return {
            'piece': piece_id,
            'tracks': tracks,
            'duration_ticks': max_time,  # Duration in ticks
            'ticks_per_beat': tpb,  # 480
            'estimated_tempo': bpm  # Server's guess at original tempo
        }

    def measure_reconstruction_accuracy(self, piece_id: str) -> dict:
        """Measure reconstruction accuracy by analyzing data completeness.

        Returns metrics about:
        - Pattern timing completeness (rhythm_ratios present)
        - Duration completeness (duration_ratios present)
        - Note count accuracy
        """
        if not self.v24_rules:
            return {'error': 'No v24_rules loaded'}

        # Collect patterns used in this piece
        patterns_used = {}
        total_notes = 0
        notes_with_timing = 0
        notes_with_duration = 0
        notes_with_velocity = 0
        patterns_complete = 0
        patterns_incomplete = 0

        for rid, rule in self.v24_rules.items():
            for occ in rule.get('occurrences', []):
                if occ.get('piece_id') == piece_id:
                    if rid not in patterns_used:
                        patterns_used[rid] = rule
                    break

        if not patterns_used:
            return {'error': f'Piece not found: {piece_id}', 'patterns_found': 0}

        issues = []

        for pid, rule in patterns_used.items():
            pcs = rule.get('pitch_classes', [])
            canonical = rule.get('canonical_pitches', [])
            rr = rule.get('rhythm_ratios', [])
            dr = rule.get('duration_ratios', [])
            vr = rule.get('velocity_ratios', [])

            n_notes = len(pcs) if pcs else len(canonical)
            total_notes += n_notes

            # Check rhythm_ratios (should have n-1 values for IOIs)
            expected_rhythms = max(0, n_notes - 1)
            if len(rr) >= expected_rhythms and expected_rhythms > 0:
                notes_with_timing += n_notes
            elif n_notes > 1:
                issues.append(f'P{pid}: {n_notes} notes but only {len(rr)} rhythm_ratios (expected {expected_rhythms})')

            # Check duration_ratios (should have n values)
            if len(dr) >= n_notes:
                notes_with_duration += n_notes
            elif n_notes > 0:
                issues.append(f'P{pid}: {n_notes} notes but only {len(dr)} duration_ratios')

            # Check velocity_ratios
            if len(vr) >= n_notes:
                notes_with_velocity += n_notes

            # Track pattern completeness
            timing_ok = len(rr) >= expected_rhythms or n_notes <= 1
            duration_ok = len(dr) >= n_notes
            if timing_ok and duration_ok:
                patterns_complete += 1
            else:
                patterns_incomplete += 1

        return {
            'piece_id': piece_id,
            'patterns_used': len(patterns_used),
            'patterns_complete': patterns_complete,
            'patterns_incomplete': patterns_incomplete,
            'pattern_completeness': f'{100 * patterns_complete / len(patterns_used):.1f}%' if patterns_used else 'N/A',
            'total_notes': total_notes,
            'notes_with_timing': notes_with_timing,
            'notes_with_duration': notes_with_duration,
            'notes_with_velocity': notes_with_velocity,
            'timing_accuracy': f'{100 * notes_with_timing / total_notes:.1f}%' if total_notes else 'N/A',
            'duration_accuracy': f'{100 * notes_with_duration / total_notes:.1f}%' if total_notes else 'N/A',
            'velocity_accuracy': f'{100 * notes_with_velocity / total_notes:.1f}%' if total_notes else 'N/A',
            'issues': issues[:20],  # First 20 issues
            'total_issues': len(issues)
        }

    def compute_transform_space(self, piece_id: str) -> dict:
        """Compute transform-space embedding where position = transform relationship.

        X-axis: Pitch transform (T0-T11, wraps at 12)
        Y-axis: Time position (accumulated τ values)

        Returns nodes with positions and edges with transform info.
        """
        from grammar.genome_graph import parse_compound_transform

        edge_ids = self.graph._edges_by_piece.get(piece_id, [])
        if not edge_ids:
            return {'error': f'Piece not found: {piece_id}', 'nodes': [], 'edges': []}

        # Group edges by track
        tracks_edges = defaultdict(list)
        for eid in edge_ids:
            edge = self.graph.edges[eid]
            tracks_edges[edge.track_id].append(edge)

        nodes = []
        edges = []
        node_positions = {}  # (track_id, pattern_id) -> (x, y)
        pattern_info = {}  # pattern_id -> pattern data

        # Scale factors for visualization
        pitch_scale = 50  # pixels per semitone
        time_scale = 0.1  # pixels per tick (τ480 = 48px)

        for track_id, track_edges in tracks_edges.items():
            # Find starting patterns (sources not in targets)
            sources = set(e.source for e in track_edges)
            targets = set(e.target for e in track_edges)
            start_patterns = sources - targets
            if not start_patterns:
                start_patterns = {min(sources)}  # Use first if circular

            # Build adjacency for BFS
            edges_from = defaultdict(list)
            for e in track_edges:
                edges_from[e.source].append(e)

            # BFS from each start, positioning by transforms
            # Handle None track_id (cross-track edges in v37 format)
            track_y_offset = (track_id or 0) * 200  # Separate tracks vertically

            for start_pid in start_patterns:
                if (track_id, start_pid) in node_positions:
                    continue

                # Place origin at (0, track_offset)
                current_x = 0
                current_y = track_y_offset
                node_positions[(track_id, start_pid)] = (current_x, current_y)

                # Get pattern info
                pattern = self.graph.patterns.get(start_pid)
                if pattern and start_pid not in pattern_info:
                    pattern_info[start_pid] = {
                        'id': start_pid,
                        'pitch_classes': pattern.pitch_classes,
                        'note_count': len(pattern.pitch_classes)
                    }

                # BFS traversal
                queue = [(start_pid, current_x, current_y)]
                visited = {start_pid}

                while queue:
                    current_pid, cx, cy = queue.pop(0)

                    for edge in edges_from.get(current_pid, []):
                        target_pid = edge.target
                        if target_pid in visited:
                            continue

                        # Parse transform to get deltas
                        transform = GenomeGraphHandler.edits.get(edge.id, edge.transform)
                        components = parse_compound_transform(transform)

                        dx = 0  # Pitch delta
                        dy = 0  # Time delta

                        for comp in components:
                            if comp.startswith('T'):
                                try:
                                    dx = int(comp[1:]) * pitch_scale
                                except ValueError:
                                    pass
                            elif comp.startswith('I'):
                                # Inversion - horizontal flip effect
                                try:
                                    dx = -int(comp[1:]) * pitch_scale
                                except ValueError:
                                    pass
                            elif comp == 'R':
                                # Retrograde - show as slight offset
                                dx += pitch_scale * 0.5
                            elif comp.startswith('τ'):
                                try:
                                    dy = int(comp[1:]) * time_scale
                                except ValueError:
                                    pass

                        # Position target relative to source
                        new_x = cx + dx
                        new_y = cy + dy
                        node_positions[(track_id, target_pid)] = (new_x, new_y)

                        # Get pattern info
                        pattern = self.graph.patterns.get(target_pid)
                        if pattern and target_pid not in pattern_info:
                            pattern_info[target_pid] = {
                                'id': target_pid,
                                'pitch_classes': pattern.pitch_classes,
                                'note_count': len(pattern.pitch_classes)
                            }

                        # Add edge
                        edges.append({
                            'id': edge.id,
                            'source': f'{track_id}_{current_pid}',
                            'target': f'{track_id}_{target_pid}',
                            'transform': transform,
                            'dx': dx / pitch_scale,  # Normalized deltas for editing
                            'dy': dy / time_scale,
                            'track_id': track_id
                        })

                        visited.add(target_pid)
                        queue.append((target_pid, new_x, new_y))

        # Build nodes list
        for (track_id, pid), (x, y) in node_positions.items():
            info = pattern_info.get(pid, {'id': pid, 'pitch_classes': [], 'note_count': 0})
            nodes.append({
                'id': f'{track_id}_{pid}',
                'pattern_id': pid,
                'track_id': track_id,
                'x': x,
                'y': y,
                'pitch_classes': info.get('pitch_classes', []),
                'note_count': info.get('note_count', 0)
            })

        return {
            'piece': piece_id,
            'nodes': nodes,
            'edges': edges,
            'pitch_scale': pitch_scale,
            'time_scale': time_scale,
            'track_count': len(tracks_edges)
        }

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == '/' or path == '/index.html':
            self.serve_editor()
        elif path == '/api/stats':
            stats = self.graph.stats()
            # Merge in v43 checkpoint stats
            if GenomeGraphHandler.checkpoint_stats:
                stats.update(GenomeGraphHandler.checkpoint_stats)
            # Add v43 feature importance summary
            if GenomeGraphHandler.feature_importance:
                stats['useful_features'] = GenomeGraphHandler.feature_importance.get('useful_features', [])
                stats['keys_discovered'] = 'pitch_offset_relative' in stats.get('useful_features', [])
            # Add v43 multi-factor summary
            if GenomeGraphHandler.multi_factor:
                def mf_count(key):
                    v = GenomeGraphHandler.multi_factor.get(key, 0)
                    return len(v) if isinstance(v, list) else (v if isinstance(v, int) else 0)
                stats['n_rhythm_transforms'] = mf_count('rhythm_transforms')
                stats['n_velocity_transforms'] = mf_count('velocity_transforms')
                stats['n_duration_transforms'] = mf_count('duration_transforms')
            # Add track derives count
            if GenomeGraphHandler.track_derives:
                stats['n_track_derives'] = len(GenomeGraphHandler.track_derives)
            self.send_json(stats)
        elif path == '/api/cytoscape':
            piece = query.get('piece', [None])[0]
            data = self.graph.to_cytoscape(piece)
            self.enrich_cytoscape_with_tracks(data, piece)
            self.send_json(data)
        elif path.startswith('/api/cytoscape/'):
            piece = unquote(path.split('/')[-1])
            data = self.graph.to_cytoscape(piece)
            self.enrich_cytoscape_with_tracks(data, piece)
            self.send_json(data)
        elif path.startswith('/api/pattern/'):
            try:
                pid = int(path.split('/')[-1])
                pattern = self.graph.get_pattern(pid)
                if pattern:
                    result = pattern.to_dict()
                    # Override with v24_rules data if available (includes swap edits)
                    if GenomeGraphHandler.v24_rules and str(pid) in GenomeGraphHandler.v24_rules:
                        rule = GenomeGraphHandler.v24_rules[str(pid)]
                        # Copy fields from rule that may have been swapped
                        # v52+: include pitch_intervals for contour-normalized patterns
                        for field in ['pitch_classes', 'octaves', 'durations', 'intervals',
                                      'canonical_pitches', 'pitch_intervals',
                                      'rhythm_ratios', 'duration_ratios', 'velocity_ratios']:
                            if field in rule:
                                result[field] = rule[field]
                    self.send_json(result)
                else:
                    self.send_json({'error': 'Pattern not found'}, 404)
            except ValueError:
                self.send_json({'error': 'Invalid pattern ID'}, 400)
        elif path.startswith('/api/edges/'):
            try:
                pid = int(path.split('/')[-1])
                edges = self.graph.get_edges_from(pid)
                self.send_json([e.to_dict() for e in edges])
            except ValueError:
                self.send_json({'error': 'Invalid pattern ID'}, 400)
        elif path == '/api/pieces':
            pieces = list(self.graph._edges_by_piece.keys())
            self.send_json({'pieces': pieces})
        elif path.startswith('/api/playback/'):
            # Reconstruct MIDI events for playback
            piece = path.split('/')[-1]
            piece = unquote(piece)  # URL decode
            events = self.reconstruct_piece_events(piece)
            self.send_json(events)
        elif path.startswith('/api/accuracy/'):
            # Measure reconstruction accuracy for a piece
            piece = unquote(path.split('/')[-1])
            accuracy = self.measure_reconstruction_accuracy(piece)
            self.send_json(accuracy)
        elif path == '/api/transforms':
            self.send_json({'transforms': self.graph.transform_vocab})
        # v43 API endpoints
        elif path == '/api/multi_factor':
            if GenomeGraphHandler.multi_factor:
                self.send_json(GenomeGraphHandler.multi_factor)
            else:
                self.send_json({'error': 'No multi-factor transforms loaded (v43+ required)'}, 404)
        elif path == '/api/track_derives':
            if GenomeGraphHandler.track_derives:
                # Return summary + paginated list
                limit = int(query.get('limit', [100])[0])
                offset = int(query.get('offset', [0])[0])
                derives = GenomeGraphHandler.track_derives[offset:offset+limit]
                self.send_json({
                    'total': len(GenomeGraphHandler.track_derives),
                    'offset': offset,
                    'limit': limit,
                    'derives': derives
                })
            else:
                self.send_json({'error': 'No track derives loaded (v43+ required)'}, 404)
        elif path == '/api/feature_importance':
            if GenomeGraphHandler.feature_importance:
                self.send_json(GenomeGraphHandler.feature_importance)
            else:
                self.send_json({'error': 'No feature importance loaded (v43+ required)'}, 404)
        elif path == '/transform-editor' or path == '/transform-editor/':
            self.serve_transform_editor()
        elif path.startswith('/api/transform-space/'):
            # Compute transform-space embedding for a piece
            piece = unquote(path.split('/')[-1])
            data = self.compute_transform_space(piece)
            self.send_json(data)
        elif path == '/api/meta_patterns':
            self.send_json({'meta_patterns': self.graph.meta_patterns})
        # DAG endpoints
        elif path == '/dag' or path == '/dag/':
            self.serve_dag_editor()
        elif path == '/api/dag/stats':
            if self.dag is None:
                self.send_json({'error': 'No DAG loaded'}, 404)
            else:
                self.send_json(self.dag.get_stats())
        elif path == '/api/dag/cytoscape':
            if self.dag is None:
                self.send_json({'error': 'No DAG loaded'}, 404)
            else:
                max_nodes = int(query.get('max_nodes', [5000])[0])
                self.send_json(self.dag.to_cytoscape(max_nodes=max_nodes))
        elif path.startswith('/api/dag/node/'):
            if self.dag is None:
                self.send_json({'error': 'No DAG loaded'}, 404)
            else:
                try:
                    nid = int(path.split('/')[-1])
                    node = self.dag.nodes.get(nid)
                    if node:
                        node_data = {
                            'id': node.id,
                            'type': node.node_type.value,
                            'name': node.name,
                            'children': node.children,
                            'transform': node.transform,
                        }
                        if node.pitch_classes:
                            node_data['pitch_classes'] = node.pitch_classes
                            node_data['note_count'] = len(node.pitch_classes)
                        self.send_json(node_data)
                    else:
                        self.send_json({'error': 'Node not found'}, 404)
                except ValueError:
                    self.send_json({'error': 'Invalid node ID'}, 400)
        elif path.startswith('/api/dag/expand/'):
            if self.dag is None:
                self.send_json({'error': 'No DAG loaded'}, 404)
            else:
                try:
                    nid = int(path.split('/')[-1])
                    events = evaluate(self.dag, nid)
                    self.send_json({
                        'node_id': nid,
                        'event_count': len(events),
                        'events': events[:100],  # First 100 events
                    })
                except (ValueError, KeyError) as e:
                    self.send_json({'error': str(e)}, 400)
        # Piece comparison endpoints
        elif path == '/compare' or path == '/compare/':
            self.serve_compare()
        elif path == '/api/compare':
            piece1 = query.get('piece1', [None])[0]
            piece2 = query.get('piece2', [None])[0]
            if not piece1 or not piece2:
                self.send_json({'error': 'Must provide piece1 and piece2 query parameters'}, 400)
            else:
                piece1 = unquote(piece1)
                piece2 = unquote(piece2)
                result = self.compare_pieces(piece1, piece2)
                self.send_json(result)
        else:
            self.send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len).decode() if content_len > 0 else '{}'
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_json({'error': 'Invalid JSON'}, 400)
            return

        if path.startswith('/api/factor/'):
            try:
                edge_id = int(path.split('/')[-1])
                new_edges = self.graph.factor_edge(edge_id)
                self.send_json({'new_edge_ids': new_edges})
            except (ValueError, KeyError) as e:
                self.send_json({'error': str(e)}, 400)

        elif path == '/api/entangle':
            edge_ids = data.get('edge_ids', [])
            try:
                new_edge = self.graph.entangle_path(edge_ids)
                self.send_json({'new_edge_id': new_edge})
            except (ValueError, KeyError) as e:
                self.send_json({'error': str(e)}, 400)

        elif path == '/api/clone':
            root_id = data.get('root_id')
            transform = data.get('transform')
            if root_id is None or transform is None:
                self.send_json({'error': 'Missing root_id or transform'}, 400)
                return
            try:
                new_root = self.graph.clone_subgraph(int(root_id), transform)
                self.send_json({'new_root_id': new_root})
            except (ValueError, KeyError) as e:
                self.send_json({'error': str(e)}, 400)

        elif path == '/api/apply':
            pattern_id = data.get('pattern_id')
            transform = data.get('transform')
            if pattern_id is None or transform is None:
                self.send_json({'error': 'Missing pattern_id or transform'}, 400)
                return
            try:
                new_id = self.graph.apply_transform(int(pattern_id), transform)
                self.send_json({'new_pattern_id': new_id})
            except (ValueError, KeyError) as e:
                self.send_json({'error': str(e)}, 400)

        elif path == '/api/edit':
            # Save an edit to an edge's transform
            edge_id = data.get('edge_id')
            transform = data.get('transform')
            if edge_id is None or transform is None:
                self.send_json({'error': 'Missing edge_id or transform'}, 400)
                return
            GenomeGraphHandler.edits[int(edge_id)] = transform
            self.send_json({'success': True, 'edge_id': edge_id, 'transform': transform})

        elif path == '/api/edits':
            # Get all current edits
            self.send_json({'edits': GenomeGraphHandler.edits})

        elif path == '/api/clear_edits':
            # Clear all edits
            GenomeGraphHandler.edits = {}
            GenomeGraphHandler.pattern_edits = {}
            self.send_json({'success': True})

        elif path == '/api/pattern_edit':
            # Save pattern edits (rhythm/contour pattern modifications)
            pattern_id = data.get('pattern_id')
            pattern_data = data.get('pattern')  # Array of edited values
            if pattern_id is None or pattern_data is None:
                self.send_json({'error': 'Missing pattern_id or pattern'}, 400)
                return
            # Store the edit keyed by pattern_id
            GenomeGraphHandler.pattern_edits[str(pattern_id)] = pattern_data
            self.send_json({'success': True, 'pattern_id': pattern_id, 'edit_count': len(GenomeGraphHandler.pattern_edits)})

        elif path == '/api/pattern_edits':
            # Get all pattern edits
            self.send_json({'pattern_edits': GenomeGraphHandler.pattern_edits})

        elif path == '/api/save':
            # Persist all edits to checkpoint file
            try:
                saved = self.save_edits_to_checkpoint()
                self.send_json({'success': True, 'saved': saved})
            except Exception as e:
                self.send_json({'error': str(e)}, 500)

        elif path == '/api/swap_pattern':
            # Swap one pattern for another - copy target's content to source
            source_id = data.get('source_pattern')
            target_id = data.get('target_pattern')
            if source_id is None or target_id is None:
                self.send_json({'error': 'Missing source_pattern or target_pattern'}, 400)
                return

            # Normalize IDs (remove P prefix if present)
            source_key = source_id[1:] if source_id.startswith('P') else source_id
            target_key = target_id[1:] if target_id.startswith('P') else target_id

            # Apply swap immediately to v24_rules for instant playback
            if GenomeGraphHandler.v24_rules and source_key in GenomeGraphHandler.v24_rules and target_key in GenomeGraphHandler.v24_rules:
                source_rule = GenomeGraphHandler.v24_rules[source_key]
                target_rule = GenomeGraphHandler.v24_rules[target_key]
                # Copy pitch content from target to source
                # v52+: include pitch_intervals for contour-normalized patterns
                for field in ['pitch_classes', 'octaves', 'durations', 'intervals', 'canonical_pitches',
                              'pitch_intervals', 'rhythm_ratios', 'duration_ratios', 'velocity_ratios']:
                    if field in target_rule:
                        source_rule[field] = target_rule[field].copy() if isinstance(target_rule[field], list) else target_rule[field]
                print(f"Applied swap immediately: {source_key} now has content of {target_key}")

            # Store the swap as a pattern edit (for tracking what was changed)
            GenomeGraphHandler.pattern_edits[f'swap_{source_id}'] = {
                'type': 'swap',
                'source': source_id,
                'target': target_id
            }
            self.send_json({'success': True, 'source': source_id, 'target': target_id})

        else:
            self.send_json({'error': 'Not found'}, 404)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def save_edits_to_checkpoint(self) -> dict:
        """Persist edge and pattern edits to the v24_rules and save to checkpoint.

        Returns dict with counts of saved edits.
        """
        if GenomeGraphHandler.v24_rules is None:
            raise ValueError("No v24_rules loaded - cannot save edits")

        saved_edges = 0
        saved_patterns = 0

        # Apply edge edits (transform changes) to v24_rules
        for edge_id, new_transform in GenomeGraphHandler.edits.items():
            # Find the edge in the graph
            edge = self.graph.edges.get(edge_id)
            if edge:
                # Update the edge's transform in memory
                edge.transform = new_transform
                saved_edges += 1

        # Apply pattern edits to v24_rules
        # Note: Swap edits are applied immediately when queued (for instant playback),
        # so here we just count them - no need to re-apply
        saved_swaps = 0
        for pattern_id, pattern_data in GenomeGraphHandler.pattern_edits.items():
            # Count swap edits (already applied in /api/swap_pattern handler)
            if pattern_data.get('type') == 'swap':
                saved_swaps += 1
                continue

            # Handle direct pattern edits
            if pattern_id in GenomeGraphHandler.v24_rules:
                rule = GenomeGraphHandler.v24_rules[pattern_id]
                # Merge edited fields into the rule
                for key, value in pattern_data.items():
                    rule[key] = value
                saved_patterns += 1

        # Save updated rules to checkpoint
        # Find the checkpoint path from graph
        checkpoint_path = getattr(GenomeGraphHandler.graph, '_checkpoint_path', None)
        if checkpoint_path:
            import shutil
            # Create backup
            backup_path = checkpoint_path + '.backup'
            if os.path.exists(checkpoint_path):
                shutil.copy2(checkpoint_path, backup_path)

            # Load existing checkpoint data
            existing_data = dict(np.load(checkpoint_path, allow_pickle=True))

            # Update patterns_json with edited rules
            existing_data['patterns_json'] = json.dumps(GenomeGraphHandler.v24_rules)

            # Save back
            np.savez_compressed(checkpoint_path, **existing_data)
            print(f"Saved {saved_edges} edge edits, {saved_patterns} pattern edits, {saved_swaps} swaps to {checkpoint_path}")

        return {'edges': saved_edges, 'patterns': saved_patterns, 'swaps': saved_swaps}

    def enrich_cytoscape_with_tracks(self, data: dict, piece: str = None):
        """Add track_id and trackColor to cytoscape node data."""
        TRACK_COLORS = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7',
                        '#dfe6e9', '#fd79a8', '#a29bfe', '#00b894', '#e17055']

        # Build pattern -> track mapping from v24_rules if available
        pattern_tracks = {}
        if GenomeGraphHandler.v24_rules:
            for rule_id, rule in GenomeGraphHandler.v24_rules.items():
                # Use the first occurrence's track_id
                occs = rule.get('occurrences', [])
                if occs and len(occs) > 0:
                    first_occ = occs[0]
                    if isinstance(first_occ, dict):
                        track_id = first_occ.get('track_id', 0)
                    elif isinstance(first_occ, (list, tuple)) and len(first_occ) > 3:
                        track_id = first_occ[3] if len(first_occ) > 3 else 0
                    else:
                        track_id = 0
                    pattern_tracks[rule_id] = track_id

        # Also try to infer from edges
        for edge in self.graph.edges.values():
            if hasattr(edge, 'track_id'):
                pattern_tracks[str(edge.source)] = edge.track_id
                pattern_tracks[str(edge.target)] = edge.track_id

        # Update nodes with track info
        nodes = data.get('elements', {})
        if isinstance(nodes, dict):
            node_list = nodes.get('nodes', [])
        else:
            node_list = [n for n in nodes if n.get('group') == 'nodes']

        for node in node_list:
            node_data = node.get('data', {})
            # Extract pattern ID from node ID (e.g., "P123" -> "123")
            node_id = node_data.get('id', '')
            pid = node_id[1:] if node_id.startswith('P') else node_id

            track_id = pattern_tracks.get(pid, pattern_tracks.get(int(pid) if pid.isdigit() else 0, None))
            if track_id is None:
                track_id = 0
            node_data['track_id'] = track_id
            node_data['trackColor'] = TRACK_COLORS[track_id % len(TRACK_COLORS)]

        # Collect unique tracks for legend
        unique_tracks = sorted(set(n.get('data', {}).get('track_id', 0) or 0 for n in node_list))
        data['tracks'] = [{'track_id': t, 'color': TRACK_COLORS[t % len(TRACK_COLORS)]} for t in unique_tracks]

    def get_piece_patterns(self, piece_id: str) -> set:
        """Get set of pattern IDs used in a piece."""
        edge_ids = self.graph._edges_by_piece.get(piece_id, [])
        patterns = set()
        for eid in edge_ids:
            edge = self.graph.edges.get(eid)
            if edge:
                patterns.add(edge.source)
                patterns.add(edge.target)
        return patterns

    def get_piece_transforms(self, piece_id: str) -> dict:
        """Get frequency distribution of transforms used in a piece."""
        edge_ids = self.graph._edges_by_piece.get(piece_id, [])
        transforms = defaultdict(int)
        for eid in edge_ids:
            edge = self.graph.edges.get(eid)
            if edge and edge.transform:
                transforms[edge.transform] += 1
        return dict(transforms)

    def get_piece_intervals(self, piece_id: str) -> list:
        """Get all pitch intervals used in a piece's patterns."""
        patterns = self.get_piece_patterns(piece_id)
        intervals = []
        for pid in patterns:
            pattern = self.graph.patterns.get(pid)
            if pattern:
                # Try pitch_intervals first (v53+), then intervals
                pi = getattr(pattern, 'pitch_intervals', None)
                if pi:
                    intervals.extend(pi)
                elif hasattr(pattern, 'intervals'):
                    intervals.extend(pattern.intervals)
        return intervals

    def compare_pieces(self, piece1: str, piece2: str) -> dict:
        """Compare two pieces using multiple similarity metrics."""
        # Get pattern sets
        patterns1 = self.get_piece_patterns(piece1)
        patterns2 = self.get_piece_patterns(piece2)

        # Get transform distributions
        transforms1 = self.get_piece_transforms(piece1)
        transforms2 = self.get_piece_transforms(piece2)

        # Get interval distributions
        intervals1 = self.get_piece_intervals(piece1)
        intervals2 = self.get_piece_intervals(piece2)

        # Compute similarity metrics
        metrics = {}

        # 1. Pattern Jaccard similarity (shared patterns / union of patterns)
        shared_patterns = patterns1 & patterns2
        union_patterns = patterns1 | patterns2
        metrics['pattern_jaccard'] = len(shared_patterns) / len(union_patterns) if union_patterns else 0

        # 2. Pattern overlap coefficient (shared / min)
        metrics['pattern_overlap'] = len(shared_patterns) / min(len(patterns1), len(patterns2)) if min(len(patterns1), len(patterns2)) > 0 else 0

        # 3. Transform Jaccard (shared transform types / union)
        shared_transforms = set(transforms1.keys()) & set(transforms2.keys())
        union_transforms = set(transforms1.keys()) | set(transforms2.keys())
        metrics['transform_jaccard'] = len(shared_transforms) / len(union_transforms) if union_transforms else 0

        # 4. Transform distribution cosine similarity
        all_transforms = union_transforms
        if all_transforms:
            vec1 = [transforms1.get(t, 0) for t in all_transforms]
            vec2 = [transforms2.get(t, 0) for t in all_transforms]
            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            metrics['transform_cosine'] = dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0
        else:
            metrics['transform_cosine'] = 0

        # 5. Interval distribution similarity
        from collections import Counter
        int_counts1 = Counter(intervals1)
        int_counts2 = Counter(intervals2)
        all_intervals = set(int_counts1.keys()) | set(int_counts2.keys())
        if all_intervals:
            vec1 = [int_counts1.get(i, 0) for i in all_intervals]
            vec2 = [int_counts2.get(i, 0) for i in all_intervals]
            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            metrics['interval_cosine'] = dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0
        else:
            metrics['interval_cosine'] = 0

        # 6. Overall similarity (weighted average)
        metrics['overall'] = (
            0.3 * metrics['pattern_jaccard'] +
            0.2 * metrics['pattern_overlap'] +
            0.2 * metrics['transform_cosine'] +
            0.3 * metrics['interval_cosine']
        )

        return {
            'piece1': piece1,
            'piece2': piece2,
            'metrics': metrics,
            'details': {
                'piece1_patterns': len(patterns1),
                'piece2_patterns': len(patterns2),
                'shared_patterns': len(shared_patterns),
                'shared_pattern_ids': list(shared_patterns)[:20],  # First 20
                'piece1_transforms': len(transforms1),
                'piece2_transforms': len(transforms2),
                'shared_transforms': list(shared_transforms),
                'piece1_intervals': len(intervals1),
                'piece2_intervals': len(intervals2),
            }
        }

    def serve_compare(self):
        """Serve the piece comparison HTML page."""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Piece Comparison</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #e94560; margin-bottom: 20px; }
        h2 { color: #0f4c75; margin: 20px 0 10px; font-size: 16px; }
        .selectors {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        .selector-group {
            flex: 1;
            min-width: 300px;
        }
        .selector-group label {
            display: block;
            margin-bottom: 8px;
            color: #888;
        }
        select {
            width: 100%;
            padding: 10px;
            background: #0f3460;
            border: 1px solid #16213e;
            color: #e0e0e0;
            border-radius: 4px;
            font-size: 14px;
        }
        button {
            background: #e94560;
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 10px;
        }
        button:hover { background: #ff6b6b; }
        button:disabled { background: #555; cursor: not-allowed; }
        .results {
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .metric-card {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
        }
        .metric-name {
            color: #888;
            font-size: 12px;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .metric-bar {
            height: 6px;
            background: #1a1a2e;
            border-radius: 3px;
            margin-top: 8px;
            overflow: hidden;
        }
        .metric-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #e94560, #4CAF50);
            border-radius: 3px;
            transition: width 0.3s;
        }
        .overall-score {
            text-align: center;
            padding: 30px;
            background: #0f3460;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .overall-score .score {
            font-size: 64px;
            font-weight: bold;
            background: linear-gradient(135deg, #e94560, #4CAF50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .overall-score .label {
            color: #888;
            margin-top: 5px;
        }
        .details {
            margin-top: 20px;
            padding: 15px;
            background: #0f3460;
            border-radius: 8px;
        }
        .detail-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #16213e;
        }
        .detail-row:last-child { border-bottom: none; }
        .detail-label { color: #888; }
        .shared-patterns {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 10px;
        }
        .pattern-chip {
            background: #e94560;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
        }
        .loading { text-align: center; padding: 40px; color: #888; }
        .back-link {
            display: inline-block;
            color: #e94560;
            text-decoration: none;
            margin-bottom: 20px;
        }
        .back-link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-link">&larr; Back to Genome Editor</a>
        <h1>Piece Comparison</h1>

        <div class="selectors">
            <div class="selector-group">
                <label>Piece 1</label>
                <select id="piece1">
                    <option value="">Loading pieces...</option>
                </select>
            </div>
            <div class="selector-group">
                <label>Piece 2</label>
                <select id="piece2">
                    <option value="">Loading pieces...</option>
                </select>
            </div>
        </div>
        <button id="compare-btn" onclick="compare()" disabled>Compare Pieces</button>

        <div id="results"></div>
    </div>

    <script>
        let pieces = [];

        async function loadPieces() {
            const response = await fetch('/api/pieces');
            const data = await response.json();
            pieces = data.pieces || [];

            const select1 = document.getElementById('piece1');
            const select2 = document.getElementById('piece2');

            const options = pieces.map(p => `<option value="${p}">${p}</option>`).join('');
            select1.innerHTML = '<option value="">Select a piece...</option>' + options;
            select2.innerHTML = '<option value="">Select a piece...</option>' + options;

            // Enable compare button when both selected
            select1.onchange = select2.onchange = () => {
                document.getElementById('compare-btn').disabled =
                    !select1.value || !select2.value || select1.value === select2.value;
            };
        }

        async function compare() {
            const piece1 = document.getElementById('piece1').value;
            const piece2 = document.getElementById('piece2').value;

            if (!piece1 || !piece2) return;

            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<div class="loading">Comparing pieces...</div>';

            try {
                const response = await fetch(`/api/compare?piece1=${encodeURIComponent(piece1)}&piece2=${encodeURIComponent(piece2)}`);
                const data = await response.json();

                if (data.error) {
                    resultsDiv.innerHTML = `<div class="results"><p style="color: #e94560;">Error: ${data.error}</p></div>`;
                    return;
                }

                const m = data.metrics;
                const d = data.details;

                resultsDiv.innerHTML = `
                    <div class="results">
                        <div class="overall-score">
                            <div class="score">${(m.overall * 100).toFixed(1)}%</div>
                            <div class="label">Overall Similarity</div>
                        </div>

                        <h2>Similarity Metrics</h2>
                        <div class="metric-grid">
                            <div class="metric-card">
                                <div class="metric-name">Pattern Jaccard</div>
                                <div class="metric-value">${(m.pattern_jaccard * 100).toFixed(1)}%</div>
                                <div class="metric-bar"><div class="metric-bar-fill" style="width: ${m.pattern_jaccard * 100}%"></div></div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Pattern Overlap</div>
                                <div class="metric-value">${(m.pattern_overlap * 100).toFixed(1)}%</div>
                                <div class="metric-bar"><div class="metric-bar-fill" style="width: ${m.pattern_overlap * 100}%"></div></div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Transform Jaccard</div>
                                <div class="metric-value">${(m.transform_jaccard * 100).toFixed(1)}%</div>
                                <div class="metric-bar"><div class="metric-bar-fill" style="width: ${m.transform_jaccard * 100}%"></div></div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Transform Cosine</div>
                                <div class="metric-value">${(m.transform_cosine * 100).toFixed(1)}%</div>
                                <div class="metric-bar"><div class="metric-bar-fill" style="width: ${m.transform_cosine * 100}%"></div></div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Interval Cosine</div>
                                <div class="metric-value">${(m.interval_cosine * 100).toFixed(1)}%</div>
                                <div class="metric-bar"><div class="metric-bar-fill" style="width: ${m.interval_cosine * 100}%"></div></div>
                            </div>
                        </div>

                        <div class="details">
                            <h2>Details</h2>
                            <div class="detail-row">
                                <span class="detail-label">${data.piece1} patterns</span>
                                <span>${d.piece1_patterns}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">${data.piece2} patterns</span>
                                <span>${d.piece2_patterns}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Shared patterns</span>
                                <span>${d.shared_patterns}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">${data.piece1} transforms</span>
                                <span>${d.piece1_transforms}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">${data.piece2} transforms</span>
                                <span>${d.piece2_transforms}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Shared transforms</span>
                                <span>${d.shared_transforms.length}</span>
                            </div>
                        </div>

                        ${d.shared_pattern_ids.length > 0 ? `
                        <div class="details">
                            <h2>Shared Pattern IDs (first 20)</h2>
                            <div class="shared-patterns">
                                ${d.shared_pattern_ids.map(p => `<span class="pattern-chip">P${p}</span>`).join('')}
                            </div>
                        </div>
                        ` : ''}
                    </div>
                `;
            } catch (error) {
                resultsDiv.innerHTML = `<div class="results"><p style="color: #e94560;">Error: ${error.message}</p></div>`;
            }
        }

        // Load pieces on page load
        loadPieces();
    </script>
</body>
</html>'''
        self.send_html(html)

    def serve_editor(self):
        """Serve the genome editor HTML."""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Genome Graph Editor</title>
    <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
        }
        #sidebar {
            width: 320px;
            background: #16213e;
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid #0f3460;
        }
        #graph-container {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        #toolbar {
            background: #0f3460;
            padding: 10px 20px;
            display: flex;
            gap: 15px;
            align-items: center;
        }
        #cy { flex: 1; }
        h1 { font-size: 18px; margin-bottom: 20px; color: #e94560; }
        h2 { font-size: 14px; margin: 15px 0 10px; color: #0f4c75; }
        .stat {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #0f3460;
        }
        .stat-label { color: #888; }
        .stat-value { font-weight: bold; }
        select, button, input {
            background: #0f3460;
            border: 1px solid #1a1a2e;
            color: #e0e0e0;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
        }
        select:hover, button:hover { background: #16213e; }
        button { background: #e94560; border-color: #e94560; }
        button:hover { background: #ff6b6b; }
        button.secondary { background: #0f4c75; border-color: #0f4c75; }
        button.play-btn { background: #4CAF50; border-color: #4CAF50; }
        button.play-btn:hover { background: #66BB6A; }
        button.stop-btn { background: #f44336; border-color: #f44336; }
        button.stop-btn:hover { background: #ef5350; }
        .filter-section { margin-top: 20px; }
        .filter-group { margin-bottom: 15px; }
        .filter-group label { display: block; margin-bottom: 5px; color: #888; font-size: 12px; }
        #pattern-detail {
            background: #0f3460;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            display: none;
        }
        #pattern-detail.active { display: block; }
        .pitch-classes {
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
            margin: 10px 0;
        }
        .pitch {
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .pitch-0 { background: #e94560; }
        .pitch-1 { background: #ff6b6b; }
        .pitch-2 { background: #f9a825; }
        .pitch-3 { background: #fdd835; }
        .pitch-4 { background: #8bc34a; }
        .pitch-5 { background: #4caf50; }
        .pitch-6 { background: #00bcd4; }
        .pitch-7 { background: #2196f3; }
        .pitch-8 { background: #3f51b5; }
        .pitch-9 { background: #673ab7; }
        .pitch-10 { background: #9c27b0; }
        .pitch-11 { background: #e91e63; }
        .edge-list { margin-top: 15px; }
        .edge-item {
            padding: 8px;
            background: #16213e;
            border-radius: 4px;
            margin-bottom: 5px;
            font-size: 12px;
            cursor: pointer;
        }
        .edge-item:hover { background: #1a1a2e; }
        .edge-transform { color: #e94560; font-weight: bold; }
        .legend {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 12px;
        }
        .legend-color {
            width: 20px;
            height: 3px;
            border-radius: 2px;
        }
        .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 15px; }
        #playback-section {
            background: #0f3460;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
        }
        #playback-controls {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            align-items: center;
        }
        #playback-status {
            font-size: 12px;
            color: #888;
            margin-top: 10px;
        }
        #playback-time {
            font-family: monospace;
            font-size: 14px;
            color: #4CAF50;
        }
        .volume-control {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 10px;
        }
        .volume-control input[type="range"] {
            width: 100px;
        }
        /* Expandable sections */
        .expandable {
            background: #0f3460;
            border-radius: 8px;
            margin-top: 10px;
            overflow: hidden;
        }
        .expandable-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 15px;
            cursor: pointer;
            user-select: none;
        }
        .expandable-header:hover { background: rgba(255,255,255,0.05); }
        .expandable-header h3 {
            font-size: 13px;
            font-weight: 500;
            color: #0f4c75;
        }
        .expand-icon {
            font-size: 12px;
            transition: transform 0.2s;
        }
        .expandable.collapsed .expand-icon { transform: rotate(-90deg); }
        .expandable-content {
            padding: 0 15px 15px;
            max-height: 300px;
            overflow-y: auto;
        }
        .expandable.collapsed .expandable-content { display: none; }
        /* Track mixer */
        .track-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 0;
            border-bottom: 1px solid #1a1a2e;
        }
        .track-item:last-child { border-bottom: none; }
        .track-color-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .track-name {
            flex: 1;
            font-size: 11px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .track-slider {
            width: 60px;
        }
        .track-mute {
            font-size: 10px;
            padding: 2px 6px;
            background: #1a1a2e;
            border: none;
            border-radius: 3px;
            color: #888;
            cursor: pointer;
        }
        .track-mute.muted { background: #e94560; color: white; }
        /* Pattern swap */
        .pattern-swap {
            margin-top: 10px;
        }
        .pattern-swap select {
            width: 100%;
            font-size: 12px;
        }
        /* Track legend */
        .track-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-left: 15px;
        }
        .track-legend-item {
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>Genome Graph Editor</h1>

        <div class="stats">
            <div class="stat">
                <span class="stat-label">Patterns</span>
                <span class="stat-value" id="stat-patterns">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Edges</span>
                <span class="stat-value" id="stat-edges">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Pieces</span>
                <span class="stat-value" id="stat-pieces">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Avg Degree</span>
                <span class="stat-value" id="stat-degree">-</span>
            </div>
        </div>

        <div class="filter-section">
            <h2>Filters</h2>
            <div class="filter-group">
                <label>Piece</label>
                <select id="piece-filter" style="width: 100%">
                    <option value="">All pieces</option>
                </select>
            </div>
            <div class="filter-group">
                <label>Transform Type</label>
                <select id="transform-filter" style="width: 100%">
                    <option value="">All transforms</option>
                    <option value="T">Transposition (T)</option>
                    <option value="I">Inversion (I)</option>
                    <option value="R">Retrograde (R)</option>
                    <option value="τ">Temporal (τ)</option>
                </select>
            </div>
        </div>

        <div id="pattern-detail">
            <h2>Pattern <span id="detail-id"></span></h2>
            <div class="stat">
                <span class="stat-label">Track</span>
                <span class="stat-value" id="detail-track">-</span>
            </div>
            <div class="pitch-classes" id="detail-pitches"></div>
            <div class="stat">
                <span class="stat-label">Length</span>
                <span class="stat-value" id="detail-length">-</span>
            </div>
            <div class="pattern-swap">
                <label style="font-size:11px;color:#888;margin-bottom:5px;display:block">Swap with Pattern:</label>
                <select id="pattern-swap-select" onchange="swapPattern(this.value)">
                    <option value="">-- Select pattern to swap --</option>
                </select>
            </div>
            <div class="edge-list">
                <h2>Outgoing Edges</h2>
                <div id="detail-edges"></div>
            </div>
            <div class="actions">
                <button onclick="applyTransform()">Apply Transform</button>
                <button class="secondary" onclick="cloneSubgraph()">Clone Subgraph</button>
            </div>
        </div>

        <div id="edge-detail" style="display:none; margin-top: 20px;">
            <h2>Edge <span id="edge-id"></span></h2>
            <div class="stat">
                <span class="stat-label">Transform</span>
                <span class="stat-value edge-transform" id="edge-transform"></span>
            </div>
            <div class="stat">
                <span class="stat-label">Type</span>
                <span class="stat-value" id="edge-type">-</span>
            </div>
            <div class="actions">
                <button onclick="factorEdge()">Factor Edge</button>
                <button class="secondary" onclick="editEdgeTransform()">Edit Transform</button>
            </div>
        </div>

        <div id="save-section" style="margin-top: 15px;">
            <button style="background:#4CAF50;border-color:#4CAF50;width:100%" onclick="saveEdits()">Save All Edits</button>
            <div id="save-status" style="font-size:11px;color:#888;margin-top:5px;"></div>
        </div>

        <div id="playback-section">
            <h2>Playback</h2>
            <div id="playback-controls">
                <button class="play-btn" id="play-btn" onclick="playPiece()">Play</button>
                <button class="stop-btn" id="stop-btn" onclick="stopPlayback()" disabled>Stop</button>
                <span id="playback-time">0:00 / 0:00</span>
            </div>
            <div class="volume-control">
                <label>Volume:</label>
                <input type="range" id="volume-slider" min="0" max="100" value="50">
                <span id="volume-value">50%</span>
            </div>
            <div class="volume-control">
                <label>Tempo:</label>
                <input type="range" id="tempo-slider" min="40" max="300" value="120">
                <span id="tempo-value">120 BPM</span>
            </div>
            <div id="playback-status">Select a piece to enable playback</div>
        </div>

        <!-- Expandable Track Mixer -->
        <div class="expandable" id="track-mixer">
            <div class="expandable-header" onclick="toggleExpand('track-mixer')">
                <h3>Track Mixer</h3>
                <span class="expand-icon">▼</span>
            </div>
            <div class="expandable-content">
                <div id="track-mixer-content">
                    <div style="color:#666;font-size:11px;">Load a piece to see tracks</div>
                </div>
            </div>
        </div>
    </div>

    <div id="graph-container">
        <div id="toolbar">
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #2196F3"></div>
                    <span>Transposition</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #F44336"></div>
                    <span>Inversion</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #9C27B0"></div>
                    <span>Retrograde</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4CAF50"></div>
                    <span>Temporal</span>
                </div>
            </div>
            <div class="track-legend" id="track-legend"></div>
            <div style="flex:1"></div>
            <button class="secondary" onclick="resetView()">Reset View</button>
            <button class="secondary" onclick="runLayout()">Relayout</button>
        </div>
        <div id="cy"></div>
    </div>

    <script>
        // Detect API base path - handles both direct access (localhost:8080) and proxied (/genome)
        const API_BASE = window.location.pathname.includes('/genome') ? '/genome/api' : '/api';

        let cy;
        let selectedPattern = null;
        let selectedEdge = null;
        let currentPiece = null;

        const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

        // ======== Web Audio Playback Engine ========
        let audioCtx = null;
        let masterGain = null;
        let isPlaying = false;
        let playbackStartTime = 0;
        let playbackData = null;
        let scheduledNodes = [];
        let playbackTimer = null;
        let playbackTempo = 120;  // BPM for playback tempo control
        let trackGains = {};  // track index -> GainNode for per-track volume
        let currentTrackNames = [];  // track names from playback data
        let trackMuted = {};  // track_id -> boolean (muted state)
        let allPatterns = [];  // All patterns for swap dropdown
        let currentGraphData = null;  // Store graph data for track info
        const TRACK_COLORS = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9', '#fd79a8', '#a29bfe', '#00b894', '#e17055'];

        // Drum samples (noise-based synthesis for simplicity)
        const DRUM_MAP = {
            35: 'kick', 36: 'kick',
            38: 'snare', 40: 'snare',
            42: 'hihat_closed', 44: 'hihat_closed', 46: 'hihat_open',
            49: 'crash', 51: 'ride', 52: 'crash', 57: 'crash',
            45: 'tom_low', 47: 'tom_mid', 48: 'tom_high',
            43: 'tom_low', 50: 'tom_high'
        };

        function initAudio() {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                masterGain = audioCtx.createGain();
                masterGain.connect(audioCtx.destination);
                masterGain.gain.value = 0.5;
            }
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
        }

        function getTrackGain(trackId) {
            if (!audioCtx) initAudio();
            if (!trackGains[trackId]) {
                trackGains[trackId] = audioCtx.createGain();
                trackGains[trackId].connect(masterGain);
                trackGains[trackId].gain.value = 1.0;
            }
            return trackGains[trackId];
        }

        function setTrackVolume(trackId, value) {
            if (audioCtx && trackGains[trackId]) {
                trackGains[trackId].gain.setValueAtTime(value / 100, audioCtx.currentTime);
            }
        }

        function updateTrackVolumesUI(tracks) {
            const container = document.getElementById('track-mixer-content');
            if (!container || !tracks) return;

            container.innerHTML = tracks.map((track, i) => {
                const color = TRACK_COLORS[i % TRACK_COLORS.length];
                const name = track.is_drum ? 'Drums' : 'Track ' + track.track_id;
                const muted = trackMuted[track.track_id] || false;
                return `<div class="track-item">
                    <div class="track-color-dot" style="background:${color}"></div>
                    <span class="track-name">${name}</span>
                    <input type="range" class="track-slider" min="0" max="100" value="${muted ? 0 : 100}"
                        oninput="setTrackVolume(${track.track_id}, this.value); this.nextElementSibling.nextElementSibling.textContent = this.value + '%'">
                    <button class="track-mute ${muted ? 'muted' : ''}" onclick="toggleTrackMute(${track.track_id}, this)">
                        ${muted ? 'M' : 'M'}
                    </button>
                    <span style="width:30px;font-size:10px;color:#888">${muted ? '0%' : '100%'}</span>
                </div>`;
            }).join('');

            // Update track legend in toolbar
            updateTrackLegend(tracks);
        }

        function toggleTrackMute(trackId, btn) {
            trackMuted[trackId] = !trackMuted[trackId];
            const muted = trackMuted[trackId];
            btn.classList.toggle('muted', muted);

            // Find the slider and update
            const slider = btn.previousElementSibling;
            slider.value = muted ? 0 : 100;
            slider.nextElementSibling.nextElementSibling.textContent = muted ? '0%' : '100%';

            setTrackVolume(trackId, muted ? 0 : 100);
        }

        function toggleExpand(id) {
            const el = document.getElementById(id);
            el.classList.toggle('collapsed');
        }

        function updateTrackLegend(tracks) {
            const legend = document.getElementById('track-legend');
            if (!legend || !tracks) return;

            legend.innerHTML = tracks.slice(0, 6).map((track, i) => {
                const color = TRACK_COLORS[i % TRACK_COLORS.length];
                const name = track.is_drum ? 'Drums' : 'T' + track.track_id;
                return `<div class="track-legend-item">
                    <div style="width:8px;height:8px;border-radius:50%;background:${color}"></div>
                    <span>${name}</span>
                </div>`;
            }).join('') + (tracks.length > 6 ? `<span style="color:#888">+${tracks.length - 6} more</span>` : '');
        }

        function midiToFreq(midiNote) {
            return 440 * Math.pow(2, (midiNote - 69) / 12);
        }

        function playTone(freq, startTime, duration, velocity = 80, trackId = 0) {
            const osc = audioCtx.createOscillator();
            const gainNode = audioCtx.createGain();

            osc.type = 'sine';
            osc.frequency.value = freq;

            const vol = (velocity / 127) * 0.3;
            gainNode.gain.setValueAtTime(0, startTime);
            gainNode.gain.linearRampToValueAtTime(vol, startTime + 0.01);
            gainNode.gain.setValueAtTime(vol, startTime + duration - 0.02);
            gainNode.gain.linearRampToValueAtTime(0, startTime + duration);

            osc.connect(gainNode);
            // Route through per-track gain for volume control
            gainNode.connect(getTrackGain(trackId));

            osc.start(startTime);
            osc.stop(startTime + duration + 0.05);

            scheduledNodes.push(osc);
            return osc;
        }

        function playDrum(drumType, startTime, velocity = 80, trackId = 0) {
            // Use short click sound for all drums to differentiate from melodic notes
            const vol = (velocity / 127) * 0.4;
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();

            // Click: very short high-frequency burst
            osc.type = 'square';
            osc.frequency.value = 1000;  // 1kHz click

            gain.gain.setValueAtTime(vol, startTime);
            gain.gain.exponentialRampToValueAtTime(0.001, startTime + 0.02);  // 20ms decay

            osc.connect(gain);
            // Route through per-track gain for volume control
            gain.connect(getTrackGain(trackId));
            osc.start(startTime);
            osc.stop(startTime + 0.03);
            scheduledNodes.push(osc);
        }

        async function playPiece() {
            if (!currentPiece) {
                alert('Please select a piece first');
                return;
            }

            initAudio();
            stopPlayback();

            document.getElementById('playback-status').textContent = 'Loading...';

            try {
                const resp = await fetch(`${API_BASE}/playback/${encodeURIComponent(currentPiece)}`);
                playbackData = await resp.json();

                if (playbackData.error) {
                    document.getElementById('playback-status').textContent = 'Error: ' + playbackData.error;
                    return;
                }

                // Use current tempo slider value (user can adjust before pressing Play)
                // playbackTempo is already set from the slider's current value

                isPlaying = true;

                // Convert ticks to seconds using the tempo slider
                // Formula: seconds = ticks * 60 / (bpm * ticks_per_beat)
                const ticksPerBeat = playbackData.ticks_per_beat || 480;
                const ticksToSec = 60.0 / (playbackTempo * ticksPerBeat);

                // Calculate total duration first
                const totalDuration = playbackData.duration_ticks * ticksToSec;

                // Flatten all events for batched scheduling
                const allEvents = [];
                for (const track of playbackData.tracks) {
                    for (const event of track.events) {
                        allEvents.push({...event, is_drum: track.is_drum, track_id: track.track_id});
                    }
                }

                // Update per-track volume sliders UI
                updateTrackVolumesUI(playbackData.tracks);

                console.log(`Playing ${allEvents.length} notes at ${playbackTempo} BPM, duration: ${totalDuration.toFixed(1)}s`);

                document.getElementById('play-btn').disabled = true;
                document.getElementById('stop-btn').disabled = false;
                document.getElementById('playback-status').textContent = `Playing ${allEvents.length} notes...`;

                // Sort events by time for JIT scheduling
                allEvents.sort((a, b) => a.time - b.time);

                // JIT scheduling: only schedule notes within a lookahead window
                // This avoids creating thousands of audio nodes upfront
                const LOOKAHEAD_SEC = 2.0;  // Schedule 2 seconds ahead
                const SCHEDULE_INTERVAL = 100;  // Check every 100ms
                let nextEventIndex = 0;

                playbackStartTime = audioCtx.currentTime + 0.1;  // Small initial delay

                function scheduleUpcoming() {
                    if (!isPlaying) return;

                    const currentPlaybackTime = audioCtx.currentTime - playbackStartTime;
                    const scheduleUntil = currentPlaybackTime + LOOKAHEAD_SEC;

                    // Schedule events that fall within the lookahead window
                    while (nextEventIndex < allEvents.length) {
                        const event = allEvents[nextEventIndex];
                        const eventTimeSec = event.time * ticksToSec;

                        if (eventTimeSec > scheduleUntil) {
                            break;  // Event is too far ahead, wait
                        }

                        const startTime = playbackStartTime + eventTimeSec;
                        const duration = Math.max(0.05, event.duration * ticksToSec);

                        if (event.is_drum) {
                            const drumType = DRUM_MAP[event.pitch] || 'generic';
                            playDrum(drumType, startTime, event.velocity, event.track_id);
                        } else {
                            const freq = midiToFreq(event.pitch);
                            playTone(freq, startTime, duration, event.velocity, event.track_id);
                        }

                        nextEventIndex++;
                    }

                    // Continue scheduling if there are more events
                    if (nextEventIndex < allEvents.length && isPlaying) {
                        setTimeout(scheduleUpcoming, SCHEDULE_INTERVAL);
                    }
                }

                // Start JIT scheduling
                scheduleUpcoming();

                // Timer using requestAnimationFrame - runs independently
                function updateTimer() {
                    if (!isPlaying) return;
                    const elapsed = audioCtx.currentTime - playbackStartTime;
                    if (elapsed >= totalDuration) {
                        stopPlayback();
                        return;
                    }
                    const displayElapsed = Math.max(0, elapsed);
                    const mins = Math.floor(displayElapsed / 60);
                    const secs = Math.floor(displayElapsed % 60);
                    const totalMins = Math.floor(totalDuration / 60);
                    const totalSecs = Math.floor(totalDuration % 60);
                    document.getElementById('playback-time').textContent =
                        `${mins}:${secs.toString().padStart(2, '0')} / ${totalMins}:${totalSecs.toString().padStart(2, '0')}`;
                    playbackTimer = requestAnimationFrame(updateTimer);
                }
                playbackTimer = requestAnimationFrame(updateTimer);

            } catch (err) {
                document.getElementById('playback-status').textContent = 'Error: ' + err.message;
            }
        }

        function stopPlayback() {
            isPlaying = false;
            if (playbackTimer) {
                cancelAnimationFrame(playbackTimer);
                playbackTimer = null;
            }

            // Stop all scheduled oscillators
            for (const node of scheduledNodes) {
                try {
                    node.stop();
                } catch (e) {}
            }
            scheduledNodes = [];

            document.getElementById('play-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
            document.getElementById('playback-time').textContent = '0:00 / 0:00';
            if (currentPiece) {
                document.getElementById('playback-status').textContent = 'Ready to play: ' + currentPiece.substring(0, 30);
            }
        }

        // Volume control
        document.getElementById('volume-slider').addEventListener('input', (e) => {
            const vol = e.target.value / 100;
            document.getElementById('volume-value').textContent = e.target.value + '%';
            if (masterGain) {
                masterGain.gain.value = vol;
            }
        });

        // Tempo control
        document.getElementById('tempo-slider').addEventListener('input', (e) => {
            playbackTempo = parseInt(e.target.value);
            document.getElementById('tempo-value').textContent = playbackTempo + ' BPM';
        });

        // Edit edge transform
        async function editEdgeTransform() {
            if (!selectedEdge) return;
            const currentTransform = document.getElementById('edge-transform').textContent;
            const newTransform = prompt('Edit transform (e.g., T5, I0, R, τ480):', currentTransform);
            if (!newTransform || newTransform === currentTransform) return;

            const resp = await fetch(`${API_BASE}/edit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ edge_id: selectedEdge, transform: newTransform })
            });
            const result = await resp.json();
            if (result.success) {
                document.getElementById('edge-transform').textContent = newTransform;
                // Update the cytoscape edge data so the edit persists when re-selecting
                const edge = cy.$('#' + selectedEdge);
                if (edge.length > 0) {
                    edge.data('transform', newTransform);
                }
                document.getElementById('playback-status').textContent = 'Edit saved - will apply on next playback';
                document.getElementById('save-status').textContent = 'Unsaved changes';
                document.getElementById('save-status').style.color = '#f59e0b';
            }
        }

        // Save all edits to checkpoint
        async function saveEdits() {
            document.getElementById('save-status').textContent = 'Saving...';
            document.getElementById('save-status').style.color = '#888';

            try {
                const resp = await fetch(`${API_BASE}/save`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                const result = await resp.json();
                if (result.success) {
                    document.getElementById('save-status').textContent =
                        `Saved ${result.saved.edges} edge edits, ${result.saved.patterns} pattern edits, ${result.saved.swaps || 0} swaps`;
                    document.getElementById('save-status').style.color = '#10b981';
                } else {
                    document.getElementById('save-status').textContent = 'Error: ' + (result.error || 'Unknown error');
                    document.getElementById('save-status').style.color = '#ef4444';
                }
            } catch (err) {
                document.getElementById('save-status').textContent = 'Error: ' + err.message;
                document.getElementById('save-status').style.color = '#ef4444';
            }
        }

        // Initialize Cytoscape
        function initGraph(data) {
            currentGraphData = data;  // Store for later use

            // Normalize elements to array format (handle both dict and array formats)
            let elements = data.elements;
            if (elements && !Array.isArray(elements)) {
                // Convert {nodes: [...], edges: [...]} to flat array
                elements = [...(elements.nodes || []), ...(elements.edges || [])];
                data.elements = elements;
            }

            // Build patterns list for swap dropdown
            allPatterns = elements.filter(el => el.group === 'nodes' || (el.data && el.data.id && el.data.id.startsWith('P'))).map(el => ({
                id: el.data.id,
                label: el.data.label,
                length: el.data.length,
                track_id: el.data.track_id || 0
            }));

            cy = cytoscape({
                container: document.getElementById('cy'),
                elements: elements,
                style: [
                    {
                        selector: 'node',
                        style: {
                            'label': 'data(label)',
                            'background-color': 'data(trackColor)',
                            'color': '#fff',
                            'font-size': '10px',
                            'text-valign': 'center',
                            'width': 'mapData(length, 1, 20, 20, 60)',
                            'height': 'mapData(length, 1, 20, 20, 60)',
                            'border-width': 2,
                            'border-color': 'data(trackColor)',
                        }
                    },
                    {
                        selector: 'node:selected',
                        style: {
                            'background-color': '#fff',
                            'color': '#000',
                            'border-width': 3,
                            'border-color': '#e94560',
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'width': 2,
                            'line-color': 'data(color)',
                            'target-arrow-color': 'data(color)',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'opacity': 0.7,
                        }
                    },
                    {
                        selector: 'edge:selected',
                        style: {
                            'width': 4,
                            'opacity': 1,
                        }
                    },
                    {
                        selector: 'node.highlighted',
                        style: {
                            'background-color': '#ffeb3b',
                            'border-color': '#ffc107',
                            'border-width': 4,
                        }
                    },
                ],
                layout: {
                    name: 'cose',
                    animate: false,
                    nodeRepulsion: 8000,
                    idealEdgeLength: 100,
                    edgeElasticity: 100,
                }
            });

            // Node click handler
            cy.on('tap', 'node', function(e) {
                selectedPattern = parseInt(e.target.id().replace('P', ''));
                selectedEdge = null;
                showPatternDetail(selectedPattern);
                document.getElementById('edge-detail').style.display = 'none';
            });

            // Edge click handler
            cy.on('tap', 'edge', function(e) {
                selectedEdge = parseInt(e.target.id().replace('E', ''));
                showEdgeDetail(e.target.data());
                document.getElementById('pattern-detail').classList.remove('active');
            });

            // Background click
            cy.on('tap', function(e) {
                if (e.target === cy) {
                    selectedPattern = null;
                    selectedEdge = null;
                    document.getElementById('pattern-detail').classList.remove('active');
                    document.getElementById('edge-detail').style.display = 'none';
                }
            });
        }

        // Load stats
        async function loadStats() {
            const resp = await fetch(`${API_BASE}/stats`);
            const stats = await resp.json();
            document.getElementById('stat-patterns').textContent = stats.n_patterns.toLocaleString();
            document.getElementById('stat-edges').textContent = stats.n_edges.toLocaleString();
            document.getElementById('stat-pieces').textContent = stats.n_pieces;
            document.getElementById('stat-degree').textContent = stats.avg_degree.toFixed(2);
        }

        // Load pieces list
        async function loadPieces() {
            const resp = await fetch(`${API_BASE}/pieces`);
            const data = await resp.json();
            const select = document.getElementById('piece-filter');
            data.pieces.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p;
                opt.textContent = p.substring(0, 30) + (p.length > 30 ? '...' : '');
                select.appendChild(opt);
            });
        }

        // Load graph data
        async function loadGraph(piece = null) {
            const url = piece ? `${API_BASE}/cytoscape/${encodeURIComponent(piece)}` : `${API_BASE}/cytoscape`;
            const resp = await fetch(url);
            const data = await resp.json();

            if (cy) {
                cy.destroy();
            }
            initGraph(data);
        }

        // Show pattern detail
        async function showPatternDetail(pid) {
            const resp = await fetch(`${API_BASE}/pattern/${pid}`);
            const pattern = await resp.json();

            document.getElementById('detail-id').textContent = `P${pid}`;
            document.getElementById('detail-length').textContent = pattern.pitch_classes.length;

            // Get track info from cytoscape node
            const nodeData = cy.$(`#P${pid}`).data();
            const trackId = nodeData?.track_id || 0;
            const trackColor = nodeData?.trackColor || TRACK_COLORS[0];
            document.getElementById('detail-track').innerHTML =
                `<span style="display:inline-flex;align-items:center;gap:6px">` +
                `<span style="width:10px;height:10px;border-radius:50%;background:${trackColor}"></span>` +
                `Track ${trackId}</span>`;

            // Render pitch classes
            const pitchDiv = document.getElementById('detail-pitches');
            pitchDiv.innerHTML = pattern.pitch_classes.map(pc =>
                `<div class="pitch pitch-${pc}" title="${NOTE_NAMES[pc]}">${NOTE_NAMES[pc]}</div>`
            ).join('');

            // Populate pattern swap dropdown - show patterns from same track first
            const swapSelect = document.getElementById('pattern-swap-select');
            const sameTrack = allPatterns.filter(p => p.track_id === trackId && p.id !== `P${pid}`);
            const otherTrack = allPatterns.filter(p => p.track_id !== trackId && p.id !== `P${pid}`);

            swapSelect.innerHTML = '<option value="">-- Select pattern to swap --</option>' +
                (sameTrack.length > 0 ? `<optgroup label="Same Track (${trackId})">` +
                    sameTrack.slice(0, 50).map(p =>
                        `<option value="${p.id}">${p.label} (len: ${p.length})</option>`
                    ).join('') + '</optgroup>' : '') +
                (otherTrack.length > 0 ? `<optgroup label="Other Tracks">` +
                    otherTrack.slice(0, 50).map(p =>
                        `<option value="${p.id}">${p.label} T${p.track_id} (len: ${p.length})</option>`
                    ).join('') + '</optgroup>' : '');

            // Load edges
            const edgeResp = await fetch(`${API_BASE}/edges/${pid}`);
            const edges = await edgeResp.json();

            const edgeDiv = document.getElementById('detail-edges');
            edgeDiv.innerHTML = edges.slice(0, 20).map(e => `
                <div class="edge-item" onclick="selectEdge(${e.id})">
                    <span class="edge-transform">${e.transform}</span>
                    → P${e.target}
                </div>
            `).join('') + (edges.length > 20 ? `<div style="color:#888">${edges.length - 20} more...</div>` : '');

            document.getElementById('pattern-detail').classList.add('active');
        }

        // Swap pattern content - copies target's notes into source pattern
        async function swapPattern(targetPatternId) {
            if (!selectedPattern || !targetPatternId) return;

            const sourceId = `P${selectedPattern}`;

            const resp = await fetch(`${API_BASE}/swap_pattern`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    source_pattern: sourceId,
                    target_pattern: targetPatternId
                })
            });
            const result = await resp.json();
            if (result.success) {
                document.getElementById('save-status').textContent = `✓ Applied: ${sourceId} now plays ${targetPatternId}'s notes (unsaved)`;
                document.getElementById('save-status').style.color = '#f59e0b';

                // Visual feedback - mark source pattern as modified
                const sourceNode = cy.$(`#${sourceId}`);
                sourceNode.style('border-color', '#f59e0b');
                sourceNode.style('border-width', 4);

                // Update label to show it's been swapped
                const currentLabel = sourceNode.data('label');
                if (!currentLabel.includes('→')) {
                    sourceNode.data('label', `${currentLabel}→${targetPatternId}`);
                }

                // Flash the target to show what content was copied
                cy.$(`#${targetPatternId}`).flashClass('highlighted', 1000);

                // Reset dropdown
                document.getElementById('pattern-swap-select').value = '';

                // Refresh the pattern detail panel to show new pitch content
                await showPatternDetail(selectedPattern);

                // Force playback data refresh by clearing cache
                playbackData = null;
            }
        }

        // Show edge detail
        function showEdgeDetail(data) {
            document.getElementById('edge-detail').style.display = 'block';
            document.getElementById('edge-id').textContent = data.id;
            document.getElementById('edge-transform').textContent = data.transform;
            document.getElementById('edge-type').textContent = data.edge_type;
        }

        // Select edge
        function selectEdge(eid) {
            selectedEdge = eid;
            cy.$(`#E${eid}`).select();
        }

        // Factor edge
        async function factorEdge() {
            if (!selectedEdge) return;
            const resp = await fetch(`${API_BASE}/factor/${selectedEdge}`, { method: 'POST' });
            const result = await resp.json();
            if (result.new_edge_ids) {
                alert(`Factored into ${result.new_edge_ids.length} edges`);
                loadGraph(document.getElementById('piece-filter').value || null);
            }
        }

        // Apply transform
        async function applyTransform() {
            if (!selectedPattern) return;
            const transform = prompt('Enter transform (e.g., T5, I0, R, τ480):');
            if (!transform) return;

            const resp = await fetch(`${API_BASE}/apply`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pattern_id: selectedPattern, transform })
            });
            const result = await resp.json();
            if (result.new_pattern_id) {
                alert(`Created pattern P${result.new_pattern_id}`);
                loadGraph(document.getElementById('piece-filter').value || null);
            }
        }

        // Clone subgraph
        async function cloneSubgraph() {
            if (!selectedPattern) return;
            const transform = prompt('Enter transform to apply to cloned subgraph:');
            if (!transform) return;

            const resp = await fetch(`${API_BASE}/clone`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ root_id: selectedPattern, transform })
            });
            const result = await resp.json();
            if (result.new_root_id) {
                alert(`Cloned subgraph with new root P${result.new_root_id}`);
                loadGraph(document.getElementById('piece-filter').value || null);
            }
        }

        // Reset view
        function resetView() {
            cy.fit();
        }

        // Run layout
        function runLayout() {
            cy.layout({ name: 'cose', animate: true }).run();
        }

        // Filter handlers
        document.getElementById('piece-filter').addEventListener('change', (e) => {
            currentPiece = e.target.value || null;
            loadGraph(currentPiece);
            if (currentPiece) {
                document.getElementById('playback-status').textContent = 'Ready to play: ' + currentPiece.substring(0, 30);
            } else {
                document.getElementById('playback-status').textContent = 'Select a piece to enable playback';
            }
            stopPlayback();
        });

        document.getElementById('transform-filter').addEventListener('change', (e) => {
            const filter = e.target.value;
            if (filter) {
                cy.edges().forEach(edge => {
                    if (edge.data('transform').includes(filter)) {
                        edge.style('opacity', 1);
                    } else {
                        edge.style('opacity', 0.1);
                    }
                });
            } else {
                cy.edges().style('opacity', 0.7);
            }
        });

        // Initialize
        loadStats();
        loadPieces();
        // Don't load full graph - too large for browser
        // User must select a piece from dropdown first
        console.log('Select a piece from the dropdown to view its subgraph');
    </script>
</body>
</html>'''
        self.send_html(html)

    def serve_dag_editor(self):
        """Serve the DAG expression editor HTML."""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Music DAG Editor</title>
    <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
    <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
    <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
        }
        #sidebar {
            width: 320px;
            background: #16213e;
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid #0f3460;
        }
        #graph-container {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        #toolbar {
            background: #0f3460;
            padding: 10px 20px;
            display: flex;
            gap: 15px;
            align-items: center;
        }
        #cy { flex: 1; }
        h1 { font-size: 18px; margin-bottom: 20px; color: #4CAF50; }
        h2 { font-size: 14px; margin: 15px 0 10px; color: #0f4c75; }
        .stat {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #0f3460;
        }
        .stat-label { color: #888; }
        .stat-value { font-weight: bold; }
        select, button, input {
            background: #0f3460;
            border: 1px solid #1a1a2e;
            color: #e0e0e0;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
        }
        select:hover, button:hover { background: #16213e; }
        button { background: #4CAF50; border-color: #4CAF50; }
        button:hover { background: #66BB6A; }
        button.secondary { background: #0f4c75; border-color: #0f4c75; }
        #node-detail {
            background: #0f3460;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            display: none;
        }
        #node-detail.active { display: block; }
        .pitch-classes {
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
            margin: 10px 0;
        }
        .pitch {
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .pitch-0 { background: #e94560; }
        .pitch-1 { background: #ff6b6b; }
        .pitch-2 { background: #f9a825; }
        .pitch-3 { background: #fdd835; }
        .pitch-4 { background: #8bc34a; }
        .pitch-5 { background: #4caf50; }
        .pitch-6 { background: #00bcd4; }
        .pitch-7 { background: #2196f3; }
        .pitch-8 { background: #3f51b5; }
        .pitch-9 { background: #673ab7; }
        .pitch-10 { background: #9c27b0; }
        .pitch-11 { background: #e91e63; }
        .children-list { margin-top: 10px; }
        .child-item {
            padding: 6px 10px;
            background: #16213e;
            border-radius: 4px;
            margin-bottom: 4px;
            font-size: 12px;
            cursor: pointer;
        }
        .child-item:hover { background: #1a1a2e; }
        .legend {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 12px;
        }
        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }
        .type-filter { margin-top: 20px; }
        .type-filter label { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; cursor: pointer; }
        .type-filter input[type="checkbox"] { width: 16px; height: 16px; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>Music DAG Editor</h1>
        <p style="color:#888;font-size:12px;margin-bottom:20px">Expression-based hierarchical compression</p>

        <div class="stats">
            <div class="stat">
                <span class="stat-label">Total Nodes</span>
                <span class="stat-value" id="stat-nodes">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Patterns</span>
                <span class="stat-value" id="stat-patterns">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Sequences</span>
                <span class="stat-value" id="stat-seq">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Transforms</span>
                <span class="stat-value" id="stat-transform">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Unique Transforms</span>
                <span class="stat-value" id="stat-unique-t">-</span>
            </div>
        </div>

        <div class="type-filter">
            <h2>Filter by Type</h2>
            <label><input type="checkbox" id="show-pattern" checked><span style="color:#2196F3">Pattern</span></label>
            <label><input type="checkbox" id="show-seq" checked><span style="color:#4CAF50">Sequence</span></label>
            <label><input type="checkbox" id="show-par" checked><span style="color:#FF9800">Parallel</span></label>
            <label><input type="checkbox" id="show-transform" checked><span style="color:#9C27B0">Transform</span></label>
        </div>

        <div id="node-detail">
            <h2>Node <span id="detail-id"></span></h2>
            <div class="stat">
                <span class="stat-label">Type</span>
                <span class="stat-value" id="detail-type">-</span>
            </div>
            <div class="stat" id="detail-transform-row" style="display:none">
                <span class="stat-label">Transform</span>
                <span class="stat-value" id="detail-transform" style="color:#9C27B0">-</span>
            </div>
            <div id="detail-pitches-container" style="display:none">
                <h2>Pitch Classes</h2>
                <div class="pitch-classes" id="detail-pitches"></div>
            </div>
            <div class="children-list" id="detail-children-container" style="display:none">
                <h2>Children (<span id="detail-child-count">0</span>)</h2>
                <div id="detail-children"></div>
            </div>
            <div style="margin-top:15px">
                <button onclick="expandNode()">Expand to Events</button>
            </div>
        </div>
    </div>

    <div id="graph-container">
        <div id="toolbar">
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #2196F3"></div>
                    <span>Pattern</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4CAF50"></div>
                    <span>Sequence</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF9800"></div>
                    <span>Parallel</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #9C27B0"></div>
                    <span>Transform</span>
                </div>
            </div>
            <div style="flex:1"></div>
            <select id="layout-select">
                <option value="dagre">Dagre (hierarchical)</option>
                <option value="cose">Force-directed</option>
                <option value="breadthfirst">Breadth-first</option>
            </select>
            <button class="secondary" onclick="resetView()">Reset View</button>
            <button class="secondary" onclick="runLayout()">Relayout</button>
        </div>
        <div id="cy"></div>
    </div>

    <script>
        let cy;
        let selectedNode = null;
        const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

        // Initialize Cytoscape
        function initGraph(data) {
            // Normalize elements to array format (handle both dict and array formats)
            let elements = data.elements;
            if (elements && !Array.isArray(elements)) {
                // Convert {nodes: [...], edges: [...]} to flat array
                elements = [...(elements.nodes || []), ...(elements.edges || [])];
                data.elements = elements;
            }

            cy = cytoscape({
                container: document.getElementById('cy'),
                elements: elements,
                style: [
                    {
                        selector: 'node',
                        style: {
                            'label': 'data(label)',
                            'background-color': 'data(color)',
                            'color': '#e0e0e0',
                            'font-size': '10px',
                            'text-valign': 'center',
                            'width': 'mapData(notes, 0, 20, 25, 60)',
                            'height': 'mapData(notes, 0, 20, 25, 60)',
                            'border-width': 2,
                            'border-color': '#1a1a2e',
                        }
                    },
                    {
                        selector: 'node:selected',
                        style: {
                            'border-color': '#fff',
                            'border-width': 3,
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'width': 2,
                            'line-color': '#555',
                            'target-arrow-color': '#555',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'opacity': 0.6,
                            'label': 'data(label)',
                            'font-size': '8px',
                            'color': '#888',
                            'text-rotation': 'autorotate',
                        }
                    },
                ],
                layout: {
                    name: 'dagre',
                    rankDir: 'TB',
                    nodeSep: 30,
                    rankSep: 50,
                    ranker: 'tight-tree',
                    spacingFactor: 0.8,
                    fit: true,
                    padding: 30
                }
            });

            cy.on('tap', 'node', function(e) {
                selectedNode = parseInt(e.target.id());
                showNodeDetail(selectedNode);
            });

            cy.on('tap', function(e) {
                if (e.target === cy) {
                    selectedNode = null;
                    document.getElementById('node-detail').classList.remove('active');
                }
            });
        }

        // Load stats
        async function loadStats() {
            const resp = await fetch(`${API_BASE}/dag/stats`);
            const stats = await resp.json();
            document.getElementById('stat-nodes').textContent = stats.node_count?.toLocaleString() || '-';
            document.getElementById('stat-patterns').textContent = stats.by_type?.pattern?.toLocaleString() || '0';
            document.getElementById('stat-seq').textContent = stats.by_type?.seq?.toLocaleString() || '0';
            document.getElementById('stat-transform').textContent = stats.by_type?.transform?.toLocaleString() || '0';
            document.getElementById('stat-unique-t').textContent = stats.unique_transforms || '0';
        }

        // Load graph
        async function loadGraph() {
            const resp = await fetch(`${API_BASE}/dag/cytoscape?max_nodes=2000`);
            const data = await resp.json();
            if (cy) cy.destroy();
            initGraph(data);
            applyFilters();
        }

        // Show node detail
        async function showNodeDetail(nid) {
            const resp = await fetch(`${API_BASE}/dag/node/${nid}`);
            const node = await resp.json();

            document.getElementById('detail-id').textContent = nid;
            document.getElementById('detail-type').textContent = node.type;
            document.getElementById('detail-type').style.color = {
                'pattern': '#2196F3',
                'seq': '#4CAF50',
                'par': '#FF9800',
                'transform': '#9C27B0'
            }[node.type] || '#fff';

            // Transform
            const transformRow = document.getElementById('detail-transform-row');
            if (node.transform) {
                document.getElementById('detail-transform').textContent = node.transform;
                transformRow.style.display = 'flex';
            } else {
                transformRow.style.display = 'none';
            }

            // Pitch classes
            const pitchContainer = document.getElementById('detail-pitches-container');
            if (node.pitch_classes && node.pitch_classes.length > 0) {
                const pitchDiv = document.getElementById('detail-pitches');
                pitchDiv.innerHTML = node.pitch_classes.slice(0, 30).map(pc =>
                    `<div class="pitch pitch-${pc}" title="${NOTE_NAMES[pc]}">${NOTE_NAMES[pc]}</div>`
                ).join('') + (node.pitch_classes.length > 30 ? '<span style="color:#888">...</span>' : '');
                pitchContainer.style.display = 'block';
            } else {
                pitchContainer.style.display = 'none';
            }

            // Children
            const childrenContainer = document.getElementById('detail-children-container');
            if (node.children && node.children.length > 0) {
                document.getElementById('detail-child-count').textContent = node.children.length;
                const childrenDiv = document.getElementById('detail-children');
                childrenDiv.innerHTML = node.children.slice(0, 20).map(cid =>
                    `<div class="child-item" onclick="selectNode(${cid})">Node ${cid}</div>`
                ).join('') + (node.children.length > 20 ? '<div style="color:#888;padding:6px">...</div>' : '');
                childrenContainer.style.display = 'block';
            } else {
                childrenContainer.style.display = 'none';
            }

            document.getElementById('node-detail').classList.add('active');
        }

        function selectNode(nid) {
            cy.$(`#${nid}`).select();
            selectedNode = nid;
            showNodeDetail(nid);
            cy.center(cy.$(`#${nid}`));
        }

        async function expandNode() {
            if (!selectedNode) return;
            const resp = await fetch(`${API_BASE}/dag/expand/${selectedNode}`);
            const data = await resp.json();
            alert(`Node ${selectedNode} expands to ${data.event_count} events\\n\\nFirst few: ${JSON.stringify(data.events.slice(0,5))}`);
        }

        // Filters
        function applyFilters() {
            const showPattern = document.getElementById('show-pattern').checked;
            const showSeq = document.getElementById('show-seq').checked;
            const showPar = document.getElementById('show-par').checked;
            const showTransform = document.getElementById('show-transform').checked;

            cy.nodes().forEach(node => {
                const type = node.data('type');
                let visible = true;
                if (type === 'pattern' && !showPattern) visible = false;
                if (type === 'seq' && !showSeq) visible = false;
                if (type === 'par' && !showPar) visible = false;
                if (type === 'transform' && !showTransform) visible = false;
                node.style('display', visible ? 'element' : 'none');
            });
        }

        document.querySelectorAll('.type-filter input').forEach(cb => {
            cb.addEventListener('change', applyFilters);
        });

        document.getElementById('layout-select').addEventListener('change', runLayout);

        function runLayout() {
            const layoutName = document.getElementById('layout-select').value;
            let options = { name: layoutName, animate: true, fit: true, padding: 30 };
            if (layoutName === 'dagre') {
                options.rankDir = 'TB';
                options.nodeSep = 30;
                options.rankSep = 50;
                options.ranker = 'tight-tree';
                options.spacingFactor = 0.8;
            } else if (layoutName === 'cose') {
                options.nodeRepulsion = 8000;
                options.idealEdgeLength = 50;
                options.gravity = 0.25;
            }
            cy.layout(options).run();
        }

        function resetView() {
            cy.fit();
        }

        // Initialize
        loadStats();
        loadGraph();
    </script>
</body>
</html>'''
        self.send_html(html)

    def serve_transform_editor(self):
        """Serve the transform-space editor where position = transform."""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Transform Space Editor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a1a;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
            overflow: hidden;
        }
        #sidebar {
            width: 300px;
            background: #12122a;
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid #2a2a4a;
        }
        #canvas-container {
            flex: 1;
            position: relative;
            overflow: hidden;
        }
        #canvas {
            position: absolute;
            top: 0;
            left: 0;
        }
        h1 { font-size: 18px; margin-bottom: 15px; color: #00ff88; }
        h2 { font-size: 14px; margin: 15px 0 10px; color: #4488ff; }
        .info-box {
            background: #1a1a3a;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .stat {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #2a2a4a;
            font-size: 13px;
        }
        .stat:last-child { border-bottom: none; }
        .stat-label { color: #888; }
        .stat-value { font-weight: bold; color: #00ff88; }
        select, button {
            width: 100%;
            background: #2a2a4a;
            border: 1px solid #3a3a5a;
            color: #e0e0e0;
            padding: 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            margin-bottom: 10px;
        }
        select:hover, button:hover { background: #3a3a5a; }
        button.play { background: #00aa55; border-color: #00aa55; }
        button.play:hover { background: #00cc66; }
        button.stop { background: #aa3333; border-color: #aa3333; }
        button.stop:hover { background: #cc4444; }
        #node-detail, #edge-detail {
            background: #1a1a3a;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            display: none;
        }
        #node-detail.active, #edge-detail.active { display: block; }
        .pitch-display {
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
            margin: 10px 0;
        }
        .pitch-note {
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
            color: #000;
        }
        .slider-group {
            margin: 10px 0;
        }
        .slider-group label {
            display: block;
            font-size: 12px;
            color: #888;
            margin-bottom: 5px;
        }
        .slider-group input[type="range"] {
            width: 100%;
        }
        .slider-value {
            text-align: center;
            font-weight: bold;
            color: #00ff88;
        }
        .axis-label {
            position: absolute;
            font-size: 12px;
            color: #4488ff;
            pointer-events: none;
        }
        #x-axis-label { bottom: 10px; left: 50%; transform: translateX(-50%); }
        #y-axis-label { top: 50%; left: 10px; transform: rotate(-90deg) translateX(-50%); transform-origin: left center; }
        #help-text {
            font-size: 11px;
            color: #666;
            margin-top: 15px;
            line-height: 1.5;
        }
        .track-colors {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 10px 0;
        }
        .track-color {
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 11px;
        }
        .track-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>Transform Space Editor</h1>
        <p style="color:#888;font-size:12px;margin-bottom:15px">Position = Transform: Drag to edit</p>

        <select id="piece-select">
            <option value="">Select a piece...</option>
        </select>

        <div class="info-box">
            <div class="stat">
                <span class="stat-label">Nodes</span>
                <span class="stat-value" id="stat-nodes">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Edges</span>
                <span class="stat-value" id="stat-edges">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Tracks</span>
                <span class="stat-value" id="stat-tracks">-</span>
            </div>
        </div>

        <div id="track-legend" class="track-colors"></div>

        <button class="play" id="play-btn" onclick="playPiece()">Play</button>
        <button class="stop" id="stop-btn" onclick="stopPlayback()" disabled>Stop</button>

        <div class="slider-group">
            <label>Tempo</label>
            <input type="range" id="tempo-slider" min="40" max="300" value="120">
            <div class="slider-value" id="tempo-value">120 BPM</div>
        </div>

        <div class="slider-group">
            <label>Master Volume</label>
            <input type="range" id="volume-slider" min="0" max="100" value="50">
            <div class="slider-value" id="volume-value">50%</div>
        </div>

        <div id="track-volumes" style="max-height:200px;overflow-y:auto;margin:10px 0;"></div>

        <div id="node-detail">
            <h2>Pattern <span id="node-id"></span></h2>
            <div class="pitch-display" id="node-pitches"></div>
            <div class="stat">
                <span class="stat-label">Notes</span>
                <span class="stat-value" id="node-note-count">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Track</span>
                <span class="stat-value" id="node-track">-</span>
            </div>
        </div>

        <div id="edge-detail">
            <h2>Transform</h2>
            <div class="stat">
                <span class="stat-label">Edge ID</span>
                <span class="stat-value" id="edge-id">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Current</span>
                <span class="stat-value" id="edge-transform">-</span>
            </div>

            <div class="slider-group">
                <label>Pitch Transform (T)</label>
                <input type="range" id="pitch-slider" min="0" max="11" value="0">
                <div class="slider-value" id="pitch-value">T0</div>
            </div>

            <div class="slider-group">
                <label>Time Transform (tau)</label>
                <input type="range" id="time-slider" min="0" max="5" value="2">
                <div class="slider-value" id="time-value">480</div>
            </div>

            <button onclick="applyTransformEdit()">Apply Change</button>
        </div>

        <div style="margin-top:15px;">
            <button style="background:#4CAF50;border-color:#4CAF50;width:100%" onclick="saveEdits()">Save All Edits</button>
            <div id="save-status" style="font-size:11px;color:#888;margin-top:5px;text-align:center;"></div>
        </div>

        <div id="help-text">
            <strong>How it works:</strong><br>
            X-axis = Pitch transform (T0-T11)<br>
            Y-axis = Time position (tau)<br>
            <br>
            Drag a node to change its incoming transform.
            Changes affect all pieces using this relationship.
        </div>
    </div>

    <div id="canvas-container">
        <canvas id="canvas"></canvas>
        <div id="x-axis-label" class="axis-label">Pitch Transform (T)</div>
        <div id="y-axis-label" class="axis-label">Time (tau)</div>
    </div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('canvas-container');

let nodes = [];
let edges = [];
let currentPiece = null;
let pitchScale = 50;
let timeScale = 0.1;

// View transform
let viewX = 0;
let viewY = 0;
let zoom = 1;

// Interaction state
let selectedNode = null;
let selectedEdge = null;
let draggingNode = null;
let dragStartX = 0;
let dragStartY = 0;
let isPanning = false;
let panStartX = 0;
let panStartY = 0;

// Track colors
const TRACK_COLORS = [
    '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7',
    '#dfe6e9', '#fd79a8', '#a29bfe', '#00b894', '#e17055'
];

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const NOTE_COLORS = [
    '#e94560', '#ff6b6b', '#f9a825', '#fdd835', '#8bc34a',
    '#4caf50', '#00bcd4', '#2196f3', '#3f51b5', '#673ab7',
    '#9c27b0', '#e91e63'
];

const TAU_VALUES = [120, 240, 480, 960, 1920, 3840];

// Web Audio
let audioCtx = null;
let masterGain = null;
let isPlaying = false;
let scheduledNodes = [];
let playbackTempo = 120;
let trackGains = {};  // track_id -> GainNode for per-track volume control
let trackInfo = {};   // track_id -> {name, color} for UI

function resizeCanvas() {
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    render();
}

function worldToScreen(x, y) {
    return {
        x: (x - viewX) * zoom + canvas.width / 2,
        y: (y - viewY) * zoom + canvas.height / 2
    };
}

function screenToWorld(sx, sy) {
    return {
        x: (sx - canvas.width / 2) / zoom + viewX,
        y: (sy - canvas.height / 2) / zoom + viewY
    };
}

function render() {
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = '#1a1a3a';
    ctx.lineWidth = 1;

    const gridSpacingX = pitchScale * zoom;
    const gridSpacingY = 48 * zoom; // tau480 = 48px base

    const startWorld = screenToWorld(0, 0);
    const endWorld = screenToWorld(canvas.width, canvas.height);

    // Vertical grid lines (pitch)
    for (let x = Math.floor(startWorld.x / pitchScale) * pitchScale; x < endWorld.x; x += pitchScale) {
        const screen = worldToScreen(x, 0);
        ctx.beginPath();
        ctx.moveTo(screen.x, 0);
        ctx.lineTo(screen.x, canvas.height);
        ctx.stroke();
    }

    // Horizontal grid lines (time)
    for (let y = Math.floor(startWorld.y / 48) * 48; y < endWorld.y; y += 48) {
        const screen = worldToScreen(0, y);
        ctx.beginPath();
        ctx.moveTo(0, screen.y);
        ctx.lineTo(canvas.width, screen.y);
        ctx.stroke();
    }

    // Draw edges
    for (const edge of edges) {
        const sourceNode = nodes.find(n => n.id === edge.source);
        const targetNode = nodes.find(n => n.id === edge.target);
        if (!sourceNode || !targetNode) continue;

        const p1 = worldToScreen(sourceNode.x, sourceNode.y);
        const p2 = worldToScreen(targetNode.x, targetNode.y);

        const color = TRACK_COLORS[edge.track_id % TRACK_COLORS.length];
        ctx.strokeStyle = selectedEdge === edge ? '#ffffff' : color;
        ctx.lineWidth = selectedEdge === edge ? 3 : 1.5;
        ctx.globalAlpha = selectedEdge === edge ? 1 : 0.6;

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();

        // Arrow
        const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x);
        const arrowLen = 8;
        ctx.beginPath();
        ctx.moveTo(p2.x, p2.y);
        ctx.lineTo(p2.x - arrowLen * Math.cos(angle - 0.3), p2.y - arrowLen * Math.sin(angle - 0.3));
        ctx.lineTo(p2.x - arrowLen * Math.cos(angle + 0.3), p2.y - arrowLen * Math.sin(angle + 0.3));
        ctx.closePath();
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fill();

        ctx.globalAlpha = 1;
    }

    // Draw nodes
    for (const node of nodes) {
        const p = worldToScreen(node.x, node.y);
        const radius = 8 + node.note_count * 2;

        const color = TRACK_COLORS[node.track_id % TRACK_COLORS.length];

        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = selectedNode === node ? '#ffffff' : color;
        ctx.fill();

        if (selectedNode === node) {
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 3;
            ctx.stroke();
        }

        // Label
        ctx.fillStyle = selectedNode === node ? '#000' : '#fff';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('P' + node.pattern_id, p.x, p.y);
    }
}

async function loadPieces() {
    const resp = await fetch(`${API_BASE}/pieces`);
    const data = await resp.json();
    const select = document.getElementById('piece-select');
    data.pieces.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p.substring(0, 35) + (p.length > 35 ? '...' : '');
        select.appendChild(opt);
    });
}

async function loadPiece(piece) {
    if (!piece) return;

    currentPiece = piece;
    const resp = await fetch(`${API_BASE}/transform-space/${encodeURIComponent(piece)}`);
    const data = await resp.json();

    if (data.error) {
        alert(data.error);
        return;
    }

    nodes = data.nodes;
    edges = data.edges;
    pitchScale = data.pitch_scale;
    timeScale = data.time_scale;

    // Update stats
    document.getElementById('stat-nodes').textContent = nodes.length;
    document.getElementById('stat-edges').textContent = edges.length;
    document.getElementById('stat-tracks').textContent = data.track_count;

    // Build track legend
    const tracks = [...new Set(nodes.map(n => n.track_id))].sort((a,b) => a-b);
    const legend = document.getElementById('track-legend');
    legend.innerHTML = tracks.map(t => `
        <div class="track-color">
            <div class="track-dot" style="background:${TRACK_COLORS[t % TRACK_COLORS.length]}"></div>
            Track ${t}
        </div>
    `).join('');

    // Build track volumes UI
    updateTrackVolumesUI();

    // Center view on nodes
    if (nodes.length > 0) {
        const avgX = nodes.reduce((s, n) => s + n.x, 0) / nodes.length;
        const avgY = nodes.reduce((s, n) => s + n.y, 0) / nodes.length;
        viewX = avgX;
        viewY = avgY;
    }

    render();
}

// Event handlers
document.getElementById('piece-select').addEventListener('change', e => {
    loadPiece(e.target.value);
});

canvas.addEventListener('mousedown', e => {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const world = screenToWorld(sx, sy);

    // Check if clicking a node
    for (const node of nodes) {
        const dx = world.x - node.x;
        const dy = world.y - node.y;
        const radius = (8 + node.note_count * 2) / zoom;
        if (dx*dx + dy*dy < radius*radius) {
            selectedNode = node;
            selectedEdge = null;
            draggingNode = node;
            dragStartX = node.x;
            dragStartY = node.y;
            showNodeDetail(node);
            render();
            return;
        }
    }

    // Check if clicking an edge
    for (const edge of edges) {
        const sourceNode = nodes.find(n => n.id === edge.source);
        const targetNode = nodes.find(n => n.id === edge.target);
        if (!sourceNode || !targetNode) continue;

        // Point-to-line distance
        const x1 = sourceNode.x, y1 = sourceNode.y;
        const x2 = targetNode.x, y2 = targetNode.y;
        const px = world.x, py = world.y;

        const dx = x2 - x1;
        const dy = y2 - y1;
        const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
        const nearX = x1 + t * dx;
        const nearY = y1 + t * dy;
        const dist = Math.sqrt((px - nearX) * (px - nearX) + (py - nearY) * (py - nearY));

        if (dist < 10 / zoom) {
            selectedEdge = edge;
            selectedNode = null;
            showEdgeDetail(edge);
            render();
            return;
        }
    }

    // Pan
    isPanning = true;
    panStartX = e.clientX;
    panStartY = e.clientY;
    selectedNode = null;
    selectedEdge = null;
    document.getElementById('node-detail').classList.remove('active');
    document.getElementById('edge-detail').classList.remove('active');
});

canvas.addEventListener('mousemove', e => {
    if (draggingNode) {
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const world = screenToWorld(sx, sy);

        draggingNode.x = world.x;
        draggingNode.y = world.y;
        render();
    } else if (isPanning) {
        const dx = e.clientX - panStartX;
        const dy = e.clientY - panStartY;
        viewX -= dx / zoom;
        viewY -= dy / zoom;
        panStartX = e.clientX;
        panStartY = e.clientY;
        render();
    }
});

canvas.addEventListener('mouseup', e => {
    if (draggingNode) {
        // Calculate new transform based on position change
        const dx = draggingNode.x - dragStartX;
        const dy = draggingNode.y - dragStartY;

        if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
            // Find incoming edge to this node
            const incomingEdge = edges.find(e => e.target === draggingNode.id);
            if (incomingEdge) {
                // Convert position delta to transform delta
                const pitchDelta = Math.round(dx / pitchScale);
                const timeDelta = Math.round(dy / timeScale);

                // Update edge display
                selectedEdge = incomingEdge;
                showEdgeDetail(incomingEdge, pitchDelta, timeDelta);
            }
        }

        draggingNode = null;
    }
    isPanning = false;
});

canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    zoom *= zoomFactor;
    zoom = Math.max(0.1, Math.min(5, zoom));
    render();
});

function showNodeDetail(node) {
    document.getElementById('edge-detail').classList.remove('active');
    document.getElementById('node-detail').classList.add('active');

    document.getElementById('node-id').textContent = 'P' + node.pattern_id;
    document.getElementById('node-note-count').textContent = node.note_count;
    document.getElementById('node-track').textContent = node.track_id;

    const pitchDiv = document.getElementById('node-pitches');
    pitchDiv.innerHTML = node.pitch_classes.slice(0, 12).map(pc =>
        `<div class="pitch-note" style="background:${NOTE_COLORS[pc]}">${NOTE_NAMES[pc]}</div>`
    ).join('') + (node.pitch_classes.length > 12 ? '...' : '');
}

function showEdgeDetail(edge, suggestedPitchDelta = 0, suggestedTimeDelta = 0) {
    document.getElementById('node-detail').classList.remove('active');
    document.getElementById('edge-detail').classList.add('active');

    document.getElementById('edge-id').textContent = edge.id;
    document.getElementById('edge-transform').textContent = edge.transform;

    // Set sliders based on current + suggested delta
    const currentPitch = edge.dx || 0;
    const currentTime = edge.dy || 0;

    let newPitch = Math.round(currentPitch + suggestedPitchDelta) % 12;
    if (newPitch < 0) newPitch += 12;

    // Find closest tau value
    const newTimeRaw = currentTime + suggestedTimeDelta;
    let tauIndex = 2; // Default to 480
    let minDist = Infinity;
    TAU_VALUES.forEach((v, i) => {
        const dist = Math.abs(newTimeRaw - v);
        if (dist < minDist) {
            minDist = dist;
            tauIndex = i;
        }
    });

    document.getElementById('pitch-slider').value = newPitch;
    document.getElementById('pitch-value').textContent = 'T' + newPitch;

    document.getElementById('time-slider').value = tauIndex;
    document.getElementById('time-value').textContent = TAU_VALUES[tauIndex];
}

document.getElementById('pitch-slider').addEventListener('input', e => {
    document.getElementById('pitch-value').textContent = 'T' + e.target.value;
});

document.getElementById('time-slider').addEventListener('input', e => {
    document.getElementById('time-value').textContent = TAU_VALUES[e.target.value];
});

async function applyTransformEdit() {
    if (!selectedEdge) return;

    const pitch = document.getElementById('pitch-slider').value;
    const tauIndex = document.getElementById('time-slider').value;
    const tau = TAU_VALUES[tauIndex];

    const newTransform = `T${pitch}:tau${tau}`;

    const resp = await fetch(`${API_BASE}/edit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ edge_id: selectedEdge.id, transform: newTransform })
    });

    const result = await resp.json();
    if (result.success) {
        // Update local edge data (don't reload - that would overwrite local edits)
        selectedEdge.transform = newTransform;
        selectedEdge.dx = parseInt(pitch);
        selectedEdge.dy = tau;

        // Update UI
        document.getElementById('edge-transform').textContent = newTransform;
        document.getElementById('save-status').textContent = 'Unsaved changes';
        document.getElementById('save-status').style.color = '#f59e0b';

        // Re-render to show updated position
        render();
    }
}

// Save all edits to checkpoint
async function saveEdits() {
    document.getElementById('save-status').textContent = 'Saving...';
    document.getElementById('save-status').style.color = '#888';

    try {
        const resp = await fetch(`${API_BASE}/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const result = await resp.json();
        if (result.success) {
            document.getElementById('save-status').textContent =
                `Saved ${result.saved.edges} edge edits, ${result.saved.patterns} pattern edits`;
            document.getElementById('save-status').style.color = '#10b981';
        } else {
            document.getElementById('save-status').textContent = 'Error: ' + (result.error || 'Unknown');
            document.getElementById('save-status').style.color = '#ef4444';
        }
    } catch (err) {
        document.getElementById('save-status').textContent = 'Error: ' + err.message;
        document.getElementById('save-status').style.color = '#ef4444';
    }
}

// Audio playback
function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        masterGain = audioCtx.createGain();
        masterGain.connect(audioCtx.destination);
        masterGain.gain.value = 0.5;
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();
}

function getTrackGain(trackId) {
    if (!trackGains[trackId]) {
        trackGains[trackId] = audioCtx.createGain();
        trackGains[trackId].connect(masterGain);
        trackGains[trackId].gain.value = 1.0;
    }
    return trackGains[trackId];
}

function updateTrackVolumesUI() {
    const container = document.getElementById('track-volumes');
    if (!container) return;

    // Collect unique track_ids from nodes
    const trackIds = [...new Set(nodes.map(n => n.track_id))].sort((a, b) => a - b);

    container.innerHTML = trackIds.map((tid, i) => {
        const color = TRACK_COLORS[i % TRACK_COLORS.length];
        const name = trackInfo[tid]?.name || 'Track ' + tid;
        return '<div style="display:flex;align-items:center;gap:8px;margin:4px 0;font-size:11px;">' +
            '<div style="width:10px;height:10px;border-radius:50%;background:' + color + '"></div>' +
            '<span style="width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + name + '</span>' +
            '<input type="range" min="0" max="100" value="100" style="width:60px" ' +
            'oninput="setTrackVolume(' + tid + ', this.value)">' +
            '</div>';
    }).join('');
}

function setTrackVolume(trackId, value) {
    if (audioCtx && trackGains[trackId]) {
        trackGains[trackId].gain.setValueAtTime(value / 100, audioCtx.currentTime);
    }
}

// Tempo slider handler
document.getElementById('tempo-slider').addEventListener('input', e => {
    playbackTempo = parseInt(e.target.value);
    document.getElementById('tempo-value').textContent = playbackTempo + ' BPM';
});

// Volume slider handler
document.getElementById('volume-slider').addEventListener('input', e => {
    const vol = parseInt(e.target.value);
    document.getElementById('volume-value').textContent = vol + '%';
    if (masterGain) {
        masterGain.gain.setValueAtTime(vol / 100, audioCtx.currentTime);
    }
});

function midiToFreq(note) {
    return 440 * Math.pow(2, (note - 69) / 12);
}

async function playPiece() {
    if (!currentPiece) return;
    initAudio();
    stopPlayback();

    isPlaying = true;

    try {
        // Fetch actual playback data from server (includes edited patterns)
        const resp = await fetch(`${API_BASE}/playback/${encodeURIComponent(currentPiece)}`);
        const playbackData = await resp.json();

        if (playbackData.error) {
            alert('Error: ' + playbackData.error);
            isPlaying = false;
            return;
        }

        // Convert ticks to seconds using tempo slider
        const ticksPerBeat = playbackData.ticks_per_beat || 480;
        const ticksToSec = 60.0 / (playbackTempo * ticksPerBeat);
        const totalDuration = playbackData.duration_ticks * ticksToSec;

        // Flatten all events with track info
        const allEvents = [];
        for (const track of playbackData.tracks) {
            for (const event of track.events) {
                allEvents.push({
                    pitch: event.pitch,
                    time: event.time,
                    duration: event.duration,
                    velocity: event.velocity,
                    track_id: track.track_id,
                    is_drum: track.is_drum
                });
        }
        }
        allEvents.sort((a, b) => a.time - b.time);

        console.log(`Playing ${allEvents.length} notes at ${playbackTempo} BPM`);

        // JIT scheduling: only schedule notes within lookahead window
        const LOOKAHEAD_SEC = 2.0;
        const SCHEDULE_INTERVAL = 100;
        let nextEventIndex = 0;
        const playbackStartTime = audioCtx.currentTime + 0.1;

        function scheduleUpcoming() {
            if (!isPlaying) return;
            const currentTime = audioCtx.currentTime - playbackStartTime;
            const scheduleUntil = currentTime + LOOKAHEAD_SEC;

            while (nextEventIndex < allEvents.length) {
                const event = allEvents[nextEventIndex];
                const eventTimeSec = event.time * ticksToSec;
                if (eventTimeSec > scheduleUntil) break;

                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.type = event.is_drum ? 'square' : 'sine';
                osc.frequency.value = event.is_drum ? 1000 : midiToFreq(event.pitch);
                const vol = (event.velocity / 127) * (event.is_drum ? 0.3 : 0.2);
                const t = playbackStartTime + eventTimeSec;
                const dur = event.is_drum ? 0.02 : Math.max(0.05, event.duration * ticksToSec);
                gain.gain.setValueAtTime(0, t);
                gain.gain.linearRampToValueAtTime(vol, t + 0.01);
                if (event.is_drum) {
                    gain.gain.exponentialRampToValueAtTime(0.001, t + dur);
                } else {
                    gain.gain.setValueAtTime(vol, t + dur - 0.02);
                    gain.gain.linearRampToValueAtTime(0, t + dur);
                }
                osc.connect(gain);
                // Route through track-specific gain for per-track volume control
                gain.connect(getTrackGain(event.track_id));
                osc.start(t);
                osc.stop(t + dur + 0.05);
                scheduledNodes.push(osc);
                nextEventIndex++;
            }

            if (nextEventIndex < allEvents.length && isPlaying) {
                setTimeout(scheduleUpcoming, SCHEDULE_INTERVAL);
            }
        }

        scheduleUpcoming();
        document.getElementById('play-btn').disabled = true;
        document.getElementById('stop-btn').disabled = false;

    } catch (err) {
        console.error('Playback error:', err);
        alert('Playback error: ' + err.message);
        isPlaying = false;
    }
}

function stopPlayback() {
    isPlaying = false;
    for (const node of scheduledNodes) {
        try { node.stop(); } catch(e) {}
    }
    scheduledNodes = [];
    document.getElementById('play-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
}

// Init
window.addEventListener('resize', resizeCanvas);
resizeCanvas();
loadPieces();
</script>
</body>
</html>'''
        self.send_html(html)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Genome Graph Server')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--dag', help='Path to DAG checkpoint (optional)')
    parser.add_argument('--v24', help='Path to original v24 checkpoint (for full playback reconstruction)')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    args = parser.parse_args()

    # Load graph
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        graph = GenomeGraph.from_checkpoint(args.checkpoint)
        print(f"Loaded genome graph format")
    except ValueError:
        print(f"Converting v24 checkpoint to genome graph...")
        graph = convert_v24_to_genome_graph(args.checkpoint)

    GenomeGraphHandler.graph = graph
    # Store checkpoint path for save functionality
    GenomeGraphHandler.graph._checkpoint_path = os.path.abspath(args.checkpoint)

    stats = graph.stats()
    print(f"Graph loaded: {stats['n_patterns']} patterns, {stats['n_edges']} edges")

    # Load DAG if provided
    if args.dag:
        print(f"\nLoading DAG: {args.dag}")
        try:
            dag = MusicDAG.from_checkpoint(args.dag)
            GenomeGraphHandler.dag = dag
            dag_stats = dag.get_stats()
            print(f"DAG loaded: {dag_stats['node_count']} nodes")
            print(f"  Patterns: {dag_stats['by_type'].get('pattern', 0)}")
            print(f"  Sequences: {dag_stats['by_type'].get('seq', 0)}")
            print(f"  Transforms: {dag_stats['by_type'].get('transform', 0)}")
        except Exception as e:
            print(f"Warning: Could not load DAG: {e}")

    # Load v24 rules if provided (for playback reconstruction with full occurrences)
    # Also check if the main checkpoint has patterns_json (v33 format)
    if args.v24:
        print(f"\nLoading v24 rules from: {args.v24}")
        try:
            v24_data = np.load(args.v24, allow_pickle=True)
            rules_json = v24_data['grammar_rules_json'][0]
            v24_rules = json.loads(rules_json)
            GenomeGraphHandler.v24_rules = v24_rules
            total_occs = sum(len(r.get('occurrences', [])) for r in v24_rules.values())
            print(f"Loaded {len(v24_rules)} rules with {total_occs} total occurrences")
        except Exception as e:
            print(f"Warning: Could not load v24 rules: {e}")

    # Try to load patterns from main checkpoint if v24_rules not loaded
    if GenomeGraphHandler.v24_rules is None:
        try:
            main_data = np.load(args.checkpoint, allow_pickle=True)
            checkpoint_dir = os.path.dirname(args.checkpoint)

            # v4 format: patterns stored in external JSON file
            if 'patterns_json_file' in main_data:
                patterns_file = main_data['patterns_json_file'].item()
                patterns_path = os.path.join(checkpoint_dir, patterns_file)
                with open(patterns_path, 'r') as f:
                    v24_rules = json.load(f)
                GenomeGraphHandler.v24_rules = v24_rules
                total_occs = sum(len(r.get('occurrences', [])) for r in v24_rules.values())
                print(f"Loaded {len(v24_rules)} patterns from v4 external format with {total_occs} total occurrences")
            # Try v33 format (patterns_json inline)
            elif 'patterns_json' in main_data:
                patterns_json = main_data['patterns_json']
                if hasattr(patterns_json, 'item'):
                    v24_rules = json.loads(str(patterns_json.item()))
                else:
                    v24_rules = json.loads(str(patterns_json))
                GenomeGraphHandler.v24_rules = v24_rules
                total_occs = sum(len(r.get('occurrences', [])) for r in v24_rules.values())
                print(f"Loaded {len(v24_rules)} patterns from v33 format with {total_occs} total occurrences")
            # Fallback to grammar_rules_json
            elif 'grammar_rules_json' in main_data:
                rules_json = main_data['grammar_rules_json'][0]
                v24_rules = json.loads(rules_json)
                GenomeGraphHandler.v24_rules = v24_rules
                total_occs = sum(len(r.get('occurrences', [])) for r in v24_rules.values())
                print(f"Loaded {len(v24_rules)} rules with {total_occs} total occurrences")

            # Load track_info for drum detection (v4 format: external file)
            if 'track_info_json_file' in main_data:
                track_info_file = main_data['track_info_json_file'].item()
                track_info_path = os.path.join(checkpoint_dir, track_info_file)
                with open(track_info_path, 'r') as f:
                    track_info = json.load(f)
                GenomeGraphHandler.track_info = track_info
                drum_count = sum(1 for ti in track_info if ti.get('is_drum', False))
                print(f"Loaded track info: {len(track_info)} tracks ({drum_count} drums)")
            # v33 format: inline
            elif 'track_info_json' in main_data:
                track_info_json = main_data['track_info_json']
                if hasattr(track_info_json, 'item'):
                    track_info = json.loads(str(track_info_json.item()))
                else:
                    track_info = json.loads(str(track_info_json))
                GenomeGraphHandler.track_info = track_info
                drum_count = sum(1 for ti in track_info if ti.get('is_drum', False))
                print(f"Loaded track info: {len(track_info)} tracks ({drum_count} drums)")

            # v43: Load multi-factor transforms (τ, v, d)
            if 'multi_factor_json_file' in main_data:
                mf_file = main_data['multi_factor_json_file'].item()
                mf_path = os.path.join(checkpoint_dir, mf_file)
                if os.path.exists(mf_path):
                    with open(mf_path, 'r') as f:
                        GenomeGraphHandler.multi_factor = json.load(f)
                    # Handle both formats: list or int count
                    def get_count(key):
                        v = GenomeGraphHandler.multi_factor.get(key, 0)
                        return len(v) if isinstance(v, list) else (v if isinstance(v, int) else 0)
                    n_tau = get_count('rhythm_transforms')
                    n_vel = get_count('velocity_transforms')
                    n_dur = get_count('duration_transforms')
                    print(f"Loaded multi-factor transforms: τ={n_tau}, v={n_vel}, d={n_dur}")

            # v43: Load track derives (cross-track arrangements)
            if 'track_derives_json_file' in main_data:
                td_file = main_data['track_derives_json_file'].item()
                td_path = os.path.join(checkpoint_dir, td_file)
                if os.path.exists(td_path):
                    with open(td_path, 'r') as f:
                        td_data = json.load(f)
                    # Handle both formats: list or {"derives_json": [...]}
                    if isinstance(td_data, dict) and 'derives_json' in td_data:
                        GenomeGraphHandler.track_derives = td_data['derives_json']
                    else:
                        GenomeGraphHandler.track_derives = td_data
                    print(f"Loaded {len(GenomeGraphHandler.track_derives)} track derive relations")

            # v43: Load feature importance (MDL-discovered useful features)
            if 'feature_importance_json_file' in main_data:
                fi_file = main_data['feature_importance_json_file'].item()
                fi_path = os.path.join(checkpoint_dir, fi_file)
                if os.path.exists(fi_path):
                    with open(fi_path, 'r') as f:
                        GenomeGraphHandler.feature_importance = json.load(f)
                    useful = GenomeGraphHandler.feature_importance.get('useful_features', [])
                    print(f"Loaded feature importance: {', '.join(useful) if useful else 'none discovered'}")

            # Load checkpoint stats for /api/stats
            stats = {}
            for key in ['n_files', 'n_tracks', 'n_notes', 'n_patterns', 'n_transform_vocabulary',
                        'n_factor_vocabulary', 'n_rhythm_transforms', 'n_velocity_transforms',
                        'n_duration_transforms', 'n_track_derives']:
                if key in main_data:
                    stats[key] = int(main_data[key].item())
            if stats:
                GenomeGraphHandler.checkpoint_stats = stats

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Warning: Could not load patterns from main checkpoint: {e}")

    # Start server
    server = HTTPServer(('0.0.0.0', args.port), GenomeGraphHandler)
    print(f"\nGenome Graph Editor running at http://localhost:{args.port}")
    if GenomeGraphHandler.dag:
        print(f"DAG Editor available at http://localhost:{args.port}/dag")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
