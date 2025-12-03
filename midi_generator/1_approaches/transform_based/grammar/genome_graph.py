"""
Genome Graph: Musical DNA as a Transform Graph
===============================================

The fundamental insight: TIME IS A TRANSFORM, not metadata.

A musical genome is a directed multigraph where:
  - Nodes = Patterns (musical objects with pitch, rhythm, velocity)
  - Edges = Transforms connecting patterns
    - Pitch transforms: T0-T11, I0-I11, R, compounds
    - Time transforms: τ_n (shift by n ticks)
    - Cross-track: Same transforms, different edge category

A piece is a subgraph with temporal edges forming the timeline.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Iterator, Any
from collections import defaultdict
from itertools import groupby


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Pattern:
    """A musical pattern - node in the genome graph."""
    id: int
    pitch_classes: List[int]
    octaves: List[int]
    velocities: List[int]
    durations: List[int]
    rhythm_ioi: List[int] = field(default_factory=list)  # Inter-onset intervals

    @property
    def length(self) -> int:
        return len(self.pitch_classes)

    def to_dict(self) -> Dict:
        # Convert all values to native Python types for JSON serialization
        def to_native(x):
            if hasattr(x, 'tolist'):
                return x.tolist()
            if isinstance(x, (list, tuple)):
                return [int(v) if hasattr(v, 'item') else v for v in x]
            if hasattr(x, 'item'):
                return x.item()
            return x

        return {
            'id': int(self.id) if hasattr(self.id, 'item') else self.id,
            'pitch_classes': to_native(self.pitch_classes),
            'octaves': to_native(self.octaves),
            'velocities': to_native(self.velocities),
            'durations': to_native(self.durations),
            'rhythm_ioi': to_native(self.rhythm_ioi),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Pattern':
        return cls(
            id=d['id'],
            pitch_classes=d['pitch_classes'],
            octaves=d['octaves'],
            velocities=d['velocities'],
            durations=d['durations'],
            rhythm_ioi=d.get('rhythm_ioi', []),
        )


@dataclass
class Edge:
    """A transform edge connecting two patterns."""
    id: int
    source: int  # Pattern ID
    target: int  # Pattern ID
    transform: str  # "T5", "τ480", "I7∘R", "T5∘τ480"
    edge_type: str  # "temporal", "harmonic", "cross-track", "derived"
    piece_id: Optional[str] = None
    track_id: Optional[int] = None
    source_onset: Optional[int] = None  # Absolute onset time for reconstruction

    def to_dict(self) -> Dict:
        # Convert numpy types to native Python types for JSON serialization
        def to_native(x):
            if x is None:
                return None
            if hasattr(x, 'item'):
                return x.item()
            return x

        d = {
            'id': to_native(self.id),
            'source': to_native(self.source),
            'target': to_native(self.target),
            'transform': self.transform,
            'edge_type': self.edge_type,
            'piece_id': self.piece_id,
            'track_id': to_native(self.track_id),
        }
        if self.source_onset is not None:
            d['source_onset'] = to_native(self.source_onset)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'Edge':
        return cls(
            id=d['id'],
            source=d['source'],
            target=d['target'],
            transform=d['transform'],
            edge_type=d['edge_type'],
            piece_id=d.get('piece_id'),
            track_id=d.get('track_id'),
            source_onset=d.get('source_onset'),
        )

    @property
    def is_temporal(self) -> bool:
        return 'τ' in self.transform

    @property
    def is_pitch_transform(self) -> bool:
        return any(t in self.transform for t in ['T', 'I', 'R'])

    @property
    def is_compound(self) -> bool:
        return '∘' in self.transform


# ============================================================================
# Transform Operations
# ============================================================================

def parse_compound_transform(transform: str) -> List[str]:
    """Parse compound transform into components.

    "T5∘τ480" -> ["T5", "τ480"]
    "I7∘R∘T3" -> ["I7", "R", "T3"]
    """
    return transform.split('∘')


def compose_transforms(transforms: List[str]) -> str:
    """Compose list of transforms into compound.

    ["T5", "τ480"] -> "T5∘τ480"
    """
    return '∘'.join(transforms)


def apply_pitch_transform(pitch_classes: List[int], transform: str) -> List[int]:
    """Apply pitch-class transform.

    T_n: transpose by n semitones (mod 12)
    I_n: invert around axis n
    R: retrograde (reverse)
    """
    if transform == 'identity' or transform == 'T0':
        return pitch_classes[:]

    if transform.startswith('T'):
        n = int(transform[1:])
        return [(p + n) % 12 for p in pitch_classes]

    if transform.startswith('I'):
        n = int(transform[1:])
        return [(2 * n - p) % 12 for p in pitch_classes]

    if transform == 'R':
        return pitch_classes[::-1]

    raise ValueError(f"Unknown pitch transform: {transform}")


def find_pitch_transform(source_pc: List[int], target_pc: List[int]) -> str:
    """Find the pitch transform that maps source to target."""
    if len(source_pc) != len(target_pc):
        return 'none'  # Different lengths, no simple transform

    if source_pc == target_pc:
        return 'identity'

    # Check transposition
    if len(source_pc) > 0:
        delta = (target_pc[0] - source_pc[0]) % 12
        if all((s + delta) % 12 == t for s, t in zip(source_pc, target_pc)):
            return f'T{delta}'

    # Check retrograde
    if source_pc[::-1] == target_pc:
        return 'R'

    # Check inversion (around each possible axis)
    for axis in range(12):
        if all((2 * axis - s) % 12 == t for s, t in zip(source_pc, target_pc)):
            return f'I{axis}'

    # Check retrograde inversion
    for axis in range(12):
        if all((2 * axis - s) % 12 == t for s, t in zip(source_pc[::-1], target_pc)):
            return f'I{axis}∘R'

    return 'none'


# Standard τ grid: 1/4 beat, 1/2 beat, 1 beat, 2 beats, 4 beats, 8 beats (at 480 ticks/beat)
TAU_GRID = [120, 240, 480, 960, 1920, 3840]

def quantize_time_delta(delta: int, resolution: int = None) -> int:
    """Quantize time delta to nearest grid value from TAU_GRID.

    This reduces thousands of unique τ values to 6 standard intervals:
    τ120, τ240, τ480, τ960, τ1920, τ3840

    Combined with T0-T11, I0-I11, and R, gives 31 total edge types.

    Args:
        delta: Raw time delta in ticks
        resolution: Ignored (kept for backward compatibility)

    Returns:
        Quantized delta from TAU_GRID
    """
    if delta <= 0:
        return 0
    # Find nearest grid value
    best = TAU_GRID[0]
    best_dist = abs(delta - best)
    for grid_val in TAU_GRID[1:]:
        dist = abs(delta - grid_val)
        if dist < best_dist:
            best = grid_val
            best_dist = dist
        elif grid_val > delta * 2:
            break  # Grid values only increase, early exit
    return best


# ============================================================================
# GenomeGraph Class
# ============================================================================

class GenomeGraph:
    """
    Musical genome as a directed multigraph.

    Nodes are patterns, edges are transforms.
    Time is treated as a transform (τ), not metadata.
    """

    def __init__(self):
        self.patterns: Dict[int, Pattern] = {}
        self.edges: Dict[int, Edge] = {}
        self._next_pattern_id = 0
        self._next_edge_id = 0

        # Indices for fast lookup
        self._edges_by_source: Dict[int, List[int]] = defaultdict(list)
        self._edges_by_target: Dict[int, List[int]] = defaultdict(list)
        self._edges_by_transform: Dict[str, List[int]] = defaultdict(list)
        self._edges_by_piece: Dict[str, List[int]] = defaultdict(list)

        # Transform vocabulary
        self.transform_vocab: List[str] = []

        # Meta-patterns (Level 3 grammar)
        self.meta_patterns: List[Dict] = []

        # Occurrences: pattern_id -> list of {piece_id, track_id, onset_time}
        # This is needed for full reconstruction (edges only capture consecutive pairs)
        self.occurrences: Dict[int, List[Dict]] = defaultdict(list)

    # ---------- Basic Operations ----------

    def add_pattern(self, pattern: Pattern) -> int:
        """Add a pattern to the graph. Returns pattern ID."""
        if pattern.id is None:
            pattern.id = self._next_pattern_id
            self._next_pattern_id += 1
        else:
            self._next_pattern_id = max(self._next_pattern_id, pattern.id + 1)

        self.patterns[pattern.id] = pattern
        return pattern.id

    def add_edge(self, source: int, target: int, transform: str,
                 edge_type: str = 'derived',
                 piece_id: str = None, track_id: int = None,
                 source_onset: int = None) -> int:
        """Add an edge to the graph. Returns edge ID."""
        edge_id = self._next_edge_id
        self._next_edge_id += 1

        edge = Edge(
            id=edge_id,
            source=source,
            target=target,
            transform=transform,
            edge_type=edge_type,
            piece_id=piece_id,
            track_id=track_id,
            source_onset=source_onset,
        )

        self.edges[edge_id] = edge
        self._edges_by_source[source].append(edge_id)
        self._edges_by_target[target].append(edge_id)
        self._edges_by_transform[transform].append(edge_id)
        if piece_id:
            self._edges_by_piece[piece_id].append(edge_id)

        return edge_id

    def remove_edge(self, edge_id: int):
        """Remove an edge from the graph."""
        if edge_id not in self.edges:
            return

        edge = self.edges[edge_id]
        self._edges_by_source[edge.source].remove(edge_id)
        self._edges_by_target[edge.target].remove(edge_id)
        self._edges_by_transform[edge.transform].remove(edge_id)
        if edge.piece_id:
            self._edges_by_piece[edge.piece_id].remove(edge_id)

        del self.edges[edge_id]

    def get_pattern(self, pattern_id: int) -> Optional[Pattern]:
        """Get a pattern by ID."""
        return self.patterns.get(pattern_id)

    def get_edge(self, edge_id: int) -> Optional[Edge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)

    # ---------- Graph Queries ----------

    def get_edges_from(self, pattern_id: int,
                       transform_filter: str = None) -> List[Edge]:
        """Get all edges originating from a pattern."""
        edge_ids = self._edges_by_source.get(pattern_id, [])
        edges = [self.edges[eid] for eid in edge_ids]

        if transform_filter:
            edges = [e for e in edges if transform_filter in e.transform]

        return edges

    def get_edges_to(self, pattern_id: int,
                     transform_filter: str = None) -> List[Edge]:
        """Get all edges pointing to a pattern."""
        edge_ids = self._edges_by_target.get(pattern_id, [])
        edges = [self.edges[eid] for eid in edge_ids]

        if transform_filter:
            edges = [e for e in edges if transform_filter in e.transform]

        return edges

    def get_reachable(self, pattern_id: int,
                      transform_filter: str = None,
                      max_depth: int = 1) -> Set[int]:
        """Get all patterns reachable from a pattern via transforms.

        Args:
            pattern_id: Starting pattern
            transform_filter: Only follow edges containing this string
            max_depth: Maximum hop distance

        Returns:
            Set of reachable pattern IDs
        """
        visited = set()
        frontier = {pattern_id}

        for _ in range(max_depth):
            next_frontier = set()
            for pid in frontier:
                if pid in visited:
                    continue
                visited.add(pid)

                for edge in self.get_edges_from(pid, transform_filter):
                    if edge.target not in visited:
                        next_frontier.add(edge.target)

            frontier = next_frontier
            if not frontier:
                break

        visited.discard(pattern_id)  # Don't include starting pattern
        return visited

    def get_piece_subgraph(self, piece_id: str) -> 'GenomeGraph':
        """Extract subgraph for a single piece."""
        subgraph = GenomeGraph()

        edge_ids = self._edges_by_piece.get(piece_id, [])
        pattern_ids = set()

        for eid in edge_ids:
            edge = self.edges[eid]
            pattern_ids.add(edge.source)
            pattern_ids.add(edge.target)

        for pid in pattern_ids:
            subgraph.add_pattern(self.patterns[pid])

        for eid in edge_ids:
            edge = self.edges[eid]
            subgraph.add_edge(
                edge.source, edge.target, edge.transform,
                edge.edge_type, edge.piece_id, edge.track_id
            )

        return subgraph

    # ---------- Transform Operations ----------

    def apply_transform(self, pattern_id: int, transform: str) -> int:
        """Apply a transform to a pattern, return result pattern ID.

        If transform already exists in graph, returns existing target.
        Otherwise computes new pattern and adds it.
        """
        # Check if this transform already exists
        for edge in self.get_edges_from(pattern_id):
            if edge.transform == transform:
                return edge.target

        # Compute new pattern
        source = self.patterns[pattern_id]

        # Parse compound transform
        components = parse_compound_transform(transform)

        # Apply pitch transforms (skip τ for now)
        new_pc = source.pitch_classes[:]
        for comp in components:
            if not comp.startswith('τ'):
                new_pc = apply_pitch_transform(new_pc, comp)

        # Create new pattern
        new_pattern = Pattern(
            id=None,
            pitch_classes=new_pc,
            octaves=source.octaves[:],  # Keep octaves
            velocities=source.velocities[:],
            durations=source.durations[:],
            rhythm_ioi=source.rhythm_ioi[:],
        )

        new_id = self.add_pattern(new_pattern)
        self.add_edge(pattern_id, new_id, transform, 'derived')

        return new_id

    def factor_edge(self, edge_id: int) -> List[int]:
        """Factor a compound edge into atomic edges.

        "T5∘τ480" becomes T5 edge + τ480 edge with intermediate node.

        Returns list of new edge IDs.
        """
        edge = self.edges.get(edge_id)
        if not edge or not edge.is_compound:
            return [edge_id]

        components = parse_compound_transform(edge.transform)
        if len(components) <= 1:
            return [edge_id]

        # Create chain of edges
        current_source = edge.source
        new_edge_ids = []

        for i, comp in enumerate(components[:-1]):
            # Create intermediate pattern by applying transform
            intermediate_id = self.apply_transform(current_source, comp)
            new_edge_ids.append(
                self.add_edge(current_source, intermediate_id, comp,
                             edge.edge_type, edge.piece_id, edge.track_id)
            )
            current_source = intermediate_id

        # Final edge to original target
        new_edge_ids.append(
            self.add_edge(current_source, edge.target, components[-1],
                         edge.edge_type, edge.piece_id, edge.track_id)
        )

        # Remove original compound edge
        self.remove_edge(edge_id)

        return new_edge_ids

    def entangle_path(self, edge_ids: List[int]) -> int:
        """Entangle a path of edges into a single compound edge.

        Returns new compound edge ID.
        """
        if not edge_ids:
            return -1

        edges = [self.edges[eid] for eid in edge_ids]

        # Verify path is connected
        for i in range(len(edges) - 1):
            if edges[i].target != edges[i + 1].source:
                raise ValueError("Edges do not form a connected path")

        # Compose transforms
        compound = compose_transforms([e.transform for e in edges])

        # Create compound edge
        new_edge_id = self.add_edge(
            edges[0].source, edges[-1].target, compound,
            edges[0].edge_type, edges[0].piece_id, edges[0].track_id
        )

        # Remove original edges
        for eid in edge_ids:
            self.remove_edge(eid)

        return new_edge_id

    def clone_subgraph(self, root_id: int, transform: str,
                       max_depth: int = 10) -> int:
        """Clone a subgraph with transform applied to all patterns.

        Returns new root pattern ID.
        """
        # BFS to find all patterns in subgraph
        visited = set()
        frontier = {root_id}
        subgraph_edges = []

        for _ in range(max_depth):
            next_frontier = set()
            for pid in frontier:
                if pid in visited:
                    continue
                visited.add(pid)

                for edge in self.get_edges_from(pid):
                    subgraph_edges.append(edge)
                    if edge.target not in visited:
                        next_frontier.add(edge.target)

            frontier = next_frontier
            if not frontier:
                break

        # Create transformed copies
        old_to_new = {}
        for pid in visited:
            new_id = self.apply_transform(pid, transform)
            old_to_new[pid] = new_id

        # Recreate edges in new subgraph
        for edge in subgraph_edges:
            if edge.source in old_to_new and edge.target in old_to_new:
                self.add_edge(
                    old_to_new[edge.source],
                    old_to_new[edge.target],
                    edge.transform,
                    'derived'
                )

        return old_to_new[root_id]

    # ---------- Serialization ----------

    def to_dict(self) -> Dict:
        """Export graph to dictionary for JSON serialization."""
        # Convert all keys to str for JSON compatibility (numpy.int64 not supported)
        return {
            'patterns': {str(pid): p.to_dict() for pid, p in self.patterns.items()},
            'edges': [e.to_dict() for e in self.edges.values()],
            'transform_vocab': self.transform_vocab,
            'meta_patterns': self.meta_patterns,
            'occurrences': {str(k): v for k, v in self.occurrences.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'GenomeGraph':
        """Create graph from dictionary."""
        graph = cls()

        for pid_str, pdata in d.get('patterns', {}).items():
            pattern = Pattern.from_dict(pdata)
            graph.add_pattern(pattern)

        for edata in d.get('edges', []):
            edge = Edge.from_dict(edata)
            graph.edges[edge.id] = edge
            graph._next_edge_id = max(graph._next_edge_id, edge.id + 1)
            graph._edges_by_source[edge.source].append(edge.id)
            graph._edges_by_target[edge.target].append(edge.id)
            graph._edges_by_transform[edge.transform].append(edge.id)
            if edge.piece_id:
                graph._edges_by_piece[edge.piece_id].append(edge.id)

        graph.transform_vocab = d.get('transform_vocab', [])
        graph.meta_patterns = d.get('meta_patterns', [])

        # Load occurrences
        for pid_str, occ_list in d.get('occurrences', {}).items():
            graph.occurrences[int(pid_str)] = occ_list

        return graph

    def to_checkpoint(self, path: str):
        """Save graph to numpy checkpoint format."""
        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        data = self.to_dict()
        np.savez(
            path,
            graph_json=np.array([json.dumps(data, cls=NumpyEncoder)]),
            is_genome_graph=np.array([True]),
            version=np.array(['v25']),
        )

    @classmethod
    def from_checkpoint(cls, path: str) -> 'GenomeGraph':
        """Load graph from checkpoint."""
        data = np.load(path, allow_pickle=True)

        if 'graph_json' in data:
            graph_data = json.loads(str(data['graph_json'][0]))
            return cls.from_dict(graph_data)

        raise ValueError("Checkpoint does not contain genome graph data")

    # ---------- Cytoscape Export ----------

    def to_cytoscape(self, piece_filter: str = None) -> Dict:
        """Export to Cytoscape.js format for web visualization."""
        nodes = []
        edges_out = []

        # Filter patterns if piece specified
        if piece_filter:
            edge_ids = self._edges_by_piece.get(piece_filter, [])
            pattern_ids = set()
            for eid in edge_ids:
                edge = self.edges[eid]
                pattern_ids.add(edge.source)
                pattern_ids.add(edge.target)
        else:
            pattern_ids = set(self.patterns.keys())
            edge_ids = list(self.edges.keys())

        for pid in pattern_ids:
            p = self.patterns[pid]
            nodes.append({
                'data': {
                    'id': f'P{pid}',
                    'label': f'P{pid}',
                    'length': p.length,
                    'pitch_classes': p.pitch_classes,
                }
            })

        for eid in edge_ids:
            e = self.edges[eid]
            # Determine edge color by edge type and transform
            if e.edge_type == 'cross-track':
                color = '#E91E63'  # Pink for cross-track
                line_style = 'dashed'
            elif 'τ' in e.transform:
                color = '#4CAF50'  # Green for temporal
                line_style = 'solid'
            elif 'I' in e.transform:
                color = '#F44336'  # Red for inversion
                line_style = 'solid'
            elif 'R' in e.transform:
                color = '#9C27B0'  # Purple for retrograde
                line_style = 'solid'
            elif 'T' in e.transform:
                color = '#2196F3'  # Blue for transposition
                line_style = 'solid'
            else:
                color = '#9E9E9E'  # Gray for other
                line_style = 'solid'

            edges_out.append({
                'data': {
                    'id': f'E{eid}',
                    'source': f'P{e.source}',
                    'target': f'P{e.target}',
                    'transform': e.transform,
                    'edge_type': e.edge_type,
                    'color': color,
                    'line_style': line_style,
                }
            })

        return {
            'elements': {
                'nodes': nodes,
                'edges': edges_out,
            }
        }

    # ---------- Statistics ----------

    def stats(self) -> Dict:
        """Get graph statistics."""
        transform_counts = defaultdict(int)
        edge_type_counts = defaultdict(int)

        for e in self.edges.values():
            edge_type_counts[e.edge_type] += 1
            for comp in parse_compound_transform(e.transform):
                if comp.startswith('T'):
                    transform_counts['transposition'] += 1
                elif comp.startswith('I'):
                    transform_counts['inversion'] += 1
                elif comp == 'R':
                    transform_counts['retrograde'] += 1
                elif comp.startswith('τ'):
                    transform_counts['temporal'] += 1

        return {
            'n_patterns': len(self.patterns),
            'n_edges': len(self.edges),
            'n_pieces': len(self._edges_by_piece),
            'transform_counts': dict(transform_counts),
            'edge_type_counts': dict(edge_type_counts),
            'avg_degree': len(self.edges) * 2 / max(1, len(self.patterns)),
        }


# ============================================================================
# Convert from v24 checkpoint to GenomeGraph
# ============================================================================

def extract_temporal_edges(patterns: Dict, occurrences_by_pattern: Dict,
                           time_resolution: int = 120) -> List[Dict]:
    """Extract τ transform edges from occurrence data.

    Args:
        patterns: Dict of pattern_id -> pattern data
        occurrences_by_pattern: Dict of pattern_id -> list of occurrences
        time_resolution: Quantization resolution for τ values

    Returns:
        List of edge dicts ready for GenomeGraph
    """
    edges = []

    # Group all occurrences by (piece_id, track_id)
    all_occurrences = []
    for pid, occs in occurrences_by_pattern.items():
        for occ in occs:
            all_occurrences.append({
                'pattern_id': pid,
                'piece_id': occ.get('piece_id', occ[0] if isinstance(occ, tuple) else None),
                'track_id': occ.get('track_id', occ[1] if isinstance(occ, tuple) else None),
                'onset_time': occ.get('onset_time', occ[2] if isinstance(occ, tuple) else None),
            })

    # Sort by piece, track, time
    all_occurrences.sort(key=lambda x: (
        x['piece_id'] or '',
        x['track_id'] or 0,
        x['onset_time'] or 0
    ))

    # Group by (piece, track)
    for (piece_id, track_id), group in groupby(
        all_occurrences,
        key=lambda x: (x['piece_id'], x['track_id'])
    ):
        group_list = list(group)

        for i in range(len(group_list) - 1):
            curr = group_list[i]
            next_occ = group_list[i + 1]

            if curr['onset_time'] is None or next_occ['onset_time'] is None:
                continue

            # Compute time delta
            delta = next_occ['onset_time'] - curr['onset_time']
            delta_q = quantize_time_delta(delta, time_resolution)

            if delta_q <= 0:
                continue  # Skip simultaneous or backwards

            # Find pitch transform between patterns
            src_pc = patterns.get(curr['pattern_id'], {}).get('pitch_classes', [])
            tgt_pc = patterns.get(next_occ['pattern_id'], {}).get('pitch_classes', [])

            pitch_t = find_pitch_transform(src_pc, tgt_pc)

            # Create edge - entangled pitch and time
            if pitch_t != 'none' and pitch_t != 'identity':
                transform = f"{pitch_t}∘τ{delta_q}"
            else:
                transform = f"τ{delta_q}"

            edges.append({
                'source': curr['pattern_id'],
                'target': next_occ['pattern_id'],
                'transform': transform,
                'edge_type': 'temporal',
                'piece_id': piece_id,
                'track_id': track_id,
                'source_onset': curr['onset_time'],  # Absolute onset time for reconstruction
            })

    return edges


def extract_cross_track_edges(patterns: Dict, occurrences_by_pattern: Dict) -> List[Dict]:
    """Extract cross-track edges: patterns in different tracks related by T, I, R.

    For patterns occurring at the same time in different tracks, checks if
    one is a pitch transformation of the other (T5, I7, etc.).

    This captures relationships like "sax plays trumpet part transposed up a fifth"

    Args:
        patterns: Dict of pattern_id -> pattern data
        occurrences_by_pattern: Dict of pattern_id -> list of occurrences

    Returns:
        List of edge dicts with edge_type='cross-track'
    """
    from itertools import combinations

    edges = []

    # Group all occurrences by (piece_id, onset_time) to find simultaneous patterns
    patterns_by_time = defaultdict(list)
    for pid, occs in occurrences_by_pattern.items():
        for occ in occs:
            piece_id = occ.get('piece_id', occ[0] if isinstance(occ, tuple) else None)
            track_id = occ.get('track_id', occ[1] if isinstance(occ, tuple) else None)
            onset_time = occ.get('onset_time', occ[2] if isinstance(occ, tuple) else None)

            if piece_id is not None and onset_time is not None:
                patterns_by_time[(piece_id, onset_time)].append({
                    'pattern_id': pid,
                    'track_id': track_id,
                })

    # For each time point, compare patterns across tracks
    for (piece_id, onset_time), track_patterns in patterns_by_time.items():
        # Skip if only one track has a pattern at this time
        if len(track_patterns) < 2:
            continue

        # Get unique tracks at this time
        tracks = set(tp['track_id'] for tp in track_patterns)
        if len(tracks) < 2:
            continue

        # Compare all pairs of patterns from different tracks
        for tp_a, tp_b in combinations(track_patterns, 2):
            if tp_a['track_id'] == tp_b['track_id']:
                continue  # Same track, skip

            pid_a = tp_a['pattern_id']
            pid_b = tp_b['pattern_id']

            # Get pitch classes
            pcs_a = patterns.get(pid_a, {}).get('pitch_classes', [])
            pcs_b = patterns.get(pid_b, {}).get('pitch_classes', [])

            if not pcs_a or not pcs_b:
                continue

            # Check if related by transform (skip identity)
            transform = find_pitch_transform(pcs_a, pcs_b)
            if transform and transform not in ('none', 'identity'):
                edges.append({
                    'source': pid_a,
                    'target': pid_b,
                    'transform': transform,
                    'edge_type': 'cross-track',
                    'piece_id': piece_id,
                    'track_id': None,  # Cross-track has no single track
                    'source_track': tp_a['track_id'],
                    'target_track': tp_b['track_id'],
                })

    return edges


def convert_v24_to_genome_graph(checkpoint_path: str, include_cross_track: bool = True) -> GenomeGraph:
    """Convert a v24 factored checkpoint to GenomeGraph format.

    Args:
        checkpoint_path: Path to v24 .npz checkpoint
        include_cross_track: If True, extract cross-track edges (patterns related
                            by T, I, R across different tracks at same time)

    Returns:
        GenomeGraph instance
    """
    data = np.load(checkpoint_path, allow_pickle=True)

    graph = GenomeGraph()

    # Load patterns from grammar_rules_json or patterns_json (v33 format)
    rules = None
    if 'grammar_rules_json' in data:
        rules_json = data['grammar_rules_json']
        if hasattr(rules_json, '__len__') and len(rules_json) == 1:
            rules = json.loads(str(rules_json[0]))
        else:
            rules = json.loads(str(rules_json.item()))
    elif 'patterns_json' in data:
        # v33 format uses patterns_json
        patterns_json = data['patterns_json']
        if hasattr(patterns_json, 'item'):
            rules = json.loads(str(patterns_json.item()))
        else:
            rules = json.loads(str(patterns_json))
        print(f"Loaded {len(rules)} patterns from v33 patterns_json")

    # Process rules (from either grammar_rules_json or patterns_json)
    if rules:
        occurrences_by_pattern = {}

        for rule_id, rule_data in rules.items():
            if isinstance(rule_data, dict):
                pattern = Pattern(
                    id=int(rule_id),
                    pitch_classes=rule_data.get('pitch_classes', []),
                    octaves=rule_data.get('octaves', []),
                    velocities=rule_data.get('velocities', []),
                    durations=rule_data.get('duration_buckets',
                                           rule_data.get('durations', [])),
                )
                graph.add_pattern(pattern)

                # Store occurrences for temporal edge extraction AND reconstruction
                if 'occurrences' in rule_data:
                    occurrences_by_pattern[int(rule_id)] = rule_data['occurrences']
                    # Also store in graph for full reconstruction
                    graph.occurrences[int(rule_id)] = rule_data['occurrences']

        # Extract temporal edges
        temporal_edges = extract_temporal_edges(
            {pid: p.to_dict() for pid, p in graph.patterns.items()},
            occurrences_by_pattern
        )

        for e in temporal_edges:
            graph.add_edge(
                e['source'], e['target'], e['transform'],
                e['edge_type'], e['piece_id'], e['track_id'],
                e.get('source_onset')  # Pass onset time for reconstruction
            )

        # Extract cross-track edges if enabled
        if include_cross_track:
            cross_track_edges = extract_cross_track_edges(
                {pid: p.to_dict() for pid, p in graph.patterns.items()},
                occurrences_by_pattern
            )

            for e in cross_track_edges:
                graph.add_edge(
                    e['source'], e['target'], e['transform'],
                    e['edge_type'], e['piece_id'], e['track_id']
                )

            if cross_track_edges:
                print(f"  Extracted {len(cross_track_edges)} cross-track edges")

    # Load transform vocabulary
    if 'transform_vocabulary_json' in data:
        vocab_json = data['transform_vocabulary_json']
        if hasattr(vocab_json, '__len__') and len(vocab_json) == 1:
            graph.transform_vocab = json.loads(str(vocab_json[0]))
        else:
            graph.transform_vocab = json.loads(str(vocab_json.item()))

    # Load meta patterns
    if 'meta_patterns_json' in data:
        meta_json = data['meta_patterns_json']
        if hasattr(meta_json, '__len__') and len(meta_json) == 1:
            graph.meta_patterns = json.loads(str(meta_json[0]))
        else:
            graph.meta_patterns = json.loads(str(meta_json.item()))

    return graph


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python genome_graph.py <checkpoint.npz> [--convert]")
        print()
        print("Options:")
        print("  --convert   Convert v24 checkpoint to genome graph format")
        print("  --stats     Show graph statistics")
        print("  --cytoscape Export to cytoscape JSON")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    if '--convert' in sys.argv:
        print(f"Converting {checkpoint_path} to genome graph...")
        graph = convert_v24_to_genome_graph(checkpoint_path)

        output_path = checkpoint_path.replace('.npz', '_graph.npz')
        graph.to_checkpoint(output_path)

        print(f"Saved to {output_path}")
        print(f"Stats: {graph.stats()}")

    elif '--stats' in sys.argv:
        try:
            graph = GenomeGraph.from_checkpoint(checkpoint_path)
        except ValueError:
            graph = convert_v24_to_genome_graph(checkpoint_path)

        stats = graph.stats()
        print(f"=== Genome Graph Statistics ===")
        print(f"Patterns: {stats['n_patterns']}")
        print(f"Edges: {stats['n_edges']}")
        print(f"Pieces: {stats['n_pieces']}")
        print(f"Avg degree: {stats['avg_degree']:.2f}")
        print(f"\nTransform counts:")
        for t, count in stats['transform_counts'].items():
            print(f"  {t}: {count}")

    elif '--cytoscape' in sys.argv:
        try:
            graph = GenomeGraph.from_checkpoint(checkpoint_path)
        except ValueError:
            graph = convert_v24_to_genome_graph(checkpoint_path)

        output_path = checkpoint_path.replace('.npz', '_cytoscape.json')
        with open(output_path, 'w') as f:
            json.dump(graph.to_cytoscape(), f, indent=2)
        print(f"Exported to {output_path}")

    else:
        try:
            graph = GenomeGraph.from_checkpoint(checkpoint_path)
        except ValueError:
            graph = convert_v24_to_genome_graph(checkpoint_path)

        print(f"=== Genome Graph ===")
        print(f"Patterns: {len(graph.patterns)}")
        print(f"Edges: {len(graph.edges)}")
        print(f"\nSample patterns:")
        for pid in list(graph.patterns.keys())[:5]:
            p = graph.patterns[pid]
            print(f"  P{pid}: {p.pitch_classes[:8]}... (len={p.length})")

        print(f"\nSample edges:")
        for eid in list(graph.edges.keys())[:10]:
            e = graph.edges[eid]
            print(f"  P{e.source} --{e.transform}--> P{e.target}")
