"""
Music DAG: Expression-Based Genome Representation
==================================================

The correct architecture: A declarative expression DAG where
NODE COUNT = DESCRIPTION LENGTH.

Node Types:
- PATTERN: Leaf node with actual musical content
- SEQ: Sequential composition (children play in order)
- PAR: Parallel composition (children play simultaneously)
- TRANSFORM: Apply transform to single child

Example:
    Piece (seq)
    ├── Section_A (seq)
    │   ├── Phrase1 (seq)
    │   │   ├── P12 (pattern)
    │   │   └── Transform(T5∘τ480, P12)
    │   └── Phrase2 = Transform(T7, Phrase1)  ← REUSE
    └── Section_A' = Transform(T5, Section_A) ← REUSE

50 nodes → 500 notes = 10x compression
This is what we want.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set, Any
from enum import Enum
from collections import defaultdict
import numpy as np
import json

# GPU support
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_device(device: str = None) -> str:
    """Get compute device, defaulting to CUDA if available."""
    if device:
        return device
    if HAS_TORCH and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# ============================================================================
# Core Data Structures
# ============================================================================

class NodeType(Enum):
    PATTERN = 'pattern'      # Leaf: actual notes
    SEQ = 'seq'              # Sequential: A then B then C
    PAR = 'par'              # Parallel: A with B with C
    TRANSFORM = 'transform'  # Apply transform to child


@dataclass
class Node:
    """A node in the music expression DAG."""
    id: int
    node_type: NodeType

    # Pattern data (only for PATTERN nodes)
    pitch_classes: Optional[List[int]] = None
    octaves: Optional[List[int]] = None
    velocities: Optional[List[int]] = None
    durations: Optional[List[int]] = None

    # Children (for SEQ, PAR, TRANSFORM nodes)
    children: List[int] = field(default_factory=list)

    # Transform (only for TRANSFORM nodes)
    transform: Optional[str] = None

    # Optional metadata
    name: Optional[str] = None

    @property
    def note_count(self) -> int:
        """Number of notes in this pattern (only for PATTERN nodes)."""
        if self.node_type == NodeType.PATTERN and self.pitch_classes:
            return len(self.pitch_classes)
        return 0

    def __hash__(self):
        """Hash for deduplication."""
        if self.node_type == NodeType.PATTERN:
            return hash((
                self.node_type,
                tuple(self.pitch_classes or []),
                tuple(self.octaves or []),
                tuple(self.velocities or []),
                tuple(self.durations or []),
            ))
        elif self.node_type == NodeType.TRANSFORM:
            return hash((self.node_type, tuple(self.children), self.transform))
        else:
            return hash((self.node_type, tuple(self.children)))


@dataclass
class MusicDAG:
    """A directed acyclic graph of music expressions.

    The key insight: len(nodes) IS the description length.
    Goal: 500 nodes reconstructs any file.
    """
    nodes: Dict[int, Node] = field(default_factory=dict)
    root: Optional[int] = None
    _next_id: int = 0

    # ========================================================================
    # Node Creation
    # ========================================================================

    def add_pattern(
        self,
        pitch_classes: List[int],
        octaves: List[int],
        velocities: List[int],
        durations: List[int],
        name: Optional[str] = None,
    ) -> int:
        """Add a leaf pattern node."""
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = Node(
            id=nid,
            node_type=NodeType.PATTERN,
            pitch_classes=pitch_classes,
            octaves=octaves,
            velocities=velocities,
            durations=durations,
            name=name,
        )
        return nid

    def add_seq(self, children: List[int], name: Optional[str] = None) -> int:
        """Add sequential composition node."""
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = Node(
            id=nid,
            node_type=NodeType.SEQ,
            children=children,
            name=name,
        )
        return nid

    def add_par(self, children: List[int], name: Optional[str] = None) -> int:
        """Add parallel composition node."""
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = Node(
            id=nid,
            node_type=NodeType.PAR,
            children=children,
            name=name,
        )
        return nid

    def add_transform(
        self,
        child: int,
        transform: str,
        name: Optional[str] = None,
    ) -> int:
        """Add transform node.

        Transforms:
        - T0-T11: Pitch transposition (semitones)
        - I0-I11: Inversion around axis
        - R: Retrograde (time reversal)
        - τN: Time shift by N ticks
        - Compound: T5∘τ480 (transpose AND shift)
        """
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = Node(
            id=nid,
            node_type=NodeType.TRANSFORM,
            children=[child],
            transform=transform,
            name=name,
        )
        return nid

    # ========================================================================
    # Statistics
    # ========================================================================

    def node_count(self) -> int:
        """Description length = number of nodes."""
        return len(self.nodes)

    def pattern_count(self) -> int:
        """Number of leaf pattern nodes."""
        return sum(1 for n in self.nodes.values() if n.node_type == NodeType.PATTERN)

    def total_notes(self) -> int:
        """Total notes across all pattern nodes."""
        return sum(n.note_count for n in self.nodes.values())

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        by_type = defaultdict(int)
        for n in self.nodes.values():
            by_type[n.node_type.value] += 1

        # Count unique transforms
        transforms = set()
        for n in self.nodes.values():
            if n.transform:
                transforms.add(n.transform)

        return {
            'node_count': len(self.nodes),
            'by_type': dict(by_type),
            'pattern_count': by_type.get('pattern', 0),
            'total_notes_in_patterns': self.total_notes(),
            'unique_transforms': len(transforms),
            'transforms': sorted(transforms),
        }

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(self) -> Dict:
        """Export DAG to dictionary for JSON serialization."""
        nodes_dict = {}
        for nid, node in self.nodes.items():
            node_data = {
                'id': node.id,
                'type': node.node_type.value,
            }
            if node.node_type == NodeType.PATTERN:
                node_data['pitch_classes'] = node.pitch_classes
                node_data['octaves'] = node.octaves
                node_data['velocities'] = node.velocities
                node_data['durations'] = node.durations
            if node.children:
                node_data['children'] = node.children
            if node.transform:
                node_data['transform'] = node.transform
            if node.name:
                node_data['name'] = node.name
            nodes_dict[str(nid)] = node_data

        return {
            'nodes': nodes_dict,
            'root': self.root,
            'stats': self.get_stats(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MusicDAG':
        """Create DAG from dictionary."""
        dag = cls()
        dag.root = data.get('root')

        for nid_str, node_data in data.get('nodes', {}).items():
            nid = int(nid_str)
            node_type = NodeType(node_data['type'])

            node = Node(
                id=nid,
                node_type=node_type,
                pitch_classes=node_data.get('pitch_classes'),
                octaves=node_data.get('octaves'),
                velocities=node_data.get('velocities'),
                durations=node_data.get('durations'),
                children=node_data.get('children', []),
                transform=node_data.get('transform'),
                name=node_data.get('name'),
            )
            dag.nodes[nid] = node
            dag._next_id = max(dag._next_id, nid + 1)

        return dag

    def to_checkpoint(self, path: str):
        """Save DAG to numpy checkpoint format."""
        data = self.to_dict()
        np.savez(
            path,
            dag_json=np.array([json.dumps(data)]),
            is_music_dag=np.array([True]),
            version=np.array(['v1']),
        )

    @classmethod
    def from_checkpoint(cls, path: str) -> 'MusicDAG':
        """Load DAG from checkpoint."""
        data = np.load(path, allow_pickle=True)

        if 'dag_json' in data:
            dag_data = json.loads(str(data['dag_json'][0]))
            return cls.from_dict(dag_data)

        raise ValueError("Checkpoint does not contain music DAG data")

    def to_cytoscape(self, max_nodes: int = 5000) -> Dict:
        """Export to Cytoscape.js format for web visualization.

        Node colors by type:
        - PATTERN: Blue (#2196F3)
        - SEQ: Green (#4CAF50)
        - PAR: Orange (#FF9800)
        - TRANSFORM: Purple (#9C27B0)
        """
        TYPE_COLORS = {
            'pattern': '#2196F3',
            'seq': '#4CAF50',
            'par': '#FF9800',
            'transform': '#9C27B0',
        }

        nodes = []
        edges = []

        # Limit nodes if too many
        node_ids = list(self.nodes.keys())[:max_nodes]
        node_id_set = set(node_ids)  # For fast lookup

        for nid in node_ids:
            node = self.nodes[nid]

            # Build label
            if node.name:
                label = node.name
            elif node.node_type == NodeType.PATTERN:
                label = f"P{nid}"
            elif node.node_type == NodeType.TRANSFORM:
                label = node.transform or f"T{nid}"
            else:
                label = f"{node.node_type.value}_{nid}"

            nodes.append({
                'data': {
                    'id': str(nid),
                    'label': label,
                    'type': node.node_type.value,
                    'color': TYPE_COLORS.get(node.node_type.value, '#9E9E9E'),
                    'notes': node.note_count if node.node_type == NodeType.PATTERN else 0,
                }
            })

            # Add edges to children (only if child is in the limited set)
            for child_id in (node.children or []):
                if child_id in node_id_set:
                    edge_label = ''
                    if node.node_type == NodeType.TRANSFORM:
                        edge_label = node.transform or ''

                    edges.append({
                        'data': {
                            'id': f"e_{nid}_{child_id}",
                            'source': str(nid),
                            'target': str(child_id),
                            'label': edge_label,
                        }
                    })

        return {
            'elements': {
                'nodes': nodes,
                'edges': edges,
            },
            'stats': self.get_stats(),
        }


# ============================================================================
# Evaluation: DAG → MIDI Events
# ============================================================================

def evaluate(dag: MusicDAG, node_id: int) -> List[Tuple[int, int, int, int, int]]:
    """Expand node to list of (time, pitch, velocity, duration, channel).

    Time is RELATIVE - starts at 0 for each node.
    This is the reconstruction proof: DAG → original music.
    """
    node = dag.nodes[node_id]

    if node.node_type == NodeType.PATTERN:
        # Leaf: emit notes starting at time 0
        events = []
        t = 0
        for i in range(len(node.pitch_classes)):
            pitch = node.pitch_classes[i] + 12 * node.octaves[i]
            vel = node.velocities[i]
            dur = node.durations[i]
            events.append((t, pitch, vel, dur, 0))
            t += dur  # Advance by duration
        return events

    if node.node_type == NodeType.SEQ:
        # Sequential: concatenate children, offset times
        events = []
        t = 0
        for child_id in node.children:
            child_events = evaluate(dag, child_id)
            for (ct, p, v, d, ch) in child_events:
                events.append((t + ct, p, v, d, ch))
            if child_events:
                # Advance by the end of the last event
                t += max(ct + d for ct, p, v, d, ch in child_events)
        return events

    if node.node_type == NodeType.PAR:
        # Parallel: merge children at same time
        events = []
        for child_id in node.children:
            events.extend(evaluate(dag, child_id))
        return events

    if node.node_type == NodeType.TRANSFORM:
        child_events = evaluate(dag, node.children[0])
        return apply_transform(child_events, node.transform)

    return []


def apply_transform(
    events: List[Tuple[int, int, int, int, int]],
    transform: str,
) -> List[Tuple[int, int, int, int, int]]:
    """Apply transform to event list.

    Supports compound transforms like T5∘τ480.
    """
    components = transform.split('∘')

    result = list(events)
    for comp in components:
        if comp.startswith('T') and comp[1:].lstrip('-').isdigit():
            # Pitch transposition
            n = int(comp[1:])
            result = [(t, p + n, v, d, ch) for t, p, v, d, ch in result]

        elif comp.startswith('τ'):
            # Time shift
            delta = int(comp[1:])
            result = [(t + delta, p, v, d, ch) for t, p, v, d, ch in result]

        elif comp.startswith('I') and comp[1:].replace('R', '').lstrip('-').isdigit():
            # Inversion around axis
            axis_str = comp[1:].replace('R', '')
            axis = int(axis_str) if axis_str else 0
            result = [(t, 2 * axis - p, v, d, ch) for t, p, v, d, ch in result]

        elif comp == 'R':
            # Retrograde (time reversal)
            if result:
                max_t = max(t + d for t, p, v, d, ch in result)
                result = [(max_t - t - d, p, v, d, ch) for t, p, v, d, ch in result]

        elif comp == 'identity':
            pass  # No-op

    return result


# ============================================================================
# Gene Editing Operations
# ============================================================================

def factor_transform(dag: MusicDAG, node_id: int) -> int:
    """Factor compound transform into chain.

    T5∘τ480(X) → τ480(T5(X))

    Returns new root node ID.
    """
    node = dag.nodes[node_id]
    if node.node_type != NodeType.TRANSFORM:
        return node_id

    components = node.transform.split('∘')
    if len(components) <= 1:
        return node_id

    # Build chain from inside out (rightmost applied first)
    current = node.children[0]
    for comp in reversed(components):
        current = dag.add_transform(current, comp)

    return current


def entangle_transforms(dag: MusicDAG, node_id: int) -> int:
    """Entangle chain of transforms into compound.

    τ480(T5(X)) → T5∘τ480(X)

    Returns new node ID.
    """
    node = dag.nodes[node_id]
    if node.node_type != NodeType.TRANSFORM:
        return node_id

    # Collect transform chain
    transforms = []
    current = node_id
    while dag.nodes[current].node_type == NodeType.TRANSFORM:
        transforms.append(dag.nodes[current].transform)
        current = dag.nodes[current].children[0]

    if len(transforms) <= 1:
        return node_id

    # Create compound transform
    compound = '∘'.join(transforms)
    return dag.add_transform(current, compound)


def substitute(dag: MusicDAG, old_id: int, new_id: int):
    """Replace all references to old_id with new_id."""
    for node in dag.nodes.values():
        if node.children:
            node.children = [new_id if c == old_id else c for c in node.children]
    if dag.root == old_id:
        dag.root = new_id


def clone_with_transform(dag: MusicDAG, node_id: int, transform: str) -> int:
    """Create transformed version of a subtree.

    Instead of cloning, just wrap in transform node.
    This is the power of the DAG: reuse with transform.
    """
    return dag.add_transform(node_id, transform)


# ============================================================================
# τ (Time Shift) Quantization
# ============================================================================

# Standard grid: 1/4 beat, 1/2 beat, 1 beat, 2 beats, 4 beats, 8 beats (at 480 ticks/beat)
TAU_GRID = [120, 240, 480, 960, 1920, 3840]

def quantize_tau(delta: int) -> int:
    """Quantize raw τ delta to nearest grid value.

    This reduces thousands of unique τ values to ~6 standard intervals,
    enabling proper hash-consing and grammar compression.
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
            # Early exit - grid values only increase
            break
    return best


# ============================================================================
# Build DAG from v24 Checkpoint
# ============================================================================

def build_dag_from_checkpoint(checkpoint_path: str, max_pieces: int = None, verbose: bool = True, quantize_time: bool = False) -> MusicDAG:
    """Convert v24 checkpoint to expression DAG.

    Strategy:
    1. Load patterns as leaf nodes (deduplicated)
    2. Group occurrences by (piece, track)
    3. Build seq nodes for each track with hash-consing for transforms
    4. Combine tracks with par nodes
    5. Apply compression (next step)

    Args:
        checkpoint_path: Path to checkpoint file
        max_pieces: Limit to first N pieces (for testing)
        verbose: Print progress
        quantize_time: Quantize τ values to grid [120, 240, 480, 960, 1920, 3840]
                      This reduces unique transforms from ~5000 to ~6
    """
    import sys

    data = np.load(checkpoint_path, allow_pickle=True)
    dag = MusicDAG()

    # Load patterns
    patterns = {}
    occurrences = []

    if 'grammar_rules_json' in data:
        rules_json = data['grammar_rules_json']
        if hasattr(rules_json, '__len__') and len(rules_json) == 1:
            rules = json.loads(str(rules_json[0]))
        else:
            rules = json.loads(str(rules_json.item()))

        for rule_id, rule_data in rules.items():
            if isinstance(rule_data, dict):
                patterns[int(rule_id)] = rule_data

                # Extract occurrences
                for occ in rule_data.get('occurrences', []):
                    if isinstance(occ, dict):
                        occurrences.append({
                            'pattern_id': int(rule_id),
                            'piece_id': occ.get('piece_id'),
                            'track_id': occ.get('track_id'),
                            'onset_time': occ.get('onset_time'),
                        })
                    elif isinstance(occ, (list, tuple)) and len(occ) >= 3:
                        occurrences.append({
                            'pattern_id': int(rule_id),
                            'piece_id': occ[0],
                            'track_id': occ[1],
                            'onset_time': occ[2],
                        })

    if verbose:
        print(f"  Loaded {len(patterns)} patterns, {len(occurrences)} occurrences")
        sys.stdout.flush()

    # Add patterns as leaf nodes
    pattern_to_node = {}
    for pid, pdata in patterns.items():
        pc = pdata.get('pitch_classes', [])
        oct = pdata.get('octaves', [])
        vel = pdata.get('velocities', [])
        dur = pdata.get('duration_buckets', pdata.get('durations', []))

        if pc:  # Only add non-empty patterns
            nid = dag.add_pattern(pc, oct, vel, dur, name=f"P{pid}")
            pattern_to_node[pid] = nid

    if verbose:
        print(f"  Created {len(pattern_to_node)} pattern nodes")
        sys.stdout.flush()

    # Group occurrences by (piece, track)
    by_piece_track = defaultdict(list)
    for occ in occurrences:
        if occ['pattern_id'] in pattern_to_node:
            key = (occ['piece_id'], occ['track_id'])
            by_piece_track[key].append(occ)

    # Limit pieces if requested
    unique_pieces = sorted(set(k[0] for k in by_piece_track.keys()))
    if max_pieces and len(unique_pieces) > max_pieces:
        unique_pieces = unique_pieces[:max_pieces]
        # Filter by_piece_track
        by_piece_track = {k: v for k, v in by_piece_track.items() if k[0] in unique_pieces}
        if verbose:
            print(f"  Limited to {max_pieces} pieces ({len(by_piece_track)} tracks)")
            sys.stdout.flush()

    # Hash-consing cache for transform nodes: (pattern_node, transform) -> node_id
    transform_cache = {}

    # Build seq nodes for each track
    track_nodes = {}
    track_count = 0
    total_tracks = len(by_piece_track)

    for (piece_id, track_id), occs in by_piece_track.items():
        occs_sorted = sorted(occs, key=lambda x: x['onset_time'] or 0)

        # Build sequence with time transforms (with hash-consing)
        children = []
        prev_end_time = 0

        for occ in occs_sorted:
            pattern_node = pattern_to_node[occ['pattern_id']]
            onset = occ['onset_time'] or 0
            delta = onset - prev_end_time

            if delta > 0:
                # Quantize τ if enabled (reduces ~5000 unique values to ~6)
                if quantize_time:
                    delta = quantize_tau(delta)
                # Hash-cons the transform node
                cache_key = (pattern_node, f"τ{delta}")
                if cache_key in transform_cache:
                    wrapped = transform_cache[cache_key]
                else:
                    wrapped = dag.add_transform(pattern_node, f"τ{delta}")
                    transform_cache[cache_key] = wrapped
                children.append(wrapped)
            else:
                children.append(pattern_node)

            # Estimate end time from pattern duration
            pattern = patterns[occ['pattern_id']]
            durations = pattern.get('duration_buckets', pattern.get('durations', [0]))
            pattern_duration = sum(durations) if durations else 0
            prev_end_time = onset + pattern_duration

        if children:
            track_node = dag.add_seq(children, name=f"Track_{piece_id}_{track_id}")
            track_nodes[(piece_id, track_id)] = track_node

        track_count += 1
        if verbose and track_count % 100 == 0:
            print(f"  Processed {track_count}/{total_tracks} tracks, {len(dag.nodes)} nodes")
            sys.stdout.flush()

    if verbose:
        print(f"  Created {len(track_nodes)} track nodes, {len(transform_cache)} unique transforms")
        sys.stdout.flush()

    # Group tracks by piece (parallel composition)
    by_piece = defaultdict(list)
    for (piece_id, track_id), node_id in track_nodes.items():
        by_piece[piece_id].append(node_id)

    piece_nodes = {}
    for piece_id, track_node_ids in by_piece.items():
        if len(track_node_ids) == 1:
            piece_nodes[piece_id] = track_node_ids[0]
        else:
            piece_nodes[piece_id] = dag.add_par(track_node_ids, name=f"Piece_{piece_id}")

    # Root = all pieces
    if len(piece_nodes) == 1:
        dag.root = list(piece_nodes.values())[0]
    elif piece_nodes:
        dag.root = dag.add_par(list(piece_nodes.values()), name="Corpus")

    if verbose:
        print(f"  Final DAG: {len(dag.nodes)} nodes, {len(piece_nodes)} pieces")
        sys.stdout.flush()

    return dag


# ============================================================================
# MDL Compression on DAG
# ============================================================================

def hash_subtree(dag: MusicDAG, node_id: int, cache: Dict[int, str] = None) -> str:
    """Compute structural hash of subtree rooted at node_id."""
    if cache is None:
        cache = {}

    if node_id in cache:
        return cache[node_id]

    node = dag.nodes[node_id]

    if node.node_type == NodeType.PATTERN:
        h = f"P:{tuple(node.pitch_classes)}:{tuple(node.octaves)}"
    elif node.node_type == NodeType.TRANSFORM:
        child_hash = hash_subtree(dag, node.children[0], cache)
        h = f"T:{node.transform}:{child_hash}"
    else:
        child_hashes = [hash_subtree(dag, c, cache) for c in node.children]
        h = f"{node.node_type.value}:[{','.join(child_hashes)}]"

    cache[node_id] = h
    return h


def find_duplicate_subtrees(dag: MusicDAG) -> Dict[str, List[int]]:
    """Find structurally identical subtrees."""
    hash_cache = {}
    hash_to_nodes = defaultdict(list)

    for nid in dag.nodes:
        h = hash_subtree(dag, nid, hash_cache)
        hash_to_nodes[h].append(nid)

    # Return only duplicates
    return {h: nodes for h, nodes in hash_to_nodes.items() if len(nodes) > 1}


def deduplicate_dag(dag: MusicDAG) -> MusicDAG:
    """Hash-cons: deduplicate identical subtrees.

    If two subtrees are structurally identical,
    make them reference the same node.
    """
    duplicates = find_duplicate_subtrees(dag)

    for h, node_ids in duplicates.items():
        # Keep first, replace others with references to first
        canonical = node_ids[0]
        for other in node_ids[1:]:
            substitute(dag, other, canonical)

    # Remove unreferenced nodes
    dag = prune_unreferenced(dag)

    return dag


def prune_unreferenced(dag: MusicDAG) -> MusicDAG:
    """Remove nodes not reachable from root."""
    if dag.root is None:
        return dag

    # BFS from root
    reachable = set()
    queue = [dag.root]
    while queue:
        nid = queue.pop(0)
        if nid in reachable:
            continue
        reachable.add(nid)
        node = dag.nodes.get(nid)
        if node and node.children:
            queue.extend(node.children)

    # Keep only reachable
    new_dag = MusicDAG()
    new_dag.root = dag.root
    new_dag._next_id = dag._next_id
    for nid in reachable:
        new_dag.nodes[nid] = dag.nodes[nid]

    return new_dag


def find_transform_relationships(dag: MusicDAG, device: str = None) -> List[Tuple[int, int, str]]:
    """Find nodes that are transforms of each other.

    If node B's content = T5(node A's content), return (A, B, "T5").
    This enables replacing B with Transform(A, T5).

    Uses GPU acceleration when available for O(n²) comparison.
    """
    device = get_device(device)
    pattern_nodes = [n for n in dag.nodes.values() if n.node_type == NodeType.PATTERN]

    if not pattern_nodes:
        return []

    # Group by length for efficient comparison
    by_length = defaultdict(list)
    for n in pattern_nodes:
        if n.pitch_classes:
            by_length[len(n.pitch_classes)].append(n)

    relationships = []

    # Use GPU for groups with many patterns of same length
    if HAS_TORCH and device == 'cuda':
        for length, nodes in by_length.items():
            if len(nodes) < 2:
                continue

            # Stack pitch classes into tensor [N, length]
            pitch_matrix = torch.tensor(
                [n.pitch_classes for n in nodes],
                dtype=torch.int32,
                device=device
            )

            n_patterns = len(nodes)

            # Fully vectorized transposition detection:
            # 1. Compute pairwise differences for first element
            first_col = pitch_matrix[:, 0]  # [N]
            diff_first = first_col.unsqueeze(1) - first_col.unsqueeze(0)  # [N, N]

            # 2. Compute expected transposed values for all pairs
            # expected[i,j,k] = pitch_matrix[i,k] + diff_first[i,j]
            expected = pitch_matrix.unsqueeze(1) + diff_first.unsqueeze(2)  # [N, N, length]

            # 3. Check if actual matches expected for all positions
            # is_transposition[i,j] = True if pitch_matrix[j] == expected[i,j]
            actual = pitch_matrix.unsqueeze(0)  # [1, N, length] broadcasts to [N, N, length]
            is_transposition = torch.all(actual == expected, dim=2)  # [N, N]

            # 4. Only keep upper triangle (avoid duplicates) and non-zero deltas
            mask = torch.triu(torch.ones(n_patterns, n_patterns, dtype=torch.bool, device=device), diagonal=1)
            mask = mask & (diff_first != 0)
            mask = mask & is_transposition

            # 5. Extract valid pairs
            pairs = torch.nonzero(mask, as_tuple=False)  # [K, 2]
            for pair in pairs:
                i, j = pair[0].item(), pair[1].item()
                delta = diff_first[i, j].item()
                relationships.append((nodes[i].id, nodes[j].id, f"T{delta}"))
    else:
        # CPU fallback
        for length, nodes in by_length.items():
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i+1:]:
                    diffs = [p2 - p1 for p1, p2 in zip(n1.pitch_classes, n2.pitch_classes)]
                    if len(set(diffs)) == 1:
                        delta = diffs[0]
                        if delta != 0:
                            relationships.append((n1.id, n2.id, f"T{delta}"))

    return relationships


def compress_via_transforms(dag: MusicDAG, device: str = None) -> MusicDAG:
    """Replace duplicate patterns with transform references.

    If B = T5(A), remove B's content and make B = Transform(A, T5).
    Uses GPU acceleration for finding relationships.
    """
    relationships = find_transform_relationships(dag, device=device)

    for source_id, target_id, transform in relationships:
        # Create transform node pointing to source
        new_node = dag.add_transform(source_id, transform)

        # Replace all references to target with new transform node
        substitute(dag, target_id, new_node)

    return prune_unreferenced(dag)


def repair_seq_patterns(dag: MusicDAG, min_count: int = 2, max_iterations: int = 200, device: str = None) -> MusicDAG:
    """Apply Re-Pair compression to seq node children.

    This is the key to achieving 500 nodes per piece.
    Uses GPU acceleration for bigram counting when available.

    Example:
        Seq([P1, τ100(P1), P2, τ100(P2), P1, τ100(P1), P2, τ100(P2)])
        → Phrase_X = Seq([P1, τ100(P1), P2, τ100(P2)])
        → Seq([Phrase_X, Phrase_X])

    Algorithm:
    1. Find all seq nodes
    2. Extract child sequences (as node ID tuples)
    3. Find repeated bigrams across all sequences
    4. Replace most frequent bigram with new seq node
    5. Repeat until no bigrams with count >= min_count
    """
    import sys
    device = get_device(device)

    # Collect all seq nodes
    seq_nodes = [n for n in dag.nodes.values() if n.node_type == NodeType.SEQ]

    if not seq_nodes:
        return dag

    iteration = 0
    while True:
        # Count bigrams - use GPU if available and worthwhile
        use_gpu = (HAS_TORCH and device == 'cuda' and
                   sum(len(n.children) for n in seq_nodes if len(n.children) >= 2) > 1000)

        if use_gpu:
            # GPU-accelerated bigram counting - fully vectorized
            all_children = []
            seq_starts = [0]  # Track where each sequence starts
            seq_node_ids = []  # Track which seq node each segment belongs to

            for seq_node in seq_nodes:
                if len(seq_node.children) >= 2:
                    all_children.extend(seq_node.children)
                    seq_starts.append(len(all_children))
                    seq_node_ids.append(seq_node.id)

            if len(all_children) < 2:
                break

            # Map node IDs to compact indices
            unique_ids = sorted(set(all_children))
            id_to_idx = {nid: i for i, nid in enumerate(unique_ids)}
            n_unique = len(unique_ids)

            # Convert to tensor
            children_tensor = torch.tensor(
                [id_to_idx[c] for c in all_children],
                dtype=torch.int64, device=device
            )

            # Create position-in-sequence tensor (vectorized)
            pos_in_seq = torch.zeros(len(all_children), dtype=torch.int64, device=device)
            seq_idx_tensor = torch.zeros(len(all_children), dtype=torch.int64, device=device)
            for seg_idx, (start, end) in enumerate(zip(seq_starts[:-1], seq_starts[1:])):
                pos_in_seq[start:end] = torch.arange(end - start, device=device)
                seq_idx_tensor[start:end] = seg_idx

            # Create valid bigram mask (exclude cross-sequence boundaries)
            valid_pos = torch.ones(len(all_children) - 1, dtype=torch.bool, device=device)
            boundary_positions = torch.tensor([end - 1 for end in seq_starts[1:-1]], device=device)
            if boundary_positions.numel() > 0:
                valid_pos[boundary_positions] = False

            # Extract bigrams - all vectorized
            first = children_tensor[:-1]
            second = children_tensor[1:]

            # Count bigrams using bincount (encode as single int)
            bigram_keys_all = first * n_unique + second
            bigram_keys = bigram_keys_all[valid_pos]

            if bigram_keys.numel() == 0:
                break

            counts = torch.bincount(bigram_keys, minlength=n_unique * n_unique)
            max_count = counts.max().item()

            if max_count < min_count:
                break

            best_idx = counts.argmax().item()
            best_a = best_idx // n_unique
            best_b = best_idx % n_unique
            best_bigram = (unique_ids[best_a], unique_ids[best_b])

            # Find all locations of best bigram - vectorized
            target_key = best_a * n_unique + best_b
            matches = (bigram_keys_all == target_key) & valid_pos

            # Get match indices and convert to (seq_node_id, pos_in_seq)
            match_indices = torch.nonzero(matches, as_tuple=False).squeeze(-1)

            bigram_locations = defaultdict(list)
            if match_indices.numel() > 0:
                match_seq_indices = seq_idx_tensor[match_indices].cpu().numpy()
                match_positions = pos_in_seq[match_indices].cpu().numpy()

                for idx, (seg_idx, pos) in enumerate(zip(match_seq_indices, match_positions)):
                    seq_id = seq_node_ids[seg_idx]
                    bigram_locations[best_bigram].append((seq_id, int(pos)))
        else:
            # CPU fallback - original implementation
            bigram_counts = defaultdict(int)
            bigram_locations = defaultdict(list)

            for seq_node in seq_nodes:
                if len(seq_node.children) < 2:
                    continue
                for i in range(len(seq_node.children) - 1):
                    bigram = (seq_node.children[i], seq_node.children[i + 1])
                    bigram_counts[bigram] += 1
                    bigram_locations[bigram].append((seq_node.id, i))

            if not bigram_counts:
                break

            best_bigram, best_count = max(bigram_counts.items(), key=lambda x: x[1])

            if best_count < min_count:
                break

        # Create new phrase node for this bigram
        phrase_id = dag.add_seq(list(best_bigram), name=f"Phrase_{iteration}")

        # Replace all occurrences
        by_seq = defaultdict(list)
        for seq_id, pos in bigram_locations[best_bigram]:
            by_seq[seq_id].append(pos)

        for seq_id, positions in by_seq.items():
            seq_node = dag.nodes[seq_id]
            positions = sorted(positions, reverse=True)

            new_children = list(seq_node.children)
            replaced = set()

            for pos in positions:
                if pos in replaced or pos + 1 in replaced:
                    continue

                if (pos + 1 < len(new_children) and
                    new_children[pos] == best_bigram[0] and
                    new_children[pos + 1] == best_bigram[1]):
                    new_children = new_children[:pos] + [phrase_id] + new_children[pos + 2:]
                    replaced.add(pos)

            seq_node.children = new_children

        iteration += 1

        # Progress output every 50 iterations
        if iteration % 50 == 0:
            print(f"    Re-Pair iteration {iteration}, nodes: {len(dag.nodes)}", flush=True)

        seq_nodes = [n for n in dag.nodes.values() if n.node_type == NodeType.SEQ]

        if iteration >= max_iterations:
            print(f"    Re-Pair reached max iterations ({max_iterations})", flush=True)
            break

    return dag


def compress_dag_mdl(dag: MusicDAG, device: str = None, verbose: bool = True) -> MusicDAG:
    """Full MDL compression pipeline with GPU acceleration.

    1. Hash-cons identical subtrees
    2. Find and use transform relationships (GPU accelerated)
    3. Apply Re-Pair on seq patterns (GPU accelerated)
    """
    import sys
    device = get_device(device)

    if verbose:
        print(f"Before compression: {dag.node_count()} nodes (device={device})")
        sys.stdout.flush()

    # Step 1: Deduplicate identical subtrees
    dag = deduplicate_dag(dag)
    if verbose:
        print(f"After dedup: {dag.node_count()} nodes")
        sys.stdout.flush()

    # Step 2: Use transform relationships (GPU accelerated)
    dag = compress_via_transforms(dag, device=device)
    if verbose:
        print(f"After transform compression: {dag.node_count()} nodes")
        sys.stdout.flush()

    # Step 3: Re-Pair on seq patterns (GPU accelerated)
    dag = repair_seq_patterns(dag, device=device)
    if verbose:
        print(f"After Re-Pair on sequences: {dag.node_count()} nodes")
        sys.stdout.flush()

    # Step 4: Final dedup (new phrase nodes may be duplicates)
    dag = deduplicate_dag(dag)
    if verbose:
        print(f"After final dedup: {dag.node_count()} nodes")
        sys.stdout.flush()

    return dag


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Music DAG: Expression-based genome compression')
    parser.add_argument('checkpoint', help='Path to v24 checkpoint file')
    parser.add_argument('--save', '-s', help='Save compressed DAG to file')
    parser.add_argument('--cytoscape', '-c', help='Export to Cytoscape JSON')
    parser.add_argument('--max-pieces', '-m', type=int, help='Limit to N pieces')
    parser.add_argument('--no-compress', action='store_true', help='Skip compression')
    args = parser.parse_args()

    checkpoint_path = args.checkpoint

    print(f"Loading {checkpoint_path}...")
    dag = build_dag_from_checkpoint(checkpoint_path, max_pieces=args.max_pieces)

    print("\n=== Initial DAG Statistics ===")
    stats = dag.get_stats()
    print(f"Total nodes: {stats['node_count']}")
    print(f"  Patterns: {stats['by_type'].get('pattern', 0)}")
    print(f"  Sequences: {stats['by_type'].get('seq', 0)}")
    print(f"  Parallel: {stats['by_type'].get('par', 0)}")
    print(f"  Transforms: {stats['by_type'].get('transform', 0)}")
    print(f"Notes in patterns: {stats['total_notes_in_patterns']}")
    print(f"Unique transforms: {stats['unique_transforms']}")

    if not args.no_compress:
        print("\n=== Applying MDL Compression ===")
        dag = compress_dag_mdl(dag)

        print("\n=== Compressed DAG Statistics ===")
        stats = dag.get_stats()
        print(f"Total nodes: {stats['node_count']}")
        print(f"  Patterns: {stats['by_type'].get('pattern', 0)}")
        print(f"  Sequences: {stats['by_type'].get('seq', 0)}")
        print(f"  Parallel: {stats['by_type'].get('par', 0)}")
        print(f"  Transforms: {stats['by_type'].get('transform', 0)}")

    # Verify reconstruction
    if dag.root is not None:
        print("\n=== Reconstruction Test ===")
        events = evaluate(dag, dag.root)
        print(f"Reconstructed {len(events)} events from {stats['node_count']} nodes")
        if events:
            print(f"Compression ratio: {len(events) / stats['node_count']:.1f}x")

    # Save checkpoint
    if args.save:
        dag.to_checkpoint(args.save)
        print(f"\nSaved compressed DAG to {args.save}")

    # Export Cytoscape
    if args.cytoscape:
        cyto_data = dag.to_cytoscape()
        with open(args.cytoscape, 'w') as f:
            json.dump(cyto_data, f, indent=2)
        print(f"Exported Cytoscape JSON to {args.cytoscape}")
