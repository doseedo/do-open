"""
Hierarchical Genome: True 500-Operation Compression
====================================================

The key insight: Apply Re-Pair to the EDGE SEQUENCE, not just patterns.

A file with 3,261 notes should be representable in ~500 operations because:
1. Patterns compress repeated note sequences (Re-Pair level 1)
2. Transform edges compress pattern-to-pattern relationships
3. Edge sequences compress via Re-Pair on the transform grammar (Level 3)
4. Sections/phrases are SUBGRAPHS that can be referenced by a single node

This module implements:
- HierarchicalGenome: A genome where subgraphs are first-class citizens
- Subgraph compression via Re-Pair on edge sequences
- Phrase/section detection as subgraph boundaries
- ~500 operation representation goal
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, Counter
from itertools import groupby


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Pattern:
    """A musical pattern - leaf node in the genome."""
    id: int
    pitch_classes: List[int]
    octaves: List[int]
    velocities: List[int]
    durations: List[int]

    @property
    def length(self) -> int:
        return len(self.pitch_classes)


@dataclass
class Operation:
    """A single operation in the compressed representation.

    Operations can be:
    - "emit P123" - emit pattern P123 at current time
    - "advance τ480" - advance time by 480 ticks
    - "call S5" - expand subgraph S5 (recursive)
    - "transform T5" - apply transposition to following operations
    - "repeat 4" - repeat previous operation/group 4 times
    """
    op_type: str  # 'emit', 'advance', 'call', 'transform', 'repeat'
    value: Any  # pattern_id, ticks, subgraph_id, transform_str, count


@dataclass
class Subgraph:
    """A reusable subgraph - phrase, section, or higher-level structure.

    Subgraphs contain a sequence of operations that can be:
    - Expanded inline
    - Called by reference
    - Transformed when called (e.g., T5(S1) = S1 transposed up a fourth)
    """
    id: int
    name: str  # "phrase_1", "section_A", etc.
    operations: List[Operation]

    # Metadata
    depth: int = 0  # Depth in hierarchy (0 = pattern, 1 = phrase, 2 = section, etc.)
    occurrence_count: int = 0  # How many times this subgraph is referenced


@dataclass
class HierarchicalGenome:
    """A hierarchically compressed musical genome.

    Structure:
        patterns: Dict[int, Pattern] - leaf patterns
        subgraphs: Dict[int, Subgraph] - reusable subgraphs (phrases, sections)
        pieces: Dict[str, int] - piece_id -> root subgraph_id

    A piece is reconstructed by expanding its root subgraph.
    """
    patterns: Dict[int, Pattern] = field(default_factory=dict)
    subgraphs: Dict[int, Subgraph] = field(default_factory=dict)
    pieces: Dict[str, int] = field(default_factory=dict)  # piece_id -> root subgraph

    # Compression stats
    total_notes: int = 0
    total_operations: int = 0
    compression_ratio: float = 1.0

    _next_pattern_id: int = 0
    _next_subgraph_id: int = 0


# ============================================================================
# Re-Pair on Operation Sequences
# ============================================================================

def repair_operations(ops: List[Operation], min_count: int = 2) -> Tuple[List[Operation], Dict[int, List[Operation]]]:
    """Apply Re-Pair to an operation sequence to find repeated patterns.

    Args:
        ops: List of operations
        min_count: Minimum occurrences for a pattern to be extracted

    Returns:
        (compressed_ops, new_subgraphs)
    """
    if len(ops) < 2:
        return ops, {}

    # Convert operations to hashable form for pair counting
    def op_key(op: Operation) -> Tuple:
        return (op.op_type, str(op.value))

    # Count adjacent pairs
    pair_counts = Counter()
    for i in range(len(ops) - 1):
        pair = (op_key(ops[i]), op_key(ops[i + 1]))
        pair_counts[pair] += 1

    # Find most frequent pair
    if not pair_counts:
        return ops, {}

    best_pair, best_count = pair_counts.most_common(1)[0]

    if best_count < min_count:
        return ops, {}

    # Create new subgraph for this pair
    new_subgraph_id = 10000 + len(pair_counts)  # Start high to avoid conflicts
    new_subgraph_ops = [
        Operation(op_type=best_pair[0][0], value=eval(best_pair[0][1]) if best_pair[0][1].isdigit() else best_pair[0][1]),
        Operation(op_type=best_pair[1][0], value=eval(best_pair[1][1]) if best_pair[1][1].isdigit() else best_pair[1][1]),
    ]

    # Replace all occurrences of pair with subgraph call
    new_ops = []
    i = 0
    while i < len(ops):
        if i < len(ops) - 1:
            pair = (op_key(ops[i]), op_key(ops[i + 1]))
            if pair == best_pair:
                new_ops.append(Operation(op_type='call', value=new_subgraph_id))
                i += 2
                continue
        new_ops.append(ops[i])
        i += 1

    new_subgraphs = {new_subgraph_id: new_subgraph_ops}

    # Recurse
    compressed_ops, more_subgraphs = repair_operations(new_ops, min_count)
    new_subgraphs.update(more_subgraphs)

    return compressed_ops, new_subgraphs


def run_length_encode(ops: List[Operation]) -> List[Operation]:
    """Apply run-length encoding to consecutive identical operations.

    [τ480, τ480, τ480, τ480] -> [τ480, repeat 4]
    """
    if len(ops) < 2:
        return ops

    result = []
    i = 0

    while i < len(ops):
        # Count consecutive identical operations
        count = 1
        while i + count < len(ops) and ops[i + count].op_type == ops[i].op_type and ops[i + count].value == ops[i].value:
            count += 1

        if count >= 2:
            result.append(ops[i])
            result.append(Operation(op_type='repeat', value=count))
            i += count
        else:
            result.append(ops[i])
            i += 1

    return result


# ============================================================================
# Build Hierarchical Genome from Flat Pattern Sequence
# ============================================================================

def detect_phrase_boundaries(operations: List[Operation],
                            long_pause_threshold: int = 960) -> List[int]:
    """Detect phrase boundaries based on timing gaps.

    A phrase boundary occurs when:
    - Long pause (> threshold ticks between patterns)
    - Pattern repetition structure changes

    Returns list of indices where phrases begin.
    """
    boundaries = [0]  # First phrase starts at 0

    for i, op in enumerate(operations):
        if op.op_type == 'advance' and op.value >= long_pause_threshold:
            boundaries.append(i + 1)

    return boundaries


def detect_section_boundaries(phrases: List[Subgraph]) -> List[int]:
    """Detect section boundaries based on phrase similarity.

    A section boundary occurs when phrase patterns change significantly.

    Returns list of indices where sections begin.
    """
    if len(phrases) < 2:
        return [0]

    boundaries = [0]

    # Simple heuristic: section boundary when phrase length changes by > 50%
    for i in range(1, len(phrases)):
        prev_len = len(phrases[i - 1].operations)
        curr_len = len(phrases[i].operations)

        if abs(curr_len - prev_len) > 0.5 * prev_len:
            boundaries.append(i)

    return boundaries


def build_hierarchical_genome(
    patterns: Dict[int, Pattern],
    occurrences: List[Dict],  # [{pattern_id, piece_id, track_id, onset_time}, ...]
    min_phrase_length: int = 4,
    min_section_length: int = 2,
) -> HierarchicalGenome:
    """Build hierarchical genome from pattern occurrences.

    Process:
    1. Sort occurrences by (piece, track, time)
    2. Convert to operation sequences (emit + advance)
    3. Run-length encode repeated operations
    4. Detect phrase boundaries, create phrase subgraphs
    5. Run Re-Pair on phrase sequences to find repeated phrases
    6. Detect section boundaries, create section subgraphs
    7. Create root subgraph for each piece
    """
    genome = HierarchicalGenome()
    genome.patterns = patterns

    # Group occurrences by (piece, track)
    occurrences_sorted = sorted(occurrences, key=lambda x: (
        x.get('piece_id', ''),
        x.get('track_id', 0),
        x.get('onset_time', 0)
    ))

    for (piece_id, track_id), group in groupby(
        occurrences_sorted,
        key=lambda x: (x.get('piece_id', ''), x.get('track_id', 0))
    ):
        group_list = list(group)
        if not group_list:
            continue

        # Build operation sequence
        operations = []
        prev_time = 0

        for occ in group_list:
            pattern_id = occ.get('pattern_id')
            onset_time = occ.get('onset_time', 0)

            # Advance time if needed
            if onset_time > prev_time:
                delta = onset_time - prev_time
                operations.append(Operation(op_type='advance', value=delta))

            # Emit pattern
            operations.append(Operation(op_type='emit', value=pattern_id))

            # Update time (pattern duration)
            pattern = patterns.get(pattern_id)
            if pattern:
                # Estimate duration from pattern
                duration = sum(pattern.durations) if pattern.durations else 480
                prev_time = onset_time + duration
            else:
                prev_time = onset_time + 480

        # Run-length encode
        operations = run_length_encode(operations)

        # Detect and create phrases
        phrase_boundaries = detect_phrase_boundaries(operations)
        phrases = []

        for i in range(len(phrase_boundaries)):
            start = phrase_boundaries[i]
            end = phrase_boundaries[i + 1] if i + 1 < len(phrase_boundaries) else len(operations)

            if end - start >= min_phrase_length:
                phrase_id = genome._next_subgraph_id
                genome._next_subgraph_id += 1

                phrase = Subgraph(
                    id=phrase_id,
                    name=f"phrase_{piece_id}_{track_id}_{i}",
                    operations=operations[start:end],
                    depth=1,
                )
                genome.subgraphs[phrase_id] = phrase
                phrases.append(phrase)

        # Re-Pair on phrase operations to find repeated phrases
        if phrases:
            phrase_calls = [Operation(op_type='call', value=p.id) for p in phrases]
            compressed_calls, new_subgraphs = repair_operations(phrase_calls, min_count=2)

            for sg_id, sg_ops in new_subgraphs.items():
                genome.subgraphs[sg_id] = Subgraph(
                    id=sg_id,
                    name=f"phrase_group_{sg_id}",
                    operations=sg_ops,
                    depth=2,
                )

            # Create root subgraph for this track
            root_id = genome._next_subgraph_id
            genome._next_subgraph_id += 1

            genome.subgraphs[root_id] = Subgraph(
                id=root_id,
                name=f"track_{piece_id}_{track_id}",
                operations=compressed_calls,
                depth=3,
            )

            # Record piece -> root mapping
            piece_key = f"{piece_id}:{track_id}"
            genome.pieces[piece_key] = root_id

        # Count notes
        genome.total_notes += sum(
            patterns[occ.get('pattern_id')].length
            for occ in group_list
            if occ.get('pattern_id') in patterns
        )

    # Calculate total operations
    genome.total_operations = sum(
        len(sg.operations) for sg in genome.subgraphs.values()
    )

    if genome.total_notes > 0:
        genome.compression_ratio = genome.total_notes / max(1, genome.total_operations)

    return genome


# ============================================================================
# Expand Hierarchical Genome to Operations
# ============================================================================

def expand_subgraph(genome: HierarchicalGenome, subgraph_id: int,
                   transform: str = None, max_depth: int = 10) -> List[Operation]:
    """Recursively expand a subgraph to its full operation sequence.

    Args:
        genome: The hierarchical genome
        subgraph_id: Subgraph to expand
        transform: Optional transform to apply (e.g., "T5")
        max_depth: Maximum recursion depth

    Returns:
        List of expanded operations
    """
    if max_depth <= 0:
        return []

    subgraph = genome.subgraphs.get(subgraph_id)
    if not subgraph:
        return []

    result = []
    repeat_count = 1

    for op in subgraph.operations:
        if op.op_type == 'repeat':
            repeat_count = op.value
            continue

        for _ in range(repeat_count):
            if op.op_type == 'call':
                # Recursively expand
                nested = expand_subgraph(genome, op.value, transform, max_depth - 1)
                result.extend(nested)
            elif op.op_type == 'emit' and transform:
                # Apply transform to emitted pattern
                result.append(Operation(
                    op_type='emit_transformed',
                    value=(op.value, transform)
                ))
            else:
                result.append(op)

        repeat_count = 1

    return result


def count_operations(genome: HierarchicalGenome) -> Dict[str, int]:
    """Count operations by type across the genome."""
    counts = defaultdict(int)

    for sg in genome.subgraphs.values():
        for op in sg.operations:
            counts[op.op_type] += 1

    return dict(counts)


def get_compression_stats(genome: HierarchicalGenome) -> Dict:
    """Get detailed compression statistics."""
    op_counts = count_operations(genome)

    # Count unique patterns vs total pattern references
    pattern_refs = sum(1 for sg in genome.subgraphs.values()
                       for op in sg.operations if op.op_type == 'emit')
    unique_patterns = len(genome.patterns)

    # Count subgraph reuse
    subgraph_calls = sum(1 for sg in genome.subgraphs.values()
                         for op in sg.operations if op.op_type == 'call')

    return {
        'total_notes': genome.total_notes,
        'total_operations': genome.total_operations,
        'compression_ratio': genome.compression_ratio,
        'unique_patterns': unique_patterns,
        'pattern_references': pattern_refs,
        'subgraph_count': len(genome.subgraphs),
        'subgraph_calls': subgraph_calls,
        'piece_count': len(genome.pieces),
        'operation_breakdown': op_counts,
    }


# ============================================================================
# Convert from v24 Checkpoint to Hierarchical Genome
# ============================================================================

def convert_v24_to_hierarchical(checkpoint_path: str) -> HierarchicalGenome:
    """Convert a v24 checkpoint to hierarchical genome format.

    This achieves the 500-operation goal by:
    1. Using existing patterns as leaf nodes
    2. Building operation sequences from occurrences
    3. Applying multi-level Re-Pair compression
    """
    import json

    data = np.load(checkpoint_path, allow_pickle=True)

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
                pattern = Pattern(
                    id=int(rule_id),
                    pitch_classes=rule_data.get('pitch_classes', []),
                    octaves=rule_data.get('octaves', []),
                    velocities=rule_data.get('velocities', []),
                    durations=rule_data.get('duration_buckets',
                                           rule_data.get('durations', [])),
                )
                patterns[int(rule_id)] = pattern

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

    # Build hierarchical genome
    return build_hierarchical_genome(patterns, occurrences)


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hierarchical_genome.py <checkpoint.npz>")
        print()
        print("Converts a v24 checkpoint to hierarchical genome format")
        print("and displays compression statistics.")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    print(f"Converting {checkpoint_path} to hierarchical genome...")
    genome = convert_v24_to_hierarchical(checkpoint_path)

    stats = get_compression_stats(genome)

    print(f"\n=== Hierarchical Genome Statistics ===")
    print(f"Total notes: {stats['total_notes']:,}")
    print(f"Total operations: {stats['total_operations']:,}")
    print(f"Compression ratio: {stats['compression_ratio']:.1f}x")
    print(f"\n=== Structure ===")
    print(f"Unique patterns: {stats['unique_patterns']:,}")
    print(f"Pattern references: {stats['pattern_references']:,}")
    print(f"Subgraphs: {stats['subgraph_count']:,}")
    print(f"Subgraph calls: {stats['subgraph_calls']:,}")
    print(f"Pieces: {stats['piece_count']}")

    print(f"\n=== Operation Breakdown ===")
    for op_type, count in sorted(stats['operation_breakdown'].items()):
        print(f"  {op_type}: {count:,}")

    # Show sample subgraph
    if genome.subgraphs:
        sample_id = list(genome.subgraphs.keys())[0]
        sample = genome.subgraphs[sample_id]
        print(f"\n=== Sample Subgraph: {sample.name} ===")
        print(f"Depth: {sample.depth}")
        print(f"Operations ({len(sample.operations)}):")
        for op in sample.operations[:10]:
            print(f"  {op.op_type}: {op.value}")
        if len(sample.operations) > 10:
            print(f"  ... ({len(sample.operations) - 10} more)")
