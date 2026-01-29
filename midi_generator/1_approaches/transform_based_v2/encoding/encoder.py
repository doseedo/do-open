"""
Transform-Relative Encoder

Converts SEQUITUR grammar output + D24 transform table into
transform-relative encoding.

Flow:
1. Take SEQUITUR grammar rules
2. Identify canonical patterns (using transform equivalence classes)
3. For each rule usage in the encoded sequence:
   - If pattern is canonical: emit INTRO token
   - If pattern is transform of earlier: emit TRANSFORM token
   - If exact repeat: emit REPEAT token
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np

from .transform_relative import (
    TokenType,
    EncodingToken,
    TransformRelativeEncoding,
    CanonicalPattern,
    find_transform,
    apply_d24_transform,
)


@dataclass
class PatternOccurrence:
    """Records where a pattern appears in the sequence."""
    rule_id: int
    position: int  # Position in flat sequence
    track_id: int
    pitch_classes: np.ndarray


class TransformRelativeEncoder:
    """
    Encodes SEQUITUR grammar + transforms into transform-relative format.
    """

    def __init__(self, d24_transform_table: Optional[np.ndarray] = None):
        """
        Args:
            d24_transform_table: Shape (N, N) where entry [i,j] gives the
                                 transform_id that maps pattern i to pattern j.
                                 -1 means no transform exists.
        """
        self.d24_table = d24_transform_table
        self._canonical_map: Dict[int, int] = {}  # rule_id -> canonical_id
        self._pattern_cache: Dict[int, np.ndarray] = {}  # rule_id -> pitch_classes

    def encode(
        self,
        grammar,
        factored_objects: List,
        pattern_extractor=None,
        cross_track_relations: Optional[List] = None,
    ) -> TransformRelativeEncoding:
        """
        Encode a piece using transform-relative encoding.

        Args:
            grammar: SEQUITUR grammar with rules
            factored_objects: List of factored MIDI objects with track info
            pattern_extractor: GrammarPatternExtractor instance (optional)
            cross_track_relations: List of cross-track relations (optional)

        Returns:
            TransformRelativeEncoding
        """
        encoding = TransformRelativeEncoding()

        # Group objects by piece
        pieces = {}
        for obj in factored_objects:
            piece_id = obj.piece_id
            if piece_id not in pieces:
                pieces[piece_id] = []
            pieces[piece_id].append(obj)

        # For now, encode first piece (can extend to multiple)
        if not pieces:
            return encoding

        piece_id = list(pieces.keys())[0]
        piece_objects = pieces[piece_id]
        encoding.piece_id = piece_id

        # Group by track
        tracks = {}
        for obj in piece_objects:
            track_id = obj.track_id
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append(obj)

        encoding.n_tracks = len(tracks)

        # Extract patterns from grammar
        rule_patterns = self._extract_rule_patterns(grammar, pattern_extractor)

        # Build canonical patterns using transform equivalence
        canonicals, rule_to_canonical = self._build_canonicals(rule_patterns)
        encoding.canonical_patterns = canonicals

        # Encode each track
        encoding.tokens = []
        introduced_canonicals: Set[int] = set()  # Track which canonicals have INTRO
        token_idx_by_canonical: Dict[int, int] = {}  # canonical_id -> first token index

        for track_id in sorted(tracks.keys()):
            track_tokens = []
            track_objects = tracks[track_id]

            # Get encoded sequences for this track
            for obj in track_objects:
                # Get the grammar-encoded sequence for this object
                encoded_seq = self._get_encoded_sequence(obj, grammar)

                for rule_id in encoded_seq:
                    if rule_id not in rule_to_canonical:
                        continue  # Skip terminal rules

                    canonical_id = rule_to_canonical[rule_id]
                    canonical = canonicals[canonical_id]

                    # Find transform from canonical to this rule's pattern
                    rule_pattern = rule_patterns.get(rule_id)
                    if rule_pattern is None:
                        continue

                    transform_id = find_transform(
                        canonical.pitch_classes, rule_pattern
                    )
                    if transform_id is None:
                        transform_id = 0  # Default to identity if not found

                    if canonical_id not in introduced_canonicals:
                        # First occurrence: INTRO
                        token = EncodingToken(
                            token_type=TokenType.INTRO,
                            pattern_idx=canonical_id,
                            transform_id=0,
                        )
                        introduced_canonicals.add(canonical_id)
                        token_idx_by_canonical[canonical_id] = len(track_tokens)
                        canonical.usage_count += 1
                    elif transform_id == 0:
                        # Identity transform: REPEAT
                        ref_idx = token_idx_by_canonical.get(canonical_id, 0)
                        token = EncodingToken(
                            token_type=TokenType.REPEAT,
                            pattern_idx=ref_idx,
                        )
                        canonical.usage_count += 1
                    else:
                        # Non-identity transform: TRANSFORM
                        ref_idx = token_idx_by_canonical.get(canonical_id, 0)
                        token = EncodingToken(
                            token_type=TokenType.TRANSFORM,
                            pattern_idx=ref_idx,
                            transform_id=transform_id,
                        )
                        canonical.usage_count += 1

                    track_tokens.append(token)

            encoding.tokens.append(track_tokens)

            # Add track boundary if not last track
            if track_id != max(tracks.keys()):
                encoding.tokens[-1].append(
                    EncodingToken(token_type=TokenType.TRACK_BOUNDARY)
                )

        # Add cross-track relations
        if cross_track_relations:
            encoding.cross_track_relations = self._encode_cross_track(
                cross_track_relations, encoding.tokens
            )

        return encoding

    def _extract_rule_patterns(
        self,
        grammar,
        pattern_extractor=None
    ) -> Dict[int, np.ndarray]:
        """Extract pitch-class patterns from grammar rules."""
        patterns = {}

        if pattern_extractor is not None:
            # Use the extractor
            rule_patterns = pattern_extractor.extract_patterns(
                min_length=2, max_length=128, min_usage=1
            )
            for rp in rule_patterns:
                patterns[rp.rule_id] = rp.pitch_class
        elif hasattr(grammar, 'rules'):
            # Direct extraction from grammar
            for rule_id, rule in grammar.rules.items():
                if not hasattr(rule, 'expansion'):
                    continue

                # Extract terminals
                terminals = []
                for symbol in rule.expansion:
                    if isinstance(symbol, int):  # Terminal
                        terminals.append(symbol)

                if len(terminals) >= 2:
                    # Convert to pitch classes
                    # Assuming token format: pitch_offset + octave*12 + pitch_class
                    pitch_offset = 128  # Standard offset
                    pitch_classes = []
                    for t in terminals:
                        if t >= pitch_offset:
                            pc = (t - pitch_offset) % 12
                            pitch_classes.append(pc)

                    if pitch_classes:
                        patterns[rule_id] = np.array(pitch_classes, dtype=np.int8)

        return patterns

    def _build_canonicals(
        self,
        rule_patterns: Dict[int, np.ndarray]
    ) -> Tuple[List[CanonicalPattern], Dict[int, int]]:
        """
        Build canonical patterns using transform equivalence.

        Returns:
            (list of CanonicalPattern, mapping from rule_id to canonical_id)
        """
        canonicals = []
        rule_to_canonical: Dict[int, int] = {}

        # If we have D24 table, use it for equivalence
        if self.d24_table is not None:
            return self._build_canonicals_from_table(rule_patterns)

        # Otherwise, compute transforms on the fly
        processed: Set[int] = set()
        rule_ids = list(rule_patterns.keys())

        for rule_id in rule_ids:
            if rule_id in processed:
                continue

            pattern = rule_patterns[rule_id]
            canonical_id = len(canonicals)

            # This rule becomes a canonical
            canonical = CanonicalPattern(
                pattern_id=canonical_id,
                pitch_classes=pattern,
                original_rule_id=rule_id,
            )
            canonicals.append(canonical)
            rule_to_canonical[rule_id] = canonical_id
            processed.add(rule_id)

            # Find all other rules that are transforms of this one
            for other_id in rule_ids:
                if other_id in processed:
                    continue

                other_pattern = rule_patterns[other_id]
                transform = find_transform(pattern, other_pattern)

                if transform is not None:
                    rule_to_canonical[other_id] = canonical_id
                    processed.add(other_id)

        return canonicals, rule_to_canonical

    def _build_canonicals_from_table(
        self,
        rule_patterns: Dict[int, np.ndarray]
    ) -> Tuple[List[CanonicalPattern], Dict[int, int]]:
        """Build canonicals using precomputed D24 transform table."""
        canonicals = []
        rule_to_canonical: Dict[int, int] = {}

        # D24 table gives us equivalence classes directly
        # Use Union-Find to build classes
        n = self.d24_table.shape[0]
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union patterns related by transforms
        for i in range(n):
            for j in range(i + 1, n):
                if self.d24_table[i, j] >= 0:
                    union(i, j)

        # Group by canonical (root of equivalence class)
        classes: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            if root not in classes:
                classes[root] = []
            classes[root].append(i)

        # Create canonicals (use root's pattern)
        root_to_canonical: Dict[int, int] = {}
        rule_ids = list(rule_patterns.keys())

        for root, members in classes.items():
            if root >= len(rule_ids):
                continue

            rule_id = rule_ids[root]
            if rule_id not in rule_patterns:
                continue

            canonical_id = len(canonicals)
            root_to_canonical[root] = canonical_id

            canonical = CanonicalPattern(
                pattern_id=canonical_id,
                pitch_classes=rule_patterns[rule_id],
                original_rule_id=rule_id,
            )
            canonicals.append(canonical)

            # Map all members to this canonical
            for member in members:
                if member < len(rule_ids):
                    member_rule = rule_ids[member]
                    rule_to_canonical[member_rule] = canonical_id

        return canonicals, rule_to_canonical

    def _get_encoded_sequence(self, obj, grammar) -> List[int]:
        """Get the grammar-encoded sequence for an object."""
        # The grammar encodes the object as a sequence of rule references
        # We need to walk the grammar to get the top-level rule sequence

        if hasattr(obj, 'encoded_sequence'):
            return obj.encoded_sequence

        if hasattr(grammar, 'start_rule') and hasattr(grammar.start_rule, 'expansion'):
            # Return non-terminal rules from expansion
            return [s for s in grammar.start_rule.expansion if not isinstance(s, int)]

        # Fallback: return all rule IDs
        if hasattr(grammar, 'rules'):
            return list(grammar.rules.keys())

        return []

    def _encode_cross_track(
        self,
        cross_track_relations: List,
        tokens: List[List[EncodingToken]]
    ) -> List[Tuple[int, int, int, int, str]]:
        """Encode cross-track relations as references."""
        encoded = []

        for rel in cross_track_relations:
            # Relation format: (track_a, track_b, relation_type, ...)
            if hasattr(rel, 'track_a') and hasattr(rel, 'track_b'):
                encoded.append((
                    rel.track_a, 0,  # token index (approximate)
                    rel.track_b, 0,
                    getattr(rel, 'relation_type', 'unknown')
                ))

        return encoded


def encode_piece(
    grammar,
    factored_objects: List,
    d24_table: Optional[np.ndarray] = None,
    pattern_extractor=None,
    cross_track_relations: Optional[List] = None,
) -> TransformRelativeEncoding:
    """
    Convenience function to encode a piece.

    Args:
        grammar: SEQUITUR grammar
        factored_objects: List of factored MIDI objects
        d24_table: D24 transform table (optional)
        pattern_extractor: GrammarPatternExtractor (optional)
        cross_track_relations: Cross-track relations (optional)

    Returns:
        TransformRelativeEncoding
    """
    encoder = TransformRelativeEncoder(d24_table)
    return encoder.encode(
        grammar,
        factored_objects,
        pattern_extractor,
        cross_track_relations,
    )
