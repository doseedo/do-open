"""
Grammar Pattern Extractor
=========================

Extracts pitch-class patterns from SEQUITUR grammar rules for transform discovery.

The key insight: SEQUITUR finds recurring motifs in the corpus as grammar rules.
These rules represent musical patterns (brass riffs, melodic phrases, etc).
We extract the pitch-class content of these rules to find D24 transforms between them.

Example:
    Rule R42: [C, E, G] in measure 4
    Rule R17: [G, B, D] in measure 1

    These are related by transposition: R42 = T_7(R17)
    (transpose up 7 semitones: C->G, E->B, G->D)

Author: Dosedo Architecture v2
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class RulePattern:
    """A pitch-class pattern extracted from a grammar rule."""
    rule_id: int
    pitch_class: np.ndarray  # Shape: (n_notes,), values 0-11
    usage_count: int
    length: int  # Number of notes
    depth: int = 0  # Rule depth in grammar hierarchy

    def __repr__(self):
        pc_str = ','.join(str(x) for x in self.pitch_class[:5])
        if len(self.pitch_class) > 5:
            pc_str += '...'
        return f"RulePattern(R{self.rule_id}, len={self.length}, usage={self.usage_count}, pc=[{pc_str}])"


class GrammarPatternExtractor:
    """
    Extract pitch-class patterns from SEQUITUR grammar rules.

    This enables transform discovery on grammar-induced patterns rather than
    raw objects, which is essential for finding musical relationships.

    Usage:
        extractor = GrammarPatternExtractor(grammar)
        patterns = extractor.extract_patterns(min_length=4)

        # Now pass patterns to UnifiedRelationDiscovery
    """

    # Token encoding constants (must match serialize_factored_object)
    RHYTHM_VOCAB_SIZE = 1000
    PITCH_CLASS_VOCAB_SIZE = 12
    OCTAVE_VOCAB_SIZE = 10
    DURATION_VOCAB_SIZE = 100
    VELOCITY_LEVELS = 8

    def __init__(self, grammar: 'SequiturGrammar'):
        """
        Initialize extractor with a grammar.

        Args:
            grammar: SequiturGrammar with induced rules
        """
        self.grammar = grammar

        # Compute token offsets
        self.pitch_offset = self.RHYTHM_VOCAB_SIZE
        self.duration_offset = self.pitch_offset + self.PITCH_CLASS_VOCAB_SIZE * self.OCTAVE_VOCAB_SIZE
        self.velocity_offset = self.duration_offset + self.DURATION_VOCAB_SIZE

        # Cache for expanded rules
        self._expansion_cache: Dict[int, List[int]] = {}

    def extract_patterns(
        self,
        min_length: int = 4,
        max_length: int = 64,
        min_usage: int = 2
    ) -> List[RulePattern]:
        """
        Extract pitch-class patterns from all grammar rules.

        Args:
            min_length: Minimum number of notes in pattern
            max_length: Maximum number of notes (avoid huge patterns)
            min_usage: Minimum rule usage count

        Returns:
            List of RulePattern objects
        """
        patterns = []

        # Get rule statistics for depth info
        stats = self.grammar.get_rule_stats() if hasattr(self.grammar, 'get_rule_stats') else {}

        for rule_id, rule in self.grammar.rules.items():
            # Skip rules that don't meet usage threshold
            if rule.usage_count < min_usage:
                continue

            # Expand rule to terminals
            terminals = self._expand_rule(rule)

            # Extract pitch classes from terminals
            pitch_classes = self._extract_pitch_classes(terminals)

            # Filter by length
            if len(pitch_classes) < min_length or len(pitch_classes) > max_length:
                continue

            # Get depth from stats if available
            depth = stats.get(rule_id, {}).get('depth', 0)

            patterns.append(RulePattern(
                rule_id=rule_id,
                pitch_class=np.array(pitch_classes, dtype=np.int8),
                usage_count=rule.usage_count,
                length=len(pitch_classes),
                depth=depth
            ))

        return patterns

    def _expand_rule(self, rule: 'Rule') -> List[int]:
        """Expand a rule to its terminal sequence (with caching)."""
        if rule.id in self._expansion_cache:
            return self._expansion_cache[rule.id]

        # Use the rule's expand method
        terminals = rule.expand()
        self._expansion_cache[rule.id] = terminals

        return terminals

    def _extract_pitch_classes(self, terminals: List[int]) -> List[int]:
        """
        Extract pitch classes from a terminal sequence.

        Tokens are structured as:
        - rhythm_id (0 to RHYTHM_VOCAB_SIZE-1)
        - pitch tokens (pitch_offset + octave*12 + pitch_class)
        - duration tokens
        - velocity tokens
        - separator (-1)

        We extract only the pitch_class values.
        """
        pitch_classes = []

        for token in terminals:
            # Skip separators and non-pitch tokens
            if token < 0:
                continue

            # Check if this is a pitch token
            if self.pitch_offset <= token < self.duration_offset:
                # Decode: token = pitch_offset + octave*12 + pitch_class
                relative = token - self.pitch_offset
                pitch_class = relative % 12
                pitch_classes.append(pitch_class)

        return pitch_classes

    def get_pattern_by_rule_id(self, rule_id: int) -> Optional[RulePattern]:
        """Get pattern for a specific rule ID."""
        if rule_id not in self.grammar.rules:
            return None

        rule = self.grammar.rules[rule_id]
        terminals = self._expand_rule(rule)
        pitch_classes = self._extract_pitch_classes(terminals)

        if not pitch_classes:
            return None

        return RulePattern(
            rule_id=rule_id,
            pitch_class=np.array(pitch_classes, dtype=np.int8),
            usage_count=rule.usage_count,
            length=len(pitch_classes),
            depth=0
        )

    def pattern_stats(self, patterns: List[RulePattern]) -> Dict:
        """Get statistics about extracted patterns."""
        if not patterns:
            return {'count': 0}

        lengths = [p.length for p in patterns]
        usages = [p.usage_count for p in patterns]

        return {
            'count': len(patterns),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': np.mean(lengths),
            'total_usage': sum(usages),
            'avg_usage': np.mean(usages),
            'unique_pc_patterns': len(set(tuple(p.pitch_class.tolist()) for p in patterns))
        }


def extract_grammar_patterns(
    grammar: 'SequiturGrammar',
    min_length: int = 4,
    max_length: int = 64,
    min_usage: int = 2,
    verbose: bool = True
) -> List[RulePattern]:
    """
    Convenience function to extract patterns from a grammar.

    Args:
        grammar: SequiturGrammar
        min_length: Minimum pattern length in notes
        max_length: Maximum pattern length
        min_usage: Minimum rule usage count
        verbose: Print statistics

    Returns:
        List of RulePattern objects
    """
    extractor = GrammarPatternExtractor(grammar)
    patterns = extractor.extract_patterns(
        min_length=min_length,
        max_length=max_length,
        min_usage=min_usage
    )

    if verbose:
        stats = extractor.pattern_stats(patterns)
        print(f"\nGrammar Pattern Extraction:")
        print(f"  Total grammar rules: {len(grammar.rules)}")
        print(f"  Patterns extracted: {stats['count']}")
        if stats['count'] > 0:
            print(f"  Length range: {stats['min_length']}-{stats['max_length']} notes")
            print(f"  Avg length: {stats['avg_length']:.1f} notes")
            print(f"  Avg usage: {stats['avg_usage']:.1f}")
            print(f"  Unique pitch patterns: {stats['unique_pc_patterns']}")

    return patterns
