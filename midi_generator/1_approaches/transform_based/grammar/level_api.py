"""
Level-Aware Grammar API
=======================

Access patterns by depth level, not just canonical ID.
Supports both V32/V33 hierarchical checkpoints and V24+ factored checkpoints.

Usage:
    from grammar.level_api import GrammarHierarchy

    grammar = GrammarHierarchy.from_checkpoint('checkpoint_v2.npz')

    # Get patterns by semantic level
    motifs = grammar.get_patterns_at_level('motif')  # depth 3-4
    phrases = grammar.get_patterns_at_level('phrase')  # depth 5-6

    # Get patterns by depth
    depth_3 = grammar.get_patterns_at_depth(3)

    # Expand to terminals
    terminals = grammar.expand_rule(rule_id)
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Iterator, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Rule:
    """A grammar rule with computed metadata."""
    id: str
    raw: Any  # Original rule data (list for hierarchical, dict for factored)
    depth: int = 0
    terminal_count: int = 0
    child_rule_count: int = 0
    expansion_length: int = 0  # Length when fully expanded
    children: List[str] = field(default_factory=list)
    # Factored fields (v24+)
    pitch_classes: List[int] = field(default_factory=list)
    octaves: List[int] = field(default_factory=list)
    velocities: List[int] = field(default_factory=list)
    durations: List[int] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)
    occurrence_count: int = 0


@dataclass
class LevelStats:
    """Statistics for a depth level."""
    depth: int
    semantic_level: str
    rule_count: int
    avg_expansion_length: float
    min_expansion_length: int
    max_expansion_length: int
    total_usage: int = 0


SEMANTIC_LEVELS = {
    1: 'interval',
    2: 'interval',
    3: 'motif',
    4: 'motif',
    5: 'phrase',
    6: 'phrase',
    7: 'section',
    8: 'section',
    9: 'large-scale',
    10: 'large-scale',
    11: 'large-scale',
    12: 'corpus',
}


class GrammarHierarchy:
    """
    Level-aware interface to grammar hierarchy.

    Provides access to patterns by:
    - Depth level (1-12)
    - Semantic level (interval, motif, phrase, section, large-scale)
    - Parent/child relationships

    Supports both hierarchical (V32/V33) and factored (V24+) checkpoint formats.
    """

    def __init__(self, rules_dict: Dict, is_factored: bool = False):
        """
        Initialize from grammar rules dictionary.

        Args:
            rules_dict: Dict mapping rule_id (str) -> rule_data
                - For hierarchical: rule_data is a list of elements
                - For factored: rule_data is a dict with tokens, pitch_classes, etc.
            is_factored: Whether this is a factored checkpoint (v24+)
        """
        self.raw_rules = rules_dict
        self.rules: Dict[str, Rule] = {}
        self.by_depth: Dict[int, List[str]] = defaultdict(list)
        self.by_level: Dict[str, List[str]] = defaultdict(list)
        self.expansion_cache: Dict[str, List[int]] = {}
        self.is_factored = is_factored

        # Factored-specific data
        self.transform_vocab: List[str] = []
        self.meta_patterns: List[Dict] = []

        if is_factored:
            self._build_factored_hierarchy()
        else:
            self._build_hierarchy()

    def _build_factored_hierarchy(self):
        """Build hierarchy from factored checkpoint format (v24+).

        Factored rules have flat structure with factored note dimensions.
        Depth is computed from pattern length (number of notes).
        """
        # Length-to-depth mapping for factored patterns
        # Short patterns (2-3 notes) = depth 1-2 (interval)
        # Medium patterns (4-8 notes) = depth 3-4 (motif)
        # Longer patterns (9-16 notes) = depth 5-6 (phrase)
        # Very long patterns (17+ notes) = depth 7+ (section)
        def length_to_depth(length: int) -> int:
            if length <= 2:
                return 1
            elif length <= 3:
                return 2
            elif length <= 5:
                return 3
            elif length <= 8:
                return 4
            elif length <= 12:
                return 5
            elif length <= 16:
                return 6
            elif length <= 24:
                return 7
            elif length <= 32:
                return 8
            elif length <= 48:
                return 9
            elif length <= 64:
                return 10
            elif length <= 96:
                return 11
            else:
                return 12

        # Build Rule objects from factored pattern data
        for rule_id, rule_data in self.raw_rules.items():
            if isinstance(rule_data, dict):
                # Factored format: dict with tokens, pitch_classes, etc.
                tokens = rule_data.get('tokens', [])
                pitch_classes = rule_data.get('pitch_classes', [])
                octaves = rule_data.get('octaves', [])
                velocities = rule_data.get('velocities', [])
                durations = rule_data.get('duration_buckets', rule_data.get('durations', []))
                occurrences = rule_data.get('occurrences', [])

                pattern_length = len(tokens) if tokens else len(pitch_classes)
                depth = length_to_depth(pattern_length)

                self.rules[rule_id] = Rule(
                    id=rule_id,
                    raw=rule_data,
                    depth=depth,
                    terminal_count=pattern_length,
                    child_rule_count=0,  # Factored patterns are flat
                    expansion_length=pattern_length,
                    children=[],  # No children in factored format
                    pitch_classes=pitch_classes,
                    octaves=octaves,
                    velocities=velocities,
                    durations=durations,
                    tokens=tokens,
                    occurrence_count=len(occurrences),
                )

                # Cache expansion (tokens are the terminals)
                self.expansion_cache[rule_id] = tokens
            else:
                # Fallback for unexpected format
                self.rules[rule_id] = Rule(
                    id=rule_id,
                    raw=rule_data,
                    depth=1,
                )

        # Build indices
        for rule_id, rule in self.rules.items():
            self.by_depth[rule.depth].append(rule_id)
            level = SEMANTIC_LEVELS.get(rule.depth, 'unknown')
            self.by_level[level].append(rule_id)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> 'GrammarHierarchy':
        """Load grammar hierarchy from checkpoint file."""
        import os
        data = np.load(checkpoint_path, allow_pickle=True)
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Detect if this is a factored checkpoint
        is_factored = 'is_factored' in data and bool(data['is_factored'].item()) if 'is_factored' in data else False

        # Handle different checkpoint formats
        rules_dict = None

        # v4+ format: patterns in external JSON file
        if 'patterns_json_file' in data:
            patterns_file = data['patterns_json_file'].item()
            patterns_path = os.path.join(checkpoint_dir, patterns_file)
            with open(patterns_path, 'r') as f:
                rules_dict = json.load(f)
        # v33 format: patterns_json inline
        elif 'patterns_json' in data:
            patterns_json = data['patterns_json']
            if hasattr(patterns_json, 'item'):
                rules_dict = json.loads(str(patterns_json.item()))
            else:
                rules_dict = json.loads(str(patterns_json))
        # Legacy format: grammar_rules_json
        elif 'grammar_rules_json' in data:
            rules_json = data['grammar_rules_json']
            if hasattr(rules_json, '__len__') and len(rules_json) == 1:
                rules_dict = json.loads(str(rules_json[0]))
            else:
                rules_dict = json.loads(str(rules_json.item()))
        else:
            raise ValueError("No grammar_rules_json or patterns_json_file in checkpoint")

        instance = cls(rules_dict, is_factored=is_factored)

        # Load factored-specific data
        # v4+ format: external JSON files
        if 'transforms_json_file' in data:
            transforms_file = data['transforms_json_file'].item()
            transforms_path = os.path.join(checkpoint_dir, transforms_file)
            if os.path.exists(transforms_path):
                with open(transforms_path, 'r') as f:
                    instance.transform_vocab = json.load(f)
        elif 'transform_vocabulary_json' in data:
            vocab_json = data['transform_vocabulary_json']
            if hasattr(vocab_json, '__len__') and len(vocab_json) == 1:
                instance.transform_vocab = json.loads(str(vocab_json[0]))
            else:
                instance.transform_vocab = json.loads(str(vocab_json.item()))

        if 'meta_patterns_json' in data:
            meta_json = data['meta_patterns_json']
            if hasattr(meta_json, '__len__') and len(meta_json) == 1:
                instance.meta_patterns = json.loads(str(meta_json[0]))
            else:
                instance.meta_patterns = json.loads(str(meta_json.item()))

        return instance

    @classmethod
    def from_json(cls, json_path: str) -> 'GrammarHierarchy':
        """Load grammar hierarchy from JSON file."""
        with open(json_path) as f:
            rules_dict = json.load(f)
        return cls(rules_dict)

    def _build_hierarchy(self):
        """Compute depths and build indices."""
        # First pass: create Rule objects
        for rule_id, rule_data in self.raw_rules.items():
            children = []
            terminal_count = 0

            for element in rule_data:
                if isinstance(element, list) and len(element) == 2 and element[0] == 'R':
                    children.append(str(element[1]))
                elif isinstance(element, int):
                    terminal_count += 1

            self.rules[rule_id] = Rule(
                id=rule_id,
                raw=rule_data,
                terminal_count=terminal_count,
                child_rule_count=len(children),
                children=children,
            )

        # Second pass: compute depths
        for rule_id in self.rules:
            self._compute_depth(rule_id)

        # Third pass: compute expansion lengths
        for rule_id in self.rules:
            exp = self._expand_rule(rule_id)
            self.rules[rule_id].expansion_length = len(exp)

        # Build indices
        for rule_id, rule in self.rules.items():
            self.by_depth[rule.depth].append(rule_id)
            level = SEMANTIC_LEVELS.get(rule.depth, 'unknown')
            self.by_level[level].append(rule_id)

    def _compute_depth(self, rule_id: str, visited: set = None) -> int:
        """Compute depth of a rule (memoized)."""
        if visited is None:
            visited = set()

        rule = self.rules.get(rule_id)
        if rule is None:
            return 0

        if rule.depth > 0:
            return rule.depth

        if rule_id in visited:
            return 1  # Cycle - shouldn't happen

        visited.add(rule_id)

        if not rule.children:
            rule.depth = 1
        else:
            child_depths = [self._compute_depth(c, visited) for c in rule.children]
            rule.depth = max(child_depths) + 1

        return rule.depth

    def _expand_rule(self, rule_id: str) -> List[int]:
        """Expand rule to terminal sequence (memoized)."""
        if rule_id in self.expansion_cache:
            return self.expansion_cache[rule_id]

        rule = self.rules.get(rule_id)
        if rule is None:
            try:
                return [int(rule_id)]
            except ValueError:
                return []

        result = []
        for element in rule.raw:
            if isinstance(element, list) and len(element) == 2 and element[0] == 'R':
                child_id = str(element[1])
                result.extend(self._expand_rule(child_id))
            elif isinstance(element, int):
                result.append(element)

        self.expansion_cache[rule_id] = result
        return result

    # =========== Public API ===========

    def get_max_depth(self) -> int:
        """Get maximum depth in hierarchy."""
        return max(self.by_depth.keys()) if self.by_depth else 0

    def get_rule_count(self) -> int:
        """Get total number of rules."""
        return len(self.rules)

    def get_patterns_at_depth(self, depth: int) -> List[Rule]:
        """Get all patterns at a specific depth."""
        return [self.rules[rid] for rid in self.by_depth.get(depth, [])]

    def get_patterns_at_level(self, level: str) -> List[Rule]:
        """
        Get patterns at a semantic level.

        Args:
            level: 'interval', 'motif', 'phrase', 'section', or 'large-scale'

        Returns:
            List of Rule objects at that level
        """
        return [self.rules[rid] for rid in self.by_level.get(level, [])]

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a specific rule by ID."""
        return self.rules.get(str(rule_id))

    def expand_rule(self, rule_id: str) -> List[int]:
        """Expand a rule to its terminal sequence."""
        return self._expand_rule(str(rule_id))

    def get_children(self, rule_id: str) -> List[Rule]:
        """Get immediate child rules."""
        rule = self.rules.get(str(rule_id))
        if rule is None:
            return []
        return [self.rules[cid] for cid in rule.children if cid in self.rules]

    def get_parent_rules(self, rule_id: str) -> List[Rule]:
        """Get all rules that reference this rule."""
        target = str(rule_id)
        parents = []
        for rule in self.rules.values():
            if target in rule.children:
                parents.append(rule)
        return parents

    def get_level_stats(self) -> Dict[int, LevelStats]:
        """Get statistics for each depth level."""
        stats = {}

        for depth, rule_ids in sorted(self.by_depth.items()):
            rules = [self.rules[rid] for rid in rule_ids]
            lengths = [r.expansion_length for r in rules]

            stats[depth] = LevelStats(
                depth=depth,
                semantic_level=SEMANTIC_LEVELS.get(depth, 'unknown'),
                rule_count=len(rules),
                avg_expansion_length=np.mean(lengths) if lengths else 0,
                min_expansion_length=min(lengths) if lengths else 0,
                max_expansion_length=max(lengths) if lengths else 0,
            )

        return stats

    def iterate_by_depth(self) -> Iterator[Tuple[int, List[Rule]]]:
        """Iterate through all rules grouped by depth."""
        for depth in sorted(self.by_depth.keys()):
            yield depth, [self.rules[rid] for rid in self.by_depth[depth]]

    def get_subtree(self, rule_id: str, max_depth: int = None) -> Dict:
        """
        Get rule and all its descendants as a tree structure.

        Args:
            rule_id: Root rule ID
            max_depth: Maximum depth to descend (None = unlimited)

        Returns:
            Dict with 'rule', 'children', 'expansion'
        """
        def build_subtree(rid: str, current_depth: int) -> Dict:
            rule = self.rules.get(str(rid))
            if rule is None:
                return {'terminal': int(rid) if rid.isdigit() else rid}

            result = {
                'rule': rule,
                'depth': rule.depth,
                'expansion_length': rule.expansion_length,
            }

            if max_depth is not None and current_depth >= max_depth:
                result['children'] = f"[{len(rule.children)} children truncated]"
            else:
                result['children'] = [
                    build_subtree(cid, current_depth + 1)
                    for cid in rule.children
                ]

            return result

        return build_subtree(str(rule_id), 0)

    def to_json_export(self) -> Dict:
        """Export hierarchy data for web interface."""
        return {
            'max_depth': self.get_max_depth(),
            'rule_count': self.get_rule_count(),
            'levels': {
                depth: {
                    'semantic': SEMANTIC_LEVELS.get(depth, 'unknown'),
                    'count': len(rule_ids),
                    'rules': [
                        {
                            'id': rid,
                            'depth': self.rules[rid].depth,
                            'expansion_length': self.rules[rid].expansion_length,
                            'children': self.rules[rid].children[:10],  # Limit for JSON size
                            'child_count': len(self.rules[rid].children),
                        }
                        for rid in rule_ids[:100]  # Limit per level
                    ]
                }
                for depth, rule_ids in sorted(self.by_depth.items())
            },
            'stats': {
                depth: {
                    'semantic': stats.semantic_level,
                    'count': stats.rule_count,
                    'avg_length': round(stats.avg_expansion_length, 1),
                    'min_length': stats.min_expansion_length,
                    'max_length': stats.max_expansion_length,
                }
                for depth, stats in self.get_level_stats().items()
            }
        }


def export_hierarchy_json(checkpoint_path: str, output_path: str):
    """Export grammar hierarchy to JSON for web interface."""
    grammar = GrammarHierarchy.from_checkpoint(checkpoint_path)
    export = grammar.to_json_export()

    with open(output_path, 'w') as f:
        json.dump(export, f, indent=2)

    print(f"Exported {grammar.get_rule_count()} rules to {output_path}")
    return export


if __name__ == '__main__':
    import sys

    checkpoint = sys.argv[1] if len(sys.argv) > 1 else 'checkpoint_v2.npz'

    grammar = GrammarHierarchy.from_checkpoint(checkpoint)

    print(f"=== Grammar Hierarchy ===")
    print(f"Total rules: {grammar.get_rule_count()}")
    print(f"Max depth: {grammar.get_max_depth()}")

    print(f"\n=== Level Statistics ===")
    for depth, stats in grammar.get_level_stats().items():
        print(f"Depth {depth:2d} ({stats.semantic_level:12s}): "
              f"{stats.rule_count:5d} rules, "
              f"avg length {stats.avg_expansion_length:6.1f}")

    print(f"\n=== Sample by Semantic Level ===")
    for level in ['motif', 'phrase', 'section']:
        patterns = grammar.get_patterns_at_level(level)[:3]
        print(f"\n{level.upper()} ({len(grammar.by_level[level])} total):")
        for p in patterns:
            exp = grammar.expand_rule(p.id)[:20]
            print(f"  R{p.id}: {len(exp)} terminals, {p.child_rule_count} children")
            print(f"    First 20: {exp}")
