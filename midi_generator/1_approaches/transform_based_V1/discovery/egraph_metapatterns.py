"""
V2: E-Graph Meta-Pattern Discovery

Discovers parameterized lambda functions from V1 derivation graphs.
Groups transforms by structural similarity, extracts parameter distributions.

Example:
    V1 finds: transpose(+4), transpose(+3), transpose(+7)
    V2 abstracts: λ(n): transpose(n) with distribution {+4: 150, +3: 120, +7: 90}

This enables:
- Compression (1 meta-pattern vs 50 specific transforms)
- Generalization (can generate unseen parameter values)
- Musical insight (parameter distributions reveal harmonic preferences)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Callable, Any
import json
import re


@dataclass
class TransformExpr:
    """
    Transform expression that can be structurally compared.

    Supports both primitive transforms and compositions.
    """
    name: str
    amount: float = 0.0
    children: List['TransformExpr'] = field(default_factory=list)
    usage_count: int = 1

    def get_structural_signature(self) -> str:
        """
        Return signature ignoring numeric parameters.

        transpose(+4) -> "transpose(_)"
        transpose(+3) -> "transpose(_)"  (SAME!)
        concat(A, B) -> "concat(_, _)"
        """
        if not self.children:
            # Primitive transform
            return f"{self.name}(_)"
        else:
            # Composite - recurse on children
            child_sigs = [c.get_structural_signature() for c in self.children]
            return f"{self.name}({', '.join(child_sigs)})"

    def get_parameters(self) -> Tuple[float, ...]:
        """Extract numeric parameters from expression as tuple."""
        if not self.children:
            return (self.amount,)
        else:
            params = []
            for c in self.children:
                params.extend(c.get_parameters())
            return tuple(params)

    def __hash__(self):
        return hash((self.name, self.amount, tuple(self.children)))

    def __eq__(self, other):
        if not isinstance(other, TransformExpr):
            return False
        return (self.name == other.name and
                self.amount == other.amount and
                self.children == other.children)


@dataclass
class MetaPattern:
    """
    A discovered meta-pattern (parameterized lambda function).

    Represents an abstraction over multiple specific transforms.
    """
    name: str
    structure: str  # e.g., "transpose(_)"
    parameter_distribution: Dict[Tuple[float, ...], int]  # params -> count
    total_uses: int
    e_class_id: int = 0

    def get_top_parameters(self, n: int = 5) -> List[Tuple[Tuple[float, ...], int]]:
        """Get top N most frequent parameter values."""
        sorted_params = sorted(
            self.parameter_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_params[:n]

    def get_parameter_entropy(self) -> float:
        """
        Calculate entropy of parameter distribution.
        Higher entropy = more diverse parameter usage.
        """
        import math
        total = sum(self.parameter_distribution.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in self.parameter_distribution.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def merge(self, other: 'MetaPattern'):
        """Merge another pattern's statistics into this one."""
        for params, count in other.parameter_distribution.items():
            self.parameter_distribution[params] = (
                self.parameter_distribution.get(params, 0) + count
            )
        self.total_uses += other.total_uses

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'name': self.name,
            'structure': self.structure,
            'parameter_distribution': {
                str(k): v for k, v in self.parameter_distribution.items()
            },
            'total_uses': self.total_uses,
            'e_class_id': self.e_class_id,
            'top_5_params': [
                {'params': list(p), 'count': c}
                for p, c in self.get_top_parameters(5)
            ],
            'entropy': round(self.get_parameter_entropy(), 3),
        }


class EGraph:
    """
    Equivalence graph for transform expressions.

    Groups transforms by structural similarity (ignoring numeric parameters).
    Uses Dict instead of Set to properly aggregate usage counts for identical expressions.
    """

    def __init__(self):
        # e_class_id -> {(name, amount): TransformExpr with aggregated usage_count}
        self.e_classes: Dict[int, Dict[Tuple, TransformExpr]] = {}
        self.signature_to_class: Dict[str, int] = {}  # signature -> e_class_id
        self.expr_to_class: Dict[TransformExpr, int] = {}  # expression -> e_class_id
        self.next_class_id = 0

    def add_expression(self, expr: TransformExpr) -> int:
        """
        Add expression to e-graph.

        Returns e-class ID (may be existing or new class).
        Aggregates usage_count for identical expressions.
        """
        # Get structural signature (e.g., "transpose(_)")
        signature = expr.get_structural_signature()
        # Key for this specific expression within its e-class
        expr_key = (expr.name, expr.amount)

        if signature in self.signature_to_class:
            # Same structure - add to existing class
            e_class = self.signature_to_class[signature]
            if expr_key in self.e_classes[e_class]:
                # Same exact expression - increment usage count
                self.e_classes[e_class][expr_key].usage_count += 1
            else:
                # New parameter variant - add to class
                self.e_classes[e_class][expr_key] = expr
        else:
            # New structure - create new class
            e_class = self.next_class_id
            self.next_class_id += 1
            self.e_classes[e_class] = {expr_key: expr}
            self.signature_to_class[signature] = e_class

        self.expr_to_class[expr] = e_class
        return e_class

    def get_expressions(self, e_class_id: int) -> List[TransformExpr]:
        """Get all expressions in an e-class as a list."""
        if e_class_id not in self.e_classes:
            return []
        return list(self.e_classes[e_class_id].values())

    def get_class_stats(self, e_class_id: int) -> Dict:
        """Get statistics for an e-class."""
        if e_class_id not in self.e_classes:
            return {}

        exprs = self.e_classes[e_class_id].values()  # Dict values
        return {
            'num_variants': len(exprs),
            'total_uses': sum(e.usage_count for e in exprs),
            'structure': list(exprs)[0].get_structural_signature() if exprs else None,
        }


class IncrementalEGraphDiscovery:
    """
    Maintains e-graph across iterations, only processing new derivations.

    This is the key optimization: instead of re-scanning all derivations
    each iteration, we only process newly discovered ones.
    """

    def __init__(self, min_variants: int = 3, min_total_uses: int = 10):
        """
        Args:
            min_variants: Minimum parameter variants to form a meta-pattern
            min_total_uses: Minimum total uses across all variants
        """
        self.e_graph = EGraph()
        self.processed_derivation_ids: Set[str] = set()
        self.meta_patterns: Dict[str, MetaPattern] = {}
        self.min_variants = min_variants
        self.min_total_uses = min_total_uses

    def update_incremental(
        self,
        derivations: List[Any],  # List of Derivation objects
        verbose: bool = True
    ) -> List[MetaPattern]:
        """
        Update e-graph with new derivations.

        Args:
            derivations: List of Derivation objects (must have transform_name, transform_amount)
            verbose: Print progress

        Returns:
            List of newly discovered or updated meta-patterns
        """
        # Filter to truly new derivations
        new_derivations = []
        for d in derivations:
            deriv_id = self._get_derivation_id(d)
            if deriv_id not in self.processed_derivation_ids:
                new_derivations.append(d)
                self.processed_derivation_ids.add(deriv_id)

        if not new_derivations:
            if verbose:
                print("  V2: No new derivations to process")
            return []

        if verbose:
            print(f"  V2: Processing {len(new_derivations)} new derivations "
                  f"(skipping {len(derivations) - len(new_derivations)} already seen)")

        # Add to e-graph
        affected_classes: Set[int] = set()
        for deriv in new_derivations:
            expr = self._derivation_to_expr(deriv)
            e_class = self.e_graph.add_expression(expr)
            affected_classes.add(e_class)

        # Analyze affected e-classes
        updated_patterns = []
        for e_class_id in affected_classes:
            pattern = self._analyze_e_class(e_class_id)
            if pattern:
                pattern_key = pattern.structure

                if pattern_key in self.meta_patterns:
                    # Update existing pattern
                    self.meta_patterns[pattern_key].merge(pattern)
                    updated_patterns.append(self.meta_patterns[pattern_key])
                else:
                    # New pattern discovered
                    self.meta_patterns[pattern_key] = pattern
                    updated_patterns.append(pattern)

        if verbose:
            print(f"  V2: Updated {len(updated_patterns)} meta-patterns")
            print(f"  V2: Total meta-patterns: {len(self.meta_patterns)}")

        return updated_patterns

    def _get_derivation_id(self, deriv: Any) -> str:
        """Create unique ID for a derivation."""
        # Use target object ID + transform to identify
        target_id = id(deriv.target) if hasattr(deriv, 'target') else id(deriv)
        return f"{target_id}:{deriv.transform_name}:{deriv.transform_amount}"

    def _derivation_to_expr(self, deriv: Any) -> TransformExpr:
        """Convert a Derivation object to TransformExpr."""
        return TransformExpr(
            name=deriv.transform_name,
            amount=deriv.transform_amount,
            children=[],
            usage_count=1,
        )

    def _analyze_e_class(self, e_class_id: int) -> Optional[MetaPattern]:
        """
        Analyze single e-class for meta-pattern potential.

        Returns MetaPattern if class meets thresholds, else None.
        """
        if e_class_id not in self.e_graph.e_classes:
            return None

        # Get expressions from the Dict (values are TransformExpr)
        expressions = list(self.e_graph.e_classes[e_class_id].values())

        # Check minimum variants (different parameter values)
        if len(expressions) < self.min_variants:
            return None

        # Get structural signature from first expression
        structure = expressions[0].get_structural_signature()

        # Collect parameter distributions (already aggregated in usage_count)
        param_counts: Dict[Tuple[float, ...], int] = defaultdict(int)
        for expr in expressions:
            params = expr.get_parameters()
            param_counts[params] = expr.usage_count

        total_uses = sum(param_counts.values())

        # Check minimum uses
        if total_uses < self.min_total_uses:
            return None

        return MetaPattern(
            name=f"meta_{structure.replace('(', '_').replace(')', '').replace(', ', '_')}",
            structure=structure,
            parameter_distribution=dict(param_counts),
            total_uses=total_uses,
            e_class_id=e_class_id,
        )

    def get_all_patterns(self) -> List[MetaPattern]:
        """Get all discovered meta-patterns sorted by usage."""
        return sorted(
            self.meta_patterns.values(),
            key=lambda p: p.total_uses,
            reverse=True
        )

    def get_pattern_summary(self) -> str:
        """Get human-readable summary of discovered patterns."""
        patterns = self.get_all_patterns()
        if not patterns:
            return "No meta-patterns discovered yet."

        lines = [
            f"Discovered {len(patterns)} meta-patterns:",
            ""
        ]

        for i, pattern in enumerate(patterns[:10]):  # Top 10
            top_params = pattern.get_top_parameters(3)
            param_str = ", ".join(
                f"{p}:{c}" for p, c in top_params
            )
            lines.append(
                f"  {i+1}. {pattern.structure} "
                f"({pattern.total_uses} uses, {len(pattern.parameter_distribution)} variants)"
            )
            lines.append(f"      Top params: {param_str}")

        if len(patterns) > 10:
            lines.append(f"  ... and {len(patterns) - 10} more")

        return "\n".join(lines)

    def export_patterns(self, filepath: str):
        """Export patterns to JSON file."""
        patterns = self.get_all_patterns()
        data = {
            'meta_patterns': [p.to_dict() for p in patterns],
            'total_patterns': len(patterns),
            'total_derivations_processed': len(self.processed_derivation_ids),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_musical_insights(self) -> Dict[str, Any]:
        """
        Extract musical insights from parameter distributions.

        Returns insights about harmonic/rhythmic patterns.
        """
        insights = {}

        # Look for transpose patterns
        transpose_pattern = self.meta_patterns.get('meta_transpose__')
        if transpose_pattern:
            params = transpose_pattern.parameter_distribution
            insights['transposition'] = {
                'total_uses': transpose_pattern.total_uses,
                'num_intervals': len(params),
                'most_common': transpose_pattern.get_top_parameters(5),
                'analysis': self._analyze_intervals(params),
            }

        # Look for time_scale patterns
        time_scale_pattern = self.meta_patterns.get('meta_time_scale__')
        if time_scale_pattern:
            params = time_scale_pattern.parameter_distribution
            insights['time_scale'] = {
                'total_uses': time_scale_pattern.total_uses,
                'num_factors': len(params),
                'most_common': time_scale_pattern.get_top_parameters(5),
                'analysis': self._analyze_time_factors(params),
            }

        return insights

    def _analyze_intervals(self, params: Dict[Tuple[float, ...], int]) -> Dict:
        """Analyze transposition intervals for musical meaning."""
        analysis = {
            'thirds': 0,  # ±3, ±4
            'fifths': 0,  # ±5, ±7
            'octaves': 0,  # ±12
            'chromatic': 0,  # ±1, ±2
        }

        for (interval,), count in params.items():
            abs_int = abs(int(interval))
            if abs_int in (3, 4):
                analysis['thirds'] += count
            elif abs_int in (5, 7):
                analysis['fifths'] += count
            elif abs_int == 12 or abs_int == 0:
                analysis['octaves'] += count
            elif abs_int in (1, 2):
                analysis['chromatic'] += count

        return analysis

    def _analyze_time_factors(self, params: Dict[Tuple[float, ...], int]) -> Dict:
        """Analyze time scaling factors for rhythmic patterns."""
        analysis = {
            'halving': 0,  # 0.5
            'doubling': 0,  # 2.0
            'triplet': 0,  # 0.667, 1.5
            'other': 0,
        }

        for (factor,), count in params.items():
            if abs(factor - 0.5) < 0.01:
                analysis['halving'] += count
            elif abs(factor - 2.0) < 0.01:
                analysis['doubling'] += count
            elif abs(factor - 0.667) < 0.05 or abs(factor - 1.5) < 0.05:
                analysis['triplet'] += count
            else:
                analysis['other'] += count

        return analysis


def integrate_v2_with_v1(
    v1_discovery,  # EmergentHierarchyDiscovery instance
    max_iterations: int = 5,
    verbose: bool = True
) -> Tuple[Dict, Dict[str, MetaPattern]]:
    """
    Run V1 discovery with incremental V2 meta-pattern extraction.

    This is the main integration point between V1 and V2.

    Args:
        v1_discovery: Initialized EmergentHierarchyDiscovery
        max_iterations: Maximum discovery iterations
        verbose: Print progress

    Returns:
        (v1_derivation_graph, v2_meta_patterns)
    """
    v2_incremental = IncrementalEGraphDiscovery()

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*60}")

        # V1: Discover derivations (expensive)
        if verbose:
            print("\n  V1: Discovering derivations...")

        # Get derivations from current graph
        derivations = list(v1_discovery.derivation_graph.values())
        new_count = len(derivations) - len(v2_incremental.processed_derivation_ids)

        if verbose:
            print(f"  V1: {len(derivations)} total derivations, {new_count} new")

        # V2: Incremental lambda extraction (cheap)
        if verbose:
            print("\n  V2: Incremental meta-pattern update...")

        updated_patterns = v2_incremental.update_incremental(derivations, verbose)

        # Check convergence
        if new_count == 0:
            if verbose:
                print("\n  Converged - no new derivations")
            break

    # Final summary
    if verbose:
        print(f"\n{'='*60}")
        print("FINAL V2 SUMMARY")
        print(f"{'='*60}")
        print(v2_incremental.get_pattern_summary())

        insights = v2_incremental.get_musical_insights()
        if insights:
            print("\nMusical Insights:")
            for category, data in insights.items():
                print(f"  {category}: {data.get('analysis', {})}")

    return v1_discovery.derivation_graph, v2_incremental.meta_patterns


# Standalone usage for post-processing V1 results
def analyze_v1_results(derivation_graph: Dict, output_path: str = None, verbose: bool = True):
    """
    Analyze V1 derivation graph to extract V2 meta-patterns.

    Can be run after V1 discovery completes.

    Args:
        derivation_graph: V1 derivation graph (target -> Derivation)
        output_path: Optional path to save results JSON
        verbose: Print progress

    Returns:
        IncrementalEGraphDiscovery instance with patterns
    """
    if verbose:
        print(f"\n{'='*60}")
        print("V2 META-PATTERN ANALYSIS")
        print(f"{'='*60}")
        print(f"  Analyzing {len(derivation_graph)} derivations...")

    v2 = IncrementalEGraphDiscovery(min_variants=2, min_total_uses=5)
    derivations = list(derivation_graph.values())
    v2.update_incremental(derivations, verbose)

    if verbose:
        print(f"\n{v2.get_pattern_summary()}")

        insights = v2.get_musical_insights()
        if insights:
            print("\nMusical Insights:")
            for category, data in insights.items():
                print(f"  {category}:")
                for k, v in data.get('analysis', {}).items():
                    print(f"    {k}: {v}")

    if output_path:
        v2.export_patterns(output_path)
        if verbose:
            print(f"\n  Exported to: {output_path}")

    return v2
