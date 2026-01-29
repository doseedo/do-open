# V2: E-Graph Meta-Pattern Abstraction

## The Gap: What V1 Produces vs. What We Want

### Current V1 System Output
```python
# V1 learns many specific derivations:
derivation_1 = Derivation(transpose(+4), source_1)  # Major 3rd
derivation_2 = Derivation(transpose(+3), source_2)  # Minor 3rd
derivation_3 = Derivation(transpose(+4), source_3)  # Major 3rd
derivation_4 = Derivation(transpose(+3), source_4)  # Minor 3rd

# Dictionary contains:
transforms = {
    'transpose(+4)': 150 uses,  # Major 3rd
    'transpose(+3)': 120 uses,  # Minor 3rd
    'transpose(+7)': 90 uses,   # Perfect 5th
    # ... many more specific transforms
}
```

**Problem**: Treats `transpose(+3)` and `transpose(+4)` as completely unrelated operations.

### Ideal V2 System Output
```python
# V2 abstracts to meta-patterns (lambda functions):
meta_patterns = {
    'diatonic_interval': λ(n) → transpose(n),  # Parameterized transposition
    'harmonic_voicing': λ(intervals) → stack([transpose(i) for i in intervals]),
}

# With usage statistics:
'diatonic_interval': {
    'total_uses': 360,
    'parameter_distribution': {
        +4: 150,  # Major 3rd
        +3: 120,  # Minor 3rd
        +7: 90,   # Perfect 5th
    }
}
```

---

## The Bridge: E-Graphs for Pattern Abstraction

### What Are E-Graphs?

E-graphs (equivalence graphs) are data structures that compactly represent many equivalent expressions. Perfect for discovering when different derivations are "the same operation with different parameters."

### How E-Graphs Enable Meta-Patterns

```
Step 1: Build Derivation Graph (V1 does this)
┌─────────────────────────────────────────────┐
│  source_A ──transpose(+4)──> harmony_A      │
│  source_B ──transpose(+3)──> harmony_B      │
│  source_C ──transpose(+4)──> harmony_C      │
│  source_D ──transpose(+7)──> bass_D         │
└─────────────────────────────────────────────┘

Step 2: E-Graph Equivalence Classes (V2 adds this)
┌─────────────────────────────────────────────┐
│  E-CLASS: "transposition"                    │
│  ├── transpose(+4)                          │
│  ├── transpose(+3)                          │
│  └── transpose(+7)                          │
│                                              │
│  E-CLASS: "harmonic_derivation"             │
│  ├── transpose(+4) ∘ TrackDerive(harmony)   │
│  ├── transpose(+3) ∘ TrackDerive(harmony)   │
│  └── (same structure, different params)     │
└─────────────────────────────────────────────┘

Step 3: Meta-Pattern Extraction
┌─────────────────────────────────────────────┐
│  harmonic_third = λ(n): transpose(n) ∘ X    │
│    where X ∈ TrackDerive(harmony_track)     │
│                                              │
│  Instances:                                  │
│    harmonic_third(+4) → major harmony       │
│    harmonic_third(+3) → minor harmony       │
└─────────────────────────────────────────────┘
```

---

## Implementation Architecture

### Phase 1: E-Graph Construction

```python
class EGraph:
    """
    Equivalence graph for transform patterns.
    Groups transforms by structural similarity.
    """

    def __init__(self):
        self.e_classes = {}  # e_class_id -> set of equivalent expressions
        self.expr_to_class = {}  # expression -> e_class_id

    def add_expression(self, expr: TransformExpr):
        """Add expression, merging with equivalent classes."""
        # Get structural signature (ignoring numeric params)
        signature = expr.get_structural_signature()

        if signature in self.signature_to_class:
            # Same structure - add to existing class
            e_class = self.signature_to_class[signature]
            self.e_classes[e_class].add(expr)
        else:
            # New structure - create new class
            e_class = self.new_class()
            self.e_classes[e_class] = {expr}
            self.signature_to_class[signature] = e_class

        self.expr_to_class[expr] = e_class
        return e_class
```

### Phase 2: Structural Signature Extraction

```python
class TransformExpr:
    """Transform expression that can be structurally compared."""

    def get_structural_signature(self) -> str:
        """
        Return signature ignoring numeric parameters.

        transpose(+4) -> "transpose(_)"
        transpose(+3) -> "transpose(_)"  (SAME!)

        time_scale(2.0) -> "time_scale(_)"
        time_scale(0.5) -> "time_scale(_)"  (SAME!)

        concat(A, B) -> "concat(_, _)"
        """
        if self.is_primitive():
            return f"{self.name}(_)"
        else:
            # Composite - recurse on children
            child_sigs = [c.get_structural_signature() for c in self.children]
            return f"{self.name}({', '.join(child_sigs)})"

    def get_parameters(self) -> List[float]:
        """Extract numeric parameters from expression."""
        if self.is_primitive():
            return [self.amount]
        else:
            params = []
            for c in self.children:
                params.extend(c.get_parameters())
            return params
```

### Phase 3: Meta-Pattern Discovery

```python
def discover_meta_patterns(e_graph: EGraph, min_variants: int = 3) -> List[MetaPattern]:
    """
    Discover meta-patterns from e-graph equivalence classes.

    A meta-pattern is an e-class with multiple parameter variants
    that all appear frequently in the corpus.
    """
    meta_patterns = []

    for e_class_id, expressions in e_graph.e_classes.items():
        if len(expressions) < min_variants:
            continue

        # Get common structure
        structure = expressions[0].get_structural_signature()

        # Collect parameter distributions
        param_counts = defaultdict(int)
        for expr in expressions:
            params = tuple(expr.get_parameters())
            param_counts[params] += expr.usage_count

        # Create meta-pattern
        meta = MetaPattern(
            name=f"meta_{structure}",
            structure=structure,
            parameter_distribution=dict(param_counts),
            total_uses=sum(param_counts.values()),
            lambda_fn=create_lambda(structure)
        )
        meta_patterns.append(meta)

    return meta_patterns
```

### Phase 4: Lambda Function Generation

```python
def create_lambda(structure: str) -> Callable:
    """
    Create parameterized lambda from structural signature.

    "transpose(_)" -> λ(n): transpose(n)
    "concat(transpose(_), transpose(_))" -> λ(a, b): concat(transpose(a), transpose(b))
    """
    # Parse structure to AST
    ast = parse_signature(structure)

    # Find parameter placeholders
    params = find_placeholders(ast)

    # Generate lambda
    param_names = [f"p{i}" for i in range(len(params))]

    def lambda_fn(*args):
        # Fill in placeholders with args
        filled = fill_placeholders(ast, dict(zip(param_names, args)))
        return evaluate(filled)

    return lambda_fn
```

---

## Example: Big Band Voicing Patterns

### Input (V1 Derivations)
```
piece_1:
  sax_1 = transpose(+4)(lead_sax)   # Major 3rd above
  sax_2 = transpose(+3)(lead_sax)   # Minor 3rd above
  sax_3 = transpose(0)(lead_sax)    # Unison (lead)
  sax_4 = transpose(-3)(lead_sax)   # Minor 3rd below

piece_2:
  sax_1 = transpose(+7)(lead_sax)   # Perfect 5th above
  sax_2 = transpose(+4)(lead_sax)   # Major 3rd above
  sax_3 = transpose(0)(lead_sax)    # Unison (lead)
  sax_4 = transpose(-5)(lead_sax)   # Perfect 4th below
```

### Output (V2 Meta-Patterns)
```python
meta_patterns = [
    MetaPattern(
        name='section_voicing',
        structure='transpose(_)',
        description='Parallel harmony voicing from lead',
        parameter_distribution={
            (+4,): 45,   # Major 3rd (most common)
            (+3,): 38,   # Minor 3rd
            (+7,): 25,   # Perfect 5th
            (0,): 50,    # Unison (always present)
            (-3,): 20,   # Minor 3rd below
            (-5,): 15,   # Perfect 4th below
        },
        lambda_fn=lambda n: Transpose(n),

        # Higher-order insight:
        musical_interpretation={
            'positive_intervals': 'Upper voices (harmonize above)',
            'negative_intervals': 'Lower voices (harmonize below)',
            'zero': 'Lead voice (melodic reference)',
        }
    ),

    MetaPattern(
        name='closed_voicing_stack',
        structure='concat(transpose(_), transpose(_), transpose(_))',
        description='Three-note close-position chord',
        parameter_distribution={
            (+4, +7, 0): 30,  # Major triad
            (+3, +7, 0): 25,  # Minor triad
            (+4, +8, 0): 10,  # Augmented
        },
        lambda_fn=lambda a, b, c: ConcatPar(
            Transpose(a), Transpose(b), Transpose(c)
        ),
    )
]
```

---

## Integration with V1 Discovery Pipeline

### Where V2 Plugs In

```
V1 Pipeline:
┌─────────────────────────────────────────────┐
│ 1. Object Extraction                        │
│ 2. FAISS Cross-Piece Derivation Search      │
│ 3. MDL Dictionary Learning                  │
│ 4. Composition Pattern Discovery            │
│         ↓                                   │
│    [DERIVATION GRAPH]                       │
│         ↓                                   │
│ ┌─────────────────────────────────────────┐ │
│ │ V2: E-Graph Meta-Pattern Discovery      │ │  ← NEW PHASE
│ │   • Build E-Graph from derivations      │ │
│ │   • Cluster by structural similarity    │ │
│ │   • Extract parameterized lambdas       │ │
│ │   • Compute parameter distributions     │ │
│ └─────────────────────────────────────────┘ │
│         ↓                                   │
│    [META-PATTERN LIBRARY]                   │
└─────────────────────────────────────────────┘
```

### API for V2 Integration

```python
# After V1 discovery completes:
from v2.egraph_patterns import EGraphMetaPatternDiscovery

# Build e-graph from V1 derivations
egraph_discovery = EGraphMetaPatternDiscovery()
egraph_discovery.build_from_derivations(derivation_graph)

# Discover meta-patterns
meta_patterns = egraph_discovery.discover_patterns(
    min_variants=3,      # At least 3 parameter variants
    min_total_uses=20,   # Used at least 20 times total
)

# Export results
egraph_discovery.export_patterns('meta_patterns.json')
egraph_discovery.visualize_egraph('egraph_visualization.html')
```

---

## Benefits of V2

1. **Compression**: Instead of 50 specific transpose transforms, learn 1 meta-pattern with parameter distribution
2. **Generalization**: Can generate new variants (e.g., transpose(+6) for tritone) even if never seen
3. **Musical Insight**: Parameter distributions reveal harmonic preferences (major vs minor prevalence)
4. **Compositionality**: Meta-patterns can compose to form even higher abstractions

---

## Critical Optimization: Incremental Discovery

### The Problem: Naive Re-scanning

```python
# DON'T do this (slow):
for iteration in range(max_iterations):
    discover_derivations()  # 2 hours
    discover_ALL_lambdas()  # 30 min scanning ALL derivations

# Total: 2.5 hours per iteration
# With 5 iterations: 12.5 hours!
```

### The Solution: Incremental Lambda Discovery

```python
# DO this (fast):
for iteration in range(max_iterations):
    new_derivations = discover_derivations()  # 2 hours
    discover_lambdas_incremental(new_derivations)  # 5 min - only NEW derivations

# Total: ~2.1 hours per iteration
# With 5 iterations: 10.5 hours (saves 2 hours)
```

### Implementation: Incremental E-Graph Updates

```python
class IncrementalEGraphDiscovery:
    """
    Maintains e-graph across iterations, only processing new derivations.
    """

    def __init__(self):
        self.e_graph = EGraph()
        self.processed_derivations = set()  # Track what we've already seen
        self.meta_patterns = {}  # Running pattern cache

    def update_incremental(self, new_derivations: List[Derivation]) -> List[MetaPattern]:
        """
        Update e-graph with only new derivations.
        Returns newly discovered or updated meta-patterns.
        """
        # Filter to truly new derivations
        truly_new = [
            d for d in new_derivations
            if d.id not in self.processed_derivations
        ]

        if not truly_new:
            return []

        print(f"  Processing {len(truly_new)} new derivations (skipping {len(new_derivations) - len(truly_new)} already seen)")

        # Add to e-graph
        affected_classes = set()
        for deriv in truly_new:
            expr = self._derivation_to_expr(deriv)
            e_class = self.e_graph.add_expression(expr)
            affected_classes.add(e_class)
            self.processed_derivations.add(deriv.id)

        # Only re-analyze affected e-classes
        updated_patterns = []
        for e_class in affected_classes:
            pattern = self._analyze_e_class(e_class)
            if pattern:
                if pattern.name in self.meta_patterns:
                    # Update existing pattern
                    self.meta_patterns[pattern.name].merge(pattern)
                else:
                    # New pattern discovered
                    self.meta_patterns[pattern.name] = pattern
                updated_patterns.append(pattern)

        return updated_patterns

    def _derivation_to_expr(self, deriv: Derivation) -> TransformExpr:
        """Convert derivation to transform expression."""
        return TransformExpr(
            name=deriv.transform_name,
            amount=deriv.transform_amount,
            usage_count=1,
        )

    def _analyze_e_class(self, e_class_id: int) -> Optional[MetaPattern]:
        """Analyze single e-class for meta-pattern potential."""
        expressions = self.e_graph.e_classes[e_class_id]

        if len(expressions) < 3:  # Need at least 3 variants
            return None

        # Get structural signature
        structure = list(expressions)[0].get_structural_signature()

        # Collect parameter distributions
        param_counts = defaultdict(int)
        for expr in expressions:
            params = tuple(expr.get_parameters())
            param_counts[params] += expr.usage_count

        return MetaPattern(
            name=f"meta_{structure}_{e_class_id}",
            structure=structure,
            parameter_distribution=dict(param_counts),
            total_uses=sum(param_counts.values()),
        )
```

### Integration with V1 Discovery Loop

```python
def run_discovery_with_incremental_v2(corpus, max_iterations=5):
    """
    V1 discovery with incremental V2 meta-pattern extraction.
    """
    v1_discovery = EmergentHierarchyDiscovery(corpus)
    v2_incremental = IncrementalEGraphDiscovery()

    for iteration in range(max_iterations):
        print(f"\n=== ITERATION {iteration + 1} ===")

        # V1: Discover derivations (expensive)
        print("  V1: Discovering derivations...")
        new_derivations = v1_discovery.run_iteration()
        print(f"  V1: Found {len(new_derivations)} new derivations")

        # V2: Incremental lambda extraction (cheap)
        print("  V2: Incremental meta-pattern update...")
        updated_patterns = v2_incremental.update_incremental(new_derivations)
        print(f"  V2: Updated {len(updated_patterns)} meta-patterns")

        # Check convergence
        if len(new_derivations) == 0:
            print("  Converged - no new derivations")
            break

    return v1_discovery.graph, v2_incremental.meta_patterns
```

### Performance Comparison

| Approach | Iteration 1 | Iteration 2 | Iteration 3 | Total |
|----------|-------------|-------------|-------------|-------|
| Naive (rescan all) | 2.5h | 2.5h | 2.5h | 7.5h |
| Incremental | 2.1h | 2.05h | 2.05h | 6.2h |

**Savings**: ~20% time reduction, scales better with more iterations

### Why Incremental Works

1. **E-graph structure is additive**: Adding new expressions only affects their e-class
2. **Meta-patterns are compositional**: New derivations either:
   - Add to existing pattern's parameter distribution (quick update)
   - Create new pattern (only when truly novel structure appears)
3. **Processed set tracking**: Never re-analyze the same derivation twice

---

## Future: V3 - Contextual Meta-Patterns

```python
# V3 would add context awareness:
contextual_third = ContextualMetaPattern(
    base_pattern='transpose(_)',
    context_rules={
        'major_key': lambda: transpose(+4),  # Major 3rd
        'minor_key': lambda: transpose(+3),  # Minor 3rd
    },
    context_extractor=extract_key_context,
)
```
