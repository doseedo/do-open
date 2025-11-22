# Rule-Based Style Transfer (Classic Approach)

## Overview
The original rule-based style transfer system uses hand-crafted musicological rules to transform MIDI from one style to another. This is the classical, deterministic approach that preceded all machine learning methods.

## Architecture

### Core Module

- **style_transfer.py** (950 lines)
  - Hand-written rules for style transformation
  - Pattern matching and replacement
  - Music theory constraints
  - Deterministic transformations

## Approach

**Rule Types:**

1. **Pitch Rules**
   - Scale transformations (major ↔ minor)
   - Chord substitutions (ii-V-I patterns)
   - Melodic ornamentations

2. **Rhythm Rules**
   - Time signature conversion
   - Groove quantization
   - Swing vs straight feel

3. **Harmony Rules**
   - Chord voicing rules
   - Voice leading constraints
   - Tension/resolution patterns

4. **Dynamics Rules**
   - Accent patterns
   - Crescendo/diminuendo shapes
   - Style-specific dynamics

## Example Rules

```python
# If major 7th chord in jazz → add extensions
if chord.quality == "major7" and style == "jazz":
    add_extension(chord, 9)  # Add 9th
    if random.random() > 0.5:
        add_extension(chord, 13)  # Add 13th

# If classical → strict voice leading
if style == "classical":
    ensure_voice_leading(chord1, chord2, max_leap=5)
```

## Capabilities

**Strengths:**
- ✅ Fully deterministic (same input = same output)
- ✅ Complete control (every rule is explicit)
- ✅ Fast execution (no neural network inference)
- ✅ No training data required
- ✅ Musicologically accurate (rules from music theory)

**Limitations:**
- ❌ Labor-intensive (rules must be hand-written)
- ❌ Brittle (doesn't generalize beyond programmed rules)
- ❌ Can't capture subtle stylistic nuances
- ❌ Hard to maintain (adding rules can break existing ones)
- ❌ Doesn't learn from data

## Use Cases

1. **Baseline Comparisons** - Compare against ML approaches
2. **Quick Prototypes** - Fast deterministic transformations
3. **Educational** - Understand explicit music theory rules
4. **Legacy Support** - Maintain existing rule-based systems

## Integration Points

- **With Transform-Based**: Rules can be expressed as transforms
- **With Semantic Learning**: Rules provide interpretable baselines
- **With LLM Expansion**: LLM can generate new rules

## Historical Context

This was the first approach implemented (before the machine learning pipelines). While superseded by more sophisticated methods, it remains useful for:
- Baseline comparisons
- Fast deterministic operations
- Educational demonstrations of music theory

## References

- Classic implementation: `transformation/style_transfer.py` (~950 lines)
- Predates Agents 3-15 (original codebase)
