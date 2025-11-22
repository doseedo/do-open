# Transform-Based Approach (Agent 8)

## Overview
The transform-based approach uses a library of 60 parametric space-level transforms to modify MIDI files. This is a compositional, interpretable approach where complex transformations are built from simple, understandable operations.

## Architecture

### Core Components (`core/`)
- **space_level_transforms.py** - Core transform primitives and execution engine
- **transform_registry.py** - Registry system for managing and discovering transforms
- **transforms_init.py** - Module initialization and exports

### Transform Library (`transforms/`)
60 parametric transforms across 7 categories:
- **pitch.py** - Transposition, inversion, pitch scaling (20 transforms)
- **rhythm.py** - Time stretching, quantization, swing (25 transforms)
- **harmony.py** - Chord substitution, voicing, tension (20 transforms)
- **texture.py** - Density, register, spacing (20 transforms)
- **expression.py** - Dynamics, articulation, phrasing (15 transforms)
- **form.py** - Structure, repetition, development (10 transforms)
- **advanced.py** - Complex multi-dimensional transforms (10 transforms)

### Discovery System (`discovery/`)
- **hybrid_synthesizer.py** - Combines transform search with neural guidance
- **llm_code_generator.py** - LLM-assisted transform code generation
- **sparse_learning.py** - Learns sparse transform combinations from examples

## Key Innovation: Space-Level Transforms

Unlike note-level or frame-level operations, space-level transforms operate on musical structures (chords, phrases, sections) as atomic units:

```python
# Example: Transpose all chord tones up by 2 semitones
transform = SpaceLevelTransform(
    iterator=IteratorType.SIMULTANEOUS_NOTES,  # Group by chords
    filter=FilterType.IS_CHORD_TONE,           # Only chord tones
    operation=OperationType.TRANSPOSE,          # Transpose operation
    amount=2                                    # +2 semitones
)
```

## Capabilities

**Strengths:**
- ✅ Fully interpretable (human can understand every operation)
- ✅ Compositional (combine simple transforms for complex effects)
- ✅ Efficient (60 transforms cover vast transformation space)
- ✅ Controllable (precise parameter control)
- ✅ Generalizable (works on any MIDI file)

**Limitations:**
- ❌ Fixed vocabulary (can't invent new transform types)
- ❌ Requires search to find right combination
- ❌ May not capture very subtle stylistic nuances

## Use Cases

1. **Style Transfer** - Transform piece from one style to another
2. **Variation Generation** - Create variations of existing material
3. **Data Augmentation** - Generate training data for ML models
4. **Interactive Editing** - Real-time musical transformations
5. **Analysis** - Understand relationships between musical pieces

## Integration Points

- **With Neural Synthesis**: Transform library provides templates for neural program synthesis
- **With Semantic Learning**: Transforms serve as interpretable operations for learned features
- **With Hierarchical MTL**: Transforms can be predicted by MTL model
- **With LLM Expansion**: LLM can generate new transform code

## References

- Agent 8 Phase 1: `transforms/space_level_transforms.py` (Agent 8, ~4000 lines)
- Agent 8 Phase 2: Transform discovery and expansion
- Related: Dimensionality theory for transform space coverage
