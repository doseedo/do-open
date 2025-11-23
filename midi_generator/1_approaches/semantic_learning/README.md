# Semantic Feature Learning (Agents 3-7)

## Overview
Semantic learning maps high-level musical concepts (rhythm feel, harmonic tension, texture density) to low-level MIDI parameters. Instead of operating on notes directly, it operates on interpretable musical dimensions.

## Architecture

### Dimension-Specific Encoders (`encoders/`)

Each musical dimension has a specialized encoder:

- **rhythm_encoder.py** (Agent 3) - Rhythm and timing
  - Swing ratio, syncopation, subdivision complexity
  - Rhythmic motifs, groove patterns
  - 50+ rhythm features → 20 semantic parameters

- **harmony_encoder.py** (Agent 4) - Harmonic structure
  - Chord quality, tension/resolution, voice leading
  - Modal interchange, chromatic movement
  - 60+ harmony features → 25 semantic parameters

- **texture_encoder.py** (Agent 5) - Musical texture
  - Density, register distribution, spacing
  - Homophonic vs polyphonic, layering
  - 40+ texture features → 18 semantic parameters

- **orchestration_encoder.py** (Agent 6) - Instrument roles
  - Section balance, doubling patterns
  - Lead/accompaniment relationships
  - 35+ orchestration features → 15 semantic parameters

- **cross_dimensional_encoder.py** (Agent 7) - Cross-dimensional interactions
  - Rhythm-harmony coupling, texture-dynamics correlation
  - Multi-dimensional musical gestures
  - 45+ interaction features → 22 semantic parameters

### Core Infrastructure (`encoders/`)

- **semantic_encoder.py** - Base encoder interface
- **semantic_decoder.py** - Decode semantic → MIDI parameters
- **semantic_features.py** - Feature extraction utilities
- **semantic_constraints.py** - Musical constraint system
- **modular_encoder_factory.py** - Factory for creating encoder pipelines

### Training System (`training/`)

- **semantic_discovery_pipeline.py** - End-to-end training
- **semantic_evaluation.py** - Evaluation metrics
- **benchmark_semantic_evaluation.py** - Benchmarking suite

## Key Innovation: Semantic Parameter Space

Instead of 1000+ low-level MIDI features, operate on 100 interpretable semantic parameters:

```python
# Low-level (traditional):
note.velocity = 87
note.duration = 0.25
note.pitch = 60

# Semantic (this approach):
semantic_params = {
    'swing_ratio': 0.67,        # Triplet swing feel
    'harmonic_tension': 0.8,    # High tension (dominant 7th)
    'texture_density': 0.3,     # Sparse texture
    'rhythmic_syncopation': 0.6 # Moderate syncopation
}
```

## Workflow

```
Input MIDI
    ↓
Feature Extraction (1000+ features)
    ↓
Dimension-Specific Encoders
    ↓
Semantic Parameter Space (100 params)
    ↓
Semantic Decoder
    ↓
MIDI Parameters
    ↓
Output MIDI
```

## Capabilities

**Strengths:**
- ✅ Highly interpretable (semantic parameters have musical meaning)
- ✅ Dimension-specialized (each encoder is an expert)
- ✅ Musically constrained (encodes music theory knowledge)
- ✅ Compositional (combine semantic edits across dimensions)
- ✅ Efficient representation (100 params vs 1000+ features)

**Limitations:**
- ❌ Requires labeled training data for each dimension
- ❌ Encoder design requires music theory expertise
- ❌ May miss nuances not captured by semantic parameters
- ❌ Decoder may not perfectly reconstruct from semantics

## Training Data Requirements

Each encoder requires training data:
- **Rhythm Encoder**: 1000+ MIDI files with rhythm labels
- **Harmony Encoder**: 1000+ files with chord annotations
- **Texture Encoder**: 1000+ files with texture labels
- **Orchestration Encoder**: 1000+ orchestrated pieces
- **Cross-Dimensional**: 1000+ files with multi-dimensional labels

## Use Cases

1. **Style Transfer** - Transfer semantic characteristics between pieces
2. **Interactive Editing** - Edit music at semantic level ("make it swingier")
3. **Music Generation** - Generate from semantic descriptions
4. **Analysis** - Understand music in interpretable dimensions
5. **Education** - Teach musical concepts through semantic parameters

## Example: Rhythm Transformation

```python
# Extract rhythm semantics
encoder = RhythmEncoder()
rhythm_params = encoder.encode(input_midi)
# {'swing_ratio': 0.5, 'syncopation': 0.3, 'subdivision': 0.25}

# Modify semantics
rhythm_params['swing_ratio'] = 0.67  # Add triplet swing
rhythm_params['syncopation'] = 0.7   # More syncopated

# Decode back to MIDI
decoder = SemanticDecoder()
output_midi = decoder.decode(rhythm_params, input_midi)
```

## Integration Points

- **With Transform-Based**: Semantic params can control transform amounts
- **With Neural Synthesis**: Learn to predict semantic transformations
- **With Hierarchical MTL**: Multi-task learning across semantic dimensions
- **With Rule-Based**: Semantic constraints guide rule application

## Performance Metrics

- **Reconstruction Error**: How well decoder recovers original MIDI
- **Semantic Consistency**: Encoded params match expected values
- **Musical Validity**: Decoded MIDI is musically plausible
- **Cross-Domain Transfer**: Semantics transfer across different pieces

## References

- Agent 3: Rhythm Encoder (`rhythm_encoder.py`, ~800 lines)
- Agent 4: Harmony Encoder (`harmony_encoder.py`, ~900 lines)
- Agent 5: Texture Encoder (`texture_encoder.py`, ~750 lines)
- Agent 6: Orchestration Encoder (`orchestration_encoder.py`, ~850 lines)
- Agent 7: Cross-Dimensional Encoder (`cross_dimensional_encoder.py`, ~950 lines)
- Total: ~5,000+ lines across semantic learning system
