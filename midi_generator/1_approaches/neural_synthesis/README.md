# Neural Program Synthesis (Agent 8)

## Overview
Neural program synthesis learns to generate transform programs from input/output MIDI examples. Instead of manually searching for the right transform, a neural network learns to synthesize the program automatically.

## Architecture

### DSL Components (`dsl/`)
- **transform_dsl.py** - Domain-Specific Language for musical transformations
  - ~300 token vocabulary
  - 9 iterator types (ALL_NOTES, SIMULTANEOUS_NOTES, etc.)
  - 20 filter types (PITCH_GREATER, IS_CHORD_TONE, etc.)
  - 16 operation types (TRANSPOSE, TIME_SCALE, etc.)
  - 13 aggregator types (MEAN, SORT_BY_PITCH, etc.)

- **synthetic_dataset.py** - Training data generation
  - 100 hand-written transform templates
  - Generates 10,000 training examples
  - 6 categories: pitch, rhythm, dynamics, harmony, texture, complex

### Neural Architecture (`architecture/`)
- **neural_synthesizer.py** - Complete neural model
  - **MIDIDifferenceEncoder**: Transformer encoder that learns what changed
  - **DSLProgramDecoder**: Grammar-constrained decoder
  - **DSLGrammarConstraints**: 11 rules ensuring syntactic correctness
  - **NeuralProgramSynthesizer**: End-to-end model

### Training Infrastructure (`training/`)
- **program_synthesis_trainer.py** - PyTorch training system
  - SyntheticMIDIDataset with variable-length padding
  - AdamW optimizer with ReduceLROnPlateau
  - Early stopping (patience=10)
  - Gradient clipping (max_norm=1.0)
  - Metrics: loss, token accuracy, exact match, functional match

## Key Innovation: Grammar-Constrained Generation

The DSLGrammarConstraints system ensures 100% of generated programs are syntactically valid:

```python
# Example: After FOREACH token, only iterator types are valid
if current_token == self.foreach_token:
    valid_tokens = self.iterator_tokens  # Only ALL_NOTES, CHORD_NOTES, etc.
    # Mask out all other tokens with -inf logits
```

**11 Grammar Rules:**
1. After `<SOS>` → statement tokens only
2. After `FOREACH` → iterator types only
3. After iterator → operations/statements
4. After `IF` → filter types only
5. After filter → operations/statements
6. After operation name → value/aggregator only
7. After value → close paren or next statement
8. Nesting depth limits (max 3 levels)
9. Statement sequence rules
10. Proper parenthesis matching
11. `<EOS>` only when all blocks closed

## Workflow

```
Input MIDI + Output MIDI
        ↓
   MIDIDifferenceEncoder (learns what changed)
        ↓
   Difference Embedding
        ↓
   DSLProgramDecoder (generates program)
        ↓
   Grammar Constraints (ensures validity)
        ↓
   Valid DSL Program
```

## Capabilities

**Strengths:**
- ✅ Learns from examples (no manual transform search)
- ✅ 100% syntactically valid output (grammar constraints)
- ✅ Interpretable (generates human-readable DSL)
- ✅ Generalizable (learns patterns across templates)
- ✅ Scalable (can train on millions of examples)

**Limitations:**
- ❌ Requires training data (10,000+ examples)
- ❌ Constrained by DSL vocabulary (can't invent new operations)
- ❌ May not capture very complex multi-step logic
- ❌ Inference is slower than direct transform execution

## Training Data

The synthetic dataset generator creates training examples:

```python
# Example: Transpose +5 semitones
Input MIDI: C major scale
Output MIDI: F major scale (C+5)
DSL Program:
  FOREACH ALL_NOTES {
    TRANSPOSE(5)
  }
```

Generated from 100 templates across:
- Pitch transforms (20 templates)
- Rhythm transforms (25 templates)
- Dynamics transforms (15 templates)
- Harmony transforms (20 templates)
- Texture transforms (20 templates)
- Complex multi-dimensional (20 templates)

## Use Cases

1. **Style Transfer Learning** - Learn transformations from example pairs
2. **Inverse MIDI Analysis** - Discover what transforms were applied
3. **Interactive Learning** - User provides examples, system learns transform
4. **Automated Arrangement** - Learn orchestration patterns from examples
5. **Music Education** - Explain transformations in interpretable DSL

## Integration Points

- **With Transform-Based**: Uses same DSL and transform library
- **With Semantic Learning**: Can learn semantic feature transformations
- **With Hierarchical MTL**: Neural model can predict transform programs
- **With Gap Detection**: Discovers missing transforms in library

## Performance Metrics

Target metrics (after training):
- **Token Accuracy**: 90%+ (correct tokens predicted)
- **Exact Match**: 60%+ (entire program correct)
- **Functional Match**: 80%+ (execution equivalent to ground truth)
- **Validity**: 100% (grammar constraints guarantee this)

## References

- Agent 8 Phase 1-2: DSL + Dataset (`transform_dsl.py`, `synthetic_dataset.py`)
- Agent 8 Phase 3: Neural Architecture (`neural_synthesizer.py`)
- Agent 8 Phase 4: Training Infrastructure (`program_synthesis_trainer.py`)
- Total: ~1,300 lines of code
