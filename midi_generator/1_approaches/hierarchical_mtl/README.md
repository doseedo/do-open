# Hierarchical Multi-Task Learning (Agents 9, 14, 15)

## Overview
Hierarchical MTL uses a shared neural architecture to predict musical parameters across multiple dimensions simultaneously. It learns hierarchical relationships between low-level features and high-level musical concepts through multi-task learning.

## Architecture

### Core Model (`architecture/`)

- **hierarchical_mtl.py** - Multi-task learning architecture
  - Shared encoder (extracts common musical representations)
  - Task-specific heads (rhythm, harmony, texture, dynamics, etc.)
  - Hierarchical task dependencies (low-level → mid-level → high-level)
  - Attention mechanisms for cross-task information flow

**Model Structure:**
```
Input MIDI (pianoroll or features)
    ↓
Shared Encoder (Transformer/CNN)
    ↓
Shared Representation (512-dim)
    ↓
├─→ Rhythm Head → Rhythm params
├─→ Harmony Head → Harmony params
├─→ Texture Head → Texture params
├─→ Dynamics Head → Dynamics params
└─→ ... (10+ task heads)
```

### Training System (`training/`)

- **hierarchical_trainer.py** - Training infrastructure
  - Multi-task loss weighting (learned or fixed)
  - Task balancing strategies
  - Curriculum learning (easy tasks → hard tasks)
  - Gradient surgery (prevent task interference)

### Integration (`integration/`)

- **hierarchical_predictor.py** - Prediction interface
  - Single forward pass predicts all parameters
  - Task-specific post-processing
  - Confidence estimation per task

## Key Innovation: Task Hierarchies

Tasks are organized in a hierarchy based on dependencies:

**Level 1 (Low-level):** Direct from input
- Note density, pitch range, velocity distribution

**Level 2 (Mid-level):** Depends on Level 1
- Rhythm patterns, chord progressions, texture types

**Level 3 (High-level):** Depends on Level 1-2
- Style, genre, emotional tone, structural form

The model learns these dependencies through:
- Hierarchical loss propagation
- Task-conditional features
- Attention between task heads

## Multi-Task Learning Benefits

**Positive Transfer:**
- Rhythm knowledge helps harmony prediction (syncopation ↔ chord changes)
- Harmony knowledge helps dynamics (tension ↔ loudness)
- Texture knowledge helps orchestration (density ↔ instrumentation)

**Data Efficiency:**
- Shared encoder learns general musical representations
- Don't need complete labels for all tasks (partial labeling OK)
- Knowledge transfers from data-rich tasks to data-poor tasks

**Regularization:**
- Multi-task objective prevents overfitting to single task
- Shared encoder must learn robust features

## Capabilities

**Strengths:**
- ✅ Single model predicts all dimensions (efficient inference)
- ✅ Positive transfer across related musical tasks
- ✅ Data-efficient (shared learning across tasks)
- ✅ Captures cross-dimensional musical relationships
- ✅ Scalable to many tasks (10+ task heads)

**Limitations:**
- ❌ Negative transfer possible (unrelated tasks can interfere)
- ❌ Balancing task weights is challenging
- ❌ Harder to debug than single-task models
- ❌ Requires careful architecture design

## Task Configuration

Currently supports 10+ tasks:

1. **Rhythm Tasks**
   - Swing ratio, syncopation, subdivision

2. **Harmony Tasks**
   - Chord quality, tension, voice leading

3. **Texture Tasks**
   - Density, register, homophonic vs polyphonic

4. **Dynamics Tasks**
   - Overall level, contour, accent patterns

5. **Articulation Tasks**
   - Staccato, legato, accent placement

6. **Form Tasks**
   - Section boundaries, repetition, development

7. **Style Tasks**
   - Genre classification, era, regional style

8. **Orchestration Tasks**
   - Instrument balance, doubling, section roles

## Training Data

Requires multi-label dataset:
- **Size**: 10,000+ MIDI files
- **Labels**: Partial labels OK (not all tasks need labels for all files)
- **Format**:
  ```json
  {
    "midi_file": "example.mid",
    "labels": {
      "rhythm": {"swing_ratio": 0.67, "syncopation": 0.5},
      "harmony": {"tension": 0.8, "chord_quality": "dominant7"},
      "texture": {"density": 0.3}
      // ... other tasks (partial labels)
    }
  }
  ```

## Use Cases

1. **Comprehensive Music Analysis** - Predict all dimensions at once
2. **Style Transfer** - Transfer multi-dimensional characteristics
3. **Music Generation** - Generate conditioned on multiple tasks
4. **Quality Control** - Check if generated music satisfies multiple criteria
5. **Interactive Editing** - Edit multiple dimensions with single model

## Loss Function

Weighted multi-task loss:

```python
total_loss = Σ(w_i × L_i)

where:
- w_i = weight for task i (learned or fixed)
- L_i = loss for task i (MSE, CE, etc.)
```

**Task Weighting Strategies:**
- Uniform: All tasks equally weighted
- Learned: Weights learned during training
- Uncertainty-based: Weight by prediction uncertainty
- GradNorm: Balance gradient magnitudes across tasks

## Performance Metrics

- **Per-Task Accuracy**: How well each task is predicted
- **Cross-Task Correlation**: Do predictions align across tasks?
- **Inference Speed**: Single forward pass for all tasks (fast)
- **Transfer Benefit**: Performance gain from multi-task vs single-task

## Integration Points

- **With Semantic Learning**: MTL can predict semantic parameters
- **With Transform-Based**: MTL predictions control transform parameters
- **With Neural Synthesis**: MTL provides targets for program synthesis
- **With Feature Extraction**: Uses consolidated features from 3_analysis/

## References

- Agent 9: Feature mapping for MTL
- Agent 14: Training infrastructure development
- Agent 15: Model training and evaluation
- Total: ~3,000 lines across MTL system
