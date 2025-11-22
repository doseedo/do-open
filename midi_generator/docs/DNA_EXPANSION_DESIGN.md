# DNA Expansion Design: 120D → 300D Hierarchical Architecture

**Agent:** Agent 3 - DNA Expansion & Hierarchical Architecture
**Date:** 2025-11-22
**Status:** Design Phase
**Version:** 1.0.0

---

## Executive Summary

This document outlines the design for expanding the Musical DNA system from 120D to 300D with a hierarchical structure. The expansion maintains backward compatibility while adding new capabilities for melody encoding, voicing analysis, and global musical context.

---

## 1. Current Architecture (120D)

### 1.1 Current DNA Structure

```
Domain Parameters (110D):
├── Harmony: 30D     (from 250D harmony features)
├── Rhythm: 20D      (from 250D rhythm features)
├── Form: 15D        (from 50D structure features)
├── Orchestration: 25D (from 150D orchestration features)
└── Texture: 20D     (from 100D texture features)

Cross-Dimensional: 10D (from concatenated 110D)

Total: 120D
```

### 1.2 Available Feature Space

DeepFeatureExtractor provides 1150D features:
- Harmony: 250D ✓ (currently using)
- **Melody: 200D** ⚠️ (NOT currently used - opportunity!)
- Rhythm: 250D ✓ (currently using)
- **Dynamics: 150D** ⚠️ (NOT currently used - opportunity!)
- Texture: 100D ✓ (currently using)
- Structure: 50D ✓ (currently using)
- Orchestration: 150D ✓ (currently using)

**Key Insight:** We have 200D melody features and 150D dynamics features that are currently unused. This provides a natural foundation for expansion.

---

## 2. Proposed 300D Hierarchical Architecture

### 2.1 Hierarchical Structure

```
GLOBAL LEVEL (60D) - Musical Context & Style
├── key_context: 12D        (tonal center, modulations, key stability)
├── tempo_feel: 8D          (tempo, tempo variations, metric feel)
├── genre_style: 20D        (jazz/classical/latin style markers)
└── form_structure: 20D     (overall form, section count, proportions)

SECTIONAL LEVEL (140D) - Musical Content
├── harmony: 60D            [EXPANDED from 30D]
│   ├── chord_types: 20D
│   ├── progressions: 20D
│   └── voice_leading: 20D
│
├── melody: 40D             [NEW MODULE]
│   ├── contour: 15D        (shape, range, intervals)
│   ├── motifs: 15D         (repetition, development)
│   └── phrasing: 10D       (phrase structure, breath marks)
│
└── rhythm: 40D             [EXPANDED from 20D]
    ├── syncopation: 15D
    ├── groove: 15D
    └── subdivision: 10D

LOCAL LEVEL (100D) - Implementation Details
├── voicing: 30D            [NEW MODULE]
│   ├── spacing: 10D        (intervals, clusters, spread)
│   ├── doubling: 10D       (unison, octave, intensity)
│   └── register: 10D       (tessitura, range distribution)
│
├── texture: 30D            [EXPANDED from 20D]
│   ├── density: 10D
│   ├── independence: 10D
│   └── layering: 10D
│
└── orchestration: 40D      [EXPANDED from 25D]
    ├── instrumentation: 15D
    ├── balance: 15D
    └── articulation: 10D

Total: 300D (60 + 140 + 100)
```

### 2.2 Dimension Mapping

| Level | Module | Old Size | New Size | Change | Source Features |
|-------|--------|----------|----------|--------|-----------------|
| GLOBAL | key_context | - | 12D | NEW | Harmony (250D) |
| GLOBAL | tempo_feel | - | 8D | NEW | Rhythm (250D) |
| GLOBAL | genre_style | - | 20D | NEW | Structure (50D) + Melody (200D) |
| GLOBAL | form_structure | 15D | 20D | +5D | Structure (50D) |
| SECTIONAL | harmony | 30D | 60D | +30D | Harmony (250D) |
| SECTIONAL | melody | - | 40D | NEW | **Melody (200D)** |
| SECTIONAL | rhythm | 20D | 40D | +20D | Rhythm (250D) |
| LOCAL | voicing | - | 30D | NEW | Harmony (250D) + **Dynamics (150D)** |
| LOCAL | texture | 20D | 30D | +10D | Texture (100D) + **Dynamics (150D)** |
| LOCAL | orchestration | 25D | 40D | +15D | Orchestration (150D) |

**Note:** Cross-dimensional encoder (10D) is removed in favor of hierarchical conditioning (higher levels condition lower levels).

---

## 3. Implementation Plan

### 3.1 Updated MusicalDNA Class

```python
@dataclass
class MusicalDNA:
    """
    Hierarchical Musical DNA (300D).

    Version 2.0 with hierarchical structure.
    """
    # Version control
    version: str = "2.0"

    # GLOBAL LEVEL (60D)
    key_context_params: np.ndarray      # 12D
    tempo_feel_params: np.ndarray       # 8D
    genre_style_params: np.ndarray      # 20D
    form_structure_params: np.ndarray   # 20D

    # SECTIONAL LEVEL (140D)
    harmony_params: np.ndarray          # 60D (expanded)
    melody_params: np.ndarray           # 40D (NEW)
    rhythm_params: np.ndarray           # 40D (expanded)

    # LOCAL LEVEL (100D)
    voicing_params: np.ndarray          # 30D (NEW)
    texture_params: np.ndarray          # 30D (expanded)
    orchestration_params: np.ndarray    # 40D (expanded)

    # Metadata
    source_file: Optional[str] = None
    extraction_timestamp: Optional[str] = None

    def to_vector(self) -> np.ndarray:
        """Flatten to 300D vector."""
        return np.concatenate([
            # Global (60D)
            self.key_context_params,
            self.tempo_feel_params,
            self.genre_style_params,
            self.form_structure_params,
            # Sectional (140D)
            self.harmony_params,
            self.melody_params,
            self.rhythm_params,
            # Local (100D)
            self.voicing_params,
            self.texture_params,
            self.orchestration_params,
        ])

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'MusicalDNA':
        """Create from 300D vector."""
        assert len(vector) == 300
        return cls(
            # Global (60D)
            key_context_params=vector[0:12],
            tempo_feel_params=vector[12:20],
            genre_style_params=vector[20:40],
            form_structure_params=vector[40:60],
            # Sectional (140D)
            harmony_params=vector[60:120],
            melody_params=vector[120:160],
            rhythm_params=vector[160:200],
            # Local (100D)
            voicing_params=vector[200:230],
            texture_params=vector[230:260],
            orchestration_params=vector[260:300],
        )
```

### 3.2 Migration Utility (120D → 300D)

**Strategy:** Map old parameters to new structure with intelligent initialization:

```python
def migrate_120d_to_300d(old_dna: MusicalDNA_v1) -> MusicalDNA_v2:
    """
    Migrate 120D DNA to 300D hierarchical structure.

    Mapping:
    - Old harmony (30D) → new harmony (60D): replicate + augment
    - Old rhythm (20D) → new rhythm (40D): replicate + augment
    - Old form (15D) → new form_structure (20D): extend
    - Old orchestration (25D) → new orchestration (40D): extend
    - Old texture (20D) → new texture (30D): extend
    - Old cross (10D) → distributed to global params

    NEW parameters initialized to zero (will be learned):
    - key_context (12D) → zeros
    - tempo_feel (8D) → zeros
    - genre_style (20D) → zeros
    - melody (40D) → zeros
    - voicing (30D) → zeros
    """
```

### 3.3 New Encoder Modules

#### 3.3.1 GlobalEncoder (60D output)

```python
class GlobalEncoder(nn.Module):
    """
    Encodes global musical context.

    Input: 1150D (full features)
    Output: 60D (key_context 12D + tempo_feel 8D + genre_style 20D + form_structure 20D)

    Architecture:
    - Attention mechanism over all features
    - Focus on long-range patterns
    - Style classification head
    """
```

#### 3.3.2 MelodyEncoder (40D output)

```python
class MelodyEncoder(SemanticFeatureEncoder):
    """
    Encodes melodic parameters from 200D melody features.

    Input: 200D melody features
    Output: 40D (contour 15D + motifs 15D + phrasing 10D)

    Locality Functions:
    - TRANSPOSE
    - INVERT
    - RETROGRADE
    - AUGMENTATION
    """
```

#### 3.3.3 VoicingEncoder (30D output)

```python
class VoicingEncoder(SemanticFeatureEncoder):
    """
    Encodes voicing details from harmony + dynamics features.

    Input: 250D harmony features + 150D dynamics features = 400D
    Output: 30D (spacing 10D + doubling 10D + register 10D)

    Locality Functions:
    - OCTAVE_SHIFT
    - VOICE_PERMUTATION
    - REGISTER_SHIFT
    """
```

### 3.4 Expanded Encoder Modules

- **HarmonyEncoder**: 30D → 60D (use more of 250D harmony features)
- **RhythmEncoder**: 20D → 40D (use more of 250D rhythm features)
- **TextureEncoder**: 20D → 30D (add dynamics features)
- **OrchestrationEncoder**: 25D → 40D (use more of 150D orchestration features)
- **FormEncoder** → **FormStructureEncoder**: 15D → 20D (expand structure analysis)

---

## 4. Backward Compatibility

### 4.1 Version Detection

```python
def load_dna(path: Path) -> MusicalDNA:
    """Load DNA with automatic version detection."""
    with open(path, 'r') as f:
        data = json.load(f)

    version = data.get('version', '1.0')

    if version == '1.0':
        # Load as 120D and migrate
        old_dna = MusicalDNA_v1.from_dict(data)
        return migrate_120d_to_300d(old_dna)
    else:
        # Load as 300D
        return MusicalDNA_v2.from_dict(data)
```

### 4.2 Checkpoint Migration

```python
def migrate_checkpoint(old_checkpoint_path: Path, new_checkpoint_path: Path):
    """
    Migrate 120D encoder checkpoint to 300D.

    Strategy:
    - Copy weights for existing modules where possible
    - Initialize new dimensions with Xavier initialization
    - Preserve optimizer state for transferred weights
    """
```

---

## 5. Training Strategy

### 5.1 Hierarchical Training

```
Phase 1: Train GLOBAL encoders (60D)
  ├── GlobalEncoder learns musical context
  └── Provides context vectors for conditioning

Phase 2: Train SECTIONAL encoders (140D) conditioned on global
  ├── HarmonyEncoder (60D) ← conditioned on key_context
  ├── MelodyEncoder (40D) ← conditioned on key_context + genre_style
  └── RhythmEncoder (40D) ← conditioned on tempo_feel

Phase 3: Train LOCAL encoders (100D) conditioned on global + sectional
  ├── VoicingEncoder (30D) ← conditioned on harmony_params
  ├── TextureEncoder (30D) ← conditioned on rhythm_params
  └── OrchestrationEncoder (40D) ← conditioned on all sectional
```

### 5.2 Conditioning Mechanism

```python
class ConditionalEncoder(nn.Module):
    """Base class for conditional encoding."""

    def forward(self, features, condition=None):
        if condition is not None:
            # FiLM conditioning: scale and shift
            features = features * self.scale_layer(condition) + self.shift_layer(condition)
        return self.encoder(features)
```

---

## 6. Memory & Performance Considerations

### 6.1 Memory Estimates

- **120D DNA**: ~1KB per sample
- **300D DNA**: ~2.4KB per sample (2.5x increase)
- **Encoder parameters**: ~50M → ~120M (2.4x increase)
- **Training batch size**: May need to reduce from 64 → 32 depending on GPU memory

### 6.2 Optimization Strategies

1. **Gradient Checkpointing**: Save memory during backprop
2. **Mixed Precision Training**: Use FP16 where possible
3. **Modular Training**: Train modules separately if memory constrained
4. **Feature Caching**: Cache extracted features to disk

---

## 7. Validation Metrics

### 7.1 Reconstruction Quality

- **Per-level MSE**: Measure reconstruction at each hierarchical level
- **Cross-level consistency**: Ensure global ↔ sectional ↔ local coherence

### 7.2 Disentanglement

- **Within-level disentanglement**: Parameters within same level should be independent
- **Cross-level interpretability**: Higher levels should be more interpretable

### 7.3 Musical Validity

- **Global coherence**: Does key_context match actual key?
- **Melodic sensibility**: Do melody params produce singable melodies?
- **Voicing realism**: Do voicing params match jazz conventions?

---

## 8. Timeline

**Week 1** (Days 1-7):
- ✓ Research existing codebase
- ✓ Design 300D architecture
- ⏳ Implement MusicalDNA v2.0 class
- ⏳ Build migration utility

**Week 2** (Days 8-14):
- Implement GlobalEncoder
- Implement MelodyEncoder
- Implement VoicingEncoder
- Test new modules independently

**Week 3** (Days 15-21):
- Expand existing encoders (Harmony, Rhythm, Texture, Orchestration)
- Implement hierarchical conditioning
- Update ModularEncoderFactory

**Week 4** (Days 22-30):
- Integration testing
- Backward compatibility testing
- Memory profiling and optimization
- Documentation and examples

---

## 9. Success Criteria

- [ ] 300D DNA class with hierarchical structure
- [ ] All 300 dimensions sum correctly
- [ ] Backward compatibility: old 120D checkpoints load successfully
- [ ] New modules (Global, Melody, Voicing) train without errors
- [ ] Expanded modules maintain or improve reconstruction quality
- [ ] Memory usage < 16GB GPU for batch_size=32
- [ ] Comprehensive tests pass
- [ ] Clear migration guide for existing users

---

## 10. Next Steps

1. ✅ Complete design document (this document)
2. ⏭️ Implement MusicalDNA v2.0 class
3. ⏭️ Build migration utility
4. ⏭️ Implement new encoder modules
5. ⏭️ Update existing encoders
6. ⏭️ Integration and testing

---

## Appendices

### A. Feature Allocation

| Encoder | Input Features | Output Dims | Source |
|---------|---------------|-------------|--------|
| GlobalEncoder | 1150D (all) | 60D | All features with attention |
| HarmonyEncoder | 250D harmony | 60D | Harmony features |
| MelodyEncoder | 200D melody | 40D | **Melody features (NEW)** |
| RhythmEncoder | 250D rhythm | 40D | Rhythm features |
| VoicingEncoder | 250D harmony + 150D dynamics | 30D | **Harmony + Dynamics (NEW)** |
| TextureEncoder | 100D texture + 150D dynamics | 30D | Texture + Dynamics |
| OrchestrationEncoder | 150D orchestration | 40D | Orchestration features |

### B. Code Files to Create/Modify

**New Files:**
- `midi_generator/learning/musical_dna_v2.py` - New hierarchical DNA class
- `midi_generator/learning/dna_migration.py` - Migration utilities
- `midi_generator/learning/global_encoder.py` - Global encoder
- `midi_generator/learning/melody_encoder.py` - Melody encoder
- `midi_generator/learning/voicing_encoder.py` - Voicing encoder
- `midi_generator/learning/conditional_encoder.py` - Base class for conditioning

**Modified Files:**
- `midi_generator/learning/modular_encoder_factory.py` - Add new encoders
- `midi_generator/learning/harmony_encoder.py` - Expand to 60D
- `midi_generator/learning/rhythm_encoder.py` - Expand to 40D
- `midi_generator/learning/texture_encoder.py` - Expand to 30D
- `midi_generator/learning/orchestration_encoder.py` - Expand to 40D
- `midi_generator/learning/form_encoder.py` - Expand to 20D
- `midi_generator/learning/modular_discovery_pipeline.py` - Support v2.0 DNA

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-22
**Status:** Ready for Implementation
