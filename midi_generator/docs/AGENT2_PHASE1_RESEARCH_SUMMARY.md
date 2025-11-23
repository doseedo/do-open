# Agent 2: Differentiable MIDI & Utilities - Phase 1 Research Summary

**Date**: November 22, 2025
**Agent**: Agent 2 - Differentiable MIDI & Utilities Support
**Phase**: Phase 1 - Research & Understanding (Days 1-3)
**Status**: COMPLETED

---

## Executive Summary

I have completed a comprehensive exploration of the existing codebase to understand:
1. **Existing MIDI I/O patterns** (mido, pretty_midi)
2. **Feature extraction architecture** (DeepFeatureExtractor: 1150D features)
3. **Current encoder patterns** (6 modular encoders)
4. **Musical transformations** (12 locality functions)
5. **Data structures and representations**

**Key Finding**: The codebase has robust MIDI analysis capabilities but **NO DECODER** or **differentiable representations** yet. This is exactly what Agent 1 and I need to build.

---

## 1. Existing MIDI I/O Infrastructure

### 1.1 MIDI Loading Libraries

The codebase uses **two MIDI libraries**:

1. **`mido`** - Low-level MIDI message processing
   - Location: `midi_generator/learning/musical_locality.py`
   - Used for: Musical transformations, note event manipulation
   - Advantages: Direct access to MIDI messages, delta times
   - File: `/home/user/Do/midi_generator/learning/musical_locality.py`

2. **`pretty_midi`** - Higher-level MIDI abstraction
   - Location: `midi_generator/feature_selection/rich_feature_extractor.py`
   - Used for: Feature extraction, note objects, instrument parsing
   - Advantages: Clean Note/Instrument API, easy tempo/key extraction
   - File: `/home/user/Do/midi_generator/synthesis/deep_feature_extractor.py:151`

### 1.2 Current MIDI Loading Pattern

```python
# Pattern 1: Using pretty_midi (most common)
import pretty_midi
midi = pretty_midi.PrettyMIDI(str(midi_path))
# Access: midi.instruments, midi.get_tempo_changes(), etc.

# Pattern 2: Using mido (for transformations)
import mido
midi_file = mido.MidiFile('song.mid')
# Access: midi_file.tracks, messages, delta times
```

**My Assessment**: We should build on `pretty_midi` for the decoder since:
- It has a cleaner API for generating MIDI
- `pretty_midi.Note` objects are easier to work with
- It handles tempo/time signature automatically
- We can convert to mido if needed for transformations

---

## 2. Feature Extraction Architecture

### 2.1 DeepFeatureExtractor (1150D Features)

**Location**: `/home/user/Do/midi_generator/synthesis/deep_feature_extractor.py`

**Feature Breakdown**:
- **Harmony**: 250 features (chord types, voicings, progressions)
- **Melody**: 200 features (intervals, contour, motifs)
- **Rhythm**: 250 features (syncopation, density, swing)
- **Dynamics**: 150 features (velocity patterns, articulation)
- **Texture**: 100 features (polyphony, register, layering)
- **Structure**: 50 features (form, sections)
- **Orchestration**: 150 features (instrumentation, timbral balance)

**Total**: 1150D feature vector

### 2.2 RichMultitrackFeatureExtractor (600D Features)

**Location**: `/home/user/Do/midi_generator/feature_selection/rich_feature_extractor.py`

**Feature Breakdown**:
- **Global features**: 200D (full-file statistics)
- **Per-track features**: 200D (8 tracks × 25D each)
- **Temporal features**: 100D (4 sections × 25D)
- **Orchestration features**: 100D (arrangement quality)

**Total**: 600D feature vector

### 2.3 Key Insight for Decoder Design

The decoder needs to **reverse** this feature extraction:
- Features → DNA (encoder)
- DNA → MIDI (decoder) ← **This is what we need to build**

The challenge: Features are continuous, MIDI events are discrete.
**Solution**: Differentiable representations (pianoroll, soft sampling).

---

## 3. Current Encoder Architecture

### 3.1 Modular Encoder System

**Location**: `/home/user/Do/midi_generator/learning/`

The system has **6 modular encoders**:

1. **HarmonyEncoder** (`harmony_encoder.py`)
   - Input: 200D features → Output: 30D harmony parameters
   - Categories: chord types, voicings, progressions, voice leading, harmonic rhythm, extensions

2. **RhythmEncoder** (`rhythm_encoder.py`)
   - Input: 250D features → Output: ~20D rhythm parameters
   - Patterns: swing, syncopation, density, groove templates

3. **FormEncoder** (`form_encoder.py`)
   - Input: Structure features → Output: ~15D form parameters
   - Structure: sections, transitions, repetition

4. **OrchestrationEncoder** (`orchestration_encoder.py`)
   - Input: Instrumentation features → Output: ~25D orchestration parameters
   - Tracks: instrument roles, balance, timbral distribution

5. **TextureEncoder** (`texture_encoder.py`)
   - Input: Texture features → Output: ~20D texture parameters
   - Layers: polyphony, register, density

6. **SemanticEncoder** (`semantic_encoder.py`)
   - Meta-encoder that coordinates other encoders
   - Uses PyTorch neural networks

### 3.2 Encoder Architecture Pattern

**Typical structure** (from `harmony_encoder.py`):
```python
Input: 200D features
  ↓
Hidden: 512D (FC layer + activation)
  ↓
Output: 30D parameters
```

**Key observation**: The decoder should **mirror** this architecture:
```python
Input: 30D DNA harmony params
  ↓
Hidden: 512D (FC layer + activation)
  ↓
Output: Intermediate representation (chords, voicings)
  ↓
Soft MIDI events (differentiable)
```

---

## 4. Musical Locality Functions (12 Transformations)

**Location**: `/home/user/Do/midi_generator/learning/musical_locality.py`

### 4.1 The 12 Transformation Types

1. **TRANSPOSE** - Shift all pitches by semitones
2. **INVERT** - Invert melodic intervals around pivot
3. **TIME_SHIFT** - Shift note onsets temporally
4. **AUGMENT** - Stretch durations (rhythmic augmentation)
5. **RETROGRADE** - Reverse note sequence
6. **DIMINUTION** - Compress durations
7. **OCTAVE_SHIFT** - Move by octave(s)
8. **VELOCITY_SCALE** - Scale dynamics
9. **REGISTER_SHIFT** - Shift pitch register
10. **INTERVAL_SCALE** - Scale interval sizes
11. **RHYTHMIC_QUANTIZE** - Align to metric grid
12. **VOICE_PERMUTATION** - Swap tracks

### 4.2 Implementation Details

All transformations:
- Are **invertible** (T(T⁻¹(x)) = x)
- Operate on `mido.MidiFile` objects
- Preserve musical validity
- Form groups under composition

**Example transformation**:
```python
transformer = MusicalLocalityFunctions()
transform = MusicalTransform(
    transform_type=LocalityType.TRANSPOSE,
    parameters={"semitones": 5}
)
transformed_midi = transformer.apply_transform(midi_file, transform)
original = transformer.invert_transform(transformed_midi, transform)
```

### 4.3 Relevance to Decoder

These transformations show **what operations are musically valid**:
- The decoder should be able to generate MIDI that can be transformed
- Testing: Generate MIDI, apply transformation, check validity
- Data augmentation: Use transformations to expand training data

---

## 5. Data Structures

### 5.1 Note Representation (from DeepFeatureExtractor)

**Location**: `midi_generator/synthesis/deep_feature_extractor.py:46`

```python
@dataclass
class Note:
    pitch: int           # MIDI note number (0-127)
    velocity: int        # Velocity (1-127)
    start_time: float    # Onset time (seconds or ticks)
    end_time: float      # Offset time
    duration: float      # Note length
    channel: int         # MIDI channel (0-15)
```

**This is what the decoder needs to generate!**

### 5.2 Chord Representation

```python
@dataclass
class Chord:
    pitches: List[int]        # All pitches in chord
    start_time: float
    end_time: float
    duration: float
    velocities: List[int]
```

### 5.3 MIDI Constants

**Location**: `/home/user/Do/midi_generator/midi/midi_constants.py`

Available resources:
- **GM_DRUM_MAP**: General MIDI drum mapping (35-81)
- **MIDI_CC**: Control change numbers
- **DYNAMICS**: Velocity mappings (pppp=8 to ffff=127)
- **GM_PROGRAMS**: All 128 GM instrument programs
- **INSTRUMENT_RANGES**: Valid pitch ranges per instrument
- **PPQN**: Ticks per quarter note (480, 960, 1920)

---

## 6. Missing Components (What We Need to Build)

### 6.1 No Existing Decoder

**Finding**: There is **NO MIDI decoder** in the codebase.
- No DNA → MIDI conversion
- No generative models for MIDI
- Only analysis direction exists (MIDI → Features → DNA)

**This is our primary task!**

### 6.2 No Differentiable MIDI Representations

**Finding**: No soft/differentiable MIDI representations exist.
- No pianoroll representation
- No soft sampling mechanisms
- No Gumbel-softmax or straight-through estimators

**These are critical for training the decoder!**

### 6.3 No MIDI Assembly Utilities

**Finding**: No utilities for assembling MIDI events into files.
- Feature extraction exists (MIDI → features)
- But no reverse (features/parameters → MIDI)
- Need to build: events → pretty_midi.PrettyMIDI objects

---

## 7. Coordination with Agent 1

### 7.1 What Agent 1 Needs from Me

Agent 1 (MIDI Decoder Architecture Lead) depends on:

1. **Differentiable pianoroll representation**
   - Continuous representation during training
   - Conversion to discrete MIDI for inference
   - Must support backpropagation

2. **Soft sampling strategies**
   - Gumbel-softmax for differentiable sampling
   - Temperature-based annealing
   - Straight-through estimators

3. **MIDI assembly utilities**
   - Convert decoder outputs (pitch, time, duration, velocity, track) to MIDI
   - Handle multi-track coordination
   - Ensure valid MIDI format

4. **Validation utilities**
   - Check MIDI validity
   - Detect common errors
   - Quality metrics

### 7.2 Interface Design (Preliminary)

**My proposed interface** (to discuss with Agent 1):

```python
# 1. Soft Pianoroll Representation
class SoftPianoRoll:
    """Differentiable pianoroll representation"""
    def __init__(self, time_steps, pitch_range=(21, 108), num_tracks=8)
    def to_midi(self, threshold=0.5) -> pretty_midi.PrettyMIDI
    def from_midi(midi: pretty_midi.PrettyMIDI) -> SoftPianoRoll

# 2. Gumbel-Softmax Sampling
class GumbelSoftmaxSampler:
    """Differentiable discrete sampling"""
    def sample(logits, temperature, hard=False) -> torch.Tensor
    def anneal_temperature(step, initial_temp, final_temp, anneal_rate)

# 3. MIDI Assembly
class MIDIAssembler:
    """Convert continuous predictions to MIDI"""
    def assemble(
        pitch_probs,      # (T, 128) probabilities
        onset_probs,      # (T,) onset probabilities
        duration_values,  # (T,) duration predictions
        velocity_values,  # (T,) velocity predictions
        track_probs       # (T, num_tracks) track assignment
    ) -> pretty_midi.PrettyMIDI

# 4. Validation
class MIDIValidator:
    """Validate generated MIDI"""
    def validate(midi: pretty_midi.PrettyMIDI) -> Dict[str, Any]
    def get_quality_metrics(midi) -> Dict[str, float]
```

---

## 8. Technical Decisions & Recommendations

### 8.1 Representation Choice: Pianoroll vs Event-Based

**Recommendation**: Start with **pianoroll** for Phase 3 implementation.

**Reasons**:
1. **Simpler to make differentiable** (continuous grid)
2. **Easier multi-track coordination** (parallel tracks)
3. **Natural CNN/Transformer input** (image-like)
4. **Proven in MusicVAE, Music Transformer**

**Pianoroll design**:
- Shape: `(time_steps, 128, num_tracks)` or `(num_tracks, time_steps, 128)`
- Time resolution: 16th notes (PPQN/4)
- Pitch range: 21-108 (A0 to C8, 88 keys)
- Values: Continuous [0, 1] during training, binary {0, 1} during inference

**Event-based** can be added later for better expressiveness.

### 8.2 Sampling Strategy: Gumbel-Softmax

**Recommendation**: Use **Gumbel-Softmax** with temperature annealing.

**Why**:
- Allows discrete sampling with continuous gradients
- Temperature controls exploration vs exploitation
- Can use "hard" mode during inference (argmax)
- Standard in VAE literature

**Temperature schedule**:
```python
temperature = max(
    final_temp,
    initial_temp * exp(-anneal_rate * step)
)
```
- Initial: 1.0 (soft, exploratory)
- Final: 0.1 (hard, deterministic)
- Anneal rate: 0.0001

### 8.3 MIDI Assembly Strategy

**Recommendation**: Multi-stage assembly process.

**Pipeline**:
1. **Decoder outputs** → Continuous logits/values
2. **Soft sampling** → Soft discrete distributions
3. **Thresholding** → Binary pianoroll
4. **Post-processing** → Note events (pitch, onset, duration, velocity)
5. **MIDI creation** → pretty_midi.PrettyMIDI object

**Post-processing steps**:
- Minimum note duration: 1 time step
- Velocity quantization: Round to nearest int [1, 127]
- Remove overlapping notes on same pitch
- Ensure note_off after note_on

### 8.4 Memory Efficiency

**Challenge**: Pianoroll can be memory-intensive.
- Shape: `(batch, tracks, time, pitch)` = `(32, 8, 2048, 128)` = 67M floats
- Memory: 67M × 4 bytes = 268 MB per batch (manageable)

**Optimizations**:
- Use `torch.float16` during training (mixed precision)
- Sparse representations for empty tracks
- Gradient checkpointing if needed
- Process long sequences in chunks

---

## 9. Dependencies & Tools

### 9.1 Required Python Packages

Already available in codebase:
- ✅ `torch` - Neural networks, autograd
- ✅ `numpy` - Numerical operations
- ✅ `pretty_midi` - MIDI I/O
- ✅ `mido` - Low-level MIDI
- ✅ `scipy` - Statistical functions

Need to verify/install:
- ❓ `torch.nn.functional.gumbel_softmax` (should be in PyTorch)
- ❓ `torch.cuda.amp` (mixed precision - in PyTorch 1.6+)

### 9.2 File Locations for Implementation

**My files** (to be created in Phase 3):
```
/home/user/Do/midi_generator/utils/
├── differentiable_midi.py       # SoftPianoRoll, representations
├── soft_sampling.py              # Gumbel-softmax, temperature annealing
├── midi_assembly.py              # Event assembly, MIDI creation
└── midi_validation.py            # Validation, quality metrics

/home/user/Do/midi_generator/tests/
├── test_differentiable_midi.py
├── test_soft_sampling.py
└── test_midi_assembly.py
```

---

## 10. Research Topics Summary

### 10.1 Papers to Reference

1. **Gumbel-Softmax**
   - "Categorical Reparameterization with Gumbel-Softmax" (Jang et al., 2017)
   - "The Concrete Distribution" (Maddison et al., 2017)

2. **Music Generation**
   - "Music Transformer" (Huang et al., 2018) - attention-based MIDI generation
   - "MusicVAE" (Roberts et al., 2018) - hierarchical VAE for music
   - "MuseGAN" (Dong et al., 2018) - multi-track generation

3. **Differentiable DSP**
   - "DDSP: Differentiable Digital Signal Processing" (Engel et al., 2020)
   - "Differentiable Audio Rendering" (Kumar et al., 2019)

### 10.2 Key Concepts

- **Reparameterization trick**: Sample from N(μ, σ) via z = μ + σ * ε where ε ~ N(0,1)
- **Gumbel-Softmax**: Differentiable approximation to argmax
- **Straight-through estimator**: Forward pass discrete, backward pass continuous
- **Temperature annealing**: Gradually reduce temperature for sharper distributions
- **Pianoroll**: Time-pitch grid representation of music

---

## 11. Next Steps (Phase 2: Design)

### 11.1 Immediate Actions

1. **Design document** for differentiable representations
   - Finalize pianoroll format (shape, resolution, range)
   - Define interface contracts with Agent 1
   - Specify data types, shapes, ranges

2. **Architecture diagrams**
   - Draw pianoroll → MIDI conversion flow
   - Show Gumbel-softmax sampling process
   - Illustrate multi-track assembly

3. **Coordinate with Agent 1**
   - Share proposed interfaces
   - Agree on representation format
   - Define decoder output specifications

### 11.2 Phase 3 Implementation Priorities

**Week 2** (Days 8-14):
- Implement `SoftPianoRoll` class
- Basic conversion: pianoroll ↔ MIDI
- Unit tests for representations

**Week 3** (Days 15-21):
- Implement Gumbel-softmax sampling
- Temperature annealing schedules
- Gradient flow tests

**Week 4** (Days 22-28):
- MIDI assembly utilities
- Validation and quality metrics
- Integration tests with Agent 1's decoder

---

## 12. Risk Assessment

### 12.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Gradient vanishing/explosion | Medium | High | Use gradient clipping, careful initialization |
| Memory overflow (large pianorolls) | Medium | Medium | Mixed precision, gradient checkpointing |
| Invalid MIDI generation | High | Medium | Robust validation, post-processing rules |
| Interface mismatch with Agent 1 | Low | High | Early coordination, clear contracts |
| Sampling too soft (blurry outputs) | Medium | Medium | Temperature annealing, hard sampling at inference |

### 12.2 Coordination Risks

- **Agent 1 dependency**: They need my representations to build decoder
  - **Mitigation**: Deliver Phase 3 Week 2 outputs early (by Day 10)

- **Agent 5 dependency**: They need my MIDI distance metrics
  - **Mitigation**: Define metric interface in Phase 2, implement in Phase 3 Week 4

---

## 13. Success Metrics (Phase 1)

### 13.1 Research Objectives ✅

- ✅ Understand existing MIDI I/O code (mido, pretty_midi)
- ✅ Identify current representations (Note, Chord structures)
- ✅ Study encoder architecture patterns (6 modular encoders)
- ✅ Map feature extraction pipeline (1150D → DNA)
- ✅ Document musical transformations (12 locality functions)
- ✅ Identify gaps (no decoder, no differentiable representations)

### 13.2 Deliverables ✅

- ✅ This comprehensive research summary document
- ✅ Clear understanding of coordination points with Agent 1
- ✅ Technical decisions made (pianoroll, Gumbel-softmax)
- ✅ Implementation plan for Phase 3

---

## 14. Appendix: Code Examples

### 14.1 Current MIDI Loading Pattern

```python
# From rich_feature_extractor.py:151
def extract(self, midi_path: str) -> np.ndarray:
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return np.zeros(self.TOTAL_DIM)

    # Extract features...
```

### 14.2 Current Note Parsing

```python
# From deep_feature_extractor.py:166
def _parse_notes(self, midi: mido.MidiFile) -> List[Note]:
    notes = []
    current_notes = {}  # key: (pitch, channel), value: (velocity, start_time)

    for track in midi.tracks:
        track_time = 0.0
        for msg in track:
            track_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                key = (msg.note, msg.channel)
                current_notes[key] = (msg.velocity, track_time)

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in current_notes:
                    velocity, start_time = current_notes.pop(key)
                    duration = track_time - start_time
                    notes.append(Note(
                        pitch=msg.note,
                        velocity=velocity,
                        start_time=start_time,
                        end_time=track_time,
                        duration=duration,
                        channel=msg.channel
                    ))

    return notes
```

### 14.3 Musical Transformation Example

```python
# From musical_locality.py:285
def transpose(self, midi_file: MidiFile, params: Dict[str, Any]) -> MidiFile:
    semitones = params.get("semitones", 0)
    result = self._copy_midi(midi_file)

    for track in result.tracks:
        for msg in track:
            if msg.type in ('note_on', 'note_off') and hasattr(msg, 'note'):
                new_note = np.clip(msg.note + semitones, 0, 127)
                msg.note = int(new_note)

    return result
```

---

## Conclusion

Phase 1 research is **COMPLETE**. I have a clear understanding of:
- Existing codebase architecture
- What needs to be built (differentiable MIDI representations)
- How to coordinate with Agent 1
- Technical approach (pianoroll + Gumbel-softmax)

**Ready to proceed to Phase 2: Design** (Days 4-7).

---

**Prepared by**: Agent 2 - Differentiable MIDI & Utilities Support
**Date**: November 22, 2025
**Next Phase**: Design Document (Days 4-7)
