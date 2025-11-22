# Agent 2: Differentiable MIDI & Utilities - Phase 2 Design Document

**Date**: November 22, 2025
**Agent**: Agent 2 - Differentiable MIDI & Utilities Support
**Phase**: Phase 2 - Design (Days 4-7)
**Status**: COMPLETE

---

## Executive Summary

This document provides detailed architectural designs for the four core components that Agent 2 will implement:

1. **SoftPianoRoll** - Differentiable MIDI representation
2. **GumbelSoftmaxSampler** - Differentiable discrete sampling
3. **MIDIAssembler** - MIDI assembly from decoder outputs
4. **MIDIValidator** - Validation and quality metrics

All designs are optimized for:
- **Differentiability** - Full gradient flow for end-to-end training
- **Efficiency** - GPU-friendly, vectorized operations
- **Compatibility** - Works with Agent 1's decoder and Agent 5's training loop
- **Robustness** - Handles edge cases, invalid inputs gracefully

---

## 1. SoftPianoRoll - Differentiable MIDI Representation

### 1.1 Architectural Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     SoftPianoRoll                            │
│                                                              │
│  Input: MIDI File (pretty_midi.PrettyMIDI)                  │
│         ↓                                                    │
│  Parse: Extract notes (pitch, onset, duration, velocity)    │
│         ↓                                                    │
│  Grid:  Quantize to time-pitch grid                         │
│         ↓                                                    │
│  Tensor: (batch, tracks, time_steps, pitches)               │
│         ↓                                                    │
│  Output: Differentiable tensor representation               │
│                                                              │
│  Reverse Direction:                                          │
│  Tensor → Threshold → Note Events → pretty_midi.PrettyMIDI  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Tensor Format Specification

**Shape**: `(batch_size, num_tracks, time_steps, num_pitches)`

**Default Configuration**:
- `batch_size`: Variable (typically 32)
- `num_tracks`: 8 (matches multi-track MIDI)
- `time_steps`: 2048 (configurable, ~8 minutes at 0.25s resolution)
- `num_pitches`: 88 (piano range A0-C8, MIDI 21-108)

**Alternative**: Full MIDI range
- `num_pitches`: 128 (full MIDI range 0-127)

**Value Range**:
- Training: Continuous [0.0, 1.0] (soft probabilities)
- Inference: Binary {0, 1} (hard decisions)

**Time Resolution**:
- Default: 0.25 seconds per step (16th note at 120 BPM)
- Configurable: 0.125s (32nd notes) or 0.5s (8th notes)

### 1.3 Data Structure Design

```python
@dataclass
class PianoRollConfig:
    """Configuration for pianoroll representation"""
    time_steps: int = 2048
    num_pitches: int = 88
    pitch_offset: int = 21  # A0 = MIDI note 21
    num_tracks: int = 8
    time_resolution: float = 0.25  # seconds per step
    fps: int = 4  # frames per second (1 / time_resolution)
    default_velocity: int = 64
    min_note_duration: float = 0.05  # 50ms minimum

class SoftPianoRoll:
    def __init__(self, config: PianoRollConfig = None, device: str = 'cpu'):
        self.config = config or PianoRollConfig()
        self.device = device
        self.data = None  # torch.Tensor

    # Core methods (see interface spec)
```

### 1.4 MIDI → Pianoroll Conversion Algorithm

```python
def from_midi(midi: PrettyMIDI) -> torch.Tensor:
    """
    Convert MIDI to pianoroll tensor.

    Algorithm:
    1. Extract all notes from all instruments
    2. Group notes by track/instrument
    3. For each note:
       - Convert onset time to time step: step = int(onset / time_resolution)
       - Convert offset time to time step
       - Convert MIDI pitch to pianoroll pitch: pr_pitch = midi_pitch - pitch_offset
       - Set pianoroll[track, step:step+duration, pr_pitch] = 1.0
    4. Handle velocity (optional channel): store in separate tensor
    5. Return tensor
    """

    # Pseudocode:
    pianoroll = torch.zeros((num_tracks, time_steps, num_pitches))

    for track_idx, instrument in enumerate(midi.instruments[:num_tracks]):
        for note in instrument.notes:
            start_step = int(note.start / time_resolution)
            end_step = int(note.end / time_resolution)
            pitch_idx = note.pitch - pitch_offset

            if 0 <= pitch_idx < num_pitches and start_step < time_steps:
                end_step = min(end_step, time_steps)
                pianoroll[track_idx, start_step:end_step, pitch_idx] = 1.0

    return pianoroll
```

### 1.5 Pianoroll → MIDI Conversion Algorithm

```python
def to_midi(pianoroll: torch.Tensor, threshold: float = 0.5) -> PrettyMIDI:
    """
    Convert pianoroll to MIDI file.

    Algorithm:
    1. Binarize: pianoroll_binary = (pianoroll > threshold).float()
    2. Detect note onsets: onset = (pr[t] > 0) & (pr[t-1] == 0)
    3. Detect note offsets: offset = (pr[t] == 0) & (pr[t-1] > 0)
    4. Match onsets with offsets to create note events
    5. Convert indices back to MIDI:
       - pitch = pitch_idx + pitch_offset
       - time = step * time_resolution
    6. Create PrettyMIDI object with notes
    """

    # Pseudocode:
    midi = PrettyMIDI()

    for track_idx in range(num_tracks):
        instrument = Instrument(program=track_programs[track_idx])
        track_roll = pianoroll[track_idx]  # (time_steps, num_pitches)

        # Find active notes using diff
        for pitch_idx in range(num_pitches):
            pitch_roll = track_roll[:, pitch_idx]  # (time_steps,)

            # Detect onsets (0 → 1 transitions)
            onsets = find_onsets(pitch_roll, threshold)
            offsets = find_offsets(pitch_roll, threshold)

            # Create notes
            for onset_step, offset_step in zip(onsets, offsets):
                note = Note(
                    velocity=default_velocity,
                    pitch=pitch_idx + pitch_offset,
                    start=onset_step * time_resolution,
                    end=offset_step * time_resolution
                )
                instrument.notes.append(note)

        midi.instruments.append(instrument)

    return midi
```

### 1.6 Efficiency Optimizations

**Memory Optimization**:
- Use sparse tensors for mostly-empty pianorolls
- Store only active regions (crop silence)
- Use `torch.float16` for storage (convert to float32 for computation)

**Computation Optimization**:
- Vectorize all operations (no Python loops over notes)
- Use batch processing for MIDI → pianoroll conversion
- Cache pianoroll tensors during training (avoid repeated conversion)

---

## 2. GumbelSoftmaxSampler - Differentiable Discrete Sampling

### 2.1 Mathematical Foundation

**Gumbel-Softmax Distribution** (Concrete Distribution):

Given logits `z = (z₁, ..., zₖ)`, sample from categorical distribution while maintaining gradients.

**Standard Softmax** (not differentiable for sampling):
```
p_i = exp(z_i) / Σⱼ exp(z_j)
```

**Gumbel-Softmax** (differentiable):
```
y_i = exp((log(p_i) + g_i) / τ) / Σⱼ exp((log(p_j) + g_j) / τ)

where g_i ~ Gumbel(0, 1) = -log(-log(u_i)), u_i ~ Uniform(0, 1)
      τ = temperature parameter
```

**Properties**:
- As τ → 0: Approaches one-hot (deterministic)
- As τ → ∞: Approaches uniform distribution
- For all τ > 0: Fully differentiable

**Straight-Through Estimator** (for hard samples):
- Forward pass: y = one_hot(argmax(gumbel_softmax(z, τ)))
- Backward pass: ∇y = ∇gumbel_softmax(z, τ) (soft gradients)

### 2.2 Implementation Architecture

```
┌──────────────────────────────────────────────────────────┐
│              GumbelSoftmaxSampler                         │
│                                                           │
│  Input: logits (unnormalized log probabilities)          │
│         ↓                                                 │
│  1. Sample Gumbel noise: g ~ Gumbel(0, 1)               │
│         ↓                                                 │
│  2. Add noise: z' = logits + g                           │
│         ↓                                                 │
│  3. Temperature scaling: z'' = z' / temperature          │
│         ↓                                                 │
│  4. Softmax: y_soft = softmax(z'')                       │
│         ↓                                                 │
│  5. [Optional] Hard: y_hard = one_hot(argmax(y_soft))   │
│         ↓                                                 │
│  6. [Optional] Straight-through: y = y_hard - y_soft.detach() + y_soft │
│         ↓                                                 │
│  Output: Differentiable sample                           │
└──────────────────────────────────────────────────────────┘
```

### 2.3 Temperature Annealing Schedule

**Exponential Decay**:
```python
temperature = max(min_temp, initial_temp * exp(-decay_rate * step))
```

**Linear Decay**:
```python
temperature = max(min_temp, initial_temp - decay_rate * step)
```

**Recommended Schedule**:
- Initial temperature: 1.0 (soft, exploratory)
- Final temperature: 0.1 (hard, near-deterministic)
- Decay rate: 0.00003 (reaches 0.1 at ~60k steps)
- Decay type: Exponential

**Visual Schedule**:
```
Temperature
1.0 ┤●
    │ ●
    │  ●
    │   ●●
0.5 ┤     ●●●
    │        ●●●●
    │            ●●●●●
    │                 ●●●●●●●●●●●●●●
0.1 ┤                                 ━━━━━━━━━━━
    └─────────────────────────────────────────────→ Steps
    0     20k    40k    60k    80k   100k
```

### 2.4 Gradient Flow Analysis

**Forward Pass** (hard=True):
```
Input logits → Gumbel noise → Softmax → Argmax → One-hot
               (continuous)              (discrete, no gradient!)
```

**Backward Pass** (straight-through):
```
Gradient ← soft Softmax ← Gumbel noise ← Input logits
           (differentiable)
```

**Key Insight**: Hard sampling in forward (sharp outputs), soft gradients in backward (learning).

---

## 3. MIDIAssembler - MIDI Assembly from Decoder Outputs

### 3.1 Multi-Modal Assembly Architecture

Decoder outputs multiple modalities that must be combined:

```
Decoder Output Streams:
┌────────────────────────────────────────────────┐
│ 1. Pitch Logits:    (batch, tracks, time, 128) │
│ 2. Onset Probs:     (batch, tracks, time)      │
│ 3. Duration Values: (batch, tracks, time)      │
│ 4. Velocity Values: (batch, tracks, time)      │
└────────────────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │   Gumbel-Softmax      │
        │   (pitch sampling)    │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  Onset Detection      │
        │  (threshold > 0.5)    │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  Note Event Creation  │
        │  (match onset/offset) │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  Value Quantization   │
        │  (velocity: 1-127)    │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  Post-Processing      │
        │  (remove overlaps)    │
        └───────────────────────┘
                    ↓
           PrettyMIDI Object
```

### 3.2 Note Event Matching Algorithm

**Problem**: Match note onsets with offsets to create complete note events.

**Algorithm**:
```python
def match_onset_offset(onset_prob, duration_pred):
    """
    Create note events from onset predictions and durations.

    Args:
        onset_prob: (time_steps,) onset probabilities
        duration_pred: (time_steps,) predicted durations

    Returns:
        List of (start_time, end_time) tuples
    """
    note_events = []

    # Detect onsets
    onsets = torch.where(onset_prob > 0.5)[0]

    for onset_step in onsets:
        # Get predicted duration
        duration = duration_pred[onset_step]
        duration_steps = int(duration / time_resolution)

        # Calculate offset
        offset_step = onset_step + duration_steps

        # Convert to seconds
        start_time = onset_step * time_resolution
        end_time = offset_step * time_resolution

        note_events.append((start_time, end_time))

    return note_events
```

### 3.3 Pianoroll-Based Assembly (Simpler Alternative)

For pianoroll representation, assembly is simpler:

```python
def assemble_from_pianoroll(pianoroll, velocities=None):
    """
    Simpler assembly when using pianoroll.

    Pianoroll already encodes onset + pitch + duration.
    Just need to extract notes and add velocities.
    """
    # Use connected components or run-length encoding
    # to extract continuous note regions

    for track_idx in range(num_tracks):
        for pitch_idx in range(num_pitches):
            # Find contiguous regions where pianoroll[track, :, pitch] > threshold
            regions = find_contiguous_regions(pianoroll[track_idx, :, pitch_idx])

            for start, end in regions:
                note = Note(
                    pitch=pitch_idx + pitch_offset,
                    start=start * time_resolution,
                    end=end * time_resolution,
                    velocity=velocities[track_idx, start, pitch_idx] if velocities else 64
                )
                # Add to instrument
```

### 3.4 Post-Processing Rules

**Remove Overlapping Notes** (same pitch, same track):
```python
def remove_overlaps(notes):
    """Sort by start time, merge overlapping notes on same pitch"""
    notes.sort(key=lambda n: (n.pitch, n.start))

    cleaned = []
    for note in notes:
        if cleaned and note.pitch == cleaned[-1].pitch:
            # If overlap, extend previous note or skip
            if note.start < cleaned[-1].end:
                cleaned[-1].end = max(cleaned[-1].end, note.end)
                continue
        cleaned.append(note)

    return cleaned
```

**Quantize Timing**:
```python
def quantize_timing(notes, grid_size=0.125):
    """Snap note onsets to nearest grid point"""
    for note in notes:
        note.start = round(note.start / grid_size) * grid_size
        note.end = round(note.end / grid_size) * grid_size
```

**Enforce Minimum Duration**:
```python
def enforce_min_duration(notes, min_duration=0.05):
    """Ensure all notes are at least min_duration long"""
    for note in notes:
        if note.end - note.start < min_duration:
            note.end = note.start + min_duration
```

---

## 4. MIDIValidator - Validation and Quality Metrics

### 4.1 Validation Checks

**Basic Validation**:
1. File format valid (PrettyMIDI can parse)
2. At least one instrument
3. At least one note
4. All pitches in valid range [0, 127]
5. All velocities in valid range [1, 127]
6. All times non-negative
7. All durations positive (end > start)

**Advanced Validation**:
1. No overlapping notes on same pitch/track
2. Reasonable tempo (40-240 BPM)
3. Reasonable time signature
4. Notes don't exceed instrument range
5. No stuck notes (very long durations > 60s)

### 4.2 Quality Metrics

**Note-Level Metrics**:
```python
def compute_note_metrics(midi):
    notes = [n for inst in midi.instruments for n in inst.notes]

    return {
        'num_notes': len(notes),
        'note_density': len(notes) / midi.get_end_time(),  # notes per second
        'avg_duration': mean([n.end - n.start for n in notes]),
        'avg_velocity': mean([n.velocity for n in notes]),
        'pitch_range': max([n.pitch for n in notes]) - min([n.pitch for n in notes]),
        'avg_pitch': mean([n.pitch for n in notes])
    }
```

**Polyphony Metrics**:
```python
def compute_polyphony(midi, time_resolution=0.01):
    """Average and max simultaneous notes"""
    end_time = midi.get_end_time()
    time_steps = int(end_time / time_resolution)

    active_notes = np.zeros(time_steps)

    for inst in midi.instruments:
        for note in inst.notes:
            start_idx = int(note.start / time_resolution)
            end_idx = int(note.end / time_resolution)
            active_notes[start_idx:end_idx] += 1

    return {
        'avg_polyphony': active_notes.mean(),
        'max_polyphony': active_notes.max()
    }
```

**Rhythmic Metrics**:
```python
def compute_rhythmic_metrics(midi):
    """Measure rhythmic regularity"""
    onsets = sorted([n.start for inst in midi.instruments for n in inst.notes])
    iois = np.diff(onsets)  # inter-onset intervals

    return {
        'avg_ioi': iois.mean(),
        'ioi_std': iois.std(),
        'rhythmic_regularity': 1.0 / (1.0 + iois.std())  # [0, 1], 1 = regular
    }
```

### 4.3 MIDI Distance Metrics (for Reconstruction Loss)

**Pitch Distance**:
```python
def pitch_distance(midi1, midi2):
    """
    Compare pitch content.

    Uses pitch class histogram distance.
    """
    hist1 = compute_pitch_class_histogram(midi1)
    hist2 = compute_pitch_class_histogram(midi2)

    # Earth mover's distance or KL divergence
    return earth_movers_distance(hist1, hist2)
```

**Rhythm Distance**:
```python
def rhythm_distance(midi1, midi2):
    """
    Compare rhythmic patterns.

    Uses inter-onset interval distribution.
    """
    ioi1 = compute_ioi_distribution(midi1)
    ioi2 = compute_ioi_distribution(midi2)

    return wasserstein_distance(ioi1, ioi2)
```

**Note-Level F1 Score**:
```python
def note_f1_score(midi1, midi2, tolerance=0.05):
    """
    Precision/recall for note matching.

    Two notes match if:
    - Same pitch
    - Onset within tolerance
    - Duration within 50%
    """
    notes1 = extract_notes(midi1)
    notes2 = extract_notes(midi2)

    matches = find_matches(notes1, notes2, tolerance)

    precision = len(matches) / len(notes2) if notes2 else 0
    recall = len(matches) / len(notes1) if notes1 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}
```

---

## 5. Integration with Other Agents

### 5.1 Agent 1 (Decoder) Integration Points

**What Agent 1 provides to me**:
```python
# Decoder output format (example)
decoder_output = {
    'pianoroll': torch.Tensor,  # (batch, tracks, time, pitch)
    # OR
    'pitch_logits': torch.Tensor,  # (batch, tracks, time, 128)
    'onset_probs': torch.Tensor,   # (batch, tracks, time)
    'duration': torch.Tensor,      # (batch, tracks, time)
    'velocity': torch.Tensor,      # (batch, tracks, time)
}
```

**What I provide to Agent 1**:
```python
# For training data
pianoroll = SoftPianoRoll.from_midi('train.mid')
train_data = pianoroll.to_tensor()  # Input to decoder

# For generated MIDI
generated_midi = SoftPianoRoll.to_midi(decoder_output['pianoroll'])
```

### 5.2 Agent 5 (Training) Integration Points

**Reconstruction Loss**:
```python
def reconstruction_loss(original_midi, reconstructed_midi):
    """
    Multi-component loss for MIDI reconstruction.
    """
    # Component losses
    pitch_loss = MIDIValidator.pitch_distance(original_midi, reconstructed_midi)
    rhythm_loss = MIDIValidator.rhythm_distance(original_midi, reconstructed_midi)
    note_f1 = MIDIValidator.note_f1_score(original_midi, reconstructed_midi)

    # Weighted combination
    total_loss = 0.4 * pitch_loss + 0.4 * rhythm_loss + 0.2 * (1.0 - note_f1['f1'])

    return total_loss
```

**Training Loop**:
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Load MIDI
        midi_files = batch['midi_paths']
        pianorolls = [SoftPianoRoll.from_midi(f) for f in midi_files]

        # Encode
        dna = encoder(features)

        # Decode
        reconstructed_pianorolls = decoder(dna)

        # Convert to MIDI for evaluation
        reconstructed_midis = [SoftPianoRoll.to_midi(pr) for pr in reconstructed_pianorolls]

        # Compute loss
        loss = sum([reconstruction_loss(orig, recon)
                    for orig, recon in zip(midi_files, reconstructed_midis)])
```

---

## 6. File Structure and Organization

```
midi_generator/
├── utils/
│   ├── __init__.py
│   ├── differentiable_midi.py       # SoftPianoRoll class
│   ├── soft_sampling.py              # GumbelSoftmaxSampler class
│   ├── midi_assembly.py              # MIDIAssembler class
│   └── midi_validation.py            # MIDIValidator class
│
├── tests/
│   ├── test_differentiable_midi.py
│   ├── test_soft_sampling.py
│   ├── test_midi_assembly.py
│   └── test_midi_validation.py
│
└── docs/
    ├── AGENT2_PHASE1_RESEARCH_SUMMARY.md  ✅
    ├── AGENT2_PHASE2_DESIGN.md            ✅
    ├── AGENT2_INTERFACES.md               ✅
    └── AGENT2_IMPLEMENTATION_GUIDE.md     (Phase 3)
```

---

## 7. Technical Specifications Summary

### 7.1 Dependencies

```python
# Core
import torch
import torch.nn.functional as F
import numpy as np

# MIDI
import pretty_midi
import mido

# Utilities
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
```

### 7.2 Key Constants

```python
# Pianoroll
DEFAULT_TIME_STEPS = 2048
DEFAULT_NUM_PITCHES = 88
DEFAULT_PITCH_OFFSET = 21  # A0
DEFAULT_NUM_TRACKS = 8
DEFAULT_TIME_RESOLUTION = 0.25  # seconds

# Sampling
DEFAULT_INITIAL_TEMP = 1.0
DEFAULT_FINAL_TEMP = 0.1
DEFAULT_TEMP_DECAY = 0.00003

# MIDI
MIN_NOTE_DURATION = 0.05  # 50ms
DEFAULT_VELOCITY = 64
VELOCITY_RANGE = (1, 127)

# Validation
ONSET_TOLERANCE = 0.05  # 50ms
PITCH_MATCH_EXACT = True
DURATION_TOLERANCE = 0.5  # 50% of original
```

### 7.3 Memory Requirements

**Pianoroll Storage**:
- Shape: (32, 8, 2048, 88)
- Float32: 32 × 8 × 2048 × 88 × 4 bytes = 183 MB
- Float16: 91 MB (2× memory savings)

**Recommendations**:
- Use float16 for storage, convert to float32 for computation
- Clear cache between batches
- Use gradient checkpointing if OOM

---

## 8. Implementation Priorities (Phase 3)

### Week 2 (Days 8-14): Core Representations
**Priority 1**: SoftPianoRoll class
- [ ] Data structure and config
- [ ] from_midi() conversion
- [ ] to_midi() conversion
- [ ] to_tensor() method
- [ ] Unit tests

**Priority 2**: Basic validation
- [ ] Validate pianoroll format
- [ ] Test MIDI I/O round-trip

### Week 3 (Days 15-21): Sampling
**Priority 1**: GumbelSoftmaxSampler
- [ ] sample() method
- [ ] Temperature annealing
- [ ] Straight-through estimator
- [ ] Gradient flow tests

**Priority 2**: Integration tests
- [ ] Test with dummy decoder outputs
- [ ] Verify differentiability

### Week 4 (Days 22-28): Assembly & Validation
**Priority 1**: MIDIAssembler
- [ ] assemble_from_pianoroll()
- [ ] Multi-modal assembly
- [ ] Post-processing

**Priority 2**: MIDIValidator
- [ ] Validation checks
- [ ] Quality metrics
- [ ] Distance metrics for Agent 5

**Priority 3**: Integration
- [ ] Full pipeline test
- [ ] Agent 1 coordination
- [ ] Documentation

---

## 9. Risk Mitigation

### Technical Risks

| Risk | Mitigation Strategy |
|------|---------------------|
| Gradient vanishing | Use gradient clipping (max_norm=1.0), careful initialization |
| Memory overflow | Mixed precision (float16), gradient checkpointing, smaller batches |
| Invalid MIDI generation | Robust post-processing, extensive validation |
| Slow conversion | Vectorize all operations, cache pianorolls, use numba for critical loops |
| Non-differentiable operations | Use Gumbel-softmax, avoid argmax/threshold in forward pass during training |

### Coordination Risks

| Risk | Mitigation Strategy |
|------|---------------------|
| Interface mismatch with Agent 1 | Early prototype sharing, clear documentation |
| Agent 5 needs metrics before ready | Implement MIDIValidator first (Week 2) |
| Changing requirements | Modular design, clear separation of concerns |

---

## 10. Success Criteria (Phase 2)

### Design Objectives ✅

- ✅ Complete architectural design for all 4 components
- ✅ Detailed algorithms and pseudocode
- ✅ Integration points clearly defined
- ✅ Memory and performance analysis
- ✅ Risk assessment and mitigation
- ✅ Implementation roadmap for Phase 3

### Deliverables ✅

- ✅ This comprehensive design document
- ✅ Clear specifications for implementation
- ✅ Ready to begin Phase 3 coding

---

## Conclusion

Phase 2 design is **COMPLETE**. All architectural decisions are made, algorithms designed, and specifications documented. Ready to proceed with implementation in Phase 3.

**Next**: Begin Week 2 implementation - SoftPianoRoll class.

---

**Prepared by**: Agent 2 - Differentiable MIDI & Utilities Support
**Date**: November 22, 2025
**Next Phase**: Implementation (Weeks 2-4)
