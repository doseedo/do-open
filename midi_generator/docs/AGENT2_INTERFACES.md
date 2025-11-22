# Agent 2: Proposed Interface Specifications

**For Coordination with**: Agent 1 (Decoder Lead), Agent 5 (Training Pipeline)
**Status**: DRAFT - Phase 1 Output
**Date**: November 22, 2025

---

## Overview

This document specifies the interfaces that Agent 2 will implement for use by other agents.

---

## 1. SoftPianoRoll - Differentiable MIDI Representation

### Purpose
Continuous, differentiable representation of MIDI for training neural decoders.

### Interface

```python
class SoftPianoRoll:
    """
    Differentiable pianoroll representation for MIDI.

    Shape: (batch, num_tracks, time_steps, num_pitches)
    Values: Continuous [0, 1] during training, binary {0, 1} during inference
    """

    def __init__(
        self,
        time_steps: int,
        num_pitches: int = 88,          # Default: A0-C8 (21-108)
        num_tracks: int = 8,
        time_resolution: float = 0.25,   # Time per step (seconds)
        device: str = 'cpu'
    ):
        """
        Initialize soft pianoroll.

        Args:
            time_steps: Number of time steps
            num_pitches: Pitch range (default 88 = piano)
            num_tracks: Number of simultaneous tracks
            time_resolution: Seconds per time step (default 0.25 = 16th note @ 120 BPM)
            device: 'cpu' or 'cuda'
        """
        pass

    @classmethod
    def from_midi(
        cls,
        midi_path: str,
        max_time_steps: int = 2048,
        num_tracks: int = 8
    ) -> 'SoftPianoRoll':
        """
        Create SoftPianoRoll from MIDI file.

        Args:
            midi_path: Path to MIDI file
            max_time_steps: Maximum time steps (truncate if longer)
            num_tracks: Number of tracks to extract

        Returns:
            SoftPianoRoll object
        """
        pass

    def to_midi(
        self,
        threshold: float = 0.5,
        min_duration: float = 0.1,
        default_velocity: int = 64
    ) -> 'pretty_midi.PrettyMIDI':
        """
        Convert soft pianoroll to MIDI file.

        Args:
            threshold: Threshold for binarizing continuous values
            min_duration: Minimum note duration (seconds)
            default_velocity: Default velocity for generated notes

        Returns:
            pretty_midi.PrettyMIDI object
        """
        pass

    def to_tensor(self) -> torch.Tensor:
        """
        Get underlying tensor representation.

        Returns:
            torch.Tensor of shape (batch, num_tracks, time_steps, num_pitches)
        """
        pass

    @staticmethod
    def get_note_events(
        pianoroll: torch.Tensor,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Extract note events from pianoroll.

        Args:
            pianoroll: Tensor of shape (num_tracks, time_steps, num_pitches)
            threshold: Threshold for note detection

        Returns:
            List of note dicts with keys: pitch, start_time, end_time, duration, velocity, track
        """
        pass
```

### Usage Example

```python
# Loading MIDI
pianoroll = SoftPianoRoll.from_midi('song.mid', max_time_steps=2048, num_tracks=8)
tensor = pianoroll.to_tensor()  # (1, 8, 2048, 88)

# For decoder output (continuous values from network)
decoder_output = decoder(dna_params)  # (batch, 8, 2048, 88) continuous [0, 1]
generated_midi = SoftPianoRoll.to_midi(decoder_output, threshold=0.5)
generated_midi.write('output.mid')
```

---

## 2. GumbelSoftmaxSampler - Differentiable Discrete Sampling

### Purpose
Sample discrete values (notes, onsets) while maintaining gradients for backpropagation.

### Interface

```python
class GumbelSoftmaxSampler:
    """
    Implements Gumbel-Softmax (Concrete distribution) for differentiable discrete sampling.

    Reference: "Categorical Reparameterization with Gumbel-Softmax" (Jang et al., 2017)
    """

    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
        dim: int = -1
    ) -> torch.Tensor:
        """
        Sample from Gumbel-Softmax distribution.

        Args:
            logits: Unnormalized log probabilities, shape (..., num_categories)
            temperature: Temperature parameter (lower = more discrete)
            hard: If True, return one-hot (forward) but soft (backward) - straight-through
            dim: Dimension to apply softmax

        Returns:
            Sampled tensor, same shape as logits
            - If hard=False: Soft probabilities (differentiable)
            - If hard=True: One-hot vectors (forward), soft (backward)
        """
        pass

    @staticmethod
    def anneal_temperature(
        step: int,
        initial_temp: float = 1.0,
        final_temp: float = 0.1,
        anneal_rate: float = 0.0001
    ) -> float:
        """
        Compute annealed temperature for training schedule.

        Args:
            step: Current training step
            initial_temp: Starting temperature (soft, exploratory)
            final_temp: Final temperature (hard, deterministic)
            anneal_rate: Exponential decay rate

        Returns:
            Current temperature

        Formula:
            temp = max(final_temp, initial_temp * exp(-anneal_rate * step))
        """
        pass
```

### Usage Example

```python
# During decoder forward pass
logits = decoder.predict_pitch_logits(hidden)  # (batch, time, 128) unnormalized
temperature = GumbelSoftmaxSampler.anneal_temperature(step, initial_temp=1.0, final_temp=0.1)

# Soft sampling (training)
soft_pitches = GumbelSoftmaxSampler.sample(logits, temperature=temperature, hard=False)
# soft_pitches shape: (batch, time, 128), continuous probabilities

# Hard sampling (inference)
hard_pitches = GumbelSoftmaxSampler.sample(logits, temperature=0.1, hard=True)
# hard_pitches shape: (batch, time, 128), one-hot vectors (but gradients flow through soft version)
```

---

## 3. MIDIAssembler - Convert Continuous Predictions to MIDI

### Purpose
Assemble multi-modal decoder outputs (pitch, onset, duration, velocity, track) into valid MIDI files.

### Interface

```python
class MIDIAssembler:
    """
    Assemble MIDI from decoder predictions.

    Handles:
    - Multi-track coordination
    - Note event creation from separate predictions
    - Velocity, duration, timing quantization
    - MIDI validation and error correction
    """

    def __init__(
        self,
        tempo: int = 120,
        time_signature: Tuple[int, int] = (4, 4),
        min_note_duration: float = 0.05,
        default_programs: List[int] = None
    ):
        """
        Initialize MIDI assembler.

        Args:
            tempo: BPM (default 120)
            time_signature: (numerator, denominator) default 4/4
            min_note_duration: Minimum note duration in seconds
            default_programs: GM program numbers for tracks (length num_tracks)
        """
        pass

    def assemble(
        self,
        pitch_probs: torch.Tensor,      # (num_tracks, time_steps, 128)
        onset_probs: torch.Tensor,      # (num_tracks, time_steps)
        duration_values: torch.Tensor,  # (num_tracks, time_steps)
        velocity_values: torch.Tensor,  # (num_tracks, time_steps)
        threshold: float = 0.5
    ) -> 'pretty_midi.PrettyMIDI':
        """
        Assemble MIDI from separate predictions.

        Args:
            pitch_probs: Pitch probabilities (can be soft or one-hot)
            onset_probs: Note onset probabilities
            duration_values: Note durations in seconds (continuous)
            velocity_values: Note velocities [0, 1] (will be scaled to [1, 127])
            threshold: Threshold for onset detection

        Returns:
            pretty_midi.PrettyMIDI object
        """
        pass

    def assemble_from_pianoroll(
        self,
        pianoroll: torch.Tensor,        # (num_tracks, time_steps, num_pitches)
        velocities: Optional[torch.Tensor] = None,  # (num_tracks, time_steps, num_pitches)
        threshold: float = 0.5
    ) -> 'pretty_midi.PrettyMIDI':
        """
        Assemble MIDI from pianoroll representation.

        Simpler interface when using pianoroll (onset + pitch are combined).

        Args:
            pianoroll: Binary or soft pianoroll
            velocities: Optional velocity for each active note (default: 64)
            threshold: Threshold for note detection

        Returns:
            pretty_midi.PrettyMIDI object
        """
        pass

    @staticmethod
    def post_process(
        midi: 'pretty_midi.PrettyMIDI',
        remove_overlap: bool = True,
        quantize_timing: bool = False,
        quantize_grid: float = 0.125  # 16th note @ 120 BPM
    ) -> 'pretty_midi.PrettyMIDI':
        """
        Post-process generated MIDI.

        Args:
            midi: Input MIDI
            remove_overlap: Remove overlapping notes on same pitch
            quantize_timing: Quantize note onsets to grid
            quantize_grid: Grid size in seconds

        Returns:
            Post-processed MIDI
        """
        pass
```

### Usage Example

```python
# From decoder outputs
assembler = MIDIAssembler(tempo=120, default_programs=[0, 32, 48, 56, 64])  # Piano, bass, strings, trumpet, sax

midi = assembler.assemble(
    pitch_probs=decoder_pitch,       # (8, 2048, 128)
    onset_probs=decoder_onset,       # (8, 2048)
    duration_values=decoder_duration, # (8, 2048)
    velocity_values=decoder_velocity, # (8, 2048)
    threshold=0.5
)

# Post-process
midi = MIDIAssembler.post_process(midi, remove_overlap=True, quantize_timing=True)
midi.write('generated.mid')
```

---

## 4. MIDIValidator - Validation and Quality Metrics

### Purpose
Validate generated MIDI and provide quality metrics for training feedback.

### Interface

```python
class MIDIValidator:
    """
    Validate MIDI files and compute quality metrics.
    """

    @staticmethod
    def validate(midi: 'pretty_midi.PrettyMIDI') -> Dict[str, Any]:
        """
        Validate MIDI file and return diagnostics.

        Args:
            midi: MIDI file to validate

        Returns:
            Dict with keys:
            - 'is_valid': bool
            - 'num_notes': int
            - 'num_tracks': int
            - 'duration': float (seconds)
            - 'errors': List[str] (validation errors)
            - 'warnings': List[str] (potential issues)
        """
        pass

    @staticmethod
    def get_quality_metrics(midi: 'pretty_midi.PrettyMIDI') -> Dict[str, float]:
        """
        Compute quality metrics for generated MIDI.

        Args:
            midi: MIDI file

        Returns:
            Dict with metrics:
            - 'note_density': Notes per second
            - 'pitch_range': Max pitch - min pitch
            - 'avg_velocity': Average velocity
            - 'polyphony': Average simultaneous notes
            - 'rhythmic_regularity': Measure of rhythmic structure [0, 1]
            - 'harmonic_consistency': Measure of harmonic coherence [0, 1]
        """
        pass

    @staticmethod
    def compute_distance(
        midi1: 'pretty_midi.PrettyMIDI',
        midi2: 'pretty_midi.PrettyMIDI',
        metrics: List[str] = ['pitch', 'rhythm', 'harmony']
    ) -> Dict[str, float]:
        """
        Compute distance between two MIDI files.

        Used by Agent 5 for reconstruction loss.

        Args:
            midi1: First MIDI
            midi2: Second MIDI
            metrics: Which metrics to compute

        Returns:
            Dict with distances:
            - 'pitch_distance': Pitch accuracy (0 = identical, 1 = completely different)
            - 'rhythm_distance': Rhythmic similarity
            - 'harmony_distance': Harmonic similarity
            - 'overall_distance': Weighted combination
        """
        pass
```

### Usage Example

```python
# Validate generated MIDI
validation = MIDIValidator.validate(generated_midi)
if not validation['is_valid']:
    print(f"Errors: {validation['errors']}")

# Quality metrics (for logging/monitoring)
quality = MIDIValidator.get_quality_metrics(generated_midi)
print(f"Note density: {quality['note_density']:.2f} notes/sec")
print(f"Polyphony: {quality['polyphony']:.2f} simultaneous notes")

# Distance for reconstruction loss (Agent 5)
distance = MIDIValidator.compute_distance(original_midi, reconstructed_midi)
reconstruction_loss = distance['overall_distance']
```

---

## 5. Data Types and Shapes

### Standard Tensor Shapes

```python
# Pianoroll
pianoroll_shape = (batch, num_tracks, time_steps, num_pitches)
# Example: (32, 8, 2048, 88)

# Pitch logits/probabilities
pitch_shape = (batch, num_tracks, time_steps, num_pitches)
# Example: (32, 8, 2048, 128)  # Full MIDI range 0-127

# Onset probabilities
onset_shape = (batch, num_tracks, time_steps)
# Example: (32, 8, 2048)

# Duration values (seconds)
duration_shape = (batch, num_tracks, time_steps)
# Example: (32, 8, 2048)

# Velocity values [0, 1]
velocity_shape = (batch, num_tracks, time_steps)
# Example: (32, 8, 2048)

# Track assignment (if needed)
track_probs_shape = (batch, time_steps, num_tracks)
# Example: (32, 2048, 8)
```

### Value Ranges

```python
# Pitch: MIDI note number
pitch_range = (0, 127)  # Full MIDI range
piano_range = (21, 108)  # A0 to C8

# Velocity: MIDI velocity
velocity_range = (1, 127)  # Note: 0 = note off

# Duration: seconds
duration_range = (0.05, 10.0)  # Minimum 50ms, maximum 10s

# Time resolution
time_resolution = 0.25  # seconds per time step (16th note @ 120 BPM)
# At 120 BPM: quarter note = 0.5s, 16th note = 0.125s, we use 0.25s for good resolution
```

---

## 6. Coordination Points

### With Agent 1 (Decoder Lead)

**Agent 1 needs from me**:
1. `SoftPianoRoll.to_midi()` - Convert decoder output to MIDI
2. `GumbelSoftmaxSampler.sample()` - Differentiable discrete sampling
3. `MIDIAssembler.assemble()` - Multi-modal output assembly

**I need from Agent 1**:
1. Decoder output format specification (pitch logits? pianoroll? separate streams?)
2. Whether to use pianoroll or event-based representation
3. Feedback on interface design (any changes needed?)

### With Agent 5 (Training Pipeline)

**Agent 5 needs from me**:
1. `MIDIValidator.compute_distance()` - MIDI reconstruction loss
2. `SoftPianoRoll.from_midi()` - Load training data
3. Validation utilities for monitoring

**I need from Agent 5**:
1. Training loop structure (when to call what?)
2. Loss function requirements (differentiable? batch-wise?)

---

## 7. Implementation Timeline

### Week 2 (Days 8-14): Core Representations
- ✅ `SoftPianoRoll` class
- ✅ `from_midi()` and `to_midi()` conversion
- ✅ Unit tests

**Deliverable**: Agent 1 can start using pianoroll representation

### Week 3 (Days 15-21): Sampling Strategies
- ✅ `GumbelSoftmaxSampler` implementation
- ✅ Temperature annealing schedules
- ✅ Gradient flow tests

**Deliverable**: Agent 1 can use differentiable sampling in decoder

### Week 4 (Days 22-28): Assembly and Validation
- ✅ `MIDIAssembler` for multi-modal outputs
- ✅ `MIDIValidator` with quality metrics
- ✅ Integration tests with Agent 1's decoder

**Deliverable**: Full pipeline working, Agent 5 can start training

---

## Questions for Other Agents

### For Agent 1:
1. Do you prefer pianoroll output or separate (pitch, onset, duration, velocity) streams?
2. What hidden dimension size are you using in decoder? (affects interface design)
3. Should I implement event-based representation in addition to pianoroll?

### For Agent 5:
1. What loss functions do you need? (MSE? Cross-entropy? Custom MIDI distance?)
2. Do you need batch-wise or sample-wise metrics?
3. What metrics should be logged during training?

---

**Status**: DRAFT - Open for feedback
**Next**: Incorporate feedback, finalize in Phase 2 design
**Contact**: Agent 2 - Differentiable MIDI & Utilities Support
