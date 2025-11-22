# Agent 1: MIDI Decoder Architecture Design

**Date:** November 22, 2025
**Agent:** Agent 1 - MIDI Decoder Architecture Lead
**Phase:** Phase 2 - Architecture Design (Days 4-7)
**Status:** COMPLETE ✅

---

## Executive Summary

This document presents the architectural design for the Features-to-MIDI synthesizer, the critical missing component identified in Phase 1. After careful analysis, I recommend a **hybrid approach** that combines rule-based and neural methods.

### Design Decision: Hybrid Architecture

**Recommendation:** Implement both rule-based and neural approaches in stages.

**Stage 1 (MVP - Week 2):** Rule-based Features→MIDI
**Stage 2 (Optimal - Weeks 3-4):** Neural MIDI Generator
**Stage 3 (Future):** Hybrid ensemble combining both

---

## Architecture Overview

### Complete Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                    End-to-End MIDI Reconstruction                     │
└──────────────────────────────────────────────────────────────────────┘

MIDI File (Input)
    │
    ▼
┌─────────────────────────────┐
│  DeepFeatureExtractor       │  EXISTING ✅
│  MIDI → Features (1150D)    │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  ModularEncoders            │  EXISTING ✅
│  Features → DNA (120D)      │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  SemanticDecoder            │  EXISTING ✅
│  DNA → Recon Features       │
│       (1150D)               │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  FeaturesToMIDI             │  NEW - AGENT 1 ⭐
│  Features → MIDI            │
│                             │
│  ┌───────────────────────┐ │
│  │ Stage 1: Rule-Based   │ │
│  │ Features→Parameters   │ │
│  │ Parameters→MIDI       │ │
│  └───────────────────────┘ │
│                             │
│  ┌───────────────────────┐ │
│  │ Stage 2: Neural       │ │
│  │ NeuralMIDIGenerator   │ │
│  │ Features→Events       │ │
│  └───────────────────────┘ │
└──────────┬──────────────────┘
           │
           ▼
MIDI File (Reconstructed)
```

---

## Component 1: FeaturesToMIDI Base Interface

### Purpose
Abstract base class defining the interface for converting features to MIDI.

### API Specification

```python
from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np
import torch
import mido

class FeaturesToMIDI(ABC):
    """
    Abstract base class for Features → MIDI conversion.

    All implementations must provide:
    1. features_to_midi() - Main conversion method
    2. features_to_parameters() - Feature interpretation
    3. validate_output() - MIDI validation
    """

    @abstractmethod
    def features_to_midi(
        self,
        features: Union[np.ndarray, torch.Tensor],
        output_path: Optional[str] = None,
        **kwargs
    ) -> mido.MidiFile:
        """
        Convert feature vector to MIDI file.

        Args:
            features: Feature vector [1150] or batch [B, 1150]
            output_path: Optional path to save MIDI
            **kwargs: Implementation-specific parameters

        Returns:
            mido.MidiFile object
        """
        pass

    @abstractmethod
    def features_to_parameters(
        self,
        features: Union[np.ndarray, torch.Tensor]
    ) -> dict:
        """
        Extract musical parameters from features.

        Args:
            features: Feature vector [1150]

        Returns:
            Dictionary of musical parameters
        """
        pass

    def validate_output(self, midi: mido.MidiFile) -> bool:
        """Validate that generated MIDI is valid."""
        try:
            # Check has tracks
            if len(midi.tracks) == 0:
                return False

            # Check has notes
            has_notes = False
            for track in midi.tracks:
                for msg in track:
                    if msg.type in ('note_on', 'note_off'):
                        has_notes = True
                        break

            return has_notes
        except Exception:
            return False
```

---

## Component 2: RuleBasedFeaturesToMIDI (Stage 1 - MVP)

### Purpose
Quick-win implementation using feature interpretation and existing generators.

### Architecture

```
Features (1150D)
    │
    ├─ Harmony Features (250D) ──► chord_progression, key, mode
    ├─ Rhythm Features (250D)  ──► tempo, time_sig, groove_pattern
    ├─ Melody Features (200D)  ──► melodic_contour, range, intervals
    ├─ Dynamics Features (150D)──► velocity_profile, accents
    ├─ Texture Features (100D) ──► density, voice_count
    ├─ Structure Features (50D)──► form, sections, bars
    └─ Orch Features (150D)    ──► instruments, programs
    │
    ▼
Parameters Dict
    │
    ▼
HarmonyModuleAPI / ParameterMIDIGenerator
    │
    ▼
MIDI File
```

### Implementation Strategy

```python
class RuleBasedFeaturesToMIDI(FeaturesToMIDI):
    """
    Rule-based Features → MIDI converter.

    Strategy:
    1. Extract musical parameters from features
    2. Map to HarmonyModuleAPI parameters
    3. Use existing generators to create MIDI

    Advantages:
    - Fast to implement (1-2 days)
    - Uses proven generation code
    - Musically valid by design

    Limitations:
    - Not differentiable
    - Lossy feature interpretation
    - Hard-coded rules
    """

    def __init__(self):
        self.harmony_api = HarmonyModuleAPI()
        self.param_generator = ParameterMIDIGenerator()

    def features_to_parameters(self, features: np.ndarray) -> dict:
        """Extract parameters from 1150D features."""
        # Split features by dimension
        harmony_feat = features[0:250]
        rhythm_feat = features[250:500]
        melody_feat = features[500:700]
        dynamics_feat = features[700:850]
        texture_feat = features[850:950]
        structure_feat = features[950:1000]
        orch_feat = features[1000:1150]

        params = {}

        # HARMONY PARAMETERS (from 250D harmony features)
        params['key'] = self._extract_key(harmony_feat)
        params['mode'] = self._extract_mode(harmony_feat)
        params['chord_progression'] = self._extract_chords(harmony_feat)
        params['harmonic_complexity'] = np.mean(harmony_feat[:20])
        params['voice_leading_smoothness'] = np.mean(harmony_feat[20:40])

        # RHYTHM PARAMETERS (from 250D rhythm features)
        params['tempo_bpm'] = self._extract_tempo(rhythm_feat)
        params['time_signature'] = self._extract_time_sig(rhythm_feat)
        params['syncopation'] = np.mean(rhythm_feat[0:20])
        params['groove_tightness'] = np.mean(rhythm_feat[20:40])

        # MELODY PARAMETERS (from 200D melody features)
        params['melodic_range'] = self._extract_range(melody_feat)
        params['melodic_contour'] = self._extract_contour(melody_feat)
        params['step_leap_ratio'] = np.mean(melody_feat[0:20])

        # DYNAMICS PARAMETERS (from 150D dynamics features)
        params['velocity_mean'] = np.mean(dynamics_feat[0:20]) * 127
        params['velocity_std'] = np.std(dynamics_feat[0:20]) * 64
        params['accent_frequency'] = np.mean(dynamics_feat[20:40])

        # TEXTURE PARAMETERS (from 100D texture features)
        params['voice_count'] = self._extract_voice_count(texture_feat)
        params['density'] = np.mean(texture_feat[0:20])

        # STRUCTURE PARAMETERS (from 50D structure features)
        params['num_bars'] = self._extract_num_bars(structure_feat)
        params['form'] = self._extract_form(structure_feat)

        # ORCHESTRATION PARAMETERS (from 150D orch features)
        params['instruments'] = self._extract_instruments(orch_feat)
        params['programs'] = self._extract_programs(orch_feat)

        return params

    def features_to_midi(
        self,
        features: np.ndarray,
        output_path: Optional[str] = None,
        **kwargs
    ) -> mido.MidiFile:
        """Convert features to MIDI using rule-based approach."""
        # Extract parameters
        params = self.features_to_parameters(features)

        # Generate MIDI using existing generator
        midi = self.param_generator.generate(params, output_path)

        # Validate
        if not self.validate_output(midi):
            raise ValueError("Generated MIDI is invalid")

        return midi
```

### Feature Extraction Rules

**Key Extraction:**
```python
def _extract_key(self, harmony_feat: np.ndarray) -> str:
    """Extract musical key from harmony features."""
    # Use pitch class distribution (first 12 features)
    pitch_class_dist = harmony_feat[0:12]

    # Find most prominent pitch class
    tonic_idx = np.argmax(pitch_class_dist)

    # Map to note name
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    return note_names[tonic_idx]
```

**Tempo Extraction:**
```python
def _extract_tempo(self, rhythm_feat: np.ndarray) -> float:
    """Extract tempo from rhythm features."""
    # Tempo likely encoded in first few rhythm features
    # Map feature value to BPM range [60, 200]
    tempo_feature = np.mean(rhythm_feat[0:5])
    tempo_bpm = 60 + (tempo_feature * 140)
    return np.clip(tempo_bpm, 60, 200)
```

---

## Component 3: NeuralMIDIGenerator (Stage 2 - Optimal)

### Purpose
Differentiable neural network for end-to-end Features → MIDI generation.

### Architecture Decision: Transformer-based Event Generator

**Choice:** Autoregressive Transformer generating MIDI events sequentially.

**Rationale:**
- Transformers excel at sequence generation
- Attention mechanism captures musical dependencies
- Proven in Music Transformer, MusicVAE
- Differentiable for end-to-end training

### Architecture Diagram

```
Features (1150D)
    │
    ▼
┌─────────────────────────────┐
│ Feature Encoder             │
│ FC(1150 → 512)              │
│ + Positional Encoding       │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Transformer Decoder         │
│ - 6 layers                  │
│ - 8 attention heads         │
│ - 512 hidden dim            │
│ - Autoregressive            │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Event Prediction Heads      │
│ - Pitch: FC(512 → 128)      │
│ - Duration: FC(512 → 32)    │
│ - Velocity: FC(512 → 128)   │
│ - Track: FC(512 → 16)       │
│ - IsEnd: FC(512 → 2)        │
└──────────┬──────────────────┘
           │
           ▼
MIDI Events [(pitch, onset, duration, velocity, track), ...]
    │
    ▼
┌─────────────────────────────┐
│ Event→MIDI Converter        │
│ Assemble mido.MidiFile      │
└──────────┬──────────────────┘
           │
           ▼
MIDI File
```

### Implementation Specification

```python
import torch
import torch.nn as nn
from typing import List, Tuple

class NeuralMIDIGenerator(nn.Module, FeaturesToMIDI):
    """
    Neural MIDI generator using Transformer architecture.

    Architecture:
    - Feature encoder: 1150D → 512D
    - Transformer decoder: Autoregressive event generation
    - Event heads: Predict (pitch, duration, velocity, track)

    Training:
    - Teacher forcing during training
    - Autoregressive sampling during inference
    - Gumbel-softmax for discrete events
    """

    def __init__(
        self,
        feature_dim: int = 1150,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_events: int = 512,
        max_tracks: int = 16
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_events = max_events
        self.max_tracks = max_tracks

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_events)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Event prediction heads
        self.pitch_head = nn.Linear(hidden_dim, 128)      # MIDI pitch 0-127
        self.duration_head = nn.Linear(hidden_dim, 32)    # Duration bins
        self.velocity_head = nn.Linear(hidden_dim, 128)   # Velocity 0-127
        self.track_head = nn.Linear(hidden_dim, max_tracks)  # Track assignment
        self.is_end_head = nn.Linear(hidden_dim, 2)       # End-of-sequence token

        # Event embedding (for autoregressive generation)
        self.event_embedding = nn.Embedding(128 + 32 + 128 + max_tracks, hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        target_events: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass: Features → MIDI events

        Args:
            features: [B, 1150] feature vectors
            target_events: [B, L, 4] target events (training only)
            max_length: Maximum sequence length (inference only)

        Returns:
            Tuple of (pitch_logits, duration_logits, velocity_logits,
                     track_logits, is_end_logits)
        """
        batch_size = features.shape[0]

        # Encode features → memory
        memory = self.feature_encoder(features)  # [B, hidden_dim]
        memory = memory.unsqueeze(1)  # [B, 1, hidden_dim]

        if target_events is not None:
            # Training mode: Teacher forcing
            return self._forward_training(memory, target_events)
        else:
            # Inference mode: Autoregressive generation
            return self._forward_inference(memory, max_length or self.max_events)

    def _forward_training(
        self,
        memory: torch.Tensor,
        target_events: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Training forward pass with teacher forcing."""
        batch_size, seq_len, _ = target_events.shape

        # Embed target events
        event_emb = self.event_embedding(target_events.sum(dim=-1))  # Simplified
        event_emb = self.pos_encoding(event_emb)

        # Transformer decode
        output = self.transformer(event_emb, memory)  # [B, L, hidden_dim]

        # Predict next events
        pitch_logits = self.pitch_head(output)
        duration_logits = self.duration_head(output)
        velocity_logits = self.velocity_head(output)
        track_logits = self.track_head(output)
        is_end_logits = self.is_end_head(output)

        return (pitch_logits, duration_logits, velocity_logits,
                track_logits, is_end_logits)

    def _forward_inference(
        self,
        memory: torch.Tensor,
        max_length: int
    ) -> List[dict]:
        """Autoregressive inference."""
        batch_size = memory.shape[0]
        device = memory.device

        # Initialize with start token
        events = []
        current_event = torch.zeros(batch_size, 1, self.hidden_dim).to(device)

        for step in range(max_length):
            # Add positional encoding
            current_event_pos = self.pos_encoding(current_event)

            # Decode
            output = self.transformer(current_event_pos, memory)
            output = output[:, -1, :]  # Take last timestep

            # Predict event components
            pitch = self.pitch_head(output).argmax(dim=-1)
            duration = self.duration_head(output).argmax(dim=-1)
            velocity = self.velocity_head(output).argmax(dim=-1)
            track = self.track_head(output).argmax(dim=-1)
            is_end = self.is_end_head(output).argmax(dim=-1)

            # Store event
            events.append({
                'pitch': pitch.cpu().numpy(),
                'duration': duration.cpu().numpy(),
                'velocity': velocity.cpu().numpy(),
                'track': track.cpu().numpy()
            })

            # Check for end token
            if is_end.item() == 1:
                break

            # Embed current prediction for next step
            # (Simplified - should embed all components)
            current_event = self.event_embedding(pitch.unsqueeze(1))

        return events

    def features_to_midi(
        self,
        features: Union[np.ndarray, torch.Tensor],
        output_path: Optional[str] = None,
        **kwargs
    ) -> mido.MidiFile:
        """Convert features to MIDI using neural generator."""
        # Convert to tensor
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()

        # Ensure batch dimension
        if features.dim() == 1:
            features = features.unsqueeze(0)

        # Generate events
        with torch.no_grad():
            events = self._forward_inference(
                self.feature_encoder(features).unsqueeze(1),
                max_length=kwargs.get('max_length', 512)
            )

        # Convert events to MIDI
        midi = self._events_to_midi(events[0])  # Take first in batch

        # Save if path provided
        if output_path:
            midi.save(output_path)

        return midi

    def _events_to_midi(self, events: List[dict]) -> mido.MidiFile:
        """Convert event list to mido.MidiFile."""
        midi = mido.MidiFile(type=1, ticks_per_beat=480)

        # Create tracks
        tracks = {i: mido.MidiTrack() for i in range(self.max_tracks)}

        # Add tempo
        tracks[0].append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

        # Convert events to MIDI messages
        current_time = {i: 0 for i in range(self.max_tracks)}

        for event in events:
            track_idx = event['track']
            track = tracks[track_idx]

            pitch = event['pitch']
            duration = event['duration'] * 10  # Convert to ticks
            velocity = event['velocity']

            # Note on
            track.append(mido.Message(
                'note_on',
                note=pitch,
                velocity=velocity,
                time=0
            ))

            # Note off
            track.append(mido.Message(
                'note_off',
                note=pitch,
                velocity=0,
                time=duration
            ))

        # Add tracks to MIDI
        for track in tracks.values():
            if len(track) > 0:
                track.append(mido.MetaMessage('end_of_track', time=0))
                midi.tracks.append(track)

        return midi


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]
```

---

## Component 4: Differentiable MIDI Representations

### Purpose
Enable gradient flow through discrete MIDI generation for end-to-end training.

### Soft Pianoroll Representation

```python
class SoftPianoroll:
    """
    Differentiable piano roll representation.

    Instead of binary (0/1), uses continuous values (0.0-1.0)
    for note presence. Enables gradient-based optimization.
    """

    def __init__(
        self,
        num_pitches: int = 128,
        num_timesteps: int = 256,
        resolution: int = 4  # Timesteps per beat
    ):
        self.num_pitches = num_pitches
        self.num_timesteps = num_timesteps
        self.resolution = resolution

    def create_empty(self, batch_size: int = 1) -> torch.Tensor:
        """Create empty soft pianoroll."""
        return torch.zeros(batch_size, num_timesteps, num_pitches)

    def from_midi(self, midi: mido.MidiFile) -> torch.Tensor:
        """Convert MIDI to soft pianoroll (for training)."""
        # Convert to hard pianoroll first
        hard_roll = self._midi_to_hard_pianoroll(midi)

        # Add small noise for soft version
        soft_roll = hard_roll.float()
        soft_roll = soft_roll + torch.randn_like(soft_roll) * 0.1
        soft_roll = torch.clamp(soft_roll, 0.0, 1.0)

        return soft_roll

    def to_midi(
        self,
        soft_roll: torch.Tensor,
        threshold: float = 0.5,
        tempo: int = 120
    ) -> mido.MidiFile:
        """Convert soft pianoroll to MIDI (sampling)."""
        # Threshold to hard pianoroll
        hard_roll = (soft_roll > threshold).float()

        # Convert to MIDI
        return self._hard_pianoroll_to_midi(hard_roll, tempo)

    def gumbel_softmax_sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False
    ) -> torch.Tensor:
        """
        Sample from logits using Gumbel-softmax trick.

        This allows differentiable sampling from discrete distributions.
        """
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y = logits + gumbel_noise

        # Softmax with temperature
        y_soft = torch.softmax(y / temperature, dim=-1)

        if hard:
            # Straight-through estimator
            y_hard = torch.zeros_like(y_soft)
            y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)

            # Gradient flows through soft, forward uses hard
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft

        return y
```

---

## Training Strategy

### Stage 1: Rule-Based (No Training Needed)
- Implement feature extraction rules
- Test on reconstructed features from decoder
- Measure MIDI reconstruction quality
- **Timeline:** 2-3 days

### Stage 2: Neural Generator Training

**Dataset:**
- Use existing MIDI corpus
- Extract features with DeepFeatureExtractor
- Create (features, MIDI) pairs

**Loss Function:**
```python
def compute_loss(
    predicted_events: Tuple,
    target_events: torch.Tensor,
    weights: dict = None
) -> torch.Tensor:
    """
    Multi-component loss for MIDI generation.

    Components:
    - Pitch cross-entropy
    - Duration cross-entropy
    - Velocity MSE
    - Track cross-entropy
    - End-of-sequence cross-entropy
    """
    pitch_logits, dur_logits, vel_logits, track_logits, end_logits = predicted_events

    # Pitch loss
    pitch_loss = F.cross_entropy(
        pitch_logits.reshape(-1, 128),
        target_events[:, :, 0].reshape(-1)
    )

    # Duration loss
    dur_loss = F.cross_entropy(
        dur_logits.reshape(-1, 32),
        target_events[:, :, 1].reshape(-1)
    )

    # Velocity loss (MSE for continuous)
    vel_loss = F.mse_loss(
        vel_logits.reshape(-1, 128),
        F.one_hot(target_events[:, :, 2], 128).float().reshape(-1, 128)
    )

    # Track loss
    track_loss = F.cross_entropy(
        track_logits.reshape(-1, 16),
        target_events[:, :, 3].reshape(-1)
    )

    # Combine
    total_loss = (
        pitch_loss * weights.get('pitch', 1.0) +
        dur_loss * weights.get('duration', 1.0) +
        vel_loss * weights.get('velocity', 0.5) +
        track_loss * weights.get('track', 0.5)
    )

    return total_loss
```

**Training Procedure:**
1. Initialize neural generator
2. Load MIDI dataset
3. Extract features for each MIDI
4. Train with teacher forcing
5. Validate on held-out set
6. Fine-tune with reconstruction loss

**Timeline:** 1-2 weeks of training

---

## Integration Points

### With Agent 2: Differentiable MIDI Utilities
**Required:**
- `SoftPianoroll` class
- `GumbelSoftmaxSampler`
- MIDI assembly utilities

**Interface:**
```python
# Agent 2 provides these utilities
from midi_generator.utils.differentiable_midi import (
    SoftPianoroll,
    gumbel_softmax_sample,
    assemble_midi_from_events
)
```

### With Agent 3: DNA Expansion
**Impact:**
- Decoder input will change from 120D → 300D
- Need to update all dimensions
- Maintain backward compatibility

**Action Items:**
- Monitor Agent 3's progress
- Update decoder when new DNA structure ready
- Test with expanded dimensions

### With Agent 5: Training Pipeline
**Provides:**
- MIDI reconstruction loss
- MIDI distance metrics
- Validation callbacks

**Receives:**
- Training loop integration
- Checkpoint management
- Data pipeline

---

## Testing Strategy

### Unit Tests
```python
def test_rule_based_converter():
    """Test rule-based Features→MIDI."""
    converter = RuleBasedFeaturesToMIDI()

    # Create dummy features
    features = np.random.randn(1150)

    # Convert to MIDI
    midi = converter.features_to_midi(features)

    # Validate
    assert converter.validate_output(midi)
    assert len(midi.tracks) > 0

def test_neural_generator():
    """Test neural MIDI generator."""
    generator = NeuralMIDIGenerator()

    # Create dummy features
    features = torch.randn(1, 1150)

    # Generate
    events = generator._forward_inference(
        generator.feature_encoder(features).unsqueeze(1),
        max_length=10
    )

    # Validate
    assert len(events) > 0
    assert 'pitch' in events[0]
```

### Integration Tests
```python
def test_end_to_end_reconstruction():
    """Test full MIDI → Features → DNA → Features → MIDI."""
    # Load original MIDI
    original_midi = mido.MidiFile('test.mid')

    # Extract features
    extractor = DeepFeatureExtractor()
    features = extractor.extract(original_midi)

    # Encode to DNA
    encoder = ModularEncoders()
    dna = encoder.encode(features)

    # Decode back to features
    decoder = SemanticDecoder()
    recon_features = decoder(dna)

    # Convert to MIDI
    converter = RuleBasedFeaturesToMIDI()
    recon_midi = converter.features_to_midi(recon_features)

    # Measure similarity
    similarity = compute_midi_similarity(original_midi, recon_midi)

    # Should be reasonably similar
    assert similarity > 0.3
```

---

## Success Metrics

### Stage 1 (Rule-Based) Success Criteria
- ✅ Generates valid MIDI from any 1150D feature vector
- ✅ No crashes or format errors
- ✅ Reconstruction similarity > 20%
- ✅ End-to-end pipeline works

### Stage 2 (Neural) Success Criteria
- ✅ Differentiable (gradients flow)
- ✅ Reconstruction similarity > 50%
- ✅ Musically coherent output
- ✅ Handles variable-length sequences

---

## Timeline

**Week 2 (Stage 1 - Rule-Based):**
- Days 1-2: Implement RuleBasedFeaturesToMIDI
- Days 3-4: Integration and testing
- Days 5: Validation and metrics

**Weeks 3-4 (Stage 2 - Neural):**
- Week 3: Build neural architecture, initial training
- Week 4: Fine-tuning, optimization, documentation

---

## Deliverables

**Phase 2 (This Document):**
- ✅ Architectural design
- ✅ Component specifications
- ✅ Interface definitions
- ✅ Training strategy

**Phase 3 (Implementation):**
- `src/models/features_to_midi.py` - Base interface
- `src/models/rule_based_midi.py` - Rule-based converter
- `src/models/neural_midi_generator.py` - Neural generator
- `src/utils/soft_pianoroll.py` - Differentiable representations
- `tests/test_midi_decoder.py` - Comprehensive tests
- `examples/generate_from_dna.py` - Usage examples

---

## Conclusion

**Architecture Decision:** Hybrid approach (rule-based MVP + neural optimal)

**Key Innovation:** Separating concerns - rule-based for quick validation, neural for optimal quality

**Critical Path:** Rule-based implementation in Week 2 unblocks the entire pipeline

**Risk Mitigation:** Having two approaches provides fallback if neural training is challenging

**Next:** Begin Phase 3 implementation starting with rule-based converter.

---

**Phase 2 Status:** ✅ COMPLETE
**Ready for Phase 3:** YES
**Next Milestone:** Implement RuleBasedFeaturesToMIDI
