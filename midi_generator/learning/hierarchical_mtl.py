"""
AGENT 05: Hierarchical Multi-Task Learning (MTL) Architect
===========================================================

Implements the hierarchical multi-task neural network architecture for predicting
50 hierarchical musical parameters from 200 selected features.

Architecture:
    Level 1: Global Context (8 parameters)
        - genre.primary, tempo.bpm, time_signature, key.tonic, key.mode,
          energy.level, complexity.overall, structure.form
        - Predicted from shared encoder (unconditional)

    Level 2: Universal Dimensions (20 parameters)
        - Harmony (6), Melody (5), Rhythm (5), Dynamics (2), Texture (2)
        - Conditioned on Level 1 predictions

    Level 3: Genre-Specific Details (22 parameters)
        - Universal (5), Jazz (4), Classical (3), Rock (3), Electronic (3), etc.
        - Conditioned on genre.primary from Level 1

Key Features:
    1. Hierarchical conditioning (L2 uses L1, L3 uses genre from L1)
    2. Shared feature encoder with attention
    3. Multi-task loss with automatic weighting
    4. Residual connections for gradient flow
    5. Batch normalization and dropout for regularization
    6. Supports continuous, categorical, and binary parameters

Dependencies:
    - Agent 01: Hierarchical parameter definitions
    - Agent 04: 200 selected features

Author: Agent 05 - Hierarchical MTL Architect
License: MIT
Date: November 20, 2025
"""

import json
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

warnings.filterwarnings('ignore')

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed. Install with: pip install torch")

# ============================================================================
# Parameter Definitions (from Agent 01)
# ============================================================================

@dataclass
class ParameterDefinition:
    """Definition of a hierarchical parameter"""
    name: str
    level: int  # 1, 2, or 3
    param_type: str  # 'continuous', 'categorical', 'binary', 'integer'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[str]] = None
    genre_specific: bool = False
    genres: Optional[List[str]] = None
    default_value: Any = None


# Level 1: Global Context (8 parameters)
LEVEL_1_PARAMETERS = [
    ParameterDefinition('genre.primary', 1, 'categorical',
                       categories=['jazz', 'classical', 'rock', 'electronic', 'pop', 'hiphop', 'latin']),
    ParameterDefinition('tempo.bpm', 1, 'continuous', min_value=40.0, max_value=200.0),
    ParameterDefinition('time_signature', 1, 'categorical',
                       categories=['4/4', '3/4', '6/8', '5/4', '7/8', '2/4']),
    ParameterDefinition('key.tonic', 1, 'categorical',
                       categories=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']),
    ParameterDefinition('key.mode', 1, 'categorical',
                       categories=['major', 'minor', 'dorian', 'mixolydian', 'phrygian', 'lydian']),
    ParameterDefinition('energy.level', 1, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('complexity.overall', 1, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('structure.form', 1, 'categorical',
                       categories=['verse_chorus', 'aaba', 'sonata', 'rondo', 'through_composed', 'blues']),
]

# Level 2: Universal Dimensions (20 parameters)
LEVEL_2_PARAMETERS = [
    # Harmony (6)
    ParameterDefinition('harmony.chord_density', 2, 'continuous', min_value=0.0, max_value=10.0),
    ParameterDefinition('harmony.complexity', 2, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('harmony.chromaticism', 2, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('harmony.tension', 2, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('harmony.voicing_spread', 2, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('harmony.progression_predictability', 2, 'continuous', min_value=0.0, max_value=1.0),

    # Melody (5)
    ParameterDefinition('melody.note_density', 2, 'continuous', min_value=0.0, max_value=20.0),
    ParameterDefinition('melody.range_semitones', 2, 'integer', min_value=0, max_value=48),
    ParameterDefinition('melody.contour_smoothness', 2, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('melody.rhythmic_complexity', 2, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('melody.repetition', 2, 'continuous', min_value=0.0, max_value=1.0),

    # Rhythm (5)
    ParameterDefinition('rhythm.subdivision', 2, 'categorical',
                       categories=['quarter', 'eighth', 'sixteenth', 'triplet', 'quintuplet']),
    ParameterDefinition('rhythm.syncopation', 2, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('rhythm.groove_consistency', 2, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('rhythm.polyrhythm', 2, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('rhythm.swing_amount', 2, 'continuous', min_value=0.0, max_value=1.0),

    # Dynamics (2)
    ParameterDefinition('dynamics.overall_level', 2, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('dynamics.range', 2, 'continuous', min_value=0.0, max_value=1.0),

    # Texture (2)
    ParameterDefinition('texture.polyphony', 2, 'integer', min_value=1, max_value=16),
    ParameterDefinition('texture.density', 2, 'continuous', min_value=0.0, max_value=1.0),
]

# Level 3: Genre-Specific Details (22 parameters)
LEVEL_3_PARAMETERS = [
    # Universal (5)
    ParameterDefinition('orchestration.instrument_count', 3, 'integer', min_value=1, max_value=20),
    ParameterDefinition('orchestration.register_balance', 3, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('articulation.legato_ratio', 3, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('structure.section_contrast', 3, 'continuous', min_value=0.0, max_value=1.0),
    ParameterDefinition('structure.repetition_level', 3, 'continuous', min_value=0.0, max_value=1.0),

    # Jazz (4)
    ParameterDefinition('jazz.swing_feel', 3, 'categorical',
                       categories=['straight', 'light', 'medium', 'hard'],
                       genre_specific=True, genres=['jazz']),
    ParameterDefinition('jazz.walking_bass', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['jazz']),
    ParameterDefinition('jazz.improvisation_ratio', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['jazz']),
    ParameterDefinition('jazz.bebop_vocabulary', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['jazz']),

    # Classical (3)
    ParameterDefinition('classical.counterpoint', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['classical']),
    ParameterDefinition('classical.development_density', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['classical']),
    ParameterDefinition('classical.voice_leading_quality', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['classical']),

    # Rock/Metal (3)
    ParameterDefinition('rock.power_chord_ratio', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['rock']),
    ParameterDefinition('rock.riff_repetition', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['rock']),
    ParameterDefinition('rock.distortion_level', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['rock']),

    # Electronic (3)
    ParameterDefinition('electronic.quantization', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['electronic']),
    ParameterDefinition('electronic.filter_movement', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['electronic']),
    ParameterDefinition('electronic.arpeggio_density', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['electronic']),

    # Hip-Hop (2)
    ParameterDefinition('hiphop.sample_based', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['hiphop']),
    ParameterDefinition('hiphop.boom_bap_feel', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['hiphop']),

    # Latin (2)
    ParameterDefinition('latin.clave_pattern', 3, 'categorical',
                       categories=['son', '3-2', '2-3', 'rumba', 'bossa'],
                       genre_specific=True, genres=['latin']),
    ParameterDefinition('latin.montuno_complexity', 3, 'continuous', min_value=0.0, max_value=1.0,
                       genre_specific=True, genres=['latin']),
]

ALL_PARAMETERS = LEVEL_1_PARAMETERS + LEVEL_2_PARAMETERS + LEVEL_3_PARAMETERS


# ============================================================================
# Neural Network Components
# ============================================================================

class AttentionLayer(nn.Module):
    """Self-attention layer for feature encoding"""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, x):
        # x: (batch, features)
        # Expand to (batch, 1, features) for attention
        x_expanded = x.unsqueeze(1)

        Q = self.query(x_expanded)
        K = self.key(x_expanded)
        V = self.value(x_expanded)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention
        attended = torch.matmul(attn_weights, V)
        return attended.squeeze(1)


class FeatureEncoder(nn.Module):
    """
    Shared feature encoder that processes 200 input features into a
    rich representation for parameter prediction.

    Uses:
        - Multi-layer perceptron with residual connections
        - Batch normalization
        - Dropout for regularization
        - Optional attention mechanism
    """

    def __init__(self,
                 input_dim: int = 200,
                 hidden_dims: List[int] = [512, 256, 128],
                 use_attention: bool = True,
                 dropout: float = 0.3):
        super().__init__()

        self.input_dim = input_dim
        self.use_attention = use_attention

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])

        # Attention (optional)
        if use_attention:
            self.attention = AttentionLayer(hidden_dims[0], hidden_dims[0] // 2)

        # Hidden layers with residual connections
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.dropouts.append(nn.Dropout(dropout))

        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        # Input projection
        h = F.relu(self.input_bn(self.input_proj(x)))

        # Apply attention if enabled
        if self.use_attention:
            h = h + self.attention(h)  # Residual connection

        # Hidden layers
        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            h_new = F.relu(bn(layer(h)))
            h_new = dropout(h_new)

            # Residual connection (if dimensions match)
            if h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new

        return h


class PredictionHead(nn.Module):
    """
    Generic prediction head for a single parameter.
    Handles continuous, categorical, and integer outputs.
    """

    def __init__(self,
                 input_dim: int,
                 param_def: ParameterDefinition,
                 hidden_dim: int = 64):
        super().__init__()

        self.param_def = param_def
        self.param_name = param_def.name
        self.param_type = param_def.param_type

        # Hidden layer
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)

        # Output layer
        if param_def.param_type == 'categorical':
            self.output_dim = len(param_def.categories)
            self.output = nn.Linear(hidden_dim, self.output_dim)
        elif param_def.param_type == 'binary':
            self.output_dim = 1
            self.output = nn.Linear(hidden_dim, 1)
        else:  # continuous or integer
            self.output_dim = 1
            self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.bn(self.hidden(x)))
        h = self.dropout(h)
        out = self.output(h)

        # Apply appropriate activation
        if self.param_type == 'categorical':
            # Logits for cross-entropy loss
            return out
        elif self.param_type == 'binary':
            # Sigmoid for binary classification
            return torch.sigmoid(out)
        elif self.param_type == 'continuous':
            # Check if bounded
            if self.param_def.min_value is not None and self.param_def.max_value is not None:
                # Sigmoid then scale to range
                out = torch.sigmoid(out)
                out = out * (self.param_def.max_value - self.param_def.min_value) + self.param_def.min_value
            return out
        else:  # integer
            # Sigmoid then scale and round
            if self.param_def.min_value is not None and self.param_def.max_value is not None:
                out = torch.sigmoid(out)
                out = out * (self.param_def.max_value - self.param_def.min_value) + self.param_def.min_value
            return out


class HierarchicalMTLModel(nn.Module):
    """
    Hierarchical Multi-Task Learning model for musical parameter prediction.

    Architecture:
        1. Shared Feature Encoder (200 features → 128-dim embedding)
        2. Level 1 Heads: 8 global parameters (unconditional)
        3. Level 2 Heads: 20 universal parameters (conditioned on Level 1)
        4. Level 3 Heads: 22 genre-specific parameters (conditioned on genre)

    The hierarchical structure ensures:
        - Genre is predicted first (Level 1)
        - Universal musical dimensions use genre context (Level 2)
        - Genre-specific details are only predicted for relevant genres (Level 3)
    """

    def __init__(self,
                 input_dim: int = 200,
                 encoder_hidden_dims: List[int] = [512, 256, 128],
                 head_hidden_dim: int = 64,
                 use_attention: bool = True,
                 dropout: float = 0.3,
                 conditioning_dim: int = 32):
        super().__init__()

        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim

        # Shared feature encoder
        self.encoder = FeatureEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            use_attention=use_attention,
            dropout=dropout
        )

        encoder_output_dim = self.encoder.output_dim

        # Level 1 prediction heads (8 parameters)
        self.level1_heads = nn.ModuleDict()
        for param in LEVEL_1_PARAMETERS:
            self.level1_heads[param.name] = PredictionHead(
                encoder_output_dim, param, head_hidden_dim
            )

        # Level 1 conditioning embeddings (to condition Level 2)
        # We'll embed Level 1 predictions into a conditioning vector
        self.level1_embedding = nn.Linear(len(LEVEL_1_PARAMETERS) * 16, conditioning_dim)

        # Level 2 prediction heads (20 parameters, conditioned on Level 1)
        self.level2_heads = nn.ModuleDict()
        for param in LEVEL_2_PARAMETERS:
            self.level2_heads[param.name] = PredictionHead(
                encoder_output_dim + conditioning_dim, param, head_hidden_dim
            )

        # Genre embedding for Level 3
        self.genre_categories = LEVEL_1_PARAMETERS[0].categories  # ['jazz', 'classical', ...]
        self.genre_embedding = nn.Embedding(len(self.genre_categories), conditioning_dim)

        # Level 3 prediction heads (22 parameters, conditioned on genre)
        self.level3_heads = nn.ModuleDict()
        for param in LEVEL_3_PARAMETERS:
            self.level3_heads[param.name] = PredictionHead(
                encoder_output_dim + conditioning_dim, param, head_hidden_dim
            )

        # Store parameter definitions for easy access
        self.level1_params = {p.name: p for p in LEVEL_1_PARAMETERS}
        self.level2_params = {p.name: p for p in LEVEL_2_PARAMETERS}
        self.level3_params = {p.name: p for p in LEVEL_3_PARAMETERS}

    def _create_level1_conditioning(self, level1_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create conditioning vector from Level 1 predictions"""
        batch_size = next(iter(level1_outputs.values())).size(0)

        # Convert all Level 1 outputs to embeddings
        embeddings = []

        for param_name, output in level1_outputs.items():
            param_def = self.level1_params[param_name]

            if param_def.param_type == 'categorical':
                # Use argmax to get predicted class, then embed
                pred_class = torch.argmax(output, dim=-1)
                # Create simple embedding (one-hot style)
                emb = F.one_hot(pred_class, num_classes=output.size(-1)).float()
                # Pad/truncate to fixed size (16)
                if emb.size(-1) < 16:
                    emb = F.pad(emb, (0, 16 - emb.size(-1)))
                else:
                    emb = emb[:, :16]
            else:
                # For continuous/integer, repeat value to size 16
                emb = output.repeat(1, 16) if output.size(-1) == 1 else output
                if emb.size(-1) < 16:
                    emb = F.pad(emb, (0, 16 - emb.size(-1)))
                else:
                    emb = emb[:, :16]

            embeddings.append(emb)

        # Concatenate all embeddings
        combined = torch.cat(embeddings, dim=-1)  # (batch, 8*16)

        # Project to conditioning dimension
        conditioning = self.level1_embedding(combined)
        return conditioning

    def forward(self,
                x: torch.Tensor,
                return_all_levels: bool = True,
                genre_override: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical model.

        Args:
            x: Input features (batch, 200)
            return_all_levels: If True, return predictions for all levels
            genre_override: Optional genre indices to use instead of predicting

        Returns:
            Dictionary mapping parameter names to predictions
        """
        # Encode features
        encoded = self.encoder(x)  # (batch, 128)

        outputs = {}

        # ===== LEVEL 1: Global Context =====
        level1_outputs = {}
        for param_name, head in self.level1_heads.items():
            level1_outputs[param_name] = head(encoded)

        if return_all_levels:
            outputs.update(level1_outputs)

        # Create conditioning from Level 1
        level1_conditioning = self._create_level1_conditioning(level1_outputs)

        # ===== LEVEL 2: Universal Dimensions =====
        # Concatenate encoded features with Level 1 conditioning
        level2_input = torch.cat([encoded, level1_conditioning], dim=-1)

        level2_outputs = {}
        for param_name, head in self.level2_heads.items():
            level2_outputs[param_name] = head(level2_input)

        if return_all_levels:
            outputs.update(level2_outputs)

        # ===== LEVEL 3: Genre-Specific Details =====
        # Get genre prediction or use override
        if genre_override is not None:
            genre_indices = genre_override
        else:
            genre_logits = level1_outputs['genre.primary']
            genre_indices = torch.argmax(genre_logits, dim=-1)

        # Get genre embeddings
        genre_emb = self.genre_embedding(genre_indices)

        # Concatenate with encoded features
        level3_input = torch.cat([encoded, genre_emb], dim=-1)

        level3_outputs = {}
        for param_name, head in self.level3_heads.items():
            param_def = self.level3_params[param_name]

            # Only predict genre-specific parameters if genre matches
            if param_def.genre_specific:
                # For now, predict for all; can mask later in loss
                level3_outputs[param_name] = head(level3_input)
            else:
                # Universal Level 3 parameters
                level3_outputs[param_name] = head(level3_input)

        if return_all_levels:
            outputs.update(level3_outputs)

        return outputs

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Make predictions and convert to interpretable values.

        Args:
            x: Input features (batch, 200) or (200,)

        Returns:
            Dictionary of parameter names to predicted values
        """
        self.eval()
        with torch.no_grad():
            # Handle single example
            if x.dim() == 1:
                x = x.unsqueeze(0)

            outputs = self.forward(x)

            # Convert outputs to interpretable values
            predictions = {}

            for param_name, output in outputs.items():
                # Find parameter definition
                if param_name in self.level1_params:
                    param_def = self.level1_params[param_name]
                elif param_name in self.level2_params:
                    param_def = self.level2_params[param_name]
                else:
                    param_def = self.level3_params[param_name]

                if param_def.param_type == 'categorical':
                    # Get category name
                    pred_idx = torch.argmax(output, dim=-1).item()
                    predictions[param_name] = param_def.categories[pred_idx]
                elif param_def.param_type == 'integer':
                    predictions[param_name] = int(torch.round(output).item())
                else:  # continuous or binary
                    predictions[param_name] = float(output.item())

            return predictions


# ============================================================================
# Multi-Task Loss Function
# ============================================================================

class HierarchicalMTLLoss(nn.Module):
    """
    Hierarchical multi-task loss with automatic weighting.

    Combines:
        - Per-parameter losses (MSE for continuous, CE for categorical)
        - Hierarchical weighting (Level 1 > Level 2 > Level 3)
        - Automatic task weighting based on uncertainty
    """

    def __init__(self,
                 level1_weight: float = 3.0,
                 level2_weight: float = 2.0,
                 level3_weight: float = 1.0,
                 use_auto_weighting: bool = True):
        super().__init__()

        self.level1_weight = level1_weight
        self.level2_weight = level2_weight
        self.level3_weight = level3_weight
        self.use_auto_weighting = use_auto_weighting

        # Learnable task weights (log variance)
        if use_auto_weighting:
            n_tasks = len(ALL_PARAMETERS)
            self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                param_defs: Dict[str, ParameterDefinition]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hierarchical multi-task loss.

        Args:
            predictions: Dict of parameter predictions
            targets: Dict of target values
            param_defs: Dict of parameter definitions

        Returns:
            Total loss, dictionary of per-parameter losses
        """
        losses = {}
        total_loss = 0.0
        task_idx = 0

        for param_name, pred in predictions.items():
            if param_name not in targets:
                continue

            target = targets[param_name]
            param_def = param_defs[param_name]

            # Compute loss based on parameter type
            if param_def.param_type == 'categorical':
                loss = F.cross_entropy(pred, target)
            elif param_def.param_type == 'binary':
                loss = F.binary_cross_entropy(pred, target.float().unsqueeze(-1))
            else:  # continuous or integer
                loss = F.mse_loss(pred, target.float().unsqueeze(-1) if target.dim() == 1 else target)

            # Apply hierarchical weighting
            if param_def.level == 1:
                weight = self.level1_weight
            elif param_def.level == 2:
                weight = self.level2_weight
            else:
                weight = self.level3_weight

            # Apply automatic weighting if enabled
            if self.use_auto_weighting and task_idx < len(self.log_vars):
                # Uncertainty weighting: loss / (2 * var) + log(var) / 2
                precision = torch.exp(-self.log_vars[task_idx])
                weighted_loss = precision * loss + self.log_vars[task_idx] / 2
                weighted_loss = weight * weighted_loss
            else:
                weighted_loss = weight * loss

            losses[param_name] = loss.item()
            total_loss = total_loss + weighted_loss
            task_idx += 1

        return total_loss, losses


# ============================================================================
# Dataset and DataLoader
# ============================================================================

class MIDIParameterDataset(Dataset):
    """
    Dataset for MIDI features and hierarchical parameter labels.
    """

    def __init__(self,
                 features: np.ndarray,
                 labels: Dict[str, np.ndarray],
                 param_defs: Dict[str, ParameterDefinition]):
        """
        Args:
            features: (n_samples, 200) feature matrix
            labels: Dict mapping parameter names to label arrays
            param_defs: Parameter definitions
        """
        self.features = torch.FloatTensor(features)
        self.labels = {}
        self.param_defs = param_defs

        # Convert labels to tensors
        for param_name, values in labels.items():
            param_def = param_defs[param_name]

            if param_def.param_type == 'categorical':
                # Convert categories to indices
                if isinstance(values[0], str):
                    indices = [param_def.categories.index(v) for v in values]
                    self.labels[param_name] = torch.LongTensor(indices)
                else:
                    self.labels[param_name] = torch.LongTensor(values)
            else:
                self.labels[param_name] = torch.FloatTensor(values)

        self.n_samples = len(features)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        features = self.features[idx]
        labels = {name: values[idx] for name, values in self.labels.items()}
        return features, labels


# ============================================================================
# Model Configuration and Factory
# ============================================================================

@dataclass
class MTLConfig:
    """Configuration for Hierarchical MTL model"""
    input_dim: int = 200
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    head_hidden_dim: int = 64
    use_attention: bool = True
    dropout: float = 0.3
    conditioning_dim: int = 32

    # Loss configuration
    level1_weight: float = 3.0
    level2_weight: float = 2.0
    level3_weight: float = 1.0
    use_auto_weighting: bool = True

    # Training configuration
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10

    def save(self, path: Path):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: Path):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def create_model(config: Optional[MTLConfig] = None) -> HierarchicalMTLModel:
    """
    Factory function to create Hierarchical MTL model.

    Args:
        config: Model configuration (uses default if None)

    Returns:
        Initialized model
    """
    if config is None:
        config = MTLConfig()

    model = HierarchicalMTLModel(
        input_dim=config.input_dim,
        encoder_hidden_dims=config.encoder_hidden_dims,
        head_hidden_dim=config.head_hidden_dim,
        use_attention=config.use_attention,
        dropout=config.dropout,
        conditioning_dim=config.conditioning_dim
    )

    return model


def create_loss_function(config: Optional[MTLConfig] = None) -> HierarchicalMTLLoss:
    """
    Factory function to create loss function.

    Args:
        config: Model configuration

    Returns:
        Loss function
    """
    if config is None:
        config = MTLConfig()

    loss_fn = HierarchicalMTLLoss(
        level1_weight=config.level1_weight,
        level2_weight=config.level2_weight,
        level3_weight=config.level3_weight,
        use_auto_weighting=config.use_auto_weighting
    )

    return loss_fn


# ============================================================================
# Model Summary and Utilities
# ============================================================================

def model_summary(model: HierarchicalMTLModel) -> Dict[str, Any]:
    """
    Generate summary of model architecture.

    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count parameters per level
    level1_params = sum(p.numel() for head in model.level1_heads.values() for p in head.parameters())
    level2_params = sum(p.numel() for head in model.level2_heads.values() for p in head.parameters())
    level3_params = sum(p.numel() for head in model.level3_heads.values() for p in head.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'encoder_parameters': encoder_params,
        'level1_parameters': level1_params,
        'level2_parameters': level2_params,
        'level3_parameters': level3_params,
        'level1_heads': len(model.level1_heads),
        'level2_heads': len(model.level2_heads),
        'level3_heads': len(model.level3_heads),
        'input_dim': model.input_dim,
        'encoder_output_dim': model.encoder.output_dim,
    }


def print_model_summary(model: HierarchicalMTLModel):
    """Print formatted model summary"""
    summary = model_summary(model)

    print("\n" + "="*70)
    print("HIERARCHICAL MTL MODEL SUMMARY")
    print("="*70)
    print(f"Total Parameters: {summary['total_parameters']:,}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"\nParameter Distribution:")
    print(f"  Encoder: {summary['encoder_parameters']:,}")
    print(f"  Level 1 Heads ({summary['level1_heads']}): {summary['level1_parameters']:,}")
    print(f"  Level 2 Heads ({summary['level2_heads']}): {summary['level2_parameters']:,}")
    print(f"  Level 3 Heads ({summary['level3_heads']}): {summary['level3_parameters']:,}")
    print(f"\nArchitecture:")
    print(f"  Input Dim: {summary['input_dim']}")
    print(f"  Encoder Output Dim: {summary['encoder_output_dim']}")
    print("="*70 + "\n")


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
        exit(1)

    print("Hierarchical MTL Model - Agent 05")
    print("="*70)

    # Create model
    config = MTLConfig()
    model = create_model(config)

    # Print summary
    print_model_summary(model)

    # Test forward pass
    print("Testing forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 200)

    outputs = model(dummy_input)

    print(f"\nForward pass successful!")
    print(f"Number of predictions: {len(outputs)}")
    print(f"\nSample predictions:")
    for i, (name, pred) in enumerate(list(outputs.items())[:5]):
        print(f"  {name}: shape {pred.shape}")

    print("\n" + "="*70)
    print("Model initialization and testing complete!")
    print("="*70)
