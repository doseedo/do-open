#!/usr/bin/env python3
"""
Unified Self-Improving Instrument Classifier

Combines all classifiers into a single network with knowledge sharing.
Uses SOTA techniques: Mean Teacher, Knowledge Distillation, Contrastive Learning.

Run overnight:
  nohup python3 unified_self_improving_classifier.py --mode full > unified_training.log 2>&1 &

Or step by step:
  python3 unified_self_improving_classifier.py --mode train --epochs 30
  python3 unified_self_improving_classifier.py --mode refine --rounds 5
  python3 unified_self_improving_classifier.py --mode eval
"""

import argparse
import copy
import gc
import logging
import random
import sys
from collections import Counter, defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import orjson
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ===================== CONFIGURATION =====================

CORRECTIONS_PATH = Path("/home/arlo/gcs-bucket/Manifests/corrections.json")
UNIFIED_CORRECTIONS_PATH = Path("/home/arlo/gcs-bucket/Manifests/unified_corrections.json")
MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/unified_manifest.json")
GROUP_MODEL_PATH = Path("/home/arlo/Data/latent_classifier/model.pt")
STEMS_CLASSIFIED_PATH = Path("/home/arlo/Data/mix_classifier/stems_classified.json")
STEMS_LATENTS_BASE = Path("/home/arlo/gcs-bucket/LatentDemucsV2")
LATENTS_ROOT = Path("/home/arlo/gcs-bucket/Latents")
OUTPUT_DIR = Path("/home/arlo/Data/unified_classifier")

# Classes
GROUP_CLASSES = [
    'bass', 'brass', 'drums', 'guitar', 'mallets', 'organ',
    'percussion', 'piano', 'plucked', 'strings', 'synth', 'voice', 'winds'
]

# Full hierarchy: group -> subgroup -> techniques
# CONSERVATIVE: Only include techniques we can reliably label
# Other techniques can be discovered via timbre_profiler.py
HIERARCHY = {
    'brass': {
        # Muted vs open is labeled from mute_manifest
        'trumpet': ['open', 'muted'],
        'trombone': ['open', 'muted'],
        'french_horn': ['open', 'muted'],
        'tuba': ['open'],
        'flugelhorn': ['open'],
        'brass_section': ['open'],
    },
    'strings': {
        # Bow technique is consistent per-take, labeled from filename
        'violin': ['arco', 'pizz', 'tremolo'],
        'viola': ['arco', 'pizz', 'tremolo'],
        'cello': ['arco', 'pizz', 'tremolo'],
        'double_bass': ['arco', 'pizz'],
        'string_section': ['arco', 'pizz'],
    },
    'guitar': {
        # Techniques vary within recordings - no reliable labels
        'acoustic_guitar': ['default'],
        'electric_guitar': ['default'],
        'classical_guitar': ['default'],
    },
    'bass': {
        'electric_bass': ['default'],
        'upright_bass': ['arco', 'pizz'],  # Bow technique from filename
        'synth_bass': ['default'],
    },
    'piano': {
        # Articulation varies within recordings
        'grand_piano': ['default'],
        'upright_piano': ['default'],
        'electric_piano': ['default'],
    },
    'winds': {
        'flute': ['default'],
        'clarinet': ['default'],
        'saxophone': ['default'],
        'oboe': ['default'],
        'bassoon': ['default'],
    },
    'drums': {
        # Stick type is consistent per-session, labeled from filename
        'kit': ['sticks', 'brushes', 'mallets', 'rods'],
    },
    'percussion': {
        'hand_percussion': ['default'],
        'tuned_percussion': ['default'],
    },
    'mallets': {
        'vibraphone': ['default'],
        'marimba': ['default'],
        'xylophone': ['default'],
        'glockenspiel': ['default'],
    },
    'organ': {
        'pipe_organ': ['default'],
        'hammond': ['default'],
    },
    'synth': {
        'lead_synth': ['default'],
        'pad_synth': ['default'],
        'bass_synth': ['default'],
    },
    'voice': {
        # Gender from vocal model
        'lead': ['male', 'female'],
        'background': ['male', 'female'],
        'choir': ['default'],
    },
    'plucked': {
        'harp': ['default'],
        'banjo': ['default'],
        'mandolin': ['default'],
    },
}

# Backward compatibility: extract subgroup map from hierarchy
SUBGROUP_MAP = {group: list(subgroups.keys()) for group, subgroups in HIERARCHY.items()}

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
REFINEMENT_ROUNDS = 5
PSEUDO_LABEL_THRESHOLD = 0.90
TEACHER_MOMENTUM = 0.999
TEMPERATURE = 3.0
CONTRASTIVE_TEMP = 0.1

# Feature extraction
POOL_METHODS = ['mean', 'std', 'max']
INPUT_DIM = 8 * 16 * len(POOL_METHODS)  # 384

# Excluded classes for training
EXCLUDED_CLASSES = {'undefined', 'room', 'fx', 'click', 'silent', 'junk', 'review_vocals', 'full-track'}

# Bleed instrument mapping: correction bleed_instruments values → GROUP_CLASSES
BLEED_INSTRUMENT_MAP = {
    'drums': 'drums', 'strings': 'strings', 'voice': 'voice', 'brass': 'brass',
    'guitar': 'guitar', 'bass': 'bass', 'piano': 'piano', 'organ': 'organ',
    'synth': 'synth', 'winds': 'winds', 'percussion': 'percussion',
    'mallets': 'mallets', 'plucked': 'plucked',
    'e-drums': 'drums',      # Map e-drums → drums GROUP_CLASS
    'dialogue': None,         # Not an instrument - handled by dialogue_head
}

# Data source weights
WEIGHT_CORRECTION = 3.0        # Manual corrections (highest quality)
WEIGHT_CORRECTION_HARD_NEG = 4.0  # Corrections where classifier was wrong
WEIGHT_STEMS_GT = 2.0          # Mix stems classified by group classifier
WEIGHT_FILENAME_HIGH = 1.5     # High-confidence filename match
WEIGHT_FILENAME_LOW = 0.8      # Lower-confidence filename match
WEIGHT_MANIFEST = 1.0          # Manifest labels (some noise)


# ===================== IMPROVED FILENAME LABELING =====================

import re

# Patterns that are TOO SHORT or AMBIGUOUS - will cause false positives
AMBIGUOUS_PATTERNS = {
    'gt', 'vl', 'vc', 'tb', 'fh', 'tp', 'ac', 'di', 'ep', 'bd', 'sn', 'hh',
    'oh', 'rm', 'vo', 'v1', 'v2', 'bs', 'el', 'hr', 'cl', 'ob', 'bn', 'fl',
    'pn', 'dr', 'ky', 'og', 'pc', 'fx', 'sx', 'tr', 'vx', 'bg', 'ld', 'rh',
    'wd', 'st', 'br', 'pz', 'ar', 'hn', 'tbn', 'hrn', 'vla', 'vln', 'clo',
}

# High-confidence patterns - unambiguous, full words
FILENAME_PATTERNS_HIGH_CONF = {
    'drums': [
        'kick', 'snare', 'hihat', 'cymbal', 'overhead', 'drumkit', 'drums',
        'floortom', 'racktom', 'crash', 'ride', 'china', 'splash',
    ],
    'bass': [
        'bass', 'bassamp', 'bassdi', 'subbass', 'upright_bass', 'electric_bass',
    ],
    'guitar': [
        'guitar', 'acoustic_guitar', 'electric_guitar', 'guit', 'strat', 'tele',
        'lespaul', 'acousticgtr', 'electricgtr',
    ],
    'piano': [
        'piano', 'grand', 'upright', 'steinway', 'yamaha_piano', 'keyboard',
        'rhodes', 'wurlitzer', 'electric_piano', 'epiano',
    ],
    'organ': [
        'organ', 'hammond', 'b3organ', 'leslie',
    ],
    'voice': [
        'vocal', 'vocals', 'voice', 'vox', 'lead_vocal', 'background_vocal',
        'bgvocal', 'choir', 'soprano', 'alto', 'tenor_voice', 'baritone',
        'adlib', 'harmony', 'double_vocal',
    ],
    'strings': [
        'violin', 'viola', 'cello', 'violoncello', 'string_section', 'strings',
        'fiddle', 'double_bass', 'contrabass', 'string_ensemble',
    ],
    'brass': [
        'trumpet', 'trombone', 'french_horn', 'tuba', 'flugelhorn', 'cornet',
        'brass_section', 'horn_section', 'brass',
    ],
    'winds': [
        'saxophone', 'flute', 'clarinet', 'oboe', 'bassoon', 'piccolo',
        'alto_sax', 'tenor_sax', 'bari_sax', 'soprano_sax', 'woodwind',
        'recorder', 'panflute',
    ],
    'mallets': [
        'glockenspiel', 'marimba', 'xylophone', 'vibraphone', 'vibes',
        'bells', 'chimes', 'tubular_bells', 'celesta',
    ],
    'percussion': [
        'percussion', 'tambourine', 'shaker', 'conga', 'bongo', 'cabasa',
        'cowbell', 'triangle', 'timpani', 'cajon', 'djembe', 'clap',
    ],
    'synth': [
        'synth', 'synthesizer', 'moog', 'prophet', 'juno', 'lead_synth',
        'pad', 'synth_pad', 'analog_synth',
    ],
    'click': [
        'click', 'metronome', 'count', '2pop', 'click_track',
    ],
    'room': [
        'room', 'room_mic', 'ambient', 'ambience',
    ],
    'fx': [
        'sfx', 'effect', 'sweep', 'riser', 'impact', 'whoosh', 'glitch',
    ],
}

# Medium-confidence patterns - need word boundary checking
FILENAME_PATTERNS_MED_CONF = {
    'drums': ['tom', 'kick', 'hat', 'kit'],
    'bass': ['bss', 'bamp'],
    'guitar': ['gtr', 'elec', 'acou'],
    'piano': ['keys', 'pno', 'nord'],
    'voice': ['vox', 'bgv', 'ldv'],
    'strings': ['viol', 'cell', 'str'],
    'brass': ['tpt', 'bone', 'horn', 'tuba'],
    'winds': ['sax', 'flut', 'clar'],
    'percussion': ['perc', 'tamb', 'shak'],
}


def tokenize_filename(filename: str) -> List[str]:
    """Split filename into tokens, handling various separators."""
    # Remove extension
    name = Path(filename).stem
    # Split on separators and numbers
    tokens = re.split(r'[\s_\-\.]+|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', name)
    # Filter and lowercase
    tokens = [t.lower() for t in tokens if t and len(t) > 0]
    return tokens


def classify_by_filename_strict(filename: str) -> Tuple[Optional[str], float]:
    """
    Classify audio file by filename with strict matching.
    Returns (group, confidence) where confidence indicates match quality.

    Rules:
    - Minimum 3 chars for any match
    - Full word matches preferred
    - No ambiguous short patterns
    - Context-aware (avoid substring matches in longer words)
    """
    tokens = tokenize_filename(filename)
    tokens_set = set(tokens)
    name_lower = filename.lower()
    name_normalized = re.sub(r'[^a-z0-9]', '', name_lower)

    # Check high-confidence patterns first (full words)
    for group, patterns in FILENAME_PATTERNS_HIGH_CONF.items():
        for pattern in patterns:
            pattern_normalized = pattern.replace('_', '')

            # Exact token match (best)
            if pattern.replace('_', '') in tokens_set:
                return group, 1.0

            # Multi-word pattern in filename
            if '_' in pattern and pattern.replace('_', ' ') in name_lower:
                return group, 0.95

            # Pattern as substring but with word boundaries
            if len(pattern) >= 4:
                # Check it's not part of a longer unrelated word
                pattern_re = r'\b' + re.escape(pattern_normalized) + r'\b'
                if re.search(pattern_re, name_normalized):
                    return group, 0.9
                # Check in original with separators
                if re.search(r'(^|[_\-\s\.])' + re.escape(pattern) + r'($|[_\-\s\.])', name_lower):
                    return group, 0.9

    # Check medium-confidence patterns (need more careful matching)
    for group, patterns in FILENAME_PATTERNS_MED_CONF.items():
        for pattern in patterns:
            # Skip ambiguous patterns
            if pattern in AMBIGUOUS_PATTERNS:
                continue

            # Must be at least 3 chars
            if len(pattern) < 3:
                continue

            # Exact token match only for medium patterns
            if pattern in tokens_set:
                # Additional check: token shouldn't be part of a longer word
                # e.g., "gtr" in "guitar" is fine, but "hat" in "whatever" is not
                return group, 0.7

    return None, 0.0


def get_filename_label(path: str) -> Optional[Dict]:
    """
    Get label from filename with confidence weighting.
    Returns None if no confident match.
    """
    filename = Path(path).name

    group, confidence = classify_by_filename_strict(filename)

    if group is None or confidence < 0.5:
        return None

    # Skip excluded groups
    if group in EXCLUDED_CLASSES:
        return None

    # Determine weight based on confidence
    if confidence >= 0.9:
        weight = WEIGHT_FILENAME_HIGH
    else:
        weight = WEIGHT_FILENAME_LOW

    return {
        'group': group,
        'confidence': confidence,
        'weight': weight,
        'source': 'filename',
    }


# ===================== LOGGING =====================

def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


# ===================== FEATURE EXTRACTION =====================

def load_latent(latent_path: Path) -> Optional[torch.Tensor]:
    """Load latent tensor from file."""
    try:
        data = torch.load(latent_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            return data.get('latents', data.get('z', None))
        return data
    except Exception:
        return None


def pool_latent(latent: torch.Tensor, mask_silent: bool = True) -> torch.Tensor:
    """Pool latent [8, 16, T] or [C, T] to fixed-size feature vector."""
    if latent is None:
        return torch.zeros(INPUT_DIM)

    # Handle different shapes
    if latent.dim() == 2:
        # [C, T] format - reshape to [8, 16, T] if possible
        if latent.shape[0] == 128:
            latent = latent.view(8, 16, -1)
        else:
            # Just pool over time
            features = []
            if 'mean' in POOL_METHODS:
                features.append(latent.mean(dim=-1))
            if 'std' in POOL_METHODS:
                std_feat = latent.std(dim=-1, unbiased=False)
                std_feat = torch.nan_to_num(std_feat, nan=0.0)
                features.append(std_feat)
            if 'max' in POOL_METHODS:
                features.append(latent.max(dim=-1)[0])
            result = torch.cat(features, dim=-1).flatten()
            return torch.nan_to_num(result, nan=0.0)

    # [8, 16, T] format
    if mask_silent and latent.shape[-1] > 2:
        energy = torch.sqrt((latent ** 2).mean(dim=(0, 1)))
        non_silent = energy > 0.01
        if non_silent.sum() >= 2:  # Need at least 2 frames for std
            latent = latent[:, :, non_silent]

    features = []
    if 'mean' in POOL_METHODS:
        features.append(latent.mean(dim=-1))
    if 'std' in POOL_METHODS:
        # Use unbiased=False to avoid NaN with small samples, fill NaN with 0
        std_feat = latent.std(dim=-1, unbiased=False)
        std_feat = torch.nan_to_num(std_feat, nan=0.0)
        features.append(std_feat)
    if 'max' in POOL_METHODS:
        features.append(latent.max(dim=-1)[0])

    stacked = torch.stack(features, dim=-1)
    result = stacked.flatten()
    # Final NaN check
    result = torch.nan_to_num(result, nan=0.0)
    return result


def audio_path_to_latent_path(audio_path: str) -> Optional[Path]:
    """Convert audio path to latent path.

    Checks both LATENTS_ROOT and STEMS_LATENTS_BASE for the latent file.
    """
    audio_path = Path(audio_path)

    # Find relative path
    parts = audio_path.parts
    if 'gcs-bucket' in parts:
        idx = parts.index('gcs-bucket')
        rel_path = Path(*parts[idx+1:])
    elif 'protools' in parts:
        idx = parts.index('protools')
        rel_path = Path(*parts[idx:])
    elif 'protoolsA' in parts:
        idx = parts.index('protoolsA')
        rel_path = Path(*parts[idx:])
    else:
        rel_path = audio_path

    # Check for latent files in multiple locations
    stem = rel_path.with_suffix('')

    # Also try without double extension (e.g., .wav.dcae.pt -> .dcae.pt)
    stem_no_ext = Path(str(stem).replace('.wav', '').replace('.mp3', '').replace('.flac', ''))

    for base in [LATENTS_ROOT, STEMS_LATENTS_BASE]:
        for s in [stem, stem_no_ext]:
            for ext in ['.dcae.pt', '.pt']:
                latent_path = base / f"{s}{ext}"
                if latent_path.exists():
                    return latent_path
                # Also check just the filename in base
                latent_path = base / f"{s.name}{ext}"
                if latent_path.exists():
                    return latent_path

    return None


# ===================== MODEL ARCHITECTURE =====================

class SharedEncoder(nn.Module):
    """Shared feature encoder for all tasks."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 256, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalSequenceModel(nn.Module):
    """Lightweight bidirectional GRU for cross-window temporal context."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """x: [B, T, embed_dim] → [B, T, embed_dim] with temporal context."""
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, _ = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.gru(x)
        return self.norm(self.proj(output) + x)  # Residual connection


class UnifiedClassifier(nn.Module):
    """
    Unified model with shared encoder and hierarchical task heads.

    Hierarchy:
        group (always predicted)
        └── subgroup (predicted per group)
            └── technique (predicted per subgroup)

    Knowledge flows through shared representations.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        group_classes: List[str] = None,
        subgroup_map: Dict[str, List[str]] = None,
        hierarchy: Dict[str, Dict[str, List[str]]] = None
    ):
        super().__init__()

        self.group_classes = group_classes or GROUP_CLASSES
        self.hierarchy = hierarchy or HIERARCHY
        # Extract subgroup map from hierarchy if not provided
        self.subgroup_map = subgroup_map or {g: list(subs.keys()) for g, subs in self.hierarchy.items()}
        self.num_groups = len(self.group_classes)
        self.embed_dim = embed_dim

        # Shared encoder
        self.encoder = SharedEncoder(input_dim, hidden_dim, embed_dim)

        # Level 1: Group head (always predicted)
        self.group_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.num_groups)
        )

        self.multilabel_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.num_groups)
        )

        self.mix_detector = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # Bleed detection head: multi-label sigmoid predicting which instruments bleed
        self.bleed_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_groups)
        )

        # Dialogue detection head: binary sigmoid for speech/dialogue presence
        self.dialogue_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        # Level 2: Subgroup heads (one per group with multiple subgroups)
        self.subgroup_heads = nn.ModuleDict()
        self.subgroup_classes = {}  # group -> [subgroup names]
        for group, subgroups in self.subgroup_map.items():
            self.subgroup_classes[group] = subgroups
            if len(subgroups) > 1:
                self.subgroup_heads[group] = nn.Sequential(
                    nn.Linear(embed_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, len(subgroups))
                )

        # Level 3: Technique heads (one per group/subgroup with multiple techniques)
        self.technique_heads = nn.ModuleDict()
        self.technique_classes = {}  # "group/subgroup" -> [technique names]
        for group, subgroups in self.hierarchy.items():
            for subgroup, techniques in subgroups.items():
                key = f"{group}/{subgroup}"
                self.technique_classes[key] = techniques
                if len(techniques) > 1:
                    # Use underscores in module key (ModuleDict doesn't like slashes)
                    module_key = f"{group}_{subgroup}"
                    self.technique_heads[module_key] = nn.Sequential(
                        nn.Linear(embed_dim, 32),
                        nn.ReLU(),
                        nn.Linear(32, len(techniques))
                    )

        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 64),
        )

        # Temporal sequence model for cross-window context
        self.temporal_model = TemporalSequenceModel(embed_dim=embed_dim)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> Dict[str, torch.Tensor]:
        emb = self.encoder(x)

        outputs = {
            'group': self.group_head(emb),
            'multilabel': self.multilabel_head(emb),
            'is_mix': self.mix_detector(emb),
            'bleed': self.bleed_head(emb),
            'dialogue': self.dialogue_head(emb),
        }

        # Subgroup predictions (per group)
        for group, head in self.subgroup_heads.items():
            outputs[f'subgroup_{group}'] = head(emb)

        # Technique predictions (per group/subgroup)
        for module_key, head in self.technique_heads.items():
            outputs[f'technique_{module_key}'] = head(emb)

        if return_embedding:
            outputs['embedding'] = emb
            outputs['projection'] = F.normalize(self.projector(emb), dim=-1)

        return outputs

    def forward_temporal(self, window_features: torch.Tensor, lengths: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for temporal sequences of windows.

        window_features: [B, T, input_dim] - pooled features per window
        lengths: [B] - number of valid windows per sequence
        Returns: dict with per-window predictions after temporal smoothing
        """
        B, T, D = window_features.shape
        flat = window_features.view(B * T, D)
        emb = self.encoder(flat)
        emb_seq = emb.view(B, T, -1)

        # Apply temporal GRU for cross-window context
        ctx_emb = self.temporal_model(emb_seq, lengths)
        ctx_flat = ctx_emb.view(B * T, -1)

        return {
            'group': self.group_head(ctx_flat).view(B, T, -1),
            'multilabel': self.multilabel_head(ctx_flat).view(B, T, -1),
            'bleed': self.bleed_head(ctx_flat).view(B, T, -1),
            'dialogue': self.dialogue_head(ctx_flat).view(B, T, -1),
        }

    def get_group_idx(self, group_name: str) -> int:
        try:
            return self.group_classes.index(group_name)
        except ValueError:
            return -1

    def get_subgroup_idx(self, group: str, subgroup: str) -> int:
        """Get index of subgroup within a group."""
        if group not in self.subgroup_classes:
            return -1
        try:
            return self.subgroup_classes[group].index(subgroup)
        except ValueError:
            return -1

    def get_technique_idx(self, group: str, subgroup: str, technique: str) -> int:
        """Get index of technique within a group/subgroup."""
        key = f"{group}/{subgroup}"
        if key not in self.technique_classes:
            return -1
        try:
            return self.technique_classes[key].index(technique)
        except ValueError:
            return -1


# ===================== MEAN TEACHER =====================

class MeanTeacher:
    """Exponential moving average model for pseudo-labeling."""

    def __init__(self, student: nn.Module, momentum: float = TEACHER_MOMENTUM):
        self.student = student
        self.momentum = momentum

        # Deep copy student to create teacher
        self.teacher = copy.deepcopy(student)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self):
        """Update teacher weights as EMA of student."""
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.mul_(self.momentum).add_(sp.data, alpha=1 - self.momentum)

    @torch.no_grad()
    def pseudo_label(self, x: torch.Tensor, threshold: float = PSEUDO_LABEL_THRESHOLD) -> Dict:
        """Generate pseudo-labels for unlabeled data."""
        self.teacher.eval()
        outputs = self.teacher(x)

        # Group predictions
        group_probs = F.softmax(outputs['group'], dim=1)
        group_conf, group_preds = group_probs.max(dim=1)
        group_mask = group_conf > threshold

        # Mix predictions
        mix_probs = F.softmax(outputs['is_mix'], dim=1)
        mix_conf, mix_preds = mix_probs.max(dim=1)
        mix_mask = mix_conf > threshold

        return {
            'group_preds': group_preds,
            'group_conf': group_conf,
            'group_mask': group_mask,
            'mix_preds': mix_preds,
            'mix_conf': mix_conf,
            'mix_mask': mix_mask,
        }

    def to(self, device):
        self.teacher = self.teacher.to(device)
        return self


# ===================== LOSS FUNCTIONS =====================

def knowledge_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = TEMPERATURE
) -> torch.Tensor:
    """KL divergence between student and teacher soft targets."""
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)


def supervised_contrastive_loss(
    projections: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = CONTRASTIVE_TEMP
) -> torch.Tensor:
    """Pull together same-class embeddings, push apart different classes."""
    device = projections.device
    batch_size = projections.shape[0]

    if batch_size < 2:
        return torch.tensor(0.0, device=device)

    # Normalize projections
    projections = F.normalize(projections, dim=1)

    # Similarity matrix
    sim = torch.matmul(projections, projections.T) / temperature

    # Positive mask (same label, excluding self)
    labels = labels.view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).float()
    pos_mask.fill_diagonal_(0)

    # Check if any positive pairs exist
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    # Log-softmax for numerical stability
    logits_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - logits_max.detach()

    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Mean log-likelihood over positive pairs
    mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)

    # Only for samples with positives
    valid = pos_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    return -mean_log_prob[valid].mean()


def hierarchical_consistency_loss(
    group_logits: torch.Tensor,
    subgroup_outputs: Dict[str, torch.Tensor],
    group_classes: List[str],
    subgroup_map: Dict[str, List[str]]
) -> torch.Tensor:
    """Enforce group-subgroup consistency."""
    loss = torch.tensor(0.0, device=group_logits.device)
    group_probs = F.softmax(group_logits, dim=1)

    for group, subgroups in subgroup_map.items():
        key = f'subgroup_{group}'
        if key not in subgroup_outputs:
            continue

        try:
            group_idx = group_classes.index(group)
        except ValueError:
            continue

        group_prob = group_probs[:, group_idx]
        sub_probs = F.softmax(subgroup_outputs[key], dim=1)
        max_sub_prob = sub_probs.max(dim=1)[0]

        # Penalize if subgroup more confident than group
        violation = F.relu(max_sub_prob - group_prob - 0.1)
        loss = loss + violation.mean()

    return loss


# ===================== DATASET =====================

class LRUCache:
    """Simple bounded LRU cache to avoid memory issues."""

    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache


class UnifiedDataset(Dataset):
    """Dataset combining all data sources with hierarchical labels."""

    def __init__(
        self,
        samples: List[Dict],
        group_classes: List[str],
        hierarchy: Dict[str, Dict[str, List[str]]] = None,
        transform=None,
        cache_features: bool = True,
        cache_maxsize: int = 10000
    ):
        self.samples = samples
        self.group_classes = group_classes
        self.hierarchy = hierarchy or HIERARCHY
        self.subgroup_map = {g: list(subs.keys()) for g, subs in self.hierarchy.items()}
        self.transform = transform
        self.cache_features = cache_features
        self._feature_cache = LRUCache(maxsize=cache_maxsize)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        path = sample.get('path', '')

        # Load features (with bounded LRU cache)
        # For temporal samples, use composite key to avoid returning wrong window
        has_region = 'region_start' in sample
        if has_region:
            cache_key = f"{path}:{sample['region_start']:.2f}-{sample['region_end']:.2f}"
        else:
            cache_key = path

        cached = self._feature_cache.get(cache_key) if self.cache_features else None
        if cached is not None:
            features = cached
        else:
            features = self._load_features(sample)
            if self.cache_features and cache_key:
                self._feature_cache.put(cache_key, features)

        if self.transform:
            features = self.transform(features)

        result = {
            'features': features,
            'path': path,
            'weight': sample.get('weight', 1.0),
        }

        # Level 1: Group label
        group_name = sample.get('group', '')
        if group_name and group_name in self.group_classes:
            result['group'] = self.group_classes.index(group_name)
            result['has_group'] = True
            result['group_name'] = group_name
        else:
            result['group'] = -1
            result['has_group'] = False
            result['group_name'] = ''

        # Level 2: Subgroup label
        subgroup_name = sample.get('subgroup', '')
        if group_name in self.subgroup_map and subgroup_name in self.subgroup_map[group_name]:
            result['subgroup'] = self.subgroup_map[group_name].index(subgroup_name)
            result['has_subgroup'] = True
            result['subgroup_name'] = subgroup_name
        else:
            result['subgroup'] = -1
            result['has_subgroup'] = False
            result['subgroup_name'] = ''

        # Level 3: Technique label
        technique_name = sample.get('technique', '')
        tech_key = f"{group_name}/{subgroup_name}"
        if tech_key in self.hierarchy.get(group_name, {}) or (
            group_name in self.hierarchy and subgroup_name in self.hierarchy[group_name]
        ):
            techniques = self.hierarchy.get(group_name, {}).get(subgroup_name, [])
            if technique_name and technique_name in techniques:
                result['technique'] = techniques.index(technique_name)
                result['has_technique'] = True
                result['technique_name'] = technique_name
            else:
                result['technique'] = -1
                result['has_technique'] = False
                result['technique_name'] = ''
        else:
            result['technique'] = -1
            result['has_technique'] = False
            result['technique_name'] = ''

        # Multi-label (for mixes)
        if 'instruments' in sample:
            multilabel = torch.zeros(len(self.group_classes))
            for inst in sample['instruments']:
                if inst in self.group_classes:
                    multilabel[self.group_classes.index(inst)] = 1.0
            result['multilabel'] = multilabel
            result['has_multilabel'] = True
        else:
            result['multilabel'] = torch.zeros(len(self.group_classes))
            result['has_multilabel'] = False

        # Is mix
        result['is_mix'] = 1 if sample.get('is_mix', False) else 0

        # Hard negative
        if 'hard_negative' in sample and sample['hard_negative'] in self.group_classes:
            result['hard_negative'] = self.group_classes.index(sample['hard_negative'])
        else:
            result['hard_negative'] = -1

        # Bleed detection: multi-hot target of bleed instrument groups
        if 'has_bleed' in sample:
            bleed_target = torch.zeros(len(self.group_classes))
            if sample.get('has_bleed') and sample.get('bleed_instruments_mapped'):
                for bi in sample['bleed_instruments_mapped']:
                    if bi in self.group_classes:
                        bleed_target[self.group_classes.index(bi)] = 1.0
            result['bleed_target'] = bleed_target
            result['has_bleed_label'] = True
        else:
            result['bleed_target'] = torch.zeros(len(self.group_classes))
            result['has_bleed_label'] = False

        # Dialogue detection: binary label
        if 'is_dialogue' in sample:
            result['is_dialogue'] = 1.0 if sample['is_dialogue'] else 0.0
            result['has_dialogue_label'] = True
        elif sample.get('has_dialogue_in_region'):
            result['is_dialogue'] = 1.0
            result['has_dialogue_label'] = True
        else:
            result['is_dialogue'] = 0.0
            result['has_dialogue_label'] = False

        return result

    def _load_features(self, sample: Dict) -> torch.Tensor:
        """Load and pool features from latent file.

        For temporal regions, extracts features from the specific time window.
        """
        latent = None

        # Try direct latent path first
        if 'latent_path' in sample:
            latent = load_latent(Path(sample['latent_path']))

        # Try converting audio path to latent path
        if latent is None and 'path' in sample:
            latent_path = audio_path_to_latent_path(sample['path'])
            if latent_path:
                latent = load_latent(latent_path)

        if latent is None:
            return torch.zeros(INPUT_DIM)

        # Handle temporal regions - extract specific time window
        if 'region_start' in sample and 'region_end' in sample:
            # ACE-Step latent: ~86.13 frames per second (44100 / 512)
            fps = 44100 / 512
            start_frame = int(sample['region_start'] * fps)
            end_frame = int(sample['region_end'] * fps)

            # Ensure valid range
            T = latent.shape[-1]
            start_frame = max(0, min(start_frame, T - 1))
            end_frame = max(start_frame + 1, min(end_frame, T))

            # Extract window
            if latent.dim() == 3:  # [8, 16, T]
                latent = latent[:, :, start_frame:end_frame]
            elif latent.dim() == 2:  # [C, T]
                latent = latent[:, start_frame:end_frame]

        return pool_latent(latent)


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function with hierarchical labels."""
    result = {
        'features': torch.stack([b['features'] for b in batch]),
        'paths': [b['path'] for b in batch],
        'weights': torch.tensor([b['weight'] for b in batch]),
        # Level 1: Group
        'group': torch.tensor([b['group'] for b in batch]),
        'has_group': torch.tensor([b['has_group'] for b in batch]),
        'group_names': [b['group_name'] for b in batch],
        # Level 2: Subgroup
        'subgroup': torch.tensor([b['subgroup'] for b in batch]),
        'has_subgroup': torch.tensor([b['has_subgroup'] for b in batch]),
        'subgroup_names': [b['subgroup_name'] for b in batch],
        # Level 3: Technique
        'technique': torch.tensor([b['technique'] for b in batch]),
        'has_technique': torch.tensor([b['has_technique'] for b in batch]),
        'technique_names': [b['technique_name'] for b in batch],
        # Multilabel and mix detection
        'multilabel': torch.stack([b['multilabel'] for b in batch]),
        'has_multilabel': torch.tensor([b['has_multilabel'] for b in batch]),
        'is_mix': torch.tensor([b['is_mix'] for b in batch]),
        'hard_negative': torch.tensor([b['hard_negative'] for b in batch]),
        # Bleed detection
        'bleed_target': torch.stack([b['bleed_target'] for b in batch]),
        'has_bleed_label': torch.tensor([b['has_bleed_label'] for b in batch]),
        # Dialogue detection
        'is_dialogue': torch.tensor([b['is_dialogue'] for b in batch]),
        'has_dialogue_label': torch.tensor([b['has_dialogue_label'] for b in batch]),
    }
    return result


# ===================== DATA LOADING =====================

def load_corrections() -> Dict:
    """Load manual corrections. Merges base + unified-specific (unified takes priority)."""
    corrections = {}
    if CORRECTIONS_PATH.exists():
        with open(CORRECTIONS_PATH, 'rb') as f:
            corrections = orjson.loads(f.read())
    # Unified corrections override base (these are from reviewing the unified model)
    if UNIFIED_CORRECTIONS_PATH.exists():
        with open(UNIFIED_CORRECTIONS_PATH, 'rb') as f:
            unified = orjson.loads(f.read())
        corrections.update(unified)
        logging.info(f"  Loaded {len(unified)} unified-specific corrections (override base)")
    return corrections


def load_stems_classified() -> Dict:
    """Load pre-classified stem data for mix GT."""
    if not STEMS_CLASSIFIED_PATH.exists():
        return {'results': []}
    with open(STEMS_CLASSIFIED_PATH, 'rb') as f:
        return orjson.loads(f.read())


def load_manifest() -> Dict:
    """Load unified manifest."""
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, 'rb') as f:
        return orjson.loads(f.read())


def build_training_data(
    corrections: Dict,
    stems_data: Dict,
    manifest: Dict,
    group_classes: List[str],
    max_isolated_per_class: int = 5000
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build training data from all sources.
    Returns (labeled_samples, unlabeled_samples).
    """
    labeled_samples = []
    unlabeled_samples = []

    logging.info("Building training data...")

    # 1. Manual corrections (highest quality)
    correction_count = 0
    temporal_count = 0
    hard_neg_count = 0
    mix_correction_count = 0
    dialogue_count = 0
    bleed_pos_count = 0
    bleed_neg_count = 0

    # Track paths the user explicitly labeled as mix/ensemble
    user_mix_paths = set()

    for path, corr in corrections.items():
        group = corr.get('group')
        if not group:
            continue

        # Handle dialogue corrections: dialogue is a PROPERTY, not a GROUP_CLASS
        if group == 'dialogue':
            sample = {
                'path': path,
                'is_dialogue': True,
                'source': 'correction',
                'weight': WEIGHT_CORRECTION,
                'is_mix': False,
            }
            if corr.get('previous_group') and corr['previous_group'] in group_classes:
                sample['hard_negative'] = corr['previous_group']
            labeled_samples.append(sample)
            dialogue_count += 1
            continue

        # Handle ensemble/mix corrections: these are is_mix labels, NOT excluded
        if group in ('ensemble', 'mix', 'full-track') or group.endswith('_mix'):
            user_mix_paths.add(path)
            mix_correction_count += 1
            continue

        if group in EXCLUDED_CLASSES:
            continue
        # Strip _roomy suffix for group classification
        clean_group = group.replace('_roomy', '')
        if clean_group not in group_classes:
            continue

        sample = {
            'path': path,
            'group': clean_group,
            'subgroup': corr.get('subgroup'),
            'source': 'correction',
            'weight': WEIGHT_CORRECTION,
            'is_mix': False,  # Explicitly isolated (user gave single instrument)
        }

        # Hard negatives get extra weight (classifier was wrong)
        if corr.get('previous_group') and corr['previous_group'] in group_classes:
            sample['hard_negative'] = corr['previous_group']
            sample['weight'] = WEIGHT_CORRECTION_HARD_NEG
            hard_neg_count += 1

        # Bleed instruments: map through BLEED_INSTRUMENT_MAP
        bleed_raw = corr.get('bleed_instruments', [])
        if 'bleed_instruments' in corr:
            mapped_bleed = []
            for bi in bleed_raw:
                mapped = BLEED_INSTRUMENT_MAP.get(bi)
                if mapped and mapped in group_classes:
                    mapped_bleed.append(mapped)
            sample['bleed_instruments_mapped'] = mapped_bleed
            sample['has_bleed'] = len(mapped_bleed) > 0
            if mapped_bleed:
                bleed_pos_count += 1
            else:
                bleed_neg_count += 1

        # Voice corrections are explicit dialogue NEGATIVES
        if clean_group == 'voice':
            sample['is_dialogue'] = False

        labeled_samples.append(sample)
        correction_count += 1

    logging.info(f"  Corrections: {correction_count} (hard negatives: {hard_neg_count}, mix labels: {mix_correction_count})")
    logging.info(f"  Dialogue labels: {dialogue_count}")
    logging.info(f"  Bleed labels: {bleed_pos_count} positive, {bleed_neg_count} negative")

    # 2. Mix GT from stems_classified.json
    # Uses ORIGINAL MIX latent + detected instruments for whole-file multilabel
    # Also extracts TEMPORAL REGIONS from timeline for windowed training
    mix_count = 0
    mix_multilabel_count = 0
    stems_temporal_count = 0
    stems_mix_paths = set()

    for result in stems_data.get('results', []):
        original_path = result['original_path']

        # Get detected instruments (whole-file multilabel GT)
        detected = result.get('detected_instruments', [])
        valid_instruments = [i for i in detected if i in group_classes] if detected else []

        if valid_instruments:
            stems_mix_paths.add(original_path)
            sample = {
                'path': original_path,
                'instruments': valid_instruments,
                'is_mix': True,
                'source': 'stems_classified',
                'weight': WEIGHT_STEMS_GT,
            }
            if len(valid_instruments) >= 2:
                mix_multilabel_count += 1
            if len(valid_instruments) == 1:
                sample['group'] = valid_instruments[0]
            labeled_samples.append(sample)
            mix_count += 1

        # 2b. Timeline temporal regions from Demucs stem energy analysis
        # Each region is a ~4.5s window with per-instrument confidence scores
        timeline = result.get('timeline', [])
        for region in timeline:
            instruments_data = region.get('instruments', [])
            if not instruments_data:
                continue

            # Filter to confident instruments in valid classes
            region_instruments = []
            for inst_info in instruments_data:
                inst_name = inst_info.get('instrument', '')
                conf = inst_info.get('confidence', 0)
                if inst_name in group_classes and conf > 0.4:
                    region_instruments.append(inst_name)

            if not region_instruments:
                continue

            sample = {
                'path': original_path,
                'instruments': region_instruments,
                'is_mix': len(region_instruments) > 1,
                'source': 'stems_temporal',
                'weight': 1.5,  # Medium weight - automated but temporal
                'region_start': region.get('start', 0),
                'region_end': region.get('end', 0),
            }
            if len(region_instruments) == 1:
                sample['group'] = region_instruments[0]
            labeled_samples.append(sample)
            stems_temporal_count += 1

    logging.info(f"  Mix stems GT: {mix_count} ({mix_multilabel_count} with 2+ instruments)")
    logging.info(f"  Stems temporal regions: {stems_temporal_count}")

    # 2c. Manual temporal region corrections (gold annotations)
    manual_temporal_count = 0
    dialogue_temporal_count = 0
    implicit_bleed_count = 0
    for path, corr in corrections.items():
        if not corr.get('regions'):
            continue
        regions = corr['regions']
        duration = corr.get('duration', 0)
        file_primary_group = corr.get('group', '')
        if file_primary_group:
            file_primary_group = file_primary_group.replace('_roomy', '')

        for region in regions:
            region_labels = region.get('labels', [])

            # Separate dialogue from instrument labels
            has_dialogue = 'dialogue' in region_labels
            valid_labels = [l for l in region_labels if l in group_classes]

            if not valid_labels and not has_dialogue:
                continue

            sample = {
                'path': path,
                'source': 'temporal_correction',
                'weight': WEIGHT_CORRECTION,
                'region_start': region.get('start', 0),
                'region_end': region.get('end', duration),
            }

            if valid_labels:
                sample['instruments'] = valid_labels
                sample['is_mix'] = len(valid_labels) > 1
                if len(valid_labels) == 1:
                    sample['group'] = valid_labels[0]

                # Derive implicit bleed: if file has a known primary group and region
                # has OTHER instruments, those are bleed in that context
                if file_primary_group and file_primary_group in group_classes and len(valid_labels) >= 2:
                    bleed_insts = [i for i in valid_labels if i != file_primary_group]
                    if bleed_insts:
                        sample['bleed_instruments_mapped'] = bleed_insts
                        sample['has_bleed'] = True
                        implicit_bleed_count += 1
            else:
                sample['is_mix'] = False

            # Flag dialogue in this temporal region
            if has_dialogue:
                sample['is_dialogue'] = True
                sample['has_dialogue_in_region'] = True
                dialogue_temporal_count += 1

            labeled_samples.append(sample)
            manual_temporal_count += 1

    logging.info(f"  Manual temporal regions: {manual_temporal_count} ({dialogue_temporal_count} with dialogue)")
    logging.info(f"  Implicit bleed from multi-label regions: {implicit_bleed_count}")

    # 2d. Build is_mix training labels from all sources
    # Positive: stems_classified mixes, user ensemble/mix corrections, manifest is_mix
    # Negative: user single-instrument corrections, isolated manifest entries
    all_mix_paths = stems_mix_paths | user_mix_paths

    # 3. Manifest + Filename labels with validation
    manifest_entries = manifest.get('entries', [])
    if isinstance(manifest, dict) and 'entries' not in manifest:
        # Old format: dict keyed by path
        manifest_entries = [{'audio_path': k, **v} for k, v in manifest.items() if isinstance(v, dict)]

    class_samples = defaultdict(list)
    filename_only_samples = []
    manifest_only_count = 0

    # Track paths we've already processed
    processed_paths = set(s['path'] for s in labeled_samples)

    logging.info(f"  Processing {len(manifest_entries)} manifest entries...")
    for i, entry in enumerate(manifest_entries):
        if i > 0 and i % 50000 == 0:
            logging.info(f"    ...processed {i}/{len(manifest_entries)}")

        if not isinstance(entry, dict):
            continue

        path = entry.get('audio_path', entry.get('path', ''))
        manifest_group = entry.get('group', 'undefined')

        # Skip if already in corrections or stems
        if path in corrections or path in processed_paths:
            continue

        # Check for latent
        has_latent = entry.get('has_latent', False)
        if not has_latent:
            continue

        # Determine if this is a mix file
        fname_lower = path.lower()
        is_mix_file = (
            entry.get('is_mix', False)
            or path in all_mix_paths
            or any(kw in fname_lower for kw in ['_mix', '/mix', 'mix.', 'mix_', 'master_', 'bounce', 'full_track', '/room', '_room'])
        )

        # Skip mix/room files for isolated group training
        if is_mix_file:
            continue

        # Determine final label and weight
        # OPTIMIZATION: Skip expensive filename regex for entries with valid manifest labels
        if manifest_group not in EXCLUDED_CLASSES and manifest_group in group_classes:
            # Manifest has a valid label - use it directly (skip filename validation for speed)
            weight = WEIGHT_MANIFEST
            source = 'manifest'
            final_group = manifest_group
            manifest_only_count += 1

            sample = {
                'path': path,
                'group': final_group,
                'source': source,
                'weight': weight,
                'is_mix': False,
            }

            # Add subgroup if present in manifest
            manifest_subgroup = entry.get('subgroup', 'undefined')
            if manifest_subgroup != 'undefined':
                sample['subgroup'] = manifest_subgroup

            # Add technique if present in manifest
            manifest_technique = entry.get('technique')
            if manifest_technique:
                sample['technique'] = manifest_technique

            class_samples[final_group].append(sample)

        else:
            # Manifest group is undefined - try filename detection (slower, but fewer entries)
            filename_label = get_filename_label(path)
            if filename_label and filename_label['group'] in group_classes:
                filename_only_samples.append({
                    'path': path,
                    'group': filename_label['group'],
                    'source': 'filename_only',
                    'weight': filename_label['weight'],
                    'is_mix': False,
                    'filename_confidence': filename_label['confidence'],
                })
            else:
                # No label from either source - unlabeled
                unlabeled_samples.append({
                    'path': path,
                    'source': 'unlabeled',
                })

    # Sample per class from manifest labels
    isolated_count = 0
    for group, samples in class_samples.items():
        if len(samples) > max_isolated_per_class:
            samples = random.sample(samples, max_isolated_per_class)
        labeled_samples.extend(samples)
        isolated_count += len(samples)

    logging.info(f"  Manifest labels: {isolated_count} (from {manifest_only_count} entries)")

    # Add filename-only labels (for undefined in manifest)
    # Sample per class to avoid imbalance
    filename_class_samples = defaultdict(list)
    for sample in filename_only_samples:
        filename_class_samples[sample['group']].append(sample)

    filename_added = 0
    max_filename_per_class = 2000  # Limit filename-only samples
    for group, samples in filename_class_samples.items():
        # Prioritize high-confidence matches
        samples.sort(key=lambda x: x.get('filename_confidence', 0), reverse=True)
        if len(samples) > max_filename_per_class:
            samples = samples[:max_filename_per_class]
        labeled_samples.extend(samples)
        filename_added += len(samples)

    logging.info(f"  Filename-only labels: {filename_added}")

    logging.info(f"  Unlabeled: {len(unlabeled_samples)}")
    logging.info(f"  Total labeled: {len(labeled_samples)}")

    # Log hierarchy coverage stats
    subgroup_count = sum(1 for s in labeled_samples if s.get('subgroup'))
    technique_count = sum(1 for s in labeled_samples if s.get('technique'))
    logging.info(f"  With subgroup labels: {subgroup_count}")
    logging.info(f"  With technique labels: {technique_count}")

    # Technique distribution
    technique_dist = Counter(s.get('technique') for s in labeled_samples if s.get('technique'))
    if technique_dist:
        logging.info(f"  Technique distribution: {dict(technique_dist.most_common(10))}")

    return labeled_samples, unlabeled_samples


# ===================== TRAINER =====================

class UnifiedTrainer:
    """Training loop with all SOTA techniques."""

    def __init__(
        self,
        model: UnifiedClassifier,
        device: str = 'cuda',
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        # Mean Teacher
        self.mean_teacher = MeanTeacher(model).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = None

        # Normalization parameters (loaded from group classifier)
        self.norm_mean = None
        self.norm_std = None

        # Load normalization from existing group classifier
        self._load_normalization()

        # Metrics
        self.train_losses = []
        self.val_accuracies = []

    def _load_normalization(self):
        """Load normalization parameters from trained group classifier (as fallback)."""
        if GROUP_MODEL_PATH.exists():
            data = torch.load(GROUP_MODEL_PATH, map_location='cpu', weights_only=False)
            self.norm_mean = data['mean'].to(self.device)
            self.norm_std = data['std'].to(self.device)
            logging.info("Loaded normalization from group classifier (fallback)")

    def compute_normalization(self, dataloader: DataLoader, max_batches: int = 100):
        """Compute fresh normalization stats from current training data.

        This is more accurate than using stats from the old classifier if
        the feature distribution has changed.
        """
        logging.info("Computing normalization stats from training data...")

        all_features = []
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            all_features.append(batch['features'])

        if not all_features:
            logging.warning("No features to compute normalization, using fallback")
            return

        all_features = torch.cat(all_features, dim=0)
        self.norm_mean = all_features.mean(dim=0).to(self.device)
        self.norm_std = all_features.std(dim=0).to(self.device)

        # Clamp std to avoid division by near-zero
        self.norm_std = torch.clamp(self.norm_std, min=1e-6)

        logging.info(f"Computed normalization from {len(all_features)} samples")

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features with NaN protection."""
        # Replace any NaN/Inf in input
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.norm_mean is not None:
            normalized = (features - self.norm_mean) / (self.norm_std + 1e-8)
            return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_losses = defaultdict(float)
        num_batches = 0

        for batch in dataloader:
            losses = self._train_step(batch)

            for k, v in losses.items():
                total_losses[k] += v
            num_batches += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        # Step scheduler
        if self.scheduler:
            self.scheduler.step()

        return avg_losses

    def _train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        features = batch['features'].to(self.device)
        features = self.normalize(features)

        # Forward pass
        outputs = self.model(features, return_embedding=True)

        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # 1. Group classification loss (isolated files)
        has_group = batch['has_group'].bool()
        if has_group.sum() > 0:
            group_targets = batch['group'][has_group].to(self.device)
            group_logits = outputs['group'][has_group]
            weights = batch['weights'][has_group].to(self.device)

            # Filter valid targets (>= 0)
            valid = group_targets >= 0
            if valid.sum() > 0:
                loss = F.cross_entropy(
                    group_logits[valid],
                    group_targets[valid],
                    reduction='none'
                )
                loss = (loss * weights[valid]).mean()
                losses['group'] = loss.item()
                total_loss = total_loss + loss

        # 2. Multi-label loss
        # Train on BOTH:
        # a) Mix files with explicit multilabel targets (2+ instruments) - HIGHER WEIGHT
        # b) Isolated files using group label as single-hot multilabel target
        #    This teaches the model what each instrument looks like in isolation
        #
        # Key insight: We weight actual mixes higher to prevent the multilabel head
        # from learning to always predict single classes (since isolated files dominate)

        has_ml = batch['has_multilabel'].bool()
        has_group_valid = (batch['group'] >= 0)
        is_mix = batch['is_mix'].bool()

        # Create multilabel targets from explicit labels or group labels
        ml_targets = batch['multilabel'].clone().to(self.device)

        # For isolated files without explicit multilabel, create one-hot from group
        isolated_mask = has_group_valid & ~has_ml
        if isolated_mask.sum() > 0:
            for i, idx in enumerate(torch.where(isolated_mask)[0]):
                group_idx = batch['group'][idx].item()
                if 0 <= group_idx < self.model.num_groups:
                    ml_targets[idx] = 0  # Clear any existing
                    ml_targets[idx, group_idx] = 1.0  # Set single class

        # Train multilabel on all samples with valid targets
        ml_mask = has_ml | isolated_mask
        if ml_mask.sum() > 0:
            ml_logits = outputs['multilabel'][ml_mask]
            targets = ml_targets[ml_mask]
            sample_weights = batch['weights'][ml_mask].to(self.device)

            # Weight actual mixes (multi-instrument) HIGHER to prevent single-class bias
            # Mixes with 2+ instruments get 3x weight, isolated files get 0.5x weight
            mix_weight_multiplier = torch.ones(ml_mask.sum(), device=self.device)
            is_actual_mix = is_mix[ml_mask]
            has_explicit_ml = has_ml[ml_mask]
            # True multilabel samples (explicit 2+ instruments)
            mix_weight_multiplier[has_explicit_ml] = 3.0
            # Isolated files get reduced weight on multilabel head
            mix_weight_multiplier[~has_explicit_ml] = 0.5

            # Weighted BCE loss per sample
            loss = F.binary_cross_entropy_with_logits(ml_logits, targets, reduction='none')
            loss = (loss.mean(dim=1) * sample_weights * mix_weight_multiplier).mean()

            losses['multilabel'] = loss.item()
            total_loss = total_loss + loss

        # 3. Mix detection loss
        mix_targets = batch['is_mix'].to(self.device)
        loss = F.cross_entropy(outputs['is_mix'], mix_targets)
        losses['mix_detect'] = loss.item()
        total_loss = total_loss + 0.5 * loss

        # 4. Contrastive loss
        if has_group.sum() >= 2:
            group_labels = batch['group'][has_group].to(self.device)
            projections = outputs['projection'][has_group]

            valid = group_labels >= 0
            if valid.sum() >= 2:
                loss = supervised_contrastive_loss(projections[valid], group_labels[valid])
                losses['contrastive'] = loss.item()
                total_loss = total_loss + 0.1 * loss

        # 5. Hierarchical consistency
        subgroup_outputs = {k: v for k, v in outputs.items() if k.startswith('subgroup_')}
        if subgroup_outputs:
            loss = hierarchical_consistency_loss(
                outputs['group'],
                subgroup_outputs,
                self.model.group_classes,
                self.model.subgroup_map
            )
            losses['hierarchical'] = loss.item()
            total_loss = total_loss + 0.1 * loss

        # 6. Hard negative penalty
        hard_neg = batch['hard_negative']
        has_hard_neg = hard_neg >= 0
        if has_hard_neg.sum() > 0:
            # Penalize predicting the wrong class it was before
            hard_neg_idx = hard_neg[has_hard_neg].to(self.device)
            group_probs = F.softmax(outputs['group'][has_hard_neg], dim=1)
            wrong_probs = group_probs.gather(1, hard_neg_idx.unsqueeze(1)).squeeze()
            loss = wrong_probs.mean()  # Minimize probability of wrong class
            losses['hard_neg'] = loss.item()
            total_loss = total_loss + 0.5 * loss

        # 7. Subgroup classification loss (when labeled)
        has_subgroup = batch['has_subgroup'].bool()
        if has_subgroup.sum() > 0:
            subgroup_loss = torch.tensor(0.0, device=self.device)
            subgroup_count = 0

            # Process each group's subgroup head
            for group in self.model.subgroup_classes.keys():
                if group not in self.model.subgroup_heads:
                    continue

                # Find samples with this group AND subgroup label
                group_mask = torch.tensor([g == group for g in batch['group_names']])
                combined_mask = has_subgroup & group_mask

                if combined_mask.sum() > 0:
                    subgroup_targets = batch['subgroup'][combined_mask].to(self.device)
                    subgroup_logits = outputs[f'subgroup_{group}'][combined_mask]
                    weights = batch['weights'][combined_mask].to(self.device)

                    valid = subgroup_targets >= 0
                    if valid.sum() > 0:
                        loss = F.cross_entropy(
                            subgroup_logits[valid],
                            subgroup_targets[valid],
                            reduction='none'
                        )
                        subgroup_loss = subgroup_loss + (loss * weights[valid]).mean()
                        subgroup_count += 1

            if subgroup_count > 0:
                subgroup_loss = subgroup_loss / subgroup_count
                losses['subgroup'] = subgroup_loss.item()
                total_loss = total_loss + 0.5 * subgroup_loss

        # 8. Technique classification loss (when labeled - sparse)
        has_technique = batch['has_technique'].bool()
        if has_technique.sum() > 0:
            technique_loss = torch.tensor(0.0, device=self.device)
            technique_count = 0

            # Process each group/subgroup technique head
            for module_key in self.model.technique_heads.keys():
                group, subgroup = module_key.split('_', 1)

                # Find samples with this group/subgroup AND technique label
                group_mask = torch.tensor([g == group for g in batch['group_names']])
                subgroup_mask = torch.tensor([s == subgroup for s in batch['subgroup_names']])
                combined_mask = has_technique & group_mask & subgroup_mask

                if combined_mask.sum() > 0:
                    technique_targets = batch['technique'][combined_mask].to(self.device)
                    technique_logits = outputs[f'technique_{module_key}'][combined_mask]
                    weights = batch['weights'][combined_mask].to(self.device)

                    valid = technique_targets >= 0
                    if valid.sum() > 0:
                        loss = F.cross_entropy(
                            technique_logits[valid],
                            technique_targets[valid],
                            reduction='none'
                        )
                        technique_loss = technique_loss + (loss * weights[valid]).mean()
                        technique_count += 1

            if technique_count > 0:
                technique_loss = technique_loss / technique_count
                losses['technique'] = technique_loss.item()
                total_loss = total_loss + 0.3 * technique_loss  # Lower weight - sparse labels

        # 9. Bleed detection loss (multi-label BCE)
        has_bleed_label = batch['has_bleed_label'].bool()
        if has_bleed_label.sum() > 0:
            bleed_logits = outputs['bleed'][has_bleed_label]
            bleed_targets = batch['bleed_target'][has_bleed_label].to(self.device)
            weights_bl = batch['weights'][has_bleed_label].to(self.device)
            pos_weight = torch.full((self.model.num_groups,), 5.0, device=self.device)
            loss = F.binary_cross_entropy_with_logits(
                bleed_logits, bleed_targets, pos_weight=pos_weight, reduction='none'
            )
            loss = (loss.mean(dim=1) * weights_bl).mean()
            losses['bleed'] = loss.item()
            total_loss = total_loss + 0.3 * loss

        # 10. Dialogue detection loss (binary BCE)
        has_dialogue_label = batch['has_dialogue_label'].bool()
        if has_dialogue_label.sum() > 0:
            dialogue_logits = outputs['dialogue'][has_dialogue_label].squeeze(-1)
            dialogue_targets = batch['is_dialogue'][has_dialogue_label].to(self.device)
            weights_dl = batch['weights'][has_dialogue_label].to(self.device)
            pos_weight_dl = torch.tensor(3.0, device=self.device)
            loss = F.binary_cross_entropy_with_logits(
                dialogue_logits, dialogue_targets,
                pos_weight=pos_weight_dl, reduction='none'
            )
            loss = (loss * weights_dl).mean()
            losses['dialogue'] = loss.item()
            total_loss = total_loss + 0.5 * loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update mean teacher
        self.mean_teacher.update()

        losses['total'] = total_loss.item()
        return losses

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()

        correct_group = 0
        total_group = 0
        correct_mix = 0
        total_mix = 0

        # Multilabel metrics
        ml_true_positives = 0
        ml_false_positives = 0
        ml_false_negatives = 0
        ml_total = 0

        # Dialogue metrics
        dialogue_correct = 0
        dialogue_total = 0

        # Bleed metrics
        bleed_tp = 0
        bleed_fp = 0
        bleed_fn = 0

        for batch in dataloader:
            features = batch['features'].to(self.device)
            features = self.normalize(features)

            outputs = self.model(features)

            # Group accuracy
            has_group = batch['has_group'].bool()
            if has_group.sum() > 0:
                group_targets = batch['group'][has_group].to(self.device)
                valid = group_targets >= 0
                if valid.sum() > 0:
                    preds = outputs['group'][has_group][valid].argmax(dim=1)
                    correct_group += (preds == group_targets[valid]).sum().item()
                    total_group += valid.sum().item()

            # Mix detection accuracy
            mix_targets = batch['is_mix'].to(self.device)
            preds = outputs['is_mix'].argmax(dim=1)
            correct_mix += (preds == mix_targets).sum().item()
            total_mix += len(mix_targets)

            # Multilabel metrics (for samples with explicit multilabel targets)
            has_ml = batch['has_multilabel'].bool()
            if has_ml.sum() > 0:
                ml_logits = outputs['multilabel'][has_ml]
                ml_targets = batch['multilabel'][has_ml].to(self.device)
                ml_preds = (torch.sigmoid(ml_logits) > 0.5).float()
                ml_true_positives += (ml_preds * ml_targets).sum().item()
                ml_false_positives += (ml_preds * (1 - ml_targets)).sum().item()
                ml_false_negatives += ((1 - ml_preds) * ml_targets).sum().item()
                ml_total += has_ml.sum().item()

            # Dialogue accuracy
            has_dl = batch['has_dialogue_label'].bool()
            if has_dl.sum() > 0:
                dl_logits = outputs['dialogue'][has_dl].squeeze(-1)
                dl_targets = batch['is_dialogue'][has_dl].to(self.device)
                dl_preds = (torch.sigmoid(dl_logits) > 0.5).float()
                dialogue_correct += (dl_preds == dl_targets).sum().item()
                dialogue_total += has_dl.sum().item()

            # Bleed metrics
            has_bl = batch['has_bleed_label'].bool()
            if has_bl.sum() > 0:
                bl_logits = outputs['bleed'][has_bl]
                bl_targets = batch['bleed_target'][has_bl].to(self.device)
                bl_preds = (torch.sigmoid(bl_logits) > 0.5).float()
                bleed_tp += (bl_preds * bl_targets).sum().item()
                bleed_fp += (bl_preds * (1 - bl_targets)).sum().item()
                bleed_fn += ((1 - bl_preds) * bl_targets).sum().item()

        # Calculate multilabel F1
        ml_precision = ml_true_positives / (ml_true_positives + ml_false_positives + 1e-8)
        ml_recall = ml_true_positives / (ml_true_positives + ml_false_negatives + 1e-8)
        ml_f1 = 2 * ml_precision * ml_recall / (ml_precision + ml_recall + 1e-8)

        # Calculate bleed F1
        bl_precision = bleed_tp / (bleed_tp + bleed_fp + 1e-8)
        bl_recall = bleed_tp / (bleed_tp + bleed_fn + 1e-8)
        bl_f1 = 2 * bl_precision * bl_recall / (bl_precision + bl_recall + 1e-8)

        return {
            'group_accuracy': correct_group / total_group if total_group > 0 else 0,
            'mix_accuracy': correct_mix / total_mix if total_mix > 0 else 0,
            'multilabel_precision': ml_precision,
            'multilabel_recall': ml_recall,
            'multilabel_f1': ml_f1,
            'dialogue_accuracy': dialogue_correct / dialogue_total if dialogue_total > 0 else 0,
            'bleed_f1': bl_f1,
        }

    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        unlabeled_samples: List[Dict],
        threshold: float = PSEUDO_LABEL_THRESHOLD,
        batch_size: int = 256
    ) -> List[Dict]:
        """Generate pseudo-labels for unlabeled data using Mean Teacher."""
        self.model.eval()

        new_labels = []

        # Create simple dataloader
        dataset = UnifiedDataset(unlabeled_samples, self.model.group_classes)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=32, pin_memory=True)
        total_batches = len(dataloader)
        logging.info(f"Generating pseudo-labels: {len(unlabeled_samples)} samples, {total_batches} batches")

        max_conf_seen = 0.0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 50 == 0:
                logging.info(f"  Pseudo-label batch {batch_idx}/{total_batches} | {len(new_labels)} labels so far | max_conf: {max_conf_seen:.4f}")
            features = batch['features'].to(self.device)
            features = self.normalize(features)
            paths = batch['paths']

            pseudo = self.mean_teacher.pseudo_label(features, threshold=threshold)
            batch_max = pseudo['group_conf'].max().item()
            if batch_max > max_conf_seen:
                max_conf_seen = batch_max

            for i, path in enumerate(paths):
                if pseudo['group_mask'][i]:
                    group_idx = pseudo['group_preds'][i].item()
                    conf = pseudo['group_conf'][i].item()

                    new_labels.append({
                        'path': path,
                        'group': self.model.group_classes[group_idx],
                        'confidence': conf,
                        'source': 'pseudo_label',
                        'weight': conf,  # Weight by confidence
                        'is_mix': pseudo['mix_preds'][i].item() == 1,
                    })

        return new_labels

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict = None):
        """Save model checkpoint with full hierarchy."""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'norm_mean': self.norm_mean.cpu() if self.norm_mean is not None else None,
            'norm_std': self.norm_std.cpu() if self.norm_std is not None else None,
            'group_classes': self.model.group_classes,
            'subgroup_map': self.model.subgroup_map,
            'hierarchy': self.model.hierarchy,
            'technique_classes': self.model.technique_classes,
            'metrics': metrics,
            'saved_at': datetime.now().isoformat(),
        }
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # strict=False for backward compat (old checkpoints lack bleed/dialogue/temporal heads)
        missing, unexpected = self.model.load_state_dict(checkpoint['model_state'], strict=False)
        if missing:
            new_modules = sorted(set(k.split('.')[0] for k in missing))
            logging.info(f"New heads initialized randomly (not in checkpoint): {new_modules}")
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if checkpoint.get('norm_mean') is not None:
            self.norm_mean = checkpoint['norm_mean'].to(self.device)
            self.norm_std = checkpoint['norm_std'].to(self.device)
        # Re-sync mean teacher from loaded student weights
        self.mean_teacher = MeanTeacher(self.model).to(self.device)
        logging.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('epoch', 0)


# ===================== MAIN TRAINING PIPELINE =====================

def run_training(
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    device: str = 'cuda'
):
    """Run initial training."""
    logging.info("=" * 60)
    logging.info("PHASE 1: INITIAL TRAINING")
    logging.info("=" * 60)

    # Load all data sources
    corrections = load_corrections()
    stems_data = load_stems_classified()
    manifest = load_manifest()

    # Build training data
    labeled_samples, unlabeled_samples = build_training_data(
        corrections, stems_data, manifest, GROUP_CLASSES
    )

    if len(labeled_samples) < 100:
        logging.error("Not enough labeled samples!")
        return None

    # Split into train/val
    random.shuffle(labeled_samples)
    split_idx = int(len(labeled_samples) * 0.9)
    train_samples = labeled_samples[:split_idx]
    val_samples = labeled_samples[split_idx:]

    logging.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Create datasets
    train_dataset = UnifiedDataset(train_samples, GROUP_CLASSES)
    val_dataset = UnifiedDataset(val_samples, GROUP_CLASSES)

    # Create weighted sampler for class balance
    class_counts = Counter(s.get('group', 'unknown') for s in train_samples if 'group' in s)
    sample_weights = []
    for s in train_samples:
        if 'group' in s:
            w = 1.0 / (class_counts.get(s['group'], 1) + 1)
            w *= s.get('weight', 1.0)
        else:
            w = 1.0
        sample_weights.append(w)

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    # Create model and trainer with full hierarchy
    model = UnifiedClassifier(
        group_classes=GROUP_CLASSES,
        subgroup_map=SUBGROUP_MAP,
        hierarchy=HIERARCHY
    )
    trainer = UnifiedTrainer(model, device=device)

    # Compute fresh normalization from training data (more accurate than old classifier's stats)
    trainer.compute_normalization(train_loader)

    # Set up scheduler
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=epochs
    )

    # Training loop
    best_acc = 0
    best_epoch = 0

    for epoch in range(epochs):
        losses = trainer.train_epoch(train_loader, epoch)
        metrics = trainer.evaluate(val_loader)

        # Build log message with optional subgroup/technique losses
        log_parts = [
            f"Epoch {epoch+1}/{epochs}",
            f"Loss: {losses['total']:.4f}",
            f"Grp: {losses.get('group', 0):.4f}",
        ]
        if 'subgroup' in losses:
            log_parts.append(f"Sub: {losses['subgroup']:.4f}")
        if 'technique' in losses:
            log_parts.append(f"Tech: {losses['technique']:.4f}")
        log_parts.extend([
            f"Val: {metrics['group_accuracy']:.3f}",
            f"ML-F1: {metrics.get('multilabel_f1', 0):.3f}"
        ])
        logging.info(" | ".join(log_parts))

        # Save best model
        if metrics['group_accuracy'] > best_acc:
            best_acc = metrics['group_accuracy']
            best_epoch = epoch
            trainer.save_checkpoint(
                OUTPUT_DIR / 'unified_model_best.pt',
                epoch,
                metrics
            )

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                OUTPUT_DIR / f'unified_model_e{epoch+1}.pt',
                epoch,
                metrics
            )

    logging.info(f"Best validation accuracy: {best_acc:.3f} at epoch {best_epoch+1}")

    # Save final model
    trainer.save_checkpoint(OUTPUT_DIR / 'unified_model_final.pt', epochs, metrics)

    return trainer, unlabeled_samples


def run_refinement(
    trainer: UnifiedTrainer,
    unlabeled_samples: List[Dict],
    rounds: int = REFINEMENT_ROUNDS,
    epochs_per_round: int = 10,
    batch_size: int = BATCH_SIZE,
    start_round: int = 0,
    initial_threshold: float = PSEUDO_LABEL_THRESHOLD,
):
    """Run iterative self-refinement using pseudo-labels."""
    logging.info("=" * 60)
    logging.info("PHASE 2: SELF-REFINEMENT")
    logging.info("=" * 60)

    # Load best model
    checkpoint_path = OUTPUT_DIR / 'unified_model_best.pt'
    if checkpoint_path.exists():
        trainer.load_checkpoint(checkpoint_path)

    # Reload base training data
    corrections = load_corrections()
    stems_data = load_stems_classified()
    manifest = load_manifest()

    labeled_samples, _ = build_training_data(
        corrections, stems_data, manifest, GROUP_CLASSES
    )

    # Create a HELD-OUT gold validation set from corrections only
    # This prevents overfitting to noisy pseudo-labels from being masked
    gold_val_samples = []
    correction_paths = set()
    for path, corr in corrections.items():
        group = corr.get('group')
        if not group:
            continue
        # Skip mix/ensemble labels (not single-instrument GT)
        if group in ('ensemble', 'mix', 'full-track') or group.endswith('_mix'):
            continue
        if group in EXCLUDED_CLASSES:
            continue
        # Strip _roomy suffix for group classification
        clean_group = group.replace('_roomy', '')
        if clean_group in GROUP_CLASSES:
            gold_val_samples.append({
                'path': path,
                'group': clean_group,
                'source': 'correction',
                'weight': 1.0,
                'is_mix': False,
            })
            correction_paths.add(path)

    # Shuffle and split corrections: 80% for training, 20% for held-out validation
    random.shuffle(gold_val_samples)
    gold_val_split = int(len(gold_val_samples) * 0.2)
    held_out_val = gold_val_samples[:gold_val_split]
    corrections_for_train = gold_val_samples[gold_val_split:]

    logging.info(f"Held-out gold validation set: {len(held_out_val)} corrections")

    # Create held-out validation dataloader (NEVER includes pseudo-labels)
    held_out_dataset = UnifiedDataset(held_out_val, GROUP_CLASSES)
    held_out_loader = DataLoader(
        held_out_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )

    # Remove held-out samples from labeled_samples
    held_out_paths = set(s['path'] for s in held_out_val)
    labeled_samples = [s for s in labeled_samples if s['path'] not in held_out_paths]

    current_threshold = initial_threshold
    best_gold_acc = 0

    # Resume from a previous round's saved state
    if start_round > 0:
        state_path = OUTPUT_DIR / f'refinement_state_r{start_round}.pt'
        if state_path.exists():
            state = torch.load(state_path, map_location='cpu', weights_only=False)
            remaining_paths = set(state['unlabeled_paths'])
            unlabeled_samples = [s for s in unlabeled_samples if s['path'] in remaining_paths]
            current_threshold = state['threshold']
            best_gold_acc = state.get('best_gold_acc', 0)
            logging.info(f"Resumed refinement state from {state_path}")
            logging.info(f"  Unlabeled pool: {len(unlabeled_samples)}, threshold: {current_threshold:.2f}, best_gold_acc: {best_gold_acc:.3f}")
        else:
            # No state file — compute threshold from start_round offset
            current_threshold = max(0.80, PSEUDO_LABEL_THRESHOLD - 0.02 * start_round)
            logging.warning(f"No state file at {state_path}, using computed threshold {current_threshold:.2f}")

    for round_num in range(start_round, rounds):
        logging.info(f"\n--- Refinement Round {round_num + 1}/{rounds} ---")
        logging.info(f"Pseudo-label threshold: {current_threshold:.2f}")

        # Generate pseudo-labels
        new_labels = trainer.generate_pseudo_labels(
            unlabeled_samples,
            threshold=current_threshold
        )

        logging.info(f"Generated {len(new_labels)} pseudo-labels")

        if len(new_labels) == 0:
            logging.info("No new pseudo-labels above threshold, stopping refinement")
            break

        # Confidence distribution
        confs = [l['confidence'] for l in new_labels]
        logging.info(f"  Confidence: min={min(confs):.3f}, max={max(confs):.3f}, mean={np.mean(confs):.3f}")

        # Class distribution
        class_dist = Counter(l['group'] for l in new_labels)
        logging.info(f"  Classes: {dict(class_dist.most_common(5))}")

        # Combine with original labeled data (excluding held-out)
        combined_samples = labeled_samples + new_labels
        random.shuffle(combined_samples)

        # Create training dataloader (no split needed, we have held-out gold val)
        train_dataset = UnifiedDataset(combined_samples, GROUP_CLASSES)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )

        # Train for a few epochs
        for epoch in range(epochs_per_round):
            losses = trainer.train_epoch(train_loader, epoch)

            # Evaluate on HELD-OUT gold set (not pseudo-labels!)
            gold_metrics = trainer.evaluate(held_out_loader)

            logging.info(
                f"  Epoch {epoch+1}/{epochs_per_round} | "
                f"Loss: {losses['total']:.4f} | "
                f"Gold Val Acc: {gold_metrics['group_accuracy']:.3f}"
            )

            # Track best gold accuracy
            if gold_metrics['group_accuracy'] > best_gold_acc:
                best_gold_acc = gold_metrics['group_accuracy']
                trainer.save_checkpoint(
                    OUTPUT_DIR / 'unified_model_refined_best.pt',
                    round_num,
                    gold_metrics
                )

        # Save checkpoint per round
        trainer.save_checkpoint(
            OUTPUT_DIR / f'unified_model_r{round_num+1}.pt',
            round_num,
            gold_metrics
        )

        # Remove used pseudo-labels from unlabeled pool
        pseudo_paths = set(l['path'] for l in new_labels)
        unlabeled_samples = [s for s in unlabeled_samples if s['path'] not in pseudo_paths]
        logging.info(f"Remaining unlabeled: {len(unlabeled_samples)}")

        # Gradually lower threshold
        current_threshold = max(0.80, current_threshold - 0.02)

        # Save refinement state for resume
        next_round = round_num + 1
        refinement_state = {
            'unlabeled_paths': [s['path'] for s in unlabeled_samples],
            'threshold': current_threshold,
            'best_gold_acc': best_gold_acc,
            'round_completed': round_num,
        }
        torch.save(refinement_state, OUTPUT_DIR / f'refinement_state_r{next_round}.pt')
        logging.info(f"Saved refinement state to refinement_state_r{next_round}.pt")

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save final refined model
    logging.info(f"Best gold validation accuracy during refinement: {best_gold_acc:.3f}")
    trainer.save_checkpoint(OUTPUT_DIR / 'unified_model_refined.pt', rounds, {'best_gold_accuracy': best_gold_acc})

    return trainer


def run_evaluation(trainer: UnifiedTrainer = None):
    """Evaluate the trained model."""
    logging.info("=" * 60)
    logging.info("PHASE 3: EVALUATION")
    logging.info("=" * 60)

    # Load model if not provided
    if trainer is None:
        model = UnifiedClassifier(group_classes=GROUP_CLASSES, subgroup_map=SUBGROUP_MAP, hierarchy=HIERARCHY)
        trainer = UnifiedTrainer(model)

        # Try to load best model
        for path in [
            OUTPUT_DIR / 'unified_model_refined.pt',
            OUTPUT_DIR / 'unified_model_best.pt',
            OUTPUT_DIR / 'unified_model_final.pt'
        ]:
            if path.exists():
                trainer.load_checkpoint(path)
                break

    # Load test data (corrections as gold standard)
    corrections = load_corrections()

    test_samples = []
    for path, corr in corrections.items():
        group = corr.get('group')
        if not group:
            continue
        if group in ('ensemble', 'mix', 'full-track') or group.endswith('_mix'):
            continue
        if group in EXCLUDED_CLASSES:
            continue
        clean_group = group.replace('_roomy', '')
        if clean_group in GROUP_CLASSES:
            test_samples.append({
                'path': path,
                'group': clean_group,
                'source': 'correction',
                'weight': 1.0,
                'is_mix': False,
            })

    logging.info(f"Evaluating on {len(test_samples)} correction samples")

    test_dataset = UnifiedDataset(test_samples, GROUP_CLASSES)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    metrics = trainer.evaluate(test_loader)

    logging.info(f"Group Accuracy: {metrics['group_accuracy']:.3f}")
    logging.info(f"Mix Detection Accuracy: {metrics['mix_accuracy']:.3f}")

    # Per-class accuracy
    trainer.model.eval()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(trainer.device)
            features = trainer.normalize(features)

            outputs = trainer.model(features)
            preds = outputs['group'].argmax(dim=1).cpu()

            has_group = batch['has_group'].bool()
            targets = batch['group']

            for i in range(len(preds)):
                if has_group[i] and targets[i] >= 0:
                    true_class = GROUP_CLASSES[targets[i]]
                    pred_class = GROUP_CLASSES[preds[i]]
                    class_total[true_class] += 1
                    if preds[i] == targets[i]:
                        class_correct[true_class] += 1

    logging.info("\nPer-class accuracy:")
    for cls in sorted(class_total.keys()):
        acc = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        logging.info(f"  {cls}: {acc:.3f} ({class_correct[cls]}/{class_total[cls]})")

    # Save evaluation results
    results = {
        'overall_accuracy': metrics['group_accuracy'],
        'mix_accuracy': metrics['mix_accuracy'],
        'per_class': {
            cls: {
                'accuracy': class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0,
                'correct': class_correct[cls],
                'total': class_total[cls]
            }
            for cls in class_total.keys()
        },
        'evaluated_at': datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / 'evaluation_results.json', 'wb') as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))

    return metrics


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description='Unified Self-Improving Classifier')
    parser.add_argument('--mode', choices=['train', 'refine', 'eval', 'full'], default='full',
                        help='Mode: train, refine, eval, or full (all steps)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Training epochs')
    parser.add_argument('--rounds', type=int, default=REFINEMENT_ROUNDS,
                        help='Refinement rounds')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--start-round', type=int, default=0,
                        help='Resume refinement at this round (0-indexed). Loads model from unified_model_r{N}.pt and restores unlabeled pool from refinement_state_r{N}.pt')
    parser.add_argument('--threshold', type=float, default=PSEUDO_LABEL_THRESHOLD,
                        help='Initial pseudo-label confidence threshold (default: 0.90)')

    args = parser.parse_args()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(OUTPUT_DIR)

    logging.info("=" * 60)
    logging.info("UNIFIED SELF-IMPROVING CLASSIFIER")
    logging.info("=" * 60)
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Refinement rounds: {args.rounds}")
    logging.info(f"Start round: {args.start_round}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Output: {OUTPUT_DIR}")
    logging.info(f"Log file: {log_file}")

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    try:
        if args.mode == 'train':
            trainer, unlabeled = run_training(
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device
            )

        elif args.mode == 'refine':
            # Load existing model
            model = UnifiedClassifier(group_classes=GROUP_CLASSES, subgroup_map=SUBGROUP_MAP, hierarchy=HIERARCHY)
            trainer = UnifiedTrainer(model, device=args.device)

            # Auto-pick checkpoint: if resuming at round N, load round N checkpoint
            if args.resume:
                checkpoint_path = Path(args.resume)
            elif args.start_round > 0:
                # Try refined_best first, then the round checkpoint
                refined_best = OUTPUT_DIR / 'unified_model_refined_best.pt'
                round_ckpt = OUTPUT_DIR / f'unified_model_r{args.start_round}.pt'
                checkpoint_path = refined_best if refined_best.exists() else round_ckpt
            else:
                checkpoint_path = OUTPUT_DIR / 'unified_model_best.pt'
            if Path(checkpoint_path).exists():
                trainer.load_checkpoint(Path(checkpoint_path))

            # Reload unlabeled samples
            manifest = load_manifest()
            corrections = load_corrections()
            _, unlabeled = build_training_data(
                corrections, {}, manifest, GROUP_CLASSES
            )

            run_refinement(
                trainer, unlabeled,
                rounds=args.rounds,
                batch_size=args.batch_size,
                start_round=args.start_round,
                initial_threshold=args.threshold,
            )

        elif args.mode == 'eval':
            run_evaluation()

        elif args.mode == 'full':
            # Full pipeline
            logging.info("\n" + "=" * 60)
            logging.info("STARTING FULL TRAINING PIPELINE")
            logging.info("=" * 60 + "\n")

            # Phase 1: Train
            trainer, unlabeled = run_training(
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device
            )

            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Phase 2: Refine
            if trainer and unlabeled:
                run_refinement(
                    trainer, unlabeled,
                    rounds=args.rounds,
                    batch_size=args.batch_size,
                    start_round=args.start_round,
                )

            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Phase 3: Evaluate
            run_evaluation(trainer)

        logging.info("\n" + "=" * 60)
        logging.info("TRAINING COMPLETE")
        logging.info("=" * 60)
        logging.info(f"Results saved to {OUTPUT_DIR}")

    except Exception as e:
        logging.exception(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
