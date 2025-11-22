"""
Data Module - Style Database & Data Pipeline
=============================================

This module contains:
- Agent 7: Musical Style Database (105+ styles with complete parameter sets)
- Agent 8: Data Pipeline & Preprocessing (MIDI datasets, caching, batching, metrics)

Authors: Agent 7 (Style Database), Agent 8 (Data Pipeline)
"""

# Agent 7: Style Database Components
from .style_database import StyleDatabase, StyleMetadata, STYLE_METADATA
from .style_generator import generate_all_styles, BASE_PARAMETERS_TEMPLATE

# Agent 8: Data Pipeline Components
from .midi_cache import MIDICache
from .dna_cache import DNACache
from .midi_reconstruction_dataset import MIDIReconstructionDataset
from .variable_length_collate import (
    variable_length_collate_fn,
    packed_sequence_collate_fn,
    create_attention_mask
)
from .reconstruction_metrics import MIDIReconstructionMetrics

__all__ = [
    # Agent 7: Style Database
    'StyleDatabase',
    'StyleMetadata',
    'STYLE_METADATA',
    'generate_all_styles',
    'BASE_PARAMETERS_TEMPLATE',

    # Agent 8: Data Pipeline
    'MIDICache',
    'DNACache',
    'MIDIReconstructionDataset',
    'variable_length_collate_fn',
    'packed_sequence_collate_fn',
    'create_attention_mask',
    'MIDIReconstructionMetrics',
]
