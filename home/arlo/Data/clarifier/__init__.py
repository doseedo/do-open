# /home/arlo/Data/clarifier/__init__.py
# Post-generation latent clarifier for performer model outputs

from .models import (
    InstrumentClarifier,
    InstrumentClarifierLarge,
    SimpleInstrumentClassifier,
    GroupClarifier,
    GroupSubgroupClassifier,
    create_group_clarifier,
    create_group_classifier,
)
from .dataset import ClarifierPairDataset, ClarifierManifestDataset, collate_clarifier
from .config import (
    get_group_config,
    GROUP_CONFIGS,
    BRASS_CONFIG,
    STRINGS_CONFIG,
    WINDS_CONFIG,
    PIANO_CONFIG,
    GUITAR_CONFIG,
    BASS_CONFIG,
)

__all__ = [
    # Full clarifier (all groups)
    "InstrumentClarifier",
    "InstrumentClarifierLarge",
    "SimpleInstrumentClassifier",
    # Group-specific clarifier
    "GroupClarifier",
    "GroupSubgroupClassifier",
    "create_group_clarifier",
    "create_group_classifier",
    # Datasets
    "ClarifierPairDataset",
    "ClarifierManifestDataset",
    "collate_clarifier",
    # Configs
    "get_group_config",
    "GROUP_CONFIGS",
    "BRASS_CONFIG",
    "STRINGS_CONFIG",
    "WINDS_CONFIG",
    "PIANO_CONFIG",
    "GUITAR_CONFIG",
    "BASS_CONFIG",
]
