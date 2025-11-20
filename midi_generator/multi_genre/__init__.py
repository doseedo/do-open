"""
Multi-Genre Data Specialist Module
Agent 07: Handles genre-specific data requirements and balancing

This module provides:
- Genre stratification for train/val/test splits
- Multi-genre data augmentation with genre-specific rules
- Genre balancing and imbalance handling
- Cross-genre transfer learning utilities
- Genre-specific validation

Author: Agent 07
Date: November 20, 2025
Version: 1.0
"""

from .genre_stratifier import GenreStratifier
from .augmentation import (
    MIDIAugmentation,
    PitchTransposition,
    TempoScaling,
    VelocityPerturbation,
    TimingJitter,
    HarmonicSubstitution,
    VoicePermutation,
    GenreAugmentationPipeline
)
from .genre_balancer import GenreBalancer
from .cross_genre_transfer import CrossGenreTransfer
from .validation import (
    AugmentationValidator,
    GenreValidationSplitter,
    GenreDataStatistics
)

__all__ = [
    'GenreStratifier',
    'MIDIAugmentation',
    'PitchTransposition',
    'TempoScaling',
    'VelocityPerturbation',
    'TimingJitter',
    'HarmonicSubstitution',
    'VoicePermutation',
    'GenreAugmentationPipeline',
    'GenreBalancer',
    'CrossGenreTransfer',
    'AugmentationValidator',
    'GenreValidationSplitter',
    'GenreDataStatistics'
]

__version__ = '1.0.0'
__author__ = 'Agent 07'
