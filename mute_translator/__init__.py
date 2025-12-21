# Mute Translator Pipeline
# Convert dry trumpet to muted trumpet using latent space translation

from .models import MuteTranslator, MuteTranslatorLarge, CycleConsistentMuteTranslator
from .dataset import MuteTranslatorDataset, load_manifest

__all__ = [
    'MuteTranslator',
    'MuteTranslatorLarge',
    'CycleConsistentMuteTranslator',
    'MuteTranslatorDataset',
    'load_manifest',
]
