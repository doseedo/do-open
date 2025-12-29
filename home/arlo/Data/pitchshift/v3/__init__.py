"""
Register-Aware Pitch Shift V2

Key innovation: Register Codebook approach (like mute translator).
- Build register centroids from real recordings
- Create synthetic targets using centroid offsets
- Clean training signal (no noisy reference matching)
"""

from .models_v2 import RegisterTranslator, RegisterTranslatorDirect, CombinedLossV2
from .dataset_v2 import RegisterTransferDatasetV2, RegisterCodebook
