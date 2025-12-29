"""
Pitch Shift V8 - Register Timbre Correction with Distribution Matching

V8 improvements over V2:
1. residual_scale increased to 0.3 (was 0.1) - stronger transformation
2. content_loss now used for residual model (was only direct) - better pitch preservation
3. silence_weight increased to 5.0 (was 1.0) - less noise in silent sections
"""

from .models_v8 import RegisterTranslator, RegisterTranslatorDirect
from .dataset_v8 import RegisterTransferDatasetV8

__all__ = ['RegisterTranslator', 'RegisterTranslatorDirect', 'RegisterTransferDatasetV8']
