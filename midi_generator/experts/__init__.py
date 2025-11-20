"""
Expert Modules for Musical Program Synthesis
==============================================

This package contains specialized expert modules for deep feature extraction
and analysis in the self-expanding inverse music generation system.

Each expert focuses on a specific musical domain and provides:
1. Feature extraction from MIDI files
2. Parameter analysis and prediction
3. Musical pattern recognition
4. Training data generation for XGBoost models

Experts:
- Agent 22: Dynamics Specialist (ADSR, curves, humanization, voice balancing)
- Agent 23: Structure Specialist (form, transitions, climax, motifs)
- Agent 24: Texture Specialist (polyphony, layering, voice independence)

Author: Musical Program Synthesis Team
License: MIT
"""

__version__ = "1.0.0"

__all__ = [
    "dynamics_specialist",
    "structure_specialist",
    "texture_specialist",
]
