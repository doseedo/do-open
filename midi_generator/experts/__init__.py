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
- Agent 18: Harmony Specialist (jazz voicings, modal harmony, voice leading)
- Agent 19: Melody Specialist (motif development, sequences, ornamentation)
- Agent 23: Structure Specialist (form, transitions, climax, motifs)

Author: Musical Program Synthesis Team
License: MIT
"""

__version__ = "1.0.0"

# Import key classes and functions
from .harmony_specialist import (
    HarmonySpecialist,
    Chord,
    ChordProgression,
    Note,
    VoicingType,
    ChordQuality,
    Mode,
    HarmonicFunction,
    ReharmonizationTechnique,
    VoiceLeadingRule,
    HarmonyFeatures,
    VoicingAnalysis,
    VoiceLeadingAnalysis,
    analyze_harmony,
    generate_jazz_voicing,
)

from .melody_specialist import (
    MelodySpecialist,
    MotifTransformation,
    SequenceType,
    OrnamentType,
    ContourShape,
    Motif,
    Phrase,
    MelodicAnalysis,
    create_example_melody,
)

__all__ = [
    # Modules
    "melody_specialist",
    "structure_specialist",
    "harmony_specialist",

    # Harmony Specialist Classes
    "HarmonySpecialist",
    "Chord",
    "ChordProgression",
    "Note",
    "VoicingType",
    "ChordQuality",
    "Mode",
    "HarmonicFunction",
    "ReharmonizationTechnique",
    "VoiceLeadingRule",
    "HarmonyFeatures",
    "VoicingAnalysis",
    "VoiceLeadingAnalysis",

    # Harmony Specialist Functions
    "analyze_harmony",
    "generate_jazz_voicing",

    # Melody Specialist Classes
    "MelodySpecialist",
    "MotifTransformation",
    "SequenceType",
    "OrnamentType",
    "ContourShape",
    "Motif",
    "Phrase",
    "MelodicAnalysis",

    # Melody Specialist Functions
    "create_example_melody",
]
