#!/usr/bin/env python3
"""
Ensemble Registry - Central Database of Musical Ensembles

Provides a unified registry for all ensemble types (big band, orchestra, chamber, etc.)
with their instrument configurations, ranges, and orchestration rules.

This enables the system to be extended to ANY musical ensemble without code changes -
just add a new ensemble configuration to the registry.

Author: Agent 19 - Genre Scalability Architect
Date: 2025-11-20
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class EnsembleType(Enum):
    """Types of musical ensembles"""
    BIG_BAND = "big_band"
    SYMPHONY_ORCHESTRA = "symphony_orchestra"
    CHAMBER_ORCHESTRA = "chamber_orchestra"
    STRING_QUARTET = "string_quartet"
    BRASS_QUINTET = "brass_quintet"
    WOODWIND_QUINTET = "woodwind_quintet"
    JAZZ_COMBO = "jazz_combo"
    ROCK_BAND = "rock_band"
    CHOIR = "choir"
    HINDUSTANI_CLASSICAL = "hindustani_classical"
    GAMELAN = "gamelan"
    MARIACHI = "mariachi"


@dataclass
class SectionConfig:
    """Configuration for a section within an ensemble"""
    name: str
    instruments: List[str]
    ranges: Dict[str, Tuple[int, int]]  # instrument_name -> (low_midi, high_midi)
    voicing_types: List[str] = field(default_factory=list)
    role: str = "harmony"  # melody, harmony, bass, rhythm, etc.
    max_voices: int = 4


@dataclass
class EnsembleConfig:
    """Complete configuration for a musical ensemble"""
    name: str
    ensemble_type: EnsembleType
    sections: Dict[str, SectionConfig]
    typical_styles: List[str]
    max_total_voices: int
    orchestration_style: str
    voice_leading_priority: str = "moderate"  # strict, moderate, loose
    cultural_origin: str = "western"


# ==============================================================================
# ENSEMBLE DEFINITIONS
# ==============================================================================

# Big Band Ensemble
BIG_BAND_ENSEMBLE = EnsembleConfig(
    name="Big Band",
    ensemble_type=EnsembleType.BIG_BAND,
    sections={
        "saxes": SectionConfig(
            name="Saxophone Section",
            instruments=["alto1", "alto2", "tenor1", "tenor2", "bari"],
            ranges={
                "alto": (52, 81),    # E3-A5
                "tenor": (47, 76),   # B2-E5
                "bari": (39, 69)     # Eb2-A4
            },
            voicing_types=["drop_2", "drop_3", "close", "spread"],
            role="melody_harmony",
            max_voices=5
        ),
        "brass": SectionConfig(
            name="Brass Section",
            instruments=["trumpet1", "trumpet2", "trumpet3", "trumpet4",
                        "trombone1", "trombone2", "trombone3", "trombone4"],
            ranges={
                "trumpet": (55, 82),   # G3-Bb5
                "trombone": (40, 72)   # E2-C5
            },
            voicing_types=["spread", "drop_2", "unison", "octaves"],
            role="harmony_accents",
            max_voices=8
        ),
        "rhythm": SectionConfig(
            name="Rhythm Section",
            instruments=["piano", "bass", "drums", "guitar"],
            ranges={
                "piano": (21, 108),    # Full piano range
                "bass": (28, 55),      # E1-G3
                "drums": (35, 81)      # Drum kit range
            },
            role="accompaniment",
            max_voices=4
        )
    },
    typical_styles=["swing", "bebop", "latin_jazz", "modern_jazz"],
    max_total_voices=17,
    orchestration_style="big_band",
    voice_leading_priority="moderate"
)


# Symphony Orchestra Ensemble
SYMPHONY_ORCHESTRA_ENSEMBLE = EnsembleConfig(
    name="Symphony Orchestra",
    ensemble_type=EnsembleType.SYMPHONY_ORCHESTRA,
    sections={
        "strings": SectionConfig(
            name="String Section",
            instruments=["violin1", "violin2", "viola", "cello", "bass"],
            ranges={
                "violin1": (55, 103),  # G3-G7
                "violin2": (55, 96),   # G3-C7
                "viola": (48, 91),     # C3-G6
                "cello": (36, 84),     # C2-C6
                "bass": (28, 67)       # E1-G4
            },
            voicing_types=["divisi", "tutti", "solo", "close", "spread"],
            role="foundation",
            max_voices=40
        ),
        "woodwinds": SectionConfig(
            name="Woodwind Section",
            instruments=["flute", "oboe", "clarinet", "bassoon"],
            ranges={
                "flute": (60, 96),     # C4-C7
                "oboe": (58, 91),      # Bb3-G6
                "clarinet": (50, 91),  # D3-G6 (written)
                "bassoon": (34, 72)    # Bb1-C5
            },
            voicing_types=["solo", "doubled", "tutti"],
            role="color_melody",
            max_voices=12
        ),
        "brass": SectionConfig(
            name="Brass Section",
            instruments=["horn1", "horn2", "horn3", "horn4",
                        "trumpet1", "trumpet2", "trumpet3",
                        "trombone1", "trombone2", "trombone3", "tuba"],
            ranges={
                "horn": (41, 77),      # F2-F5 (written)
                "trumpet": (55, 82),   # G3-Bb5
                "trombone": (40, 72),  # E2-C5
                "tuba": (28, 55)       # E1-G3
            },
            voicing_types=["spread", "close", "tutti"],
            role="power_harmony",
            max_voices=11
        ),
        "percussion": SectionConfig(
            name="Percussion Section",
            instruments=["timpani", "snare", "cymbals", "triangle", "xylophone"],
            ranges={
                "timpani": (40, 60),   # E2-C4
                "percussion": (50, 84)  # General percussion range
            },
            role="rhythm_color",
            max_voices=5
        )
    },
    typical_styles=["classical", "romantic", "impressionist", "modern", "film"],
    max_total_voices=80,
    orchestration_style="orchestral",
    voice_leading_priority="strict"
)


# String Quartet Ensemble
STRING_QUARTET_ENSEMBLE = EnsembleConfig(
    name="String Quartet",
    ensemble_type=EnsembleType.STRING_QUARTET,
    sections={
        "strings": SectionConfig(
            name="String Quartet",
            instruments=["violin1", "violin2", "viola", "cello"],
            ranges={
                "violin1": (55, 103),  # G3-G7
                "violin2": (55, 96),   # G3-C7
                "viola": (48, 91),     # C3-G6
                "cello": (36, 84)      # C2-C6
            },
            voicing_types=["close", "spread", "open"],
            role="complete_texture",
            max_voices=4
        )
    },
    typical_styles=["classical", "romantic", "contemporary", "modern"],
    max_total_voices=4,
    orchestration_style="chamber",
    voice_leading_priority="strict"
)


# Jazz Combo Ensemble
JAZZ_COMBO_ENSEMBLE = EnsembleConfig(
    name="Jazz Combo",
    ensemble_type=EnsembleType.JAZZ_COMBO,
    sections={
        "frontline": SectionConfig(
            name="Frontline",
            instruments=["saxophone", "trumpet"],
            ranges={
                "saxophone": (47, 76),  # Tenor sax range
                "trumpet": (55, 82)     # G3-Bb5
            },
            voicing_types=["unison", "harmony", "counterpoint"],
            role="melody",
            max_voices=2
        ),
        "rhythm": SectionConfig(
            name="Rhythm Section",
            instruments=["piano", "bass", "drums"],
            ranges={
                "piano": (21, 108),
                "bass": (28, 55),
                "drums": (35, 81)
            },
            role="accompaniment",
            max_voices=3
        )
    },
    typical_styles=["bebop", "hard_bop", "modal_jazz", "post_bop"],
    max_total_voices=5,
    orchestration_style="jazz_combo",
    voice_leading_priority="loose"
)


# Brass Quintet Ensemble
BRASS_QUINTET_ENSEMBLE = EnsembleConfig(
    name="Brass Quintet",
    ensemble_type=EnsembleType.BRASS_QUINTET,
    sections={
        "brass": SectionConfig(
            name="Brass Quintet",
            instruments=["trumpet1", "trumpet2", "horn", "trombone", "tuba"],
            ranges={
                "trumpet": (55, 82),   # G3-Bb5
                "horn": (41, 77),      # F2-F5
                "trombone": (40, 72),  # E2-C5
                "tuba": (28, 55)       # E1-G3
            },
            voicing_types=["close", "spread", "chorale"],
            role="complete_texture",
            max_voices=5
        )
    },
    typical_styles=["classical", "romantic", "modern", "jazz"],
    max_total_voices=5,
    orchestration_style="chamber",
    voice_leading_priority="strict"
)


# Hindustani Classical Ensemble
HINDUSTANI_ENSEMBLE = EnsembleConfig(
    name="Hindustani Classical Ensemble",
    ensemble_type=EnsembleType.HINDUSTANI_CLASSICAL,
    sections={
        "melody": SectionConfig(
            name="Melodic Instruments",
            instruments=["sitar", "bansuri", "sarangi", "vocals"],
            ranges={
                "sitar": (48, 84),      # C3-C6
                "bansuri": (60, 84),    # C4-C6
                "sarangi": (52, 79),    # E3-G5
                "vocals": (55, 84)      # G3-C6
            },
            voicing_types=["monophonic"],  # Indian classical is primarily monophonic
            role="melodic_improvisation",
            max_voices=1  # Typically one melody at a time
        ),
        "drone": SectionConfig(
            name="Drone",
            instruments=["tanpura"],
            ranges={
                "tanpura": (36, 55)  # C2-G3
            },
            role="harmonic_foundation",
            max_voices=4  # Tanpura has 4 strings
        ),
        "rhythm": SectionConfig(
            name="Percussion",
            instruments=["tabla", "pakhawaj"],
            ranges={
                "tabla": (50, 70)  # Approximate tabla range
            },
            role="rhythmic_cycle",
            max_voices=2
        )
    },
    typical_styles=["hindustani", "dhrupad", "khayal", "thumri"],
    max_total_voices=7,
    orchestration_style="modal_monophonic",
    voice_leading_priority="loose",
    cultural_origin="indian"
)


# Rock Band Ensemble
ROCK_BAND_ENSEMBLE = EnsembleConfig(
    name="Rock Band",
    ensemble_type=EnsembleType.ROCK_BAND,
    sections={
        "guitars": SectionConfig(
            name="Guitars",
            instruments=["lead_guitar", "rhythm_guitar"],
            ranges={
                "guitar": (40, 88)  # E2-E6
            },
            voicing_types=["power_chords", "open_voicing", "solo"],
            role="melody_harmony",
            max_voices=2
        ),
        "rhythm": SectionConfig(
            name="Rhythm Section",
            instruments=["bass", "drums", "keyboards"],
            ranges={
                "bass": (28, 55),      # E1-G3
                "drums": (35, 81),     # Drum kit
                "keyboards": (21, 108) # Full keyboard
            },
            role="accompaniment",
            max_voices=3
        ),
        "vocals": SectionConfig(
            name="Vocals",
            instruments=["lead_vocals", "backing_vocals"],
            ranges={
                "vocals": (55, 84)  # G3-C6
            },
            role="melody",
            max_voices=3
        )
    },
    typical_styles=["rock", "hard_rock", "progressive_rock", "alternative"],
    max_total_voices=8,
    orchestration_style="rock_band",
    voice_leading_priority="loose"
)


# ==============================================================================
# ENSEMBLE REGISTRY
# ==============================================================================

ENSEMBLE_REGISTRY: Dict[str, EnsembleConfig] = {
    "big_band": BIG_BAND_ENSEMBLE,
    "symphony_orchestra": SYMPHONY_ORCHESTRA_ENSEMBLE,
    "chamber_orchestra": SYMPHONY_ORCHESTRA_ENSEMBLE,  # Use symphony with fewer players
    "string_quartet": STRING_QUARTET_ENSEMBLE,
    "brass_quintet": BRASS_QUINTET_ENSEMBLE,
    "jazz_combo": JAZZ_COMBO_ENSEMBLE,
    "hindustani": HINDUSTANI_ENSEMBLE,
    "rock_band": ROCK_BAND_ENSEMBLE,
}


# ==============================================================================
# REGISTRY FUNCTIONS
# ==============================================================================

def get_ensemble(ensemble_type: str) -> Optional[EnsembleConfig]:
    """
    Get ensemble configuration by type.

    Args:
        ensemble_type: Name of ensemble (e.g., "big_band", "string_quartet")

    Returns:
        EnsembleConfig if found, None otherwise
    """
    return ENSEMBLE_REGISTRY.get(ensemble_type)


def register_ensemble(ensemble_type: str, config: EnsembleConfig):
    """
    Register a new ensemble configuration.

    This allows users to add custom ensembles at runtime.

    Args:
        ensemble_type: Unique identifier for this ensemble
        config: EnsembleConfig object
    """
    ENSEMBLE_REGISTRY[ensemble_type] = config


def list_ensembles() -> List[str]:
    """List all registered ensemble types."""
    return list(ENSEMBLE_REGISTRY.keys())


def get_ensemble_sections(ensemble_type: str) -> Dict[str, SectionConfig]:
    """Get sections for an ensemble."""
    ensemble = get_ensemble(ensemble_type)
    return ensemble.sections if ensemble else {}


def get_ensemble_instruments(ensemble_type: str) -> List[str]:
    """Get all instruments in an ensemble."""
    ensemble = get_ensemble(ensemble_type)
    if not ensemble:
        return []

    instruments = []
    for section in ensemble.sections.values():
        instruments.extend(section.instruments)
    return instruments


def get_section_by_role(ensemble_type: str, role: str) -> Optional[SectionConfig]:
    """
    Find section by role (e.g., "melody", "harmony", "bass").

    Args:
        ensemble_type: Name of ensemble
        role: Role to search for

    Returns:
        First section matching the role, or None
    """
    ensemble = get_ensemble(ensemble_type)
    if not ensemble:
        return None

    for section in ensemble.sections.values():
        if role.lower() in section.role.lower():
            return section
    return None


if __name__ == "__main__":
    # Test the registry
    print("Registered Ensembles:")
    for name in list_ensembles():
        ensemble = get_ensemble(name)
        print(f"\n{ensemble.name}:")
        print(f"  Type: {ensemble.ensemble_type.value}")
        print(f"  Max Voices: {ensemble.max_total_voices}")
        print(f"  Sections: {', '.join(ensemble.sections.keys())}")
        print(f"  Styles: {', '.join(ensemble.typical_styles)}")
