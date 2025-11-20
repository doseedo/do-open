# Genre Scalability Architecture
## Agent 19: Comprehensive Design Document

**Author**: Agent 19 - Genre Scalability Architect
**Date**: 2025-11-20
**Purpose**: Design system architecture to scale beyond big band to any genre without rewriting code

---

## Executive Summary

This document defines the architecture that makes the MIDI Generator system **universally scalable** across all musical genres - from big band to orchestra, chamber music, world music, electronic, and beyond. The key insight is that **the same fundamental principles apply across all genres**: voice leading, phrasing, harmony, rhythm, and orchestration are universal concepts that can be parameterized and configured per genre.

### Key Architectural Principles

1. **Separation of Concerns**: Universal components vs. genre-specific profiles
2. **Dependency Injection**: Components don't hardcode genre assumptions
3. **Configuration over Code**: New genres added via configuration, not new classes
4. **Composition over Inheritance**: Mix and match components across genres
5. **Factory Pattern**: Centralized component creation with genre awareness

---

## Component Classification Matrix

### UNIVERSAL COMPONENTS (Genre-Agnostic)
These components work across ALL genres with no modification:

| Component | Module | Purpose | Scalability |
|-----------|--------|---------|-------------|
| **VoiceLeadingOptimizer** | `transformation/voice_leading_optimizer.py` | Minimize voice movement between chords | Works for: strings, brass, choir, synth pads, ANY harmony |
| **DynamicShaping** | `transformation/dynamic_shaping.py` | Apply crescendo, diminuendo, phrasing | Works for: ANY musical expression |
| **FormGenerator** | `generators/form_generator.py` | Generate musical forms (AABA, sonata, etc.) | Works for: jazz, classical, pop, world music |
| **HumanizationEngine** | `algorithms/rhythm_engine.py` | Timing/velocity variation | Works for: ANY performed music |
| **ArticulationEngine** | `midi/articulation_engine.py` | Apply articulations (needs extension) | Extensible to: strings (pizz, arco), brass (mute), winds |
| **GrooveTemplateEngine** | `algorithms/groove_library.py` | Extract and apply groove timing | Works for: ANY rhythmic music |
| **ComponentSystem** | `core/component_system.py` | Modular component architecture | **Universal framework** |
| **InstrumentLibrary** | `core/instrument_library.py` | Instrument specifications | **Universal database** |
| **Orchestrator** | `generators/orchestrator.py` | Intelligent orchestration | **Multi-style support** |
| **MultiGenreArranger** | `core/multi_genre_arranger.py` | Track-level genre control | **Cross-genre fusion** |

### GENRE-SPECIFIC COMPONENTS
These components need different implementations per genre:

| Component Type | Current Implementations | Scalability Approach |
|----------------|------------------------|----------------------|
| **Melody Generators** | `genres/jazz.py:BebopMelodyGenerator` | Create `genres/classical.py:ClassicalMelodyGenerator`, etc. |
| **Harmony Generators** | `genres/jazz.py:JazzHarmonyGenerator` | Create `genres/indian.py:RagaHarmonyGenerator`, etc. |
| **Rhythm Generators** | `genres/funk_soul.py:FunkDrumGenerator` | Create per-genre rhythm engines |
| **Voicing Engines** | `transformation/arrangement_engine.py:BigBandArranger` | Create `OrchestraArranger`, `ChoirArranger`, etc. |
| **Style Profiles** | `styles/ellington_profile.py` (to be created) | Create per-composer/style profiles |

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│  USER API LAYER                                             │
│  - UnifiedAPI (api/unified_api.py)                         │
│  - Simple methods: quick_fusion(), generate_arrangement()  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ENSEMBLE & STYLE LAYER (NEW - AGENT 19)                   │
│  - EnsembleRegistry: Register ensembles (big band, orch)   │
│  - StyleProfileRegistry: Register styles (Basie, Mozart)   │
│  - GenreConfigLoader: Load genre configurations            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ABSTRACTION LAYER                                          │
│  - GenericArranger (base class for all arrangers)          │
│  - GenericMelodyGenerator (base for melody generators)     │
│  - GenericHarmonyGenerator (base for harmony generators)   │
│  - ComponentFactory (creates genre-specific instances)     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  UNIVERSAL ENGINES (Genre-Agnostic)                         │
│  - VoiceLeadingOptimizer                                    │
│  - DynamicShaping                                           │
│  - HumanizationEngine                                       │
│  - FormGenerator                                            │
│  - Orchestrator                                             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  GENRE-SPECIFIC IMPLEMENTATIONS                             │
│  Jazz: BebopMelodyGenerator, JazzHarmonyGenerator          │
│  Classical: SonataFormGenerator, OrchestralVoicing         │
│  Indian: RagaMelodyGenerator, TalaRhythmGenerator          │
│  Electronic: SynthPadGenerator, EDMGrooveGenerator         │
└─────────────────────────────────────────────────────────────┘
```

---

## Ensemble Definition System

### Configuration Format

Each ensemble is defined as a configuration that specifies:
- Sections (groups of instruments)
- Instruments per section
- Ranges (comfortable playing ranges)
- Voicing preferences
- Orchestration rules

### Example: Big Band Ensemble

```python
BIG_BAND_ENSEMBLE = {
    "name": "Big Band",
    "sections": {
        "saxes": {
            "instruments": ["alto1", "alto2", "tenor1", "tenor2", "bari"],
            "ranges": {
                "alto": (52, 81),    # E3-A5
                "tenor": (47, 76),   # B2-E5
                "bari": (39, 69)     # Eb2-A4
            },
            "voicing_types": ["drop_2", "drop_3", "close"],
            "role": "melody_harmony"
        },
        "brass": {
            "instruments": ["trumpet1", "trumpet2", "trumpet3", "trumpet4",
                          "trombone1", "trombone2", "trombone3", "trombone4"],
            "ranges": {
                "trumpet": (55, 82),   # G3-Bb5
                "trombone": (40, 72)   # E2-C5
            },
            "voicing_types": ["spread", "drop_2", "unison"],
            "role": "harmony_accents"
        },
        "rhythm": {
            "instruments": ["piano", "bass", "drums", "guitar"],
            "role": "accompaniment"
        }
    },
    "typical_styles": ["swing", "bebop", "latin_jazz"],
    "max_voices": 17,
    "orchestration_style": "big_band"
}
```

### Example: Symphony Orchestra Ensemble

```python
SYMPHONY_ORCHESTRA_ENSEMBLE = {
    "name": "Symphony Orchestra",
    "sections": {
        "strings": {
            "instruments": ["violin1", "violin2", "viola", "cello", "bass"],
            "ranges": {
                "violin": (55, 103),   # G3-G7
                "viola": (48, 91),     # C3-G6
                "cello": (36, 84),     # C2-C6
                "bass": (28, 67)       # E1-G4
            },
            "voicing_types": ["divisi", "tutti", "solo"],
            "role": "foundation"
        },
        "woodwinds": {
            "instruments": ["flute", "oboe", "clarinet", "bassoon"],
            "ranges": {
                "flute": (60, 96),     # C4-C7
                "oboe": (58, 91),      # Bb3-G6
                "clarinet": (50, 91),  # D3-G6 (written)
                "bassoon": (34, 72)    # Bb1-C5
            },
            "role": "color_melody"
        },
        "brass": {
            "instruments": ["horn", "trumpet", "trombone", "tuba"],
            "ranges": {
                "horn": (41, 77),      # F2-F5 (written)
                "trumpet": (55, 82),   # G3-Bb5
                "trombone": (40, 72),  # E2-C5
                "tuba": (28, 55)       # E1-G3
            },
            "role": "power_harmony"
        },
        "percussion": {
            "instruments": ["timpani", "snare", "cymbals", "triangle"],
            "role": "rhythm_color"
        }
    },
    "typical_styles": ["classical", "romantic", "impressionist", "modern", "film"],
    "max_voices": 80,  # Full orchestra
    "orchestration_style": "orchestral"
}
```

### Example: String Quartet Ensemble

```python
STRING_QUARTET_ENSEMBLE = {
    "name": "String Quartet",
    "sections": {
        "strings": {
            "instruments": ["violin1", "violin2", "viola", "cello"],
            "ranges": {
                "violin1": (55, 103),  # G3-G7 (first violin higher tessitura)
                "violin2": (55, 96),   # G3-C7 (second violin)
                "viola": (48, 91),     # C3-G6
                "cello": (36, 84)      # C2-C6
            },
            "voicing_types": ["close", "spread", "open"],
            "role": "complete_texture"
        }
    },
    "typical_styles": ["classical", "romantic", "contemporary"],
    "max_voices": 4,
    "orchestration_style": "chamber",
    "voice_leading_priority": "STRICT"  # Classical voice leading rules
}
```

---

## Style Profile System

### Configuration Format

Style profiles define composer/performer-specific characteristics:

```python
@dataclass
class StyleProfile:
    """Configuration for a specific musical style"""
    name: str
    composer_era: str  # "baroque", "classical", "romantic", "modern", "jazz_swing", etc.

    # Orchestration preferences
    voicing_preference: str  # "close", "spread", "mixed"
    voicing_spacing: str     # "tight", "medium", "wide"
    doubling_rules: Dict[str, float]  # Instrument doubling probabilities

    # Harmonic characteristics
    harmony_complexity: float  # 0.0-1.0
    chord_extensions: List[int]  # [7, 9, 11, 13]
    chromaticism: float  # 0.0-1.0

    # Articulation characteristics
    articulation_variety: float  # 0.0-1.0
    articulation_probabilities: Dict[str, float]

    # Dynamic characteristics
    dynamic_range: str  # "narrow", "medium", "wide", "very_wide"
    use_crescendo: float  # 0.0-1.0

    # Form preferences
    intro_style: str  # "fanfare", "rubato", "vamp", "quiet", etc.
    ending_style: str  # "fade", "fermata", "tag", "abrupt", etc.

    # Texture characteristics
    texture_density: float  # 0.0-1.0 (sparse to dense)
    texture_variation: float  # 0.0-1.0 (static to highly varied)
```

### Example: Count Basie Style Profile

```python
BASIE_STYLE = StyleProfile(
    name="Count Basie",
    composer_era="jazz_swing",

    # Orchestration
    voicing_preference="unison_and_octaves",
    voicing_spacing="open",
    doubling_rules={
        "saxes_unison": 0.6,
        "brass_octaves": 0.5,
        "section_hits": 0.9
    },

    # Harmony
    harmony_complexity=0.3,  # Simple, functional
    chord_extensions=[7],    # Basic 7th chords
    chromaticism=0.2,

    # Articulation
    articulation_variety=0.4,
    articulation_probabilities={
        "staccato": 0.7,
        "accent": 0.5,
        "fall_short": 0.3
    },

    # Dynamics
    dynamic_range="medium",
    use_crescendo=0.3,

    # Form
    intro_style="vamp",
    ending_style="button",

    # Texture
    texture_density=0.5,  # Sparser than Ellington
    texture_variation=0.4
)
```

### Example: Wolfgang Amadeus Mozart Style Profile

```python
MOZART_STYLE = StyleProfile(
    name="Mozart",
    composer_era="classical",

    # Orchestration
    voicing_preference="balanced",
    voicing_spacing="medium",
    doubling_rules={
        "strings_tutti": 0.7,
        "winds_solo_vs_ensemble": 0.6,
        "horn_doubling_bassoon": 0.4
    },

    # Harmony
    harmony_complexity=0.4,  # Elegant simplicity
    chord_extensions=[],     # Triads and 7th chords, no extensions
    chromaticism=0.3,

    # Articulation
    articulation_variety=0.6,
    articulation_probabilities={
        "staccato": 0.5,
        "legato": 0.4,
        "tenuto": 0.2
    },

    # Dynamics
    dynamic_range="medium",
    use_crescendo=0.4,

    # Form
    intro_style="fanfare",
    ending_style="authentic_cadence",

    # Texture
    texture_density=0.6,  # Transparent, clear
    texture_variation=0.7  # Varied textures
)
```

---

## Generic Arranger Base Class

All genre-specific arrangers inherit from this base class:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass

class GenericArranger(ABC):
    """
    Base class for all arrangers (BigBand, Orchestra, Choir, etc.)

    Provides universal arranging pipeline while allowing genre-specific customization.
    """

    def __init__(self, ensemble_config: Dict, style_profile: Optional[StyleProfile] = None):
        """
        Initialize arranger with ensemble configuration and style profile.

        Args:
            ensemble_config: Ensemble definition (instruments, ranges, sections)
            style_profile: Optional style profile (Basie, Mozart, etc.)
        """
        self.ensemble = ensemble_config
        self.style = style_profile
        self.voice_leading_optimizer = VoiceLeadingOptimizer()
        self.dynamic_shaper = DynamicShaping()
        self.humanizer = HumanizationEngine()

    def arrange(self,
                melody: List[NoteEvent],
                harmony: List[ChordEvent],
                form: Optional[MusicalForm] = None) -> Dict[str, List[NoteEvent]]:
        """
        Main arranging pipeline (TEMPLATE METHOD PATTERN).

        This method defines the universal arranging process.
        Subclasses override specific steps.
        """
        arrangement = {}

        # Step 1: Prepare form structure
        if form is None:
            form = self._default_form(melody, harmony)

        # Step 2: Create lead/melody line
        arrangement['lead'] = self._arrange_melody(melody, form)

        # Step 3: Create harmonic voicings
        arrangement['harmony'] = self._arrange_harmony(harmony, melody, form)

        # Step 4: Create bass line
        arrangement['bass'] = self._arrange_bass(harmony, form)

        # Step 5: Create rhythm section
        arrangement['rhythm'] = self._arrange_rhythm(harmony, form)

        # Step 6: Apply voice leading optimization (UNIVERSAL)
        arrangement = self._optimize_voice_leading(arrangement)

        # Step 7: Apply dynamics and phrasing (UNIVERSAL)
        arrangement = self._apply_dynamics(arrangement, form)

        # Step 8: Apply articulations (genre-specific)
        arrangement = self._apply_articulations(arrangement)

        # Step 9: Apply humanization (UNIVERSAL)
        arrangement = self._apply_humanization(arrangement)

        # Step 10: Create intro/outro if needed
        if self.style and self.style.intro_style:
            arrangement = self._add_intro_outro(arrangement, harmony, form)

        return arrangement

    # =========================================================================
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def _arrange_melody(self, melody: List[NoteEvent], form: MusicalForm) -> List[NoteEvent]:
        """Arrange melody for this ensemble (genre-specific)"""
        pass

    @abstractmethod
    def _arrange_harmony(self, harmony: List[ChordEvent],
                        melody: List[NoteEvent], form: MusicalForm) -> List[NoteEvent]:
        """Create harmonic voicings for this ensemble (genre-specific)"""
        pass

    @abstractmethod
    def _arrange_bass(self, harmony: List[ChordEvent], form: MusicalForm) -> List[NoteEvent]:
        """Create bass line for this ensemble (genre-specific)"""
        pass

    @abstractmethod
    def _arrange_rhythm(self, harmony: List[ChordEvent], form: MusicalForm) -> List[NoteEvent]:
        """Create rhythm section/percussion for this ensemble (genre-specific)"""
        pass

    @abstractmethod
    def _apply_articulations(self, arrangement: Dict) -> Dict:
        """Apply genre-specific articulations"""
        pass

    # =========================================================================
    # UNIVERSAL METHODS (Used by all arrangers)
    # =========================================================================

    def _optimize_voice_leading(self, arrangement: Dict) -> Dict:
        """Apply voice leading optimization (UNIVERSAL)"""
        if 'harmony' in arrangement:
            optimized = self.voice_leading_optimizer.optimize_chord_sequence(
                arrangement['harmony'],
                num_voices=self._get_num_harmony_voices(),
                voice_ranges=self._get_voice_ranges()
            )
            arrangement['harmony'] = optimized
        return arrangement

    def _apply_dynamics(self, arrangement: Dict, form: MusicalForm) -> Dict:
        """Apply dynamic shaping based on form (UNIVERSAL)"""
        dynamic_map = self.dynamic_shaper.generate_dynamic_map_for_form(form)

        for section_name, notes in arrangement.items():
            arrangement[section_name] = self.dynamic_shaper.apply_phrase_contour(
                notes,
                phrase_length_bars=4,
                contour="arch"
            )

        return arrangement

    def _apply_humanization(self, arrangement: Dict) -> Dict:
        """Apply humanization (timing/velocity variation) (UNIVERSAL)"""
        for section_name, notes in arrangement.items():
            arrangement[section_name] = self.humanizer.apply_timing_variation(
                notes,
                amount=0.02  # 2% variation
            )
        return arrangement

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_num_harmony_voices(self) -> int:
        """Get number of harmony voices from ensemble config"""
        # Count instruments in harmony sections
        count = 0
        for section in self.ensemble['sections'].values():
            if 'harmony' in section.get('role', ''):
                count += len(section['instruments'])
        return count

    def _get_voice_ranges(self) -> List[Tuple[int, int]]:
        """Get voice ranges from ensemble config"""
        ranges = []
        for section in self.ensemble['sections'].values():
            for inst_type, range_tuple in section.get('ranges', {}).items():
                ranges.append(range_tuple)
        return ranges

    def _default_form(self, melody: List[NoteEvent],
                     harmony: List[ChordEvent]) -> MusicalForm:
        """Create default form if none provided"""
        # Simple AABA form by default
        return FormGenerator.generate("aaba", key=0, tempo=120)
```

---

## How to Add a New Genre: Step-by-Step Guide

### Example: Adding Indian Classical Music (Raga)

#### Step 1: Define the Ensemble

```python
# File: core/ensembles/indian_classical.py

HINDUSTANI_ENSEMBLE = {
    "name": "Hindustani Classical Ensemble",
    "sections": {
        "melody": {
            "instruments": ["sitar", "bansuri", "sarangi", "vocals"],
            "ranges": {
                "sitar": (48, 84),      # C3-C6
                "bansuri": (60, 84),    # C4-C6
                "sarangi": (52, 79),    # E3-G5
                "vocals": (55, 84)      # G3-C6
            },
            "role": "melodic_improvisation"
        },
        "drone": {
            "instruments": ["tanpura"],
            "ranges": {"tanpura": (36, 55)},  # C2-G3
            "role": "harmonic_foundation"
        },
        "rhythm": {
            "instruments": ["tabla", "pakhawaj"],
            "role": "rhythmic_cycle"
        }
    },
    "typical_styles": ["hindustani", "dhrupad", "khayal"],
    "max_voices": 6,
    "orchestration_style": "modal_monophonic"
}
```

#### Step 2: Create Genre-Specific Generators

```python
# File: genres/indian_classical.py

class RagaMelodyGenerator:
    """Generate melodies based on raga rules"""

    def __init__(self, raga_name: str):
        self.raga = RAGA_LIBRARY[raga_name]  # Load raga definition
        self.ascent_notes = self.raga['aroha']  # Ascending pattern
        self.descent_notes = self.raga['avaroha']  # Descending pattern
        self.vadi = self.raga['vadi']  # Main note
        self.samvadi = self.raga['samvadi']  # Second important note

    def generate_alap(self, duration_bars: int) -> List[NoteEvent]:
        """Generate alap (slow, unmetered introduction)"""
        # Implement raga exploration logic
        pass

    def generate_gat(self, tala: str, duration_bars: int) -> List[NoteEvent]:
        """Generate gat (composed melody in rhythm)"""
        # Implement raga melody with tala (rhythmic cycle)
        pass


class TalaRhythmGenerator:
    """Generate tabla patterns based on tala (rhythmic cycle)"""

    def __init__(self, tala_name: str):
        self.tala = TALA_LIBRARY[tala_name]  # Load tala definition
        self.matras = self.tala['matras']  # Number of beats
        self.vibhag = self.tala['vibhag']  # Beat divisions

    def generate_theka(self) -> List[NoteEvent]:
        """Generate theka (basic rhythmic pattern)"""
        # Implement tala pattern
        pass
```

#### Step 3: Create Arranger for Ensemble

```python
# File: transformation/indian_classical_arranger.py

class HindustaniArranger(GenericArranger):
    """Arranger for Hindustani classical music"""

    def _arrange_melody(self, melody: List[NoteEvent], form: MusicalForm) -> List[NoteEvent]:
        """Arrange melody with raga ornamentation"""
        # Add meend (glides), gamak (oscillations), etc.
        ornamented = self._add_raga_ornamentation(melody)
        return ornamented

    def _arrange_harmony(self, harmony: List[ChordEvent],
                        melody: List[NoteEvent], form: MusicalForm) -> List[NoteEvent]:
        """Create tanpura drone (not harmonic voicing)"""
        # Indian classical uses drone, not chord progressions
        drone_notes = self._create_tanpura_drone(harmony[0].root if harmony else 60)
        return drone_notes

    def _arrange_bass(self, harmony: List[ChordEvent], form: MusicalForm) -> List[NoteEvent]:
        """No traditional bass in Indian classical"""
        return []  # Drone serves as harmonic foundation

    def _arrange_rhythm(self, harmony: List[ChordEvent], form: MusicalForm) -> List[NoteEvent]:
        """Generate tabla patterns based on tala"""
        tabla_generator = TalaRhythmGenerator(self.style.tala if self.style else "teental")
        return tabla_generator.generate_theka()

    def _apply_articulations(self, arrangement: Dict) -> Dict:
        """Apply Indian classical articulations (meend, gamak, etc.)"""
        # Implement pitch bends for meend, oscillations for gamak
        return arrangement
```

#### Step 4: Register with Factory

```python
# File: core/ensemble_registry.py

ENSEMBLE_REGISTRY = {
    "big_band": BIG_BAND_ENSEMBLE,
    "orchestra": SYMPHONY_ORCHESTRA_ENSEMBLE,
    "string_quartet": STRING_QUARTET_ENSEMBLE,
    "hindustani": HINDUSTANI_ENSEMBLE,  # NEW
}

ARRANGER_REGISTRY = {
    "big_band": BigBandArranger,
    "orchestra": OrchestraArranger,
    "string_quartet": StringQuartetArranger,
    "hindustani": HindustaniArranger,  # NEW
}
```

#### Step 5: Use the New Genre

```python
from api.unified_api import HarmonyModuleAPI

# Generate Indian classical music
api = HarmonyModuleAPI()
composition = api.generate_arrangement(
    ensemble="hindustani",
    style="raga_yaman",
    tempo=60,  # Slow alap
    measures=32
)
composition.export("raga_yaman.mid")
```

---

## Summary: What's Universal vs. What's Genre-Specific

### Universal (Works Everywhere)
- ✅ Voice leading optimization
- ✅ Dynamic shaping (crescendo, diminuendo)
- ✅ Humanization (timing variation)
- ✅ Form generation (structural templates)
- ✅ Instrument library (ranges, capabilities)
- ✅ Component system (modular architecture)
- ✅ Orchestrator (intelligent instrument selection)

### Genre-Specific (Needs Custom Implementation)
- 🎵 Melody generators (bebop vs. raga vs. gregorian chant)
- 🎵 Harmony generators (jazz vs. classical vs. modal)
- 🎵 Rhythm generators (swing vs. tala vs. polyrhythm)
- 🎵 Voicing strategies (big band drop-2 vs. string quartet close position)
- 🎵 Articulation types (brass falls vs. string pizzicato vs. tabla bols)
- 🎵 Style profiles (Basie vs. Mozart vs. Ravi Shankar)

---

## Next Steps for Implementation

1. **Create Ensemble Registry** (`core/ensemble_registry.py`)
2. **Create Style Profile Registry** (`styles/style_registry.py`)
3. **Implement GenericArranger Base Class** (`transformation/generic_arranger.py`)
4. **Refactor BigBandArranger to Inherit from GenericArranger**
5. **Create Example: String Quartet Arranger** (demonstrate scalability)
6. **Write User Guide**: "How to Add a New Genre in 5 Steps"

---

## Conclusion

This architecture ensures that:
1. **Big band work is NOT wasted** - it's one configuration in a universal system
2. **New genres don't require rewrites** - just new configurations and profiles
3. **Components are reusable** - voice leading optimizer works for ANY harmony
4. **System is extensible** - add new genres via plugin pattern
5. **Code stays maintainable** - clear separation of universal vs. specific

**The system is now ready to scale from big band to the entire world of music.**
