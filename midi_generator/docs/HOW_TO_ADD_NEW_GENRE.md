# How to Add a New Genre in 5 Steps

**Author**: Agent 19 - Genre Scalability Architect
**Audience**: Developers extending the MIDI Generator to new genres
**Difficulty**: Intermediate
**Time Required**: 2-4 hours for a complete genre implementation

---

## Overview

This guide shows you how to add support for a completely new genre or ensemble to the MIDI Generator system. The architecture is designed so that you can add new genres **without modifying existing code** - just add new configuration files and implement a few genre-specific classes.

### What You'll Create

1. **Ensemble Configuration** - Define instruments and ranges
2. **Style Profile** (optional) - Define aesthetic characteristics
3. **Genre-Specific Generators** - Melody, harmony, rhythm for your genre
4. **Arranger Class** - Orchestrate for your ensemble
5. **Registration** - Make it available to users

---

## Architecture Overview

```
Universal Components (YOU DON'T MODIFY THESE)
├── VoiceLeadingOptimizer  ← Works for ANY harmony
├── DynamicShaping         ← Works for ANY music
├── HumanizationEngine     ← Works for ANY rhythm
└── GenericArranger        ← Template for ALL arrangers

Genre-Specific (YOU CREATE THESE)
├── Ensemble Config        ← Instruments, ranges, sections
├── Style Profile          ← Aesthetic choices
├── Melody Generator       ← How to generate melodies
├── Harmony Generator      ← How to generate chords
├── Rhythm Generator       ← How to generate rhythms
└── Arranger              ← Orchestration rules
```

---

## Step 1: Define Your Ensemble

Create a configuration that defines:
- What instruments are in the ensemble
- Their playable ranges
- How they're grouped into sections
- Their roles (melody, harmony, bass, rhythm)

### Example: String Quartet

Create file: `core/ensembles/string_quartet.py`

```python
from core.ensemble_registry import EnsembleConfig, SectionConfig, EnsembleType

STRING_QUARTET_ENSEMBLE = EnsembleConfig(
    name="String Quartet",
    ensemble_type=EnsembleType.STRING_QUARTET,
    sections={
        "strings": SectionConfig(
            name="String Quartet",
            instruments=["violin1", "violin2", "viola", "cello"],
            ranges={
                "violin1": (55, 103),  # G3-G7 (first violin goes higher)
                "violin2": (55, 96),   # G3-C7 (second violin)
                "viola": (48, 91),     # C3-G6
                "cello": (36, 84)      # C2-C6
            },
            voicing_types=["close", "spread", "open"],
            role="complete_texture",  # Quartet handles everything
            max_voices=4
        )
    },
    typical_styles=["classical", "romantic", "contemporary"],
    max_total_voices=4,
    orchestration_style="chamber",
    voice_leading_priority="strict"  # Classical voice leading rules
)
```

### Key Decisions:

**Instrument Ranges**: Use MIDI note numbers
- Middle C (C4) = 60
- Calculator: `note_number = 12 * octave + note_offset`
- Example: G3 = 12*3 + 7 = 55

**Voicing Types**: What voicing styles are idiomatic?
- Big band: "drop_2", "drop_3", "close"
- Orchestra: "divisi", "tutti", "solo"
- String quartet: "close", "spread", "open"

**Voice Leading Priority**: How strict are the rules?
- `"strict"`: Classical (no parallel 5ths/octaves)
- `"moderate"`: Jazz-influenced
- `"loose"`: Rock, pop, world music

---

## Step 2: Create Style Profile (Optional)

Style profiles define how a particular composer or performer approaches music. This is optional but highly recommended for genre-specific aesthetics.

Create file: `styles/beethoven_string_quartet.py`

```python
from styles.style_registry import StyleProfile

BEETHOVEN_STRING_QUARTET_STYLE = StyleProfile(
    name="Beethoven String Quartet",
    composer_era="classical_romantic",
    cultural_origin="western",

    # Orchestration
    voicing_preference="balanced",
    voicing_spacing="medium",
    doubling_rules={
        "octave_doubling": 0.4,
        "unison_emphasis": 0.3
    },

    # Harmony - Beethoven's harmonic language
    harmony_complexity=0.7,  # More complex than Mozart
    chord_extensions=[7],    # 7th chords, not 9/11/13
    chromaticism=0.6,        # Chromatic passages
    modulation_frequency=0.6,  # Frequent modulations

    # Articulation
    articulation_variety=0.8,
    articulation_probabilities={
        "sforzando": 0.7,    # Beethoven's signature
        "marcato": 0.6,
        "accent": 0.5,
        "legato": 0.4
    },
    use_ornamentation=0.3,

    # Dynamics - Extreme contrasts
    dynamic_range="very_wide",
    use_crescendo=0.8,
    sudden_dynamic_changes=0.8,  # Subito piano/forte

    # Rhythm
    rhythmic_complexity=0.7,
    syncopation=0.5,
    swing_factor=0.0,  # No swing in classical
    rubato_tendency=0.2,

    # Form
    intro_style="dramatic",
    ending_style="triumphant",
    form_adherence=0.8,  # Beethoven expanded classical forms

    # Texture
    texture_density=0.7,
    texture_variation=0.8,  # Highly varied
    counterpoint_usage=0.7,  # Strong contrapuntal writing

    # Special
    signature_sounds=["sforzando_accents", "dramatic_pauses", "motivic_development"],
    special_techniques=["fugato", "development_sections"]
)
```

---

## Step 3: Implement Genre-Specific Generators

Create generators for melody, harmony, and rhythm appropriate to your genre.

### Example: String Quartet Melody Generator

Create file: `genres/classical_chamber.py`

```python
from typing import List
from analysis.midi_analyzer import NoteEvent, JazzNote
import random

class ClassicalMelodyGenerator:
    """Generate classical-style melodies for string quartet"""

    def __init__(self, key: int = 0, mode: str = "major"):
        self.key = key
        self.mode = mode
        self.scale = self._get_scale()

    def _get_scale(self) -> List[int]:
        """Get scale degrees for key and mode"""
        if self.mode == "major":
            intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale
        else:
            intervals = [0, 2, 3, 5, 7, 8, 10]  # Natural minor
        return [(self.key + i) % 12 for i in intervals]

    def generate_phrase(self,
                       chord: ChordEvent,
                       length_beats: int = 4,
                       contour: str = "arch") -> List[NoteEvent]:
        """
        Generate a classical phrase.

        Args:
            chord: Current chord
            length_beats: Length of phrase in beats
            contour: "arch", "ascending", "descending", "wave"

        Returns:
            List of NoteEvents forming the phrase
        """
        notes = []
        current_pitch = 60 + self.key  # Start on tonic

        # Generate contour
        if contour == "arch":
            # Start mid, rise to peak at 2/3, descend
            target_pitches = self._generate_arch_contour(current_pitch, length_beats)
        elif contour == "ascending":
            target_pitches = self._generate_ascending_contour(current_pitch, length_beats)
        else:
            target_pitches = self._generate_descending_contour(current_pitch, length_beats)

        # Create notes
        for i, target in enumerate(target_pitches):
            note = NoteEvent(
                start_time=i * 0.5,  # Eighth notes
                duration=0.45,
                start_tick=int(i * 0.5 * 480),
                duration_ticks=int(0.45 * 480),
                pitch=target,
                velocity=80 + random.randint(-10, 10),  # Dynamic variation
                channel=0,
                track_idx=0
            )
            notes.append(note)

        return notes

    def _generate_arch_contour(self, start_pitch: int, length: int) -> List[int]:
        """Generate arch-shaped melodic contour"""
        contour = []
        peak_position = int(length * 0.67)  # Peak at 2/3 point

        for i in range(length * 2):  # 2 notes per beat (eighth notes)
            if i < peak_position:
                # Ascending
                step = random.choice([0, 2, 4])  # Stepwise or skip
                start_pitch += step
            elif i == peak_position:
                # Peak
                pass
            else:
                # Descending
                step = random.choice([0, -2, -4])
                start_pitch += step

            # Quantize to scale
            pitch = self._quantize_to_scale(start_pitch)
            contour.append(pitch)

        return contour

    def _quantize_to_scale(self, pitch: int) -> int:
        """Quantize pitch to nearest scale degree"""
        pitch_class = pitch % 12
        if pitch_class in self.scale:
            return pitch

        # Find nearest scale degree
        distances = [abs(pitch_class - s) for s in self.scale]
        nearest_idx = distances.index(min(distances))
        nearest_scale_degree = self.scale[nearest_idx]

        return pitch - pitch_class + nearest_scale_degree


class ClassicalHarmonyGenerator:
    """Generate classical harmony progressions"""

    def generate_progression(self, key: int, length_bars: int = 8) -> List[ChordEvent]:
        """Generate common practice period harmony"""
        # Simple I-IV-V-I progression for example
        progression = [
            ChordEvent(0.0, 4.0, key, "major", [key, key+4, key+7]),      # I
            ChordEvent(4.0, 4.0, key+5, "major", [key+5, key+9, key+12]), # IV
            ChordEvent(8.0, 4.0, key+7, "dom7", [key+7, key+11, key+14, key+17]), # V7
            ChordEvent(12.0, 4.0, key, "major", [key, key+4, key+7]),     # I
        ]
        return progression
```

---

## Step 4: Create Your Arranger

Extend `GenericArranger` and implement the abstract methods.

Create file: `transformation/string_quartet_arranger.py`

```python
from transformation.generic_arranger import GenericArranger
from typing import List, Dict
from analysis.midi_analyzer import NoteEvent, ChordEvent
import copy

class StringQuartetArranger(GenericArranger):
    """
    Arranger for string quartet.

    Implements classical voice leading and string quartet idioms.
    """

    def _arrange_melody(self, melody: List[NoteEvent], form) -> List[NoteEvent]:
        """Assign melody to first violin"""
        arranged = []
        for note in melody:
            new_note = copy.copy(note)
            # Ensure it's in violin range
            while new_note.pitch < 55:  # G3
                new_note.pitch += 12
            while new_note.pitch > 96:  # C7
                new_note.pitch -= 12

            new_note.channel = 0
            new_note.track_idx = 0
            arranged.append(new_note)

        return arranged

    def _arrange_harmony(self,
                        harmony: List[ChordEvent],
                        melody: List[NoteEvent],
                        form) -> List[NoteEvent]:
        """
        Create 4-part harmony for string quartet.

        Uses classical voice leading:
        - Soprano (violin 1): melody
        - Alto (violin 2): 3rd or 5th of chord
        - Tenor (viola): 5th or root
        - Bass (cello): root
        """
        harmony_notes = []

        for chord in harmony:
            # Create 4-part voicing
            root = chord.root
            third = root + 4 if "major" in chord.chord_type else root + 3
            fifth = root + 7

            # Distribute to instruments
            voices = {
                "cello": root + 36,      # Bass (low)
                "viola": fifth + 48,     # Tenor (middle)
                "violin2": third + 60,   # Alto (upper middle)
                "violin1": root + 72     # Soprano (highest)
            }

            # Create notes
            for i, (instrument, pitch) in enumerate(voices.items()):
                note = NoteEvent(
                    start_time=chord.start_time,
                    duration=chord.duration * 0.9,
                    start_tick=int(chord.start_time * 480),
                    duration_ticks=int(chord.duration * 0.9 * 480),
                    pitch=pitch,
                    velocity=75,
                    channel=i,
                    track_idx=i
                )
                harmony_notes.append(note)

        return harmony_notes

    def _arrange_bass(self, harmony: List[ChordEvent], form) -> List[NoteEvent]:
        """Cello plays bass line"""
        bass_notes = []

        for chord in harmony:
            # Simple bass: root on beat 1, fifth on beat 3
            root_note = NoteEvent(
                start_time=chord.start_time,
                duration=1.9,
                start_tick=int(chord.start_time * 480),
                duration_ticks=int(1.9 * 480),
                pitch=chord.root + 36,  # Low cello range
                velocity=85,
                channel=3,
                track_idx=3
            )
            bass_notes.append(root_note)

            if chord.duration >= 4.0:
                # Add fifth on beat 3
                fifth_note = NoteEvent(
                    start_time=chord.start_time + 2.0,
                    duration=1.9,
                    start_tick=int((chord.start_time + 2.0) * 480),
                    duration_ticks=int(1.9 * 480),
                    pitch=chord.root + 36 + 7,
                    velocity=80,
                    channel=3,
                    track_idx=3
                )
                bass_notes.append(fifth_note)

        return bass_notes

    def _arrange_rhythm(self, harmony: List[ChordEvent], form) -> List[NoteEvent]:
        """String quartet has no percussion"""
        return []

    def _apply_articulations(self, arrangement: Dict) -> Dict:
        """
        Apply string articulations.

        Classical string quartet uses:
        - Legato for smooth passages
        - Staccato for detached notes
        - Sforzando for accents
        - Pizzicato occasionally
        """
        # For now, just return as-is
        # Full implementation would add articulation markers
        return arrangement
```

---

## Step 5: Register Your Genre

Add your ensemble and style to the registries.

### Register Ensemble

Edit `core/ensemble_registry.py`:

```python
# Add import
from core.ensembles.string_quartet import STRING_QUARTET_ENSEMBLE

# Add to ENSEMBLE_REGISTRY
ENSEMBLE_REGISTRY = {
    "big_band": BIG_BAND_ENSEMBLE,
    "symphony_orchestra": SYMPHONY_ORCHESTRA_ENSEMBLE,
    "string_quartet": STRING_QUARTET_ENSEMBLE,  # NEW!
    # ... other ensembles
}
```

### Register Style (if created)

Edit `styles/style_registry.py`:

```python
# Add import
from styles.beethoven_string_quartet import BEETHOVEN_STRING_QUARTET_STYLE

# Add to STYLE_REGISTRY
STYLE_REGISTRY = {
    "basie": BASIE_STYLE,
    "ellington": ELLINGTON_STYLE,
    "beethoven_quartet": BEETHOVEN_STRING_QUARTET_STYLE,  # NEW!
    # ... other styles
}
```

### Register Arranger

Create `transformation/arranger_registry.py` (if it doesn't exist):

```python
from transformation.string_quartet_arranger import StringQuartetArranger

ARRANGER_REGISTRY = {
    "string_quartet": StringQuartetArranger,
    # ... other arrangers
}

def get_arranger(ensemble_type: str, style_name: str = None):
    """Get arranger class for ensemble type"""
    arranger_class = ARRANGER_REGISTRY.get(ensemble_type)
    if arranger_class:
        return arranger_class(ensemble_type, style_name)
    return None
```

---

## Step 6: Use Your New Genre

Now you can use your genre in the unified API:

```python
from api.unified_api import HarmonyModuleAPI

api = HarmonyModuleAPI()

# Generate string quartet arrangement
composition = api.generate_arrangement(
    ensemble="string_quartet",
    style="beethoven_quartet",
    tempo=120,
    key="C",
    measures=32,
    form="sonata"
)

composition.export("beethoven_quartet.mid")
```

Or use directly:

```python
from transformation.string_quartet_arranger import StringQuartetArranger
from genres.classical_chamber import ClassicalHarmonyGenerator, ClassicalMelodyGenerator

# Generate content
harmony_gen = ClassicalHarmonyGenerator()
melody_gen = ClassicalMelodyGenerator(key=0, mode="major")

harmony = harmony_gen.generate_progression(key=0, length_bars=8)
melody = melody_gen.generate_phrase(harmony[0], length_beats=16)

# Arrange
arranger = StringQuartetArranger("string_quartet", "beethoven_quartet")
arrangement = arranger.arrange(melody, harmony)

# Export to MIDI...
```

---

## Checklist: Is Your Genre Implementation Complete?

- [ ] Ensemble configuration created
- [ ] Instruments and ranges defined
- [ ] Sections and roles specified
- [ ] Style profile created (optional but recommended)
- [ ] Melody generator implemented
- [ ] Harmony generator implemented
- [ ] Rhythm generator implemented (if applicable)
- [ ] Arranger class created (extends GenericArranger)
- [ ] All abstract methods implemented
- [ ] Ensemble registered in ENSEMBLE_REGISTRY
- [ ] Style registered in STYLE_REGISTRY (if created)
- [ ] Arranger registered in ARRANGER_REGISTRY
- [ ] Test generation works end-to-end
- [ ] Documentation written for your genre

---

## Tips and Best Practices

### 1. Start Simple, Then Refine

Don't try to implement everything perfectly the first time. Start with:
1. Basic ensemble definition
2. Simple melody generator (even random notes)
3. Simple harmony (just root notes)
4. Minimal arranger

Then iterate and improve.

### 2. Study Real Examples

- Analyze MIDI files of your target genre
- Study scores and transcriptions
- Measure ranges, patterns, voicing styles
- Extract statistical patterns

### 3. Reuse Universal Components

Don't reinvent the wheel:
- Use VoiceLeadingOptimizer for smooth voice leading
- Use DynamicShaping for crescendo/diminuendo
- Use HumanizationEngine for realistic timing
- Use FormGenerator for structure

### 4. Test Incrementally

Test each component individually:
```python
# Test melody generator alone
melody = gen.generate_phrase(chord, 8)
print(f"Generated {len(melody)} notes")

# Test harmony generator alone
harmony = gen.generate_progression(key=0, length_bars=8)
print(f"Generated {len(harmony)} chords")

# Test arranger with sample data
arrangement = arranger.arrange(sample_melody, sample_harmony)
```

### 5. Document Your Choices

Add comments explaining:
- Why certain ranges were chosen
- What voicing techniques are used
- How your implementation differs from others
- Any simplifications or compromises made

---

## Common Pitfalls to Avoid

### ❌ Hardcoding MIDI Programs

**Don't**:
```python
note.channel = 40  # Hardcoded to violin
```

**Do**:
```python
instrument = self.ensemble.get_instrument("violin1")
note.program = instrument.midi_program
```

### ❌ Ignoring Instrument Ranges

**Don't**:
```python
note.pitch = 120  # Way too high for any instrument!
```

**Do**:
```python
while note.pitch > section.ranges["violin"][1]:
    note.pitch -= 12  # Transpose down
```

### ❌ Tight Coupling

**Don't**:
```python
class MyArranger:
    def arrange(self):
        # Hardcoded for one specific use case
        return fixed_pattern
```

**Do**:
```python
class MyArranger(GenericArranger):
    def _arrange_melody(self, melody, form):
        # Adapts to input
        return self._process(melody, form)
```

---

## Next Steps

1. **Add More Styles**: Create multiple style profiles for your genre (early vs. late Beethoven, Haydn vs. Mozart, etc.)

2. **Improve Generators**: Add more sophisticated melody generation (motific development, ornamentation, etc.)

3. **Add Articulations**: Implement genre-specific articulations (pizzicato, tremolo, mutes, etc.)

4. **Create Examples**: Write example scripts showing off your genre

5. **Contribute Back**: Share your genre implementation with the community!

---

## Questions?

If you get stuck, check:
1. **Architecture Doc**: `core/GENRE_SCALABILITY_ARCHITECTURE.md`
2. **Existing Implementations**: Look at `BigBandArranger` for patterns
3. **Component System**: `core/component_system.py` for modular generation
4. **Style Registry**: `styles/style_registry.py` for style profile examples

Happy music generation! 🎵
