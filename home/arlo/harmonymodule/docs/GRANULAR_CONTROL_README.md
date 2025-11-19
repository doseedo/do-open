# Granular Control System - Documentation

## Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Core Components](#core-components)
4. [Quick Start](#quick-start)
5. [Detailed API Reference](#detailed-api-reference)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Examples](#examples)
9. [Integration with Other Modules](#integration)

---

## Overview

The Granular Control System provides Photoshop-level precision for music generation by combining:

- **User-defined rhythm patterns** - Specify exact note timings
- **Chord progressions** - Define harmonic context
- **Instrument sections** - Select brass, strings, woodwinds, percussion
- **Idiomatic writing** - Automatic application of professional orchestration rules

### Key Features

✅ **Rhythm-to-Notes Mapping** - Convert abstract rhythms to pitched notes based on chord progressions
✅ **Section-Specific Articulation** - Brass hits, string staccato, woodwind slurs, etc.
✅ **Automatic Voicing** - Professional voicings for different ensembles
✅ **Range Validation** - Automatic transposition and range checking
✅ **Dynamic Control** - Crescendo, decrescendo, accent patterns
✅ **Phrase Shaping** - Ritardando, breath marks, ornaments
✅ **Humanization** - Subtle timing and velocity variations
✅ **Multi-Layer Textures** - Combine different sections seamlessly

---

## Key Concepts

### 1. Rhythm Patterns

A `RhythmPattern` defines the temporal structure without specific pitches:

```python
from midi_generator.generators.granular_control import RhythmPattern

rhythm = RhythmPattern(
    onsets=[0.0, 1.0, 2.0, 3.0],      # Beat positions
    durations=[0.5, 0.5, 0.5, 0.5],   # Note lengths
    accents=[True, False, True, False], # Emphasis
    velocities=[100, 80, 100, 80]     # MIDI velocity (optional)
)
```

**Key Properties:**
- `onsets`: Beat positions where notes occur (0.0 = downbeat)
- `durations`: Length of each note in beats
- `accents`: Boolean array marking emphasized notes
- `velocities`: MIDI velocity (1-127), auto-generated from accents if not specified
- `articulations`: Per-note articulation types (optional)

### 2. Instrument Sections

Sections have distinct characteristics and idiomatic writing styles:

```python
from midi_generator.generators.granular_control import InstrumentSection

InstrumentSection.BRASS      # Trumpet, trombone, horn, tuba
InstrumentSection.STRINGS    # Violin, viola, cello, double bass
InstrumentSection.WOODWINDS  # Flute, oboe, clarinet, bassoon
InstrumentSection.PERCUSSION # Drums, percussion
InstrumentSection.RHYTHM_SECTION  # Piano, bass, guitar, drums
InstrumentSection.CHOIR      # SATB vocals
```

### 3. Voicing Strategies

Different approaches to distributing chord tones across instruments:

```python
from midi_generator.generators.granular_control import VoicingStrategy

VoicingStrategy.CLOSE        # Notes within an octave
VoicingStrategy.OPEN         # Spread voicing
VoicingStrategy.DROP_2       # Drop second voice down (jazz)
VoicingStrategy.DROP_3       # Drop third voice down
VoicingStrategy.TRADITIONAL  # Flute on top, bassoon on bottom
VoicingStrategy.INTERLOCKING # Mixed timbres
VoicingStrategy.UNISON       # All same pitch
VoicingStrategy.OCTAVES      # Octave doublings
```

### 4. Articulation Types

Musical articulations define how notes are played:

```python
from midi_generator.generators.granular_control import ArticulationType

# Universal
ArticulationType.STACCATO     # Short, detached
ArticulationType.LEGATO       # Smooth, connected
ArticulationType.ACCENT       # Emphasized
ArticulationType.MARCATO      # Heavy accent

# Brass-specific
ArticulationType.TONGUED      # Single tongue attack
ArticulationType.DOUBLE_TONGUE  # Fast repeated notes
ArticulationType.SLURRED      # Smooth connection
ArticulationType.FALL_OFF     # Jazz fall

# String-specific
ArticulationType.DETACHE      # Separate bows
ArticulationType.SPICCATO     # Bouncing bow
ArticulationType.PIZZICATO    # Plucked
ArticulationType.TREMOLO      # Rapid bow movement
```

---

## Core Components

### GranularControl

Main engine for generating musical notation:

```python
from midi_generator.generators.granular_control import GranularControl

gc = GranularControl()

output = gc.generate(
    rhythm_pattern=rhythm,
    chord_progression=["Cmaj7", "Dm7", "G7", "Cmaj7"],
    section=InstrumentSection.BRASS,
    instruments=['trumpet', 'trombone'],  # Optional: specify exact instruments
    voicing_strategy=VoicingStrategy.DROP_2,
    articulation_style='hits',  # Or None for automatic
    measures=4,
    beats_per_measure=4,
    apply_swing=False,
    swing_factor=0.67  # If apply_swing=True
)
```

**Returns:** `SectionOutput` containing:
- `section`: The instrument section used
- `notes`: List of `GeneratedNote` objects
- `voicing_quality`: `Playability` rating
- `warnings`: List of range violations or issues
- `suggestions`: Recommendations for improvement

### GeneratedNote

Represents a single musical note with all parameters:

```python
@dataclass
class GeneratedNote:
    pitch: int                     # MIDI note number
    onset: float                   # Beat position
    duration: float                # Duration in beats
    velocity: int                  # MIDI velocity (1-127)
    articulation: ArticulationType # How to play the note
    instrument: str                # Instrument name
    written_pitch: Optional[int] = None  # For transposing instruments
```

---

## Quick Start

### Example 1: Brass Hits on Beats 1 and 3

```python
from midi_generator.generators.granular_control import (
    GranularControl, RhythmPattern, create_brass_hits
)

# Define rhythm
rhythm = RhythmPattern(
    onsets=[0.0, 2.0],  # Beats 1 and 3
    durations=[0.25, 0.25],
    accents=[True, True]
)

# Generate brass hits
output = create_brass_hits(
    onsets=[0.0, 2.0],
    chord_progression=["Cmaj7", "Dm7", "G7", "Cmaj7"],
    measures=4
)

print(f"Generated {len(output.notes)} notes")
print(f"Playability: {output.playability.name}")

# Export to MIDI
gc = GranularControl()
gc.to_midi(output, "brass_hits.mid", tempo=120)
```

### Example 2: String Pad

```python
from midi_generator.generators.granular_control import create_string_pad

output = create_string_pad(
    duration=4.0,  # Whole notes
    chord_progression=["Cmaj7", "Am7", "Fmaj7", "G7"],
    measures=4
)

gc = GranularControl()
gc.to_midi(output, "string_pad.mid", tempo=80)
```

### Example 3: Syncopated Funk Brass

```python
from midi_generator.generators.granular_control import (
    GranularControl, RhythmPattern, InstrumentSection, VoicingStrategy
)

# Syncopated funk pattern
rhythm = RhythmPattern(
    onsets=[0.0, 0.75, 1.5, 2.5, 3.25],
    durations=[0.25, 0.25, 0.5, 0.25, 0.5],
    accents=[True, False, True, False, True]
)

gc = GranularControl()
output = gc.generate(
    rhythm_pattern=rhythm,
    chord_progression=["Em7", "A7", "Dm7", "G7"],
    section=InstrumentSection.BRASS,
    articulation_style='hits',
    voicing_strategy=VoicingStrategy.DROP_2,
    measures=4
)
```

---

## Detailed API Reference

### RhythmPattern

#### Methods

##### `apply_swing(swing_factor: float = 0.67) -> RhythmPattern`

Apply swing feel to straight 8th notes:

```python
straight = RhythmPattern(
    onsets=[0.0, 0.5, 1.0, 1.5],
    durations=[0.5, 0.5, 0.5, 0.5]
)

swung = straight.apply_swing(swing_factor=0.67)  # Triplet swing
```

**Parameters:**
- `swing_factor`: 0.5 = straight, 0.67 = triplet swing, 0.75 = heavy swing

---

### GranularControl

#### Methods

##### `generate(...) -> SectionOutput`

Main generation method. See [Core Components](#core-components) for full signature.

##### `generate_hits(rhythm, chord_progression, **kwargs) -> SectionOutput`

Convenience method for hit-style writing (short, accented notes):

```python
output = gc.generate_hits(
    rhythm=rhythm,
    chord_progression=["C7", "F7", "C7", "G7"],
    section=InstrumentSection.BRASS,
    measures=4
)
```

##### `generate_sustained(rhythm, chord_progression, **kwargs) -> SectionOutput`

Convenience method for sustained pad-like writing:

```python
output = gc.generate_sustained(
    rhythm=long_notes,
    chord_progression=["Cmaj7", "Fmaj7", "Cmaj7", "G7"],
    section=InstrumentSection.STRINGS,
    measures=4
)
```

##### `to_midi(output: SectionOutput, filename: str, tempo: int = 120)`

Export generated output to MIDI file:

```python
gc.to_midi(output, "my_arrangement.mid", tempo=140)
```

---

### Voicing Engines

#### BrassVoicingEngine

```python
from midi_generator.generators.granular_control import BrassVoicingEngine

voicing = BrassVoicingEngine.voice_chord(
    chord="Cmaj7",
    ensemble='big_band',  # or 'brass_quartet', 'brass_quintet'
    voicing_type=VoicingStrategy.DROP_2
)
# Returns: [(instrument, pitch), ...]
```

**Ensembles:**
- `'big_band'`: 4 trumpets, 4 trombones
- `'brass_quartet'`: 2 trumpets, trombone, tuba
- `'brass_quintet'`: 2 trumpets, horn, trombone, tuba

#### StringVoicingEngine

```python
from midi_generator.generators.granular_control import StringVoicingEngine

voicing = StringVoicingEngine.voice_chord(
    chord="Dm7",
    ensemble='string_quartet',  # or 'string_section', 'chamber'
    voicing_type=VoicingStrategy.CLOSE
)
```

**Ensembles:**
- `'string_quartet'`: 2 violins, viola, cello
- `'string_section'`: Multiple violins, violas, cellos, double bass
- `'chamber'`: Violin, viola, cello

#### WoodwindVoicingEngine

```python
from midi_generator.generators.granular_control import WoodwindVoicingEngine

voicing = WoodwindVoicingEngine.voice_chord(
    chord="G7",
    ensemble='wind_quartet',  # or 'wind_quintet', 'symphonic'
    voicing_type=VoicingStrategy.TRADITIONAL
)
```

**Ensembles:**
- `'wind_quartet'`: Flute, oboe, clarinet, bassoon
- `'wind_quintet'`: Flute, oboe, clarinet, horn, bassoon
- `'symphonic'`: Pairs of flutes, oboes, clarinets, bassoons

---

## Advanced Features

### 1. Percussion/Drums

```python
from midi_generator.generators.granular_control import PercussionVoicingEngine

# Convert rhythm to drum hits
rhythm = RhythmPattern(
    onsets=[0.0, 0.5, 1.0, 1.5],
    durations=[0.25, 0.25, 0.25, 0.25]
)

drum_notes = PercussionVoicingEngine.rhythm_to_drums(
    rhythm,
    drum_voices=['kick', 'snare', 'kick', 'snare']
)

# Or create basic beat
rock_beat = PercussionVoicingEngine.create_basic_beat(
    style='rock',  # or 'jazz', 'funk', 'latin'
    measures=4
)
```

**Available Drum Voices:**
- `'kick'`, `'snare'`, `'closed_hihat'`, `'open_hihat'`
- `'crash'`, `'ride'`, `'tom_low'`, `'tom_mid'`, `'tom_high'`
- `'cowbell'`, `'tambourine'`, `'clap'`, `'shaker'`
- `'conga_low'`, `'conga_high'`, `'bongo_low'`, `'bongo_high'`

### 2. Dynamics and Expression

```python
from midi_generator.generators.granular_control import DynamicsEngine

# Apply crescendo
DynamicsEngine.apply_dynamics_curve(
    notes,
    curve_type='crescendo',  # or 'decrescendo', 'swell', 'arch'
    start_dynamic='p',       # pp, p, mp, mf, f, ff, fff
    end_dynamic='f'
)

# Apply accent pattern
DynamicsEngine.apply_accents(
    notes,
    accent_pattern=[True, False, True, False],
    accent_amount=20  # Velocity increase
)
```

**Dynamic Levels:**
- `'ppp'`: 20, `'pp'`: 35, `'p'`: 50
- `'mp'`: 65, `'mf'`: 80
- `'f'`: 95, `'ff'`: 110, `'fff'`: 127

**Curve Types:**
- `'crescendo'`: Gradual increase
- `'decrescendo'`: Gradual decrease
- `'swell'`: Crescendo then decrescendo
- `'arch'`: Bell curve (smooth swell)

### 3. Phrase Shaping

```python
from midi_generator.generators.granular_control import PhraseShaper

# Add phrase ending
PhraseShaper.add_phrase_ending(
    notes,
    ending_type='ritardando'  # or 'decrescendo', 'breath', 'fermata'
)

# Add ornaments
ornamented_notes = PhraseShaper.add_ornaments(
    notes,
    ornament_type='grace_note',  # or 'mordent', 'turn', 'trill'
    positions=[0, -1]  # First and last notes
)
```

### 4. Humanization

```python
from midi_generator.generators.granular_control import AdvancedControlEngine

# Add human feel
AdvancedControlEngine.apply_humanization(
    notes,
    timing_variance=0.02,   # Max timing deviation (beats)
    velocity_variance=5     # Max velocity deviation
)
```

### 5. Layered Textures

```python
from midi_generator.generators.granular_control import (
    AdvancedControlEngine, InstrumentSection, RhythmPattern
)

base_rhythm = RhythmPattern(
    onsets=[0.0, 1.0, 2.0, 3.0],
    durations=[0.5, 0.5, 0.5, 0.5]
)

layers = [
    {
        'section': InstrumentSection.BRASS,
        'instruments': ['trumpet', 'trombone'],
        'measures': 4
    },
    {
        'section': InstrumentSection.STRINGS,
        'instruments': ['violin', 'cello'],
        'offset': 0.5,  # Delay by half beat
        'duration_multiplier': 2.0,  # Double note lengths
        'measures': 4
    }
]

outputs = AdvancedControlEngine.create_layered_texture(
    base_rhythm,
    chord_progression=["Cmaj7", "Dm7", "G7", "Cmaj7"],
    layers=layers
)
```

---

## Best Practices

### 1. Rhythm Design

**✅ DO:**
- Start with simple patterns and add complexity gradually
- Use accent patterns to create musical interest
- Consider the instrument section's technical capabilities

**❌ DON'T:**
- Create patterns with notes too fast for the section (check `fast_passage_limit`)
- Use extreme register jumps in quick succession
- Ignore natural phrase boundaries

### 2. Voicing Selection

**Brass:**
- Use `DROP_2` for big band style
- Use `UNISON` for powerful punches
- Keep trumpets below G5 for comfortable playing

**Strings:**
- Use `CLOSE` for intimate quartet sound
- Use `OPEN` for fuller orchestral sound
- Consider bowing limitations (fast passages need spiccato/staccato)

**Woodwinds:**
- Use `TRADITIONAL` for classic orchestral sound (flute on top)
- Use `INTERLOCKING` for blended color
- Respect individual instrument sweet spots

### 3. Articulation

**Match to Style:**
- Jazz: Tongued attacks, fall-offs, shakes
- Classical: Legato, detaché, clean attacks
- Funk/Pop: Staccato, marcato, tight rhythms

**Match to Tempo:**
- Fast tempos: Use double/triple tonguing for brass, spiccato for strings
- Slow tempos: Legato, sustained tones, vibrato

### 4. Range Management

Always check output warnings:

```python
output = gc.generate(...)

if output.warnings:
    print("⚠️  Warnings:")
    for warning in output.warnings:
        print(f"   {warning}")

if output.suggestions:
    print("💡 Suggestions:")
    for suggestion in output.suggestions:
        print(f"   {suggestion}")
```

### 5. MIDI Export

```python
# Export with appropriate tempo
gc.to_midi(output, "fast_swing.mid", tempo=180)  # Fast
gc.to_midi(output, "ballad.mid", tempo=60)       # Slow

# Combine multiple sections
all_notes = brass_output.notes + string_output.notes + drums_output.notes
combined = SectionOutput(
    section=InstrumentSection.BRASS,  # Arbitrary
    notes=all_notes,
    voicing_quality=brass_output.voicing_quality,
    warnings=[],
    suggestions=[]
)
gc.to_midi(combined, "full_arrangement.mid", tempo=120)
```

---

## Examples

### Complete Big Band Arrangement

```python
from midi_generator.generators.granular_control import *

gc = GranularControl()
chords = ["Cmaj7", "Am7", "Dm7", "G7"]

# 1. Brass hits on 1 and 3
brass_rhythm = RhythmPattern(
    onsets=[0.0, 2.0],
    durations=[0.25, 0.25],
    accents=[True, True]
)

brass = gc.generate_hits(
    rhythm=brass_rhythm,
    chord_progression=chords,
    section=InstrumentSection.BRASS,
    voicing_strategy=VoicingStrategy.DROP_2,
    measures=4
)

# 2. String pad (sustained)
string_rhythm = RhythmPattern(
    onsets=[0.0, 4.0, 8.0, 12.0],
    durations=[4.0, 4.0, 4.0, 4.0]
)

strings = gc.generate_sustained(
    rhythm=string_rhythm,
    chord_progression=chords,
    section=InstrumentSection.STRINGS,
    measures=4
)

# 3. Jazz drums
drums = PercussionVoicingEngine.create_basic_beat(
    style='jazz',
    measures=4
)

# 4. Apply dynamics
DynamicsEngine.apply_dynamics_curve(
    strings.notes,
    curve_type='arch',
    start_dynamic='p',
    end_dynamic='mf'
)

# 5. Humanize
AdvancedControlEngine.apply_humanization(brass.notes)
AdvancedControlEngine.apply_humanization(strings.notes)

# 6. Export
gc.to_midi(brass, "brass_section.mid", tempo=140)
gc.to_midi(strings, "string_section.mid", tempo=140)

# Combine all
from midi_generator.generators.granular_control import SectionOutput, GeneratedNote

all_notes = brass.notes + strings.notes + [
    GeneratedNote(
        pitch=note.pitch,
        onset=note.onset,
        duration=note.duration,
        velocity=note.velocity,
        articulation=note.articulation,
        instrument='drums'
    ) for note in drums
]

combined = SectionOutput(
    section=InstrumentSection.BRASS,
    notes=all_notes,
    voicing_quality=brass.voicing_quality,
    warnings=[],
    suggestions=[]
)

gc.to_midi(combined, "big_band_arrangement.mid", tempo=140)
```

### Film Score: Dramatic Crescendo

```python
from midi_generator.generators.granular_control import *

gc = GranularControl()

# Long sustained notes
rhythm = RhythmPattern(
    onsets=[0.0, 8.0],
    durations=[8.0, 8.0]
)

chords = ["Cm", "Ab", "Fm", "G"]

# Generate string section
strings = gc.generate_sustained(
    rhythm=rhythm,
    chord_progression=chords,
    section=InstrumentSection.STRINGS,
    voicing_strategy=VoicingStrategy.OPEN,
    measures=16,
    beats_per_measure=4
)

# Apply dramatic crescendo
DynamicsEngine.apply_dynamics_curve(
    strings.notes,
    curve_type='crescendo',
    start_dynamic='pp',
    end_dynamic='fff'
)

# Add brass for climax (last 4 measures)
brass_rhythm = RhythmPattern(
    onsets=[0.0, 1.0, 2.0, 3.0],
    durations=[1.0, 1.0, 1.0, 1.0],
    accents=[False, False, True, True]
)

brass = gc.generate(
    rhythm_pattern=brass_rhythm,
    chord_progression=chords[-4:],
    section=InstrumentSection.BRASS,
    voicing_strategy=VoicingStrategy.OCTAVES,
    measures=4
)

# Shift brass to start at measure 13
for note in brass.notes:
    note.onset += 48.0  # 12 measures * 4 beats

# Combine
all_notes = strings.notes + brass.notes

combined = SectionOutput(
    section=InstrumentSection.STRINGS,
    notes=all_notes,
    voicing_quality=strings.voicing_quality,
    warnings=[],
    suggestions=[]
)

gc.to_midi(combined, "dramatic_crescendo.mid", tempo=60)
```

---

## Integration

### With Genre Fusion System

```python
# Coming soon: Integration with style_fusion.py for genre-specific patterns
from midi_generator.generators.style_fusion import GenreBlender

# Generate brass with jazz articulation
jazz_brass = gc.generate(
    rhythm_pattern=swing_rhythm,
    chord_progression=jazz_changes,
    section=InstrumentSection.BRASS,
    articulation_style='jazz_articulation',
    apply_swing=True,
    swing_factor=0.67
)
```

### With Harmony Modules

```python
from advanced_modules.harmony_advanced import AdvancedSubstitutions

# Get sophisticated chord progression
original = ["C", "F", "G", "C"]
reharmonized = AdvancedSubstitutions.jazz_reharmonization(original)

# Use with granular control
output = gc.generate(
    rhythm_pattern=rhythm,
    chord_progression=reharmonized,
    section=InstrumentSection.BRASS
)
```

### With MIDI Analysis

```python
# Coming soon: Integration with genre_detector.py
from midi_generator.analysis.genre_detector import GenreDetector

# Detect style from existing MIDI
detector = GenreDetector("my_song.mid")
style = detector.to_genre_features()

# Generate in detected style
# (Future feature)
```

---

## Troubleshooting

### Issue: Notes Out of Range

**Problem:** Warnings about notes outside instrument range

**Solution:**
```python
# 1. Check register_preference
output = gc.generate(
    ...,
    register_preference=Register.NEUTRAL  # Try DARK or BRIGHT
)

# 2. Manually specify target instruments
output = gc.generate(
    ...,
    instruments=['trombone', 'bass_trombone']  # Lower range
)
```

### Issue: Mechanical Sound

**Problem:** Generated music sounds robotic

**Solution:**
```python
# Apply humanization
AdvancedControlEngine.apply_humanization(
    output.notes,
    timing_variance=0.02,
    velocity_variance=8
)

# Add phrase shaping
PhraseShaper.add_phrase_ending(output.notes, ending_type='ritardando')

# Add ornaments
output.notes = PhraseShaper.add_ornaments(
    output.notes,
    ornament_type='grace_note',
    positions=[0, -1]
)
```

### Issue: Playability Warnings

**Problem:** `playability` rating is DIFFICULT or PROBLEMATIC

**Solution:**
```python
# Check suggestions
for suggestion in output.suggestions:
    print(suggestion)

# Simplify rhythm
rhythm = RhythmPattern(
    onsets=[0.0, 1.0, 2.0, 3.0],  # Slower
    durations=[0.5, 0.5, 0.5, 0.5]
)

# Use more conservative voicing
output = gc.generate(
    ...,
    voicing_strategy=VoicingStrategy.CLOSE  # Easier than OPEN
)
```

---

## Performance Tips

### Batch Generation

```python
# Generate multiple variations efficiently
gc = GranularControl()  # Reuse instance

variations = []
for voicing in [VoicingStrategy.CLOSE, VoicingStrategy.DROP_2, VoicingStrategy.OPEN]:
    output = gc.generate(
        rhythm_pattern=rhythm,
        chord_progression=chords,
        section=InstrumentSection.BRASS,
        voicing_strategy=voicing
    )
    variations.append(output)
```

### Caching

```python
# Cache voicings for repeated use
brass_voicings = {}
for chord in ["Cmaj7", "Dm7", "G7", "Cmaj7"]:
    if chord not in brass_voicings:
        brass_voicings[chord] = BrassVoicingEngine.voice_chord(
            chord, ensemble='big_band', voicing_type=VoicingStrategy.DROP_2
        )
```

---

## API Summary

### Classes

- `GranularControl` - Main generation engine
- `RhythmPattern` - Rhythm definition
- `GeneratedNote` - Output note representation
- `SectionOutput` - Generation result

### Enums

- `InstrumentSection` - Instrument families
- `ArticulationType` - Note articulations
- `VoicingStrategy` - Voicing approaches
- `Register` - Pitch register preference

### Voicing Engines

- `BrassVoicingEngine`
- `StringVoicingEngine`
- `WoodwindVoicingEngine`
- `PercussionVoicingEngine`

### Utilities

- `ChordToPitchMapper` - Chord parsing
- `ArticulationLibrary` - Articulation patterns
- `DynamicsEngine` - Dynamic shaping
- `PhraseShaper` - Musical phrasing
- `AdvancedControlEngine` - Advanced features

### Convenience Functions

- `create_brass_hits()`
- `create_string_pad()`

---

## Version History

### v1.0.0 (2025-11-19) - Initial Release

- Core rhythm pattern system
- Brass, string, woodwind voicing engines
- Articulation library with 15+ patterns
- Percussion/drum support
- Dynamics and expression control
- Phrase shaping utilities
- Humanization
- Layered texture generation
- MIDI export
- Comprehensive test suite (50+ tests)
- Full documentation

---

## Credits

**Author:** Agent 8 - Modular Genre Fusion Enhancement
**Date:** 2025-11-19
**Part of:** HarmonyModule Advanced MIDI Library (85,000+ lines)

**Research Sources:**
- Samuel Adler: "The Study of Orchestration" (4th Edition)
- Rimsky-Korsakov: "Principles of Orchestration" (1913)
- Alfred Blatter: "Instrumentation and Orchestration"
- Brass articulation research (2024)
- String bowing techniques
- Woodwind voicing strategies

---

## License

MIT License - Part of HarmonyModule library

---

## Support

For issues, feature requests, or questions:
- See `examples/granular_control_examples.py` for more examples
- Check test suite: `tests/test_granular_control.py`
- Review existing orchestration modules in `advanced_modules/`

---

**End of Documentation**
