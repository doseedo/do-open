# Modern Big Band Style Profiles

**Author:** Agent 15 - Modern Big Band Style Analyzer
**Date:** 2025
**Status:** Implemented and Validated

## Overview

This module provides comprehensive style profiles for three groundbreaking modern big band arrangers:

1. **Thad Jones** (1923-1986) - Angular melodies, quartal harmony, wide intervals
2. **Maria Schneider** (1960-present) - Orchestral colors, impressionistic, cinematic
3. **Gordon Goodwin** (1954-present) - High energy, contemporary swing, complex rhythms

These profiles capture the distinctive characteristics of each arranger's style and can be used to generate authentic-sounding arrangements that emulate their approaches.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Style Profiles](#style-profiles)
- [ModernBigBandArranger](#modernbigbandarranger)
- [Integration Points](#integration-points)
- [Validation](#validation)
- [Research Sources](#research-sources)

## Features

### Style Profile System

Each style profile includes comprehensive parameters:

- **Orchestration**: Voicing preferences, spacing, variety
- **Harmony**: Complexity, quartal/cluster usage, chord extensions
- **Melody**: Interval preferences, chromaticism, phrase variance
- **Rhythm**: Complexity, odd meters, syncopation
- **Articulation**: Variety, falls, doits, shakes
- **Dynamics**: Range, crescendo usage, terraced dynamics
- **Texture**: Density, unison/tutti usage, section contrast
- **Form**: Intro/ending styles, interlude usage
- **Special Techniques**: Woodwind doublings, pedal tones, ostinatos

### Modern Arranging Techniques

The `ModernBigBandArranger` class implements:

- **Quartal Voicings**: Stacked 4ths (Thad Jones/McCoy Tyner technique)
- **Wide Spacing**: Wide intervals between voices (modern sound)
- **Dynamic Shaping**: Context-aware velocity curves
- **Style-Specific Suggestions**: Intro/outro types, tempo ranges

## Installation

The styles module is part of the midi_generator package:

```python
# Add midi_generator to your Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "midi_generator"))

# Import style profiles
from styles import (
    THAD_JONES_STYLE,
    MARIA_SCHNEIDER_STYLE,
    GORDON_GOODWIN_STYLE,
    ModernBigBandArranger
)
```

## Usage

### Basic Usage

```python
from styles.modern_profiles import (
    THAD_JONES_STYLE,
    ModernBigBandArranger,
    get_style_profile
)

# Create an arranger with a style profile
arranger = ModernBigBandArranger(THAD_JONES_STYLE)

# Get style suggestions
intro_type = arranger.suggest_intro_type()      # "ostinato"
ending_type = arranger.suggest_ending_type()    # "fermata"
tempo = arranger.get_typical_tempo()            # 100-220 BPM

# Generate quartal voicing (Thad Jones signature technique)
voicing = arranger.generate_quartal_voicing(root_note=60, num_voices=4)
# Returns: [60, 65, 70, 75] (C, F, Bb, Eb - stacked 4ths)

# Generate wide spacing voicing
chord_tones = [60, 64, 67, 71]  # Cmaj7
wide_voicing = arranger.generate_wide_spacing_voicing(chord_tones, min_spacing=7)
# Returns: [60, 76, 91, 107] - wide spacing
```

### Retrieving Style Profiles

```python
from styles.modern_profiles import get_style_profile, list_available_styles

# List all available styles
styles = list_available_styles()
# Returns: ["thad_jones", "maria_schneider", "gordon_goodwin"]

# Get a specific profile
profile = get_style_profile("thad_jones")
# or shorthand:
profile = get_style_profile("thad")

# Access profile attributes
print(f"Harmony complexity: {profile.harmony_complexity}")
print(f"Quartal usage: {profile.use_quartal}")
print(f"Tempo range: {profile.typical_tempo_range}")
```

### Comparing Styles

```python
from styles.modern_profiles import compare_style_characteristics

# Print comprehensive comparison of all three styles
compare_style_characteristics()
```

## Style Profiles

### Thad Jones Style

**Era:** Modern (1950s-1980s)

**Key Characteristics:**
- Angular melodies with wide intervals
- Heavy use of quartal harmony (60% of voicings)
- Wide spacing between voices
- Complex, sophisticated harmony (0.85/1.0)
- Moderate to fast tempos (100-220 BPM)

**Signature Techniques:**
- Quartal voicings (stacked 4ths)
- Wide interval spacing (15+ semitones between voices)
- Complex harmonic progressions
- Occasional odd meters (5/4, 7/4)

**Famous Works:**
- "A Child is Born" - Lush harmony
- "Three and One" - Angular, modern
- "The Deacon" - Complex rhythms

**Mood:** Sophisticated, modern, intellectual

```python
# Key attributes
THAD_JONES_STYLE.harmony_complexity  # 0.85
THAD_JONES_STYLE.use_quartal        # 0.6 (60%)
THAD_JONES_STYLE.voicing_spacing    # "wide"
THAD_JONES_STYLE.angular_melodies   # True
```

### Maria Schneider Style

**Era:** Contemporary (1990s-present)

**Key Characteristics:**
- Orchestral colors and impressionistic textures
- Highest harmonic complexity (0.9/1.0)
- Woodwind doublings (signature technique)
- Very wide dynamic range (cinematic)
- Slow to moderate tempos (60-140 BPM)

**Signature Techniques:**
- Woodwind doublings (flute, clarinet)
- Pedal tones (70% usage)
- Impressionistic harmonies
- Layered textures (sparse to dense)

**Famous Works:**
- "Concert in the Garden" - Orchestral colors
- "Coming About" - Cinematic textures
- "Bulería, Soleá y Rumba" - Spanish influences

**Mood:** Atmospheric, cinematic, evocative

```python
# Key attributes
MARIA_SCHNEIDER_STYLE.harmony_complexity  # 0.9 (highest)
MARIA_SCHNEIDER_STYLE.woodwind_doublings  # True (signature)
MARIA_SCHNEIDER_STYLE.use_pedal_tones    # 0.7 (70%)
MARIA_SCHNEIDER_STYLE.impressionistic    # True
MARIA_SCHNEIDER_STYLE.dynamic_range      # "very_wide"
```

### Gordon Goodwin Style

**Era:** Contemporary (1990s-present)

**Key Characteristics:**
- High energy, virtuosic
- Highest rhythmic complexity (0.9/1.0)
- Fast tempos (160-260 BPM)
- Dense texture, powerful tutti sections
- Complex syncopation (0.9/1.0)

**Signature Techniques:**
- Fast tempos (signature: 180-240 BPM)
- Complex rhythmic patterns
- Powerful full ensemble sections
- Metric modulation

**Famous Works:**
- "Hunting Wabbits" - Contemporary swing
- "Rippin' n Runnin'" - Breakneck tempos
- "Hit the Ground Running" - Complex rhythms

**Mood:** Energetic, exciting, virtuosic

```python
# Key attributes
GORDON_GOODWIN_STYLE.rhythmic_complexity  # 0.9 (highest)
GORDON_GOODWIN_STYLE.typical_tempo_range  # (160, 260) BPM
GORDON_GOODWIN_STYLE.syncopation_level    # 0.9
GORDON_GOODWIN_STYLE.texture_density      # "dense"
```

## ModernBigBandArranger

The `ModernBigBandArranger` class provides methods for creating modern big band arrangements.

### Constructor

```python
arranger = ModernBigBandArranger(style_profile)
```

**Parameters:**
- `style_profile` (StyleProfile): One of THAD_JONES_STYLE, MARIA_SCHNEIDER_STYLE, or GORDON_GOODWIN_STYLE

### Methods

#### `arrange(melody, chords, form=None)`

Create a modern big band arrangement.

**Parameters:**
- `melody`: List of melody notes
- `chords`: List of chord events
- `form`: Musical form (optional)

**Returns:** Dictionary with arrangement tracks

#### `generate_quartal_voicing(root_note, num_voices=4)`

Generate quartal (stacked 4ths) voicing.

**Parameters:**
- `root_note` (int): Base MIDI note
- `num_voices` (int): Number of voices (default 4)

**Returns:** List of MIDI note numbers

**Example:**
```python
voicing = arranger.generate_quartal_voicing(60, 4)
# Returns: [60, 65, 70, 75]
# Intervals: [5, 5, 5] (all perfect 4ths)
```

#### `generate_wide_spacing_voicing(chord_tones, min_spacing=7)`

Generate wide-spaced voicing (Thad Jones technique).

**Parameters:**
- `chord_tones` (List[int]): Chord tone MIDI notes
- `min_spacing` (int): Minimum spacing between adjacent voices (semitones)

**Returns:** Wide-spaced voicing

**Example:**
```python
chord_tones = [60, 64, 67, 71]  # Cmaj7
wide_voicing = arranger.generate_wide_spacing_voicing(chord_tones, min_spacing=7)
# Returns: [60, 76, 91, 107]
# Average spacing: ~15 semitones
```

#### `apply_dynamic_shape(notes, section_type="a_section")`

Apply dynamic shaping based on style profile.

**Parameters:**
- `notes`: List of notes to shape
- `section_type` (str): Section type for context

**Returns:** Notes with applied dynamics

#### `suggest_intro_type()`

Suggest intro type based on style profile.

**Returns:** Intro style string ("ostinato", "rubato", "full", etc.)

#### `suggest_ending_type()`

Suggest ending type based on style profile.

**Returns:** Ending style string ("fermata", "fade", "abrupt", etc.)

#### `get_typical_tempo()`

Get typical tempo for this style.

**Returns:** Tempo in BPM (within style's typical range)

## Integration Points

The modern style profiles integrate with existing midi_generator components:

### With BigBandArranger

```python
from transformation.arrangement_engine import BigBandArranger
from styles.modern_profiles import THAD_JONES_STYLE

# Use style profile to configure arrangement
style = THAD_JONES_STYLE

# Set voicing preference
voicing_type = style.voicing_preference  # "quartal_and_spread"

# Set dynamic range
dynamic_range = style.dynamic_range  # "wide"

# Apply to arrangement
arrangement = BigBandArranger.arrange(melody, chords)
```

### With Harmony Generators

```python
from genres.jazz import ComprehensiveHarmonyGenerator
from styles.modern_profiles import MARIA_SCHNEIDER_STYLE

# Use style to inform harmony generation
style = MARIA_SCHNEIDER_STYLE

# High complexity, use pedal tones
use_pedal_tones = style.use_pedal_tones > 0.5  # True for Schneider
complexity = style.harmony_complexity  # 0.9
```

### With generate_professional.py

```python
# Command-line usage (future integration)
python generate_professional.py \
    --style thad_jones \
    --form aaba \
    --tempo 140 \
    --output my_arrangement.mid

# Or Maria Schneider ballad style
python generate_professional.py \
    --style maria_schneider \
    --form through_composed \
    --tempo 80 \
    --output ballad.mid
```

## Validation

The module includes comprehensive validation tests in `validation_test.py`.

### Running Tests

```bash
cd midi_generator
python3 styles/validation_test.py
```

### Test Coverage

The validation suite includes 8 comprehensive tests:

1. **Style Profile Integrity** - Validates all required attributes
2. **Quartal Voicing Generation** - Tests stacked 4ths voicings
3. **Wide Spacing Voicing** - Tests wide interval voicings
4. **Style Retrieval** - Tests get_style_profile() function
5. **Style Comparison** - Tests compare_style_characteristics()
6. **Arranger Creation** - Tests ModernBigBandArranger methods
7. **Harmonic Complexity** - Validates style differences
8. **Tempo Ranges** - Validates tempo characteristics

### Expected Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ALL TESTS PASSED - SYSTEM VALIDATED                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Research Sources

### Thad Jones

1. **Scores and Recordings:**
   - "A Child is Born" (lush harmony, wide spacing)
   - "Three and One" (angular melodies, odd meters)
   - "The Deacon" (complex voice leading)
   - livingjazzarchives.org Thad Jones archive

2. **Analysis:**
   - Quartal harmony techniques (stacked 4ths)
   - Wide interval spacing (signature sound)
   - Modern jazz harmony (altered scales, extensions)

### Maria Schneider

1. **Scores and Recordings:**
   - "Concert in the Garden" (Grammy-winning album)
   - Orchestration masterclasses and interviews
   - Personal website: mariaschneider.com

2. **Techniques:**
   - Woodwind doublings (flute, clarinet)
   - Impressionistic harmonies
   - Cinematic textures
   - Gil Evans influence

### Gordon Goodwin

1. **Scores and Recordings:**
   - "Hunting Wabbits" (high energy)
   - Big Phat Band recordings (1998-present)
   - Grammy-nominated arrangements

2. **Techniques:**
   - Fast tempos (160-260 BPM typical)
   - Complex rhythmic patterns
   - Contemporary swing feel
   - High-energy tutti sections

### Academic References

1. **Modern Jazz Harmony:**
   - Mark Levine: "The Jazz Theory Book"
   - Modern voicing techniques chapter

2. **Orchestration:**
   - Ted Pease & Ken Pullig: "Modern Jazz Voicings"
   - Gary Lindsay: "Jazz Arranging Techniques"

## Future Enhancements

### Planned Features

1. **Additional Arrangers:**
   - Bob Brookmeyer (valve trombone, modern harmony)
   - Gil Evans (impressionistic orchestration)
   - Woody Herman (progressive big band)

2. **Enhanced Voicing Algorithms:**
   - Polychord generation
   - Cluster chord spacing
   - Upper structure triads

3. **MIDI Export Integration:**
   - Direct export to MIDI with style characteristics
   - Integration with AudioWorklet plugins
   - Real-time style parameter control

4. **Machine Learning:**
   - Train on actual scores from each arranger
   - Learn style-specific patterns from MIDI datasets
   - Generate new variations in each style

## License

MIT License - See main repository for details

## Contributing

Contributions welcome! Areas of interest:
- Additional arranger profiles
- Enhanced voicing algorithms
- Integration with other modules
- Validation against actual scores

## Contact

**Agent 15 - Modern Big Band Style Analyzer**
Part of the 20-Agent Big Band Generator Excellence System
Repository: github.com/doseedo/Do

---

*"The goal is not to imitate, but to capture the essence of each arranger's approach to create authentic-sounding arrangements in their style."*
