# Big Band Generator API - User Guide

## Introduction

The Big Band Generator provides a simple, powerful API for creating professional big band arrangements in various styles (Count Basie, Duke Ellington, Thad Jones, etc.).

This system integrates all modules from the 20-agent big band excellence system into a cohesive, easy-to-use interface.

## Quick Start

### Python API

```python
from api import BigBandGenerator

# Simple generation
generator = BigBandGenerator(style="basie", tempo=140)
midi = generator.generate()
midi.save("basie_swing.mid")
```

### Command Line

```bash
# Simple usage
python midi_generator/tools/big_band/generate_big_band.py --style basie --tempo 140

# Full options
python midi_generator/tools/big_band/generate_big_band.py \
    --style ellington \
    --tempo 120 \
    --key Eb \
    --form aaba \
    --output my_arrangement.mid
```

## Available Styles

### Count Basie
- **Era**: Swing (1930s-1950s)
- **Characteristics**:
  - Simple, riff-based arrangements
  - Powerful rhythm section (Freddie Green guitar, feathered kick drum)
  - Punchy brass section hits
  - Sparse piano comping (Basie's famous minimalism)
  - Open, spread voicings
  - Blues-based harmony
- **Typical Tempo**: 120-180 BPM
- **Best For**: Swing tunes, blues, up-tempo swingers

```python
generator = BigBandGenerator(style="basie", tempo=140)
```

### Duke Ellington
- **Era**: Swing (1920s-1970s)
- **Characteristics**:
  - Complex, exotic harmonies (whole tone, diminished, bitonal)
  - Rich orchestral colors
  - Plunger mute brass (Bubber Miley, Cootie Williams signature)
  - Growls and "jungle" sounds
  - Unusual instrument doublings
  - Wide dynamic range
  - Rich chord extensions (9ths, 11ths, 13ths)
- **Typical Tempo**: 80-160 BPM (wide range)
- **Best For**: Ballads, exotic/orchestral pieces, rich harmony

```python
generator = BigBandGenerator(style="ellington", tempo=80, key="Eb")
```

### Thad Jones / Modern
- **Era**: Modern (1960s-present)
- **Characteristics**:
  - Modern harmony (quartal voicings, clusters)
  - Wide interval voicings (not close like swing era)
  - Angular, contemporary melodies
  - Sophisticated voice leading
  - Varied textures (sparse to dense)
  - Contemporary jazz harmony
- **Typical Tempo**: 60-200 BPM (very wide range)
- **Best For**: Modern jazz, ballads, contemporary arrangements

```python
generator = BigBandGenerator(style="thad_jones", tempo=160)
```

## API Reference

### BigBandGenerator Class

```python
class BigBandGenerator:
    def __init__(self,
                 style: str = "basie",
                 tempo: int = 140,
                 key: Union[str, int] = "C",
                 form: str = "aaba",
                 progression_type: str = "jazz_blues",
                 swing_ratio: Optional[float] = None):
```

**Parameters:**
- `style` (str): Arranging style - "basie", "ellington", "thad_jones", or "modern"
- `tempo` (int): Tempo in BPM (typical range: 60-200)
- `key` (str or int): Key signature - "C", "Eb", "F#", etc. or MIDI note number (0-11)
- `form` (str): Musical form - "aaba" (32 bars) or "blues" (12 bars)
- `progression_type` (str): Chord progression type (see Progression Types below)
- `swing_ratio` (float, optional): Override swing ratio (0.5=straight, 0.67=triplet, default=style-specific)

**Methods:**

```python
def generate(self) -> MidiFile:
    """Generate complete big band arrangement. Returns MIDI file."""
```

**Attributes:**
- `form`: Generated MusicalForm structure
- `progression`: Generated chord progression (list of JazzChord)
- `arrangement`: Generated arrangement (dict mapping section names to note lists)
- `style_profile`: StyleProfile for the selected style

### Convenience Functions

```python
from api import generate_big_band, list_available_styles

# Quick generation
midi = generate_big_band("basie", 140, "C", "output.mid")

# List styles
styles = list_available_styles()  # ['basie', 'ellington', 'thad_jones', 'modern']
```

## Musical Parameters

### Keys

Specify key as a string or MIDI note number:
- `"C"` = 0
- `"C#"` or `"Db"` = 1
- `"D"` = 2
- `"Eb"` = 3
- `"E"` = 4
- `"F"` = 5
- `"F#"` or `"Gb"` = 6
- `"G"` = 7
- `"Ab"` = 8
- `"A"` = 9
- `"Bb"` = 10
- `"B"` = 11

### Forms

**AABA** (32 bars):
- A1 (8 bars) - Main theme
- A2 (8 bars) - Repeat of theme
- B (8 bars) - Bridge (contrasting section)
- A3 (8 bars) - Return of theme (often with variations)

**Blues** (12 bars):
- Traditional 12-bar blues form

### Progression Types

Common progression types (more available in advanced harmony module):

**Basic Jazz:**
- `jazz_blues` - Standard jazz blues changes
- `rhythm_changes` - "I Got Rhythm" changes (Gershwin)
- `ii_V_I` - Basic ii-V-I progression
- `minor_ii_V_i` - Minor key ii-V-i

**Jazz Standards:**
- `autumn_leaves` - "Autumn Leaves" changes
- `take_five` - "Take Five" progression
- `blue_bossa` - "Blue Bossa" changes
- `coltrane_changes` - Giant Steps changes (advanced)

**Modal:**
- `dorian_vamp` - Dorian mode vamp
- `mixolydian_rock` - Mixolydian rock progression
- `lydian_dream` - Lydian progression

**Advanced:**
- `modal_interchange` - Modal interchange progression
- `reharmonized_blues` - Blues with reharmonization
- `quartal_harmony` - Quartal harmony progression

### Swing Ratios

The swing ratio determines the "swing feel":
- `0.50` - Straight 8ths (no swing)
- `0.56-0.58` - Light swing
- `0.60-0.64` - Medium swing (standard jazz)
- `0.65-0.67` - Heavy swing (close to triplet)

Each style has a default ratio, but you can override:

```python
generator = BigBandGenerator(
    style="basie",
    tempo=140,
    swing_ratio=0.67  # Heavy swing
)
```

## Complete Examples

### Example 1: Count Basie Swing

```python
from api import BigBandGenerator

# Classic Basie swing at 180 BPM
generator = BigBandGenerator(
    style="basie",
    tempo=180,
    key="Bb",
    form="blues",
    progression_type="jazz_blues"
)

midi = generator.generate()
midi.save("basie_blues.mid")

# Print arrangement info
print(f"Form: {generator.form.total_bars} bars")
print(f"Chords: {len(generator.progression)}")
print(f"Sections: {list(generator.arrangement.keys())}")
```

### Example 2: Duke Ellington Ballad

```python
from api import BigBandGenerator

# Lush Ellington ballad in Eb
generator = BigBandGenerator(
    style="ellington",
    tempo=80,
    key="Eb",
    form="aaba",
    progression_type="autumn_leaves"
)

midi = generator.generate()
midi.save("ellington_ballad.mid")

# Check style characteristics
print(f"Harmonic complexity: {generator.style_profile.harmony_complexity}")
print(f"Plunger mute probability: {generator.style_profile.use_plunger_mutes}")
print(f"Chord extensions: {generator.style_profile.chord_extensions}")
```

### Example 3: Modern Big Band

```python
from api import BigBandGenerator

# Modern arrangement with Coltrane changes
generator = BigBandGenerator(
    style="thad_jones",
    tempo=160,
    key="C",
    form="aaba",
    progression_type="coltrane_changes"
)

midi = generator.generate()
midi.save("modern_coltrane.mid")
```

### Example 4: Batch Generation

```python
from api import BigBandGenerator

styles = ["basie", "ellington", "thad_jones"]
tempos = [140, 100, 160]

for style, tempo in zip(styles, tempos):
    generator = BigBandGenerator(
        style=style,
        tempo=tempo,
        key="C",
        form="blues"
    )

    midi = generator.generate()
    midi.save(f"{style}_blues_{tempo}bpm.mid")
    print(f"✓ Generated {style} at {tempo} BPM")
```

## Generated Arrangement Structure

The generated arrangement includes the following sections:

- **Lead**: Lead melody line (alto sax or trumpet)
- **Saxes**: 5-part sax soli (close harmony)
- **Brass**: Brass section (background figures)
- **Piano**: Piano comping (style-dependent: sparse for Basie, rich for Ellington)
- **Bass**: Walking bass line
- **Drums**: Swing drum pattern (ride, hi-hat, kick)

Access the arrangement after generation:

```python
arrangement = generator.arrangement

lead_notes = arrangement['lead']      # List[NoteEvent]
sax_notes = arrangement['saxes']      # List[NoteEvent]
brass_notes = arrangement['brass']    # List[NoteEvent]
piano_notes = arrangement['piano']    # List[NoteEvent]
bass_notes = arrangement['bass']      # List[NoteEvent]
drum_notes = arrangement['drums']     # List[NoteEvent]
```

## Advanced Features

### Accessing Style Profiles

```python
from api.styles import get_style, BASIE_STYLE, ELLINGTON_STYLE

# Get style profile
basie = get_style("basie")

# Inspect characteristics
print(f"Piano style: {basie.piano_style}")
print(f"Harmony complexity: {basie.harmony_complexity}")
print(f"Swing ratio: {basie.swing_ratio}")
print(f"Typical tempo range: {basie.typical_tempo_range}")
```

### Custom Style Configuration (Advanced)

```python
from api.styles import StyleProfile
from api import BigBandGenerator

# Create custom style profile
custom_style = StyleProfile(
    style_name="My Custom Style",
    harmony_complexity=0.7,
    swing_ratio=0.65,
    piano_style="comping",
    voicing_preference="spread",
    # ... many more parameters
)

# Use with generator (requires modifying registry)
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the correct directory:

```bash
cd /path/to/Do
export PYTHONPATH=/path/to/Do:$PYTHONPATH
python -c "from api import BigBandGenerator"
```

### Tempo Warnings

If you use a tempo outside the typical range for a style, you'll get a warning:

```
UserWarning: Count Basie typically played at 120-180 BPM, but 60 BPM requested. Proceeding anyway.
```

This is informational - the generation will still work.

### Module Not Found

Some advanced features require modules that may not be implemented yet. The system will print messages like:

```
⊗ Step 6: Articulations (not yet available)
```

This indicates the feature is planned but not yet implemented by the corresponding agent.

## Future Enhancements

As other agents complete their modules, new features will automatically become available:

- **Agent 1**: Enhanced bebop vocabulary
- **Agent 2**: Drop-2 and drop-3 sax voicings
- **Agent 3**: Authentic stride piano
- **Agent 4**: Advanced reharmonization
- **Agent 5**: Professional brass writing
- **Agent 6**: Enhanced walking bass
- **Agent 7**: Complete drum pattern library
- **Agent 8**: Articulations (falls, rips, growls)
- **Agent 9**: Dynamic shaping and phrasing
- **Agent 10**: Intro/outro generation
- **Agent 11**: Voice leading optimization
- **Agent 12**: Enhanced swing calibration

The integration layer requires NO changes - it automatically detects and uses new modules!

## Support

For issues, questions, or feature requests:
- GitHub Issues: https://github.com/doseedo/Do/issues
- Documentation: `/home/user/Do/midi_generator/docs/`

## Credits

**Author**: Agent 18 - Integration Architecture Designer
**Version**: 1.0.0
**Date**: 2025-11-20

Built on the 20-Agent Big Band Excellence System as specified in `MASTER_PROMPT_20_AGENTS.md`.
