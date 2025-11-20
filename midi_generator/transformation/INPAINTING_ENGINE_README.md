# MIDI Inpainting Engine

## Content-Aware Fill for Music

The MIDI Inpainting Engine is a comprehensive system for regenerating sections of MIDI files with different chords, genres, or styles while maintaining seamless transitions. Think of it as Photoshop's content-aware fill, but for music.

**Agent 4** - Part of the 10-Agent Modular Fusion Enhancement System

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Concepts](#core-concepts)
6. [API Reference](#api-reference)
7. [Use Cases](#use-cases)
8. [Examples](#examples)
9. [Advanced Techniques](#advanced-techniques)
10. [Integration](#integration)
11. [Research Background](#research-background)
12. [Future Enhancements](#future-enhancements)

---

## Overview

The Inpainting Engine enables:

- **Section Regeneration** - Regenerate specific measures with new musical content
- **Reharmonization** - Change chord progressions while preserving melody and structure
- **Genre Morphing** - Smoothly transition between different musical styles
- **Boundary Smoothing** - Seamless voice leading at section boundaries
- **Melody Preservation** - Keep melodic content while changing harmony
- **Context-Aware Generation** - Analyze surrounding material for coherent results

### What Makes It Unique?

Unlike simple MIDI editing or style transfer, the Inpainting Engine:

1. **Analyzes Context** - Understands the musical material surrounding the inpaint region
2. **Smooth Boundaries** - Uses voice leading rules to create seamless transitions
3. **Genre-Aware** - Can regenerate in different styles while maintaining coherence
4. **Preserves Intent** - Optional preservation of rhythm, melody, or other elements
5. **Professional Results** - Based on music theory and reharmonization techniques

---

## Key Features

### 1. InpaintingEngine

The main engine for section regeneration:

```python
from midi_generator.transformation.inpainting_engine import InpaintingEngine

engine = InpaintingEngine("song.mid")
engine.analyze()

# Reharmonize measures 9-16 with jazz chords
regenerated = engine.inpaint_measures(
    track_numbers=[1, 2],
    start_measure=9,
    end_measure=16,
    new_chords=['Dm9', 'G7#9', 'Cmaj9#11', 'A7alt',
                'Dm9', 'Db7#11', 'Cmaj9#11', 'Cmaj9#11']
)

engine.export("song_reharmonized.mid")
```

### 2. ChordSubstitutionEngine

Advanced chord substitution and reharmonization:

```python
from midi_generator.transformation.inpainting_engine import ChordSubstitutionEngine

# Tritone substitution
sub = ChordSubstitutionEngine.tritone_substitute('G7')  # → Db7

# Secondary dominants
dom, target = ChordSubstitutionEngine.secondary_dominant('Dm7', 'C')  # → (A7, Dm7)

# Reharmonize entire progression
original = ['Dm7', 'G7', 'Cmaj7', 'Cmaj7']
jazzed = ChordSubstitutionEngine.reharmonize(original, style='jazz')
# Result: ['Dm9', 'G7#9', 'Cmaj9#11', 'Cmaj9#11']
```

### 3. StyleTransitionBlender

Smooth transitions between genres:

```python
from midi_generator.transformation.inpainting_engine import StyleTransitionBlender

# 4-measure transition from jazz to EDM
blended = StyleTransitionBlender.blend_styles(
    style_a='jazz',
    style_b='edm',
    blend_measures=4,
    blend_type='linear'  # or 'exponential', 's-curve'
)
```

### 4. MelodyPreserver

Preserve melody while changing harmony:

```python
from midi_generator.transformation.inpainting_engine import MelodyPreserver

preserver = MelodyPreserver(melody_notes, original_chords)
adjusted = preserver.reharmonize(
    new_chords=['Am7', 'D7', 'Gmaj7', 'Gmaj7'],
    adjustment_strategy='minimal'  # or 'chord_tones', 'chromatic'
)
```

---

## Installation

The Inpainting Engine is part of the HarmonyModule library:

```bash
# Clone the repository
git clone https://github.com/doseedo/Do
cd Do/home/arlo/harmonymodule

# Install dependencies
pip install mido numpy

# Run tests
python midi_generator/transformation/test_inpainting_engine.py
```

---

## Quick Start

### Basic Reharmonization

```python
from midi_generator.transformation.inpainting_engine import InpaintingEngine

# 1. Load MIDI file
engine = InpaintingEngine("input.mid")

# 2. Analyze the file
analysis = engine.analyze()
print(f"Detected {analysis['num_measures']} measures")
print(f"Original chords: {analysis['chords']}")

# 3. Define new chord progression
new_chords = ['Cmaj9', 'Am9', 'Dm9', 'G7#9']

# 4. Reharmonize measures 1-4
engine.inpaint_measures(
    track_numbers=[0, 1],  # Which tracks to regenerate
    start_measure=1,
    end_measure=4,
    new_chords=new_chords
)

# 5. Export result
engine.export("output.mid")
```

### Genre Change

```python
# Change measures 9-16 to EDM style
engine.inpaint_measures(
    track_numbers=[0, 1, 2],
    start_measure=9,
    end_measure=16,
    new_genre='edm',  # Use EDM style characteristics
    new_chords=None   # Keep original chord progression
)
```

### Preserve Melody

```python
# Reharmonize while keeping melody intact
engine.inpaint_measures(
    track_numbers=[0, 1],
    start_measure=5,
    end_measure=8,
    new_chords=['Dm7', 'G7', 'Cmaj7', 'A7'],
    preserve_melody=True  # Melody track stays the same
)
```

---

## Core Concepts

### Boundary Context

The engine analyzes measures before and after the inpaint region to ensure smooth transitions:

```python
@dataclass
class BoundaryContext:
    measure_number: int
    chord: Optional[str]                    # Harmonic context
    last_pitches: List[int]                 # Voice leading info
    rhythm_density: float                   # Notes per beat
    melodic_direction: str                  # 'ascending', 'descending'
    average_velocity: int                   # Dynamic level
```

This enables:
- **Stepwise voice leading** at entry/exit points
- **Rhythmic consistency** across boundaries
- **Dynamic continuity** (no sudden volume changes)
- **Harmonic preparation** (anticipating chord changes)

### Section Analysis

Before regenerating, the engine analyzes the section's musical characteristics:

```python
@dataclass
class SectionAnalysis:
    chords: List[str]                       # Chord progression
    harmonic_rhythm: float                  # Chords per measure
    tempo: int                              # BPM
    time_signature: Tuple[int, int]         # (numerator, denominator)
    density_per_measure: List[float]        # Rhythmic density profile
    melodic_range: Tuple[int, int]          # MIDI note range
    interval_distribution: Dict[str, float] # Step/leap ratios
```

This ensures regenerated material matches the original's:
- Rhythmic activity level
- Melodic contour patterns
- Harmonic complexity
- Dynamic range

### Voice Leading

The engine applies classical voice leading rules:

1. **Contrary Motion** - Voices move in opposite directions
2. **Stepwise Motion** - Prefer small intervals between chords
3. **Avoid Parallels** - No parallel fifths or octaves
4. **Common Tones** - Preserve common tones between chords
5. **Smooth Transitions** - Minimize melodic leaps

Example:
```
Entry Context: Last note is C (MIDI 60)
Generated First Note: Would be E (64)
Adjustment: Change to D (62) for stepwise motion
Result: Smooth C → D transition
```

---

## API Reference

### InpaintingEngine

#### Constructor

```python
InpaintingEngine(midi_file: str)
```

Creates an inpainting engine for the given MIDI file.

**Parameters:**
- `midi_file` (str): Path to MIDI file

#### Methods

##### `analyze() -> Dict[str, Any]`

Analyzes the MIDI file and builds internal data structures.

**Returns:**
- Dictionary with analysis results:
  - `key`: Detected key signature
  - `tempo`: Average tempo in BPM
  - `time_signatures`: List of time signature changes
  - `num_measures`: Number of measures in the file
  - `num_tracks`: Number of tracks
  - `chords`: Detected chord progression

**Example:**
```python
analysis = engine.analyze()
print(f"Key: {analysis['key']}")
print(f"Tempo: {analysis['tempo']} BPM")
print(f"Measures: {analysis['num_measures']}")
```

##### `inpaint_measures(...) -> Dict[int, List[NoteEvent]]`

Regenerate measures with new chords/style.

**Parameters:**
- `track_numbers` (List[int]): Which tracks to regenerate
- `start_measure` (int): Start measure (1-indexed)
- `end_measure` (int): End measure (inclusive)
- `new_chords` (Optional[List[str]]): New chord progression (None = keep existing)
- `new_genre` (Optional[str]): New genre (None = keep existing)
- `preserve_rhythm` (bool): Keep rhythmic pattern, change only pitches
- `preserve_melody` (bool): Keep melody, change only harmony

**Returns:**
- Dictionary mapping track numbers to lists of regenerated notes

**Example:**
```python
regenerated = engine.inpaint_measures(
    track_numbers=[0, 1],
    start_measure=9,
    end_measure=16,
    new_chords=['Dm7', 'G7', 'Cmaj7', 'A7'] * 2,
    preserve_rhythm=True
)
```

##### `export(output_path: str)`

Export modified MIDI to file.

**Parameters:**
- `output_path` (str): Path for output MIDI file

**Example:**
```python
engine.export("output_reharmonized.mid")
```

### ChordSubstitutionEngine

All methods are static.

##### `tritone_substitute(chord: str) -> str`

Applies tritone substitution to dominant 7th chords.

**Parameters:**
- `chord` (str): Chord symbol (e.g., 'G7')

**Returns:**
- Substituted chord (e.g., 'Db7')

**Example:**
```python
sub = ChordSubstitutionEngine.tritone_substitute('G7')  # → 'Db7'
```

##### `secondary_dominant(chord: str, key: str = 'C') -> Tuple[str, str]`

Creates secondary dominant before a chord.

**Parameters:**
- `chord` (str): Target chord
- `key` (str): Key context

**Returns:**
- Tuple of (secondary_dominant, target_chord)

**Example:**
```python
dom, target = ChordSubstitutionEngine.secondary_dominant('Dm7', 'C')
# Result: ('A7', 'Dm7')
```

##### `extended_harmony(chord: str, extensions: List[int]) -> str`

Adds extensions to a chord.

**Parameters:**
- `chord` (str): Base chord
- `extensions` (List[int]): Extension numbers (9, 11, 13)

**Returns:**
- Extended chord symbol

**Example:**
```python
extended = ChordSubstitutionEngine.extended_harmony('Cmaj7', [9, 11])
# Result: 'Cmaj9#11'
```

##### `reharmonize(chord_progression: List[str], style: str = 'jazz', key: str = 'C') -> List[str]`

Reharmonizes an entire progression.

**Parameters:**
- `chord_progression` (List[str]): Original chords
- `style` (str): Reharmonization style ('jazz', 'romantic', 'chromatic', 'modern')
- `key` (str): Key context

**Returns:**
- Reharmonized progression

**Example:**
```python
original = ['Dm7', 'G7', 'Cmaj7']
jazzed = ChordSubstitutionEngine.reharmonize(original, style='jazz')
# Result: ['Dm9', 'G7#9', 'Cmaj9#11']
```

### StyleTransitionBlender

##### `blend_styles(style_a: str, style_b: str, blend_measures: int = 2, blend_type: str = 'linear') -> List[Dict]`

Creates gradual transition from one style to another.

**Parameters:**
- `style_a` (str): Starting genre
- `style_b` (str): Ending genre
- `blend_measures` (int): Number of measures for transition
- `blend_type` (str): 'linear', 'exponential', or 's-curve'

**Returns:**
- List of blended style parameters, one per measure

**Example:**
```python
blended = StyleTransitionBlender.blend_styles(
    style_a='jazz',
    style_b='funk',
    blend_measures=4,
    blend_type='linear'
)

# Use blended styles for generation
for i, style_params in enumerate(blended):
    print(f"Measure {i+1}: {style_params['weight_a']*100}% jazz, "
          f"{style_params['weight_b']*100}% funk")
```

### MelodyPreserver

#### Constructor

```python
MelodyPreserver(melody: List[NoteEvent], original_chords: List[str])
```

**Parameters:**
- `melody` (List[NoteEvent]): Melody notes to preserve
- `original_chords` (List[str]): Original chord progression

#### Methods

##### `reharmonize(new_chords: List[str], adjustment_strategy: str = 'minimal') -> List[NoteEvent]`

Adjusts melody to fit new chords.

**Parameters:**
- `new_chords` (List[str]): New chord progression
- `adjustment_strategy` (str): 'minimal', 'chord_tones', or 'chromatic'

**Returns:**
- Adjusted melody notes

**Strategies:**
- `minimal`: Only adjust notes that clash badly
- `chord_tones`: Prefer chord tones, adjust passing tones
- `chromatic`: Allow chromatic alterations

**Example:**
```python
preserver = MelodyPreserver(melody_notes, ['Cmaj7', 'Fmaj7', 'G7'])
adjusted = preserver.reharmonize(
    new_chords=['Dm7', 'G7', 'Cmaj7'],
    adjustment_strategy='chord_tones'
)
```

---

## Use Cases

### 1. Jazz Reharmonization

Transform simple progressions into rich jazz harmony:

```python
engine = InpaintingEngine("standards.mid")
engine.analyze()

# Transform ii-V-I to more complex jazz changes
original = ['Dm7', 'G7', 'Cmaj7', 'Cmaj7']
jazzed = ChordSubstitutionEngine.reharmonize(original, style='jazz')

engine.inpaint_measures(
    track_numbers=[0, 1],
    start_measure=1,
    end_measure=4,
    new_chords=jazzed
)

engine.export("standards_reharmonized.mid")
```

### 2. Create Arrangement Variations

Generate variations of repeated sections:

```python
# Original verse: measures 1-8
# Generate variation for second verse: measures 17-24

engine.inpaint_measures(
    track_numbers=[0, 1, 2],
    start_measure=17,
    end_measure=24,
    new_chords=ChordSubstitutionEngine.reharmonize(
        original_chords,
        style='chromatic'  # Add chromatic passing chords
    )
)
```

### 3. Genre Morphing

Smoothly transition between genres:

```python
# Measures 1-8: Jazz
# Measures 9-12: Transition (jazz → EDM)
# Measures 13-20: EDM

# Create the transition
blended = StyleTransitionBlender.blend_styles('jazz', 'edm', 4)

engine.inpaint_measures(
    track_numbers=[0, 1, 2],
    start_measure=9,
    end_measure=12,
    new_genre='edm'
)
```

### 4. Fix Problematic Progressions

Replace awkward chord progressions:

```python
# Replace measures 5-8 with smoother voice leading
better_chords = ['Dm7', 'G7', 'Em7', 'Am7']

engine.inpaint_measures(
    track_numbers=[0, 1],
    start_measure=5,
    end_measure=8,
    new_chords=better_chords
)
```

### 5. Add Harmonic Interest

Spice up static harmony:

```python
# Original: Cmaj7 for 4 measures
# New: Cmaj7 → Cmaj9 → Cmaj9#11 → Cmaj13#11

static = ['Cmaj7', 'Cmaj7', 'Cmaj7', 'Cmaj7']
interesting = [
    'Cmaj7',
    ChordSubstitutionEngine.extended_harmony('Cmaj7', [9]),
    ChordSubstitutionEngine.extended_harmony('Cmaj7', [9, 11]),
    ChordSubstitutionEngine.extended_harmony('Cmaj7', [9, 11, 13])
]

engine.inpaint_measures(
    track_numbers=[1],
    start_measure=1,
    end_measure=4,
    new_chords=interesting
)
```

---

## Advanced Techniques

### Custom Boundary Smoothing

For more control over transitions:

```python
# Extract contexts manually
entry = engine._extract_entry_context([0, 1], measure=8)
exit = engine._extract_exit_context([0, 1], measure=17)

# Analyze entry pitch: aim for smooth voice leading
entry_pitch = entry[0].last_pitches[-1]
print(f"Entry pitch: {entry_pitch}, use nearby chord tones")

# Generate with this knowledge
regenerated = engine.inpaint_measures(
    track_numbers=[0, 1],
    start_measure=9,
    end_measure=16,
    new_chords=custom_chords
)

# Manually verify smoothness
first_pitch = regenerated[0][0].pitch
print(f"Interval: {abs(first_pitch - entry_pitch)} semitones")
```

### Multi-Genre Sections

Different genres per section:

```python
# Verse: Jazz (measures 1-8)
engine.inpaint_measures(
    track_numbers=[0, 1, 2],
    start_measure=1,
    end_measure=8,
    new_genre='jazz'
)

# Chorus: Funk (measures 9-16)
engine.inpaint_measures(
    track_numbers=[0, 1, 2],
    start_measure=9,
    end_measure=16,
    new_genre='funk'
)

# Bridge: EDM (measures 17-24)
engine.inpaint_measures(
    track_numbers=[0, 1, 2],
    start_measure=17,
    end_measure=24,
    new_genre='edm'
)
```

### Preserve Specific Elements

Fine-grained preservation:

```python
# Track 0: Melody (preserve)
# Track 1: Harmony (regenerate)
# Track 2: Bass (regenerate with new rhythm)

# Preserve melody
engine.inpaint_measures(
    track_numbers=[0],
    start_measure=1,
    end_measure=8,
    new_chords=new_chords,
    preserve_melody=True
)

# Regenerate harmony
engine.inpaint_measures(
    track_numbers=[1],
    start_measure=1,
    end_measure=8,
    new_chords=new_chords,
    preserve_rhythm=False
)

# Regenerate bass with new rhythm
engine.inpaint_measures(
    track_numbers=[2],
    start_measure=1,
    end_measure=8,
    new_chords=new_chords,
    preserve_rhythm=False
)
```

---

## Integration

### With Genre Detector (Agent 1)

```python
from midi_generator.analysis.genre_detector import GenreDetector

# Detect current genre
detector = GenreDetector("input.mid")
current_genre = detector.classify_genre(top_n=1)[0][0]

print(f"Current genre: {current_genre}")

# Reharmonize in detected style
chords = detector.extract_chord_progression()
reharmonized = ChordSubstitutionEngine.reharmonize(
    chords,
    style=current_genre
)
```

### With Component System (Agent 2)

```python
from midi_generator.core.component_system import CompositionBuilder, ComponentType

# Use inpainting for component-level regeneration
builder = CompositionBuilder()
builder.add_component(ComponentType.HARMONY, genre='jazz')
builder.add_component(ComponentType.RHYTHM, genre='funk')

# Build composition
comp = builder.build()

# Use inpainting to refine specific sections
# (integration details would be in full implementation)
```

### With Context-Aware Generator (Agent 3)

```python
from midi_generator.generators.context_aware_generator import ContextAwareGenerator

# Combine context-aware generation with inpainting
context_gen = ContextAwareGenerator("existing.mid")
context_gen.analyze()

# Generate new track
new_track = context_gen.add_track(instrument=34, track_type='bass')

# Refine with inpainting
engine = InpaintingEngine("existing.mid")
# ... inpaint specific sections
```

---

## Research Background

The Inpainting Engine is based on research from multiple domains:

### 1. Image Inpainting Algorithms

**Content-Aware Fill (Photoshop)**
- Analyze surrounding pixels to fill missing regions
- Adapted for music: analyze surrounding measures for context

**Boundary Smoothing**
- Smooth transitions at boundaries using gradient analysis
- Adapted: voice leading rules for smooth pitch transitions

**Context Propagation**
- Use context from multiple directions (before/after, above/below)
- Adapted: harmonic, rhythmic, and melodic context

### 2. Music Reharmonization

**Barry Harris Method**
- Diminished scale approaches
- Sixth diminished concept
- Implemented in ChordSubstitutionEngine

**Mark Levine - "The Jazz Theory Book"**
- Chord substitution rules
- Voice leading guidelines
- Tritone substitution, secondary dominants

**Functional Harmony**
- Tonic-Subdominant-Dominant functions
- Modal interchange and borrowed chords
- Used for context-aware chord selection

### 3. Style Transfer

**Neural Style Transfer for Music (2024 AAAI)**
- Separate content (melody, structure) from style (harmony, instrumentation)
- Applied in StyleTransitionBlender

**Timbre-Invariant Representations**
- Preserve melodic contour while changing harmony
- Implemented in MelodyPreserver

### 4. Voice Leading Theory

**Dmitri Tymoczko - "A Geometry of Music"**
- Voice leading spaces
- Efficient voice leading (minimal motion)
- Applied in boundary smoothing

**Fux - "Gradus ad Parnassum"**
- Classical counterpoint rules
- No parallel fifths/octaves
- Used for voice leading constraints

---

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**
   - Train on successful reharmonizations
   - Learn style-specific voice leading patterns
   - Automated suggestion of optimal chord substitutions

2. **Advanced Rhythm Preservation**
   - Preserve groove and microtiming
   - Swing factor preservation
   - Syncopation patterns

3. **Multi-Track Voice Leading**
   - Coordinate voice leading across all tracks
   - Prevent voice crossing
   - Optimize register distribution

4. **Genre-Specific Generators**
   - Use specialized generators from library
   - Jazz: walking bass, comp rhythms
   - Funk: syncopated bass lines, chicken scratch guitar
   - EDM: synthesizer arpeggios, sidechaining

5. **Interactive Mode**
   - Real-time preview of inpainting
   - Undo/redo functionality
   - A/B comparison

6. **Harmonic Analysis**
   - Roman numeral analysis
   - Functional harmony labeling
   - Chord complexity metrics

### Community Contributions

We welcome contributions! Areas where help is needed:

- **Genre Profiles** - Add more genre-specific reharmonization styles
- **Voice Leading Rules** - Implement more sophisticated voice leading
- **Test Coverage** - Expand test suite with more musical examples
- **Documentation** - Add more examples and tutorials
- **Integration** - Connect with other modules in the library

---

## Troubleshooting

### Common Issues

**Issue: Regenerated section doesn't sound good**

Solution:
- Check boundary contexts - may need manual adjustment
- Try different `blend_measures` values
- Use `preserve_rhythm=True` to maintain rhythmic feel
- Analyze the section first to understand its characteristics

**Issue: Voice leading sounds jumpy**

Solution:
- Increase smoothing at boundaries
- Use stepwise chord progressions
- Check register of generated notes
- Ensure chord tones are in appropriate range

**Issue: Genre transitions too abrupt**

Solution:
- Increase `blend_measures` value
- Use 's-curve' blend type for smoother transitions
- Manually craft intermediate style parameters

**Issue: Melody clashes with new harmony**

Solution:
- Use `MelodyPreserver` with `adjustment_strategy='chord_tones'`
- Manually check for melodic notes that don't fit chords
- Consider using passing tones or chromatic approach notes

---

## License

Part of the HarmonyModule library - see main repository for license.

---

## Contact

For questions, issues, or contributions:
- GitHub: https://github.com/doseedo/Do
- Issues: https://github.com/doseedo/Do/issues

---

## Acknowledgments

Research foundation:
- David Cope (EMI - Experiments in Musical Intelligence)
- Mark Levine (The Jazz Theory Book)
- Dmitri Tymoczko (Geometry of Music)
- Barry Harris (Harmonic Movement)
- Richard Cohn (Neo-Riemannian Theory)

Inspiration from:
- Photoshop Content-Aware Fill
- Audio editing tools (iZotope RX Spectral Repair)
- Music notation software (Dorico, Sibelius)

---

**Built with ❤️ by Agent 4 as part of the 10-Agent Modular Fusion Enhancement System**
