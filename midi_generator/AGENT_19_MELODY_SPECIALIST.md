# AGENT 19: Melody Specialist

## Overview

The Melody Specialist provides advanced melody analysis and generation capabilities beyond the basic melodic parameters defined in Agent 4. This agent implements specialized melody processing techniques used in composition, analysis, and musical AI systems.

## Features

### 1. Motif Development (10+ Techniques)
- **Inversion**: Mirror motif around pitch axis
- **Retrograde**: Reverse note order
- **Retrograde Inversion**: Combined retrograde and inversion
- **Augmentation**: Lengthen note durations (2x, 3x, etc.)
- **Diminution**: Shorten note durations (0.5x, 0.25x, etc.)
- **Transposition**: Pitch shift by interval
- **Intervallic Expansion**: Widen intervals between notes
- **Intervallic Contraction**: Narrow intervals between notes
- **Rhythmic Displacement**: Shift timing of motif
- **Fragmentation**: Extract portions of motif

### 2. Sequence Generation (6 Types)
- **Ascending**: Sequence rising by interval
- **Descending**: Sequence falling by interval
- **Tonal**: Sequence by scale degrees (diatonic)
- **Real**: Exact chromatic transposition
- **Chromatic**: Sequence by semitones
- **Rosalia**: Classic step-wise sequence

### 3. Contour Optimization (7 Shapes)
- **Arch**: Rise to apex, then fall
- **Inverted Arch**: Fall to nadir, then rise
- **Ascending**: Generally rising line
- **Descending**: Generally falling line
- **Wave**: Undulating pattern
- **Plateau**: Relatively flat
- **Terraced**: Stepwise register changes

### 4. Ornamentation System (10 Types)
- **Trill**: Rapid alternation with upper neighbor
- **Turn**: Upper neighbor, main, lower neighbor, main
- **Mordent**: Main, neighbor, main (upper)
- **Inverted Mordent**: Main, neighbor, main (lower)
- **Appoggiatura**: Accented non-chord tone
- **Acciaccatura**: Quick grace note
- **Grace Note**: Unaccented ornamental note
- **Slide**: Portamento effect
- **Tremolo**: Rapid repetition
- **Glissando**: Pitch sweep

### 5. Phrase Structure Analysis
- **Phrase Segmentation**: Automatic phrase boundary detection
- **Motif Extraction**: Identify repeating patterns
- **Cadence Detection**: Classify phrase endings
  - Authentic cadence
  - Half cadence
  - Plagal cadence
  - Imperfect cadence
- **Phrase Duration Analysis**: Measure phrase lengths

### 6. Comprehensive Melodic Analysis
- **Contour Analysis**: Identify overall shape
- **Intervallic Profile**: Distribution of intervals
- **Chromaticism Score**: Measure chromatic content
- **Stepwise Motion Ratio**: Proportion of steps vs leaps
- **Range Analysis**: Melodic span in semitones
- **Tessitura**: Average pitch register
- **Climax Position**: Location of highest note
- **Direction Changes**: Contour complexity
- **Sequence Detection**: Identify repeated patterns

## File Structure

```
midi_generator/
├── experts/
│   ├── melody_specialist.py      # Main implementation (~2,000 lines)
│   └── __init__.py                # Module exports
├── examples/
│   └── agent19_melody_demo.py    # Usage examples and demos
└── AGENT_19_MELODY_SPECIALIST.md  # This file
```

## Usage Examples

### Basic Usage

```python
from midi_generator.experts.melody_specialist import (
    MelodySpecialist, Note, Motif,
    MotifTransformation, SequenceType, OrnamentType
)

# Initialize specialist
specialist = MelodySpecialist(key="C", mode="major")

# Create a motif
motif = Motif(notes=[
    Note(60, 1.0, 80, 0.0),
    Note(64, 1.0, 80, 1.0),
    Note(67, 1.0, 80, 2.0),
])

# Develop motif
variations = specialist.develop_motif(
    motif,
    techniques=[MotifTransformation.INVERSION, MotifTransformation.RETROGRADE],
    n_variations=3
)
```

### Motif Development

```python
# Apply inversion
inverted = specialist.develop_motif(
    motif,
    [MotifTransformation.INVERSION],
    n_variations=1
)[0]

# Apply retrograde
retrograde = specialist.develop_motif(
    motif,
    [MotifTransformation.RETROGRADE],
    n_variations=1
)[0]

# Apply augmentation (double durations)
augmented = specialist.develop_motif(
    motif,
    [MotifTransformation.AUGMENTATION],
    n_variations=1
)[0]
```

### Sequence Generation

```python
# Generate ascending sequence
sequence = specialist.generate_sequence(
    pattern=motif,
    sequence_type=SequenceType.ASCENDING,
    repetitions=4,
    interval=2  # Up by major 2nd each time
)

# Generate descending chromatic sequence
chromatic_seq = specialist.generate_sequence(
    pattern=motif,
    sequence_type=SequenceType.CHROMATIC,
    repetitions=5,
    interval=1
)
```

### Contour Optimization

```python
from midi_generator.experts.melody_specialist import ContourShape

# Optimize melody to arch shape
optimized = specialist.optimize_contour(
    melody=notes,
    target_shape=ContourShape.ARCH,
    smoothing=0.7  # 0.0-1.0, higher = more transformation
)

# Analyze contour
contour_analysis = specialist.analyze_contour(melody)
print(f"Shape: {contour_analysis['shape']}")
print(f"Apex position: {contour_analysis['apex_position']}")
```

### Ornamentation

```python
# Add trills and turns
ornamented = specialist.add_ornamentation(
    melody=notes,
    ornament_types=[OrnamentType.TRILL, OrnamentType.TURN],
    density=0.3  # Ornament 30% of notes
)

# Add appoggiaturas
with_appogg = specialist.add_ornamentation(
    melody=notes,
    ornament_types=[OrnamentType.APPOGGIATURA],
    density=0.5
)
```

### Phrase Analysis

```python
# Analyze phrase structure
phrases = specialist.analyze_phrase_structure(
    melody=notes,
    phrase_length=4.0  # Expected phrase length in beats
)

for phrase in phrases:
    print(f"Phrase: {phrase.start_time} - {phrase.end_time}")
    print(f"  Motifs: {len(phrase.motifs)}")
    print(f"  Cadence: {phrase.cadence_type}")
```

### Comprehensive Analysis

```python
# Full melodic analysis
analysis = specialist.analyze_melody(notes)

print(f"Contour: {analysis.contour_shape.value}")
print(f"Range: {analysis.range_semitones} semitones")
print(f"Tessitura: MIDI {analysis.tessitura}")
print(f"Stepwise motion: {analysis.stepwise_motion_ratio:.1%}")
print(f"Chromaticism: {analysis.chromaticism_score:.1%}")
print(f"Sequence detected: {analysis.sequence_detected}")

# Intervallic profile
for interval, count in analysis.intervallic_profile.items():
    print(f"  {interval:+d} semitones: {count} occurrences")
```

## Integration with Agent 4 Parameters

Agent 19 provides the **implementation** for many of Agent 4's melody parameters:

### Contour Parameters
- `melody.contour.arch_probability`
- `melody.contour.ascending_prob`
- `melody.contour.climax_placement`
- `melody.contour.apex_note_emphasis`

### Interval Parameters
- `melody.intervals.stepwise_motion_ratio`
- `melody.intervals.leap_resolution`
- `melody.intervals.chromatic_intensity`

### Ornamentation Parameters
- `melody.ornamentation.trill_density`
- `melody.ornamentation.turn_probability`
- `melody.ornamentation.appoggiatura_usage`
- `melody.ornamentation.grace_note_style`

### Phrase Parameters
- `melody.phrase.length_mean`
- `melody.phrase.cadence_strength`
- `melody.phrase.motif_repetition_ratio`

## API Reference

### Classes

#### `MelodySpecialist`
Main class for melody analysis and generation.

**Constructor:**
```python
MelodySpecialist(key: str = "C", mode: str = "major")
```

**Methods:**
- `develop_motif()` - Apply transformation techniques
- `generate_sequence()` - Generate melodic sequences
- `optimize_contour()` - Optimize melodic shape
- `add_ornamentation()` - Add ornaments to melody
- `analyze_phrase_structure()` - Segment into phrases
- `analyze_melody()` - Comprehensive analysis
- `analyze_contour()` - Contour-specific analysis
- `get_statistics()` - Processing statistics
- `reset_statistics()` - Clear statistics

#### `Note`
Represents a musical note.

**Attributes:**
- `pitch` - MIDI note number (0-127)
- `duration` - Duration in beats
- `velocity` - Velocity (0-127)
- `start_time` - Start time in beats
- `articulation` - Articulation marking (optional)
- `ornament` - Ornament type (optional)

**Properties:**
- `pitch_class` - Pitch class (0-11)
- `octave` - Octave number

**Methods:**
- `transpose(semitones)` - Transpose note

#### `Motif`
Represents a melodic motif.

**Attributes:**
- `notes` - List of Note objects
- `name` - Motif name (optional)
- `category` - Motif category (optional)

**Methods:**
- `duration()` - Total duration
- `pitch_intervals()` - Intervals between notes
- `contour()` - Contour as [-1, 0, 1]

#### `Phrase`
Represents a melodic phrase.

**Attributes:**
- `motifs` - List of Motif objects
- `start_time` - Phrase start time
- `end_time` - Phrase end time
- `cadence_type` - Type of cadence

**Methods:**
- `all_notes()` - Get all notes in phrase

#### `MelodicAnalysis`
Results of comprehensive melodic analysis.

**Attributes:**
- `contour_shape` - Overall contour
- `range_semitones` - Melodic range
- `tessitura` - Average pitch
- `climax_position` - Position of highest note (0.0-1.0)
- `intervallic_profile` - Interval distribution
- `chromaticism_score` - Chromatic content (0.0-1.0)
- `stepwise_motion_ratio` - Steps vs leaps
- `leap_count` - Number of leaps
- `direction_changes` - Contour complexity
- `sequence_detected` - Sequence present
- `phrases` - Detected phrases
- `motifs` - Detected motifs

### Enums

#### `MotifTransformation`
- `INVERSION`
- `RETROGRADE`
- `RETROGRADE_INVERSION`
- `AUGMENTATION`
- `DIMINUTION`
- `TRANSPOSITION`
- `INTERVALLIC_EXPANSION`
- `INTERVALLIC_CONTRACTION`
- `RHYTHMIC_DISPLACEMENT`
- `FRAGMENTATION`
- `SEQUENCE`

#### `SequenceType`
- `ASCENDING`
- `DESCENDING`
- `TONAL`
- `REAL`
- `CHROMATIC`
- `DIATONIC`
- `ROSALIA`

#### `OrnamentType`
- `TRILL`
- `TURN`
- `MORDENT`
- `INVERTED_MORDENT`
- `APPOGGIATURA`
- `ACCIACCATURA`
- `GRACE_NOTE`
- `SLIDE`
- `TREMOLO`
- `GLISSANDO`

#### `ContourShape`
- `ARCH`
- `INVERTED_ARCH`
- `ASCENDING`
- `DESCENDING`
- `WAVE`
- `PLATEAU`
- `TERRACED`
- `ZIGZAG`

## Performance Characteristics

- **Motif Development**: O(n) where n = number of notes in motif
- **Sequence Generation**: O(n × r) where r = repetitions
- **Contour Optimization**: O(n) per melody
- **Ornamentation**: O(n × d) where d = density
- **Phrase Analysis**: O(n) with phrase segmentation
- **Comprehensive Analysis**: O(n²) for sequence detection

## Testing

Run the demo:
```bash
python midi_generator/examples/agent19_melody_demo.py
```

Run unit tests:
```bash
pytest tests/test_melody_specialist.py
```

## Statistics Tracking

The specialist tracks processing statistics:
```python
stats = specialist.get_statistics()
# Returns:
# {
#     'motifs_developed': int,
#     'sequences_generated': int,
#     'ornaments_added': int,
#     'phrases_analyzed': int
# }
```

## Dependencies

- `numpy` - Numerical operations
- `dataclasses` - Data structures
- `enum` - Enumerations
- `typing` - Type hints

## Integration Points

### With Agent 4 (Melody Parameters)
- Implements functionality for 50+ melody parameters
- Provides analysis for parameter validation
- Generates examples for parameter training

### With Agent 8 (Feature Extraction)
- Melodic features used for analysis
- Contour features for pattern matching
- Interval profile for style detection

### With Agent 14 (Training Data Generation)
- Generates melodic variations for training
- Creates diverse melodic patterns
- Validates musical coherence

### With Other Specialists
- **Agent 18 (Harmony)**: Melody-harmony relationships
- **Agent 20 (Rhythm)**: Rhythmic aspects of melody
- **Agent 22 (Dynamics)**: Dynamic shaping of melody

## Future Enhancements

1. **Machine Learning Integration**
   - Style-based melody generation
   - Motif similarity detection
   - Automatic variation selection

2. **Advanced Techniques**
   - Schenkerian analysis
   - Set theory operations
   - Twelve-tone row manipulation

3. **Performance Optimization**
   - Caching of analysis results
   - Parallel processing for batch operations
   - GPU acceleration for pattern matching

4. **Extended Ornamentation**
   - Style-specific ornaments
   - Period-appropriate ornaments
   - Cultural ornamentation patterns

## References

- Schoenberg, A. (1967). *Fundamentals of Musical Composition*
- LaRue, J. (1992). *Guidelines for Style Analysis*
- Narmour, E. (1990). *The Analysis and Cognition of Basic Melodic Structures*
- Huron, D. (2006). *Sweet Anticipation: Music and the Psychology of Expectation*

## License

MIT License - See LICENSE file for details

## Author

Agent 19 - Melody Specialist
Part of the Musical Program Synthesis System

## Version History

- **v1.0.0** (2024): Initial implementation
  - 10 motif transformation techniques
  - 6 sequence types
  - 10 ornament types
  - 7 contour shapes
  - Comprehensive analysis system
  - ~2,000 lines of code
  - Full integration with Agent 4

---

**Status**: ✅ Complete - 50+ specialized melody parameters implemented
**Integration**: ✅ Fully integrated with Agent 4's melody foundation
**Testing**: ✅ Comprehensive demo and examples provided
