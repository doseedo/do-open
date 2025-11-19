# Multi-Genre Arranger - Track-Level Genre Control System

**Agent 9: Track-Level Genre Control**

A comprehensive system for creating seamless multi-genre arrangements where different tracks can use different genres while maintaining harmonic compatibility, rhythmic synchronization, and proper voice leading.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Research Foundation](#research-foundation)
4. [Quick Start](#quick-start)
5. [Core Concepts](#core-concepts)
6. [API Reference](#api-reference)
7. [Examples](#examples)
8. [Integration](#integration)
9. [Best Practices](#best-practices)
10. [Advanced Usage](#advanced-usage)

---

## Overview

The Multi-Genre Arranger enables Photoshop-level control for music generation, allowing you to assign different genres to different tracks in your arrangement. Unlike simple style fusion that blends genres uniformly, this system provides **per-track genre control** while ensuring:

- **Harmonic Compatibility**: All tracks follow the same underlying harmonic structure
- **Rhythmic Synchronization**: Tracks stay in sync despite genre-specific timing characteristics
- **Voice Leading**: Smooth motion across style boundaries
- **Register Balance**: Intelligent frequency allocation to avoid muddiness

### Example Use Cases

- **Track 1 (Bass)**: Funk style with syncopation and slap technique
- **Track 2 (Piano)**: Jazz style with extended harmonies
- **Track 3 (Drums)**: Hip-hop style with laid-back timing
- **Track 4 (Strings)**: Classical style with smooth voice leading

**Result**: A coherent fusion arrangement where each track plays in its native style.

---

## Features

### 1. **Genre Compatibility Analysis**
- Multi-dimensional compatibility scoring (rhythmic, harmonic, timbral, cultural)
- Automatic detection of potential issues
- Suggestions for optimal blending strategies
- Known fusion precedent database (jazz-funk, latin-electronic, etc.)

### 2. **Harmonic Unification**
- Genre-specific chord substitutions
- Voice-aware chord voicing
- Reharmonization with style preservation
- Slash chord and extension handling

### 3. **Rhythmic Synchronization**
- Multiple sync strategies (strict grid, accompaniment reference, loose pocket)
- Genre-weighted timing characteristics
- Swing and laid-back timing application
- Automatic reference track selection

### 4. **Voice Leading Management**
- Parallel fifth/octave detection
- Stepwise motion optimization
- Priority-based voice leading rules
- Genre-appropriate motion handling

### 5. **Register Allocation**
- Automatic register analysis
- Intelligent frequency distribution
- Role-based register suggestions
- Overlap prevention

---

## Research Foundation

This system is based on cutting-edge research in multi-genre music arrangement:

### Harmonic Compatibility
- **Jazz Fusion Techniques**: Quartal/quintal harmony from modal jazz combined with rock power chords (Stack Exchange, 2024)
- **Harmonic Innovation**: Blending musical genres for new global sounds (Royal School of Music, 2024)

### Rhythmic Synchronization
- **Scale-Free Cross-Correlations**: Ensemble timing shows historical dependencies (PNAS, 2014)
- **Role-Optimized Coupling**: West African drum ensemble synchronization (PMC, 2021)
- **Genre-Specific Timing**: Malian drummers achieve near-perfect synchrony at 600 events/minute

### Voice Leading
- **Common Practice Rules**: Classical voice leading adapted for multi-genre contexts
- **Genre-Specific Approaches**: Jazz vs. classical vs. electronic voice leading conventions

### Orchestration
- **Universal Techniques**: Harmonic voicing, layering, dynamics remain universal across genres
- **Register Distribution**: Frequency balance principles from orchestration theory

---

## Quick Start

### Basic Example

```python
from midi_generator.core.multi_genre_arranger import (
    MultiGenreArranger,
    HarmonicContext,
    TrackSpec,
    TrackRole
)

# Define harmonic context (shared by all tracks)
harmonic_context = HarmonicContext(
    chord_progression=['Cmaj7', 'Dm7', 'G7', 'Cmaj7'],
    key='C',
    time_signature=(4, 4),
    tempo_bpm=120,
    length_measures=16
)

# Define tracks with different genres
tracks = [
    TrackSpec(1, 'funk', TrackRole.BASS, 33),       # Funk bass
    TrackSpec(2, 'jazz', TrackRole.HARMONY, 0),     # Jazz piano
    TrackSpec(3, 'hiphop', TrackRole.PERCUSSION, 128),  # Hip-hop drums
    TrackSpec(4, 'electronic', TrackRole.MELODY, 81)    # EDM synth lead
]

# Create arranger and generate
arranger = MultiGenreArranger()
result = arranger.arrange(harmonic_context, tracks)

# Export to MIDI
arranger.export_to_midi(result, 'fusion_arrangement.mid')
```

### Even Simpler

```python
from midi_generator.core.multi_genre_arranger import create_simple_arrangement

# Quick arrangement with preset roles
result = create_simple_arrangement(
    genres=['jazz', 'funk', 'hiphop'],
    key='Dm',
    tempo=95,
    measures=16
)
```

---

## Core Concepts

### 1. TrackSpec - Track Specification

Each track is defined by a `TrackSpec` that includes:

```python
TrackSpec(
    track_number=1,          # Track index
    genre='jazz',            # Genre from GENRE_PROFILES
    role=TrackRole.BASS,     # Musical role
    instrument=33,           # MIDI program number

    # Optional parameters
    register=RegisterRange.BASS,
    velocity_range=(60, 100),
    sync_strategy=SyncStrategy.ACCOMPANIMENT_REFERENCE,
    voice_leading_priority=VoiceLeadingPriority.MODERATE,

    # Custom overrides
    custom_swing_factor=0.67,
    custom_syncopation=0.8,

    # Timing
    timing_offset_ms=0.0,    # Rush (-) or drag (+)
    humanize_amount=0.05     # Random variation
)
```

### 2. HarmonicContext - Shared Harmony

All tracks share the same underlying harmonic structure:

```python
HarmonicContext(
    chord_progression=['Cmaj7', 'Am7', 'Dm7', 'G7'],
    key='C',
    time_signature=(4, 4),
    tempo_bpm=120,
    length_measures=16,

    # Advanced
    modal_center='Dorian',
    harmonic_rhythm_pattern=[2, 2, 1, 1],  # Measures per chord
    allow_reharmonization=True  # Genre-specific substitutions
)
```

### 3. TrackRole - Musical Function

Available roles:
- `BASS`: Bass line foundation
- `HARMONY`: Chordal accompaniment
- `MELODY`: Lead melodic line
- `PERCUSSION`: Drums and percussion
- `COUNTER_MELODY`: Secondary melodic line
- `PAD`: Sustained harmonic texture
- `RHYTHMIC_ACCENT`: Rhythmic hits (brass stabs, etc.)
- `ORNAMENT`: Decorative elements

### 4. SyncStrategy - Timing Approach

- `STRICT_GRID`: All parts align to strict tempo grid (EDM, pop)
- `ACCOMPANIMENT_REFERENCE`: Bass/drums lead, others follow (jazz, funk)
- `GENRE_WEIGHTED_TIMING`: Each genre's natural timing variability
- `LOOSE_POCKET`: Intentional timing variations (jazz, hip-hop)
- `POLYRHYTHMIC`: Independent rhythmic layers (African, progressive)
- `ADAPTIVE`: Dynamically adjust based on section

### 5. VoiceLeadingPriority

- `STRICT`: Classical-style (no parallel 5ths/8ves)
- `MODERATE`: Prefer smooth motion, allow some parallels
- `LOOSE`: Genre-appropriate, don't enforce classical rules
- `GENRE_SPECIFIC`: Apply genre's native voice leading

---

## API Reference

### MultiGenreArranger Class

Main orchestration class for multi-genre arrangements.

#### Methods

##### `analyze_arrangement_compatibility(tracks: List[TrackSpec]) -> Dict`

Analyzes compatibility before generation.

**Returns:**
```python
{
    'pairwise_scores': {
        ('jazz', 'funk'): {
            'overall': 0.75,
            'rhythmic': 0.80,
            'harmonic': 0.70,
            'timbral': 0.65,
            'cultural': 0.85
        },
        ...
    },
    'overall_compatibility': 0.72,
    'potential_issues': [
        "Low rhythmic compatibility between electronic and jazz"
    ],
    'recommendations': [
        "Use STRICT_GRID sync for electronic and jazz tracks",
        "jazz and funk blend well - emphasize this combination"
    ]
}
```

##### `arrange(harmonic_context, tracks, auto_optimize=True) -> Dict`

Generate complete arrangement.

**Parameters:**
- `harmonic_context`: HarmonicContext object
- `tracks`: List of TrackSpec objects
- `auto_optimize`: Enable voice leading and register optimization

**Returns:**
```python
{
    'tracks': [GeneratedTrack, ...],
    'harmonic_context': HarmonicContext,
    'compatibility_analysis': Dict,
    'metadata': {
        'reference_track': 3,
        'register_usage': {...},
        'track_count': 4
    }
}
```

##### `export_to_midi(arrangement: Dict, filename: str)`

Export arrangement to MIDI file.

### GenreCompatibilityAnalyzer Class

Analyzes genre compatibility.

#### Methods

##### `calculate_compatibility(genre_a: str, genre_b: str) -> Dict`

Calculate pairwise compatibility.

**Example:**
```python
analyzer = GenreCompatibilityAnalyzer()
scores = analyzer.calculate_compatibility('jazz', 'funk')

print(f"Overall: {scores['overall']:.2f}")
print(f"Rhythmic: {scores['rhythmic']:.2f}")
print(f"Harmonic: {scores['harmonic']:.2f}")
```

### HarmonicUnifier Class

Manages harmonic consistency across genres.

#### Methods

##### `parse_chord_symbol(chord: str) -> Dict`

Parse chord into components.

**Example:**
```python
parsed = HarmonicUnifier.parse_chord_symbol('Cmaj7/E')
# Returns: {'root': 'C', 'quality': 'maj7', 'bass_note': 'E', 'extensions': []}
```

##### `get_chord_tones(chord: str, octave: int) -> List[int]`

Get MIDI notes for chord.

##### `apply_genre_substitution(chord: str, genre: str) -> str`

Apply genre-appropriate substitution.

**Example:**
```python
# Jazz adds extensions
HarmonicUnifier.apply_genre_substitution('Cmaj7', 'jazz')
# Returns: 'Cmaj9'

# Blues converts to dominant
HarmonicUnifier.apply_genre_substitution('C', 'blues')
# Returns: 'C7'
```

### RhythmicSynchronizer Class

Manages timing across genres.

#### Methods

##### `apply_genre_timing(notes, genre, sync_strategy, is_reference_track=False)`

Apply genre-specific timing.

##### `quantize_to_grid(notes, grid_division=0.25)`

Quantize to rhythmic grid.

##### `determine_reference_track(tracks: List[TrackSpec]) -> int`

Select timing reference (usually percussion or bass).

### VoiceLeadingManager Class

Voice leading optimization.

#### Methods

##### `check_parallel_motion(voice1, voice2, interval_type='fifth') -> List[int]`

Detect parallel fifths/octaves.

##### `smooth_voice_leading(notes, chord_progression, priority)`

Optimize for smooth motion.

### RegisterAllocator Class

Frequency register management.

#### Methods

##### `analyze_register_usage(tracks: List[GeneratedTrack]) -> Dict`

Analyze frequency distribution.

##### `suggest_register(existing_tracks, role: TrackRole) -> RegisterRange`

Suggest optimal register for new track.

---

## Examples

### Example 1: Jazz-Funk-HipHop Fusion

```python
# Create classic jazz-hop fusion
harmonic_context = HarmonicContext(
    chord_progression=['Cmaj7', 'Am7', 'Dm7', 'G7'],
    key='C',
    time_signature=(4, 4),
    tempo_bpm=95,
    length_measures=16
)

tracks = [
    TrackSpec(1, 'funk', TrackRole.BASS, 33,
              custom_syncopation=0.9),  # Extra funky
    TrackSpec(2, 'jazz', TrackRole.HARMONY, 0,
              voice_leading_priority=VoiceLeadingPriority.STRICT),
    TrackSpec(3, 'hiphop', TrackRole.PERCUSSION, 128,
              timing_offset_ms=15.0)  # Laid-back feel
]

arranger = MultiGenreArranger()

# Check compatibility first
compat = arranger.analyze_arrangement_compatibility(tracks)
print(f"Compatibility: {compat['overall_compatibility']:.2f}")

# Generate
result = arranger.arrange(harmonic_context, tracks)
arranger.export_to_midi(result, 'jazz_hop.mid')
```

### Example 2: Electronic-Latin Fusion

```python
# Latin house fusion
harmonic_context = HarmonicContext(
    chord_progression=['Am', 'F', 'C', 'G'],
    key='Am',
    time_signature=(4, 4),
    tempo_bpm=124,
    length_measures=32
)

tracks = [
    TrackSpec(1, 'latin', TrackRole.PERCUSSION, 128,
              sync_strategy=SyncStrategy.STRICT_GRID),
    TrackSpec(2, 'electronic', TrackRole.BASS, 38,
              register=RegisterRange.SUB_BASS),
    TrackSpec(3, 'electronic', TrackRole.PAD, 88),
    TrackSpec(4, 'latin', TrackRole.MELODY, 73)  # Flute
]

result = MultiGenreArranger().arrange(harmonic_context, tracks)
```

### Example 3: Progressive Multi-Genre

```python
# Different genres in different sections
from midi_generator.generators.style_fusion import GenreBlender

# Measures 1-8: Jazz
# Measures 9-16: Transition
# Measures 17-24: Electronic

harmonic_context = HarmonicContext(
    chord_progression=['Cmaj7', 'Fmaj7', 'Dm7', 'G7'],
    key='C',
    time_signature=(7, 8),  # Odd meter
    tempo_bpm=140,
    length_measures=24
)

# Section 1: Jazz
tracks_section1 = [
    TrackSpec(1, 'jazz', TrackRole.BASS, 33),
    TrackSpec(2, 'jazz', TrackRole.HARMONY, 0)
]

# Section 3: Electronic
tracks_section3 = [
    TrackSpec(1, 'electronic', TrackRole.BASS, 38),
    TrackSpec(2, 'electronic', TrackRole.PAD, 88)
]

# Generate each section and blend
```

### Example 4: Orchestral + Jazz Fusion

```python
# Big band meets orchestra
harmonic_context = HarmonicContext(
    chord_progression=['Cmaj7', 'A7#9', 'Dm7', 'G7alt'],
    key='C',
    time_signature=(4, 4),
    tempo_bpm=160,
    length_measures=32
)

tracks = [
    # Rhythm section - jazz
    TrackSpec(1, 'jazz', TrackRole.BASS, 32),
    TrackSpec(2, 'jazz', TrackRole.PERCUSSION, 128),
    TrackSpec(3, 'jazz', TrackRole.HARMONY, 0),

    # Brass - jazz
    TrackSpec(4, 'jazz', TrackRole.RHYTHMIC_ACCENT, 56),  # Trumpet
    TrackSpec(5, 'jazz', TrackRole.RHYTHMIC_ACCENT, 57),  # Trombone

    # Strings - classical
    TrackSpec(6, 'classical', TrackRole.PAD, 48,
              voice_leading_priority=VoiceLeadingPriority.STRICT),
    TrackSpec(7, 'classical', TrackRole.COUNTER_MELODY, 40)
]

result = MultiGenreArranger().arrange(harmonic_context, tracks)
```

---

## Integration

### With Existing Generators

The multi-genre arranger integrates with existing library generators:

```python
# Use existing bass engine for funk track
from advanced_modules.bass_engine import BassEngine

# Use existing drum patterns for hip-hop
from midi_generator.algorithms.drum_patterns import DrumPatternGenerator

# In actual implementation, MultiGenreArranger delegates to these
```

### With Style Fusion

Combine with style_fusion.py for additional blending:

```python
from midi_generator.generators.style_fusion import GenreBlender, GENRE_PROFILES

# Get genre features
jazz_features = GENRE_PROFILES['jazz']
funk_features = GENRE_PROFILES['funk']

# Blend for intermediate style
blended = GenreBlender.blend_features(jazz_features, funk_features, 0.6)

# Use in track
tracks = [
    TrackSpec(1, 'jazz', TrackRole.BASS, 33,
              custom_swing_factor=blended.swing_factor)
]
```

### With Component System (Agent 2)

When Agent 2's component system is available:

```python
from midi_generator.core.component_system import (
    ComponentFactory,
    ComponentType,
    CompositionBuilder
)

# Multi-genre arranger becomes a high-level wrapper
# around component system
```

---

## Best Practices

### 1. Genre Selection

**Do:**
- Choose genres with overlapping tempo ranges
- Mix genres with compatible harmonic vocabularies
- Use compatibility analysis before generating

**Don't:**
- Mix extremely disparate tempos without strategy
- Ignore compatibility warnings
- Overload arrangement with too many genres

### 2. Track Roles

**Do:**
- Assign complementary roles across tracks
- Have clear bass and percussion reference
- Balance frequency ranges

**Don't:**
- Put multiple tracks in same role and register
- Neglect low-end foundation
- Create muddy mid-range overlaps

### 3. Sync Strategy

**Do:**
- Use STRICT_GRID for electronic + acoustic mixes
- Use ACCOMPANIMENT_REFERENCE for organic feel
- Let percussion/bass be timing reference

**Don't:**
- Use LOOSE_POCKET with electronic genres
- Change sync strategy mid-arrangement
- Fight against natural genre timing

### 4. Voice Leading

**Do:**
- Use STRICT for classical elements
- Use MODERATE as default
- Use LOOSE for jazz, blues, rock

**Don't:**
- Apply classical rules to all genres
- Ignore register when checking parallels
- Over-optimize natural genre motion

### 5. Harmonic Approach

**Do:**
- Allow genre-specific reharmonization
- Use shared progressions
- Consider modal centers for fusion

**Don't:**
- Force all genres to identical voicings
- Ignore genre chord preferences
- Over-complicate harmony

---

## Advanced Usage

### Custom Genre Profiles

Add your own genres to GENRE_PROFILES:

```python
from midi_generator.generators.style_fusion import GENRE_PROFILES, GenreFeatures

GENRE_PROFILES['my_fusion'] = GenreFeatures(
    name='My Fusion Style',
    tempo_range=(100, 130),
    swing_factor=0.6,
    syncopation=0.7,
    rhythmic_complexity=0.75,
    chord_types=['maj7', 'min7', 'sus2', '9'],
    harmonic_rhythm=2.0,
    use_extensions=True,
    chromaticism=0.5,
    interval_preference='balanced',
    ornamentation=0.5,
    melodic_range=(48, 84),
    instruments=[81, 33, 0, 64],
    texture='polyphonic',
    register_preference='mid',
    cultural_origin='Hybrid',
    rhythmic_basis='hybrid',
    groove_type='swing-straight'
)
```

### Custom Compatibility Rules

Extend compatibility analyzer:

```python
GenreCompatibilityAnalyzer.FUSION_PRECEDENTS[('my_fusion', 'jazz')] = 0.85
```

### Manual Register Control

Override automatic allocation:

```python
from midi_generator.core.multi_genre_arranger import RegisterRange

tracks = [
    TrackSpec(1, 'funk', TrackRole.BASS, 33,
              register=RegisterRange.BASS),  # Force bass register
    TrackSpec(2, 'jazz', TrackRole.HARMONY, 0,
              register=RegisterRange.MID),   # Force mid register
]
```

### Custom Timing Offsets

Create specific feels:

```python
tracks = [
    TrackSpec(1, 'funk', TrackRole.BASS, 33,
              timing_offset_ms=-5.0,    # Rush slightly (ahead of beat)
              humanize_amount=0.02),     # Tight

    TrackSpec(2, 'jazz', TrackRole.HARMONY, 0,
              timing_offset_ms=10.0,     # Laid-back (behind beat)
              humanize_amount=0.08),     # Loose
]
```

### Programmatic Arrangement Evolution

Gradually change genres over time:

```python
def create_morphing_arrangement(measures_per_section=8):
    sections = []

    # Section 1: Pure jazz
    section1 = create_arrangement_section(
        genres=['jazz', 'jazz', 'jazz'],
        measures=measures_per_section
    )

    # Section 2: Jazz-funk blend
    section2 = create_arrangement_section(
        genres=['funk', 'jazz', 'jazz'],
        measures=measures_per_section
    )

    # Section 3: Funk-electronic blend
    section3 = create_arrangement_section(
        genres=['funk', 'electronic', 'electronic'],
        measures=measures_per_section
    )

    # Concatenate sections
    return concatenate_sections([section1, section2, section3])
```

---

## Troubleshooting

### Common Issues

**Issue**: Tracks sound disconnected
- **Solution**: Ensure `allow_reharmonization=True` and check compatibility scores

**Issue**: Timing feels off
- **Solution**: Verify reference track selection, consider STRICT_GRID strategy

**Issue**: Muddy mix
- **Solution**: Check register allocation, ensure tracks occupy different frequency ranges

**Issue**: Voice leading errors
- **Solution**: Adjust `voice_leading_priority`, check chord voicings

**Issue**: Low compatibility scores
- **Solution**: Review genre selection, adjust sync strategy, consider intermediate blending

---

## Performance Considerations

- **Compatibility Analysis**: O(n²) for n genres, cache results
- **Generation**: O(n*m) for n tracks and m measures
- **Voice Leading**: O(n²) for checking parallels between tracks
- **Register Analysis**: O(n*m) for n tracks and m notes

For large arrangements (>10 tracks), consider:
- Disable auto_optimize for initial generation
- Generate sections separately
- Use simpler sync strategies

---

## Future Enhancements

Planned features for future versions:

1. **Machine Learning Integration**
   - Train on successful fusion examples
   - Automatic parameter optimization
   - Style transfer between arrangements

2. **Real-Time Adaptation**
   - Dynamic sync adjustment during playback
   - Adaptive voice leading
   - Context-aware humanization

3. **Extended Genre Database**
   - More world music genres
   - Micro-genres and sub-styles
   - User-contributed profiles

4. **Advanced Orchestration**
   - Automatic doubling suggestions
   - Harmonic series optimization
   - Spectral analysis integration

---

## References

### Research Papers

1. **Synchronization in human musical rhythms and mutually interacting complex systems** (PNAS, 2014)
   - Scale-free cross-correlations in ensemble timing

2. **Extreme precision in rhythmic interaction is enabled by role-optimized sensorimotor coupling** (PMC, 2021)
   - West African drum ensemble synchronization research

3. **Sounds familiar(?): Expertise with specific musical genres modulates timing perception** (PMC, 2022)
   - Genre-specific timing expertise

4. **Harmonic Innovation: Blending Musical Genres for a New Global Sound** (Royal School of Music, 2024)
   - Modern approaches to genre fusion

### Online Resources

- Stack Exchange Music: Jazz-rock fusion harmonic devices
- StudySmarter: Arranging music techniques across genres
- Soundtrap Blog: Genre blending in modern music production

---

## Contributing

To contribute to the Multi-Genre Arranger:

1. Add new genre profiles to `GENRE_PROFILES`
2. Extend compatibility precedents in `GenreCompatibilityAnalyzer`
3. Implement genre-specific generators
4. Add test cases to `test_multi_genre_arranger.py`
5. Update this documentation

---

## License

Part of the HarmonyModule library.

---

## Contact

**Agent 9**: Track-Level Genre Control System
**Date**: 2025
**Version**: 1.0.0

For issues, questions, or suggestions, please refer to the main HarmonyModule documentation.

---

**Happy Multi-Genre Arranging!** 🎵🎹🎸🥁
