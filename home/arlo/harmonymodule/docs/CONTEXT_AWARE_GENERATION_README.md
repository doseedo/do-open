# Context-Aware Generation System

**Agent 3: Context-Aware Generation**

> Analyze existing MIDI arrangements and generate new tracks that seamlessly fit the existing style, harmony, rhythm, and overall musical context.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Core Classes](#core-classes)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Advanced Topics](#advanced-topics)
8. [Examples](#examples)
9. [Integration](#integration)
10. [Research Foundation](#research-foundation)

---

## Overview

The Context-Aware Generation system is a powerful tool for adding tracks to existing MIDI arrangements or regenerating sections with different harmonies or styles. Unlike traditional MIDI generation that starts from scratch, this system **understands** existing musical material and generates new content that fits seamlessly.

### What Makes It Context-Aware?

- **Harmonic Analysis**: Extracts chord progressions from existing tracks
- **Rhythmic Profiling**: Analyzes note density, groove patterns, and timing
- **Voice Leading**: Ensures smooth motion between voices, avoiding parallel fifths/octaves
- **Texture Matching**: Generates material that complements existing arrangement density
- **Style Detection**: Identifies genre characteristics and maintains consistency
- **Boundary Smoothing**: Creates seamless transitions when regenerating sections

### Use Cases

1. **Add Missing Parts**: Generate bass, drums, or harmony for incomplete arrangements
2. **Reharmonization**: Change chord progressions while preserving melody
3. **Genre Fusion**: Add tracks in different genres (e.g., EDM synths to jazz arrangement)
4. **Inpainting**: Regenerate specific sections (like Photoshop content-aware fill)
5. **Smart Orchestration**: Get AI-powered suggestions for what instruments to add

---

## Key Features

### 1. Comprehensive MIDI Analysis

```python
gen = ContextAwareGenerator('existing_song.mid')
analysis = gen.analyze()

print(f"Tempo: {analysis.tempo} BPM")
print(f"Key: {analysis.key}")
print(f"Time Signature: {analysis.time_signature}")
print(f"Chords: {analysis.chord_progression}")
print(f"Groove: {analysis.groove_type}")
print(f"Texture: {analysis.texture}")
```

**Extracted Features:**
- Tempo and time signature
- Key signature (Krumhansl-Schmuckler algorithm)
- Chord progression with harmonic rhythm
- Note density per measure
- Groove type (swing, straight, shuffle, half-time)
- Texture (monophonic, homophonic, polyphonic)
- Register distribution (low/mid/high)
- Instrument classification
- Track role detection (melody, harmony, bass, drums)

### 2. Intelligent Track Generation

Generate new tracks that **fit** the existing arrangement:

```python
# Add funk bass to jazz piano arrangement
bass_notes = gen.add_track(
    instrument=33,      # Fingered bass
    genre='funk',       # Funk style bass
    track_type='bass'   # Bass line
)

gen.export_with_new_track(bass_notes, 'with_funk_bass.mid')
```

**Track Types:**
- **Bass**: Walking bass, funk bass, reggae bass, etc.
- **Harmony**: Chord voicings, pads, comping
- **Melody**: Lead lines, countermelodies
- **Percussion**: Drums, rhythmic patterns

### 3. Section-Level Control

Add tracks to specific sections only:

```python
# Add strings to bridge section (measures 8-16)
string_notes = gen.add_section(
    start_measure=8,
    end_measure=16,
    instrument=48,      # String ensemble
    track_type='harmony'
)
```

### 4. Inpainting (Regenerate Sections)

Regenerate measures with new chords or style:

```python
inpainter = TrackInpainter('song.mid')

# Reharmonize measures 5-8
new_section = inpainter.inpaint_measures(
    track=0,
    start=5,
    end=8,
    new_chords=['Dm7', 'G7alt', 'Cmaj9', 'A7#9'],
    smooth_boundaries=True  # Seamless transitions
)

inpainter.export('reharmonized.mid')
```

**Boundary Smoothing:**
- Analyzes notes before/after inpaint region
- Applies voice leading rules for smooth entry
- Matches rhythmic density at boundaries
- Preserves melodic contour

### 5. Genre Change Within Song

```python
# Change chorus to EDM style while keeping jazz verse
edm_section = inpainter.inpaint_with_genre_change(
    track=0,
    start=8,      # Chorus starts at measure 8
    end=16,
    new_genre='edm',
    smooth_boundaries=True
)
```

### 6. Smart Orchestration Suggestions

```python
orchestrator = SmartOrchestrator('piano_drums.mid')
suggestions = orchestrator.suggest_additions()

# Output:
# 1. No bass line - would provide harmonic foundation (Priority: 0.9)
# 2. Mid-range sparse - strings would fill space (Priority: 0.75)
# 3. Limited harmony - piano would add richness (Priority: 0.7)

# Auto-add top suggestion
orchestrator.add_suggested_track(suggestions[0])
orchestrator.export('orchestrated.mid')
```

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                  Context-Aware Generator                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Analysis   │  │  Generation  │  │   Export     │    │
│  │              │  │              │  │              │    │
│  │ • Chords     │  │ • Bass       │  │ • MIDI       │    │
│  │ • Rhythm     │  │ • Harmony    │  │ • Multi-     │    │
│  │ • Texture    │  │ • Melody     │  │   track      │    │
│  │ • Style      │  │ • Percussion │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
│  ┌──────────────────────────────────────────────────┐     │
│  │            Integration Modules                    │     │
│  ├──────────────────────────────────────────────────┤     │
│  │ • MidiAnalyzer (chord/key detection)            │     │
│  │ • BassEngine (existing bass patterns)           │     │
│  │ • GenreFeatures (style definitions)             │     │
│  │ • Counterpoint (voice leading rules)            │     │
│  └──────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

        ┌─────────────────┐         ┌─────────────────┐
        │  TrackInpainter │         │ SmartOrchestrator│
        │                 │         │                  │
        │ • Regenerate    │         │ • Suggestions    │
        │   sections      │         │ • Auto-add       │
        │ • Reharmonize   │         │ • Balance        │
        │ • Genre change  │         │   analysis       │
        └─────────────────┘         └─────────────────┘
```

### Data Flow

```
Input MIDI → Analysis → Generation → Boundary Smoothing → Output MIDI
                ↓           ↓              ↓
           Chords      Track Type    Voice Leading
           Tempo       Genre         Density Matching
           Rhythm      Constraints   Register Balance
```

---

## Core Classes

### 1. ContextAwareGenerator

Main class for analyzing and generating tracks.

**Constructor:**
```python
gen = ContextAwareGenerator(existing_midi: str)
```

**Key Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `analyze()` | Comprehensive MIDI analysis | `ArrangementAnalysis` |
| `add_track(...)` | Generate new track | `List[Note]` |
| `add_section(...)` | Add to specific section | `List[Note]` |
| `regenerate_section(...)` | Inpaint/regenerate | `List[Note]` |
| `suggest_additions()` | AI suggestions | `List[Dict]` |
| `export_with_new_track(...)` | Save MIDI | `None` |

### 2. ArrangementAnalysis

Complete analysis of existing arrangement.

**Attributes:**
```python
@dataclass
class ArrangementAnalysis:
    tempo: int
    time_signature: Tuple[int, int]
    length_measures: int
    key: KeySignature
    chord_progression: List[str]
    chords_per_measure: List[List[ChordEvent]]
    harmonic_rhythm: float
    density_per_measure: List[float]
    rhythmic_profile: Dict[str, Any]
    groove_type: str
    texture: str
    register_distribution: Dict[str, float]
    instruments: List[int]
    tracks: List[List[NoteEvent]]
    track_roles: Dict[int, str]
    detected_style: Optional[GenreFeatures]
```

### 3. TrackInpainter

Specialized for regenerating sections.

**Constructor:**
```python
inpainter = TrackInpainter(midi_file: str)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `inpaint_measures(track, start, end, new_chords)` | Regenerate with new chords |
| `inpaint_with_genre_change(track, start, end, new_genre)` | Change genre in section |
| `export(output_file)` | Save modified MIDI |

### 4. SmartOrchestrator

AI-powered orchestration suggestions.

**Constructor:**
```python
orchestrator = SmartOrchestrator(midi_file: str)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `suggest_additions()` | Get suggestions for tracks to add |
| `add_suggested_track(suggestion)` | Add suggested track |
| `analyze_orchestral_balance()` | Analyze arrangement balance |
| `export(output_file)` | Save with added tracks |

### 5. BoundaryContext

Context at section boundaries for smooth transitions.

```python
@dataclass
class BoundaryContext:
    measure: int
    last_notes: List[int]
    last_rhythm: List[float]
    last_velocities: List[int]
    harmonic_context: Optional[str]
    voice_leading_tendency: str  # 'ascending', 'descending', 'static'
    register: str  # 'low', 'mid', 'high'
```

### 6. GenerationConstraints

Control generation behavior.

```python
@dataclass
class GenerationConstraints:
    follow_harmony: bool = True
    match_density: bool = True
    avoid_voice_leading_errors: bool = True
    preserve_texture: bool = True
    max_voice_leading_distance: int = 7  # Semitones
    preferred_motion: str = 'contrary'
```

---

## Usage Guide

### Quick Start

```python
from midi_generator.generators.context_aware_generator import (
    ContextAwareGenerator,
    TrackInpainter,
    SmartOrchestrator
)

# 1. Basic track addition
gen = ContextAwareGenerator('my_song.mid')
bass_notes = gen.add_track(instrument=33, track_type='bass')
gen.export_with_new_track(bass_notes, 'with_bass.mid')

# 2. Inpainting
inpainter = TrackInpainter('my_song.mid')
new_section = inpainter.inpaint_measures(
    track=0, start=4, end=8,
    new_chords=['Em7', 'A7', 'Dmaj7', 'Gmaj7']
)
inpainter.export('reharmonized.mid')

# 3. Smart suggestions
orchestrator = SmartOrchestrator('my_song.mid')
suggestions = orchestrator.suggest_additions()
for s in suggestions:
    print(f"{s['reason']} - Priority: {s['priority']}")
```

### Step-by-Step: Adding Bass Line

```python
# Step 1: Initialize
gen = ContextAwareGenerator('piano_only.mid')

# Step 2: Analyze
analysis = gen.analyze()
print(f"Detected {len(analysis.chord_progression)} chords")
print(f"Tempo: {analysis.tempo} BPM")
print(f"Style: {analysis.detected_style.name if analysis.detected_style else 'Unknown'}")

# Step 3: Generate bass
bass_notes = gen.add_track(
    instrument=33,      # Fingered bass
    genre='jazz',       # Jazz walking bass
    track_type='bass',
    constraints=GenerationConstraints(
        follow_harmony=True,
        avoid_voice_leading_errors=True,
        max_voice_leading_distance=5
    )
)

print(f"Generated {len(bass_notes)} bass notes")

# Step 4: Export
gen.export_with_new_track(
    bass_notes,
    'with_bass.mid',
    instrument=33,
    track_name='Jazz Bass'
)

print("Done! Check 'with_bass.mid'")
```

### Step-by-Step: Reharmonization

```python
# Original: C - F - G - C
# Target: Cmaj7 - Fmaj7 - Em7 - Am7 - Dm7 - G7 - Cmaj7

from midi_generator.generators.context_aware_generator import TrackInpainter

inpainter = TrackInpainter('simple_song.mid')

# Reharmonize entire song
new_progression = [
    'Cmaj7', 'Fmaj7#11', 'Em7', 'Am7',
    'Dm7', 'G7alt', 'Cmaj7', 'Abmaj7'
]

new_track = inpainter.inpaint_measures(
    track=0,          # Piano track
    start=0,          # From beginning
    end=8,            # 8 measures
    new_chords=new_progression,
    smooth_boundaries=True
)

# Optionally adjust other tracks to match
for track_idx in range(1, len(inpainter.generator.analysis.tracks)):
    new_harmony = inpainter.inpaint_measures(
        track=track_idx,
        start=0,
        end=8,
        new_chords=new_progression
    )

inpainter.export('jazz_reharmonized.mid')
```

### Step-by-Step: Genre Fusion

```python
# Jazz verse + EDM chorus + Jazz verse

from midi_generator.generators.context_aware_generator import TrackInpainter

inpainter = TrackInpainter('jazz_song.mid')

# Keep verse 1 (measures 0-8) as jazz
# Change chorus (measures 8-16) to EDM
edm_chorus = inpainter.inpaint_with_genre_change(
    track=0,
    start=8,
    end=16,
    new_genre='edm',
    smooth_boundaries=True  # Smooth transition from jazz → EDM
)

# Verse 2 (measures 16-24) stays jazz

inpainter.export('jazz_edm_fusion.mid')
```

---

## API Reference

### ContextAwareGenerator

#### `__init__(existing_midi: str)`

Initialize generator with existing MIDI file.

**Parameters:**
- `existing_midi` (str): Path to MIDI file

**Example:**
```python
gen = ContextAwareGenerator('input.mid')
```

---

#### `analyze() -> ArrangementAnalysis`

Perform comprehensive analysis of MIDI arrangement.

**Returns:**
- `ArrangementAnalysis`: Complete analysis object

**Analysis Includes:**
- Basic info: tempo, time signature, key, length
- Harmonic: chord progression, harmonic rhythm
- Rhythmic: density per measure, groove type
- Textural: texture type, register distribution
- Instrumental: track roles, instruments

**Example:**
```python
analysis = gen.analyze()

print(f"Tempo: {analysis.tempo} BPM")
print(f"Chords: {', '.join(analysis.chord_progression[:4])}")
print(f"Density: {sum(analysis.density_per_measure) / len(analysis.density_per_measure):.1f} notes/measure")
```

---

#### `add_track(instrument, genre=None, track_type='auto', constraints=None) -> List[Tuple[int, float, float, int]]`

Generate new track fitting existing arrangement.

**Parameters:**
- `instrument` (int): MIDI program number (0-127) or 128 for drums
- `genre` (str, optional): Genre style ('jazz', 'funk', 'edm', etc.). If None, uses detected style
- `track_type` (str): Track type - 'bass', 'harmony', 'melody', 'percussion', 'auto'
- `constraints` (GenerationConstraints, optional): Generation constraints

**Returns:**
- List of tuples: `(pitch, start_time, duration, velocity)`

**Track Type Inference (when `track_type='auto'`):**
- Instruments 32-39 → bass
- Instruments 112+ or 128 → percussion
- Lead instruments (brass, winds) → melody
- Others → harmony

**Example:**
```python
# Funk bass
bass = gen.add_track(instrument=33, genre='funk', track_type='bass')

# Auto-detect type from instrument
strings = gen.add_track(instrument=48, track_type='auto')  # → harmony

# With constraints
melody = gen.add_track(
    instrument=73,  # Flute
    track_type='melody',
    constraints=GenerationConstraints(
        max_voice_leading_distance=3,
        preferred_motion='contrary'
    )
)
```

---

#### `add_section(start_measure, end_measure, instrument, track_type='auto', custom_chords=None) -> List[Tuple]`

Add instrument to specific section only.

**Parameters:**
- `start_measure` (int): Section start (0-indexed)
- `end_measure` (int): Section end (exclusive)
- `instrument` (int): MIDI program number
- `track_type` (str): Track type
- `custom_chords` (List[str], optional): Override detected chords

**Returns:**
- List of notes for that section only

**Example:**
```python
# Add brass to bridge section
brass_section = gen.add_section(
    start_measure=16,
    end_measure=24,
    instrument=56,  # Trumpet
    track_type='melody',
    custom_chords=['Bb7', 'Eb7', 'Ab7', 'Db7', 'Gb7', 'B7', 'E7', 'A7']
)
```

---

#### `regenerate_section(track_number, start_measure, end_measure, new_chords=None, new_genre=None, blend_boundaries=True) -> List[Tuple]`

Regenerate section of existing track (INPAINTING).

**Parameters:**
- `track_number` (int): Track index to regenerate
- `start_measure` (int): Start of section
- `end_measure` (int): End of section (exclusive)
- `new_chords` (List[str], optional): New chord progression
- `new_genre` (str, optional): New genre
- `blend_boundaries` (bool): Apply boundary smoothing

**Returns:**
- New notes for section

**Boundary Smoothing:**
When `blend_boundaries=True`:
1. Analyzes notes before/after section
2. Applies voice leading for smooth entry
3. Matches rhythmic density at boundaries
4. Prepares exit using anticipation

**Example:**
```python
# Reharmonize chorus
new_chorus = gen.regenerate_section(
    track_number=0,
    start_measure=8,
    end_measure=16,
    new_chords=[
        'Fmaj7', 'Bm7b5', 'E7alt', 'Am7',
        'Dm7', 'G7', 'Cmaj7', 'A7'
    ],
    blend_boundaries=True
)
```

---

#### `suggest_additions() -> List[Dict[str, Any]]`

Analyze arrangement and suggest tracks to add.

**Returns:**
- List of suggestion dictionaries, sorted by priority

**Suggestion Dictionary:**
```python
{
    'instrument': int,      # MIDI program number
    'track_type': str,      # 'bass', 'harmony', 'melody', 'percussion'
    'reason': str,          # Human-readable explanation
    'priority': float,      # 0.0-1.0, higher = more important
    'genre': str           # Suggested genre
}
```

**Suggestion Logic:**
- Missing bass → priority 0.9
- Sparse mid-range → suggest harmony
- No drums → suggest percussion
- Low harmonic density → suggest chordal instrument

**Example:**
```python
suggestions = gen.suggest_additions()

for i, s in enumerate(suggestions, 1):
    print(f"{i}. [{s['priority']:.2f}] {s['reason']}")
    print(f"   → Add {s['track_type']} ({s['instrument']})")
```

---

#### `export_with_new_track(new_notes, output_file, instrument=0, track_name='Generated Track')`

Export MIDI with new track added.

**Parameters:**
- `new_notes` (List[Tuple]): Generated notes
- `output_file` (str): Output MIDI filename
- `instrument` (int): MIDI program for new track
- `track_name` (str): Track name in MIDI

**Example:**
```python
bass = gen.add_track(instrument=33, track_type='bass')
gen.export_with_new_track(
    bass,
    'output.mid',
    instrument=33,
    track_name='Funk Bass Line'
)
```

---

### TrackInpainter

#### `inpaint_measures(track, start, end, new_chords=None, smooth_boundaries=True) -> List[Tuple]`

Regenerate measures with smooth boundaries.

**Parameters:**
- `track` (int): Track index
- `start` (int): Start measure
- `end` (int): End measure (exclusive)
- `new_chords` (List[str], optional): New chords (None = keep existing)
- `smooth_boundaries` (bool): Apply smoothing

**Returns:**
- New notes for section

**Example:**
```python
inpainter = TrackInpainter('song.mid')

# Reharmonize bridge
new_bridge = inpainter.inpaint_measures(
    track=0,
    start=16,
    end=24,
    new_chords=['Bbmaj7', 'Am7', 'Gm7', 'Fmaj7', 'Em7b5', 'A7', 'Dm7', 'G7']
)
```

---

#### `inpaint_with_genre_change(track, start, end, new_genre, smooth_boundaries=True) -> List[Tuple]`

Regenerate section in different genre.

**Parameters:**
- `track` (int): Track index
- `start`, `end` (int): Section boundaries
- `new_genre` (str): New genre ('jazz', 'edm', 'funk', etc.)
- `smooth_boundaries` (bool): Apply smoothing

**Returns:**
- New notes

**Example:**
```python
# EDM drop in jazz song
edm_drop = inpainter.inpaint_with_genre_change(
    track=0,
    start=32,  # Drop at measure 32
    end=48,
    new_genre='edm',
    smooth_boundaries=True
)
```

---

### SmartOrchestrator

#### `suggest_additions() -> List[Dict]`

Get suggestions for tracks to add.

**Returns:**
- List of suggestion dictionaries (sorted by priority)

---

#### `add_suggested_track(suggestion) -> List[Tuple]`

Add a track based on suggestion.

**Parameters:**
- `suggestion` (Dict): Suggestion from `suggest_additions()`

**Returns:**
- Generated notes

**Example:**
```python
orchestrator = SmartOrchestrator('sparse_arrangement.mid')
suggestions = orchestrator.suggest_additions()

# Add all high-priority suggestions
for suggestion in suggestions:
    if suggestion['priority'] > 0.7:
        orchestrator.add_suggested_track(suggestion)

orchestrator.export('full_arrangement.mid')
```

---

## Advanced Topics

### Voice Leading Rules

The generator applies classical voice leading rules:

1. **Avoid Parallel Fifths/Octaves**: Consecutive perfect fifths or octaves between voices
2. **Stepwise Motion**: Prefer stepwise motion (±1-2 semitones) over large leaps
3. **Contrary Motion**: When possible, move voices in opposite directions
4. **Preparation**: Approach dissonances by step or common tone
5. **Resolution**: Resolve tendency tones (7→8, 4→3)

**Implementation:**
```python
# Voice leading is checked in _avoid_voice_leading_errors()
# Uses counterpoint_engine.py from existing modules

constraints = GenerationConstraints(
    avoid_voice_leading_errors=True,
    max_voice_leading_distance=5,  # Max semitones between adjacent notes
    preferred_motion='contrary'     # Prefer contrary motion
)
```

### Boundary Smoothing Algorithm

When regenerating sections, boundary smoothing ensures seamless transitions:

```
Original:  [────A────][────B────][────C────]
Regenerate:         [──── B' ───]

Boundary smoothing:
1. Extract entry context from end of A
2. Extract exit context from start of C
3. Generate B' matching harmony
4. Smooth entry: adjust first notes of B' to approach from A
5. Smooth exit: adjust last notes of B' to lead into C
```

**Smoothing Techniques:**
- **Pitch**: Use notes close to boundary notes (stepwise motion)
- **Rhythm**: Match rhythmic density at boundaries
- **Dynamics**: Fade in/out velocity
- **Preparation**: Add anticipation notes before transitions

### Register Balance

The system analyzes register distribution:

```python
# Register ranges (MIDI note numbers)
LOW:  0-47   (C-1 to B2)
MID:  48-71  (C3 to B4)
HIGH: 72-127 (C5 to G9)

# Analysis
register_dist = analysis.register_distribution
# {'low': 0.2, 'mid': 0.5, 'high': 0.3}

# System suggests filling sparse registers
if register_dist['mid'] < 0.3:
    suggestions.append({
        'reason': 'Mid-range sparse - add harmonic padding',
        'instrument': 48,  # Strings
        'priority': 0.75
    })
```

### Harmonic Rhythm Matching

**Harmonic rhythm** = rate of chord change

```python
# Analysis
harmonic_rhythm = len(chords) / num_measures
# 8 chords / 4 measures = 2.0 chords/measure

# Generation matches this density:
# - Fast harmonic rhythm (3-4 chords/measure) → add more chord changes
# - Slow harmonic rhythm (0.5 chords/measure) → sustain chords longer
```

### Texture Preservation

**Texture Types:**
- **Monophonic**: Single melodic line (1 note at a time)
- **Homophonic**: Melody + accompaniment (2-3 simultaneous notes)
- **Polyphonic**: Multiple independent voices (4+ simultaneous notes)

The system preserves texture when adding tracks:

```python
if analysis.texture == 'monophonic':
    # Don't add dense chords - would change texture
    # Add single-note bass line or simple accompaniment
    pass
elif analysis.texture == 'polyphonic':
    # Can add complex voicings
    pass
```

---

## Examples

### Example 1: Add Funk Bass to Jazz Piano

```python
from midi_generator.generators.context_aware_generator import ContextAwareGenerator

gen = ContextAwareGenerator('jazz_piano.mid')
analysis = gen.analyze()

print(f"Original: {analysis.tempo} BPM, {len(analysis.tracks)} tracks")
print(f"Chords: {', '.join(analysis.chord_progression[:4])}...")

# Generate funk bass (syncopated, rhythmic)
funk_bass = gen.add_track(
    instrument=33,      # Fingered bass
    genre='funk',       # Funk feel
    track_type='bass'
)

gen.export_with_new_track(
    funk_bass,
    'jazz_piano_with_funk_bass.mid',
    instrument=33,
    track_name='Funk Bass'
)

print(f"Added {len(funk_bass)} bass notes")
```

**Result**: Jazz piano chords + funky, syncopated bass line

---

### Example 2: Reharmonize Bridge Section

```python
from midi_generator.generators.context_aware_generator import TrackInpainter

# Original bridge: C - F - G - C
# New bridge: Cmaj7 - Fmaj7#11 - Bm7b5 - E7alt - Am7 - Dm7 - G7alt - Cmaj7

inpainter = TrackInpainter('song.mid')

# Reharmonize measures 16-24 (bridge)
new_bridge = inpainter.inpaint_measures(
    track=0,  # Piano track
    start=16,
    end=24,
    new_chords=[
        'Cmaj7', 'Fmaj7#11', 'Bm7b5', 'E7alt',
        'Am7', 'Dm7', 'G7alt', 'Cmaj7'
    ],
    smooth_boundaries=True
)

inpainter.export('reharmonized_bridge.mid')
```

**Result**: Bridge has rich jazz harmony while verse/chorus unchanged

---

### Example 3: Genre Fusion (Jazz Verse + EDM Chorus)

```python
from midi_generator.generators.context_aware_generator import TrackInpainter

inpainter = TrackInpainter('jazz_ballad.mid')

# Verse 1 (0-8): Keep jazz
# Chorus (8-16): Change to EDM
# Verse 2 (16-24): Keep jazz

edm_chorus = inpainter.inpaint_with_genre_change(
    track=0,
    start=8,
    end=16,
    new_genre='edm',
    smooth_boundaries=True
)

# Add EDM drums to chorus only
gen = inpainter.generator
edm_drums = gen.add_section(
    start_measure=8,
    end_measure=16,
    instrument=128,  # Drums
    track_type='percussion'
)

gen.export_with_new_track(edm_drums, 'jazz_edm_fusion.mid')
```

**Result**: Smooth transition jazz → EDM → jazz

---

### Example 4: Smart Auto-Orchestration

```python
from midi_generator.generators.context_aware_generator import SmartOrchestrator

orchestrator = SmartOrchestrator('piano_only.mid')

# Get AI suggestions
suggestions = orchestrator.suggest_additions()

print("Orchestration Suggestions:")
print("=" * 60)
for i, s in enumerate(suggestions, 1):
    print(f"{i}. [{s['priority']:.2f}] {s['reason']}")
    print(f"   Instrument: {s['instrument']} ({s['track_type']})")
    print()

# Auto-add all high-priority suggestions
added = 0
for suggestion in suggestions:
    if suggestion['priority'] >= 0.75:
        orchestrator.add_suggested_track(suggestion)
        added += 1

print(f"Added {added} tracks")
orchestrator.export('full_orchestration.mid')
```

**Example Output:**
```
Orchestration Suggestions:
==========================================================
1. [0.90] No bass line - would provide harmonic foundation
   Instrument: 33 (bass)

2. [0.85] No rhythmic foundation - drums would provide groove
   Instrument: 128 (percussion)

3. [0.75] Mid-range sparse - strings would fill harmonic space
   Instrument: 48 (harmony)

Added 3 tracks
```

---

### Example 5: Complex Multi-Track Addition

```python
from midi_generator.generators.context_aware_generator import ContextAwareGenerator

gen = ContextAwareGenerator('melody_chords.mid')
gen.analyze()

# Add multiple tracks with different genres
tracks_to_add = [
    (33, 'funk', 'bass'),         # Funk bass
    (128, 'jazz', 'percussion'),  # Jazz drums
    (48, 'classical', 'harmony'), # String pad
    (56, 'jazz', 'melody')        # Trumpet countermelody
]

output = MidiFile(ticks_per_beat=gen.ticks_per_beat)

# Copy original tracks
for track in gen.midi.tracks:
    output.tracks.append(track.copy())

# Generate and add each new track
for instrument, genre, track_type in tracks_to_add:
    notes = gen.add_track(instrument, genre, track_type)

    # Convert to MIDI track (simplified)
    new_track = create_midi_track(notes, instrument)
    output.tracks.append(new_track)

output.save('full_arrangement.mid')
```

---

## Integration

### With Existing HarmonyModule Components

The Context-Aware Generator integrates seamlessly with existing modules:

#### 1. MidiAnalyzer (analysis/midi_analyzer.py)

```python
# Used for:
# - Chord extraction
# - Key detection
# - Tempo/time signature
# - Note extraction

from analysis.midi_analyzer import MidiAnalyzer

analyzer = MidiAnalyzer('song.mid')
chords = analyzer.extract_chords()  # Used internally
```

#### 2. BassEngine (advanced_modules/bass_engine.py)

```python
# Used for bass line generation
# Context-aware generator calls BassEngine with detected chords

from advanced_modules.bass_engine import BassEngine

bass_gen = BassEngine()
bass_line = bass_gen.generate_walking_bass(
    chord_progression=analysis.chord_progression,
    style='jazz'
)
```

#### 3. GenreFeatures (generators/style_fusion.py)

```python
# Used for style detection and genre-specific generation

from generators.style_fusion import GENRE_PROFILES, GenreFeatures

# Get genre characteristics
jazz_features = GENRE_PROFILES['jazz']
funk_features = GENRE_PROFILES['funk']
```

#### 4. Counterpoint Engine (advanced_modules/counterpoint_engine.py)

```python
# Used for voice leading validation

from advanced_modules.counterpoint_engine import VoiceLeadingChecker

checker = VoiceLeadingChecker()
is_valid = checker.check_parallel_fifths(voice1, voice2)
```

### Integration Example

```python
from midi_generator.generators.context_aware_generator import ContextAwareGenerator
from midi_generator.generators.style_fusion import GenreBlender
from advanced_modules.bass_engine import BassEngine
from advanced_modules.counterpoint_engine import VoiceLeadingChecker

# Analyze existing arrangement
gen = ContextAwareGenerator('song.mid')
analysis = gen.analyze()

# Use GenreBlender for hybrid style
blended_style = GenreBlender.blend_features(
    GENRE_PROFILES['jazz'],
    GENRE_PROFILES['funk'],
    weight_a=0.6
)

# Generate bass with BassEngine
bass_engine = BassEngine()
bass_line = bass_engine.generate_walking_bass(
    chord_progression=analysis.chord_progression,
    style='fusion'
)

# Validate voice leading
checker = VoiceLeadingChecker()
if checker.is_valid_voice_leading(bass_line, analysis.tracks[0]):
    print("Voice leading validated ✓")
```

---

## Research Foundation

This system is based on cutting-edge research in:

### 1. Music Accompaniment Systems

**Francois Pachet - The Continuator (2003)**
- Real-time style matching
- Context-aware generation
- Markov-based prediction

**Application**: Our system uses similar principles for analyzing existing style and generating matching material.

### 2. Voice Leading Theory

**Dmitri Tymoczko - A Geometry of Music (2011)**
- Voice leading geometry
- OPTIC spaces
- Efficient voice leading

**Application**: Implemented in `_avoid_voice_leading_errors()` to ensure smooth transitions.

### 3. Chord Extraction Algorithms

**Music21 Library (MIT)**
- Polyphonic chord detection
- Key-finding algorithms (Krumhansl-Schmuckler)
- Harmonic analysis

**Application**: Used in `MidiAnalyzer` for chord progression extraction.

### 4. Style Matching

**Foroughmand-Aarabi et al. - Genre Classification (2024)**
- Timbral features
- Rhythmic patterns
- Harmonic characteristics

**Application**: Informs our `_detect_style_basic()` method.

### 5. Inpainting Techniques

**Content-Aware Fill (Adobe Photoshop)**
- Context analysis
- Boundary blending
- Seamless transitions

**Application**: Adapted for music in `TrackInpainter` class.

---

## Performance Considerations

### Optimization Tips

1. **Analyze Once**: Store analysis result if generating multiple tracks

```python
gen = ContextAwareGenerator('song.mid')
analysis = gen.analyze()  # Do this once

# Generate multiple tracks without re-analyzing
bass = gen.add_track(instrument=33, track_type='bass')
drums = gen.add_track(instrument=128, track_type='percussion')
harmony = gen.add_track(instrument=48, track_type='harmony')
```

2. **Batch Export**: Export all tracks at once instead of one by one

3. **Constraint Tuning**: Disable unnecessary constraints for faster generation

```python
fast_constraints = GenerationConstraints(
    follow_harmony=True,
    match_density=False,         # Disable
    avoid_voice_leading_errors=False,  # Disable
    preserve_texture=False       # Disable
)
```

### Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Analysis | O(n) | n = number of notes |
| Chord extraction | O(n log n) | Sorting notes by time |
| Track generation | O(m) | m = number of measures |
| Voice leading check | O(k²) | k = simultaneous voices |
| Export | O(n) | n = number of notes |

---

## Troubleshooting

### Common Issues

**1. "No chords detected"**

**Cause**: MIDI file has monophonic melody only

**Solution**:
```python
# Provide manual chord progression
gen.add_track(
    instrument=33,
    track_type='bass',
    custom_chords=['C', 'F', 'G', 'C']
)
```

---

**2. "Generated track sounds off-key"**

**Cause**: Key detection error

**Solution**:
```python
# Manually set key in analysis
gen.analysis.key = KeySignature(tonic=0, mode='major', confidence=1.0)  # C major
```

---

**3. "Boundary transitions sound abrupt"**

**Cause**: `smooth_boundaries=False` or insufficient context

**Solution**:
```python
# Always use smooth boundaries for inpainting
inpainter.inpaint_measures(..., smooth_boundaries=True)

# Ensure at least 1 measure of context before/after
```

---

## Future Enhancements (Planned)

1. **Genre Detection Integration**: Use Agent 1's GenreDetector for accurate style detection
2. **Advanced Bass Patterns**: Full integration with BassEngine for all genre styles
3. **Multi-Track Inpainting**: Regenerate multiple tracks simultaneously with harmony preservation
4. **ML-Based Style Matching**: Use pattern recognition for more accurate style matching
5. **Real-Time Generation**: Support for real-time MIDI input/output

---

## Version History

- **v1.0** (2025-11-19): Initial release by Agent 3
  - Core context-aware generation
  - Track inpainting
  - Smart orchestration
  - 25+ comprehensive tests
  - Full documentation

---

## Credits

**Author**: Agent 3 - Context-Aware Generation
**Date**: November 2025
**Part of**: HarmonyModule Library (85,989+ lines)
**Enhancement**: Modular Genre Fusion (10-Agent System)

**Research Sources**:
- Francois Pachet - The Continuator
- Dmitri Tymoczko - A Geometry of Music
- Music21 - MIT Music Analysis Library
- Foroughmand-Aarabi et al. - Genre Classification

---

## License

Part of the HarmonyModule library. See main repository for license information.

---

## Contact & Support

For issues, questions, or contributions:
- GitHub: https://github.com/doseedo/Do
- Issues: Report bugs or request features via GitHub Issues

---

**End of Documentation**

Total Lines: ~800 lines
