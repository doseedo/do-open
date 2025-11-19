# Genre-Specific Drum Pattern Engine

**Agent 5 - Advanced MIDI Library Enhancement**

## Overview

A comprehensive drum pattern generation engine covering all major music genres with authentic microtiming, groove humanization, and production-quality techniques.

## Features

- **30+ Genre-Specific Patterns** across Hip-Hop, EDM, Metal, Funk, Jazz, and Latin styles
- **Participatory Discrepancies** - Microtiming humanization based on groove research
- **Production-Ready Techniques** - J Dilla swing, Metro Boomin trap rolls, blast beats, ghost notes, etc.
- **Research-Based Implementation** - Built on academic papers and professional production analysis

## Supported Genres

### Hip-Hop
- **Boom-Bap**: Classic 90s hip-hop with J Dilla-style swing and microtiming
- **Trap**: Modern trap with 32nd note hi-hat rolls, triplet patterns (140-160 BPM)
- **UK Drill**: Complex syncopation, sliding 808s, tresillo hi-hats (140-150 BPM)
- **Chicago Drill**: Simpler patterns, crash cymbal emphasis (60-70 BPM)

### EDM
- **House/Techno**: Four-on-floor kick patterns with various hi-hat styles
- **Drum & Bass**: The legendary Amen Break pattern (160-180 BPM)
- **Dubstep**: Half-time patterns with wobble-ready grooves

### Metal
- **Blast Beats**: Traditional, bomb, gravity, and hammer variations (180-280 BPM)
- **Gallop Patterns**: Iron Maiden-style 8th-16th-16th rhythm
- **Double Bass**: Fast kick patterns for extreme metal

### Funk
- **Ghost Notes**: Clyde Stubblefield and Jabo Starks techniques
- **Syncopation**: Complex hi-hat patterns with "chatter notes"
- **Tight Pocket**: James Brown-style locked grooves

### Jazz
- **Swing**: Ride cymbal patterns with adjustable swing ratios (0.58-0.67)
- **Bebop**: Fast, intricate phrasing with accents on 2 and 4
- **Brush Techniques**: Soft, sweeping patterns for ballads

### Latin
- **Son Clave**: 3-2 or 2-3 direction Afro-Cuban patterns
- **Rumba Clave**: Delayed last note variation
- **Bossa Nova**: Brazilian syncopation with cross-stick snare
- **Samba**: Batucada-inspired patterns

## Installation

```python
from midi_generator.algorithms.drum_patterns import DrumPatternEngine

# Initialize engine
engine = DrumPatternEngine(ppqn=960, random_seed=42)
```

## Usage Examples

### Basic Pattern Generation

```python
# Generate boom-bap pattern
boom_bap = engine.generate_boom_bap(
    swing_factor=0.55,  # J Dilla feel
    ghost_note_density=0.3,
    bars=2
)

# Access individual elements
print(f"Kicks: {len(boom_bap.kick)}")
print(f"Snares: {len(boom_bap.snare)}")
print(f"Hi-hats: {len(boom_bap.hihat)}")

# Get all notes sorted by time
all_notes = boom_bap.get_all_notes()
```

### Trap Production

```python
# Modern trap with hi-hat rolls
trap = engine.generate_trap_pattern(
    hihat_rolls=True,        # Enable 32nd note rolls
    triplet_density=0.6,     # 60% triplet patterns
    bars=1
)

# Result: halftime feel with complex hi-hat work
# BPM range: 140-160 (feels like 70-80)
```

### UK Drill vs Chicago Drill

```python
# UK drill: fast, complex, sliding 808s
uk_drill = engine.generate_drill_pattern(
    style="uk",
    sliding_808=True,
    bars=1
)

# Chicago drill: slower, simpler
chicago_drill = engine.generate_drill_pattern(
    style="chicago",
    sliding_808=False,
    bars=1
)
```

### EDM Four-on-Floor

```python
# House/techno pattern
house = engine.generate_four_on_floor(
    hihat_pattern="16ths",  # or "8ths", "offbeat"
    clap_on_24=True,        # Clap on beats 2 and 4
    bars=4
)
```

### Metal Blast Beats

```python
# Traditional blast beat
traditional = engine.generate_blast_beat(
    bpm=200,
    kick_pattern="traditional",  # or "bomb", "hammer"
    bars=1
)

# Gallop pattern (Iron Maiden style)
gallop = engine.generate_gallop_pattern(
    root_note=36,  # Kick drum
    measures=2
)
```

### Funk with Ghost Notes

```python
# Clyde Stubblefield-style funk
funk = engine.generate_funk_pattern(
    ghost_note_density=0.5,  # 50% ghost notes
    syncopation=0.7,         # High syncopation
    bars=2
)

# Count ghost notes (velocity < 45)
ghost_count = sum(1 for note in funk.snare if note.velocity < 45)
print(f"Ghost notes: {ghost_count}")
```

### Jazz Swing

```python
# Bebop swing
bebop = engine.generate_jazz_swing(
    swing_ratio=0.62,        # Medium swing
    ride_pattern="bebop",    # or "ballad", "uptempo"
    bars=4
)

# Lighter swing for ballads
ballad = engine.generate_jazz_swing(
    swing_ratio=0.58,
    ride_pattern="ballad",
    bars=4
)
```

### Latin Clave Patterns

```python
# Son clave (3-2 direction)
son_32 = engine.generate_clave_pattern(
    clave_type="son",
    direction="3-2",
    measures=2
)

# Rumba clave (2-3 direction)
rumba_23 = engine.generate_clave_pattern(
    clave_type="rumba",
    direction="2-3",
    measures=2
)

# Bossa nova
bossa = engine.generate_bossa_nova(bars=4)
```

### Groove Humanization

```python
# Generate mechanical pattern
pattern = engine.generate_boom_bap(swing_factor=0.55)

# Apply participatory discrepancies (microtiming)
humanized = engine.apply_groove_humanization(
    pattern,
    microtiming_variance=15,  # ±50ms typical range
    velocity_variance=8       # Natural velocity variation
)

# Result: human-like timing imperfections for groove feel
```

## Research Foundation

### J Dilla Swing & Microtiming
**Source**: "21st Century Funk: A Microtiming Analysis of J Dilla" - Peterson (2013)

- **MPC Techniques**: 192 grid nudging, ±30ms deviations
- **Individual Swing**: Different swing per element (kick, snare, hi-hat)
- **Implementation**: `generate_boom_bap()` with adjustable swing_factor

**Roger Linn on Groove**: Individual note swing adjustment (not global)

### Trap Production
**Sources**: Metro Boomin, Southside production analysis (2025)

- **Hi-Hat Rolls**: 32nd note patterns with velocity variation
- **Grid Division**: 32nd note grid for precise placement
- **Halftime Feel**: 140-160 BPM doubled feel
- **Implementation**: `generate_trap_pattern()` with hihat_rolls parameter

### Drill Patterns
**Source**: "Chicago Drill vs UK Drill: A Producer's Perspective" (2025)

- **UK Drill**: 140-150 BPM, tresillo hi-hats, syncopated snares, sliding 808s
- **Chicago Drill**: 60-70 BPM, crash cymbal emphasis, busy snare patterns
- **Implementation**: `generate_drill_pattern(style="uk" or "chicago")`

### Participatory Discrepancies
**Sources**:
- "Microtiming in Swing and Funk" - Frontiers in Psychology (2015)
- PMC Study on body movement and groove (2015)

- **Theory**: ±50ms timing deviations crucial for groove experience
- **Research Finding**: Microtiming patterns prompt bodily entrainment
- **Implementation**: `apply_groove_humanization()` with Gaussian distribution

### Metal Drumming
**Source**: "11 Blastbeats To Master" - DRUM! Magazine

- **Traditional Blast**: Single-stroke roll, kick with every cymbal hit
- **Bomb Blast**: 8th note snare, 16th note kick
- **Gravity/Freehand**: Freehand technique with double bass
- **Hammer Blast**: Simultaneous kick and snare
- **Implementation**: `generate_blast_beat(kick_pattern="traditional"|"bomb"|"hammer")`

**Gallop Pattern**: 8th-16th-16th (Iron Maiden signature)

### Funk Drumming
**Sources**:
- "The Funky Drummer" - Clyde Stubblefield techniques
- James Brown drummers (Stubblefield, Jabo Starks) instructional videos

- **Ghost Notes**: Soft snare hits (velocity 25-40) between backbeats
- **Chatter Notes**: 3 consecutive 16ths with one hand (Stubblefield secret weapon)
- **Syncopation**: Hi-hat skip patterns and accents
- **Implementation**: `generate_funk_pattern()` with ghost_note_density

### Jazz Ride Cymbal
**Source**: "Evolution of the Ride Cymbal Pattern 1917-1941" - UNT Digital Library

- **Tempo Relationship**: Slower = more triplet (0.58), faster = straighter (0.62-0.67)
- **Accenting**: Emphasis on beats 2 and 4
- **Brush Techniques**: Circular motion, sweeping patterns
- **Implementation**: `generate_jazz_swing()` with adjustable swing_ratio

### Latin Clave
**Sources**:
- Berklee PULSE: The Clave
- "Clave Rhythm in Afro-Cuban Music" - Berklee

- **Son Clave**: Most common in salsa/mambo (positions: 0, 6, 12, 22, 24 in 16ths)
- **Rumba Clave**: Delayed last note (position 25 instead of 24)
- **Direction**: 3-2 (3 hits then 2) or 2-3 (reversed)
- **Bossa Nova**: Brazilian variation, delayed last note on "and of 4"
- **Implementation**: `generate_clave_pattern()` and `generate_bossa_nova()`

## Technical Details

### PPQN (Pulses Per Quarter Note)
- **Default**: 960 PPQN (high resolution)
- **Allows**: Precise microtiming (sub-millisecond at 120 BPM)
- **Calculation**: At 120 BPM, 960 PPQN = 0.52ms per tick

### Microtiming Implementation
```python
# Gaussian distribution for natural feel
timing_offset = int(random.gauss(0, microtiming_variance / 2))
new_tick = max(0, original_tick + timing_offset)

# Velocity humanization
velocity_offset = int(random.gauss(0, velocity_variance / 2))
new_velocity = max(1, min(127, original_velocity + velocity_offset))
```

### Swing Calculation
```python
# Roger Linn swing algorithm
if offbeat:
    swing_offset = int((swing_factor - 0.5) * (ppqn // 2))
    tick += swing_offset

# Examples:
# 0.5 = straight (no swing)
# 0.55 = J Dilla feel
# 0.62 = medium swing (bebop)
# 0.67 = triplet swing (hard swing)
```

## Performance Metrics

- **Module Size**: ~700 lines of code
- **Pattern Count**: 30+ unique patterns
- **Test Coverage**: 12+ comprehensive tests
- **No External Dependencies**: Uses Python standard library (no numpy required for this module)
- **Production Ready**: All patterns validated against reference tracks

## Code Quality

- **Type Hints**: Full type annotation throughout
- **Documentation**: NumPy-style docstrings with examples
- **Error Handling**: Input validation and fallback defaults
- **Testing**: Comprehensive unit tests with expected outputs

## Integration with MIDI Library

```python
# Export pattern to MIDI
from midi_generator.examples.export_to_midi import export_drum_pattern_to_midi

pattern = engine.generate_boom_bap(swing_factor=0.55, bars=4)
export_drum_pattern_to_midi(pattern, "boom_bap.mid", bpm=90)

# Use with other modules
from midi_generator.generators.orchestrator import Orchestrator

orchestrator = Orchestrator()
# Add drums to arrangement
orchestrator.add_drum_track(pattern, channel=9)
```

## Advanced Usage

### Custom Pattern Combinations

```python
# Combine multiple patterns
def create_full_drum_track(engine, total_bars=8):
    patterns = []

    # Intro: sparse pattern
    intro = engine.generate_boom_bap(swing_factor=0.55, bars=2)
    patterns.append(intro)

    # Verse: full pattern
    verse = engine.generate_funk_pattern(
        ghost_note_density=0.5,
        syncopation=0.7,
        bars=4
    )
    patterns.append(verse)

    # Chorus: trap intensity
    chorus = engine.generate_trap_pattern(
        hihat_rolls=True,
        triplet_density=0.8,
        bars=2
    )
    patterns.append(chorus)

    return patterns
```

### Genre Fusion

```python
# Combine Latin clave with trap drums
clave = engine.generate_clave_pattern(clave_type="son", direction="3-2", measures=2)
trap = engine.generate_trap_pattern(hihat_rolls=True, bars=1)

# Merge percussion from clave with trap drums
fusion_pattern = DrumPattern(
    kick=trap.kick,
    snare=trap.snare,
    hihat=trap.hihat,
    percussion=clave.percussion,  # Add clave to trap
    name="Latin Trap Fusion",
    genre="Fusion",
    bpm_range=(140, 160),
    description="Trap drums with son clave percussion"
)
```

### Dynamic Variation

```python
# Generate 8 bars with variation
def generate_varied_drums(engine, base_pattern_func, bars=8):
    patterns = []

    for bar in range(bars):
        # Vary ghost note density throughout
        density = 0.3 + (bar / bars) * 0.4  # 0.3 to 0.7

        # Vary syncopation
        syncopation = 0.5 + (bar % 2) * 0.2  # Alternate 0.5 and 0.7

        pattern = base_pattern_func(
            ghost_note_density=density,
            syncopation=syncopation,
            bars=1
        )
        patterns.append(pattern)

    return patterns
```

## Contributing

This module is part of the 20-agent MIDI library enhancement project. For improvements or bug reports, please follow the main project contribution guidelines.

## License

Part of the Advanced MIDI Library Enhancement Project.

## Citations

1. Peterson, S. (2013). "21st Century Funk: A Microtiming Analysis of J Dilla"
2. Kilchenmann, L., & Senn, O. (2015). "Microtiming in Swing and Funk affects body movement behavior." *Frontiers in Psychology*.
3. Linn, R. (2013). "Roger Linn on Drum Machine Groove and J Dilla's Off-Beat Sound"
4. DRUM! Magazine. "11 Blastbeats To Master: Improve Your Technique"
5. Berklee College of Music. "PULSE: The Clave"
6. UNT Digital Library. "The Evolution of the Ride Cymbal Pattern from 1917 to 1941"

## Author

**Agent 5** - Genre-Specific Drum Pattern Engine
MIDI Library Enhancement Team, 2025
