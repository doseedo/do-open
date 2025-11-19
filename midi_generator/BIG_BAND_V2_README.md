# Professional Big Band Generator V2

## What's New in V2

### ✅ Fixed Issues from V1:
1. **NO MORE REPETITIVE BRASS** - Brass patterns now vary across sections with:
   - Staccato hits (short punches)
   - Sustained notes (shout chorus)
   - Call-and-response patterns
   - Tower of Power-style horn arrangements

2. **SAX SOLI DOESN'T PLAY CONSTANTLY** - Proper sectional structure:
   - **Intro**: Drums and brass hits only
   - **Head**: Solo melody (lead alto)
   - **Sax Soli**: Full 5-part sax section harmony
   - **Solo Section**: Background pads only (sparse)
   - **Shout Chorus**: Full ensemble with heavy brass
   - **Outro**: Big finish with full section

### New Features

#### 🎯 Proper Arrangement Structure
Based on classic big band form (Basie, Ellington):
- **Intro** (4 bars): Sets up the groove
- **Head** (12 bars): Main melody statement
- **Sax Soli** (12 bars): Feature the sax section
- **Solo Section** (12 bars): Space for improvisation
- **Shout Chorus** (12 bars): Full ensemble climax
- **Head Out** (12 bars): Final melody statement
- **Outro** (12 bars): Big finish

Total: ~84 bars of professionally arranged music

#### 🎺 Tower of Power Horn Arrangements
Uses authentic funk/soul horn writing techniques:
- 90% unison/octave-based (like Greg Adams)
- Staccato hits that "never get in the way"
- Short, punchy brass stabs
- Sustained notes for shout chorus
- Call-and-response between trumpets and trombones

#### 🎷 Intelligent Sax Section Writing
- **Solo melody**: Lead alto only (other instruments rest)
- **Sax soli**: Full 5-part close harmony
- **Background pads**: Sparse, soft pads during solos
- **Dynamic variation**: Changes based on section type

#### 🥁 Varied Drum Patterns
- Swing ride cymbal pattern
- Backbeat on 2 & 4
- Drum fills at section transitions
- Varies intensity by section

## Installation

```bash
pip install mido matplotlib
```

## Usage

### Basic Usage
```bash
python generate_big_band_v2.py
```

### Custom Options
```bash
# Specify output file, tempo, and key
python generate_big_band_v2.py my_arrangement.mid 140 0

# Different tempos and keys
python generate_big_band_v2.py ballad.mid 80 5      # F major ballad
python generate_big_band_v2.py uptempo.mid 200 10   # Bb major bebop
python generate_big_band_v2.py swing.mid 140 3      # Eb major swing
```

## Architecture

### Libraries Used

1. **genres/jazz.py**
   - `JazzGenerator` - Overall jazz generation
   - `BebopMelodyGenerator` - Melody creation
   - `JazzWalkingBass` - Bass line generation
   - `PianoComping` - Piano chord voicings
   - `JazzProgressions` - Chord progressions

2. **genres/funk_soul.py**
   - `FunkSoulGenerator.generate_horn_section()` - Tower of Power-style brass
   - Authentic horn arrangement techniques
   - Staccato hits and sustained notes

3. **algorithms/groove_library.py**
   - Famous drum grooves
   - Swing timing profiles

4. **algorithms/rhythm_engine.py**
   - Groove templates
   - Humanization engine

### Class Structure

```python
ArrangementSection(Enum)
├── INTRO
├── HEAD
├── SAX_SOLI
├── SOLO
├── SHOUT_CHORUS
└── OUTRO

SectionConfig(dataclass)
├── section_type: ArrangementSection
├── num_choruses: int
├── brass_activity: str      # "none", "sparse", "moderate", "heavy"
├── sax_activity: str         # "melody", "soli", "background", "none"
└── solo_instrument: Optional[str]

BigBandArrangementGenerator
├── generate_arrangement()           # Main orchestrator
├── _generate_section()              # Section-specific generation
├── _generate_varied_brass()         # Tower of Power brass
├── _add_sustained_brass()           # Shout chorus sustains
├── _add_call_response()             # Call-response patterns
├── _generate_sax_backgrounds()      # Sparse solo backgrounds
├── _harmonize_sax_section()         # 5-part sax harmony
├── _generate_drum_fill()            # Transition fills
└── export_midi()                     # MIDI file creation
```

## Arrangement Breakdown

### Section-by-Section

#### 1. INTRO (12 bars)
- **Brass**: Moderate activity (punches)
- **Saxes**: Silent (none)
- **Rhythm**: Full (piano, bass, drums)
- **Purpose**: Establish groove and key

#### 2. HEAD (12 bars)
- **Brass**: Sparse hits (don't overpower)
- **Saxes**: Solo melody (lead alto only)
- **Rhythm**: Walking bass, piano comp, swing drums
- **Purpose**: State main theme

#### 3. SAX SOLI (12 bars)
- **Brass**: Sparse background
- **Saxes**: Full 5-part harmony (soli)
- **Rhythm**: Full section
- **Purpose**: Feature sax section

#### 4. SOLO SECTION (12 bars)
- **Brass**: Sparse punctuation
- **Saxes**: Soft background pads
- **Rhythm**: Full section
- **Solo**: Tenor sax (improvised)
- **Purpose**: Create space for soloist

#### 5. SHOUT CHORUS (12 bars)
- **Brass**: HEAVY activity (sustained + hits)
- **Saxes**: Full soli
- **Rhythm**: Drums with fills
- **Purpose**: Climactic ensemble peak

#### 6. HEAD OUT (12 bars)
- **Brass**: Moderate (same as head)
- **Saxes**: Melody (lead alto)
- **Rhythm**: Full section
- **Purpose**: Restate theme

#### 7. OUTRO (12 bars)
- **Brass**: Heavy (big finish)
- **Saxes**: Full soli
- **Rhythm**: Builds to final hit
- **Purpose**: Conclude with power

## Comparison: V1 vs V2

| Feature | V1 (Old) | V2 (New) |
|---------|----------|----------|
| **Brass patterns** | Same hits, same beats, all the time | Varied: staccato, sustained, call-response |
| **Sax section** | Plays constantly, never stops | Alternates: solo, soli, backgrounds, rest |
| **Structure** | No form, just loops | 7 distinct sections with purpose |
| **Horn writing** | Generic | Tower of Power style |
| **Dynamics** | Flat | Builds and releases |
| **Total bars** | ~24 bars | ~84 bars |

## Technical Details

### Brass Voicing Ranges
- **Trumpet**: 60-82 (C4-A#5) - comfortable high brass
- **Trombone**: 40-72 (E2-C5) - comfortable low brass

### Sax Voicing Ranges
- **Alto Sax**: 64-81 (E4-A5)
- **Tenor Sax**: 55-76 (G3-E5)
- **Baritone Sax**: 48-67 (C3-G4)

### Swing Timing
- **Swing Ratio**: 66.7% (2:1 triplet feel)
- **Ride Pattern**: Swing eighths
- **Backbeat**: Beats 2 & 4

### Chord Progression
12-bar jazz blues with sophisticated changes:
```
| I7 | ivm7 VII7 | I7 | I7 |
| IV7 | IV7 | I7 | VIm7 |
| iim7 | V7 | Imaj7 | vim7 |
```

## Output

### MIDI File Structure
- **17 tracks** (one per instrument)
- **General MIDI** compatible
- **Channel 9**: Drums
- **Channels 0-8, 10-15**: Melodic instruments

### Opening the File
Works with:
- **DAWs**: Logic, Ableton, FL Studio, GarageBand, Pro Tools
- **Notation**: MuseScore, Sibelius, Finale, Dorico
- **Players**: Any General MIDI player

## Customization

### Change Arrangement Structure

Edit the `sections` list in `generate_arrangement()`:

```python
sections = [
    SectionConfig(ArrangementSection.INTRO, 1, "moderate", "none"),
    SectionConfig(ArrangementSection.HEAD, 2, "sparse", "melody"),  # 2 choruses
    SectionConfig(ArrangementSection.SOLO, 2, "sparse", "background", "trumpet"),  # Trumpet solo
    SectionConfig(ArrangementSection.SHOUT_CHORUS, 1, "heavy", "soli"),
    SectionConfig(ArrangementSection.OUTRO, 1, "heavy", "soli"),
]
```

### Brass Activity Levels
- **"none"**: No brass
- **"sparse"**: Occasional hits (10-20% coverage)
- **"moderate"**: Regular patterns (40-50% coverage)
- **"heavy"**: Constant brass (80-90% coverage)

### Sax Activity Types
- **"none"**: Saxes rest
- **"melody"**: Lead alto only (solo)
- **"soli"**: Full 5-part sax section
- **"background"**: Sparse pads

## Credits

- **Research**: Tower of Power horn techniques (Greg Adams)
- **Jazz Theory**: Mark Levine, Jerry Coker
- **Arranging**: Duke Ellington, Count Basie principles
- **Implementation**: Consolidated from multiple Phase 3 modules

## License

MIT License - Part of the Dø (Doseedo) MIDI Generator Library
