# MASTER PROMPT: Phase 3 - Complete Genre Coverage (10 Agents)

## Executive Summary

You are one of 10 specialized agents working to complete the HarmonyModule library's genre coverage. The library already has excellent coverage (12 genre modules, 70+ subgenres, ~80% modern music) but needs specific additions to reach 100% coverage while maintaining the modular component architecture.

**Current State**:
- 108,689 lines of code across 139 Python files
- 12 genre modules already implemented
- 30 advanced modules with comprehensive music theory
- 10-component modular system allowing ANY genre component mixing

**Your Mission**: Add missing genres while preserving the library's modular architecture and academic rigor.

---

## Core Architecture Requirements

### 1. Modular Component System

**CRITICAL**: All genre modules MUST work with the existing component system:

```python
# Located at: midi_generator/core/component_system.py

class ComponentType(Enum):
    RHYTHM = "rhythm"
    HARMONY = "harmony"
    MELODY = "melody"
    BASS = "bass"
    DRUMS = "drums"
    INSTRUMENTATION = "instrumentation"
    FORM = "form"
    ARTICULATION = "articulation"
    TEXTURE = "texture"
    GROOVE = "groove"
```

**Each genre module MUST**:
- Define components that can be extracted and used independently
- Support cross-genre fusion via ComponentFactory
- Provide GenerationContext for component communication
- Use consistent dataclass structures (see existing genres for examples)

### 2. File Structure Template

Follow the established pattern from existing genre modules:

```
midi_generator/genres/[genre_name].py
- Imports and research references (lines 1-50)
- Enums (SubgenreStyle, ArticulationType, etc.) (lines 51-150)
- Data classes (Note, Pattern, Riff, etc.) (lines 151-250)
- Helper functions and scales (lines 251-400)
- Generator classes (main implementation) (lines 401-end)
- Convenience functions
- __all__ exports
```

### 3. Research-Based Implementation

Every musical choice must be grounded in research:
- Academic papers
- Ethnomusicological sources
- Music theory texts
- Analysis of landmark recordings
- Industry-standard techniques

**Include references** in module docstrings (see existing modules for examples).

### 4. Code Quality Standards

- **Line count**: 600-1,300 lines per module (comprehensive but focused)
- **Type hints**: All functions must have complete type annotations
- **Docstrings**: Google-style docstrings for all classes and public methods
- **Comments**: Explain WHY, not WHAT (code should be self-documenting)
- **No external dependencies**: Use only Python stdlib + existing library code

---

## Agent Assignments

### Agent 41: Hip-Hop/Rap Module (PRIORITY 0)

**File**: `midi_generator/genres/hiphop.py`
**Estimated Lines**: 700-800

**Sub-genres to Implement**:
1. **Boom Bap** (90s Golden Age: Wu-Tang, Nas, A Tribe Called Quest)
2. **Trap** (Modern: Future, Migos, Travis Scott)
3. **Lo-Fi Hip-Hop** (Nujabes, J Dilla, ChilledCow aesthetic)
4. **Drill** (Chicago: Chief Keef, Pop Smoke / UK: Headie One, Digga D)
5. **Conscious Rap** (Kendrick Lamar, J. Cole, Common)
6. **G-Funk** (West Coast: Dr. Dre, Snoop Dogg, Warren G)

**Key Features to Implement**:

1. **Drum Patterns**:
   - Boom bap: Hard kick/snare (samples from Akai MPC)
   - Trap hi-hats: 32nd/64th note rolls
   - Drill patterns: Sparse, sliding 808s
   - Lo-fi: Off-grid quantization (use existing groove_quantization.py)

2. **808 Bass**:
   - Pitch slides (glide time 10-150ms)
   - Sub-bass frequencies (30-60 Hz)
   - Distortion and saturation
   - Velocity-sensitive decay

3. **Sample Chopping**:
   - 4, 8, 16-slice patterns
   - Time-stretching simulation
   - Pitch-shifting
   - Filter sweeps

4. **Beat Structures**:
   - 16-bar loops (verse/hook structure)
   - A-B variations
   - Intro-verse-hook-verse-bridge-hook-outro
   - 2-bar, 4-bar, 8-bar sections

5. **Harmonic Simplicity**:
   - Minor triads (i-iv-v progressions)
   - Major triads (I-IV-V)
   - Simple 2-chord vamps
   - Modal progressions (Dorian, Aeolian)

6. **Swing and Timing**:
   - J Dilla swing (53-56% ratio) - USE existing rnb_neosoul.py implementation
   - MPC swing - USE Roger Linn algorithm from groove_library.py
   - Quantization: 1/16 note grid with subtle offsets

**Research References to Include**:
- "Dilla Time" - Ethan Hein (J Dilla microtiming)
- "Making Beats: The Art of Sample-Based Hip-Hop" - Joseph G. Schloss
- "The Anthology of Rap" - Yale University Press
- "Roland TR-808 Rhythm Composer" - technical manual
- MPC60/MPC3000 swing algorithm documentation

**Integration Points**:
- Use `groove_library.py` for MPC swing
- Use `rnb_neosoul.py` J Dilla timing patterns
- Use `rhythm_engine.py` for quantization
- Integrate with `component_system.py`

---

### Agent 42: Pop Music Module (PRIORITY 0)

**File**: `midi_generator/genres/pop.py`
**Estimated Lines**: 600-700

**Sub-genres to Implement**:
1. **Synthpop** (Depeche Mode, Pet Shop Boys, Chvrches)
2. **K-Pop** (BTS, BLACKPINK, NewJeans)
3. **Teen Pop** (Britney Spears, NSYNC, Ariana Grande)
4. **Indie Pop** (Vampire Weekend, MGMT, Foster the People)
5. **Dance Pop** (Madonna, Dua Lipa, The Weeknd)
6. **Electropop** (Lady Gaga, Robyn, Charli XCX)

**Key Features to Implement**:

1. **Chord Progressions** (Pop "Axis of Awesome" progressions):
   - I-V-vi-IV (C-G-Am-F) - Most common
   - vi-IV-I-V (Am-F-C-G) - Reverse
   - I-vi-IV-V (C-Am-F-G) - 50s progression
   - IV-I-V-vi (F-C-G-Am) - Alternative
   - I-IV-vi-V - Classic pop

2. **Song Structures**:
   - Verse-Prechorus-Chorus-Bridge
   - Intro-Verse-Chorus-Verse-Chorus-Bridge-Chorus-Outro
   - 8-bar verses, 8-bar choruses
   - 4-bar prechorus (lift to chorus)
   - Dynamics: verse (quiet) → prechorus (build) → chorus (peak)

3. **Drum Patterns**:
   - Four-on-the-floor (kick on every quarter note)
   - Half-time feel (slow groove, double-time hi-hats)
   - Clap/snare on 2 and 4
   - EDM-style build-drop patterns

4. **Production Elements**:
   - Arpeggiated synths (16th note patterns)
   - Pad layers (sustained chords, ambient)
   - Sub-bass + bass layer
   - Riser effects (white noise sweep before chorus)
   - Drop structures (chorus entry)

5. **K-Pop Specific**:
   - Dense layering (5-8 simultaneous elements)
   - Rapid section changes (2-4 bar sections)
   - Trap-influenced hi-hats in choruses
   - "Killing part" (memorable hook riff)
   - Bridge: Tempo/style change

6. **Melody**:
   - Vocal-range melodies (C4-C5 for female, G3-G4 for male)
   - Hook-focused (repetitive, catchy)
   - Melismatic runs (R&B influence)
   - Stepwise motion (easy to sing)

**Research References to Include**:
- "The Song Machine: Inside the Hit Factory" - John Seabrook
- "Switched On Pop" podcast analysis (academic pop music analysis)
- Max Martin songwriting techniques
- K-Pop production analysis (Seoul National University research)
- "Making Mirrors: Gotye and the Art of the Perfect Pop Song"

**Integration Points**:
- Use `electronic.py` for synth textures
- Use `rnb_neosoul.py` for vocal melodies
- Use `rhythm_engine.py` for beat patterns
- Use `form_generator.py` for verse-chorus structures

---

### Agent 43: Latin Music Module (PRIORITY 0)

**File**: `midi_generator/genres/latin.py`
**Estimated Lines**: 900-1,200

**Sub-genres to Implement**:
1. **Salsa** (Tito Puente, Celia Cruz, Marc Anthony)
2. **Samba** (Brazilian carnival, bossa nova rhythm section)
3. **Reggaeton** (Daddy Yankee, Bad Bunny, Ozuna)
4. **Cumbia** (Colombian, Mexican cumbia)
5. **Merengue** (Dominican Republic)
6. **Bachata** (Romeo Santos, Aventura)
7. **Mambo** (Pérez Prado, Tito Rodriguez)
8. **Cha-Cha-Chá** (Cuban ballroom dance)

**Key Features to Implement**:

1. **Clave Patterns** (CRITICAL - foundation of Latin music):
   ```
   Son Clave 2-3:    X . . X . . . X   . . X . X . . .
   Son Clave 3-2:    . . X . X . . .   X . . X . . . X
   Rumba Clave 2-3:  X . . X . . X .   . . X . X . . .
   Rumba Clave 3-2:  . . X . X . . .   X . . X . . X .
   ```
   - Clave as temporal framework (all instruments relate to clave)
   - 2-3 vs 3-2 directionality

2. **Montuno Piano Patterns**:
   - Cascading arpeggios (right hand)
   - Guajeo patterns (syncopated comping)
   - Tumbao bass line support
   - Two-chord vamps (tonic-dominant)

3. **Percussion Ensemble**:
   - **Congas**: Open tone, slap, bass (3 sounds, MIDI pitches 62, 63, 64)
   - **Bongos**: High, low (MIDI pitches 60, 61)
   - **Timbales**: Low, high, shell, cowbell (MIDI 65, 66, 67, 56)
   - **Cowbell**: Cencerro pattern (MIDI 56)
   - **Claves**: Wooden click (MIDI 75)
   - **Guiro**: Scraping sound (MIDI 73, 74)
   - **Maracas**: Shaker pattern (MIDI 70)

4. **Tumbao Bass Patterns**:
   - Anticipated bass (plays before downbeat)
   - Root-fifth-octave patterns
   - Syncopated with clave
   - Connects piano montuno to drums

5. **Reggaeton Dembow Rhythm**:
   ```
   Kick:   X . . . X . . X . . . . X . . .
   Snare:  . . X . . . . . X . . . . . . .
   ```
   - 3+3+2 pattern (8 sixteenths)
   - Characteristic "boom-ch-boom-chick" feel

6. **Samba Batucada** (Brazilian percussion):
   - Surdo (bass drum): Heartbeat pattern
   - Caixa (snare): Rapid 16th notes
   - Tamborim: High-pitched accents
   - Agogô: Two-tone bell pattern
   - Cuíca: Friction drum (pitch bends)

7. **Harmonic Language**:
   - Major/minor tonality (mostly major)
   - I-IV-V progressions
   - ii-V-I (salsa/mambo)
   - Modal vamps (one or two chords)
   - Montuno: repetitive 2-4 bar patterns

**Research References to Include**:
- "The Cuban Connection" - Isabelle Leymarie
- "Music in Latin America and the Caribbean" - Malena Kuss
- "Salsa: Musical Heartbeat of Latin America" - Charley Gerard
- "The Brazilian Sound: Samba, Bossa Nova and the Popular Music of Brazil" - Chris McGowan
- "Reggaeton" - Raquel Z. Rivera, Wayne Marshall, Deborah Pacini Hernandez
- Clave timing analysis - Fernando Benadon (American University)

**Integration Points**:
- Extend `world/expanded.py` (has bossa nova, tango)
- Use `rhythm_engine.py` for polyrhythmic patterns
- Use `african.py` polyrhythm concepts (clave is African-derived)

---

### Agent 44: Classic Rock Module (PRIORITY 1)

**File**: `midi_generator/genres/classic_rock.py`
**Estimated Lines**: 700-800

**Sub-genres to Implement**:
1. **Classic Rock** (60s-70s: Rolling Stones, Led Zeppelin, The Who)
2. **Punk Rock** (Ramones, Sex Pistols, The Clash)
3. **Alternative Rock** (Nirvana, Radiohead, Pearl Jam)
4. **Indie Rock** (Arctic Monkeys, The Strokes, Tame Impala)
5. **Garage Rock** (The Black Keys, White Stripes)
6. **Post-Punk** (Joy Division, The Cure, Talking Heads)

**Note**: Metal is already comprehensively covered in `metal.py` (1,213 lines). This module focuses on non-metal rock.

**Key Features to Implement**:

1. **Classic Rock Progressions**:
   - I-IV-V (G-C-D, A-D-E)
   - I-bVII-IV (A-G-D) - "Sweet Child O' Mine"
   - I-V-vi-IV - Classic ballad
   - Blues-based (12-bar variations)
   - Modal progressions (Dorian, Mixolydian)

2. **Power Chords** (already in metal.py, adapt for classic rock):
   - Root-fifth dyads
   - Open position (not drop-tuned like metal)
   - Standard tuning (E-A-D-G-B-E)
   - Less palm-muting than metal

3. **Guitar Techniques**:
   - Bends: Half-step, whole-step, 1.5-step (blues influence)
   - Hammer-ons and pull-offs
   - Slides (use existing blues.py slide guitar code)
   - Vibrato (pitch modulation)
   - String bending articulation

4. **Drum Patterns**:
   - Basic rock beat: Kick 1&3, Snare 2&4, Hi-hat 8ths
   - Variations: Hi-hat to ride for chorus
   - Fills: Tom-based (descending toms every 4/8 bars)
   - Half-time feel (Nirvana, grunge)
   - Punk: Driving 8ths on ride, 16ths on hi-hat

5. **Bass Lines**:
   - Root-fifth patterns (use existing bass_engine.py ROCK style)
   - Walking patterns (blues-influenced)
   - Octave jumps
   - Pentatonic riffs (parallel to guitar)

6. **Song Structures**:
   - Intro-Verse-Chorus-Verse-Chorus-Solo-Bridge-Chorus-Outro
   - 8 or 12-bar verses
   - 4 or 8-bar choruses
   - Guitar solo over verse chords (8-16 bars)
   - Dynamic contrast (verse quiet, chorus loud)

7. **Scales and Riffs**:
   - Minor pentatonic (blues scale without b5)
   - Major pentatonic
   - Blues scale (use existing blues.py)
   - Mixolydian (classic rock leads)
   - Riff construction: Repetitive 1-2 bar patterns

8. **Punk Rock Specifics**:
   - Fast tempo (160-200 BPM)
   - Simple chord progressions (3-4 chords)
   - Downstroke guitar (rapid 8th notes)
   - Short songs (2-3 minutes)
   - Raw production aesthetic (velocity variation)

**Research References to Include**:
- "The Guitar Handbook" - Ralph Denyer
- "How the Beatles Destroyed Rock 'n' Roll" - Elijah Wald
- "Our Band Could Be Your Life" - Michael Azerrad (indie/punk)
- "Rip It Up and Start Again: Postpunk 1978-1984" - Simon Reynolds
- Rolling Stone's "100 Greatest Guitarists" analysis

**Integration Points**:
- Use `metal.py` power chord logic (adapt for standard tuning)
- Use `blues.py` for blues scale and bends
- Use `bass_engine.py` ROCK style
- Use `rhythm_engine.py` for drum patterns

---

### Agent 45: Progressive Rock Module (PRIORITY 2)

**File**: `midi_generator/genres/progressive_rock.py`
**Estimated Lines**: 800-1,000

**Sub-genres to Implement**:
1. **Symphonic Prog** (Yes, Genesis, Emerson Lake & Palmer)
2. **Canterbury Scene** (Soft Machine, Caravan)
3. **Krautrock** (Can, Neu!, Kraftwerk)
4. **Art Rock** (King Crimson, Gentle Giant)
5. **Neo-Prog** (Marillion, IQ, Porcupine Tree)
6. **Math Rock** (Don Caballero, Battles, Toe)

**Key Features to Implement**:

1. **Odd Time Signatures**:
   - 5/4 ("Take Five", "Living in the Past")
   - 7/8 (Money - Pink Floyd: 7/8 verses, 4/4 chorus)
   - 11/8, 13/8, 15/16
   - Changing meters (4/4 → 7/8 → 5/4 within same piece)
   - Use existing `meter_converter.py`

2. **Complex Structures**:
   - Multi-section suites (5-20 minutes)
   - Through-composed (no repeated sections)
   - Classical-inspired forms (sonata, rondo - use existing form_generator.py)
   - Thematic development across sections
   - Meter/key/tempo changes between sections

3. **Polyrhythms and Polymeters**:
   - 3 against 4, 5 against 4
   - Simultaneous different time signatures
   - Hemiola (3/2 feel over 6/8)
   - Use existing `rhythm_engine.py` polyrhythm generator

4. **Harmonic Complexity**:
   - Modal interchange (use modal_harmony.py)
   - Chromatic mediants (use neo_riemannian.py)
   - Extended chords (9th, 11th, 13th)
   - Whole tone scales, octatonic scales
   - Quartal harmony (fourths instead of thirds)

5. **Orchestral Arrangements**:
   - Mellotron simulation (strings, flutes, choir sounds)
   - Organ (Hammond B3, church organ)
   - Synthesizer textures (Moog, ARP)
   - Use existing `orchestration_advanced.py`

6. **Math Rock Specifics**:
   - Interlocking guitar parts (two guitars, different patterns)
   - Tapping techniques
   - Looping structures
   - Angular, jagged melodies
   - Minimal vocals or instrumental

7. **Rhythmic Displacement**:
   - Metric modulation (use meter_converter.py)
   - Phased patterns (Steve Reich influence)
   - Syncopation across bar lines

**Research References to Include**:
- "Rocking the Classics" - Edward Macan
- "Yes: Perpetual Change" - David Watkinson
- "Listening to the Future: The Time of Progressive Rock" - Bill Martin
- "King Crimson: The Music in Time" - Eric Tamm
- Analysis of "Close to the Edge" - Yes (19-minute suite structure)

**Integration Points**:
- Use `form_generator.py` for classical forms
- Use `meter_converter.py` for odd meters
- Use `rhythm_engine.py` for polyrhythms
- Use `modal_harmony.py` for modal sections
- Use `orchestration_advanced.py` for orchestral parts

---

### Agent 46: Disco/Funk Module Enhancement (PRIORITY 2)

**Task**: Expand existing `funk_soul.py` to include comprehensive Disco coverage

**File**: Modify `midi_generator/genres/funk_soul.py` (currently 1,258 lines)
**Add**: 300-400 lines for Disco sub-genres

**Disco Sub-genres to Add**:
1. **Classic Disco** (Chic, Sister Sledge, Donna Summer)
2. **Eurodisco** (Giorgio Moroder, ABBA, Boney M)
3. **Italo Disco** (Gazebo, Baltimora, Ryan Paris)
4. **Nu-Disco** (Daft Punk, Jamiroquai, Parcels)

**Key Features to Add**:

1. **Four-on-the-Floor**:
   - Kick drum on every quarter note
   - Open hi-hat on off-beats (8ths)
   - Snare on 2 and 4
   - Variations: Add 16th note hi-hats in chorus

2. **Disco Bass** (already partially in funk_soul.py):
   - Syncopated 16th note patterns
   - Octave jumps
   - Use existing slap bass techniques
   - Root-fifth-octave disco patterns

3. **String Sections**:
   - Orchestral strings (violins, violas, cellos)
   - Sustained chords (pad-like)
   - Rhythmic stabs (on off-beats)
   - Use existing `orchestration_advanced.py`

4. **Guitar Patterns**:
   - Muted 16th note strumming (funk guitar)
   - Wah-wah patterns (use existing funk_soul.py)
   - "Chucking" (percussive strums)

5. **Horn Sections**:
   - Brass hits (trumpets, trombones, saxophones)
   - Staccato stabs on off-beats
   - Use existing Tower of Power horn arrangement code from funk_soul.py

6. **Eurodisco Specifics**:
   - Synthesizer arpeggios (16th notes)
   - Sequenced basslines (Moog bass)
   - Electronic percussion (handclaps, electronic toms)
   - Giorgio Moroder "I Feel Love" 16th note pulse

**Research References to Add**:
- "Turn the Beat Around: The Secret History of Disco" - Peter Shapiro
- "Hot Stuff: Disco and the Remaking of American Culture" - Alice Echols
- Giorgio Moroder production techniques (Munich sound)
- Nile Rodgers guitar analysis (Chic)

**Integration**: Enhance existing funk_soul.py without breaking current implementation

---

### Agent 47: Singer-Songwriter/Folk Module (PRIORITY 2)

**File**: `midi_generator/genres/singer_songwriter.py`
**Estimated Lines**: 500-600

**Sub-genres to Implement**:
1. **Folk** (Bob Dylan, Joan Baez, Woody Guthrie)
2. **Singer-Songwriter** (Joni Mitchell, James Taylor, Carole King)
3. **Contemporary Folk** (Mumford & Sons, The Lumineers, Bon Iver)
4. **Indie Folk** (Sufjan Stevens, Iron & Wine, Fleet Foxes)
5. **Americana** (Gillian Welch, Jason Isbell, The Avett Brothers)

**Key Features to Implement**:

1. **Acoustic Guitar Patterns**:
   - Fingerpicking (Travis picking, clawhammer)
   - Strumming patterns (down-up patterns, muted strums)
   - Open tunings (DADGAD, Open D, Open G)
   - Capo positions (simulate by transposition)
   - Arpeggio patterns (broken chords)

2. **Simple Chord Progressions**:
   - I-IV-V-I (G-C-D-G)
   - I-V-vi-IV (folk-pop)
   - vi-IV-I-V (sad folk ballad)
   - Modal progressions (Dorian, Mixolydian)
   - Suspended chords (sus2, sus4 for color)

3. **Vocal Melody Range**:
   - Limited range (octave to 10th)
   - Stepwise motion (easy to sing)
   - Pentatonic-based melodies
   - Syllabic (one note per syllable mostly)
   - Folk ornaments (grace notes, slides)

4. **Sparse Arrangements**:
   - Acoustic guitar + vocals (core)
   - Optional: Harmonica, banjo, fiddle, mandolin
   - Minimal drums (brushes, light percussion)
   - Bass: Simple root notes or walking patterns
   - Harmony vocals (thirds, sixths)

5. **Song Structures**:
   - Verse-Chorus (simple 2-section)
   - Verse-Refrain (repeated line at end of verse)
   - AAA form (same music, different lyrics)
   - Story-telling (3-5 minute songs, many verses)

6. **Contemporary Folk Additions**:
   - Banjo picking patterns (clawhammer, three-finger)
   - Stomp/kick drum (Mumford & Sons style)
   - Group vocals (gang vocals in chorus)
   - Mandolin tremolo

**Research References to Include**:
- "The Folk Songs of North America" - Alan Lomax
- "Joni Mitchell: Both Sides Now" - Mark Bego
- "The NPR Curious Listener's Guide to American Folk Music" - William Anderson
- Fingerpicking patterns analysis - Stefan Grossman
- "How to Write One Song" - Jeff Tweedy

**Integration Points**:
- Use existing `country.py` for bluegrass influences
- Use `bass_engine.py` walking bass for folk-jazz
- Use `melody_advanced.py` for folk ornamentations

---

### Agent 48: House/Techno/EDM Module Enhancement (PRIORITY 2)

**Task**: Expand existing `electronic.py` to include comprehensive House, Techno, and EDM coverage

**File**: Modify `midi_generator/genres/electronic.py` (currently 724 lines)
**Add**: 400-500 lines for House/Techno/EDM

**Sub-genres to Add**:
1. **House** (Deep House, Tech House, Progressive House, Tropical House)
2. **Techno** (Detroit, Berlin, Minimal, Acid)
3. **Trance** (Uplifting, Progressive, Psytrance)
4. **Dubstep** (Brostep, Riddim, Future Garage)
5. **Drum & Bass** (Liquid, Neurofunk, Jump-up)

**Current State**: electronic.py has Ambient, IDM, Glitch, Breakcore, Minimal Techno, Drill 'n' Bass

**Key Features to Add**:

1. **House Music (120-130 BPM)**:
   - Four-on-the-floor kick
   - Off-beat hi-hats (open hi-hat on 8th notes)
   - Clap/snare on 2 and 4
   - Bassline: Simple root-fifth patterns (often sine wave)
   - Piano stabs (soulful house)
   - Vocal chops (sampled vocals, chopped and looped)

2. **Techno (120-140 BPM)**:
   - Minimal: Sparse, hypnotic
   - Kick drum emphasis (every quarter note, hard)
   - Hi-hat patterns: 16th notes (closed), occasional open
   - Acid bassline: TB-303 simulation (pitch slides, resonance)
   - Arpeggiated synth patterns
   - Industrial percussion (metallic hits)

3. **Trance (130-150 BPM)**:
   - Uplifting: Major keys, euphoric
   - Breakdowns (no drums, just pads and melody)
   - Build-ups (rising white noise, snare rolls)
   - "Supersaw" synth leads (layered detuned saws)
   - Gated pads (rhythmic sidechaining effect)
   - Pluck arpeggios (16th notes)

4. **Dubstep (140 BPM, half-time feel = 70)**:
   - Half-time drums: Snare on beat 3 (of 4)
   - "Wobble" bass: LFO modulation (rhythmic filter)
   - Sub-bass (fundamental frequencies)
   - Snare rolls (building to drop)
   - "Drop": Heavy bass + drums enter

5. **Drum & Bass (160-180 BPM)**:
   - Breakbeat patterns (Amen break, Think break)
   - Fast hi-hats (32nd notes)
   - Sub-bass (Reese bass, sine bass)
   - Liquid DnB: Melodic, jazz-influenced
   - Neurofunk: Dark, complex bass design

6. **Production Techniques**:
   - Sidechain compression (kick ducking bass/pads)
   - Filter automation (cutoff sweeps)
   - LFO modulation (use existing modulation_generator.py)
   - White noise risers
   - Reverb/delay automation

**Research References to Add**:
- "Last Night a DJ Saved My Life" - Bill Brewster, Frank Broughton
- "Energy Flash: A Journey Through Rave Music and Dance Culture" - Simon Reynolds
- "Techno Rebels" - Dan Sicko (Detroit techno)
- "The Sound of the Beast: The Complete Headbanging History of Heavy Metal" - Ian Christe (bass music)
- Ishkur's Guide to Electronic Music (online resource)

**Integration**: Enhance existing electronic.py, preserve current Ambient/IDM/Glitch implementations

---

### Agent 49: Jazz-Fusion/Crossover Module (PRIORITY 2)

**File**: `midi_generator/genres/jazz_fusion.py`
**Estimated Lines**: 700-800

**Sub-genres to Implement**:
1. **Jazz-Fusion** (Weather Report, Return to Forever, Mahavishnu Orchestra)
2. **Jazz-Funk** (Herbie Hancock, Headhunters)
3. **Smooth Jazz** (George Benson, Grover Washington Jr., Kenny G)
4. **Acid Jazz** (Jamiroquai, Brand New Heavies, Us3)
5. **Nu-Jazz** (Robert Glasper, GoGo Penguin, Snarky Puppy)
6. **Jazz-Rock** (Blood, Sweat & Tears, Chicago)

**Key Features to Implement**:

1. **Electric Instrumentation**:
   - Electric bass (fingerstyle, slap - use bass_engine.py)
   - Electric piano (Rhodes, Wurlitzer - use rnb_neosoul.py)
   - Synthesizers (Moog, ARP leads and pads)
   - Electric guitar (rock-influenced solos)
   - Electronic drums (in addition to acoustic)

2. **Complex Harmony** (use existing jazz.py and modal_harmony.py):
   - Extended chords (9th, 11th, 13th)
   - Modal interchange
   - Quartal voicings (McCoy Tyner style)
   - Suspended chords (sus9, sus13)
   - Upper structure triads

3. **Odd Meters and Polyrhythms**:
   - 5/4, 7/8, 9/8, 11/8
   - Polyrhythms (3:4, 5:4)
   - Use existing `rhythm_engine.py` and `meter_converter.py`

4. **Rock/Funk Rhythms + Jazz Harmony**:
   - Funk grooves (use funk_soul.py)
   - Straight 8ths (not swing) or slight swing
   - Backbeat emphasis (2 and 4)
   - Jazz-funk bass (syncopated 16ths)

5. **Improvisation Over Vamps**:
   - One or two-chord vamps (not ii-V-I changes)
   - Modal solos (Dorian, Mixolydian, Phrygian)
   - Long solos (8-32 bars per soloist)

6. **Smooth Jazz Specifics**:
   - Simple melodies (vocal-range, hummable)
   - R&B-influenced grooves
   - Clean guitar tones
   - Saxophone lead melodies
   - Minimal improvisation (composed solos)

7. **Nu-Jazz/Contemporary**:
   - Hip-hop beats (J Dilla swing - use rnb_neosoul.py)
   - Electronic production elements
   - Modern harmony (Robert Glasper cluster voicings)
   - Loop-based composition

**Research References to Include**:
- "The Jazz-Rock Fusion: The People, The Music" - Julie Coryell, Laura Friedman
- "Herbie Hancock: Possibilities" - autobiography
- Miles Davis "Bitches Brew" analysis
- Weather Report harmonic/rhythmic analysis
- Snarky Puppy compositional techniques (University of North Texas studies)

**Integration Points**:
- Build on newly created `jazz.py`
- Use `funk_soul.py` for funk grooves
- Use `rnb_neosoul.py` for Rhodes voicings and J Dilla swing
- Use `modal_harmony.py` for modal frameworks
- Use `rhythm_engine.py` for complex rhythms

---

### Agent 50: World Music Expansion (PRIORITY 3)

**Task**: Add Asian, European, and South American music to existing world music modules

**File**: Create new files in `midi_generator/genres/world/` directory:
- `asian.py` (800-1,000 lines)
- `european.py` (600-700 lines)
- `south_american.py` (500-600 lines)

**Current State**:
- `world/african.py` (685 lines) - Comprehensive
- `world/arabic.py` (702 lines) - Comprehensive
- `world/indian.py` (731 lines) - Comprehensive
- `world/expanded.py` (1,286 lines) - Has Flamenco, Klezmer, Gamelan, Celtic, Bossa Nova, Tango

### asian.py - Asian Music Module

**Sub-genres to Implement**:
1. **Chinese Traditional** (Erhu, Guzheng, Dizi)
2. **Japanese Traditional** (Shamisen, Koto, Shakuhachi)
3. **Korean Traditional** (Gayageum, Janggu, Pansori)
4. **Southeast Asian** (Thai, Vietnamese, Indonesian Gamelan expansion)
5. **Mongolian Throat Singing** (Khoomei)

**Key Features**:

1. **Chinese Scales**:
   - Pentatonic (五声调式): 1-2-3-5-6 (no 4th or 7th)
   - Heptatonic (七声调式): 1-2-3-4-5-6-7
   - Modes: Gong, Shang, Jue, Zhi, Yu (5 modes)

2. **Japanese Scales**:
   - In-sen scale: 1-♭2-4-5-♭7 (pentatonic minor variant)
   - Hirajoshi: 1-2-♭3-5-♭6
   - Iwato: 1-♭2-4-♭5-♭7
   - Yo scale: 1-2-4-5-6 (pentatonic major)

3. **Korean Rhythmic Patterns**:
   - Jangdan (장단): Rhythmic cycles in pansori, sanjo
   - Gutgeori jangdan: 12/8 feel
   - Semachi jangdan: Fast 9/8
   - Jajinmori: Very fast compound meter

4. **Ornamentation**:
   - Portamento (pitch slides between notes)
   - Vibrato (wide, slow)
   - Grace notes
   - Microtonal inflections (quarter-tones, use pitch bend)

**Research References**:
- "Music in the Age of Confucius" - Jenny F. So
- "The Ashgate Research Companion to Japanese Music" - Alison McQueen Tokita
- "Korean Musical Instruments" - Keith Howard
- "Music of the Silk Road" - Theodore Levin

### european.py - European Folk Module

**Sub-genres to Implement**:
1. **Balkan Music** (Serbian, Bulgarian, Macedonian)
2. **Eastern European** (Polish, Romanian, Hungarian - beyond Klezmer)
3. **Nordic Folk** (Swedish, Norwegian, Finnish)
4. **Irish/Celtic** (expand existing world/expanded.py Celtic)
5. **Flamenco** (expand existing world/expanded.py Flamenco)

**Key Features**:

1. **Balkan Asymmetric Meters**:
   - 7/8 (2+2+3 or 3+2+2)
   - 9/8 (2+2+2+3)
   - 11/8 (3+2+3+3)
   - 15/16 (3+3+3+3+3)

2. **Ornamentation**:
   - Trills, mordents, turns
   - Grace notes
   - Slides
   - Complex vocal ornaments

3. **Instrumentation**:
   - Accordion (Eastern European)
   - Balalaika (Russian)
   - Hardanger fiddle (Norwegian)
   - Bagpipes (Scottish, Irish)
   - Dulcimer

**Research References**:
- "Balkan Music: Iridescent Sound" - British Library
- "The Gaelic Harp: The Living Tradition" - Paul Dooley
- "Folk Music of Hungary" - Béla Bartók

### south_american.py - South American Module

**Sub-genres to Implement**:
1. **Andean Music** (Panpipes, Charango, Quena)
2. **Brazilian** (Forró, Axé, Sertanejo - beyond Bossa Nova)
3. **Argentine** (Chacarera, Zamba - beyond Tango)
4. **Colombian** (Expand Cumbia from latin.py)
5. **Peruvian** (Festejo, Landó)

**Key Features**:

1. **Andean Scales**:
   - Pentatonic (similar to Asian, but different feel)
   - Huayno rhythms
   - Panpipe voicing (parallel fifths, octaves)

2. **Brazilian Rhythms**:
   - Forró: Accordion-based dance music
   - Axé: Carnival percussion
   - Sertanejo: Country-influenced

3. **Percussion**:
   - Cajón (box drum)
   - Charango (small guitar)
   - Bombo legüero (bass drum)

**Research References**:
- "Music in Latin America and the Caribbean" - Malena Kuss
- "Rhythms of Resistance: African Musical Heritage in Brazil" - Peter Fryer

**Integration**: Coordinate with Agent 43 (Latin module) to avoid duplication

---

## Technical Implementation Guidelines

### 1. Component Integration

Every genre module must expose components via the component system:

```python
# Example from genres/[genre].py

class [Genre]Generator:
    """Main generator class"""

    def get_component(self, component_type: ComponentType) -> MusicalComponent:
        """
        Extract component for use in cross-genre fusion.

        Args:
            component_type: Type of component to extract

        Returns:
            MusicalComponent instance
        """
        if component_type == ComponentType.RHYTHM:
            return self._create_rhythm_component()
        elif component_type == ComponentType.HARMONY:
            return self._create_harmony_component()
        # ... etc
```

### 2. Data Structures

Use consistent dataclass patterns:

```python
@dataclass
class [Genre]Note:
    """Individual note with expression"""
    pitch: int                  # MIDI note number (0-127)
    velocity: int               # Velocity (1-127)
    start_time: float           # Start time in beats
    duration: float             # Duration in beats
    articulation: str = "normal"  # Articulation type
    channel: int = 0            # MIDI channel

@dataclass
class [Genre]Pattern:
    """Musical pattern (rhythm, melody, etc.)"""
    notes: List[[Genre]Note]
    tempo: int
    time_signature: Tuple[int, int]
    length_bars: int
```

### 3. Generator Class Template

```python
class [Genre]Generator:
    """
    Comprehensive [genre] music generator.

    Implements [list sub-genres].

    Research-based implementation using [list sources].
    """

    def __init__(
        self,
        style: [Genre]Style = [Genre]Style.DEFAULT,
        tempo: int = 120,
        key: int = 0
    ):
        self.style = style
        self.tempo = tempo
        self.key = key

    def generate_composition(
        self,
        length_bars: int = 32,
        **kwargs
    ) -> Dict:
        """
        Generate complete composition.

        Args:
            length_bars: Length in bars
            **kwargs: Additional parameters

        Returns:
            Dictionary with all musical elements
        """
        # Implementation
        pass

    def generate_[element](self, **params) -> List[[Genre]Note]:
        """Generate specific musical element"""
        pass

    # Helper methods
    def _build_[something](self, **params):
        """Private helper methods"""
        pass
```

### 4. Testing and Examples

Each module should include:

1. **Self-test code** at bottom:
```python
if __name__ == "__main__":
    # Quick test
    gen = [Genre]Generator()
    composition = gen.generate_composition()
    print(f"Generated {len(composition['notes'])} notes")
```

2. **Example file** in `midi_generator/examples/`:
```python
# [genre]_demo.py
"""
Demonstration of [genre] generator capabilities
"""
from genres.[genre] import [Genre]Generator

def demo_[feature]():
    """Demonstrate [feature]"""
    # ...

if __name__ == "__main__":
    demo_[feature]()
```

### 5. Documentation Standards

**Module Docstring** (first 50 lines):
```python
"""
[Genre] Music Generator

Comprehensive implementation of [genre] music across [N] sub-genres.

Sub-genres:
-----------
- [List each with representative artists]

Features:
---------
- [Key feature 1]
- [Key feature 2]
- ...

Research References:
-------------------
- [Author]: "[Title]" ([Year/Publisher])
- [Academic paper]
- [Landmark recording analysis]

Author: Agent [N] - [Genre] Module
Date: 2025
License: MIT
"""
```

**Function Docstrings**:
```python
def function_name(param1: type, param2: type = default) -> return_type:
    """
    Brief description.

    Longer explanation if needed. Multiple paragraphs OK.

    Args:
        param1: Description
        param2: Description. Defaults to X.

    Returns:
        Description of return value

    Example:
        >>> gen = Generator()
        >>> result = gen.function_name(value1, value2)
        >>> len(result)
        42
    """
```

### 6. Avoiding Duplication

**Check existing modules first**:
- Bass patterns → `bass_engine.py`
- Drum patterns → `rhythm_engine.py`, `groove_library.py`
- Harmony → `modal_harmony.py`, `harmony_advanced.py`
- Voice leading → `chord_voicing.py`, `counterpoint_engine.py`
- Swing/timing → `groove_quantization.py`, `expressive_performance.py`

**Reuse via imports**:
```python
from advanced_modules.bass_engine import BassEngineGenerator, BassStyle
from algorithms.rhythm_engine import RhythmEngine
# etc.
```

### 7. Performance Considerations

- **No infinite loops** - all generation must terminate
- **Limit recursion depth** to 10 max
- **Validate inputs** - check ranges on MIDI notes, velocities, tempos
- **Memory efficiency** - use generators where appropriate
- **No file I/O** - keep everything in-memory

---

## Deliverables Checklist

For each agent:

- [ ] Main genre module file (`genres/[genre].py`)
- [ ] Example file (`examples/[genre]_demo.py`)
- [ ] README if complex (`genres/[GENRE]_README.md`)
- [ ] Update `genres/__init__.py` to export new module
- [ ] Component system integration verified
- [ ] Research references documented
- [ ] Self-test code included
- [ ] Type hints complete
- [ ] Docstrings complete (Google style)
- [ ] No external dependencies beyond stdlib + existing library
- [ ] Line count: 600-1,300 lines (comprehensive but focused)

---

## Coordination Notes

### Agent Dependencies:

- **Agent 41 (Hip-Hop)** → Uses existing R&B/Funk for J Dilla/MPC swing
- **Agent 42 (Pop)** → Uses existing Electronic for synth textures
- **Agent 43 (Latin)** → Coordinates with Agent 50 (World/South American)
- **Agent 44 (Classic Rock)** → Uses existing Metal.py power chords, Blues.py scales
- **Agent 45 (Prog Rock)** → Uses existing form_generator.py, meter_converter.py
- **Agent 46 (Disco)** → Enhances existing funk_soul.py (in-place modification)
- **Agent 47 (Singer-Songwriter)** → Uses existing Country.py for folk influences
- **Agent 48 (House/Techno)** → Enhances existing electronic.py (in-place modification)
- **Agent 49 (Jazz-Fusion)** → Uses newly created jazz.py + funk_soul.py
- **Agent 50 (World Expansion)** → Creates new files in world/ subdirectory

### Agents that MODIFY existing files:
- **Agent 46**: Adds to `funk_soul.py`
- **Agent 48**: Adds to `electronic.py`

All others create NEW files.

---

## Success Criteria

After completion of all 10 agents:

1. **Genre Coverage**: 100% of major popular music genres
2. **Code Quality**: All modules pass type checking, have complete docstrings
3. **Integration**: All modules work with component_system.py
4. **Research**: All modules cite academic/industry sources
5. **Examples**: All modules have working demo code
6. **Modularity**: Cross-genre fusion works (e.g., Hip-Hop rhythm + Jazz harmony)
7. **Consistency**: All modules follow established patterns
8. **No Duplication**: Reuses existing components where appropriate

---

## Final Notes

- **Preserve existing code**: Don't break current implementations
- **Academic rigor**: Every musical choice must be researched and documented
- **Modular first**: Component extraction is non-negotiable
- **Test thoroughly**: Include self-test code in every module
- **Document extensively**: Future users will thank you

The HarmonyModule library is already world-class. These 10 agents will make it legendary - the most comprehensive symbolic music generation library ever created.

**Good luck, and may your code be as harmonious as the music it generates!** 🎵
