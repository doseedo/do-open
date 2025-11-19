# HARMONYMODULE LIBRARY - COMPREHENSIVE INVENTORY

## Executive Summary
The Harmonymodule library is a state-of-the-art MIDI generation and music composition system with modular architecture supporting 35+ genres and featuring 10 distinct component types. This inventory catalogs all available modular components, transformation capabilities, algorithmic techniques, analysis capabilities, and music theory coverage.

---

## SECTION 1: COMPONENT SYSTEM & MODULAR ARCHITECTURE

### ComponentType Options (from component_system.py)
The library defines 10 core musical component types that can be mixed and matched across genres:

1. **RHYTHM** - Rhythmic patterns and timing
2. **HARMONY** - Chord progressions and voicings
3. **MELODY** - Melodic lines and motifs
4. **BASS** - Bass lines and bass patterns
5. **DRUMS** - Drum patterns and percussion
6. **INSTRUMENTATION** - Instrument selection and orchestration
7. **FORM** - Song structure (verse/chorus/bridge/etc)
8. **ARTICULATION** - Performance articulation (staccato, legato, etc)
9. **TEXTURE** - Musical texture (homophonic/polyphonic/heterophonic)
10. **GROOVE** - Groove and feel (swing, syncopation, microtiming)

### Dependency Resolution (Automatic Generation Order)
- RHYTHM: No dependencies (generates first)
- HARMONY: Requires RHYTHM
- FORM: Requires HARMONY
- GROOVE: Requires RHYTHM
- MELODY: Requires HARMONY + FORM
- BASS: Requires HARMONY + RHYTHM
- DRUMS: Requires RHYTHM + GROOVE
- TEXTURE: Requires HARMONY
- ARTICULATION: Requires MELODY
- INSTRUMENTATION: No dependencies

---

## SECTION 2: GENERATOR MODULES

### Location: `/midi_generator/generators/`

#### 2.1 Form Generator (`form_generator.py`)
**Musical Forms Supported:**
- Classical Forms:
  - SONATA (exposition-development-recapitulation)
  - RONDO (refrain-couplet structure)
  - THEME_AND_VARIATIONS
  - FUGUE
  - BINARY (two-part)
  - TERNARY (ABA)
  
- Popular Forms:
  - VERSE_CHORUS
  - AABA
  - TWELVE_BAR_BLUES
  - VERSE_CHORUS_BRIDGE
  - THROUGH_COMPOSED

**Key Relationships:**
- TONIC, DOMINANT, SUBDOMINANT
- RELATIVE_MAJOR, RELATIVE_MINOR
- PARALLEL_MAJOR, PARALLEL_MINOR
- MEDIANT, SUBMEDIANT

**FormSection Properties:**
- Key relationships to tonic
- Section length in bars
- Character/mood
- Thematic material
- Development level (0-1)
- Dynamic level (0-1)
- Texture density (0-1)

#### 2.2 Development Engine (`development_engine.py`)
**Motivic Development Techniques (15 techniques):**
1. REPETITION - Exact and varied repetition
2. TRANSPOSITION - Moving to different keys
3. SEQUENCE - Transposed patterns
4. INVERSION - Melodic mirror/inversion
5. RETROGRADE - Reversed sequence
6. RETROGRADE_INVERSION - Both retrograde and inverted
7. AUGMENTATION - Slower/longer durations
8. DIMINUTION - Faster/shorter durations
9. FRAGMENTATION - Using part of motif
10. EXTENSION - Adding notes
11. INTERPOLATION - Inserting material
12. RHYTHMIC_SHIFT - Changing beat placement
13. INTERVALLIC_EXPANSION - Making intervals larger
14. INTERVALLIC_CONTRACTION - Making intervals smaller
15. OCTAVE_DISPLACEMENT - Changing register

#### 2.3 Orchestrator (`orchestrator.py`)
**Orchestration Styles:**
- CLASSICAL (transparent, balanced)
- ROMANTIC (lush, full)
- IMPRESSIONIST (colorful, sparse)
- MODERN (angular, percussive)
- FILM (powerful, emotional)
- CHAMBER (intimate, 3-10 players)
- BIG_BAND (jazz orchestra)
- POP (contemporary)

**Texture Types:**
- MELODY, HARMONY, BASS
- COUNTERMELODY, OSTINATO
- PEDAL, TUTTI

**Intelligent Features:**
- Automatic instrument selection by register
- Voice leading optimization
- Tessitura-aware writing
- Range validation and transposition
- Professional doubling rules
- Dynamic balance management

#### 2.4 Texture Generator (`texture_generator.py`)
**Accompaniment Patterns:**
- BLOCK_CHORDS - Static chord voicings
- ALBERTI_BASS - Classic broken chord pattern
- BROKEN_CHORDS - Varied broken chord patterns
- ARPEGGIATED - Multiple arpeggiation styles
- WALTZ - 3/4 time patterns
- STRIDE - Stride piano patterns
- OSTINATO - Repeating patterns
- PEDAL_POINT - Held bass note
- COUNTERMELODY - Secondary melodic line
- WALKING_BASS - Jazz walking patterns
- REPEATED_CHORDS - Rhythmic chord strumming

**Texture Types:**
- MONOPHONIC - Single line
- HOMOPHONIC - Melody + accompaniment
- POLYPHONIC - Multiple independent lines
- HETEROPHONIC - Variations of same melody

#### 2.5 Advanced Harmony Generator (`advanced_harmony_generator.py`)
- Voice leading optimization
- Secondary dominants
- Borrowed chords
- Chromatic mediant progressions
- Harmonic rhythm control

#### 2.6 Transition Engine (`transition_engine.py`)
**Modulation Types:**
- COMMON_CHORD (pivot chord)
- DIRECT (abrupt key change)
- SEQUENTIAL (through sequence)
- ENHARMONIC (reinterpretation)
- CHROMATIC_MEDIANT (third relation)
- MODAL_MIXTURE (borrowed chords)

**Transition Types:**
- BUILD_UP (increase energy)
- BREAKDOWN (reduce texture)
- FILL (drum/melodic fill)
- CRESCENDO / DECRESCENDO
- RISER (EDM-style)
- TURNAROUND (harmonic)
- PAUSE, ACCELERANDO, RITARDANDO

---

## SECTION 3: TRANSFORMATION MODULES

### Location: `/midi_generator/transformation/`

#### 3.1 Style Transfer Engine (`style_transfer.py`)
**Transformation Dimensions:**

1. **Harmonic Transformation:**
   - Reharmonization (classical, jazz, pop styles)
   - Chord substitution and voice leading
   - Modal interchange and borrowed chords
   - Upper structure triads
   - Polychord generation

2. **Rhythmic Transformation:**
   - Groove application (swing, shuffle, straight)
   - Time signature conversion (4/4 → 7/8, 5/4, etc.)
   - Syncopation addition/removal
   - Quantization styles (strict, loose, human)

3. **Melodic Transformation:**
   - Ornamentation (baroque, romantic styles)
   - Simplification/elaboration
   - Contour modification (stepwise ↔ angular)
   - Range shifting

4. **Instrumental Transformation:**
   - Re-orchestration (piano → orchestra, etc.)
   - Timbre mapping
   - Register optimization
   - Texture redistribution

**Predefined Style Profiles:**
- Classical, Jazz, Blues, Funk, Rock
- Latin, Electronic, Pop, Folk, World Music
- Each with chord types, harmonic rhythm, swing ratio, syncopation level, ornamentation density, interval preference, chromatic usage, and register preferences

#### 3.2 Arrangement Engine (`arrangement_engine.py`)
**Arrangement Templates:**
- BIG_BAND (4 trumpets, 4 trombones, 5 saxes, rhythm section)
- STRING_QUARTET (2 violins, viola, cello)
- SOLO_PIANO (complete arrangement)
- POP_BAND (drums, bass, guitar, keys, vocals)
- JAZZ_COMBO (piano, bass, drums, horn)
- ORCHESTRA (full symphonic)

**Arranging Principles:**
- Voice leading optimization
- Register distribution
- Instrumental idioms
- Texture variation
- Dynamic balance

#### 3.3 MIDI Inpainting Engine (`inpainting_engine.py`)
**Content-Aware Fill for Music:**
- Section regeneration with new chords/style
- Boundary smoothing for seamless transitions
- Reharmonization while preserving content
- Genre morphing with smooth blending
- Melody preservation with harmony changes
- Context-aware generation

**Use Cases:**
- Reharmonize bridge with jazz chords
- Change verse to EDM style while keeping chorus as rock
- Add variation to repeated sections
- Fix awkward progressions
- Dynamic style changes in arrangements

#### 3.4 Tempo Converter (`tempo_converter.py`)
**Intelligent Tempo Conversion:**
- Style-aware conversion (jazz, classical, EDM, world)
- Double-time and half-time feel generation
- Automatic subdivision adjustment
- Swing/groove preservation and adaptation
- Articulation adjustment
- Genre-appropriate tempo range validation
- Phrase-aware conversion

**Genre-Specific Tempo Ranges:**
- Ballad: 60-80 BPM
- Medium swing: 120-160 BPM
- Up-tempo jazz: 200-300 BPM
- Funk: 90-110 BPM
- House: 120-130 BPM
- Techno: 120-140 BPM
- Dubstep: 140 BPM (half-time at 70)
- Drum & Bass: 160-180 BPM

#### 3.5 Meter Converter (`meter_converter.py`)
**Time Signature Conversion:**
- Simple conversions (4/4 → 3/4, 2/4, 6/8)
- Odd meter conversions (4/4 → 5/4, 7/8, 11/8)
- Compound to simple conversions
- Multiple strategies: STRETCH, REDISTRIBUTE, TRUNCATE

**Metric Modulation:**
- Elliott Carter-style metric modulation
- Smooth tempo transitions via pivot rhythms
- Mathematically precise calculations

**Phrase Preservation:**
- Intelligent phrase boundary detection
- Maintains phrase structure
- Preserves melodic/harmonic content

---

## SECTION 4: ALGORITHMIC COMPOSITION MODULES

### Location: `/midi_generator/algorithms/`

#### 4.1 L-System (`lsystem.py`)
**Formal Grammar-Based Melody Generation:**

**Musical Symbols:**
- Pitch Movement: FORWARD, BACKWARD, OCTAVE_UP, OCTAVE_DOWN, JUMP_THIRD, JUMP_FIFTH
- Rhythm: LENGTHEN, SHORTEN, REST
- Dynamics: LOUDER, SOFTER
- Structure: PUSH, POP
- Notes: NOTE, CHORD

**L-System Types:**
- Context-free (basic production rules)
- Context-sensitive (rules depend on neighbors)
- Parametric (control pitch, rhythm, dynamics)
- Stochastic (probabilistic rule selection)

**Pre-built Musical Grammars:**
- Bach style
- Minimalist
- Jazz

**Research Base:** Prusinkiewicz (1986), Mason & Saffle (1994)

#### 4.2 Cellular Automata (`cellular_automata.py`)
**1D and 2D Cellular Automata for Music:**

**1D - Elementary CA:**
- Wolfram's elementary CA rules (256 possible)
- Famous rules: 30 (chaotic), 110 (Turing complete), 90 (Sierpiński), 184 (traffic)

**2D - Conway's Game of Life**

**Interpretation Modes:**
- Pitch mapping
- Rhythm mapping
- Dynamics mapping

**Applications:**
- Generative melodies from evolving patterns
- Rhythmic sequences from CA evolution
- Harmonic progressions from 2D CA
- Ambient/experimental texture generation

#### 4.3 Constraint Solver (`constraint_solver.py`)
**CSP-Based Music Generation:**

**Constraint Types:**
- HARD constraints: Must be satisfied (voice leading rules, range limits)
- SOFT constraints: Preferences (stepwise motion, climax placement)

**Algorithms:**
- Backtracking search with forward checking
- Arc consistency algorithms (AC-3)
- Heuristics: Most Constrained Variable, Least Constraining Value

**Applications:**
- Melody generation with voice leading
- Harmony generation with spacing rules
- Counterpoint generation
- Style-based constraint sets

#### 4.4 Rhythm Engine (`rhythm_engine.py`)
**Advanced Rhythm & Groove System:**

**Timing Styles:**
- LOCKED (quantized)
- TIGHT (slight deviation)
- LAID_BACK (notes slightly late)
- RUSHING (notes slightly early)
- DRUNK (heavy deviation)
- HUMAN (natural variation)
- MACHINE (perfect)

**Groove Features:**
- Groove template extraction and application
- Advanced polyrhythm generation (3 against 4, etc.)
- Humanization engine
- Rhythm transformations (augmentation, retrograde, shuffle)

**Groove Intensity:**
- SUBTLE (0.25), LIGHT (0.5), MEDIUM (0.75), HEAVY (1.0), EXTREME (1.5)

#### 4.5 Groove Library (`groove_library.py`)
**Famous Drum Grooves Database:**
- Purdie shuffle (Bernard Pretty Purdie)
- Motown backbeat
- Afrobeat patterns
- Genre-specific microtiming profiles
- Professional drummer timing models
- Ghost notes and embellishments

---

## SECTION 5: ANALYSIS & DETECTION MODULES

### Location: `/midi_generator/analysis/`

#### 5.1 MIDI Analyzer (`midi_analyzer.py`)
**Comprehensive MIDI Analysis:**

1. **Key Detection:**
   - Krumhansl-Schmuckler algorithm
   - Tonic and mode detection (major/minor)
   - Confidence scoring

2. **Tempo Detection:**
   - Automatic tempo discovery
   - Tempo stability analysis
   - Rubato detection

3. **Time Signature Detection:**
   - Meter detection
   - Time signature extraction
   - Polymetric detection

4. **Chord Recognition:**
   - Automatic chord detection
   - Chord inversion detection
   - Confidence scoring
   - Root motion analysis

5. **Melody Extraction:**
   - Melodic line identification
   - Contour analysis
   - Range analysis
   - Intervallic patterns

6. **Rhythm Analysis:**
   - Rhythm pattern extraction
   - Syncopation measurement
   - Microtiming deviation analysis
   - Swing factor detection

7. **Groove Analysis:**
   - Groove pattern recognition
   - Timing deviation profiling
   - Velocity pattern analysis
   - Feel characterization

8. **Statistical Analysis:**
   - Pitch class distribution
   - Interval distribution
   - Duration distribution
   - Velocity statistics
   - Density analysis (notes per beat)

**Data Structures:**
- NoteEvent (pitch, timing, velocity, channel)
- ChordEvent (root, quality, confidence)
- KeySignature (tonic, mode, confidence)
- TimeSignature (numerator, denominator)

---

## SECTION 6: ADVANCED MODULES

### Location: `/home/arlo/harmonymodule/advanced_modules/`

#### 6.1 Harmony Advanced (`harmony_advanced.py`)
**Advanced Harmony System (10x Enhanced):**

1. **Voice Leading:**
   - Fux Counterpoint (Species 1-5)
   - Voice crossing prevention
   - Spacing rules (SATB, string quartet)
   - Motion type detection (contrary, oblique, similar, parallel)

2. **Functional Harmony:**
   - Cadence types: Authentic, Plagal, Half, Deceptive, Phrygian Half
   - Harmonic functions: Tonic, Subdominant, Dominant, Secondary Dominant
   - Functional analysis and generation

3. **Modal Interchange:**
   - Borrowed chords from parallel modes
   - Phrygian, Dorian, Mixolydian modes
   - Parallel major/minor relationships

4. **Neo-Riemannian Transformations:**
   - PLR operations (Parallel, Leading-tone, Relative)
   - Tonnetz navigation
   - Hexatonic and octatonic cycles
   - Chromatic mediant progressions

5. **Advanced Substitutions:**
   - Tritone substitutions
   - Diminished chord substitutions
   - Augmented sixth chords
   - Upper structure triads

6. **Harmonic Rhythm:**
   - Chord change frequency control
   - Harmonic pace variation
   - Tension curves

7. **Voice Leading Quality:**
   - Perfect (5), Excellent (4), Good (3), Acceptable (2), Poor (1), Unacceptable (0)

#### 6.2 Bass Engine (`bass_engine.py`)
**Advanced Bass Line Generation:**

**Bass Styles:**
- WALKING (Jazz walking bass)
- FUNK (Syncopation, ghost notes, slap)
- REGGAE (One-drop, roots reggae)
- DISCO (Four-on-floor with octaves)
- METAL (Gallop patterns, power)
- BOSSA (Brazilian syncopation)
- POP (Simple root-fifth patterns)
- ROCK (Driving eighth notes)

**Articulation Types:**
- NORMAL, SLAP, POP, GHOST, SLIDE
- HARMONIC, STACCATO, LEGATO, PALM_MUTE

**Bass Registers:**
- LOW (E1 to G2)
- MID (E2 to G3)
- HIGH (E3 to G4)
- FULL (E1 to G4)

**Features:**
- Contour-based algorithms
- Voice leading geometry (Tymoczko)
- Melody contour matching
- Genre-specific patterns
- Harmonic awareness
- Chord tone targeting

**Research Base:** Dias & Guedes (2013), Tymoczko (2011)

#### 6.3 Counterpoint Engine (`counterpoint_engine.py`)
**Species Counterpoint Generation:**

**Species Levels:**
- FIRST (1:1 note-against-note)
- SECOND (2:1 two notes against one)
- THIRD (4:1 four notes against one)
- FOURTH (4:1 with syncopation/suspension)
- FIFTH (5 florid, mixed rhythms)

**Styles:**
- STRICT_FUX (very strict)
- PALESTRINA (16th century polyphonic)
- BACH (more baroque freedom)
- RELAXED (educational/modern)

**Consonance Classification:**
- PERFECT_CONSONANCE (unison, P5, P8)
- IMPERFECT_CONSONANCE (m3, M3, m6, M6)
- DISSONANCE (all others)

**Motion Types:**
- CONTRARY, OBLIQUE, SIMILAR, PARALLEL

**Algorithms:**
- Variable Neighborhood Search (VNS)
- Backtracking search
- Cantus firmus validation
- Multi-voice support (2-4 voices)

**Research Base:** Fux (1725), Herremans & Sörensen (2012-2013)

#### 6.4 Chord Voicing (`chord_voicing.py`)
**Advanced Chord Voicing:**

**Voicing Types:**
- CLOSE (within octave)
- DROP_2, DROP_3, DROP_2_4, DROP_3_5
- SPREAD (wide spacing)
- ROOTLESS (Bill Evans style)
- UPPER_STRUCTURE (triad over shell)
- POLYCHORD (two triads)
- CLUSTER (chromatic/diatonic)

**Qualities:**
- MAJOR, MINOR, DOMINANT, HALF_DIMINISHED
- DIMINISHED, AUGMENTED, SUS2, SUS4
- MAJOR7, MINOR7, DOMINANT7, DIMINISHED7

**Cluster Types:**
- CHROMATIC, DIATONIC, PENTATONIC
- QUARTAL, QUINTAL

**Ensemble Types:**
- SATB, STRING_QUARTET, JAZZ_COMBO
- BIG_BAND, PIANO, GUITAR

**Spacing Rules:**
- SATB standards
- String quartet rules
- Jazz voicing conventions
- Piano idioms

**Research Base:** Tymoczko (2011), Levine (1995), Evans, Bartók, Stravinsky

#### 6.5 Extended Harmony (`extended_harmony.py`)
**20th-21st Century Harmonic Techniques:**

**Upper Structure Triads:**
- MAJ_#11, MAJ_B9, MIN_B9, MAJ_#9
- MIN_5, MAJ_B13, DIM_3, AUG_ROOT

**Polychord Relations:**
- TRITONE (Petrushka chord)
- CHROMATIC_MEDIANT
- SYMMETRIC
- PARALLEL
- RELATIVE
- ARBITRARY

**Cluster Types:**
- CHROMATIC, DIATONIC, PENTATONIC
- WHOLE_TONE, QUARTAL, QUINTAL, SECUNDAL

**Features:**
- Polychord generation
- Cluster voicings
- Slash chords
- Altered dominants
- Multi-tonic systems
- Constant structures (Messiaen)

**Research Base:** Ligeti, Cowell, Bartók, Stravinsky, Messiaen

#### 6.6 Expressive Performance (`expressive_performance.py`)
**Advanced MIDI Expression & Humanization:**

**Dynamic Curves:**
- LINEAR, EXPONENTIAL, EASE_IN, EASE_OUT
- EASE_IN_OUT, LOGARITHMIC

**Articulations:**
- LEGATO, STACCATO, STACCATISSIMO
- TENUTO, MARCATO, PORTATO
- ACCENT, GHOST_NOTE, FERMATA

**Features:**
- Dynamic curve generation
- Velocity humanization (Gaussian variation)
- Microtiming and swing
- Roger Linn algorithm for swing
- Rubato and tempo curves
- Style-specific expression profiles
- GigaMIDI dataset-based patterns
- MAESTRO dataset alignment

**Research Base:** GigaMIDI, MAESTRO, PMC research, Kilchenmann & Senn

#### 6.7 Groove Quantization (`groove_quantization.py`)
**Groove Template System:**
- Extract and apply timing patterns
- Microtiming deviations
- Velocity profiles
- Genre-specific quantization

#### 6.8 Harmonic Rhythm (`harmonic_rhythm.py`)
**Harmonic Rhythm Control:**
- Chord change frequency
- Tension/release curves
- Genre-specific patterns
- Rhythmic analysis

#### 6.9 Film Scoring Engine (`film_scoring_engine.py`)
**State-of-the-Art Film Scoring:**

1. **Video Analysis:**
   - Scene change detection (PySceneDetect)
   - Visual intensity analysis (motion, cuts, color)
   - Color mood mapping
   - Dialogue detection

2. **Film Scoring Techniques:**
   - Leitmotif system (character/location themes)
   - Tension arc mapping
   - Mickey-Mousing (action sync)
   - Chromatic harmony (half-step modulations)
   - Ostinato and pedal point
   - Progression morphing

3. **Synchronization:**
   - SMPTE timecode (HH:MM:SS:FF)
   - Hit point marking
   - Elastic tempo mapping
   - Frame-accurate generation

**Tension Levels:**
- VERY_LOW through VERY_HIGH

**Research Base:** Zimmer, Williams, PySceneDetect

#### 6.10 Tempo Engine (`tempo_engine.py`)
**Tempo Control & Conversion:**
- Tempo mapping for scenes
- Genre-appropriate ranges
- Double-time/half-time feels
- Gradual tempo changes

#### 6.11 Microtonality (`microtonality.py`)
**Microtonal Systems & World Music:**

**Equal Temperaments:**
- 24-TET (quarter tones)
- 19-TET (meantone approximation)
- 31-TET (meantone)
- 53-TET (Pythagorean comma)
- Custom N-TET

**Just Intonation:**
- Pure frequency ratios (3:2, 5:4, etc.)
- Natural acoustic relationships

**Non-Western Systems:**
- Arabic maqam (24 maqamat)
- Indian raga (72 melakarta)
- Turkish makam (53-TET)
- Persian dastgah

**MIDI Implementation:**
- Pitch bend for microtones
- Cents-based calculation
- Proper pitch bend range setting

**Research Base:** Partch, Johnston, Touma, Jairazbhoy

#### 6.12 MIDI CC Automation (`midi_cc_automation.py`)
**Control Change Automation:**
- Parameter curves
- CC value assignment
- Expression pedal mapping
- Volume dynamics
- Modulation depth
- Filter cutoff automation

#### 6.13 Orchestration Advanced (`orchestration_advanced.py`)
**Professional Orchestration:**
- Instrument range validation
- Register optimization
- Doubling rules
- Texture-based assignment
- Dynamic balance
- Style-specific orchestration

---

## SECTION 7: CORE THEORY MODULES

### Location: `/midi_generator/core/`

#### 7.1 Neo-Riemannian Theory (`neo_riemannian.py`)
**Neo-Riemannian Transformations:**

**Triad Qualities:**
- MAJOR, MINOR, AUGMENTED, DIMINISHED

**Primary Transformations (PLR):**
- P (Parallel): C maj → C min
- L (Leading-tone): C maj → E min
- R (Relative): C maj → A min

**Advanced Features:**
- Tonnetz navigation
- Hexatonic cycles
- Octatonic cycles
- Voice leading optimization
- Chromatic progressions

**Applications:**
- Film scoring (Williams, Zimmer, Desplat)
- Late-Romantic harmony
- Maximally smooth voice leading
- Contemporary composition

#### 7.2 Modal Harmony (`modal_harmony.py`)
**Church Modes & Modal Systems:**

**Diatonic Modes:**
- IONIAN (major)
- DORIAN (minor with raised 6)
- PHRYGIAN (minor with lowered 2)
- LYDIAN (major with raised 4)
- MIXOLYDIAN (major with lowered 7)
- AEOLIAN (natural minor)
- LOCRIAN (darkest, diminished)

**Harmonic Minor Modes:**
- HARMONIC_MINOR
- LOCRIAN_NAT6, IONIAN_SHARP5, DORIAN_SHARP4
- PHRYGIAN_DOMINANT, LYDIAN_SHARP2, SUPER_LOCRIAN_BB7

**Melodic Minor Modes:**
- MELODIC_MINOR
- DORIAN_FLAT2, LYDIAN_AUGMENTED, LYDIAN_DOMINANT
- MIXOLYDIAN_FLAT6, LOCRIAN_NAT2, ALTERED

**Symmetrical Scales:**
- WHOLE_TONE, DIMINISHED_HALF_WHOLE, DIMINISHED_WHOLE_HALF
- AUGMENTED, PROMETHEUS, TRITONE

**Features:**
- Modal interchange
- Color and brightness variation
- Horizontal melodic motion
- Modal composition

**Research Base:** George Russell, Jerry Coker, Messiaen

#### 7.3 Instrument Library (`instrument_library.py`)
**Comprehensive Instrument Database:**
- Range (low/high, comfortable/full)
- MIDI program assignment
- Tessitura optimization
- Transposition information
- Orchestral families
- Solo/ensemble preferences
- Articulation capabilities
- Special techniques

---

## SECTION 8: LEARNING & PATTERN MODULES

### Location: `/midi_generator/learning/`

#### 8.1 Corpus Learner (`corpus_learner.py`)
**Corpus-Based Style Learning:**

**Style Features:**
- PITCH_DISTRIBUTION
- INTERVAL_DISTRIBUTION
- RHYTHM_DISTRIBUTION
- CHORD_DISTRIBUTION
- MELODIC_RANGE
- HARMONIC_COMPLEXITY
- RHYTHMIC_COMPLEXITY
- TEMPO
- KEY_PREFERENCE
- CADENCE_PATTERNS

**Learning Methods:**
- Statistical modeling
- Markov chains
- N-gram models
- Composer/genre classification
- Style interpolation and hybridization
- Machine learning integration

**Research Base:** David Cope (EMI), Conklin (2003), Saunders et al. (2004)

#### 8.2 Pattern Extractor (`pattern_extractor.py`)
**Music Pattern Extraction:**
- Rhythmic patterns
- Melodic contours
- Harmonic progressions
- Leitmotif identification
- Phrase structure
- Structural analysis

#### 8.3 Motif Library (`motif_library.py`)
**Motif Management:**
- Motif extraction
- Variation generation
- Transformation tracking
- Reuse and development

---

## SECTION 9: MUSIC THEORY COVERAGE SUMMARY

### Harmonic Systems
1. **Functional Harmony** (Classical/Romantic)
   - Tonic, Subdominant, Dominant functions
   - Cadences (authentic, plagal, deceptive)
   - Voice leading
   - Spacing rules

2. **Modal Harmony** (Jazz, Folk, Contemporary)
   - Seven church modes
   - Harmonic and melodic minor modes
   - Symmetrical scales
   - Modal interchange

3. **Neo-Riemannian Theory** (Late Romantic, Film Music)
   - PLR transformations
   - Tonnetz navigation
   - Chromatic relationships
   - Smooth voice leading

4. **Extended Harmony** (20th-21st Century)
   - Polychords
   - Cluster voicings
   - Upper structure triads
   - Multi-tonic systems

5. **Microtonal Systems** (World Music, Contemporary)
   - Equal temperaments (24, 19, 31, 53-TET)
   - Just intonation
   - Arabic maqam
   - Indian raga
   - Turkish makam
   - Persian dastgah

### Voice Leading & Counterpoint
1. **Fux Counterpoint**
   - Species 1-5
   - Cantus firmus
   - Consonance classification
   - Parallel/direct motion rules

2. **Jazz Voicing**
   - Drop voicings (2, 3, 2&4, 3&5)
   - Rootless voicings (Bill Evans)
   - Upper structures
   - Polychords

3. **Classical Spacing**
   - SATB rules
   - String quartet techniques
   - Doubling rules
   - Tessitura optimization

4. **Tymoczko Voice Leading**
   - OPTIC spaces
   - Geometric approaches
   - Minimally smooth voice leading
   - Orbifolds

### Melodic Techniques
1. **Motivic Development**
   - 15 development techniques
   - Thematic transformation
   - Phrase structure
   - Contour analysis

2. **Ornamentation**
   - Baroque, Classical, Romantic styles
   - Turn, trill, grace note patterns
   - Embellishment density

3. **Melody Extraction & Analysis**
   - Contour analysis
   - Range analysis
   - Intervallic patterns
   - Melodic complexity metrics

### Rhythm & Groove
1. **Groove Systems**
   - Famous grooves (Purdie shuffle, Motown)
   - Genre-specific microtiming
   - Swing feels (Roger Linn algorithm)
   - Humanization patterns

2. **Polyrhythm**
   - Complex cross-rhythms (3 vs 4, 5 vs 4, etc.)
   - Metric modulation
   - Elliott Carter techniques

3. **Rhythm Transformation**
   - Augmentation/diminution
   - Retrograde
   - Syncopation control
   - Quantization strategies

4. **Meter & Time Signature**
   - Simple, compound, complex meters
   - Metric modulation
   - Meter conversion strategies
   - Odd meter support (5/4, 7/8, 11/8)

### Form & Structure
1. **Classical Forms**
   - Sonata, rondo, theme & variations
   - Binary, ternary
   - Fugue

2. **Popular Forms**
   - Verse-chorus-bridge
   - AABA
   - 12-bar blues
   - Through-composed

3. **Modulation & Transitions**
   - Common chord modulation
   - Sequential modulation
   - Enharmonic reinterpretation
   - Chromatic mediant transitions

### Performance & Expression
1. **Articulation**
   - Legato, staccato, marcato, tenuto
   - Dynamics curves
   - Velocity humanization
   - Microtiming

2. **Expressive Techniques**
   - Rubato and tempo curves
   - Dynamic crescendo/decrescendo
   - Swing and groove feel
   - Style-specific expression profiles

3. **Humanization**
   - Gaussian velocity variation
   - Natural timing deviations
   - Groove-based timing
   - Style-appropriate expression

---

## SECTION 10: GENRE COVERAGE

The library supports 35+ genres through modular components and specialized generators:

### Major Genre Categories
1. **Jazz** - Bebop, Fusion, Modal Jazz, Big Band
2. **Blues** - Classic, Modern, Electric
3. **Funk/Soul** - Funk, Soul, Disco, R&B, Neo-Soul
4. **Rock** - Classic, Progressive, Metal, Alternative
5. **Pop** - Pop, Electronic, Dance, House, Techno, Dubstep, Drum & Bass
6. **Classical** - Baroque, Classical, Romantic, Modern
7. **Latin** - Salsa, Bossa Nova, Samba, Mambo, Tango, Reggae
8. **World Music** - African, Indian, Arabic, Turkish, Persian
9. **Gospel** - Gospel, Spiritual
10. **Film/Orchestral** - Film scoring, orchestral, ballet

---

## SECTION 11: EXPORT & OUTPUT CAPABILITIES

### MIDI Export
- Standard MIDI file generation (Format 0 and 1)
- CC automation (volume, modulation, effects)
- Sustain pedal, expression pedal
- Pitch bend information
- Articulation data

### Timecode & Synchronization
- SMPTE timecode (HH:MM:SS:FF)
- Frame-accurate timing
- Hit point marking
- Tempo mapping for picture

### Analysis & Annotation
- Key signature metadata
- Time signature information
- Chord progression data
- Structural annotations
- Performance metrics

---

## SECTION 12: RESEARCH & THEORETICAL FOUNDATIONS

### Music Theory References
- Fux: "Gradus ad Parnassum" (Counterpoint)
- Piston: "Harmony"
- Persichetti: "Twentieth-Century Harmony"
- Schoenberg: "Fundamentals of Musical Composition"
- Tymoczko: "A Geometry of Music"
- Cohn: "Audacious Euphony"
- Levine: "The Jazz Theory Book"

### MIR (Music Information Retrieval)
- Krumhansl & Kessler: Key detection algorithm
- Temperley: "Music and Probability"
- Müllensiefen: Melodic analysis
- Saunders et al.: Style classification

### Performance Research
- Bengtsson & Gabrielsson: "Timing Patterns in Music"
- Repp & Su: "Timing in Music Performance"
- Kilchenmann & Senn: "Microtiming Deviations"
- Roger Linn: Swing algorithm (MPC)

### Datasets
- GigaMIDI (1.4M files, microtiming patterns)
- MAESTRO (200 hours piano, 3ms alignment)
- Music21 Corpus

---

## SECTION 13: SUMMARY STATISTICS

### Component Types: 10
### Form Types: 14
### Development Techniques: 15
### Orchestration Styles: 8
### Accompaniment Patterns: 11
### Texture Types: 4
### Bass Styles: 8
### Modulation Types: 6
### Transition Types: 10
### L-System Symbol Types: 10+
### CA Rules: 256+ (elementary) + Conway's Life
### Constraint Types: Multiple (hard/soft)
### Tempo Feel Types: 7
### Groove Intensity Levels: 5
### Chord Voicing Types: 9
### Articulation Types: 10+
### Analysis Capabilities: 8 major categories
### Modal Systems: 25+ scales/modes
### Temperament Options: 10+ equal temperaments
### Supported Genres: 35+

---

## END OF INVENTORY

This comprehensive inventory documents the complete modular architecture of the harmonymodule library, demonstrating professional-grade capabilities for music composition, transformation, analysis, and generation across multiple genres and compositional styles.
