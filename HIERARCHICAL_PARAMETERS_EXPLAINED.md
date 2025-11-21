# 🎼 The 50 Hierarchical Parameters: Complete Explanation

**System**: Musical Program Synthesis - Hierarchical Parameter Architecture v2.0
**Date**: November 20, 2025
**Reduction**: 118 parameters → 50 parameters (57.6% reduction)

---

## 📊 OVERVIEW: Why 50 Parameters?

Your system uses **50 hierarchical parameters** organized in **3 levels** to represent any musical style with precision while maintaining computational efficiency.

### Design Philosophy

**The Pyramid Structure**:
```
Level 1: Global Context (8 params)          ← Sets the stage
         ↓ conditions ↓
Level 2: Universal Dimensions (20 params)   ← Core musical elements
         ↓ conditions ↓
Level 3: Genre-Specific (22 params)         ← Style-specific nuances
```

**Key Insight**: Music has a natural hierarchy. You don't need to specify every detail independently - high-level decisions (genre, tempo, key) constrain and inform lower-level details (jazz swing feel, classical counterpoint).

---

## 🎯 LEVEL 1: GLOBAL CONTEXT (8 Parameters)

**Purpose**: Establish the fundamental musical context that conditions everything else

**These parameters answer**: "What kind of piece is this, in broad strokes?"

---

### 1. `genre.primary` (categorical)
**Type**: Categorical
**Options**: jazz, classical, rock, electronic, pop, latin, hiphop
**Default**: jazz

**What it does**:
- **Most important parameter** - determines which Level 3 genre-specific parameters are active
- Gates entire sections of the parameter space
- Example: If `genre.primary = 'jazz'`, then `jazz.swing_feel`, `jazz.walking_bass`, `jazz.improvisation_ratio`, and `jazz.bebop_vocabulary` become active

**Musical Function**: Sets the stylistic universe for generation

**Examples**:
- `"jazz"` → Activates swing feel, walking bass, improvisation
- `"classical"` → Activates counterpoint, voice leading, development
- `"rock"` → Activates power chords, riffs, distortion

**Why it's Level 1**: Everything else depends on genre - harmony rules, rhythm patterns, instrumentation, etc.

---

### 2. `tempo.bpm` (continuous, 40-200)
**Type**: Continuous
**Range**: 40 BPM (slow ballad) to 200 BPM (fast bebop)
**Default**: 120 BPM

**What it does**:
- Sets the speed of the piece
- Affects perceived energy and mood
- Influences rhythm subdivision choices

**Musical Function**: Controls temporal flow

**Examples**:
- 60 BPM → Slow ballad, contemplative
- 120 BPM → Medium swing, comfortable
- 180 BPM → Fast bebop, energetic

**Formula**: Directly extracted from MIDI tempo meta-events or beat tracking

**Why it's Level 1**: Tempo affects rhythm, melody density, and overall energy

---

### 3. `time_signature` (categorical)
**Type**: Categorical
**Options**: 4/4, 3/4, 6/8, 5/4, 7/8, 12/8, 2/4
**Default**: 4/4

**What it does**:
- Defines the metric organization (how many beats per measure)
- Establishes rhythmic framework
- Affects groove patterns and phrasing

**Musical Function**: Sets the rhythmic container

**Examples**:
- `4/4` → Common time (most popular music)
- `3/4` → Waltz feel (one-two-three)
- `7/8` → Complex meter (progressive rock, folk)

**Why it's Level 1**: All rhythmic decisions happen within this metric framework

---

### 4. `key.tonic` (categorical)
**Type**: Categorical
**Options**: C, C#, D, D#, E, F, F#, G, G#, A, A#, B (12 keys)
**Default**: C

**What it does**:
- Establishes the tonal center (root note)
- Determines available pitches and chords
- Affects harmonic relationships

**Musical Function**: Sets the pitch reference point

**Examples**:
- `C` → Key of C (no sharps/flats)
- `Bb` → Key of Bb (common in jazz - horns)
- `F#` → Key of F# (6 sharps)

**Extraction**: Krumhansl-Schmuckler key detection algorithm (pitch class distribution)

**Why it's Level 1**: All harmony and melody use this tonal center as reference

---

### 5. `key.mode` (categorical)
**Type**: Categorical
**Options**: major, minor, dorian, phrygian, lydian, mixolydian, aeolian, locrian
**Default**: major

**What it does**:
- Defines the scale/mode within the key
- Determines harmonic color (bright vs dark)
- Affects available chord progressions

**Musical Function**: Establishes tonal quality

**Examples**:
- `major` → Bright, happy (Ionian mode)
- `minor` → Dark, sad (Aeolian mode)
- `dorian` → Modal jazz (minor with raised 6th)
- `mixolydian` → Blues, rock (major with lowered 7th)

**Why it's Level 1**: Mode fundamentally changes the harmonic palette

---

### 6. `energy.level` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (calm) to 1.0 (intense)
**Default**: 0.5

**What it does**:
- High-level measure of overall intensity
- Aggregates dynamics, tempo, and texture density
- Provides a single "energy" knob

**Musical Function**: Controls overall intensity

**Formula**:
```
energy.level = 0.3 × dynamics.overall_level
             + 0.3 × min(tempo/200, 1.0)
             + 0.4 × texture.density
```

**Examples**:
- 0.2 → Ballad (soft, slow, sparse)
- 0.5 → Medium (moderate intensity)
- 0.9 → Intense rock (loud, fast, dense)

**Why it's Level 1**: Sets the overall vibe that influences all Level 2 parameters

---

### 7. `complexity.overall` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (simple) to 1.0 (complex)
**Default**: 0.5

**What it does**:
- High-level measure of musical complexity
- Aggregates harmonic, melodic, and rhythmic complexity
- Provides a single "sophistication" knob

**Musical Function**: Controls overall sophistication

**Formula**:
```
complexity.overall = 0.5 × harmony.complexity
                   + 0.3 × melody.rhythmic_complexity
                   + 0.2 × rhythm.syncopation
```

**Examples**:
- 0.2 → Simple pop (triads, simple rhythms)
- 0.6 → Standard jazz (7th chords, moderate syncopation)
- 0.95 → Free jazz (extended harmony, complex rhythms)

**Why it's Level 1**: Guides the sophistication level of all musical elements

---

### 8. `structure.form` (categorical)
**Type**: Categorical
**Options**: AABA, ABAC, verse_chorus, verse_chorus_bridge, through_composed, theme_variations, sonata, rondo
**Default**: AABA

**What it does**:
- Defines the large-scale structure
- Determines how sections are organized
- Affects repetition and contrast patterns

**Musical Function**: Establishes formal architecture

**Examples**:
- `AABA` → Jazz standard (32 bars: 8+8+8+8)
- `verse_chorus_bridge` → Pop song structure
- `sonata` → Classical sonata form

**Why it's Level 1**: Form determines how all other parameters evolve over time

---

## 🎵 LEVEL 2: UNIVERSAL DIMENSIONS (20 Parameters)

**Purpose**: Core musical elements that apply across all genres
**Conditioning**: These parameters are influenced by Level 1 (especially genre, key, tempo)

**These parameters answer**: "What are the universal musical characteristics?"

### Subcategories:
- **Harmony** (6 params): Chord structure and progression
- **Melody** (5 params): Melodic line characteristics
- **Rhythm** (5 params): Temporal patterns
- **Dynamics** (2 params): Loudness and expression
- **Texture** (2 params): Vertical and horizontal density

---

## 🎹 HARMONY (6 parameters)

### 9. `harmony.chord_density` (continuous, 1.0-12.0)
**Type**: Continuous
**Range**: 1.0 (sparse) to 12.0 (very dense)
**Default**: 4.0

**What it does**: Measures harmonic activity
**Formula**: `chords_per_measure × avg_notes_per_chord`

**Examples**:
- 2.0 → Sparse (2 triads per measure = 6 notes total)
- 4.0 → Medium (4 triads = 12 notes, or 2 seventh chords = 8 notes)
- 8.0 → Dense (8 chords or thick voicings)

**Musical Impact**: Higher density = richer harmony, busier accompaniment

---

### 10. `harmony.complexity` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (simple triads) to 1.0 (extended chords)
**Default**: 0.5

**What it does**: Measures chord sophistication
**Formula**: `0.3 × use_9ths + 0.3 × use_11ths + 0.4 × use_13ths`

**Examples**:
- 0.2 → Simple (mostly triads: C, F, G)
- 0.7 → Jazz standard (mostly 7ths and 9ths: Cmaj7, F9, G7)
- 0.9 → Modern jazz (extended: C13#11, F7alt)

**Why separate from density**: Complexity is about **chord quality**, density is about **chord quantity**

---

### 11. `harmony.chromaticism` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (diatonic) to 1.0 (highly chromatic)
**Default**: 0.3

**What it does**: Measures departure from the key
**Formula**: `(tritone_substitutions + modal_interchange) / total_chords`

**Examples**:
- 0.1 → Diatonic (C major: C, Dm, Em, F, G, Am)
- 0.4 → Moderate (some secondary dominants: C, A7→Dm, G7)
- 0.8 → Bebop (tritone subs, altered chords)

**Musical Impact**: Higher chromaticism = more tension, jazz/modern sound

---

### 12. `harmony.tension` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (consonant) to 1.0 (dissonant)
**Default**: 0.5

**What it does**: Measures harmonic dissonance
**Formula**: `avg_dissonance_score` (based on interval content)

**Examples**:
- 0.2 → Consonant (major/minor triads, stable)
- 0.5 → Moderate (7th chords, some tension)
- 0.95 → Dissonant (clusters, atonal harmony)

**Why separate from complexity**: You can have complex but consonant chords (Cmaj9) or simple but tense chords (C+E+F tritone)

---

### 13. `harmony.voicing_spread` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (close voicing) to 1.0 (wide voicing)
**Default**: 0.5

**What it does**: Measures vertical spacing of chord notes
**Formula**: `avg_chord_range_semitones / 36.0` (normalized to 3 octaves)

**Examples**:
- 0.2 → Close voicing (C-E-G within 1 octave)
- 0.5 → Medium spread (C-G-E spanning 1.5 octaves)
- 0.9 → Wide voicing (C2-G3-E5 spanning 3 octaves)

**Musical Impact**: Wider = fuller, more orchestral; closer = tighter, more compact

---

### 14. `harmony.progression_predictability` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (unpredictable) to 1.0 (very predictable)
**Default**: 0.5

**What it does**: Measures harmonic expectancy
**Formula**: `1.0 - normalized_entropy(chord_sequence)`

**Examples**:
- 0.8 → Predictable (I-IV-V-I loops, standard progressions)
- 0.5 → Moderate (some surprises, modal interchange)
- 0.1 → Avant-garde (random progressions, no patterns)

**Musical Impact**: Predictability affects listener comfort vs surprise

---

## 🎤 MELODY (5 parameters)

### 15. `melody.note_density` (continuous, 1.0-16.0)
**Type**: Continuous
**Range**: 1.0 (sparse) to 16.0 (very dense)
**Default**: 4.0

**What it does**: Melodic activity level
**Formula**: `total_melody_notes / total_measures`

**Examples**:
- 2.0 → Sparse (whole notes, half notes)
- 4.0 → Moderate (quarter notes, some eighths)
- 12.0 → Bebop (sixteenth note runs)

---

### 16. `melody.range_semitones` (integer, 3-36)
**Type**: Integer
**Range**: 3 semitones (narrow) to 36 semitones (3 octaves)
**Default**: 12

**What it does**: Melodic pitch span
**Formula**: `highest_note - lowest_note`

**Examples**:
- 8 → Narrow (minor 6th)
- 12 → Octave (common)
- 24 → 2 octaves (wide)

---

### 17. `melody.contour_smoothness` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (jagged/disjunct) to 1.0 (smooth/conjunct)
**Default**: 0.7

**What it does**: Measures stepwise vs leap motion
**Formula**: `stepwise_motion_ratio × (1 - avg_leap_size/12)`

**Examples**:
- 0.3 → Jagged (many large leaps)
- 0.7 → Moderate (mostly steps, some leaps)
- 0.95 → Legato (almost all stepwise motion)

---

### 18. `melody.rhythmic_complexity` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (simple rhythm) to 1.0 (complex rhythm)
**Default**: 0.5

**What it does**: Measures rhythmic variety
**Formula**: `normalized_entropy(note_durations)`

**Examples**:
- 0.2 → Simple (mostly quarter notes)
- 0.5 → Moderate (mix of durations)
- 0.9 → Complex (many different durations, tuplets)

---

### 19. `melody.repetition` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (through-composed) to 1.0 (highly repetitive)
**Default**: 0.5

**What it does**: Measures melodic recurrence
**Formula**: `repeated_motif_ratio`

**Examples**:
- 0.1 → Through-composed (constant new material)
- 0.5 → Moderate (some motifs return)
- 0.95 → Ostinato (constant repetition)

---

## 🥁 RHYTHM (5 parameters)

### 20. `rhythm.subdivision` (categorical)
**Type**: Categorical
**Options**: whole, half, quarter, eighth, triplet, sixteenth, quintuplet, sextuplet
**Default**: eighth

**What it does**: Primary note duration level
**Formula**: Detect smallest common duration

**Examples**:
- `quarter` → Slow ballad (♩ as base unit)
- `eighth` → Medium swing (♪ as base unit)
- `sixteenth` → Fast bebop (𝅘𝅥𝅯 as base unit)

---

### 21. `rhythm.syncopation` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (on-beat) to 1.0 (highly syncopated)
**Default**: 0.3

**What it does**: Measures off-beat emphasis
**Formula**: `notes_on_off_beats / total_notes`

**Examples**:
- 0.1 → Classical (mostly on-beat)
- 0.4 → Jazz (moderate syncopation)
- 0.8 → Funk (heavy syncopation)

---

### 22. `rhythm.groove_consistency` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (loose/rubato) to 1.0 (tight/quantized)
**Default**: 0.7

**What it does**: Measures timing precision
**Formula**: `1.0 - normalized_std(note_timings)`

**Examples**:
- 0.3 → Loose/human feel (expressive timing)
- 0.7 → Moderate (slight humanization)
- 0.99 → Perfectly quantized (electronic)

---

### 23. `rhythm.polyrhythm` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (single rhythm) to 1.0 (complex polyrhythms)
**Default**: 0.1

**What it does**: Measures rhythmic layering
**Formula**: `conflicting_rhythms / total_voices`

**Examples**:
- 0.0 → Homophonic (everyone plays same rhythm)
- 0.3 → Moderate (some independence)
- 0.8 → Polyrhythmic (3 against 4, complex layers)

---

### 24. `rhythm.swing_amount` (continuous, 0.5-0.75)
**Type**: Continuous
**Range**: 0.5 (straight) to 0.75 (hard swing)
**Default**: 0.67

**What it does**: Controls swing feel
**Formula**: `duration_ratio` of paired eighth notes

**Examples**:
- 0.5 → Straight eighths (1:1 ratio) ♪♪♪♪
- 0.67 → Standard swing (2:1 ratio) ♪. ♬ ♪. ♬
- 0.75 → Hard swing (3:1 ratio)

**Why 0.67?**: This creates the triplet feel (♩♩♩ = ♪.♬♪.♬)

---

## 🔊 DYNAMICS (2 parameters)

### 25. `dynamics.overall_level` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (pp - pianissimo) to 1.0 (ff - fortissimo)
**Default**: 0.6

**What it does**: Average loudness
**Formula**: `avg_velocity / 127`

**Examples**:
- 0.2 → pp (very soft)
- 0.6 → mf (mezzo-forte)
- 0.9 → ff (very loud)

---

### 26. `dynamics.range` (continuous, 0.0-1.0)
**Type**: Continuous
**Range**: 0.0 (flat dynamics) to 1.0 (wide dynamics)
**Default**: 0.3

**What it does**: Dynamic variation
**Formula**: `velocity_std / 127`

**Examples**:
- 0.1 → Flat (little variation)
- 0.3 → Moderate (some expression)
- 0.7 → Expressive (wide dynamic range)

**Why separate from level?**: Level is **average** loudness, range is **variation**

---

## 🎻 TEXTURE (2 parameters)

### 27. `texture.polyphony` (integer, 1-12)
**Type**: Integer
**Range**: 1 (monophonic) to 12 (highly polyphonic)
**Default**: 4

**What it does**: Maximum simultaneous notes
**Formula**: `max(notes_at_time_t for all t)`

**Examples**:
- 1 → Monophonic (single melody line)
- 4 → Quartet (4-part harmony)
- 10 → Orchestral (dense texture)

---

### 28. `texture.density` (continuous, 0.5-20.0)
**Type**: Continuous
**Range**: 0.5 (very sparse) to 20.0 (very dense)
**Default**: 5.0

**What it does**: Temporal note density across all tracks
**Formula**: `total_notes / duration_seconds`

**Examples**:
- 2.0 → Sparse (ballad)
- 5.0 → Moderate (medium tempo jazz)
- 15.0 → Dense (fast bebop ensemble)

**Why separate from polyphony?**: Polyphony is **vertical** (simultaneous), density is **horizontal** (over time)

---

## 🎺 LEVEL 3: GENRE-SPECIFIC DETAILS (22 Parameters)

**Purpose**: Style-specific nuances activated by `genre.primary`
**Conditioning**: Only relevant parameters are active based on genre

**These parameters answer**: "What are the genre-specific characteristics?"

---

## 🌍 UNIVERSAL ORCHESTRATION (5 parameters - always active)

### 29. `orchestration.instrument_count` (integer, 1-20)
**Range**: 1 (solo) to 20 (orchestra)
**Default**: 5

**What it does**: Number of distinct instruments
**Formula**: `count(distinct_midi_programs)`

**Examples**:
- 1 → Solo piano
- 5 → Jazz combo (sax, trumpet, piano, bass, drums)
- 15 → Big band orchestra

---

### 30. `orchestration.register_balance` (continuous, 0.0-1.0)
**Range**: 0.0 (bass-heavy) to 1.0 (treble-heavy)
**Default**: 0.5

**What it does**: High vs low note distribution
**Formula**: `notes_above_C4 / notes_below_C4` (normalized)

**Examples**:
- 0.2 → Bass-heavy (rock, dubstep)
- 0.5 → Balanced (most music)
- 0.8 → Treble-heavy (piccolo ensemble)

---

### 31. `articulation.legato_ratio` (continuous, 0.3-1.0)
**Range**: 0.3 (staccato) to 1.0 (legato)
**Default**: 0.9

**What it does**: Note connection
**Formula**: `note_duration / inter_onset_interval`

**Examples**:
- 0.5 → Staccato (short, detached)
- 0.75 → Portato (semi-connected)
- 0.95 → Legato (smooth, connected)

---

### 32. `structure.section_contrast` (continuous, 0.0-1.0)
**Range**: 0.0 (similar sections) to 1.0 (contrasting sections)
**Default**: 0.5

**What it does**: Variation between formal sections
**Formula**: `avg_distance(section_features)`

**Examples**:
- 0.2 → Unified (sections sound similar)
- 0.5 → Moderate contrast
- 0.9 → High contrast (verse vs chorus very different)

---

### 33. `structure.repetition_level` (continuous, 0.0-1.0)
**Range**: 0.0 (through-composed) to 1.0 (highly repetitive)
**Default**: 0.5

**What it does**: Material repetition
**Formula**: `repeated_measures / total_measures`

**Examples**:
- 0.1 → Through-composed (constant new material)
- 0.5 → Moderate (some repetition)
- 0.9 → Loop-based (heavy repetition)

---

## 🎷 JAZZ-SPECIFIC (4 parameters - active when genre='jazz')

### 34. `jazz.swing_feel` (categorical)
**Options**: straight, light, medium, hard
**Default**: medium

**What it does**: Categorizes swing intensity
**Mapping**:
- straight: rhythm.swing_amount = 0.5
- light: 0.57
- medium: 0.67
- hard: 0.75

**Examples**:
- `straight` → Latin jazz (even eighths)
- `medium` → Standard swing (bebop)
- `hard` → Big band (hard-driving)

---

### 35. `jazz.walking_bass` (continuous, 0.0-1.0)
**Range**: 0.0 (no walking) to 1.0 (full walking bass)
**Default**: 0.8

**What it does**: Walking bass line presence
**Formula**: `quarter_note_bass_ratio × chord_tone_ratio`

**Examples**:
- 0.2 → Ballad (sparse, whole notes)
- 0.8 → Medium swing (classic walking)
- 0.95 → Bebop (constant quarter notes)

---

### 36. `jazz.improvisation_ratio` (continuous, 0.0-1.0)
**Range**: 0.0 (all composed) to 1.0 (all improvised)
**Default**: 0.3

**What it does**: Estimates improvised content
**Formula**: `non_repeating_material / total_material`

**Examples**:
- 0.1 → Head arrangement (mostly composed)
- 0.5 → Standard (half solos, half melody)
- 0.95 → Free jazz (almost all improvised)

---

### 37. `jazz.bebop_vocabulary` (continuous, 0.0-1.0)
**Range**: 0.0 (no bebop) to 1.0 (heavy bebop)
**Default**: 0.3

**What it does**: Bebop language detection
**Formula**: `detected_bebop_patterns / total_phrases`

**Patterns detected**: Enclosures, approach notes, chromatic runs, bebop scales

**Examples**:
- 0.1 → Swing era (simple lines)
- 0.8 → Classic bebop (Parker, Gillespie)
- 0.2 → Modal jazz (fewer bebop clichés)

---

## 🎻 CLASSICAL-SPECIFIC (3 parameters - active when genre='classical')

### 38. `classical.counterpoint` (continuous, 0.0-1.0)
**Range**: 0.0 (homophonic) to 1.0 (strict counterpoint)
**Default**: 0.5

**What it does**: Voice independence
**Formula**: `melodic_independence × rhythmic_independence`

**Examples**:
- 0.1 → Homophonic (chordal accompaniment)
- 0.5 → Moderate (some independence)
- 0.95 → Fugue (strict contrapuntal rules)

---

### 39. `classical.development_density` (continuous, 0.0-1.0)
**Range**: 0.0 (simple) to 1.0 (highly developed)
**Default**: 0.5

**What it does**: Thematic development
**Formula**: `variation_techniques / total_measures`

**Techniques**: Inversion, retrograde, augmentation, fragmentation

**Examples**:
- 0.2 → Simple song form
- 0.5 → Moderate development
- 0.9 → Sonata development section (Beethoven)

---

### 40. `classical.voice_leading_quality` (continuous, 0.0-1.0)
**Range**: 0.0 (poor) to 1.0 (excellent)
**Default**: 0.8

**What it does**: Voice leading smoothness
**Formula**: `1.0 - normalized_voice_leading_cost`

**Rules checked**: Parallel fifths/octaves, contrary motion, smooth connections

**Examples**:
- 0.3 → Poor (many violations)
- 0.7 → Good (mostly smooth)
- 0.95 → Excellent (Bach-level)

---

## 🎸 ROCK-SPECIFIC (3 parameters - active when genre='rock')

### 41. `rock.power_chord_ratio` (continuous, 0.0-1.0)
**Range**: 0.0 (no power chords) to 1.0 (all power chords)
**Default**: 0.7

**What it does**: Power chord prevalence
**Formula**: `power_chords / total_chords`

**Power chord**: Root + 5th (no 3rd) - C-G, D-A, etc.

**Examples**:
- 0.2 → Soft rock (full chords)
- 0.7 → Hard rock (mostly power chords)
- 0.95 → Metal (almost all power chords)

---

### 42. `rock.riff_repetition` (continuous, 0.0-1.0)
**Range**: 0.0 (no riffs) to 1.0 (constant riffs)
**Default**: 0.7

**What it does**: Riff-based structure
**Formula**: `repeated_riff_measures / total_measures`

**Examples**:
- 0.3 → Progressive rock (through-composed)
- 0.7 → Classic rock (iconic riffs)
- 0.9 → Punk (2-bar riff loops)

---

### 43. `rock.distortion_level` (continuous, 0.0-1.0)
**Range**: 0.0 (clean) to 1.0 (heavy distortion)
**Default**: 0.5

**What it does**: Timbral aggression (MIDI proxy)
**Formula**: `velocity_intensity × accent_density`

**Examples**:
- 0.2 → Clean guitar (soft rock)
- 0.5 → Moderate crunch (classic rock)
- 0.95 → Heavy distortion (metal)

**Note**: In MIDI, we approximate with velocity and articulation

---

## 🎹 ELECTRONIC-SPECIFIC (3 parameters - active when genre='electronic')

### 44. `electronic.quantization` (continuous, 0.0-1.0)
**Range**: 0.0 (loose timing) to 1.0 (perfectly quantized)
**Default**: 0.9

**What it does**: Grid alignment precision
**Formula**: `notes_on_grid / total_notes`

**Examples**:
- 0.4 → Live performance feel
- 0.7 → Moderate quantization
- 1.0 → DAW-perfect timing (techno)

---

### 45. `electronic.filter_movement` (continuous, 0.0-1.0)
**Range**: 0.0 (static) to 1.0 (dynamic filtering)
**Default**: 0.5

**What it does**: Spectral modulation (MIDI proxy)
**Formula**: `timbre_change_rate` (via CC/velocity changes)

**Examples**:
- 0.2 → Static pads (no filter sweep)
- 0.5 → Moderate (some modulation)
- 0.9 → Dynamic (constant filter movement)

**Note**: MIDI approximation using CC modulation, velocity envelopes

---

### 46. `electronic.arpeggio_density` (continuous, 0.0-1.0)
**Range**: 0.0 (no arpeggios) to 1.0 (constant arpeggios)
**Default**: 0.3

**What it does**: Arpeggio pattern prevalence
**Formula**: `arpeggio_notes / total_notes`

**Arpeggio**: Notes of a chord played sequentially (C-E-G-C)

**Examples**:
- 0.1 → Pad-focused (block chords)
- 0.4 → Moderate (some arps)
- 0.9 → Arp-heavy (trance, synthwave)

---

## 🎤 HIP-HOP-SPECIFIC (2 parameters - active when genre='hiphop')

### 47. `hiphop.sample_based` (continuous, 0.0-1.0)
**Range**: 0.0 (composed) to 1.0 (loop-based)
**Default**: 0.7

**What it does**: Loop-based structure
**Formula**: `looped_material / total_material`

**Examples**:
- 0.2 → Live instruments (The Roots)
- 0.8 → Sample-based (classic hip-hop)
- 0.95 → Heavy loops (boom-bap)

---

### 48. `hiphop.boom_bap_feel` (continuous, 0.0-1.0)
**Range**: 0.0 (modern/trap) to 1.0 (classic boom-bap)
**Default**: 0.6

**What it does**: Drum pattern style
**Formula**: `boom_bap_pattern_score`

**Boom-bap**: Kick (boom) + snare (bap) = 💥👏 pattern

**Examples**:
- 0.1 → Trap (hi-hat rolls, 808s)
- 0.5 → Modern hip-hop (hybrid)
- 0.9 → Classic boom-bap (90s golden era)

---

## 🎺 LATIN-SPECIFIC (2 parameters - active when genre='latin')

### 49. `latin.clave_pattern` (categorical)
**Options**: none, son_clave_2-3, son_clave_3-2, rumba_clave_2-3, rumba_clave_3-2, bossa_clave
**Default**: son_clave_2-3

**What it does**: Identifies rhythmic foundation
**Clave**: The fundamental rhythm pattern that organizes all other parts

**Patterns**:
- **Son clave 2-3**: ×・・×・・×　・×・・×・・
- **Son clave 3-2**: ・×・・×・・　×・・×・・×
- **Rumba clave**: Similar but displaced by 1 pulse
- **Bossa clave**: Bossa nova pattern (Brazilian)

**Examples**:
- `son_clave_2-3` → Salsa, mambo
- `rumba_clave_2-3` → Rumba, afro-cuban
- `bossa_clave` → Bossa nova, samba

---

### 50. `latin.montuno_complexity` (continuous, 0.0-1.0)
**Range**: 0.0 (simple) to 1.0 (complex)
**Default**: 0.5

**What it does**: Piano montuno sophistication
**Montuno**: Repeating rhythmic piano pattern (tumbao)

**Formula**: `montuno_variation_score`

**Examples**:
- 0.2 → Simple 2-bar loop
- 0.5 → Moderate (some variations)
- 0.9 → Complex (virtuosic montuno)

---

## 🔗 HIERARCHICAL CONDITIONING

**Key Insight**: Parameters don't exist in isolation - they influence each other!

### Level 2 conditioned on Level 1:

```python
# Example: Harmony parameters conditioned by key and genre
if genre.primary == 'jazz':
    harmony.complexity = high (0.7-0.9)
    harmony.chromaticism = high (0.5-0.8)
elif genre.primary == 'rock':
    harmony.complexity = low (0.2-0.4)
    harmony.chromaticism = low (0.1-0.3)

# Melody conditioned by tempo
if tempo.bpm > 150:
    melody.note_density = high (8-16)
    melody.rhythmic_complexity = high
elif tempo.bpm < 80:
    melody.note_density = low (1-4)
```

### Level 3 conditioned on Level 1:

```python
# Genre gates entire parameter groups
if genre.primary == 'jazz':
    # Activate jazz-specific parameters
    jazz.swing_feel = ACTIVE
    jazz.walking_bass = ACTIVE
    jazz.improvisation_ratio = ACTIVE
    jazz.bebop_vocabulary = ACTIVE
    # Deactivate others
    rock.* = INACTIVE
    classical.* = INACTIVE
```

---

## 📐 PARAMETER RELATIONSHIPS

### Aggregation Examples:

**Energy Level** (derived):
```
energy.level = 0.3 × dynamics.overall_level
             + 0.3 × min(tempo/200, 1.0)
             + 0.4 × texture.density
```

**Complexity Overall** (derived):
```
complexity.overall = 0.5 × harmony.complexity
                   + 0.3 × melody.rhythmic_complexity
                   + 0.2 × rhythm.syncopation
```

### Validation Rules:

1. **Swing consistency**:
   - If `rhythm.swing_amount > 0.6`, then `genre.primary` should be jazz

2. **Rock simplicity**:
   - If `rock.power_chord_ratio > 0.7`, then `harmony.complexity < 0.5`

3. **Electronic precision**:
   - If `electronic.quantization > 0.9`, then `rhythm.groove_consistency > 0.8`

4. **Energy correlation**:
   - `energy.level` should correlate with `dynamics.overall_level` (r > 0.5)

5. **Complexity correlation**:
   - `complexity.overall` should correlate with `harmony.complexity` (r > 0.6)

---

## 🎯 WHY THIS HIERARCHY WORKS

### 1. **Dimensionality Reduction**: 118 → 50 parameters (57.6% reduction)
- Removed redundancy (8 rhythm parameters → 5)
- Merged correlated features
- Kept only musically meaningful distinctions

### 2. **Hierarchical Efficiency**:
- Level 1 gates entire sections (8 params control which 22 are active)
- Reduces parameter space exponentially
- Natural conditional generation

### 3. **Musical Validity**:
- Parameters map to actual musical concepts
- Each parameter has clear musical function
- Extractable from real MIDI files

### 4. **Computational Tractability**:
- 50 parameters feasible for MTL neural network
- Hierarchical structure enables efficient training
- Genre conditioning allows specialized expertise

### 5. **Human Interpretability**:
- Each parameter has musical meaning
- Musicians can understand and control each dimension
- Enables semantic music generation

---

## 📊 EXTRACTION PIPELINE

### Stage 1: Basic MIDI Analysis
- tempo.bpm, time_signature, key.tonic, key.mode
- orchestration.instrument_count, structure.form

### Stage 2: Level 2 Feature Extraction
- Extract all 20 universal dimensions
- harmony.*, melody.*, rhythm.*, dynamics.*, texture.*

### Stage 3: Level 1 Aggregation
- Calculate derived parameters:
  - energy.level (from dynamics, tempo, texture)
  - complexity.overall (from harmony, melody, rhythm)
  - genre.primary (classification from Level 2 features)

### Stage 4: Genre-Specific Analysis
- Based on detected genre, extract relevant Level 3 params
- jazz.* if jazz, classical.* if classical, etc.

**Target Speed**: <100ms per MIDI file for all 50 parameters

---

## 🎵 USAGE EXAMPLE

### Big Band Jazz Analysis:

```json
{
  "level1": {
    "genre.primary": "jazz",
    "tempo.bpm": 180,
    "time_signature": "4/4",
    "key.tonic": "Bb",
    "key.mode": "major",
    "energy.level": 0.85,
    "complexity.overall": 0.7,
    "structure.form": "AABA"
  },
  "level2": {
    "harmony.chord_density": 6.0,
    "harmony.complexity": 0.8,
    "harmony.chromaticism": 0.6,
    "harmony.tension": 0.5,
    "harmony.voicing_spread": 0.7,
    "harmony.progression_predictability": 0.6,
    "melody.note_density": 8.0,
    "melody.range_semitones": 18,
    "melody.contour_smoothness": 0.6,
    "melody.rhythmic_complexity": 0.7,
    "melody.repetition": 0.4,
    "rhythm.subdivision": "eighth",
    "rhythm.syncopation": 0.5,
    "rhythm.groove_consistency": 0.85,
    "rhythm.polyrhythm": 0.2,
    "rhythm.swing_amount": 0.67,
    "dynamics.overall_level": 0.75,
    "dynamics.range": 0.4,
    "texture.polyphony": 8,
    "texture.density": 12.0
  },
  "level3": {
    "orchestration.instrument_count": 17,
    "orchestration.register_balance": 0.5,
    "articulation.legato_ratio": 0.8,
    "structure.section_contrast": 0.6,
    "structure.repetition_level": 0.5,
    "jazz.swing_feel": "hard",
    "jazz.walking_bass": 0.95,
    "jazz.improvisation_ratio": 0.4,
    "jazz.bebop_vocabulary": 0.6
  }
}
```

This describes: **Fast big band swing** (Duke Ellington style) - 17 instruments, hard swing, walking bass, moderate bebop vocabulary, complex harmony, high energy.

---

## 🔬 SCIENTIFIC VALIDATION

### White-Box Learning Connection:

Your 50 parameters ARE semantic features as defined in the "Towards White Box Deep Learning" paper:

- **Level 1**: Two-Step layer (establishes invariances)
- **Level 2**: Convolutional layer (universal features)
- **Level 3**: Affine layer (genre-specific transformations)
- **Output**: Logical layer (MIDI generation)

Each parameter has:
- **Semantic meaning**: Musically interpretable
- **Locality**: Small changes preserve musical identity
- **Invariance**: Robust to irrelevant variations

---

## 📚 SUMMARY

**50 Hierarchical Parameters** = Complete Musical Description

- **8 Level 1**: Global context (genre, tempo, key, energy, complexity, form)
- **20 Level 2**: Universal dimensions (harmony 6, melody 5, rhythm 5, dynamics 2, texture 2)
- **22 Level 3**: Genre-specific (universal 5, jazz 4, classical 3, rock 3, electronic 3, hiphop 2, latin 2)

**Result**: Any musical style expressible with 50 numbers.

**Training Target**: Neural network learns to predict all 50 from 200D MIDI features.

**Generation**: Specify 50 parameters → HarmonyModule generates MIDI → 90%+ reconstruction fidelity.

---

**Document prepared by**: Claude (Hierarchical Parameter Analysis)
**Date**: November 20, 2025
**Based on**: hierarchical_parameters.json v2.0
