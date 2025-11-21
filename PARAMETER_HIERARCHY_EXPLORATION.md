# MIDI Generator Parameter Hierarchy and Preset System
## Comprehensive Exploration Report

**Date:** November 21, 2025
**System:** HierarchicalParameterExtractorV2 (v2.1)
**Scope:** Complete parameter hierarchy, extraction system, and relationship mapping

---

## EXECUTIVE SUMMARY

The MIDI generator implements a **three-level hierarchical parameter system** with **50 core parameters** extracted from MIDI files:
- **Level 1:** 8 global context parameters
- **Level 2:** 20 universal dimension parameters (5 categories)
- **Level 3:** 22 genre-specific parameters (7 genres)

This represents a **57.6% reduction** from the original 118-parameter system, achieving cleaner organization while maintaining musical expressiveness.

**Additionally**, there are **484+ expanded parameters** across specialized expansion modules for deep control of instrumentation, dynamics, harmony, melody, rhythm, and structure.

---

## 1. PARAMETER EXTRACTORS AND FILES

### Core Parameter Extraction Files

| File | Purpose | Type | Status |
|------|---------|------|--------|
| `hierarchical_extractor_v2.py` (889 lines) | **ACTIVE** - Integrated extractor producing 200D features + 50 hierarchical parameters | Implemented | Fully Functional |
| `hierarchical_extractor.py` (966 lines) | V1 extractor - standalone 50-parameter extraction | Implemented | Available |
| `hierarchical_parameters.json` (33KB) | Complete schema defining all 50 parameters with types, ranges, defaults | Schema | Authoritative |
| `universal_registry.py` (1147 lines) | Central registry for 2000+ parameters with type system, validation, dependencies | Framework | Implemented |

### Expansion Module Files (484+ Parameters)

| Module | Parameters | Status | Focus |
|--------|-----------|--------|-------|
| `harmony_deep_expansion.py` | 94 parameters | Implemented | Advanced voicing, extensions, substitutions (6 register functions) |
| `melody_rhythm_expansion.py` | 120 parameters | Implemented | Contour, intervals, chromaticism, rhythm patterns (3 register functions) |
| `dynamics_articulation_expansion.py` | 90 parameters | Implemented | Velocity expression, articulation curves, string/wind techniques (7 register functions) |
| `dynamics_specialist_parameters.py` | 40 parameters | Implemented | ADSR, dynamic curves, humanization, voice balancing (5 register functions) |
| `structure_expansion.py` | 60 parameters | Implemented | Form, transitions, development, repetition (4 register functions) |
| `instrumentation_expansion.py` | 80 parameters | Implemented | Piano, bass, brass, woodwind, strings, percussion (6 register functions) |
| `instrumentation_params.py` | 50+ parameters | Data Structure | Piano-specific parameters (voice density, pedal usage, etc.) |

**Total Expansion Parameters:** 484+ (across 484 ParameterDefinition instances)

### Supporting Files

| File | Purpose |
|------|---------|
| `registry_expansion.py` (857 lines) | Core parameter registration and expansion coordinator |
| `parameter_migration_map.json` (24KB) | Mapping from 118 legacy → 50 hierarchical parameters with formulas |
| `registry.json` (45KB) | Flat registry of all parameters with impact/genre metadata |
| `registry_with_structure.json` (33KB) | Hierarchically organized parameter registry |
| `musical_validator.py` (3638 lines) | Parameter validation framework |
| `hierarchical_validator.py` (659 lines) | Validation specific to hierarchical system |
| `legacy_adapter.py` (513 lines) | Adapter for backward compatibility |

---

## 2. HIERARCHICAL PARAMETER SYSTEM ARCHITECTURE

### Level 1: Global Context (8 Parameters)

High-level conditioning parameters that establish the musical context for all other parameters.

| Parameter | Type | Range/Options | Default | Extraction Method |
|-----------|------|----------------|---------|-------------------|
| **genre.primary** | categorical | jazz, classical, rock, electronic, pop, latin, hiphop | jazz | Genre classification algorithm |
| **tempo.bpm** | continuous | [40, 200] | 120 | MIDI meta events or beat tracking |
| **time_signature** | categorical | 4/4, 3/4, 6/8, 5/4, 7/8, 12/8, 2/4 | 4/4 | MIDI meta events |
| **key.tonic** | categorical | C, C#, D, D#, E, F, F#, G, G#, A, A#, B | C | Key detection algorithm |
| **key.mode** | categorical | major, minor, dorian, phrygian, lydian, mixolydian, aeolian, locrian | major | Key detection algorithm |
| **energy.level** | continuous | [0.0, 1.0] | 0.5 | Formula: 0.3*dynamics.overall_level + 0.3*min(tempo/200, 1.0) + 0.4*texture.density |
| **complexity.overall** | continuous | [0.0, 1.0] | 0.5 | Formula: 0.5*harmony.complexity + 0.3*melody.rhythmic_complexity + 0.2*rhythm.syncopation |
| **structure.form** | categorical | AABA, ABAC, verse_chorus, verse_chorus_bridge, through_composed, theme_variations, sonata, rondo | AABA | Structure analysis |

**Musical Function:** Level 1 parameters condition all downstream extraction - genre determines which Level 3 parameters activate, tempo/time_signature affect rhythm calculations, key/mode affect harmonic analysis.

---

### Level 2: Universal Dimensions (20 Parameters - 5 Categories)

Core musical dimensions that apply across ALL genres. Extracted from MIDI analysis.

#### **HARMONY (6 parameters)**

| Parameter | Type | Range | Default | Extraction | Formula |
|-----------|------|-------|---------|-----------|---------|
| **harmony.chord_density** | continuous | [1.0, 12.0] | 4.0 | chord_detection + voicing_analysis | chords_per_measure * avg_notes_per_chord |
| **harmony.complexity** | continuous | [0.0, 1.0] | 0.5 | analyze_chord_extensions | 0.3*use_9ths + 0.3*use_11ths + 0.4*use_13ths |
| **harmony.chromaticism** | continuous | [0.0, 1.0] | 0.3 | analyze_chromatic_notes | (tritone_sub_prob + modal_interchange_prob) / 2 |
| **harmony.tension** | continuous | [0.0, 1.0] | 0.5 | dissonance_analysis | avg_dissonance_score across chords |
| **harmony.voicing_spread** | continuous | [0.0, 1.0] | 0.5 | analyze_pitch_range_in_chords | avg(chord_range_semitones) / 36.0 |
| **harmony.progression_predictability** | continuous | [0.0, 1.0] | 0.5 | progression_pattern_entropy | 1.0 - normalized_entropy(progression_sequence) |

#### **MELODY (5 parameters)**

| Parameter | Type | Range | Default | Extraction | Formula |
|-----------|------|-------|---------|-----------|---------|
| **melody.note_density** | continuous | [1.0, 16.0] | 4.0 | count_notes_per_measure | total_melody_notes / total_measures |
| **melody.range_semitones** | integer | [3, 36] | 12 | pitch_analysis | highest_note - lowest_note |
| **melody.contour_smoothness** | continuous | [0.0, 1.0] | 0.7 | analyze_interval_sizes | stepwise_ratio * (1 - avg_leap_size/12) |
| **melody.rhythmic_complexity** | continuous | [0.0, 1.0] | 0.5 | rhythm_entropy_analysis | normalized_entropy(note_durations) |
| **melody.repetition** | continuous | [0.0, 1.0] | 0.5 | motif_detection | repeated_motif_ratio |

#### **RHYTHM (5 parameters)**

| Parameter | Type | Range/Options | Default | Extraction | Formula/Notes |
|-----------|------|----------------|---------|-----------|----------------|
| **rhythm.subdivision** | categorical | whole, half, quarter, eighth, triplet, sixteenth, quintuplet, sextuplet | eighth | detect_smallest_duration | Temporal granularity |
| **rhythm.syncopation** | continuous | [0.0, 1.0] | 0.3 | off_beat_ratio | notes_on_off_beats / total_notes |
| **rhythm.groove_consistency** | continuous | [0.0, 1.0] | 0.7 | timing_deviation_analysis | 1.0 - normalized_std(note_timings) |
| **rhythm.polyrhythm** | continuous | [0.0, 1.0] | 0.1 | cross_rhythm_detection | count_conflicting_rhythms / total_voices |
| **rhythm.swing_amount** | continuous | [0.5, 0.75] | 0.67 | measure_swing_ratio | duration_ratio of paired eighth notes |

#### **DYNAMICS (2 parameters)**

| Parameter | Type | Range | Default | Extraction | Formula |
|-----------|------|-------|---------|-----------|---------|
| **dynamics.overall_level** | continuous | [0.0, 1.0] | 0.6 | velocity_analysis | mean(all_velocities) / 127 |
| **dynamics.range** | continuous | [0.0, 1.0] | 0.3 | velocity_spread | std(all_velocities) / 127 |

#### **TEXTURE (2 parameters)**

| Parameter | Type | Range | Default | Extraction | Formula |
|-----------|------|-------|---------|-----------|---------|
| **texture.polyphony** | integer | [1, 12] | 4 | simultaneity_analysis | max(notes_at_time_t for all t) |
| **texture.density** | continuous | [0.5, 20.0] | 5.0 | temporal_fullness | total_notes / total_duration_seconds |

---

### Level 3: Genre-Specific Details (22 Parameters - 7 Genre Groups)

Genre-specific parameters that activate based on `genre.primary`. Sets unused genre parameters to 0.0.

#### **UNIVERSAL ORCHESTRATION (5 parameters - ALWAYS ACTIVE)**

| Parameter | Type | Range | Default | Extraction | Active For |
|-----------|------|-------|---------|-----------|-----------|
| **orchestration.instrument_count** | integer | [1, 20] | 5 | count(distinct_midi_programs) | All genres |
| **orchestration.register_balance** | continuous | [0.0, 1.0] | 0.5 | low_notes_ratio / high_notes_ratio | All genres |
| **articulation.legato_ratio** | continuous | [0.3, 1.0] | 0.9 | note_duration / inter_onset_interval | All genres |
| **structure.section_contrast** | continuous | [0.0, 1.0] | 0.5 | section_variation_analysis | All genres |
| **structure.repetition_level** | continuous | [0.0, 1.0] | 0.5 | repeated_material_ratio | All genres |

#### **JAZZ-SPECIFIC (4 parameters - genre.primary == 'jazz')**

| Parameter | Type | Range/Options | Default | Extraction |
|-----------|------|----------------|---------|-----------|
| **jazz.swing_feel** | categorical | straight, light, medium, hard | medium | categorize(rhythm.swing_amount) |
| **jazz.walking_bass** | continuous | [0.0, 1.0] | 0.8 | detect_walking_pattern (quarter_note_bass_ratio) |
| **jazz.improvisation_ratio** | continuous | [0.0, 1.0] | 0.3 | non_thematic_material_analysis |
| **jazz.bebop_vocabulary** | continuous | [0.0, 1.0] | 0.3 | bebop_lick_detection |

#### **CLASSICAL-SPECIFIC (3 parameters - genre.primary == 'classical')**

| Parameter | Type | Range | Default | Extraction |
|-----------|------|-------|---------|-----------|
| **classical.counterpoint** | continuous | [0.0, 1.0] | 0.5 | voice_independence_analysis |
| **classical.development_density** | continuous | [0.0, 1.0] | 0.5 | thematic_variation_analysis |
| **classical.voice_leading_quality** | continuous | [0.0, 1.0] | 0.8 | voice_leading_cost_function |

#### **ROCK-SPECIFIC (3 parameters - genre.primary == 'rock')**

| Parameter | Type | Range | Default | Extraction |
|-----------|------|-------|---------|-----------|
| **rock.power_chord_ratio** | continuous | [0.0, 1.0] | 0.7 | detect_power_chords (power_chords / total_chords) |
| **rock.riff_repetition** | continuous | [0.0, 1.0] | 0.7 | riff_pattern_detection |
| **rock.distortion_level** | continuous | [0.0, 1.0] | 0.5 | articulation_intensity_proxy (velocity * accent_density) |

#### **ELECTRONIC-SPECIFIC (3 parameters - genre.primary == 'electronic')**

| Parameter | Type | Range | Default | Extraction |
|-----------|------|-------|---------|-----------|
| **electronic.quantization** | continuous | [0.0, 1.0] | 0.9 | grid_alignment_measure |
| **electronic.filter_movement** | continuous | [0.0, 1.0] | 0.5 | spectral_variation_proxy (via velocity/cc) |
| **electronic.arpeggio_density** | continuous | [0.0, 1.0] | 0.3 | arpeggio_pattern_detection |

#### **HIP-HOP-SPECIFIC (2 parameters - genre.primary == 'hiphop')**

| Parameter | Type | Range | Default | Extraction |
|-----------|------|-------|---------|-----------|
| **hiphop.sample_based** | continuous | [0.0, 1.0] | 0.7 | loop_repetition_analysis |
| **hiphop.boom_bap_feel** | continuous | [0.0, 1.0] | 0.6 | drum_pattern_classification |

#### **LATIN-SPECIFIC (2 parameters - genre.primary == 'latin')**

| Parameter | Type | Range/Options | Default | Extraction |
|-----------|------|----------------|---------|-----------|
| **latin.clave_pattern** | categorical | none, son_clave_2-3, son_clave_3-2, rumba_clave_2-3, rumba_clave_3-2, bossa_clave | son_clave_2-3 | clave_detection_algorithm |
| **latin.montuno_complexity** | continuous | [0.0, 1.0] | 0.5 | montuno_pattern_analysis |

---

## 3. EXTRACTION PIPELINE AND DATA FLOW

### HierarchicalParameterExtractorV2 Extraction Stages

```
MIDI File Input
    ↓
Stage 1: MIDI Analysis
    ├─ Extract tempo, time signature, key
    ├─ Parse all note events with timing/velocity
    ├─ Build instrument-to-notes mapping
    ├─ Identify melody track
    └─ Output: MIDIAnalysis dataclass (246 data points)
    ↓
Stage 2: 200D Feature Vector Extraction
    └─ OptimizedFeatureExtractor
        └─ Output: 200-dimensional numpy array (neural encoder INPUT)
    ↓
Stage 3: Level 2 Extraction (Universal Dimensions)
    ├─ Harmony extraction (6 parameters)
    ├─ Melody extraction (5 parameters)
    ├─ Rhythm extraction (5 parameters)
    ├─ Dynamics extraction (2 parameters)
    ├─ Texture extraction (2 parameters)
    └─ Output: 20 Level 2 parameters in 5 categories
    ↓
Stage 4: Level 1 Extraction (Global Context)
    ├─ Genre classification (IMPROVED with Level 2 features)
    ├─ Energy level calculation
    ├─ Complexity calculation
    ├─ Key detection
    └─ Output: 8 Level 1 parameters
    ↓
Stage 5: Level 3 Extraction (Genre-Specific)
    ├─ Universal Orchestration (5 params - ALWAYS)
    ├─ Jazz-specific (4 params IF genre='jazz')
    ├─ Classical-specific (3 params IF genre='classical')
    ├─ Rock-specific (3 params IF genre='rock')
    ├─ Electronic-specific (3 params IF genre='electronic')
    ├─ Hip-Hop-specific (2 params IF genre='hiphop')
    ├─ Latin-specific (2 params IF genre='latin')
    └─ Output: 22 Level 3 parameters (5 universal + 17 genre-conditional)
    ↓
Final Output:
├─ features: List[float] (200 dimensions for neural encoder)
├─ parameters:
│  ├─ level1_global: 8 parameters
│  ├─ level2_universal: 20 parameters (organized in 5 heads)
│  └─ level3_genre_specific: 22 parameters (5 universal + genre-specific)
└─ metadata: extraction info, file path, note count, duration
```

### Parameter Dependencies and Conditioning

**Level 2 Conditions on Level 1:**
- `harmony.*` ← depends on: genre.primary, key.tonic, key.mode
- `melody.*` ← depends on: key.tonic, key.mode, tempo.bpm
- `rhythm.*` ← depends on: tempo.bpm, time_signature, genre.primary
- `dynamics.*` ← depends on: energy.level
- `texture.*` ← depends on: complexity.overall, energy.level

**Level 3 Conditions on Level 1:**
- `jazz.*` ← activated when: genre.primary == 'jazz'
- `classical.*` ← activated when: genre.primary == 'classical'
- `rock.*` ← activated when: genre.primary == 'rock'
- `electronic.*` ← activated when: genre.primary == 'electronic'
- `hiphop.*` ← activated when: genre.primary == 'hiphop'
- `latin.*` ← activated when: genre.primary == 'latin'

---

## 4. PARAMETER COUNTS BY CATEGORY AND HEAD

### Summary Table

| Category | Head | Level 1 | Level 2 | Level 3 | Total | Type |
|----------|------|---------|---------|---------|-------|------|
| **Harmony** | Harmony | 0 | 6 | 0 | 6 | Extracted |
| **Melody** | Melody | 0 | 5 | 0 | 5 | Extracted |
| **Rhythm** | Rhythm | 0 | 5 | 0 | 5 | Extracted |
| **Dynamics** | Dynamics | 0 | 2 | 0 | 2 | Extracted |
| **Texture** | Texture | 0 | 2 | 0 | 2 | Extracted |
| **Global** | Global Context | 8 | 0 | 0 | 8 | Extracted |
| **Orchestration** | Universal Orchestration | 0 | 0 | 5 | 5 | Extracted (Always) |
| **Jazz** | Jazz-Specific | 0 | 0 | 4 | 4 | Extracted (Conditional) |
| **Classical** | Classical-Specific | 0 | 0 | 3 | 3 | Extracted (Conditional) |
| **Rock** | Rock-Specific | 0 | 0 | 3 | 3 | Extracted (Conditional) |
| **Electronic** | Electronic-Specific | 0 | 0 | 3 | 3 | Extracted (Conditional) |
| **Hip-Hop** | Hip-Hop-Specific | 0 | 0 | 2 | 2 | Extracted (Conditional) |
| **Latin** | Latin-Specific | 0 | 0 | 2 | 2 | Extracted (Conditional) |
| | **TOTAL** | **8** | **20** | **22** | **50** | **Core System** |

### Parameter Types Distribution

| Type | Count | Examples |
|------|-------|----------|
| **Continuous** | 32 | tempo.bpm, energy.level, harmony.complexity, dynamics.overall_level |
| **Integer** | 3 | melody.range_semitones, orchestration.instrument_count, texture.polyphony |
| **Categorical** | 13 | genre.primary, time_signature, key.tonic, key.mode, rhythm.subdivision, jazz.swing_feel, latin.clave_pattern |
| **Boolean** | 0 | (Only in expansion modules) |
| **Probability** | 0 | (Normalized to continuous [0.0, 1.0]) |

---

## 5. EXPANSION MODULES: ADVANCED PARAMETER SETS (484+ Parameters)

The system extends the 50 core parameters with specialized expansion modules for fine-grained control.

### Expansion Module Breakdown

#### **harmony_deep_expansion.py (94 parameters)**
**6 register functions for:**
- Advanced voicing techniques (quartal, quintal, cluster voicings)
- Extended chord extensions (9ths, 11ths, 13ths)
- Substitution techniques (tritone, modal interchange)
- Advanced voice leading algorithms
- Chord progression patterns
- Harmonic function analysis

**Sample Parameters:**
- `harmony.voicing.quartal_probability`, `harmony.voicing.quintal_probability`
- `harmony.extensions.*`, `harmony.substitution.*`
- `harmony.voice_leading.smoothness`, `harmony.voice_leading.parallel_motion_tolerance`

#### **melody_rhythm_expansion.py (120 parameters)**
**3 register functions for:**
- Melodic contour shapes (arch, ascending, descending, wave, static)
- Interval patterns and stepwise motion
- Chromatic alteration amounts
- Rhythm pattern analysis
- Microtiming humanization

**Sample Parameters:**
- `melody.contour.arch_probability`, `melody.contour.inverted_arch_prob`
- `melody.intervals.stepwise_probability`, `melody.intervals.max_leap`
- `rhythm.swing.amount`, `rhythm.microtiming.variation`

#### **dynamics_articulation_expansion.py (90 parameters)**
**7 register functions for:**
- Velocity expression curves
- Articulation shapes (staccato, legato, portato)
- Dynamic envelope control
- Expression mapping
- String techniques (pizzicato, tremolo, sul ponticello)
- Wind/brass techniques (vibrato, growl, flutter-tongue)

**Sample Parameters:**
- `dynamics.velocity.overall_level`, `dynamics.velocity.range`
- `articulation.staccato_duration`, `articulation.portato_gap`
- `dynamics.expression_curve`, `dynamics.vibrato_depth`

#### **dynamics_specialist_parameters.py (40 parameters)**
**5 register functions for:**
- ADSR envelope control (attack, decay, sustain, release)
- Dynamic curves (linear, exponential, S-curve)
- Humanization parameters
- Voice balancing across instruments

**Sample Parameters:**
- `dynamics.adsr.attack_time`, `dynamics.adsr.decay_time`
- `dynamics.curve.shape`, `dynamics.curve.sustain_time`

#### **structure_expansion.py (60 parameters)**
**4 register functions for:**
- Form architecture (intro length, verse length, bridge length, coda)
- Section transitions and drum fills
- Thematic development techniques
- Repetition and motif recall

**Sample Parameters:**
- `structure.form.type`, `structure.form.length_bars`, `structure.form.intro_bars`
- `structure.transition.drum_fill_prob`, `structure.transition.key_change_prob`
- `structure.development.variation_intensity`, `structure.development.sequence_prob`

#### **instrumentation_expansion.py (80 parameters)**
**6 register functions for:**
- Piano voicing and comping patterns
- Bass playing styles
- Brass section control
- Woodwind articulation
- String section techniques
- Percussion and drum patterns

**Sample Parameters:**
- `instrumentation.piano.voicing_density`, `instrumentation.piano.comping_pattern`
- `instrumentation.bass.walking_pattern`, `instrumentation.bass.syncopation`
- `instrumentation.brass.vibrato_depth`, `instrumentation.woodwind.breath_amount`

#### **instrumentation_params.py (50+ parameters)**
**Data structure (not extractors) for:**
- Piano-specific parameter definitions
- Pre-computed defaults and presets
- Instrument-specific constraints

**Sample Parameters:**
- `instrumentation.piano.pedal_usage`, `instrumentation.piano.cluster_probability`
- `instrumentation.piano.shell_voicing_prob`, `instrumentation.piano.alberti_bass_prob`

---

## 6. PARAMETER VALUE RANGES AND DEFAULTS

### Default Value Strategy

**Continuous Parameters:** Normalized to [0.0, 1.0] or domain-specific ranges:
- `energy.level`: [0.0, 1.0] ← combines dynamics (0.3), tempo (0.3), density (0.4)
- `complexity.overall`: [0.0, 1.0] ← combines harmony (0.5), melody (0.3), rhythm (0.2)
- `harmony.chord_density`: [1.0, 12.0] ← chords per measure
- `dynamics.overall_level`: [0.0, 1.0] ← MIDI velocity normalized
- `rhythm.swing_amount`: [0.5, 0.75] ← standard swing ratios

**Categorical Parameters:** Genre/style-specific options:
- `genre.primary`: 7 options (jazz, classical, rock, electronic, pop, latin, hiphop)
- `rhythm.subdivision`: 8 options (whole, half, quarter, eighth, triplet, sixteenth, quintuplet, sextuplet)
- `jazz.swing_feel`: 4 options (straight, light, medium, hard)
- `latin.clave_pattern`: 6 options (none, son_clave_2-3, son_clave_3-2, rumba_clave_2-3, rumba_clave_3-2, bossa_clave)

**Integer Parameters:** Countable dimensions:
- `melody.range_semitones`: [3, 36] semitones
- `orchestration.instrument_count`: [1, 20] instruments
- `texture.polyphony`: [1, 12] simultaneous voices

---

## 7. PARAMETER MIGRATION AND LEGACY COMPATIBILITY

### 118 → 50 Parameter Consolidation (parameter_migration_map.json)

**Consolidation Strategies Used:**

1. **Direct Mapping (1:1)** - 6 parameters kept as-is:
   - `harmony.voicing.spread` → `harmony.voicing_spread`
   - `rhythm.swing.amount` → `rhythm.swing_amount`
   - `articulation.duration.ratio` → `articulation.legato_ratio`

2. **Merged Mapping (N:1)** - Multiple old params consolidated:
   - `harmony.voicing.density` + `harmony.voicing.type` → `harmony.chord_density`
   - `harmony.extensions.use_9ths/11ths/13ths` → `harmony.complexity`
   - `harmony.substitution.tritone_probability` + `modal_interchange_probability` → `harmony.chromaticism`
   - `melody.intervals.max_leap` + `melody.intervals.stepwise_probability` → `melody.contour_smoothness`
   - `dynamics.velocity.base` + drum velocities → `dynamics.overall_level`
   - `texture` parameters (5 old) → `texture.polyphony` + `texture.density`

3. **Computed Mapping (Formula-based)** - Parameters derived via computation:
   - `energy.level` = 0.3*dynamics + 0.3*tempo_norm + 0.4*density
   - `complexity.overall` = 0.5*harmony.complexity + 0.3*melody.rhythmic_complexity + 0.2*rhythm.syncopation

4. **New Parameters** - 8 parameters added from MIDI analysis:
   - `genre.primary`, `energy.level`, `complexity.overall`
   - `harmony.progression_predictability`, `harmony.tension`
   - `melody.rhythmic_complexity`, `melody.repetition`
   - `rhythm.groove_consistency`, `rhythm.polyrhythm`

5. **Dropped Parameters** - Parameters removed with justification:
   - Instrument-specific drum parameters (moved to expansion modules)
   - Redundant articulation variants
   - Over-specialized timbre parameters

---

## 8. IMPLEMENTED VS. DEFINED PARAMETERS

### Fully Implemented (Actively Extracted from MIDI)

**Core System (50 parameters):**
- ✅ All Level 1 parameters (8/8)
- ✅ All Level 2 parameters (20/20)
- ✅ Universal Orchestration (5/5 - always)
- ✅ Jazz-specific (4/4 - when active)
- ✅ Classical-specific (3/3 - when active)
- ✅ Rock-specific (3/3 - when active)
- ✅ Electronic-specific (3/3 - when active)
- ✅ Hip-Hop-specific (2/2 - when active)
- ✅ Latin-specific (2/2 - when active)

**Total: 50/50 (100% implemented)**

### Expansion Modules (Defined but Not Auto-Extracted)

These are registered in the parameter registry but require custom analysis functions:

- **harmony_deep_expansion.py**: 94 parameters defined, NOT auto-extracted
  - Status: Registered, awaiting extraction implementation
  - Extractors needed: Advanced voicing detection, extension analysis

- **melody_rhythm_expansion.py**: 120 parameters defined, NOT auto-extracted
  - Status: Registered, awaiting extraction implementation
  - Extractors needed: Contour analysis, interval pattern detection

- **dynamics_articulation_expansion.py**: 90 parameters defined, NOT auto-extracted
  - Status: Registered, awaiting extraction implementation
  - Extractors needed: Velocity curve analysis, articulation pattern detection

- **dynamics_specialist_parameters.py**: 40 parameters defined, NOT auto-extracted
  - Status: Registered, awaiting extraction implementation
  - Extractors needed: ADSR detection, envelope curve fitting

- **structure_expansion.py**: 60 parameters defined, NOT auto-extracted
  - Status: Registered, awaiting extraction implementation
  - Extractors needed: Form detection, transition analysis

- **instrumentation_expansion.py**: 80 parameters defined, NOT auto-extracted
  - Status: Registered, awaiting extraction implementation
  - Extractors needed: Instrument-specific pattern recognition

- **instrumentation_params.py**: 50+ parameters defined, NOT auto-extracted
  - Status: Data structure only, no extraction functions
  - Type: Parameter definition container

**Total Expansion: 484+ parameters defined, ~0 actively extracted (in core pipeline)**

### Extraction Status Summary

| System | Parameters | Fully Implemented | Defined Only | Implementation %|
|--------|-----------|------------------|--------------|-----------------|
| **Hierarchical Core** | 50 | 50 | 0 | 100% |
| **Expansion Modules** | 484+ | 0 | 484+ | 0% |
| **Total** | 534+ | 50 | 484+ | 9.4% |

**Note:** Expansion modules are registered but require custom extraction implementations. The 50-parameter core system is production-ready and fully auto-extracted.

---

## 9. PARAMETER IMPACT AND MUSICAL FUNCTION

### Critical Parameters (Fundamentally Change Musical Character)

| Parameter | Impact | Reason |
|-----------|--------|--------|
| **genre.primary** | CRITICAL | Determines entire genre-specific behavior |
| **harmony.complexity** | CRITICAL | Transforms chord sophistication |
| **rhythm.swing_amount** | CRITICAL | Defines jazz vs. straight feel |
| **rock.power_chord_ratio** | CRITICAL | Core rock/metal identity |
| **electronic.quantization** | CRITICAL | Electronic vs. human feel |
| **jazz.walking_bass** | CRITICAL | Defines jazz rhythm section |

### High Impact Parameters (Significant Perceptual Impact)

| Parameter | Impact | Reason |
|-----------|--------|--------|
| **tempo.bpm** | HIGH | Controls overall speed and energy |
| **harmony.chord_density** | HIGH | Determines harmonic activity |
| **melody.note_density** | HIGH | Melodic activity level |
| **rhythm.syncopation** | HIGH | Rhythmic displacement |
| **dynamics.overall_level** | HIGH | Overall loudness |
| **texture.density** | HIGH | Fullness of arrangement |

### Medium Impact Parameters (Noticeable Refinement)

| Parameter | Impact | Reason |
|-----------|--------|--------|
| **key.tonic** / **key.mode** | MEDIUM | Establish tonal center (less perceptual change than content) |
| **energy.level** | MEDIUM | High-level intensity (aggregated from other params) |
| **melody.contour_smoothness** | MEDIUM | Melodic continuity |
| **harmony.tension** | MEDIUM | Dissonance level |
| **time_signature** | MEDIUM | Metric structure |

---

## 10. VALIDATION AND CONSTRAINT RULES

### Cross-Parameter Validation Rules

From `hierarchical_parameters.json`:

| Rule | Type | Musical Justification |
|------|------|----------------------|
| `if rhythm.swing_amount > 0.6 then genre.primary should be 'jazz' or 'blues'` | consistency_check | Swing is primarily a jazz feature |
| `if rock.power_chord_ratio > 0.7 then harmony.complexity should be < 0.5` | musical_consistency | Power chords reduce harmonic complexity |
| `if electronic.quantization > 0.9 then rhythm.groove_consistency should be > 0.8` | consistency_check | Perfect quantization correlates with tight groove |
| `energy.level correlates with dynamics.overall_level (r > 0.5)` | correlation_check | Energy combines dynamics, tempo, density |
| `complexity.overall correlates with harmony.complexity (r > 0.6)` | correlation_check | Overall complexity driven largely by harmony |

### Activation Rules (Genre Conditioning)

```
if genre.primary == 'jazz':
    → ACTIVATE: jazz.swing_feel, jazz.walking_bass, jazz.improvisation_ratio, jazz.bebop_vocabulary
    → DEACTIVATE: rock.*, electronic.*, hiphop.*, latin.*
    
if genre.primary == 'classical':
    → ACTIVATE: classical.counterpoint, classical.development_density, classical.voice_leading_quality
    → DEACTIVATE: jazz.*, rock.*, electronic.*, hiphop.*, latin.*
    
if genre.primary == 'rock':
    → ACTIVATE: rock.power_chord_ratio, rock.riff_repetition, rock.distortion_level
    → DEACTIVATE: jazz.*, classical.*, electronic.*, hiphop.*, latin.*

# etc. for electronic, hiphop, latin
```

---

## 11. CONFIGURATION FILES AND PRESETS

### Key Configuration Files

| File | Purpose | Structure |
|------|---------|-----------|
| **hierarchical_parameters.json** | Master schema | Nested: levels → subcategories → parameters |
| **registry.json** | Flat parameter registry | {param_name: {type, default, range, impact, genres}} |
| **registry_with_structure.json** | Hierarchical registry | Category-organized parameters |
| **parameter_migration_map.json** | Legacy→New mapping | {old_param: {new_param, mapping_type, formula}} |

### Genre Presets (in structure_expansion.py)

Pre-defined parameter sets for major genres:

**Jazz Presets:**
- `structure.form.type`: "AABA" (standard 32-bar form)
- `rhythm.swing_amount`: 0.67 (medium swing)
- `jazz.walking_bass`: 0.8 (walking bass probability)
- `structure.transition.drum_fill_prob`: 0.7
- `structure.transition.texture_change_prob`: 0.6

**Blues Presets:**
- `structure.form.type`: "blues"
- `structure.form.length_bars`: 12
- `structure.form.vamp_probability`: 0.5
- `rhythm.syncopation`: 0.3 (moderate)

**Classical Presets:**
- `structure.form.type`: "sonata"
- `classical.counterpoint`: 0.5
- `classical.development_density`: 0.5
- `structure.development.variation_intensity`: 0.7

---

## 12. EXTRACTION VALIDATION AND COMPLETENESS CHECKS

### Level Count Verification (hierarchical_extractor_v2.py)

```python
# Verification in extract_complete():
if level1_count != 8:
    warnings.warn(f"Level 1: expected 8 params, got {level1_count}")
if level2_count != 20:
    warnings.warn(f"Level 2: expected 20 params, got {level2_count}")
if level3_count != 22:
    warnings.warn(f"Level 3: expected 22 params, got {level3_count}")
    
# Feature vector validation:
if len(features_list) != 200:
    warnings.warn(f"Expected 200 features, got {len(features_list)}")
    # Pad or truncate to 200
```

### Data Quality Checks

- ✅ All required metadata extracted (tempo, time signature, key)
- ✅ All notes parsed with timing and velocity
- ✅ Instrument program changes tracked
- ✅ Melody voice identified (highest average pitch)
- ✅ Feature vector dimensions validated (200D)
- ✅ Parameter ranges validated (no out-of-bounds values)

---

## 13. RELATIONSHIP BETWEEN EXTRACTORS

```
┌─────────────────────────────────────────────────────────┐
│  HierarchicalParameterExtractorV2 (v2.1) - RECOMMENDED  │
│  Integrated: 200D features + 50 parameters              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ├─ Uses OptimizedFeatureExtractor
                     │  for 200D feature vector
                     │
                     ├─ Uses _analyze_midi()
                     │  for basic MIDI parsing
                     │
                     ├─ Calls _extract_level2() → 20 params
                     │
                     ├─ Calls _extract_level1() → 8 params
                     │  (depends on Level 2)
                     │
                     └─ Calls _extract_level3_complete() → 22 params
                        (depends on Levels 1 & 2)

┌────────────────────────────────────────────────────────┐
│  HierarchicalParameterExtractor (v1 - Legacy)          │
│  Standalone: 50 parameters only                        │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  UniversalParameterRegistry                            │
│  Central registry for all 2000+ parameters across      │
│  entire system with type system, validation            │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  Expansion Modules (7 modules, 484+ parameters)        │
│  Registered but not auto-extracted from MIDI           │
│  Require custom analysis implementations               │
└────────────────────────────────────────────────────────┘
```

---

## 14. USAGE EXAMPLE

```python
from midi_generator.parameters.hierarchical_extractor_v2 import HierarchicalParameterExtractorV2

# Initialize extractor
extractor = HierarchicalParameterExtractorV2(verbose=True)

# Extract complete training data
result = extractor.extract_complete("path/to/file.mid")

# Result structure:
{
    'features': [200 floats],  # Neural encoder input
    'parameters': {
        'level1_global': {
            'genre.primary': 'jazz',
            'tempo.bpm': 120.0,
            'time_signature': '4/4',
            'key.tonic': 'C',
            'key.mode': 'major',
            'energy.level': 0.65,
            'complexity.overall': 0.58,
            'structure.form': 'AABA'
        },
        'level2_universal': {
            'harmony': {
                'chord_density': 4.2,
                'complexity': 0.65,
                'chromaticism': 0.35,
                'tension': 0.45,
                'voicing_spread': 0.55,
                'progression_predictability': 0.60
            },
            'melody': {
                'note_density': 4.8,
                'range_semitones': 18,
                'contour_smoothness': 0.72,
                'rhythmic_complexity': 0.55,
                'repetition': 0.48
            },
            'rhythm': {
                'subdivision': 'eighth',
                'syncopation': 0.35,
                'groove_consistency': 0.78,
                'polyrhythm': 0.12,
                'swing_amount': 0.68
            },
            'dynamics': {
                'overall_level': 0.58,
                'range': 0.32
            },
            'texture': {
                'polyphony': 4,
                'density': 5.2
            }
        },
        'level3_genre_specific': {
            'orchestration': {
                'instrument_count': 5,
                'register_balance': 0.52,
                'legato_ratio': 0.85,
                'section_contrast': 0.48,
                'repetition_level': 0.52
            },
            'jazz': {
                'swing_feel': 'medium',
                'walking_bass': 0.75,
                'improvisation_ratio': 0.35,
                'bebop_vocabulary': 0.32
            },
            'classical': {  # Set to 0 for non-classical
                'counterpoint': 0.0,
                'development_density': 0.0,
                'voice_leading_quality': 0.0
            }
            # ... other genres similarly
        }
    },
    'metadata': {
        'file': 'path/to/file.mid',
        'extraction_version': '2.1.0',
        'total_notes': 2847,
        'duration_seconds': 245.5,
        'feature_count': 200,
        'parameter_count': 50
    }
}
```

---

## SUMMARY STATISTICS

| Metric | Count |
|--------|-------|
| **Total Parameter Extractors** | 2 (v1, v2.1) |
| **Core Hierarchical Parameters** | 50 |
| **Core Parameter Categories** | 5 (Level 2 heads) |
| **Genre-Specific Parameter Groups** | 7 |
| **Expansion Module Parameters** | 484+ |
| **Total System Parameters** | 534+ |
| **Continuous Parameters** | 32 |
| **Categorical Parameters** | 13 |
| **Integer Parameters** | 3 |
| **Parameter Registration Functions** | 31 (across modules) |
| **Validation Rules** | 5+ |
| **Genre Presets Defined** | 7+ |
| **Features Extracted (Neural Input)** | 200 |
| **Extraction Pipeline Stages** | 5 |
| **MIDI Analysis Data Points** | 246+ |

---

**End of Report**
