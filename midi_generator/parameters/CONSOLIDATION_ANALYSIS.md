# Parameter Consolidation Analysis - Agent 01
## Phase 1: Current State Analysis

**Date:** November 20, 2025
**Current Parameter Count:** 118 parameters
**Target Parameter Count:** 50 parameters
**Reduction:** 68 parameters (57.6% reduction)

---

## 1. Current Parameter Distribution

### By Category
| Category | Count | Percentage |
|----------|-------|------------|
| Harmony | 24 | 20.3% |
| Articulation | 22 | 18.6% |
| Rhythm | 19 | 16.1% |
| Drums | 15 | 12.7% |
| Texture | 10 | 8.5% |
| Bass | 7 | 5.9% |
| Timbre | 7 | 5.9% |
| Melody | 6 | 5.1% |
| Dynamics | 4 | 3.4% |
| Genre | 4 | 3.4% |
| **TOTAL** | **118** | **100%** |

### Key Findings

1. **Heavy Instrumentation Focus**: 44 parameters (37%) are instrumentation-specific (piano, bass, drums, brass, strings)
2. **Harmony Dominance**: 24 parameters for harmony alone - opportunity for consolidation
3. **Articulation Granularity**: 22 articulation parameters - many are instrument-specific
4. **Missing High-Level Parameters**: No explicit `energy.level`, `complexity.overall`, `tension`, etc.

---

## 2. Identified Redundancies

### Harmony Parameters (24 → 6 consolidated)
**Current Parameters:**
- `harmony.voicing.type`, `harmony.voicing.spread`, `harmony.voicing.density`
- `harmony.extensions.use_9ths`, `use_11ths`, `use_13ths`
- `harmony.substitution.tritone_probability`, `modal_interchange_probability`
- `harmony.voice_leading.smoothness`, `parallel_motion_tolerance`
- **+ 14 more in expansion modules**

**Consolidation Strategy:**
1. **`harmony.chord_density`** ← merge `voicing.density` + chord_changes_per_measure
2. **`harmony.complexity`** ← aggregate `use_9ths`, `use_11ths`, `use_13ths` (0.0-1.0 scale)
3. **`harmony.chromaticism`** ← merge `tritone_sub` + `modal_interchange` probabilities
4. **`harmony.tension`** ← NEW: compute from dissonance analysis
5. **`harmony.voicing_spread`** ← keep as-is
6. **`harmony.progression_predictability`** ← NEW: analyze progression patterns

### Melody Parameters (6 → 5 consolidated)
**Current Parameters:**
- `melody.contour.type`, `melody.intervals.stepwise_probability`
- `melody.intervals.max_leap`, `melody.chromaticism.amount`
- `melody.ornaments.probability`

**Consolidation Strategy:**
1. **`melody.note_density`** ← derive from intervals + ornaments
2. **`melody.range_semitones`** ← calculate from `max_leap`
3. **`melody.contour_smoothness`** ← inverse of `max_leap` + `stepwise_probability`
4. **`melody.rhythmic_complexity`** ← NEW: from rhythm analysis
5. **`melody.repetition`** ← NEW: motif detection

### Rhythm Parameters (19 → 5 consolidated)
**Current Parameters:**
- `rhythm.swing.amount`, `rhythm.syncopation.probability`
- `rhythm.microtiming.variation`
- **+ 16 drum-specific rhythm parameters**

**Consolidation Strategy:**
1. **`rhythm.subdivision`** ← categorical: sixteenth, eighth, triplet, etc.
2. **`rhythm.syncopation`** ← keep probability
3. **`rhythm.groove_consistency`** ← NEW: timing stability
4. **`rhythm.polyrhythm`** ← NEW: detect cross-rhythms
5. **`rhythm.swing_amount`** ← keep as-is

### Dynamics Parameters (4 → 2 consolidated)
**Current Parameters:**
- `dynamics.velocity.base`, `dynamics.velocity.variation`
- `drums.kick.velocity_min`, `drums.kick.velocity_max`

**Consolidation Strategy:**
1. **`dynamics.overall_level`** ← merge `velocity.base` across instruments
2. **`dynamics.range`** ← merge `velocity.variation` metrics

### Texture Parameters (10 → 2 consolidated)
**Current Parameters:**
- `texture.polyphonic.density`, `texture.voice.independence`
- `texture.vertical.density`, `texture.horizontal.density`
- `texture.layering.count`, `texture.register.spread`
- `texture.contrast.rate`, `texture.homophonic.ratio`
- `texture.voice_crossing.density`, `texture.rhythmic.independence`

**Consolidation Strategy:**
1. **`texture.polyphony`** ← max simultaneous voices (merge `polyphonic.density` + `layering.count`)
2. **`texture.density`** ← aggregate note density across all dimensions

---

## 3. Proposed Hierarchical Structure (50 Parameters)

### Level 1: Global Context (8 parameters)
High-level contextual parameters that condition all other parameters.

| Parameter | Type | Source | Notes |
|-----------|------|--------|-------|
| `genre.primary` | categorical | NEW/inferred | jazz, classical, rock, electronic, pop, latin |
| `tempo.bpm` | continuous | MIDI meta | 40-200 bpm |
| `time_signature` | categorical | MIDI meta | 4/4, 3/4, 6/8, 5/4, 7/8 |
| `key.tonic` | categorical | existing | C, C#, D, ... |
| `key.mode` | categorical | existing | major, minor, dorian, etc. |
| `energy.level` | continuous | **NEW** | 0.0-1.0, aggregate dynamics + tempo + density |
| `complexity.overall` | continuous | **NEW** | 0.0-1.0, aggregate harmony + melody complexity |
| `structure.form` | categorical | existing | AABA, ABAC, verse-chorus, etc. |

### Level 2: Universal Dimensions (20 parameters)
Core musical dimensions that apply across all genres.

#### Harmony (6 parameters)
| Parameter | Type | Source | Formula |
|-----------|------|--------|---------|
| `harmony.chord_density` | continuous | merge | chords_per_measure × voicing_notes |
| `harmony.complexity` | continuous | merge | (use_9ths×0.3 + use_11ths×0.3 + use_13ths×0.4) |
| `harmony.chromaticism` | continuous | merge | (tritone_sub + modal_interchange) / 2 |
| `harmony.tension` | continuous | **NEW** | dissonance_score from interval analysis |
| `harmony.voicing_spread` | continuous | existing | 0.0-1.0 |
| `harmony.progression_predictability` | continuous | **NEW** | entropy of progression patterns |

#### Melody (5 parameters)
| Parameter | Type | Source | Formula |
|-----------|------|--------|---------|
| `melody.note_density` | continuous | derive | notes_per_measure |
| `melody.range_semitones` | integer | calculate | max_pitch - min_pitch |
| `melody.contour_smoothness` | continuous | derive | stepwise_prob × (1 - max_leap/24) |
| `melody.rhythmic_complexity` | continuous | **NEW** | rhythm_entropy |
| `melody.repetition` | continuous | **NEW** | motif_repetition_ratio |

#### Rhythm (5 parameters)
| Parameter | Type | Source | Formula |
|-----------|------|--------|---------|
| `rhythm.subdivision` | categorical | derive | sixteenth, eighth, triplet, quarter |
| `rhythm.syncopation` | continuous | existing | 0.0-1.0 |
| `rhythm.groove_consistency` | continuous | **NEW** | 1 - timing_deviation_std |
| `rhythm.polyrhythm` | continuous | **NEW** | cross_rhythm_detection |
| `rhythm.swing_amount` | continuous | existing | 0.5-0.75 |

#### Dynamics (2 parameters)
| Parameter | Type | Source | Formula |
|-----------|------|--------|---------|
| `dynamics.overall_level` | continuous | merge | mean(all_velocities) / 127 |
| `dynamics.range` | continuous | merge | std(all_velocities) / 127 |

#### Texture (2 parameters)
| Parameter | Type | Source | Formula |
|-----------|------|--------|---------|
| `texture.polyphony` | integer | merge | max_simultaneous_notes |
| `texture.density` | continuous | merge | notes_per_second_all_tracks |

### Level 3: Genre-Specific Details (22 parameters)

#### Universal Orchestration (5 parameters)
| Parameter | Type | Source | Notes |
|-----------|------|--------|-------|
| `orchestration.instrument_count` | integer | derive | count(distinct_instruments) |
| `orchestration.register_balance` | continuous | **NEW** | low_notes / high_notes ratio |
| `articulation.legato_ratio` | continuous | existing | note_duration / inter_onset_interval |
| `structure.section_contrast` | continuous | **NEW** | variation between sections |
| `structure.repetition_level` | continuous | **NEW** | repeated_material_ratio |

#### Jazz-Specific (4 parameters) - Active when `genre.primary == "jazz"`
| Parameter | Type | Source | Notes |
|-----------|------|--------|-------|
| `jazz.swing_feel` | categorical | derive | straight, light, medium, hard (from swing_amount) |
| `jazz.walking_bass` | continuous | existing | walking_bass_probability |
| `jazz.improvisation_ratio` | continuous | **NEW** | non_thematic_material_ratio |
| `jazz.bebop_vocabulary` | continuous | **NEW** | bebop_lick_detection |

#### Classical-Specific (3 parameters) - Active when `genre.primary == "classical"`
| Parameter | Type | Source | Notes |
|-----------|------|--------|-------|
| `classical.counterpoint` | continuous | **NEW** | contrapuntal_motion_analysis |
| `classical.development_density` | continuous | **NEW** | thematic_development_score |
| `classical.voice_leading_quality` | continuous | existing | voice_leading_optimizer_score |

#### Rock/Metal-Specific (3 parameters) - Active when `genre.primary == "rock"`
| Parameter | Type | Source | Notes |
|-----------|------|--------|-------|
| `rock.power_chord_ratio` | continuous | existing | power_chord_probability |
| `rock.riff_repetition` | continuous | **NEW** | riff_pattern_repetition |
| `rock.distortion_level` | continuous | **NEW** | articulation_intensity_proxy |

#### Electronic-Specific (3 parameters) - Active when `genre.primary == "electronic"`
| Parameter | Type | Source | Notes |
|-----------|------|--------|-------|
| `electronic.quantization` | continuous | **NEW** | grid_alignment_measure |
| `electronic.filter_movement` | continuous | **NEW** | spectral_variation (proxy) |
| `electronic.arpeggio_density` | continuous | **NEW** | arpeggio_pattern_density |

#### Hip-Hop-Specific (2 parameters) - Active when `genre.primary == "hiphop"`
| Parameter | Type | Source | Notes |
|-----------|------|--------|-------|
| `hiphop.sample_based` | continuous | **NEW** | loop_repetition_analysis |
| `hiphop.boom_bap_feel` | continuous | **NEW** | drum_pattern_classification |

#### Latin-Specific (2 parameters) - Active when `genre.primary == "latin"`
| Parameter | Type | Source | Notes |
|-----------|------|--------|-------|
| `latin.clave_pattern` | categorical | **NEW** | son_clave, rumba_clave, bossa_clave |
| `latin.montuno_complexity` | continuous | **NEW** | montuno_pattern_analysis |

---

## 4. Parameter Migration Map (118 → 50)

### Direct Mappings (Keep As-Is)
```
✓ tempo.bpm → tempo.bpm
✓ key.tonic → key.tonic
✓ key.mode → key.mode
✓ structure.form → structure.form
✓ harmony.voicing.spread → harmony.voicing_spread
✓ rhythm.swing.amount → rhythm.swing_amount
✓ rhythm.syncopation.probability → rhythm.syncopation
✓ bass.style.walking_probability → jazz.walking_bass
✓ genre.rock.power_chord_probability → rock.power_chord_ratio
✓ articulation.duration.ratio → articulation.legato_ratio
```

### Merged Parameters (N → 1)
```
harmony.voicing.density + harmony.chord_changes → harmony.chord_density
harmony.extensions.* (3 params) → harmony.complexity (aggregated)
harmony.substitution.* (2 params) → harmony.chromaticism
melody.intervals.max_leap + stepwise_probability → melody.contour_smoothness
melody.intervals.* (2 params) → melody.range_semitones
dynamics.velocity.* (2-4 params) → dynamics.overall_level + dynamics.range
texture.* (10 params) → texture.polyphony + texture.density
```

### New Parameters (Computed/Inferred)
```
✓ genre.primary ← NEW: genre detection from features
✓ energy.level ← NEW: aggregate(dynamics, tempo, density)
✓ complexity.overall ← NEW: aggregate(harmony.complexity, melody.rhythmic_complexity)
✓ harmony.tension ← NEW: dissonance analysis
✓ harmony.progression_predictability ← NEW: progression pattern entropy
✓ melody.rhythmic_complexity ← NEW: rhythm entropy
✓ melody.repetition ← NEW: motif detection
✓ rhythm.groove_consistency ← NEW: timing stability
✓ rhythm.polyrhythm ← NEW: cross-rhythm detection
✓ orchestration.register_balance ← NEW: pitch register analysis
✓ structure.section_contrast ← NEW: section variation analysis
✓ structure.repetition_level ← NEW: material repetition
✓ jazz.improvisation_ratio ← NEW: thematic vs non-thematic
✓ jazz.bebop_vocabulary ← NEW: bebop pattern detection
✓ classical.counterpoint ← NEW: contrapuntal analysis
✓ classical.development_density ← NEW: thematic development
✓ rock.riff_repetition ← NEW: riff pattern analysis
✓ rock.distortion_level ← NEW: intensity proxy
✓ electronic.quantization ← NEW: grid alignment
✓ electronic.filter_movement ← NEW: spectral proxy
✓ electronic.arpeggio_density ← NEW: arpeggio detection
✓ hiphop.sample_based ← NEW: loop analysis
✓ hiphop.boom_bap_feel ← NEW: drum classification
✓ latin.clave_pattern ← NEW: clave detection
✓ latin.montuno_complexity ← NEW: montuno analysis
```

### Dropped Parameters (Justification)
```
✗ harmony.voice_leading.parallel_motion_tolerance → LOW IMPACT (validation report)
✗ melody.ornaments.probability → LOW IMPACT, covered by melody.note_density
✗ rhythm.microtiming.variation → LOW IMPACT, human feel not critical for ML
✗ drums.* (most specific patterns) → INSTRUMENTATION-SPECIFIC, not generalizable
✗ piano.* (20+ params) → INSTRUMENTATION-SPECIFIC, too granular
✗ bass.* (many specific techniques) → covered by bass walking + orchestration
✗ brass.* (10+ params) → INSTRUMENTATION-SPECIFIC
✗ strings.* (10+ params) → INSTRUMENTATION-SPECIFIC
✗ texture.* (many granular params) → consolidated into 2 params
```

---

## 5. Extraction Methodology

### Direct Extraction (from MIDI)
- `tempo.bpm`: MIDI meta events or beat tracking
- `time_signature`: MIDI meta events
- `key.tonic`, `key.mode`: Key detection (music21 or custom)
- `melody.range_semitones`: max(pitch) - min(pitch)
- `dynamics.overall_level`: mean(velocities)
- `dynamics.range`: std(velocities)
- `texture.polyphony`: max(simultaneous_notes)
- `orchestration.instrument_count`: count(distinct MIDI programs)

### Algorithmic Analysis
- `genre.primary`: Genre classification (use existing genre detection)
- `harmony.chord_density`: Chord detection + count per measure
- `harmony.complexity`: Analyze chord extensions (9ths, 11ths, 13ths)
- `harmony.chromaticism`: Chromatic note ratio
- `harmony.tension`: Dissonance score (interval analysis)
- `harmony.progression_predictability`: Progression pattern entropy
- `melody.note_density`: Notes per measure in melody track
- `melody.contour_smoothness`: Stepwise motion ratio + leap analysis
- `melody.rhythmic_complexity`: Rhythm entropy
- `melody.repetition`: Repeated motif detection
- `rhythm.subdivision`: Detect smallest note duration
- `rhythm.syncopation`: Off-beat note ratio
- `rhythm.groove_consistency`: Timing deviation analysis
- `rhythm.polyrhythm`: Detect conflicting rhythms
- `texture.density`: Notes per second across all tracks

### High-Level Aggregation
- `energy.level`: weighted_avg(dynamics.overall_level, tempo/200, texture.density)
- `complexity.overall`: weighted_avg(harmony.complexity, melody.rhythmic_complexity, rhythm.syncopation)

### Genre-Specific Detection
- `jazz.swing_feel`: Categorize swing_amount (0.5=straight, 0.57=light, 0.67=medium, 0.75=hard)
- `jazz.walking_bass`: Detect walking bass pattern (quarter notes, chord tones)
- `jazz.bebop_vocabulary`: Pattern matching for bebop licks
- `classical.counterpoint`: Analyze voice independence and contrapuntal motion
- `classical.voice_leading_quality`: Use existing voice leading optimizer
- `rock.power_chord_ratio`: Detect power chords (root + fifth only)
- `electronic.quantization`: Measure grid alignment (on-grid note percentage)
- `latin.clave_pattern`: Clave pattern detection (2-3 or 3-2)

---

## 6. Coverage Analysis

### Musical Dimensions Covered
✅ **Harmony**: chord density, complexity, chromaticism, tension, voicing, progressions
✅ **Melody**: density, range, contour, rhythm, repetition
✅ **Rhythm**: subdivision, syncopation, groove, polyrhythm, swing
✅ **Dynamics**: overall level, range
✅ **Texture**: polyphony, density
✅ **Timbre/Orchestration**: instrument count, register balance, articulation
✅ **Structure**: form, section contrast, repetition
✅ **Genre-Specific**: Jazz (4), Classical (3), Rock (3), Electronic (3), Hip-Hop (2), Latin (2)
✅ **Context**: genre, tempo, time signature, key, energy, complexity

### Critical Dimensions Preserved
According to `musical_validator.py` and validation reports:
- ✅ Harmonic correctness (chord density, complexity, voice leading)
- ✅ Melodic coherence (contour, range, repetition)
- ✅ Rhythmic stability (syncopation, groove consistency)
- ✅ Dynamic expressiveness (level, range)
- ✅ Genre authenticity (genre-specific parameters)

### Efficiency Gains
- **Parameter reduction**: 118 → 50 (57.6% reduction)
- **Reduced feature space for ML**: More efficient training
- **Hierarchical conditioning**: Enables contextual generation
- **Maintains musical quality**: All critical dimensions preserved

---

## 7. Next Steps (Phase 2)

1. **Implement extraction functions** for all 50 parameters
2. **Create validation suite** to test extraction accuracy
3. **Implement migration adapter** (old API → new API)
4. **Test with existing examples** to ensure backward compatibility
5. **Generate hierarchical_parameters.json** schema
6. **Document each parameter** with musical semantics

---

## 8. Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Loss of nuanced control | MEDIUM | Provide backward compatibility layer |
| Genre misclassification | HIGH | Use robust genre detection + manual override |
| Instrument-specific loss | LOW | Focus on musical function, not instrumentation |
| Feature correlation issues | MEDIUM | Run correlation analysis after extraction |
| Extraction algorithm errors | HIGH | Extensive validation with ground truth data |

---

**Status:** Phase 1 Complete - Ready for Phase 2 Implementation
**Next Agent:** Agent 01 (self) - Phase 2: Implementation
**Dependencies:** None (foundational work)
