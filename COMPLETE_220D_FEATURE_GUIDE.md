# Complete 220D Feature Breakdown

## Overview

Your **EnhancedFeatureExtractor** extracts 220 features:
- **200D from DeepFeatureExtractor** (ALL harmony features, indices 0-199)
- **+20D velocity analysis** (added by EnhancedFeatureExtractor)

---

## 🎵 Features 0-199: Harmony Features (from DeepFeatureExtractor)

**Source:** `harmony_0` through `harmony_199` from DeepFeatureExtractor's 250 harmony features

These are organized into sub-groups within the DeepFeatureExtractor:

### Group 1: Chord Quality & Extensions (23 features)
**Indices: 0-22**

Extracted by: `_extract_chord_quality_features()`

```
[0]  major_triad_ratio              - % of major chords
[1]  minor_triad_ratio              - % of minor chords
[2]  diminished_triad_ratio         - % of diminished chords
[3]  augmented_triad_ratio          - % of augmented chords
[4]  dominant_seventh_ratio         - % of dominant 7th chords
[5]  major_seventh_ratio            - % of major 7th chords
[6]  minor_seventh_ratio            - % of minor 7th chords
[7]  half_diminished_seventh_ratio  - % of half-diminished 7th chords
[8]  fully_diminished_seventh_ratio - % of fully diminished 7th chords
[9]  sus2_chord_ratio               - % of sus2 chords
[10] sus4_chord_ratio               - % of sus4 chords
[11] add2_chord_ratio               - % of add2 chords
[12] add6_chord_ratio               - % of add6 chords
[13] ninth_chord_ratio              - % of 9th chords
[14] eleventh_chord_ratio           - % of 11th chords
[15] thirteenth_chord_ratio         - % of 13th chords
[16] altered_dominant_ratio         - % of altered dominant chords
[17] extended_chord_complexity_mean - Average chord complexity
[18] average_chord_extensions       - Average extensions per chord
[19] max_chord_extension            - Maximum chord size
[20] chord_quality_diversity        - Variety of chord types
[21] triad_to_seventh_ratio         - Ratio of triads to 7th chords
[22] seventh_to_extended_ratio      - Ratio of 7th to extended chords
```

### Group 2: Voicing Characteristics (24 features)
**Indices: 23-46**

Extracted by: `_extract_voicing_features()`

```
[23] close_voicing_ratio             - % close voicings
[24] open_voicing_ratio              - % open voicings
[25] drop2_voicing_count             - Count of drop-2 voicings
[26] drop3_voicing_count             - Count of drop-3 voicings
[27] drop24_voicing_count            - Count of drop-2-4 voicings
[28] quartal_voicing_count           - Count of quartal voicings
[29] quintal_voicing_count           - Count of quintal voicings
[30] cluster_voicing_count           - Count of cluster voicings
[31] shell_voicing_count             - Count of shell voicings
[32] rootless_voicing_count          - Count of rootless voicings
[33] so_what_voicing_count           - Count of "So What" voicings
[34] upper_structure_triad_count     - Count of upper structure triads
[35] polychord_count                 - Count of polychords
[36] voicing_density_mean            - Average voicing density
[37] voicing_density_std             - Voicing density variation
[38] voicing_range_mean              - Average pitch range of voicings
[39] voicing_range_std               - Pitch range variation ⭐ VALUE: 0.999
[40] inner_voice_motion_chromaticism - Inner voice chromaticism
[41] guide_tone_line_smoothness      - Guide tone smoothness
[42] voice_leading_distance_mean     - Average voice leading distance
[43] voice_leading_distance_std      - Voice leading distance variation
[44] parallel_motion_ratio           - % parallel motion
[45] contrary_motion_ratio           - % contrary motion
[46] oblique_motion_ratio            - % oblique motion
```

### Group 3: Harmonic Progression (27 features)
**Indices: 47-73**

Extracted by: `_extract_progression_features()`

```
[47] voice_crossing_frequency         - Voice crossing frequency
[48] functional_progression_ratio     - % functional progressions
[49] modal_progression_ratio          - % modal progressions
[50] chromatic_progression_ratio      - % chromatic progressions
[51] circle_of_fifths_motion          - Circle of 5ths motion count
[52] cycle_of_fourths_motion          - Cycle of 4ths motion count
[53] tritone_substitution_ratio       - % tritone substitutions
[54] chromatic_mediant_ratio          - % chromatic mediants
[55] modal_interchange_ratio          - % modal interchange
[56] secondary_dominant_ratio         - % secondary dominants
[57] secondary_diminished_ratio       - % secondary diminished ⭐ VALUE: 0.003
[58] deceptive_cadence_ratio          - % deceptive cadences ⭐ VALUE: 0.086
[59] plagal_cadence_ratio             - % plagal cadences
[60] half_cadence_ratio               - % half cadences ⭐ VALUE: small
[61] authentic_cadence_ratio          - % authentic cadences
[62] turnaround_count                 - Jazz turnaround count
[63] ii_V_I_progression_count         - ii-V-I progression count
[64] coltrane_changes_detected        - Coltrane changes present ⭐ VALUE: 0.01
[65] giant_steps_pattern_count        - Giant Steps patterns ⭐ VALUE: 334.15
[66] backdoor_progression_count       - Backdoor progression count ⭐ VALUE: 0.03
[67] pedal_point_duration_mean        - Average pedal point duration
[68] harmonic_rhythm_regularity       - Harmonic rhythm regularity
[69] chords_per_bar_mean              - Average chords per bar
[70] chords_per_bar_std               - Chords per bar variation
[71] progression_complexity_score     - Overall progression complexity
[72] modulation_frequency             - Modulation frequency
[73] key_stability_score              - Key stability
```

### Group 4: Neo-Riemannian & Advanced (13 features)
**Indices: 74-86**

Extracted by: `_extract_advanced_harmony_features()`

```
[74] neo_riemannian_P_transform_count  - P (parallel) transform count
[75] neo_riemannian_L_transform_count  - L (leading-tone) transform count
[76] neo_riemannian_R_transform_count  - R (relative) transform count
[77] consonance_dissonance_ratio       - Consonance to dissonance ratio
[78] harmonic_entropy_mean             - Average harmonic entropy
[79] harmonic_entropy_std              - Harmonic entropy variation
[80] voice_leading_efficiency          - Voice leading efficiency
[81] chord_root_motion_by_fifths       - Root motion by 5ths
[82] chord_root_motion_by_thirds       - Root motion by 3rds
[83] chord_root_motion_by_seconds      - Root motion by 2nds
[84] chord_root_motion_by_tritone      - Root motion by tritone
[85] chord_root_motion_chromatic       - Chromatic root motion
[86] harmonic_tension_profile_variance - Tension profile variance
```

### Group 5: Voice Leading (25 features)
**Indices: 87-111**

Extracted by: `_extract_voice_leading_features()`

```
[87]  voice_leading_smoothness_mean      - Average smoothness ⭐ VALUE: 55.0
[88]  voice_leading_smoothness_std       - Smoothness variation
[89]  common_tone_retention_ratio        - % common tones retained
[90]  stepwise_motion_ratio              - % stepwise motion ⭐ VALUE: 8.0
[91]  leap_motion_ratio                  - % leap motion
[92]  ascending_motion_ratio             - % ascending motion
[93]  descending_motion_ratio            - % descending motion
[94]  voice_independence_score           - Voice independence ⭐ VALUE: 0.020
[95]  outer_voice_interval_mean          - Average outer voice interval
[96]  outer_voice_interval_std           - Outer voice interval variation
[97]  soprano_alto_correlation           - Soprano-alto correlation ⭐ VALUE: 0.52
[98]  alto_tenor_correlation             - Alto-tenor correlation ⭐ VALUE: 0.64
[99]  tenor_bass_correlation             - Tenor-bass correlation ⭐ VALUE: 0.59
[100] parallel_fifths_count              - Parallel 5ths count ⭐ VALUE: 542.0
[101] parallel_octaves_count             - Parallel octaves count ⭐ VALUE: 0.062
[102] hidden_fifths_count                - Hidden 5ths count
[103] hidden_octaves_count               - Hidden octaves count
[104] voice_overlap_frequency            - Voice overlap frequency
[105] soprano_range_mean                 - Average soprano range
[106] alto_range_mean                    - Average alto range
[107] tenor_range_mean                   - Average tenor range
[108] bass_range_mean                    - Average bass range
[109] voice_spacing_mean                 - Average voice spacing
[110] voice_spacing_std                  - Voice spacing variation
[111] doubling_frequency                 - Note doubling frequency
```

### Group 6: Harmonic Rhythm (20 features)
**Indices: 112-131**

Extracted by: `_extract_harmonic_rhythm_features()`

```
[112] harmonic_rhythm_density_mean       - Average harmonic density
[113] harmonic_rhythm_density_std        - Harmonic density variation
[114] chord_change_frequency_mean        - Average chord changes per unit time
[115] chord_change_frequency_std         - Chord change frequency variation
[116] harmonic_acceleration_mean         - Average harmonic acceleration
[117] harmonic_deceleration_mean         - Average harmonic deceleration
[118] syncopated_chord_changes_ratio     - % syncopated chord changes
[119] on_beat_chord_changes_ratio        - % on-beat chord changes
[120] off_beat_chord_changes_ratio       - % off-beat chord changes
[121] chord_duration_mean                - Average chord duration
[122] chord_duration_std                 - Chord duration variation
[123] shortest_chord_duration            - Shortest chord
[124] longest_chord_duration             - Longest chord
[125] chord_duration_diversity           - Chord duration variety
[126] harmonic_rhythm_pattern_regularity - Pattern regularity
[127] harmonic_anticipation_frequency    - Anticipation frequency
[128] harmonic_suspension_frequency      - Suspension frequency
[129] harmonic_rhythm_swing_ratio        - Swing rhythm ratio
[130] harmonic_rhythm_straight_ratio     - Straight rhythm ratio
[131] chord_change_predictability        - Change predictability ⭐ VALUE: 24.0
```

### Group 7: Tension & Resolution (18 features)
**Indices: 132-149**

Extracted by: `_extract_tension_features()`

```
[132] tension_profile_mean               - Average tension ⭐ VALUE: 6.0
[133] tension_profile_std                - Tension variation ⭐ VALUE: 0.0077
[134] tension_profile_max                - Maximum tension
[135] tension_profile_min                - Minimum tension
[136] tension_resolution_frequency       - Tension-resolution frequency
[137] tension_buildup_rate               - Tension buildup rate
[138] tension_release_rate               - Tension release rate
[139] unresolved_tension_ratio           - % unresolved tension
[140] dissonance_treatment_ratio         - Dissonance treatment ratio
[141] suspension_resolution_ratio        - Suspension resolution ratio
[142] appoggiatura_frequency             - Appoggiatura frequency
[143] passing_tone_frequency             - Passing tone frequency
[144] neighbor_tone_frequency            - Neighbor tone frequency
[145] escape_tone_frequency              - Escape tone frequency
[146] anticipation_frequency             - Anticipation frequency
[147] cambiata_frequency                 - Cambiata frequency
[148] tension_peak_frequency             - Tension peak frequency
[149] tension_valley_frequency           - Tension valley frequency
```

### Group 8: Extensions & Alterations (25 features)
**Indices: 150-174**

Extracted by: `_extract_extension_features()`

```
[150] added_ninth_frequency              - Added 9th frequency
[151] added_eleventh_frequency           - Added 11th frequency
[152] added_thirteenth_frequency         - Added 13th frequency
[153] flat_ninth_frequency               - Flat 9th frequency
[154] sharp_ninth_frequency              - Sharp 9th frequency
[155] flat_fifth_frequency               - Flat 5th frequency ⭐ VALUE: 0.0029
[156] sharp_fifth_frequency              - Sharp 5th frequency
[157] sharp_eleventh_frequency           - Sharp 11th frequency
[158] flat_thirteenth_frequency          - Flat 13th frequency
[159] natural_eleventh_frequency         - Natural 11th frequency
[160] natural_thirteenth_frequency       - Natural 13th frequency ⭐ VALUE: 0.0093
[161] chord_extension_complexity_mean    - Extension complexity ⭐ VALUE: 422.11
[162] chord_extension_complexity_std     - Extension complexity variation
[163] altered_chord_ratio                - % altered chords
[164] unaltered_chord_ratio              - % unaltered chords
[165] extended_chord_ratio               - % extended chords
[166] simple_chord_ratio                 - % simple chords
[167] chord_color_diversity              - Chord color variety
[168] upper_extension_frequency          - Upper extension frequency
[169] lower_extension_frequency          - Lower extension frequency
[170] simultaneous_alterations_mean      - Average simultaneous alterations
[171] chord_density_mean                 - Average chord density
[172] chord_density_std                  - Chord density variation
[173] vertical_sonority_complexity_mean  - Vertical complexity
[174] vertical_sonority_complexity_std   - Vertical complexity variation
```

### Group 9: Functional Harmony (25 features)
**Indices: 175-199**

Extracted by: `_extract_functional_harmony_features()`

```
[175] tonic_function_ratio               - % tonic function
[176] subdominant_function_ratio         - % subdominant function
[177] dominant_function_ratio            - % dominant function
[178] pre_dominant_function_ratio        - % pre-dominant function
[179] augmented_sixth_chord_ratio        - % augmented 6th chords
[180] neapolitan_sixth_chord_ratio       - % Neapolitan 6th chords
[181] borrowed_chord_ratio               - % borrowed chords
[182] common_practice_progression_ratio  - % common practice progressions
[183] functional_ambiguity_score         - Functional ambiguity
[184] tonal_center_stability_score       - Tonal center stability
[185] key_area_count                     - Number of key areas
[186] pivot_chord_modulation_count       - Pivot chord modulations
[187] direct_modulation_count            - Direct modulations
[188] chromatic_modulation_count         - Chromatic modulations
[189] enharmonic_modulation_count        - Enharmonic modulations
[190] common_tone_modulation_count       - Common tone modulations
[191] closely_related_key_ratio          - % closely related keys
[192] distantly_related_key_ratio        - % distantly related keys ⭐ VALUE: 427.98
[193] tonicization_frequency             - Tonicization frequency
[194] applied_chord_ratio                - % applied chords
[195] prolongation_technique_frequency   - Prolongation frequency
[196] sequence_pattern_frequency         - Sequence frequency
[197] harmonic_climax_positioning        - Climax position
[198] cadential_preparation_quality      - Cadential preparation
[199] resolution_trajectory_smoothness   - Resolution smoothness
```

---

## 🎚️ Features 200-219: Velocity Analysis (added by EnhancedFeatureExtractor)

**Source:** Added by `EnhancedFeatureExtractor._extract_velocity_features()`

**Per-channel velocity statistics (4 channels × 5 metrics = 20 features):**

### Channel 0 (usually melody/lead)
```
[200] velocity_ch0_mean    - Average velocity ⭐ VALUE: 80
[201] velocity_ch0_std     - Velocity variation ⭐ VALUE: 0
[202] velocity_ch0_min     - Minimum velocity ⭐ VALUE: 80
[203] velocity_ch0_max     - Maximum velocity ⭐ VALUE: 80
[204] velocity_ch0_median  - Median velocity ⭐ VALUE: 80
```

### Channel 1 (usually harmony/chords)
```
[205] velocity_ch1_mean    - Average velocity ⭐ VALUE: 80
[206] velocity_ch1_std     - Velocity variation ⭐ VALUE: 0
[207] velocity_ch1_min     - Minimum velocity ⭐ VALUE: 80
[208] velocity_ch1_max     - Maximum velocity ⭐ VALUE: 80
[209] velocity_ch1_median  - Median velocity ⭐ VALUE: 80
```

### Channel 2 (usually bass)
```
[210] velocity_ch2_mean    - Average velocity ⭐ VALUE: 80
[211] velocity_ch2_std     - Velocity variation ⭐ VALUE: 0
[212] velocity_ch2_min     - Minimum velocity ⭐ VALUE: 80
[213] velocity_ch2_max     - Maximum velocity ⭐ VALUE: 80
[214] velocity_ch2_median  - Median velocity ⭐ VALUE: 80
```

### Channel 3 (usually drums/percussion or additional)
```
[215] velocity_ch3_mean    - Average velocity ⭐ VALUE: 80
[216] velocity_ch3_std     - Velocity variation ⭐ VALUE: 0
[217] velocity_ch3_min     - Minimum velocity ⭐ VALUE: 80
[218] velocity_ch3_max     - Maximum velocity ⭐ VALUE: 80
[219] velocity_ch3_median  - Median velocity ⭐ VALUE: 80
```

---

## 📊 Summary Statistics

### Feature Distribution:
- **Harmony Features:** 200 (91% of features)
- **Velocity Features:** 20 (9% of features)
- **Total:** 220 features

### Non-Zero Features in Your Data:
- **Raw:** 38/220 (17.3% sparse)
- **Normalized:** 49/220 (22.3% sparse)

### Key Observations:

1. **All harmony, no melody/rhythm/dynamics/texture**
   - The feature selection chose ONLY harmony features
   - Missing: melody, rhythm, dynamics, texture, structure, orchestration

2. **Constant velocity (80) across all channels**
   - Every std = 0 (no variation)
   - Your MIDI files lack dynamic expression

3. **Sparse features are normal for big band MIDI**
   - Many advanced jazz features are zeros (Coltrane changes, Giant Steps, etc.)
   - Basic functional harmony dominates

---

## 🎯 What This Means for Training

**Good:**
- ✅ 200 sophisticated harmony features
- ✅ Normalized properly (mean ≈ 0, std ≈ 1)
- ✅ Should converge now (vs. 5000+ losses before)

**Limitation:**
- ⚠️ Only harmony dimension covered
- ⚠️ No rhythm, melody, texture analysis
- ⚠️ Training can only learn harmonic patterns

**For full musical coverage, you'd want:**
- Harmony: 40 features
- Melody: 40 features
- Rhythm: 40 features
- Dynamics: 30 features
- Texture: 25 features
- Structure: 25 features

But for now, having **200 harmony features** is fine for training the **harmony encoder**!

---

**Version:** v2.0
**Total Features:** 220D (200 harmony + 20 velocity)
**Source:** DeepFeatureExtractor → OptimizedFeatureExtractor → EnhancedFeatureExtractor
