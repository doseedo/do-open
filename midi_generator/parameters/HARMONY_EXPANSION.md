# Harmony Deep Expansion - Complete Parameter Documentation

**Agent 3: Harmony Deep Expansion Specialist**

Total Harmony Parameters: 129

---

## Advanced Voicing Techniques

| Parameter | Type | Default | Range/Options | Description |
|-----------|------|---------|---------------|-------------|
| `harmony.voicing.quartal_probability` | probability | 0.1 | [0.0, 1.0] | Probability of using quartal voicings (stacked fourths) |
| `harmony.voicing.quintal_probability` | probability | 0.05 | [0.0, 1.0] | Probability of using quintal voicings (stacked fifths) |
| `harmony.voicing.cluster_probability` | probability | 0.05 | [0.0, 1.0] | Probability of using tone clusters (closely spaced dissonances) |
| `harmony.voicing.cluster_width` | categorical | 3 | [2, 3, 4, 5] | Width of tone clusters in semitones |
| `harmony.voicing.upper_structure_triad` | probability | 0.2 | [0.0, 1.0] | Probability of adding upper structure triads over bass |
| `harmony.voicing.polychord_probability` | probability | 0.1 | [0.0, 1.0] | Probability of using polychords (two simultaneous triads) |
| `harmony.voicing.bitonal_probability` | probability | 0.02 | [0.0, 1.0] | Probability of true bitonality (two independent key centers) |
| `harmony.voicing.shell_voicing_prob` | probability | 0.3 | [0.0, 1.0] | Probability of using shell voicings (root, 3rd, 7th only) |
| `harmony.voicing.rootless_prob` | probability | 0.5 | [0.0, 1.0] | Probability of rootless voicings (3rd, 7th, extensions, no root) |
| `harmony.voicing.so_what_voicing` | probability | 0.15 | [0.0, 1.0] | Probability of 'So What' voicings (3 perfect 4ths + major 3rd) |
| `harmony.voicing.drop2_prob` | probability | 0.4 | [0.0, 1.0] | Probability of drop-2 voicings (2nd voice from top dropped octave) |
| `harmony.voicing.drop3_prob` | probability | 0.2 | [0.0, 1.0] | Probability of drop-3 voicings (3rd voice from top dropped octave) |
| `harmony.voicing.drop24_prob` | probability | 0.1 | [0.0, 1.0] | Probability of drop-2-and-4 voicings (both 2nd and 4th dropped) |
| `harmony.voicing.spread_voicing_prob` | probability | 0.3 | [0.0, 1.0] | Probability of spread voicings (wide intervals between voices) |
| `harmony.voicing.close_position_prob` | probability | 0.4 | [0.0, 1.0] | Probability of close position voicings (all voices within octave) |
| `harmony.voicing.open_position_prob` | probability | 0.3 | [0.0, 1.0] | Probability of open position voicings (voices span > 1 octave) |
| `harmony.voicing.doubled_root` | boolean | True | - | Whether to double the root in voicings |
| `harmony.voicing.doubled_third` | boolean | False | - | Whether to double the third in voicings |
| `harmony.voicing.doubled_fifth` | boolean | True | - | Whether to double the fifth in voicings |
| `harmony.voicing.doubled_seventh` | boolean | False | - | Whether to double the seventh in voicings |
| `harmony.voicing.omit_root` | boolean | True | - | Whether to omit root (assumes bass provides it) |
| `harmony.voicing.omit_fifth` | boolean | True | - | Whether to omit fifth (least important chord tone) |
| `harmony.voicing.guide_tone_emphasis` | probability | 0.8 | [0.0, 1.0] | How much to emphasize guide tones (3rd and 7th) |
| `harmony.voicing.color_tone_emphasis` | probability | 0.6 | [0.0, 1.0] | How much to emphasize color tones (9th, 11th, 13th) |
| `harmony.voicing.avoid_note_handling` | categorical | avoid | ['include', 'avoid', 'approach_from_below', 'approach_from_above'] | How to handle avoid notes (e.g., 4th over major chord) |

## Advanced Chord Extensions

| Parameter | Type | Default | Range/Options | Description |
|-----------|------|---------|---------------|-------------|
| `harmony.extension.ninth_usage` | probability | 0.7 | [0.0, 1.0] | Overall probability of including 9th extensions |
| `harmony.extension.flat_ninth_prob` | probability | 0.3 | [0.0, 1.0] | Probability of b9 (typically on dominant chords) |
| `harmony.extension.sharp_ninth_prob` | probability | 0.2 | [0.0, 1.0] | Probability of #9 (Hendrix chord when on dominant) |
| `harmony.extension.eleventh_usage` | probability | 0.5 | [0.0, 1.0] | Overall probability of including 11th extensions |
| `harmony.extension.sharp_eleventh_prob` | probability | 0.4 | [0.0, 1.0] | Probability of #11 (Lydian sound) |
| `harmony.extension.thirteenth_usage` | probability | 0.4 | [0.0, 1.0] | Overall probability of including 13th extensions |
| `harmony.extension.flat_thirteenth_prob` | probability | 0.2 | [0.0, 1.0] | Probability of b13 (enharmonic with #5) |
| `harmony.extension.add2_prob` | probability | 0.15 | [0.0, 1.0] | Probability of add2 chords (1-2-3-5, no 7th) |
| `harmony.extension.add4_prob` | probability | 0.05 | [0.0, 1.0] | Probability of add4 chords (1-3-4-5, creates dissonance) |
| `harmony.extension.add6_prob` | probability | 0.25 | [0.0, 1.0] | Probability of add6 chords (major 6th, no 7th) |
| `harmony.extension.sus2_prob` | probability | 0.2 | [0.0, 1.0] | Probability of sus2 chords (1-2-5, no 3rd) |
| `harmony.extension.sus4_prob` | probability | 0.3 | [0.0, 1.0] | Probability of sus4 chords (1-4-5, no 3rd) |
| `harmony.extension.augmented_prob` | probability | 0.1 | [0.0, 1.0] | Probability of augmented chords (raised 5th) |
| `harmony.extension.diminished_prob` | probability | 0.15 | [0.0, 1.0] | Probability of fully diminished 7th chords |
| `harmony.extension.half_diminished_prob` | probability | 0.25 | [0.0, 1.0] | Probability of half-diminished 7th chords (m7b5) |
| `harmony.extension.altered_dominant_prob` | probability | 0.3 | [0.0, 1.0] | Probability of fully altered dominants (b9,#9,b5,b13) |
| `harmony.extension.lydian_dominant_prob` | probability | 0.25 | [0.0, 1.0] | Probability of Lydian dominant (7#11) |
| `harmony.extension.phrygian_dominant_prob` | probability | 0.15 | [0.0, 1.0] | Probability of Phrygian dominant (7b9b13) |
| `harmony.extension.dominant_bebop_prob` | probability | 0.4 | [0.0, 1.0] | Probability of bebop dominant scale usage |
| `harmony.extension.stacked_intervals` | categorical | thirds | ['thirds', 'fourths', 'fifths', 'mixed'] | Primary interval for chord construction |

## Advanced Progressions

| Parameter | Type | Default | Range/Options | Description |
|-----------|------|---------|---------------|-------------|
| `harmony.progression.tritone_substitution` | probability | 0.3 | [0.0, 1.0] | Probability of tritone substitution (substitute dominant) |
| `harmony.progression.chromatic_mediant` | probability | 0.2 | [0.0, 1.0] | Probability of chromatic mediant relationships |
| `harmony.progression.modal_interchange` | probability | 0.4 | [0.0, 1.0] | Probability of borrowing chords from parallel modes |
| `harmony.progression.secondary_dominant_prob` | probability | 0.5 | [0.0, 1.0] | Probability of secondary dominants (V7/x) |
| `harmony.progression.secondary_diminished_prob` | probability | 0.2 | [0.0, 1.0] | Probability of secondary diminished chords |
| `harmony.progression.deceptive_cadence_prob` | probability | 0.2 | [0.0, 1.0] | Probability of deceptive cadences (V-vi instead of V-I) |
| `harmony.progression.plagal_cadence_prob` | probability | 0.3 | [0.0, 1.0] | Probability of plagal cadences (IV-I, 'Amen' cadence) |
| `harmony.progression.half_cadence_prob` | probability | 0.25 | [0.0, 1.0] | Probability of half cadences (ending on V) |
| `harmony.progression.turnaround_type` | categorical | I-VI-II-V | ['I-VI-II-V', 'I-III-VI-II-V', 'rhythm_changes', 'bird_blues', 'custom'] | Type of turnaround progression |
| `harmony.progression.cycle_movement` | categorical | fifths | ['fifths', 'fourths', 'thirds', 'chromatic'] | Type of cycle movement for progressions |
| `harmony.progression.coltrane_changes` | boolean | False | - | Use Coltrane changes (major thirds cycle) |
| `harmony.progression.giant_steps_prob` | probability | 0.05 | [0.0, 1.0] | Probability of Giant Steps substitution patterns |
| `harmony.progression.backdoor_progression` | probability | 0.25 | [0.0, 1.0] | Probability of backdoor progression (bVII7-I) |
| `harmony.progression.pedal_point_harmony` | probability | 0.2 | [0.0, 1.0] | Probability of using pedal point (sustained bass note) |
| `harmony.progression.static_harmony_duration` | categorical | 4 | [2, 4, 8, 16] | Duration of static harmony in bars |
| `harmony.progression.harmonic_rhythm` | categorical | medium | ['slow', 'medium', 'fast', 'variable'] | Overall harmonic rhythm (rate of chord change) |
| `harmony.progression.chords_per_bar` | continuous | 1.0 | [0.25, 4.0] | Average number of chords per bar |
| `harmony.progression.reharmonization_prob` | probability | 0.3 | [0.0, 1.0] | Probability of reharmonizing existing progressions |
| `harmony.progression.polytonality` | probability | 0.0 | [0.0, 1.0] | Degree of polytonal writing (multiple keys simultaneously) |
| `harmony.progression.parallel_harmony` | probability | 0.2 | [0.0, 1.0] | Probability of parallel chord motion (planing) |
| `harmony.progression.planing_type` | categorical | none | ['none', 'diatonic', 'chromatic', 'quartal'] | Type of parallel motion |
| `harmony.progression.upper_structure_reharmonization` | probability | 0.25 | [0.0, 1.0] | Probability of upper structure reharmonization |
| `harmony.progression.bass_cliche_prob` | probability | 0.2 | [0.0, 1.0] | Probability of bass cliché (chromatic bass line) |
| `harmony.progression.chromatic_bass_motion` | probability | 0.3 | [0.0, 1.0] | Degree of chromatic bass motion in progressions |
| `harmony.progression.contrary_motion_bass` | probability | 0.6 | [0.0, 1.0] | Preference for contrary motion between bass and melody |

## Neo-Riemannian & Advanced Theory

| Parameter | Type | Default | Range/Options | Description |
|-----------|------|---------|---------------|-------------|
| `harmony.neo_riemannian.parallel_transform` | probability | 0.15 | [0.0, 1.0] | Probability of Parallel (P) transform (major↔minor, common root) |
| `harmony.neo_riemannian.relative_transform` | probability | 0.2 | [0.0, 1.0] | Probability of Relative (R) transform (major↔minor, minor 3rd) |
| `harmony.neo_riemannian.leading_tone_transform` | probability | 0.1 | [0.0, 1.0] | Probability of Leading-tone (L) transform (major↔minor, semitone) |
| `harmony.neo_riemannian.slide_transform` | probability | 0.1 | [0.0, 1.0] | Probability of Slide (S) transform (parallel move keeping 3rd/5th) |
| `harmony.neo_riemannian.hexatonic_pole_probability` | probability | 0.08 | [0.0, 1.0] | Probability of hexatonic pole relationships (L-P cycle) |
| `harmony.set_theory.pitch_class_sets` | boolean | False | - | Use pitch class set theory for chord construction |
| `harmony.set_theory.interval_vector_preference` | array_int | [0, 0, 0, 0, 0, 0] | - | Preferred interval vector [c1,c2,c3,c4,c5,c6] for pc sets |
| `harmony.set_theory.z_relation_exploration` | probability | 0.0 | [0.0, 1.0] | Probability of exploring Z-related sets (same interval vector) |
| `harmony.set_theory.aggregate_completion` | boolean | False | - | Ensure all 12 pitch classes used before repeating |
| `harmony.set_theory.combinatoriality` | probability | 0.0 | [0.0, 1.0] | Degree of combinatorial relationships in set usage |
| `harmony.microtonality.quarter_tone_usage` | probability | 0.0 | [0.0, 1.0] | Probability of using quarter tones (50 cent intervals) |
| `harmony.microtonality.just_intonation` | boolean | False | - | Use just intonation (pure ratios) instead of equal temperament |
| `harmony.microtonality.cent_deviation` | continuous | 0.0 | [0.0, 50.0] | Maximum pitch deviation in cents from equal temperament |
| `harmony.spectral.harmonic_series_chords` | probability | 0.0 | [0.0, 1.0] | Probability of using harmonic series-based chords |

## Specialized Techniques

| Parameter | Type | Default | Range/Options | Description |
|-----------|------|---------|---------------|-------------|
| `harmony.advanced.negative_harmony_prob` | probability | 0.0 | [0.0, 1.0] | Probability of negative harmony transformation |
| `harmony.advanced.slash_chord_prob` | probability | 0.3 | [0.0, 1.0] | Probability of slash chords (chord over alternate bass) |
| `harmony.advanced.first_inversion_prob` | probability | 0.3 | [0.0, 1.0] | Probability of first inversion chords (third in bass) |
| `harmony.advanced.second_inversion_prob` | probability | 0.15 | [0.0, 1.0] | Probability of second inversion chords (fifth in bass) |
| `harmony.advanced.extended_tertian_prob` | probability | 0.2 | [0.0, 1.0] | Probability of chords with 9th, 11th, and 13th simultaneously |
| `harmony.advanced.ragam_influence` | probability | 0.0 | [0.0, 1.0] | Degree of Indian ragam/raga influence on harmony |
| `harmony.advanced.maqam_influence` | probability | 0.0 | [0.0, 1.0] | Degree of Arabic maqam influence on harmony |
| `harmony.advanced.blue_note_influence` | probability | 0.3 | [0.0, 1.0] | Degree of blue note influence (b3, b5, b7 in major context) |
| `harmony.advanced.superimposition_prob` | probability | 0.1 | [0.0, 1.0] | Probability of chord superimposition (playing 'outside') |
| `harmony.advanced.harmonic_complexity_target` | probability | 0.5 | [0.0, 1.0] | Target harmonic complexity (0.0=simple, 1.0=maximum) |
