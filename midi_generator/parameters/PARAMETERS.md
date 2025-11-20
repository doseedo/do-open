# Universal Parameter Registry

Total Parameters: 101

## Harmony Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `harmony.chromatic.augmented_sixth_probability` | probability | 0.1 | Probability of augmented sixth chords |
| `harmony.chromatic.diminished_passing_probability` | probability | 0.2 | Probability of diminished passing chords |
| `harmony.chromatic.mediant_pattern` | categorical | UCM LCM | Chromatic mediant progression pattern |
| `harmony.chromatic.secondary_dominant_probability` | probability | 0.3 | Probability of using secondary dominants |
| `harmony.extensions.add9_probability` | probability | 0.3 | Probability of add9 chords |
| `harmony.extensions.altered_dominant_probability` | probability | 0.3 | Probability of altered dominants (b9, #9, b13, etc.) |
| `harmony.extensions.seventh_probability` | probability | 0.8 | Probability of using 7th chords |
| `harmony.extensions.sus_probability` | probability | 0.2 | Probability of suspended chords |
| `harmony.extensions.use_11ths` | boolean | True | Whether to include 11th extensions |
| `harmony.extensions.use_13ths` | boolean | False | Whether to include 13th extensions |
| `harmony.extensions.use_9ths` | boolean | True | Whether to include 9th extensions |
| `harmony.modal.aeolian_probability` | probability | 0.14 | Probability of using Aeolian mode |
| `harmony.modal.cadence_type` | categorical | plagal | Type of modal cadence |
| `harmony.modal.dorian_probability` | probability | 0.14 | Probability of using Dorian mode |
| `harmony.modal.interchange_intensity` | probability | 0.2 | How often to borrow from parallel modes (0.0-1.0) |
| `harmony.modal.locrian_probability` | probability | 0.14 | Probability of using Locrian mode |
| `harmony.modal.lydian_probability` | probability | 0.14 | Probability of using Lydian mode |
| `harmony.modal.mixolydian_probability` | probability | 0.14 | Probability of using Mixolydian mode |
| `harmony.modal.phrygian_probability` | probability | 0.14 | Probability of using Phrygian mode |
| `harmony.modal.progression_type` | categorical | characteristic | Type of modal progression |
| `harmony.neo_riemannian.apply_voice_leading` | boolean | True | Apply smooth voice leading to transformations |
| `harmony.neo_riemannian.hexatonic_pole` | integer | 0 | Hexatonic system pole (0-3: Northern, Southern, Eastern, Western) |
| `harmony.neo_riemannian.transformation_sequence` | categorical | PLR | Sequence of transformations (P, L, R) |
| `harmony.substitution.modal_interchange_probability` | probability | 0.2 | Probability of modal interchange |
| `harmony.substitution.tritone_probability` | probability | 0.3 | Probability of tritone substitution |
| `harmony.voice_leading.allow_voice_crossing` | boolean | False | Allow voices to cross |
| `harmony.voice_leading.common_tone_weight` | probability | 0.7 | Weight for maintaining common tones (0.0-1.0) |
| `harmony.voice_leading.max_motion` | integer | 7 | Maximum voice motion in semitones |
| `harmony.voice_leading.parallel_motion_tolerance` | probability | 0.1 | Tolerance for parallel 5ths/octaves (0.0=strict, 1.0=permissive) |
| `harmony.voice_leading.prefer_contrary_motion` | probability | 0.6 | Preference for contrary motion (0.0-1.0) |
| `harmony.voice_leading.smoothness` | probability | 0.8 | Preference for smooth voice leading (0.0=none, 1.0=always) |
| `harmony.voicing.density` | integer | 4 | Number of notes in chord (3-7) |
| `harmony.voicing.spread` | probability | 0.5 | How spread out the voicing is (0.0=close, 1.0=wide) |
| `harmony.voicing.type` | categorical | close | Type of chord voicing (close, spread, drop2, etc.) |
| `transformation.transposition.semitones` | integer | 0 | Transposition in semitones |

## Melody Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `melody.chromaticism.amount` | probability | 0.3 | Amount of chromatic passing tones (0.0=diatonic, 1.0=chromatic) |
| `melody.contour.arch_probability` | probability | 0.35 | Probability of arch contour (low-high-low) |
| `melody.contour.ascending_probability` | probability | 0.2 | Probability of ascending contour |
| `melody.contour.descending_probability` | probability | 0.2 | Probability of descending contour |
| `melody.contour.inverted_arch_probability` | probability | 0.15 | Probability of inverted arch (high-low-high) |
| `melody.contour.type` | categorical | arch | Melodic contour shape |
| `melody.contour.wave_probability` | probability | 0.1 | Probability of wave contour |
| `melody.intervals.fifth_probability` | probability | 0.05 | Probability of fifth leaps |
| `melody.intervals.fourth_probability` | probability | 0.1 | Probability of fourth leaps |
| `melody.intervals.large_leap_probability` | probability | 0.05 | Probability of leaps > fifth |
| `melody.intervals.leap_recovery_probability` | probability | 0.8 | Probability of stepwise recovery after leap |
| `melody.intervals.max_leap` | integer | 12 | Maximum melodic leap in semitones |
| `melody.intervals.second_probability` | probability | 0.5 | Probability of stepwise motion (seconds) |
| `melody.intervals.stepwise_probability` | probability | 0.7 | Probability of stepwise motion (vs leaps) |
| `melody.intervals.third_probability` | probability | 0.2 | Probability of third leaps |
| `melody.intervals.unison_probability` | probability | 0.1 | Probability of repeated notes |
| `melody.motif.augmentation_probability` | probability | 0.15 | Probability of rhythmic augmentation |
| `melody.motif.diminution_probability` | probability | 0.15 | Probability of rhythmic diminution |
| `melody.motif.inversion_probability` | probability | 0.2 | Probability of motif inversion |
| `melody.motif.repetition_probability` | probability | 0.4 | Probability of exact motif repetition |
| `melody.motif.retrograde_probability` | probability | 0.1 | Probability of motif retrograde |
| `melody.motif.sequence_probability` | probability | 0.3 | Probability of motif sequencing |
| `melody.ornaments.appoggiatura_probability` | probability | 0.1 | Probability of appoggiaturas |
| `melody.ornaments.grace_note_probability` | probability | 0.15 | Probability of grace notes |
| `melody.ornaments.mordent_probability` | probability | 0.05 | Probability of mordents |
| `melody.ornaments.probability` | probability | 0.2 | Probability of adding ornaments (trills, mordents, etc.) |
| `melody.ornaments.trill_probability` | probability | 0.05 | Probability of trills |
| `melody.ornaments.turn_probability` | probability | 0.05 | Probability of turns |
| `melody.phrasing.antecedent_consequent_probability` | probability | 0.6 | Probability of antecedent-consequent phrase pairs |
| `melody.phrasing.length_max` | integer | 16 | Maximum phrase length in beats |
| `melody.phrasing.length_min` | integer | 4 | Minimum phrase length in beats |
| `melody.phrasing.rest_probability` | probability | 0.8 | Probability of rest between phrases |

## Rhythm Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rhythm.density.note_density` | continuous | 2.0 | Average notes per beat (0.5-8.0) |
| `rhythm.density.rest_frequency` | probability | 0.2 | Frequency of rests (0.0-1.0) |
| `rhythm.density.sustained_note_probability` | probability | 0.15 | Probability of long sustained notes |
| `rhythm.groove.pocket_depth` | integer | 0 | How far behind/ahead of beat (ms, negative=behind) |
| `rhythm.groove.shuffle_feel` | probability | 0.0 | Shuffle feel intensity (0.0-1.0) |
| `rhythm.metric_modulation.probability` | probability | 0.05 | Probability of metric modulation |
| `rhythm.microtiming.variation` | integer | 10 | Amount of microtiming humanization (ms) |
| `rhythm.polyrhythm.probability` | probability | 0.1 | Probability of polyrhythmic patterns |
| `rhythm.polyrhythm.ratio` | categorical | 3:2 | Polyrhythm ratio (e.g., 3:2, 4:3, 5:4) |
| `rhythm.subdivision.quintuplet_probability` | probability | 0.05 | Probability of quintuplet subdivisions |
| `rhythm.subdivision.sextuplet_probability` | probability | 0.1 | Probability of sextuplet subdivisions |
| `rhythm.subdivision.triplet_probability` | probability | 0.2 | Probability of triplet subdivisions |
| `rhythm.swing.amount` | continuous | 0.67 | Swing ratio (0.5=straight, 0.67=standard swing, 0.75=hard swing) |
| `rhythm.swing.intensity` | probability | 1.0 | How consistently swing is applied (0.0-1.0) |
| `rhythm.swing.ratio` | continuous | 0.67 | Swing ratio (0.5=straight, 0.67=standard, 0.75=hard) |
| `rhythm.syncopation.anticipation_probability` | probability | 0.3 | Probability of anticipating strong beats |
| `rhythm.syncopation.level` | probability | 0.3 | Overall syncopation level (0.0=none, 1.0=maximum) |
| `rhythm.syncopation.offbeat_emphasis_probability` | probability | 0.4 | Probability of emphasizing offbeats |
| `rhythm.syncopation.probability` | probability | 0.3 | Probability of syncopated rhythms |
| `transformation.humanization.amount` | probability | 0.5 | Humanization intensity (0.0=robotic, 1.0=very human) |
| `transformation.humanization.timing_variance` | integer | 15 | Timing variance in milliseconds |
| `transformation.tempo.change_ratio` | continuous | 1.0 | Tempo change ratio (0.5=half, 2.0=double) |

## Bass Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bass.style.walking_probability` | probability | 0.8 | Probability of walking bass line |

## Drums Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drums.kick.velocity_max` | velocity | 110 | Maximum kick drum velocity |
| `drums.kick.velocity_min` | velocity | 80 | Minimum kick drum velocity |

## Dynamics Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dynamics.velocity.base` | velocity | 80 | Base velocity for notes |
| `dynamics.velocity.variation` | integer | 20 | Amount of velocity variation (+/-) |
| `transformation.dynamics.scaling` | continuous | 1.0 | Dynamic scaling factor (0.5=softer, 2.0=louder) |
| `transformation.humanization.velocity_variance` | integer | 10 | Velocity variance (+/-) |

## Articulation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `articulation.duration.ratio` | probability | 0.9 | Note duration as ratio of full length (0.5=staccato, 1.0=legato) |

## Genre Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genre.rock.bend_probability` | probability | 0.3 | Probability of guitar bends |
| `genre.rock.power_chord_probability` | probability | 0.7 | Probability of using power chords |
| `genre.rock.vibrato_depth` | continuous | 30.0 | Vibrato depth range (cents) |
| `genre.rock.vibrato_probability` | probability | 0.4 | Probability of vibrato on sustained notes |
