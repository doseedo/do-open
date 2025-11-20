# Universal Parameter Registry

Total Parameters: 88

## Harmony Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `harmony.extensions.use_11ths` | boolean | True | Whether to include 11th extensions |
| `harmony.extensions.use_13ths` | boolean | False | Whether to include 13th extensions |
| `harmony.extensions.use_9ths` | boolean | True | Whether to include 9th extensions |
| `harmony.substitution.modal_interchange_probability` | probability | 0.2 | Probability of modal interchange |
| `harmony.substitution.tritone_probability` | probability | 0.3 | Probability of tritone substitution |
| `harmony.voice_leading.parallel_motion_tolerance` | probability | 0.1 | Tolerance for parallel 5ths/octaves (0.0=strict, 1.0=permissive) |
| `harmony.voice_leading.smoothness` | probability | 0.8 | Preference for smooth voice leading (0.0=none, 1.0=always) |
| `harmony.voicing.density` | integer | 4 | Number of notes in chord (3-7) |
| `harmony.voicing.spread` | probability | 0.5 | How spread out the voicing is (0.0=close, 1.0=wide) |
| `harmony.voicing.type` | categorical | close | Type of chord voicing (close, spread, drop2, etc.) |

## Melody Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `melody.chromaticism.amount` | probability | 0.3 | Amount of chromatic passing tones (0.0=diatonic, 1.0=chromatic) |
| `melody.contour.type` | categorical | arch | Melodic contour shape |
| `melody.intervals.max_leap` | integer | 12 | Maximum melodic leap in semitones |
| `melody.intervals.stepwise_probability` | probability | 0.7 | Probability of stepwise motion (vs leaps) |
| `melody.ornaments.probability` | probability | 0.2 | Probability of adding ornaments (trills, mordents, etc.) |

## Rhythm Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rhythm.microtiming.variation` | integer | 10 | Amount of microtiming humanization (ms) |
| `rhythm.swing.amount` | continuous | 0.67 | Swing ratio (0.5=straight, 0.67=standard swing, 0.75=hard swing) |
| `rhythm.syncopation.probability` | probability | 0.3 | Probability of syncopated rhythms |

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

## Articulation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `articulation.duration.ratio` | probability | 0.9 | Note duration as ratio of full length (0.5=staccato, 1.0=legato) |

## Structure Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structure.development.augmentation_prob` | probability | 0.2 | Probability of rhythmic augmentation (lengthening note values) |
| `structure.development.climax_placement` | probability | 0.75 | Where to place structural climax (0.0=beginning, 1.0=end) |
| `structure.development.diminution_prob` | probability | 0.2 | Probability of rhythmic diminution (shortening note values) |
| `structure.development.fragmentation_prob` | probability | 0.3 | Probability of breaking themes into smaller fragments |
| `structure.development.inversion_prob` | probability | 0.2 | Probability of melodic inversion (mirror contour) |
| `structure.development.motivic_transformation` | probability | 0.5 | Amount of motivic transformation across sections (0.0=none, 1.0=maximum) |
| `structure.development.retrograde_prob` | probability | 0.1 | Probability of retrograde (theme in reverse) |
| `structure.development.sequence_interval` | categorical | 2 | Interval for sequences in semitones (2=whole step, 5=fourth, etc.) |
| `structure.development.sequence_prob` | probability | 0.4 | Probability of sequential repetition at different pitch levels |
| `structure.development.variation_intensity` | probability | 0.5 | Overall intensity of thematic variation (0.0=literal, 1.0=free) |
| `structure.form.asymmetric_sections` | boolean | False | Allow asymmetric phrase lengths (7, 11, 13 bars) |
| `structure.form.breakdown_prob` | probability | 0.3 | Probability of breakdown sections (reduced texture/dynamics) |
| `structure.form.bridge_length` | categorical | 8 | Bridge/middle eight section length in bars |
| `structure.form.chorus_length` | categorical | 8 | Chorus section length in bars |
| `structure.form.coda_type` | categorical | fade | Type of ending/coda |
| `structure.form.grand_pause_prob` | probability | 0.1 | Probability of grand pause (complete silence for dramatic effect) |
| `structure.form.interlude_prob` | probability | 0.2 | Probability of instrumental interlude sections |
| `structure.form.intro_bars` | categorical | 4 | Introduction length in bars (0 = no intro) |
| `structure.form.length_bars` | categorical | 32 | Total form length in bars (standard: 12, 16, 24, 32, 48, 64) |
| `structure.form.outro_bars` | categorical | 4 | Outro/coda length in bars (0 = direct ending) |
| `structure.form.section_markers` | boolean | True | Use explicit rehearsal marks/section boundaries in output |
| `structure.form.shout_chorus` | boolean | True | Include shout chorus (climactic ensemble section) |
| `structure.form.solo_section_length` | categorical | 32 | Solo/improvisation section length in bars |
| `structure.form.solo_trading` | boolean | True | Enable solo trading (4s, 8s, 12s) between instruments |
| `structure.form.stop_time_prob` | probability | 0.2 | Probability of stop-time sections (rhythm section drops out) |
| `structure.form.tag_repetitions` | categorical | 2 | Number of tag ending repetitions (country, gospel style) |
| `structure.form.trading_length` | categorical | 8 | Length of solo trading exchanges in bars (4s, 8s, etc.) |
| `structure.form.type` | categorical | AABA | Primary form structure type (AABA, ABAB, blues, sonata, etc.) |
| `structure.form.vamp_probability` | probability | 0.3 | Probability of extended vamp sections for improvisation/intensity |
| `structure.form.verse_length` | categorical | 8 | Verse section length in bars |
| `structure.repetition.antecedent_consequent` | boolean | True | Use antecedent-consequent phrase pairs (question-answer) |
| `structure.repetition.call_response` | boolean | False | Use call-and-response phrase structure |
| `structure.repetition.contrast_middle` | boolean | True | Create contrast in middle sections (B section different from A) |
| `structure.repetition.exact_repeat_prob` | probability | 0.5 | Probability of exact repetition of sections |
| `structure.repetition.hook_emphasis` | probability | 0.7 | Emphasis on catchy hooks/memorable phrases (0.0=none, 1.0=maximum) |
| `structure.repetition.melodic_cell_recurrence` | probability | 0.6 | How often melodic cells/fragments recur (0.0=rarely, 1.0=constantly) |
| `structure.repetition.motif_recall_prob` | probability | 0.7 | Probability of recalling earlier motifs in later sections |
| `structure.repetition.motivic_consistency` | probability | 0.7 | Consistency of motivic material across sections (0.0=varied, 1.0=consistent) |
| `structure.repetition.ostinato_usage` | probability | 0.4 | Amount of ostinato (repeated pattern) usage (0.0=none, 1.0=extensive) |
| `structure.repetition.parallel_period` | boolean | True | Use parallel period construction (similar phrase beginnings) |
| `structure.repetition.pedal_point_duration` | categorical | 8 | Duration of pedal points (sustained bass notes) in bars |
| `structure.repetition.phrase_grouping` | categorical | regular | How phrases are grouped and related |
| `structure.repetition.phrase_length` | categorical | 4 | Standard phrase length in bars |
| `structure.repetition.return_variation` | probability | 0.4 | Amount of variation when themes return (0.0=exact, 1.0=extensively varied) |
| `structure.repetition.rhythmic_motif_recurrence` | probability | 0.6 | How often rhythmic motifs recur (0.0=rarely, 1.0=constantly) |
| `structure.repetition.riff_based` | boolean | False | Construct piece around repeated riffs (blues/rock style) |
| `structure.repetition.thematic_unity` | probability | 0.7 | Overall thematic unity/coherence (0.0=episodic, 1.0=highly unified) |
| `structure.repetition.theme_return_prob` | probability | 0.9 | Probability of returning to opening theme (recapitulation) |
| `structure.repetition.variation_amount` | probability | 0.4 | How much variation on repeated sections (0.0=minimal, 1.0=extensive) |
| `structure.repetition.varied_repeat_prob` | probability | 0.5 | Probability of varied repetition (with embellishments/changes) |
| `structure.transition.drum_fill_prob` | probability | 0.7 | Probability of drum fill at section transitions |
| `structure.transition.dynamic_change_prob` | probability | 0.7 | Probability of dynamic level change at transitions |
| `structure.transition.key_change_prob` | probability | 0.2 | Probability of modulation/key change at transitions |
| `structure.transition.meter_change_prob` | probability | 0.05 | Probability of meter/time signature change at transitions |
| `structure.transition.pickup_notes` | boolean | True | Use pickup/anacrusis notes before sections |
| `structure.transition.smoothness` | probability | 0.7 | How smooth transitions are (0.0=abrupt/dramatic, 1.0=seamless) |
| `structure.transition.tempo_change_prob` | probability | 0.1 | Probability of tempo change at transitions |
| `structure.transition.texture_change_prob` | probability | 0.6 | Probability of significant texture change at transitions |
| `structure.transition.turnaround_complexity` | continuous | 3.0 | Harmonic complexity of turnarounds (1=simple I-V, 7=complex jazz) |
| `structure.transition.type` | categorical | pivot_chord | Primary transition technique between sections |

## Genre Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genre.rock.bend_probability` | probability | 0.3 | Probability of guitar bends |
| `genre.rock.power_chord_probability` | probability | 0.7 | Probability of using power chords |
| `genre.rock.vibrato_depth` | continuous | 30.0 | Vibrato depth range (cents) |
| `genre.rock.vibrato_probability` | probability | 0.4 | Probability of vibrato on sustained notes |
