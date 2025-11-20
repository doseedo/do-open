# Universal Parameter Registry

<<<<<<< HEAD
Total Parameters: 108
=======
Total Parameters: 28
>>>>>>> origin/claude/music-generation-agents-01Gdbm7ZPnSUT25SKLbzQdUX

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

## Genre Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genre.rock.bend_probability` | probability | 0.3 | Probability of guitar bends |
| `genre.rock.power_chord_probability` | probability | 0.7 | Probability of using power chords |
| `genre.rock.vibrato_depth` | continuous | 30.0 | Vibrato depth range (cents) |
| `genre.rock.vibrato_probability` | probability | 0.4 | Probability of vibrato on sustained notes |
