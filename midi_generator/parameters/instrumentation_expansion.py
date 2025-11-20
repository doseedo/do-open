"""
Instrumentation Parameter Expansion - Agent 1: Strategic Parameter Expansion Coordinator
========================================================================================

Complete parameter definitions for ALL instrumentation categories (0 → 80 parameters)

This addresses the CRITICAL GAP in the system where instrumentation has 0 parameters.
This expansion adds 80 comprehensive parameters covering:
- Piano: 20 parameters
- Bass: 15 parameters
- Drums: 25 parameters
- Brass: 10 parameters
- Strings: 10 parameters

Phase 1 Target: 165 → 515 parameters
Current Expansion: +80 instrumentation parameters

Author: Agent 1 - Strategic Parameter Expansion Coordinator
License: MIT
"""

# Handle both relative and absolute imports
try:
    from .universal_registry import (
        ParameterDefinition, ParameterType, ParameterCategory,
        MusicalImpact, REGISTRY
    )
except ImportError:
    from universal_registry import (
        ParameterDefinition, ParameterType, ParameterCategory,
        MusicalImpact, REGISTRY
    )


def register_piano_parameters():
    """
    Register comprehensive piano parameters (20 params)

    Musical Rationale:
    Piano is the most versatile harmonic/melodic instrument in modern music.
    These parameters control voicing, articulation, pedaling, and stylistic
    elements that define piano performance across genres.
    """

    # ========================================================================
    # Piano Voicing and Density
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="piano_voicing_density",
        full_path="instrumentation.piano.voicing_density",
        description="Number of simultaneous notes in piano voicings (1=single notes, 7=dense clusters)",
        param_type=ParameterType.INTEGER,
        default_value=4,
        min_value=1,
        max_value=7,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "classical", "pop", "rock", "gospel"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_spread_type",
        full_path="instrumentation.piano.spread_type",
        description="Spatial arrangement of chord notes across piano range",
        param_type=ParameterType.CATEGORICAL,
        options=["close", "open", "drop2", "drop3", "drop24", "quartal"],
        default_value="close",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "classical", "contemporary"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_left_hand_style",
        full_path="instrumentation.piano.left_hand_style",
        description="Accompaniment style for left hand",
        param_type=ParameterType.CATEGORICAL,
        options=["stride", "shell", "rootless", "block", "walking"],
        default_value="shell",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "stride", "bebop", "latin", "gospel"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_right_hand_density",
        full_path="instrumentation.piano.right_hand_density",
        description="Average notes per beat in right hand (melodic/chordal density)",
        param_type=ParameterType.CONTINUOUS,
        default_value=2.0,
        min_value=1.0,
        max_value=4.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "classical", "latin", "bebop"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # Piano Comping and Rhythmic Style
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="piano_comping_pattern",
        full_path="instrumentation.piano.comping_pattern",
        description="Rhythmic comping pattern style for accompaniment",
        param_type=ParameterType.CATEGORICAL,
        options=["hits", "montuno", "stride", "flowing"],
        default_value="hits",
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "latin", "stride", "swing"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_register_preference",
        full_path="instrumentation.piano.register_preference",
        description="Preferred piano register for voicings",
        param_type=ParameterType.CATEGORICAL,
        options=["low", "mid", "high", "wide"],
        default_value="mid",
        category=ParameterCategory.TIMBRE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz", "pop", "ambient"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_pedal_usage",
        full_path="instrumentation.piano.pedal_usage",
        description="Sustain pedal usage intensity (0.0=no pedal, 1.0=constant pedal)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "romantic", "ambient", "ballad"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_cluster_probability",
        full_path="instrumentation.piano.cluster_probability",
        description="Probability of using tone clusters (dissonant adjacent notes)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary_classical", "experimental", "free_jazz"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # Piano Voicing Techniques
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="piano_octave_doubling",
        full_path="instrumentation.piano.octave_doubling",
        description="Whether to double melody/bass in octaves for power",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["rock", "gospel", "classical", "romantic"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_inner_voice_motion",
        full_path="instrumentation.piano.inner_voice_motion",
        description="Movement style of inner voices in chord progressions",
        param_type=ParameterType.CATEGORICAL,
        options=["static", "chromatic", "diatonic"],
        default_value="diatonic",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "classical", "choral"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_rhythmic_density",
        full_path="instrumentation.piano.rhythmic_density",
        description="Overall rhythmic activity level (0.0=sparse, 1.0=constant)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.6,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "latin", "funk", "bebop"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_syncopation_level",
        full_path="instrumentation.piano.syncopation_level",
        description="Syncopation intensity in piano part (0.0=on-beat, 1.0=heavily syncopated)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "funk", "latin", "afrobeat"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # Piano Chord Voicing Details
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="piano_chord_inversion_preference",
        full_path="instrumentation.piano.chord_inversion_preference",
        description="Preferred chord inversion for smooth voice leading",
        param_type=ParameterType.CATEGORICAL,
        options=["root", "first", "second", "mixed"],
        default_value="mixed",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz", "pop", "choral"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_shell_voicing_prob",
        full_path="instrumentation.piano.shell_voicing_prob",
        description="Probability of using shell voicings (root-3rd-7th only)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "bebop", "latin"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # Piano Articulation and Ornamentation
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="piano_tremolo_usage",
        full_path="instrumentation.piano.tremolo_usage",
        description="Frequency of tremolo effects (rapid note repetition)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "romantic", "film_score"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_grace_note_density",
        full_path="instrumentation.piano.grace_note_density",
        description="Density of grace notes and quick ornaments",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "baroque", "jazz", "gospel"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_alberti_bass_prob",
        full_path="instrumentation.piano.alberti_bass_prob",
        description="Probability of Alberti bass pattern (broken chord accompaniment)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "baroque", "classical_period"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_broken_chord_style",
        full_path="instrumentation.piano.broken_chord_style",
        description="Style of broken chord accompaniment patterns",
        param_type=ParameterType.CATEGORICAL,
        options=["arpeggio", "alberti", "waltz", "stride"],
        default_value="arpeggio",
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "waltz", "stride", "romantic"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_touch_articulation",
        full_path="instrumentation.piano.touch_articulation",
        description="Overall touch/articulation style",
        param_type=ParameterType.CATEGORICAL,
        options=["legato", "staccato", "portato", "mixed"],
        default_value="mixed",
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "pop", "all"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="piano_dynamic_range",
        full_path="instrumentation.piano.dynamic_range",
        description="Dynamic range utilization (0.3=narrow/consistent, 1.0=full range pp-ff)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.7,
        min_value=0.3,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "romantic", "jazz", "expressive"],
        module_file="generators/orchestrator.py"
    ))


def register_bass_parameters():
    """
    Register comprehensive bass parameters (15 params)

    Musical Rationale:
    Bass provides harmonic foundation and rhythmic drive. These parameters
    control walking patterns, register, rhythmic feel, and techniques that
    define bass style across genres from jazz to funk to rock.
    """

    # ========================================================================
    # Bass Pattern and Style
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="bass_walking_pattern",
        full_path="instrumentation.bass.walking_pattern",
        description="Whether to use walking bass pattern (continuous quarter notes)",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.BASS,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "swing", "bebop"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="bass_pattern_type",
        full_path="instrumentation.bass.pattern_type",
        description="Bass line pattern style",
        param_type=ParameterType.CATEGORICAL,
        options=["roots", "walking", "two_feel", "latin", "pedal"],
        default_value="roots",
        category=ParameterCategory.BASS,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "latin", "rock", "pop"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="bass_register",
        full_path="instrumentation.bass.register",
        description="Preferred bass register range",
        param_type=ParameterType.CATEGORICAL,
        options=["low", "mid", "high"],
        default_value="low",
        category=ParameterCategory.BASS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="bass_rhythmic_density",
        full_path="instrumentation.bass.rhythmic_density",
        description="Rhythmic activity level (0.0=whole notes, 1.0=sixteenth notes)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["funk", "jazz", "latin", "R&B"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # Bass Note Selection
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="bass_chromatic_approach",
        full_path="instrumentation.bass.chromatic_approach",
        description="Probability of chromatic approach notes to targets",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.BASS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "bebop", "walking_bass"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="bass_scale_tone_prob",
        full_path="instrumentation.bass.scale_tone_prob",
        description="Probability of using scale tones vs. chromatic passing tones",
        param_type=ParameterType.PROBABILITY,
        default_value=0.8,
        category=ParameterCategory.BASS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "walking_bass", "bebop"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="bass_anticipation_prob",
        full_path="instrumentation.bass.anticipation_prob",
        description="Probability of anticipating chord changes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "latin", "funk"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # Bass Techniques and Articulation
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="bass_octave_displacement",
        full_path="instrumentation.bass.octave_displacement",
        description="Probability of octave jumps for interest",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.BASS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["funk", "jazz", "fusion"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="bass_slap_technique_prob",
        full_path="instrumentation.bass.slap_technique_prob",
        description="Probability of slap bass technique (percussive thumb/pop)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["funk", "fusion", "R&B"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="bass_ghost_note_density",
        full_path="instrumentation.bass.ghost_note_density",
        description="Density of ghost notes (muted percussive notes)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["funk", "R&B", "neo_soul"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="bass_slide_usage",
        full_path="instrumentation.bass.slide_usage",
        description="Frequency of slides/glissando between notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["funk", "R&B", "jazz", "blues"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="bass_syncopation",
        full_path="instrumentation.bass.syncopation",
        description="Syncopation level in bass line",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["funk", "latin", "afrobeat", "jazz"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # Bass Special Patterns
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="bass_two_feel_swing",
        full_path="instrumentation.bass.two_feel_swing",
        description="Use two-feel swing pattern (half notes on 1 and 3)",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["swing", "big_band", "jazz_ballad"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="bass_pedal_point_prob",
        full_path="instrumentation.bass.pedal_point_prob",
        description="Probability of sustained pedal point (repeated note under changing harmony)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "rock", "modal", "ambient"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="bass_harmonic_complexity",
        full_path="instrumentation.bass.harmonic_complexity",
        description="Harmonic complexity of bass line (1=roots only, 7=chord tones+passing+approach)",
        param_type=ParameterType.INTEGER,
        default_value=4,
        min_value=1,
        max_value=7,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "fusion", "progressive"],
        module_file="generators/orchestrator.py"
    ))


def register_drums_parameters():
    """
    Register comprehensive drum parameters (25 params)

    Musical Rationale:
    Drums provide the rhythmic foundation and groove. These parameters control
    patterns, density, feel, and techniques across all drum kit pieces
    (kick, snare, hi-hat, cymbals, toms). Essential for defining genre and feel.
    """

    # ========================================================================
    # Drum Pattern Type
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="drums_pattern_type",
        full_path="instrumentation.drums.pattern_type",
        description="Primary drum pattern style/genre",
        param_type=ParameterType.CATEGORICAL,
        options=["swing", "straight", "latin", "funk", "brushes", "custom"],
        default_value="straight",
        category=ParameterCategory.DRUMS,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "rock", "latin", "funk", "pop"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # Custom Drum Patterns (for pattern_type="custom")
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="drums_kick_pattern",
        full_path="instrumentation.drums.kick_pattern",
        description="Custom kick drum pattern (array of beat positions)",
        param_type=ParameterType.ARRAY_INT,
        default_value=[0, 2],  # Beats 1 and 3 in 4/4
        category=ParameterCategory.DRUMS,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["all"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_snare_pattern",
        full_path="instrumentation.drums.snare_pattern",
        description="Custom snare drum pattern (array of beat positions)",
        param_type=ParameterType.ARRAY_INT,
        default_value=[1, 3],  # Backbeat on 2 and 4
        category=ParameterCategory.DRUMS,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["all"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_hihat_pattern",
        full_path="instrumentation.drums.hihat_pattern",
        description="Custom hi-hat pattern (array of beat positions)",
        param_type=ParameterType.ARRAY_INT,
        default_value=[0, 1, 2, 3],  # Quarter notes
        category=ParameterCategory.DRUMS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_ride_pattern",
        full_path="instrumentation.drums.ride_pattern",
        description="Custom ride cymbal pattern (array of beat positions)",
        param_type=ParameterType.ARRAY_INT,
        default_value=[0, 1, 2, 3],  # Quarter notes (jazz ride)
        category=ParameterCategory.DRUMS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "swing"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # Drum Density Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="drums_kick_density",
        full_path="instrumentation.drums.kick_density",
        description="Kick drum hit density (0.0=sparse, 1.0=constant)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.DRUMS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["rock", "metal", "electronic", "double_bass"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_snare_density",
        full_path="instrumentation.drums.snare_density",
        description="Snare drum hit density",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.DRUMS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_hihat_density",
        full_path="instrumentation.drums.hihat_density",
        description="Hi-hat hit density",
        param_type=ParameterType.PROBABILITY,
        default_value=0.6,
        category=ParameterCategory.DRUMS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_cymbal_density",
        full_path="instrumentation.drums.cymbal_density",
        description="Crash/ride cymbal density",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.DRUMS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["rock", "jazz", "metal"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # Drum Techniques and Accents
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="drums_tom_fill_prob",
        full_path="instrumentation.drums.tom_fill_prob",
        description="Probability of tom fills at phrase endings",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.DRUMS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["rock", "pop", "metal"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_crash_accent_prob",
        full_path="instrumentation.drums.crash_accent_prob",
        description="Probability of crash cymbal on accents/downbeats",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.DRUMS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["rock", "pop", "metal"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_ride_bell_prob",
        full_path="instrumentation.drums.ride_bell_prob",
        description="Probability of using ride bell instead of bow",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "latin"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_hihat_openness",
        full_path="instrumentation.drums.hihat_openness",
        description="Hi-hat openness (0.0=closed, 1.0=fully open)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.2,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["rock", "funk", "disco"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_brush_technique",
        full_path="instrumentation.drums.brush_technique",
        description="Use brush technique instead of sticks",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "ballad", "swing"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_cross_stick_prob",
        full_path="instrumentation.drums.cross_stick_prob",
        description="Probability of cross-stick (rim knock) on snare",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "latin", "ballad"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_rim_shot_prob",
        full_path="instrumentation.drums.rim_shot_prob",
        description="Probability of rim shot (stick + head simultaneously)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["rock", "funk", "R&B"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_ghost_note_density",
        full_path="instrumentation.drums.ghost_note_density",
        description="Density of ghost notes (soft grace notes between main hits)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["funk", "jazz", "R&B", "gospel"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # Drum Feel and Complexity
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="drums_polyrhythm_complexity",
        full_path="instrumentation.drums.polyrhythm_complexity",
        description="Complexity of polyrhythmic patterns between limbs",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.3,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["afrobeat", "progressive", "jazz", "fusion"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_fill_frequency",
        full_path="instrumentation.drums.fill_frequency",
        description="How often to insert drum fills",
        param_type=ParameterType.PROBABILITY,
        default_value=0.25,
        category=ParameterCategory.DRUMS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["rock", "pop", "jazz"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_fill_complexity",
        full_path="instrumentation.drums.fill_complexity",
        description="Complexity of drum fills (1=simple, 7=virtuosic)",
        param_type=ParameterType.INTEGER,
        default_value=4,
        min_value=1,
        max_value=7,
        category=ParameterCategory.DRUMS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["rock", "jazz", "fusion", "metal"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_swing_ratio",
        full_path="instrumentation.drums.swing_ratio",
        description="Swing ratio for ride/hi-hat (0.5=straight, 0.67=standard swing)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.5,
        max_value=0.67,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "swing", "shuffle"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_syncopation_level",
        full_path="instrumentation.drums.syncopation_level",
        description="Overall syncopation in drum patterns",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["funk", "jazz", "afrobeat", "latin"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_metric_displacement",
        full_path="instrumentation.drums.metric_displacement",
        description="Use metric displacement (shifting pattern by offset)",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["progressive", "jazz", "metal"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_clave_adherence",
        full_path="instrumentation.drums.clave_adherence",
        description="Adherence to clave pattern for Latin styles",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["latin", "salsa", "afro_cuban", "bossa_nova"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drums_dynamic_variation",
        full_path="instrumentation.drums.dynamic_variation",
        description="Dynamic variation between hits (0.0=robotic/uniform, 1.0=very expressive)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.6,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "acoustic", "expressive"],
        module_file="generators/orchestrator.py"
    ))


def register_brass_parameters():
    """
    Register comprehensive brass section parameters (10 params)

    Musical Rationale:
    Brass sections (trumpets, trombones, saxophones) provide power,
    color, and harmonic richness. These parameters control section size,
    voicing, articulation, and characteristic brass techniques.
    """

    # ========================================================================
    # Brass Section Configuration
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="brass_section_size",
        full_path="instrumentation.brass.section_size",
        description="Number of brass voices in section",
        param_type=ParameterType.CATEGORICAL,
        options=[2, 3, 4, 5],
        default_value=4,
        category=ParameterCategory.TIMBRE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "big_band", "funk", "soul"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="brass_voicing_type",
        full_path="instrumentation.brass.voicing_type",
        description="Brass section voicing style",
        param_type=ParameterType.CATEGORICAL,
        options=["close", "open", "drop2", "spread"],
        default_value="close",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "big_band", "funk"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="brass_soli_probability",
        full_path="instrumentation.brass.soli_probability",
        description="Probability of brass soli (section playing melody in unison/harmony)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["big_band", "jazz", "funk"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="brass_background_density",
        full_path="instrumentation.brass.background_density",
        description="Density of brass background figures (punches, stabs, pads)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["big_band", "funk", "soul", "R&B"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # Brass Articulation and Techniques
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="brass_fall_off_prob",
        full_path="instrumentation.brass.fall_off_prob",
        description="Probability of fall-off (descending pitch bend at note end)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "big_band", "swing"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="brass_doit_prob",
        full_path="instrumentation.brass.doit_prob",
        description="Probability of doit (ascending pitch bend at note start)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "big_band", "swing"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="brass_shake_ornament_prob",
        full_path="instrumentation.brass.shake_ornament_prob",
        description="Probability of shake ornament (rapid lip trill)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "big_band"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # Brass Mutes and Tone Color
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="brass_plunger_mute_prob",
        full_path="instrumentation.brass.plunger_mute_prob",
        description="Probability of plunger mute (wah-wah effect)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.TIMBRE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "big_band", "swing"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="brass_cup_mute_prob",
        full_path="instrumentation.brass.cup_mute_prob",
        description="Probability of cup mute (mellower, distant tone)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.TIMBRE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "ballad"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="brass_harmonic_series_voicing",
        full_path="instrumentation.brass.harmonic_series_voicing",
        description="Use harmonic series-based voicing (natural overtone spacing)",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary", "minimalist"],
        module_file="generators/orchestrator.py"
    ))


def register_strings_parameters():
    """
    Register comprehensive string section parameters (10 params)

    Musical Rationale:
    String sections (violin, viola, cello, bass) provide lush harmonic
    textures, sustained pads, and expressive melodic lines. These parameters
    control section size, voicing density, and bowing techniques.
    """

    # ========================================================================
    # String Section Configuration
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="strings_section_size",
        full_path="instrumentation.strings.section_size",
        description="String section size",
        param_type=ParameterType.CATEGORICAL,
        options=[4, 8, 12, 16, "full"],
        default_value=12,
        category=ParameterCategory.TIMBRE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "film_score", "pop_orchestral"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="strings_voicing_density",
        full_path="instrumentation.strings.voicing_density",
        description="Number of simultaneous notes in string voicings",
        param_type=ParameterType.INTEGER,
        default_value=4,
        min_value=3,
        max_value=8,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "film_score"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="strings_divisi_probability",
        full_path="instrumentation.strings.divisi_probability",
        description="Probability of divisi (section splitting into multiple parts)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "romantic", "film_score"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # String Articulation Techniques
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="strings_pizzicato_prob",
        full_path="instrumentation.strings.pizzicato_prob",
        description="Probability of pizzicato (plucked) vs. arco (bowed)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "pop", "film_score"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="strings_tremolo_prob",
        full_path="instrumentation.strings.tremolo_prob",
        description="Probability of tremolo bowing (rapid bow changes)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "film_score", "horror"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="strings_sul_ponticello_prob",
        full_path="instrumentation.strings.sul_ponticello_prob",
        description="Probability of sul ponticello (bowing near bridge for glassy tone)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.TIMBRE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary_classical", "experimental", "film_score"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="strings_col_legno_prob",
        full_path="instrumentation.strings.col_legno_prob",
        description="Probability of col legno (hitting strings with wood of bow)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary_classical", "experimental"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="strings_harmonic_prob",
        full_path="instrumentation.strings.harmonic_prob",
        description="Probability of natural/artificial harmonics (ethereal high notes)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.TIMBRE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "contemporary", "film_score"],
        module_file="generators/orchestrator.py"
    ))

    # ========================================================================
    # String Expression
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="strings_vibrato_intensity",
        full_path="instrumentation.strings.vibrato_intensity",
        description="Vibrato intensity (0.0=none, 1.0=wide/expressive)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "romantic", "film_score"],
        module_file="generators/orchestrator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="strings_portamento_usage",
        full_path="instrumentation.strings.portamento_usage",
        description="Frequency of portamento/glissando between notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["romantic", "film_score", "expressive"],
        module_file="generators/orchestrator.py"
    ))


def register_all_instrumentation_parameters():
    """
    Register all instrumentation parameters (80 total)

    Breakdown:
    - Piano: 20 parameters
    - Bass: 15 parameters
    - Drums: 25 parameters
    - Brass: 10 parameters
    - Strings: 10 parameters

    Total: 80 parameters addressing the critical instrumentation gap
    """
    print("=" * 80)
    print("INSTRUMENTATION PARAMETER EXPANSION - AGENT 1")
    print("=" * 80)
    print("\n🎹 Registering Piano parameters (20)...")
    register_piano_parameters()

    print("🎸 Registering Bass parameters (15)...")
    register_bass_parameters()

    print("🥁 Registering Drums parameters (25)...")
    register_drums_parameters()

    print("🎺 Registering Brass parameters (10)...")
    register_brass_parameters()

    print("🎻 Registering Strings parameters (10)...")
    register_strings_parameters()

    stats = REGISTRY.get_statistics()
    instrumentation_params = [
        p for p in REGISTRY.parameters.values()
        if p.full_path.startswith("instrumentation.")
    ]

    print(f"\n{'=' * 80}")
    print("✅ INSTRUMENTATION EXPANSION COMPLETE!")
    print(f"{'=' * 80}")
    print(f"   Instrumentation parameters added: {len(instrumentation_params)}")
    print(f"   Total parameters in registry: {stats['total_parameters']}")
    print(f"\n   Breakdown by instrument:")
    print(f"      Piano:   {len([p for p in instrumentation_params if '.piano.' in p.full_path])} parameters")
    print(f"      Bass:    {len([p for p in instrumentation_params if '.bass.' in p.full_path])} parameters")
    print(f"      Drums:   {len([p for p in instrumentation_params if '.drums.' in p.full_path])} parameters")
    print(f"      Brass:   {len([p for p in instrumentation_params if '.brass.' in p.full_path])} parameters")
    print(f"      Strings: {len([p for p in instrumentation_params if '.strings.' in p.full_path])} parameters")
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    # When run as script, use absolute imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from parameters.universal_registry import REGISTRY

    # Register all instrumentation parameters
    register_all_instrumentation_parameters()

    # Export updated registry
    print("\n💾 Exporting updated registry...")
    REGISTRY.export_to_json("/home/user/Do/midi_generator/parameters/registry.json")
    REGISTRY.generate_documentation("/home/user/Do/midi_generator/parameters/PARAMETERS.md")

    print("✅ Registry exported to JSON and documentation generated")
    print("=" * 80)
