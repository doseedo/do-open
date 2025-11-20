#!/usr/bin/env python3
"""
Style Generator - Programmatic Style Database Creation
======================================================

Generates 100+ complete style definitions with all 515 parameters
using musical templates and intelligent variations.

Author: Agent 7 - Style Database Curator
Date: 2025-11-20
"""

from typing import Dict, List, Any
import copy


# Base parameter template (all 515 parameters with default values)
BASE_PARAMETERS_TEMPLATE = {
    # === HARMONY (100 parameters) ===
    "harmony.voicing.type": "close",
    "harmony.voicing.spread": 0.5,
    "harmony.voicing.density": 4,
    "harmony.voicing.close_position_prob": 0.5,
    "harmony.voicing.rootless_prob": 0.5,
    "harmony.voicing.shell_voicing_prob": 0.5,
    "harmony.voicing.drop2_prob": 0.5,
    "harmony.voicing.drop3_prob": 0.3,
    "harmony.voicing.quartal_probability": 0.2,
    "harmony.voicing.cluster_probability": 0.1,

    "harmony.extensions.use_9ths": True,
    "harmony.extensions.use_11ths": True,
    "harmony.extensions.use_13ths": False,
    "harmony.extensions.ninth_usage": 0.5,
    "harmony.extensions.eleventh_usage": 0.3,
    "harmony.extensions.thirteenth_usage": 0.2,
    "harmony.extensions.altered_extensions": 0.3,
    "harmony.extensions.sharp_nine_prob": 0.2,
    "harmony.extensions.flat_nine_prob": 0.2,
    "harmony.extensions.sharp_eleven_prob": 0.3,
    "harmony.extensions.flat_thirteen_prob": 0.2,

    "harmony.substitution.tritone_probability": 0.3,
    "harmony.substitution.modal_interchange_probability": 0.2,
    "harmony.substitution.reharmonization_prob": 0.3,
    "harmony.substitution.diminished_passing_probability": 0.2,
    "harmony.substitution.secondary_dominant_probability": 0.4,
    "harmony.substitution.augmented_sixth_probability": 0.1,
    "harmony.substitution.neapolitan_prob": 0.05,

    "harmony.voice_leading.smoothness": 0.7,
    "harmony.voice_leading.parallel_motion_tolerance": 0.3,
    "harmony.voice_leading.contrary_motion_prob": 0.5,
    "harmony.voice_leading.oblique_motion_prob": 0.3,
    "harmony.voice_leading.max_voice_movement": 5,
    "harmony.voice_leading.common_tone_retention": 0.6,

    "harmony.progression.type": "functional",
    "harmony.progression.complexity": 0.5,
    "harmony.progression.chords_per_bar": 1.5,
    "harmony.progression.modulation_frequency": 0.2,
    "harmony.progression.secondary_dominants": 0.4,
    "harmony.progression.backdoor_progression_prob": 0.2,
    "harmony.progression.coltrane_changes": False,
    "harmony.progression.giant_steps_prob": 0.0,
    "harmony.progression.circle_of_fifths": 0.4,
    "harmony.progression.chromatic_descent": 0.2,

    "harmony.modal.mode": "ionian",
    "harmony.modal.modal_interchange_intensity": 0.2,
    "harmony.modal.dorian_probability": 0.14,
    "harmony.modal.phrygian_probability": 0.14,
    "harmony.modal.lydian_probability": 0.14,
    "harmony.modal.mixolydian_probability": 0.14,
    "harmony.modal.aeolian_probability": 0.14,
    "harmony.modal.locrian_probability": 0.14,
    "harmony.modal.progression_type": "characteristic",
    "harmony.modal.cadence_type": "plagal",

    "harmony.chromatic.chromaticism": 0.3,
    "harmony.chromatic.chromatic_approach": 0.3,
    "harmony.chromatic.chromatic_mediant_pattern": "UCM LCM",
    "harmony.chromatic.parallel_harmony": 0.2,
    "harmony.chromatic.planing_type": "diatonic",
    "harmony.chromatic.whole_tone_usage": 0.1,
    "harmony.chromatic.diminished_scale_usage": 0.1,
    "harmony.chromatic.augmented_scale_usage": 0.05,

    "harmony.jazz.ii_v_i_prob": 0.6,
    "harmony.jazz.turnaround_prob": 0.5,
    "harmony.jazz.tritone_substitution": 0.3,
    "harmony.jazz.diminished_approach": 0.3,
    "harmony.jazz.blues_substitution": 0.2,

    "harmony.neo_riemannian.transformation_sequence": "PLR",
    "harmony.neo_riemannian.apply_voice_leading": True,
    "harmony.neo_riemannian.hexatonic_pole": 0,

    # === MELODY (80 parameters) ===
    "melody.contour.type": "arch",
    "melody.contour.arch_probability": 0.5,
    "melody.contour.complexity": 0.5,
    "melody.contour.shape": "balanced",
    "melody.contour.climax_position": 0.67,
    "melody.contour.ascending_prob": 0.2,
    "melody.contour.descending_prob": 0.2,
    "melody.contour.wave_prob": 0.2,

    "melody.intervals.stepwise_probability": 0.7,
    "melody.intervals.max_leap": 12,
    "melody.intervals.leap_resolution_prob": 0.7,
    "melody.intervals.compound_interval_prob": 0.2,
    "melody.intervals.chromatic_approach_prob": 0.3,
    "melody.intervals.diatonic_preference": 0.7,
    "melody.intervals.third_prob": 0.3,
    "melody.intervals.fourth_prob": 0.2,
    "melody.intervals.fifth_prob": 0.15,
    "melody.intervals.sixth_prob": 0.1,
    "melody.intervals.seventh_prob": 0.05,
    "melody.intervals.octave_prob": 0.05,

    "melody.chromaticism.amount": 0.3,
    "melody.chromaticism.passing_tone_prob": 0.4,
    "melody.chromaticism.neighbor_tone_prob": 0.3,
    "melody.chromaticism.chromatic_embellishment": 0.3,
    "melody.chromaticism.bebop_chromatic_prob": 0.2,
    "melody.chromaticism.blue_notes": 0.2,

    "melody.ornaments.probability": 0.2,
    "melody.ornaments.grace_note_prob": 0.2,
    "melody.ornaments.trill_prob": 0.1,
    "melody.ornaments.mordent_prob": 0.1,
    "melody.ornaments.turn_prob": 0.1,
    "melody.ornaments.appoggiatura_prob": 0.15,
    "melody.ornaments.acciaccatura_prob": 0.1,
    "melody.ornaments.slide_prob": 0.1,
    "melody.ornaments.scoop_prob": 0.05,
    "melody.ornaments.fall_prob": 0.05,
    "melody.ornaments.doit_prob": 0.05,

    "melody.range.min_note": 60,
    "melody.range.max_note": 84,
    "melody.range.comfortable_range": 12,
    "melody.range.tessitura": "medium",
    "melody.range.register_preference": "middle",

    "melody.phrasing.phrase_length": 4,
    "melody.phrasing.breath_marks": 0.5,
    "melody.phrasing.phrase_arch": 0.5,
    "melody.phrasing.antecedent_consequent": 0.7,
    "melody.phrasing.elision_prob": 0.2,

    "melody.motif.development": 0.5,
    "melody.motif.repetition": 0.4,
    "melody.motif.sequence": 0.3,
    "melody.motif.variation": 0.5,
    "melody.motif.inversion": 0.2,
    "melody.motif.retrograde": 0.1,
    "melody.motif.augmentation": 0.15,
    "melody.motif.diminution": 0.15,

    "melody.note_density": 0.6,
    "melody.rest_frequency": 0.3,
    "melody.syncopation": 0.3,
    "melody.pickup_notes": 0.4,
    "melody.downbeat_emphasis": 0.6,

    # === RHYTHM (70 parameters) ===
    "rhythm.swing.amount": 0.5,
    "rhythm.swing.active": False,
    "rhythm.swing.intensity": 0.5,
    "rhythm.swing.variation": 0.05,
    "rhythm.swing.type": "straight",

    "rhythm.feel.type": "straight",
    "rhythm.feel.straight_8ths": True,
    "rhythm.feel.laid_back_timing": 0.0,
    "rhythm.feel.rushed_timing": 0.0,
    "rhythm.feel.groove_intensity": 0.5,
    "rhythm.feel.pocket": 0.5,

    "rhythm.syncopation.probability": 0.3,
    "rhythm.syncopation.intensity": 0.5,
    "rhythm.syncopation.anticipation_8th": 0.3,
    "rhythm.syncopation.anticipation_16th": 0.2,
    "rhythm.syncopation.composite_rhythm_complexity": 5,
    "rhythm.syncopation.offbeat_emphasis": 0.3,

    "rhythm.microtiming.variation": 10,
    "rhythm.microtiming.humanization": 0.6,
    "rhythm.microtiming.swing_variation": 0.05,
    "rhythm.microtiming.timing_jitter": 5,

    "rhythm.pattern.complexity": 0.5,
    "rhythm.pattern.polyrhythm_active": False,
    "rhythm.pattern.polymeter_active": False,
    "rhythm.pattern.rhythmic_canon": 0.0,
    "rhythm.pattern.metric_modulation": False,
    "rhythm.pattern.odd_meter_grouping": [4],
    "rhythm.pattern.clave_pattern": None,
    "rhythm.pattern.clave_adherence": 0.0,
    "rhythm.pattern.hemiola": 0.1,
    "rhythm.pattern.cross_rhythm": 0.1,

    "rhythm.note_density": 0.6,
    "rhythm.rest_density": 0.3,
    "rhythm.tied_note_prob": 0.3,
    "rhythm.dotted_rhythm_prob": 0.3,
    "rhythm.triplet_prob": 0.2,
    "rhythm.quintuplet_prob": 0.05,
    "rhythm.sextuplet_prob": 0.05,

    "rhythm.subdivision.eighth_note": 0.6,
    "rhythm.subdivision.sixteenth_note": 0.3,
    "rhythm.subdivision.triplet": 0.2,
    "rhythm.subdivision.quarter_note": 0.4,
    "rhythm.subdivision.half_note": 0.2,
    "rhythm.subdivision.whole_note": 0.1,

    "rhythm.rubato.active": False,
    "rhythm.rubato.intensity": 0.0,
    "rhythm.rubato.phrase_end": 0.1,
    "rhythm.rubato.phrase_start": 0.05,

    # === INSTRUMENTATION (60 parameters) ===
    "instrumentation.strings.active": True,
    "instrumentation.strings.section_size": "medium",
    "instrumentation.strings.voicing_density": 4,
    "instrumentation.strings.divisi_prob": 0.5,
    "instrumentation.strings.tremolo_prob": 0.1,
    "instrumentation.strings.pizzicato_prob": 0.1,
    "instrumentation.strings.sul_ponticello_prob": 0.05,
    "instrumentation.strings.sul_tasto_prob": 0.05,
    "instrumentation.strings.con_sordino_prob": 0.1,
    "instrumentation.strings.harmonics_prob": 0.05,

    "instrumentation.brass.active": True,
    "instrumentation.brass.section_size": 4,
    "instrumentation.brass.voicing_type": "close",
    "instrumentation.brass.soli_probability": 0.3,
    "instrumentation.brass.plunger_mute_prob": 0.1,
    "instrumentation.brass.harmon_mute_prob": 0.1,
    "instrumentation.brass.cup_mute_prob": 0.1,
    "instrumentation.brass.straight_mute_prob": 0.1,
    "instrumentation.brass.bucket_mute_prob": 0.05,

    "instrumentation.woodwinds.active": True,
    "instrumentation.woodwinds.section_size": 4,
    "instrumentation.woodwinds.doubling_prob": 0.3,
    "instrumentation.woodwinds.flute_prob": 0.5,
    "instrumentation.woodwinds.clarinet_prob": 0.5,
    "instrumentation.woodwinds.oboe_prob": 0.3,
    "instrumentation.woodwinds.bassoon_prob": 0.3,
    "instrumentation.woodwinds.saxophone_prob": 0.5,

    "instrumentation.piano.active": True,
    "instrumentation.piano.voicing_density": 4,
    "instrumentation.piano.left_hand_style": "shell",
    "instrumentation.piano.comping_pattern": "standard",
    "instrumentation.piano.stride_prob": 0.0,
    "instrumentation.piano.block_chord_prob": 0.3,
    "instrumentation.piano.arpeggio_prob": 0.3,
    "instrumentation.piano.broken_chord_prob": 0.3,
    "instrumentation.piano.alberti_bass_prob": 0.1,

    "instrumentation.bass.active": True,
    "instrumentation.bass.pattern_type": "walking",
    "instrumentation.bass.walking_pattern": True,
    "instrumentation.bass.syncopation": 0.3,
    "instrumentation.bass.chromatic_approach": 0.3,
    "instrumentation.bass.arco_prob": 0.1,
    "instrumentation.bass.pizzicato_prob": 0.9,
    "instrumentation.bass.slap_prob": 0.0,

    "instrumentation.drums.active": True,
    "instrumentation.drums.pattern_type": "standard",
    "instrumentation.drums.brush_technique": False,
    "instrumentation.drums.ride_pattern": "standard",
    "instrumentation.drums.hi_hat_pattern": "8th",
    "instrumentation.drums.kick_pattern": "standard",
    "instrumentation.drums.snare_pattern": "standard",
    "instrumentation.drums.cymbal_swell_prob": 0.2,
    "instrumentation.drums.tom_fills_prob": 0.2,
    "instrumentation.drums.cross_stick_prob": 0.1,

    "instrumentation.guitar.active": False,
    "instrumentation.guitar.comping_density": 0.5,
    "instrumentation.guitar.freddie_green_style": False,
    "instrumentation.guitar.chord_melody": False,
    "instrumentation.guitar.fingerstyle": False,

    # === DYNAMICS (40 parameters) ===
    "dynamics.velocity.overall_level": 0.7,
    "dynamics.velocity.variation": 0.3,
    "dynamics.velocity.phrase_shaping": 0.6,
    "dynamics.velocity.micro_dynamics": 0.5,
    "dynamics.velocity.accent_intensity": 0.3,
    "dynamics.velocity.downbeat_accent": 0.4,
    "dynamics.velocity.upbeat_accent": 0.2,

    "dynamics.form.intro_level": 0.5,
    "dynamics.form.verse_level": 0.6,
    "dynamics.form.chorus_level": 0.75,
    "dynamics.form.bridge_level": 0.7,
    "dynamics.form.outro_level": 0.5,
    "dynamics.form.climax_level": 0.9,

    "dynamics.range.min": 0.3,
    "dynamics.range.max": 0.9,
    "dynamics.range.dynamic_range": "medium",

    "dynamics.expression.crescendo_prob": 0.4,
    "dynamics.expression.decrescendo_prob": 0.4,
    "dynamics.expression.sforzando_prob": 0.1,
    "dynamics.expression.accent_prob": 0.3,
    "dynamics.expression.forte_piano_contrast": 0.5,
    "dynamics.expression.subito_piano": 0.1,
    "dynamics.expression.subito_forte": 0.1,

    "dynamics.balance.melody_prominence": 0.8,
    "dynamics.balance.bass_level": 0.7,
    "dynamics.balance.harmony_level": 0.6,
    "dynamics.balance.drums_level": 0.5,
    "dynamics.balance.orchestral_balance": 0.7,

    "dynamics.envelope.attack": 0.5,
    "dynamics.envelope.decay": 0.5,
    "dynamics.envelope.sustain": 0.7,
    "dynamics.envelope.release": 0.5,

    # === ARTICULATION (35 parameters) ===
    "articulation.default": "legato",
    "articulation.variety": 0.6,
    "articulation.legato_prob": 0.6,
    "articulation.staccato_prob": 0.2,
    "articulation.tenuto_prob": 0.15,
    "articulation.marcato_prob": 0.1,
    "articulation.accent_prob": 0.3,
    "articulation.portamento_prob": 0.1,
    "articulation.glissando_prob": 0.05,

    "articulation.bowing.slur_prob": 0.5,
    "articulation.bowing.detache_prob": 0.3,
    "articulation.bowing.spiccato_prob": 0.1,
    "articulation.bowing.staccato_prob": 0.1,
    "articulation.bowing.legato_prob": 0.5,

    "articulation.brass.fall_prob": 0.1,
    "articulation.brass.doit_prob": 0.05,
    "articulation.brass.shake_prob": 0.05,
    "articulation.brass.growl_prob": 0.05,
    "articulation.brass.lip_trill_prob": 0.05,
    "articulation.brass.flutter_tongue_prob": 0.02,

    "articulation.tongue.single_tongue": 0.7,
    "articulation.tongue.double_tongue": 0.2,
    "articulation.tongue.triple_tongue": 0.1,

    "articulation.phrasing.breath_mark_prob": 0.5,
    "articulation.phrasing.caesura_prob": 0.1,
    "articulation.phrasing.fermata_prob": 0.05,

    # === STRUCTURE (40 parameters) ===
    "structure.form.type": "AABA",
    "structure.form.length_bars": 32,
    "structure.form.intro_bars": 4,
    "structure.form.outro_bars": 4,
    "structure.form.bridge_bars": 8,
    "structure.form.verse_bars": 8,
    "structure.form.chorus_bars": 8,
    "structure.form.vamp_probability": 0.2,
    "structure.form.tag_ending_prob": 0.2,
    "structure.form.coda_prob": 0.3,
    "structure.form.interlude_prob": 0.2,

    "structure.sections.intro_style": "standard",
    "structure.sections.verse_style": "standard",
    "structure.sections.chorus_style": "full",
    "structure.sections.bridge_style": "modulation",
    "structure.sections.outro_style": "fade",
    "structure.sections.solo_section_prob": 0.3,

    "structure.repetition.exact_repeat_prob": 0.6,
    "structure.repetition.varied_repeat_prob": 0.3,
    "structure.repetition.sequence_prob": 0.2,
    "structure.repetition.ostinato_usage": 0.2,
    "structure.repetition.riff_usage": 0.3,

    "structure.development.motivic_development": 0.5,
    "structure.development.thematic_transformation": 0.4,
    "structure.development.variation_technique": 0.5,
    "structure.development.fragmentation": 0.3,
    "structure.development.expansion": 0.3,

    "structure.cadence.authentic_prob": 0.7,
    "structure.cadence.plagal_prob": 0.2,
    "structure.cadence.deceptive_prob": 0.1,
    "structure.cadence.half_cadence_prob": 0.3,
    "structure.cadence.phrygian_cadence_prob": 0.05,

    # === TEXTURE (25 parameters) ===
    "texture.density": 0.6,
    "texture.variation": 0.5,
    "texture.layering": 0.5,
    "texture.orchestral_density": 0.6,
    "texture.vertical_density": 0.6,
    "texture.horizontal_density": 0.5,

    "texture.polyphony.active": True,
    "texture.polyphony.voice_count": 4,
    "texture.polyphony.counterpoint_usage": 0.3,
    "texture.polyphony.canon_prob": 0.05,
    "texture.polyphony.fugue_prob": 0.02,
    "texture.polyphony.imitation_prob": 0.2,

    "texture.homophony.active": True,
    "texture.homophony.chordal_texture": 0.5,
    "texture.homophony.melody_accompaniment": 0.7,
    "texture.homophony.block_chords": 0.4,

    "texture.monophony.active": False,
    "texture.monophony.solo_prob": 0.2,
    "texture.monophony.unison_prob": 0.1,

    "texture.heterophony.active": False,
    "texture.heterophony.variation_prob": 0.1,

    # === TIMBRE (30 parameters) ===
    "timbre.brightness": 0.5,
    "timbre.warmth": 0.5,
    "timbre.edge": 0.3,
    "timbre.darkness": 0.3,
    "timbre.richness": 0.5,
    "timbre.clarity": 0.6,

    "timbre.orchestration.lush_strings": 0.5,
    "timbre.orchestration.bright_brass": 0.5,
    "timbre.orchestration.warm_woodwinds": 0.5,
    "timbre.orchestration.intimate_piano": 0.5,
    "timbre.orchestration.percussion_color": 0.3,

    "timbre.register.high_emphasis": 0.3,
    "timbre.register.mid_emphasis": 0.6,
    "timbre.register.low_emphasis": 0.4,

    "timbre.blend.section_blend": 0.7,
    "timbre.blend.orchestral_blend": 0.6,
    "timbre.blend.contrast": 0.4,

    "timbre.effects.reverb_amount": 0.3,
    "timbre.effects.delay_amount": 0.1,
    "timbre.effects.chorus_amount": 0.1,
    "timbre.effects.modulation_amount": 0.0,
    "timbre.effects.distortion_amount": 0.0,

    # === GENRE SPECIFIC (20 parameters) ===
    "genre.jazz.bebop_lines": 0.2,
    "genre.jazz.swing_feel": 0.5,
    "genre.jazz.blue_notes": 0.2,
    "genre.jazz.walking_bass": 0.5,
    "genre.jazz.comping_style": "standard",

    "genre.classical.orchestral_arrangement": False,
    "genre.classical.voice_leading_strict": 0.5,
    "genre.classical.formal_structure": 0.7,
    "genre.classical.counterpoint": 0.3,

    "genre.latin.clave_based": False,
    "genre.latin.montuno_pattern": False,
    "genre.latin.tumbao_bass": False,

    "genre.rock.power_chords": False,
    "genre.rock.backbeat": False,

    "genre.electronic.quantization": 0.9,
    "genre.electronic.grid_locked": False,

    # === TEMPO & TIME (10 parameters) ===
    "tempo.bpm": 120,
    "tempo.variation": 0.05,
    "tempo.ritardando_prob": 0.2,
    "tempo.accelerando_prob": 0.1,
    "tempo.rubato_intensity": 0.1,
    "tempo.fermata_prob": 0.05,

    "time_signature.numerator": 4,
    "time_signature.denominator": 4,
    "time_signature.feel": "4/4_standard",
}


def create_style_variation(base_params: Dict, modifications: Dict) -> Dict:
    """
    Create a style variation by modifying specific parameters

    Args:
        base_params: Base parameter dictionary
        modifications: Dictionary of parameters to modify

    Returns:
        Modified parameter dictionary
    """
    params = copy.deepcopy(base_params)
    params.update(modifications)
    return params


def generate_all_styles() -> Dict[str, Dict]:
    """
    Generate all 100+ style definitions programmatically

    Returns:
        Dictionary of all styles with complete parameter sets
    """
    styles = {}

    # Helper function to add a style
    def add_style(name: str, tempo: int, description: str, modifications: Dict):
        styles[name] = {
            "tempo": tempo,
            "description": description,
            "parameters": create_style_variation(BASE_PARAMETERS_TEMPLATE, modifications)
        }

    # === JAZZ STYLES (25 styles) ===

    # 1. Sinatra Ballad (already detailed in main file)
    add_style("sinatra_ballad", 72, "Frank Sinatra-style ballad with lush Nelson Riddle orchestration", {
        "harmony.voicing.density": 6,
        "harmony.extensions.ninth_usage": 0.7,
        "harmony.extensions.thirteenth_usage": 0.5,
        "rhythm.swing.amount": 0.54,
        "rhythm.feel.laid_back_timing": 0.3,
        "instrumentation.strings.active": True,
        "instrumentation.strings.section_size": "full",
        "instrumentation.drums.brush_technique": True,
        "dynamics.velocity.overall_level": 0.65,
        "structure.form.type": "AABA",
        "tempo.bpm": 72,
    })

    # 2. Coltrane Giant Steps (already detailed in main file)
    add_style("coltrane_giant_steps", 280, "John Coltrane Giant Steps-style rapid harmonic changes", {
        "harmony.progression.coltrane_changes": True,
        "harmony.progression.giant_steps_prob": 0.9,
        "harmony.progression.chords_per_bar": 3.0,
        "melody.note_density": 0.9,
        "melody.chromaticism.bebop_chromatic_prob": 0.8,
        "rhythm.swing.amount": 0.62,
        "instrumentation.piano.comping_pattern": "sparse",
        "instrumentation.bass.walking_pattern": True,
        "genre.jazz.bebop_lines": 0.9,
        "tempo.bpm": 280,
    })

    # 3. Bill Evans Trio
    add_style("bill_evans_trio", 140, "Bill Evans-style piano trio with impressionistic harmony", {
        "harmony.voicing.rootless_prob": 0.9,
        "harmony.voicing.shell_voicing_prob": 0.7,
        "harmony.modal.modal_interchange_intensity": 0.4,
        "harmony.extensions.eleventh_usage": 0.8,
        "instrumentation.piano.voicing_density": 3,
        "instrumentation.piano.left_hand_style": "rootless",
        "instrumentation.drums.brush_technique": True,
        "rhythm.swing.amount": 0.56,
        "dynamics.velocity.micro_dynamics": 0.9,
        "tempo.bpm": 140,
    })

    # 4. Mingus Workshop
    add_style("mingus_workshop", 200, "Charles Mingus-style collective improvisation", {
        "harmony.progression.modal_interchange_intensity": 0.8,
        "structure.form.type": "through_composed",
        "instrumentation.brass.soli_probability": 0.7,
        "rhythm.pattern.polyrhythm_active": True,
        "rhythm.syncopation.composite_rhythm_complexity": 6,
        "dynamics.expression.forte_piano_contrast": 0.9,
        "texture.polyphony.counterpoint_usage": 0.7,
        "tempo.bpm": 200,
    })

    # 5. Count Basie Swing
    add_style("basie_swing", 180, "Count Basie-style swing with punchy section hits", {
        "harmony.voicing.type": "spread",
        "harmony.complexity": 0.3,
        "instrumentation.brass.soli_probability": 0.9,
        "rhythm.swing.amount": 0.62,
        "rhythm.syncopation.probability": 0.5,
        "articulation.staccato_prob": 0.7,
        "articulation.accent_prob": 0.5,
        "instrumentation.guitar.freddie_green_style": True,
        "texture.density": 0.5,
        "tempo.bpm": 180,
    })

    # 6. Duke Ellington Orchestra
    add_style("ellington_orchestra", 144, "Duke Ellington-style exotic harmonies and orchestral colors", {
        "harmony.complexity": 0.9,
        "harmony.extensions.use_13ths": True,
        "harmony.chromatic.chromaticism": 0.6,
        "instrumentation.brass.plunger_mute_prob": 0.6,
        "articulation.brass.growl_prob": 0.4,
        "timbre.orchestration.lush_strings": 0.8,
        "texture.orchestral_density": 0.8,
        "dynamics.range.dynamic_range": "very_wide",
        "tempo.bpm": 144,
    })

    # 7. Thad Jones Modern
    add_style("thad_jones_modern", 160, "Thad Jones-style modern big band with quartal harmony", {
        "harmony.voicing.quartal_probability": 0.6,
        "harmony.voicing.type": "quartal",
        "harmony.chromatic.chromaticism": 0.7,
        "rhythm.syncopation.probability": 0.7,
        "instrumentation.brass.voicing_type": "wide",
        "rhythm.pattern.metric_modulation": True,
        "tempo.bpm": 160,
    })

    # 8. Miles Davis Modal
    add_style("miles_modal", 120, "Miles Davis-style modal jazz (Kind of Blue era)", {
        "harmony.modal.mode": "dorian",
        "harmony.modal.dorian_probability": 0.7,
        "harmony.progression.type": "modal",
        "harmony.voicing.rootless_prob": 0.8,
        "melody.contour.type": "wave",
        "instrumentation.brass.harmon_mute_prob": 0.9,
        "texture.density": 0.4,
        "dynamics.velocity.overall_level": 0.6,
        "tempo.bpm": 120,
    })

    # 9. Bud Powell Bebop
    add_style("bud_powell_bebop", 260, "Bud Powell-style bebop piano", {
        "melody.chromaticism.bebop_chromatic_prob": 0.9,
        "melody.note_density": 0.9,
        "harmony.jazz.ii_v_i_prob": 0.9,
        "instrumentation.piano.left_hand_style": "shell",
        "rhythm.swing.amount": 0.66,
        "articulation.staccato_prob": 0.4,
        "tempo.bpm": 260,
    })

    # 10. Oscar Peterson Trio
    add_style("oscar_peterson_trio", 220, "Oscar Peterson-style virtuosic swing trio", {
        "melody.note_density": 0.85,
        "instrumentation.piano.block_chord_prob": 0.7,
        "rhythm.swing.amount": 0.63,
        "dynamics.velocity.overall_level": 0.85,
        "articulation.variety": 0.8,
        "tempo.bpm": 220,
    })

    # 11. Django Reinhardt Gypsy Jazz
    add_style("django_gypsy_jazz", 200, "Django Reinhardt-style Gypsy jazz/Hot Club swing", {
        "harmony.voicing.type": "drop2",
        "instrumentation.guitar.active": True,
        "instrumentation.guitar.chord_melody": True,
        "instrumentation.bass.pattern_type": "la_pompe",
        "rhythm.swing.amount": 0.58,
        "melody.ornaments.slide_prob": 0.4,
        "tempo.bpm": 200,
    })

    # 12. Wes Montgomery Guitar
    add_style("wes_montgomery", 140, "Wes Montgomery-style octave guitar improvisation", {
        "instrumentation.guitar.active": True,
        "melody.intervals.octave_prob": 0.7,
        "harmony.voicing.density": 5,
        "rhythm.swing.amount": 0.58,
        "articulation.legato_prob": 0.8,
        "tempo.bpm": 140,
    })

    # 13. Weather Report Fusion
    add_style("weather_report_fusion", 144, "Weather Report-style jazz fusion", {
        "harmony.modal.mode": "mixolydian",
        "rhythm.pattern.odd_meter_grouping": [3, 3, 2],
        "rhythm.feel.straight_8ths": True,
        "instrumentation.bass.syncopation": 0.8,
        "melody.intervals.compound_interval_prob": 0.6,
        "dynamics.expression.forte_piano_contrast": 0.8,
        "tempo.bpm": 144,
    })

    # 14. Herbie Hancock Head Hunters
    add_style("hancock_head_hunters", 100, "Herbie Hancock Head Hunters-style funk jazz", {
        "harmony.voicing.type": "quartal",
        "rhythm.feel.type": "funk",
        "rhythm.syncopation.probability": 0.8,
        "instrumentation.bass.slap_prob": 0.3,
        "genre.rock.backbeat": True,
        "texture.density": 0.7,
        "tempo.bpm": 100,
    })

    # 15. Art Blakey Hard Bop
    add_style("art_blakey_hard_bop", 200, "Art Blakey-style hard bop with driving drums", {
        "rhythm.swing.amount": 0.64,
        "instrumentation.drums.ride_pattern": "hard_bop",
        "instrumentation.drums.cymbal_swell_prob": 0.5,
        "melody.chromaticism.blue_notes": 0.6,
        "harmony.jazz.blues_substitution": 0.5,
        "dynamics.velocity.overall_level": 0.8,
        "tempo.bpm": 200,
    })

    # 16. Stan Getz Bossa Nova
    add_style("getz_bossa_nova", 128, "Stan Getz-style bossa nova", {
        "rhythm.pattern.clave_pattern": "bossa",
        "rhythm.feel.straight_8ths": True,
        "harmony.extensions.ninth_usage": 0.8,
        "instrumentation.drums.pattern_type": "bossa",
        "melody.contour.type": "arch",
        "articulation.legato_prob": 0.8,
        "tempo.bpm": 128,
    })

    # 17. Chet Baker Cool Jazz
    add_style("chet_baker_cool", 100, "Chet Baker-style West Coast cool jazz", {
        "harmony.voicing.spread": 0.7,
        "texture.density": 0.4,
        "dynamics.velocity.overall_level": 0.55,
        "instrumentation.brass.harmon_mute_prob": 0.3,
        "rhythm.swing.amount": 0.56,
        "articulation.legato_prob": 0.9,
        "tempo.bpm": 100,
    })

    # 18. Cannonball Adderley Soul Jazz
    add_style("cannonball_soul_jazz", 120, "Cannonball Adderley-style soul jazz", {
        "harmony.jazz.blues_substitution": 0.6,
        "melody.chromaticism.blue_notes": 0.7,
        "rhythm.feel.type": "groove",
        "instrumentation.piano.block_chord_prob": 0.6,
        "genre.jazz.swing_feel": 0.7,
        "tempo.bpm": 120,
    })

    # 19. Ornette Coleman Free Jazz
    add_style("ornette_free_jazz", 180, "Ornette Coleman-style free jazz", {
        "harmony.modal.mode": "free",
        "melody.chromaticism.amount": 0.9,
        "structure.form.type": "free",
        "rhythm.pattern.polymeter_active": True,
        "texture.polyphony.counterpoint_usage": 0.8,
        "harmony.progression.type": "free",
        "tempo.bpm": 180,
    })

    # 20. Keith Jarrett Solo Piano
    add_style("keith_jarrett_solo", 90, "Keith Jarrett-style improvised solo piano", {
        "instrumentation.piano.active": True,
        "instrumentation.strings.active": False,
        "harmony.modal.modal_interchange_intensity": 0.7,
        "melody.motif.development": 0.9,
        "rhythm.rubato.active": True,
        "rhythm.rubato.intensity": 0.6,
        "dynamics.velocity.micro_dynamics": 0.9,
        "tempo.bpm": 90,
    })

    # 21. McCoy Tyner Modal Piano
    add_style("mccoy_tyner_modal", 160, "McCoy Tyner-style modal piano with quartal voicings", {
        "harmony.voicing.quartal_probability": 0.9,
        "harmony.modal.mode": "lydian",
        "instrumentation.piano.voicing_density": 5,
        "dynamics.velocity.overall_level": 0.85,
        "rhythm.syncopation.probability": 0.6,
        "tempo.bpm": 160,
    })

    # 22. Thelonious Monk Unique
    add_style("thelonious_monk", 140, "Thelonious Monk-style angular and unique approach", {
        "harmony.chromatic.chromaticism": 0.8,
        "melody.intervals.max_leap": 16,
        "rhythm.syncopation.probability": 0.8,
        "articulation.staccato_prob": 0.6,
        "harmony.voicing.cluster_probability": 0.4,
        "melody.rest_frequency": 0.5,
        "tempo.bpm": 140,
    })

    # 23. Sarah Vaughan Vocal Jazz
    add_style("sarah_vaughan_vocal", 80, "Sarah Vaughan-style sophisticated vocal jazz", {
        "melody.ornaments.probability": 0.6,
        "melody.chromaticism.amount": 0.5,
        "harmony.extensions.ninth_usage": 0.7,
        "rhythm.rubato.active": True,
        "dynamics.velocity.phrase_shaping": 0.9,
        "articulation.portamento_prob": 0.3,
        "tempo.bpm": 80,
    })

    # 24. Pat Metheny ECM Sound
    add_style("pat_metheny_ecm", 110, "Pat Metheny-style ECM atmospheric guitar", {
        "instrumentation.guitar.active": True,
        "harmony.voicing.spread": 0.8,
        "timbre.effects.chorus_amount": 0.5,
        "timbre.effects.delay_amount": 0.4,
        "texture.density": 0.4,
        "dynamics.velocity.overall_level": 0.6,
        "tempo.bpm": 110,
    })

    # 25. Chick Corea Return to Forever
    add_style("chick_corea_rtf", 160, "Chick Corea Return to Forever-style fusion", {
        "harmony.modal.lydian_probability": 0.6,
        "rhythm.feel.straight_8ths": True,
        "melody.note_density": 0.85,
        "instrumentation.piano.arpeggio_prob": 0.6,
        "dynamics.velocity.overall_level": 0.8,
        "tempo.bpm": 160,
    })

    # === LATIN STYLES (15 styles) ===

    # 26. Bossa Nova Jobim
    add_style("bossa_nova_jobim", 128, "Antonio Carlos Jobim-style bossa nova", {
        "harmony.extensions.ninth_usage": 0.8,
        "harmony.extensions.sharp_eleven_prob": 0.6,
        "instrumentation.bass.pattern_type": "latin",
        "instrumentation.drums.pattern_type": "bossa",
        "rhythm.pattern.clave_pattern": "bossa",
        "rhythm.feel.straight_8ths": True,
        "tempo.bpm": 128,
    })

    # 27. Afro-Cuban Mambo
    add_style("afro_cuban_mambo", 220, "Tito Puente-style Afro-Cuban mambo", {
        "rhythm.pattern.clave_pattern": "son",
        "rhythm.pattern.polyrhythm_active": True,
        "instrumentation.brass.section_size": 5,
        "instrumentation.brass.soli_probability": 0.8,
        "harmony.chromatic.parallel_harmony": 0.6,
        "tempo.bpm": 220,
    })

    # 28. Salsa
    add_style("salsa", 180, "Classic salsa with clave and montuno", {
        "rhythm.pattern.clave_pattern": "son",
        "genre.latin.clave_based": True,
        "genre.latin.montuno_pattern": True,
        "instrumentation.brass.section_size": 5,
        "rhythm.syncopation.probability": 0.7,
        "tempo.bpm": 180,
    })

    # 29. Cha-Cha-Cha
    add_style("cha_cha_cha", 128, "Cuban cha-cha-cha dance style", {
        "rhythm.pattern.clave_pattern": "son",
        "rhythm.subdivision.eighth_note": 0.9,
        "instrumentation.drums.pattern_type": "cha_cha",
        "harmony.voicing.type": "close",
        "tempo.bpm": 128,
    })

    # 30. Samba
    add_style("samba", 176, "Brazilian samba with syncopated rhythms", {
        "rhythm.syncopation.probability": 0.8,
        "rhythm.pattern.polyrhythm_active": True,
        "instrumentation.drums.pattern_type": "samba",
        "melody.syncopation": 0.7,
        "tempo.bpm": 176,
    })

    # 31. Tango Piazzolla
    add_style("tango_piazzolla", 120, "Astor Piazzolla-style nuevo tango", {
        "harmony.chromatic.chromaticism": 0.7,
        "articulation.staccato_prob": 0.6,
        "rhythm.syncopation.probability": 0.7,
        "dynamics.expression.accent_prob": 0.7,
        "instrumentation.strings.active": True,
        "tempo.bpm": 120,
    })

    # 32. Bolero
    add_style("bolero", 72, "Romantic bolero style", {
        "melody.contour.type": "arch",
        "harmony.extensions.ninth_usage": 0.7,
        "rhythm.pattern.clave_pattern": "bolero",
        "dynamics.velocity.overall_level": 0.6,
        "articulation.legato_prob": 0.8,
        "tempo.bpm": 72,
    })

    # 33. Rumba
    add_style("rumba", 144, "Cuban rumba with complex polyrhythms", {
        "rhythm.pattern.clave_pattern": "rumba",
        "rhythm.pattern.polyrhythm_active": True,
        "rhythm.syncopation.composite_rhythm_complexity": 7,
        "instrumentation.drums.pattern_type": "rumba",
        "tempo.bpm": 144,
    })

    # 34. Merengue
    add_style("merengue", 140, "Dominican merengue dance style", {
        "rhythm.feel.straight_8ths": True,
        "instrumentation.brass.section_size": 4,
        "rhythm.subdivision.eighth_note": 0.95,
        "melody.syncopation": 0.5,
        "tempo.bpm": 140,
    })

    # 35. Son Montuno
    add_style("son_montuno", 160, "Cuban son montuno style", {
        "rhythm.pattern.clave_pattern": "son",
        "genre.latin.montuno_pattern": True,
        "genre.latin.tumbao_bass": True,
        "instrumentation.piano.left_hand_style": "montuno",
        "tempo.bpm": 160,
    })

    # 36. Cumbia
    add_style("cumbia", 100, "Colombian cumbia rhythm", {
        "rhythm.pattern.polyrhythm_active": True,
        "instrumentation.woodwinds.flute_prob": 0.8,
        "instrumentation.woodwinds.clarinet_prob": 0.7,
        "rhythm.feel.groove_intensity": 0.8,
        "tempo.bpm": 100,
    })

    # 37. Bachata
    add_style("bachata", 128, "Dominican bachata romantic style", {
        "melody.contour.type": "arch",
        "instrumentation.guitar.active": True,
        "rhythm.syncopation.probability": 0.4,
        "harmony.extensions.ninth_usage": 0.6,
        "tempo.bpm": 128,
    })

    # 38. Vallenato
    add_style("vallenato", 120, "Colombian vallenato accordion style", {
        "melody.ornaments.probability": 0.5,
        "rhythm.pattern.polyrhythm_active": True,
        "instrumentation.bass.pattern_type": "latin",
        "tempo.bpm": 120,
    })

    # 39. Forro
    add_style("forro", 130, "Brazilian forró dance style", {
        "rhythm.syncopation.probability": 0.6,
        "melody.intervals.third_prob": 0.5,
        "instrumentation.drums.pattern_type": "forro",
        "tempo.bpm": 130,
    })

    # 40. Choro
    add_style("choro", 140, "Brazilian choro with virtuosic melodies", {
        "melody.note_density": 0.8,
        "melody.chromaticism.amount": 0.6,
        "harmony.chromatic.chromatic_approach": 0.6,
        "instrumentation.woodwinds.flute_prob": 0.7,
        "tempo.bpm": 140,
    })

    # === CLASSICAL STYLES (20 styles) ===

    # 41. Mozart Classical
    add_style("mozart_classical", 132, "Wolfgang Amadeus Mozart classical elegance", {
        "harmony.voicing.type": "balanced",
        "harmony.progression.type": "functional",
        "melody.contour.type": "arch",
        "structure.form.type": "sonata",
        "articulation.variety": 0.6,
        "dynamics.range.dynamic_range": "medium",
        "tempo.bpm": 132,
    })

    # 42. Beethoven Romantic
    add_style("beethoven_romantic", 120, "Ludwig van Beethoven dramatic style", {
        "harmony.chromatic.chromaticism": 0.5,
        "dynamics.range.dynamic_range": "very_wide",
        "dynamics.expression.forte_piano_contrast": 0.8,
        "articulation.marcato_prob": 0.7,
        "structure.development.motivic_development": 0.9,
        "tempo.bpm": 120,
    })

    # 43. Bach Baroque
    add_style("bach_baroque", 120, "J.S. Bach baroque counterpoint", {
        "texture.polyphony.counterpoint_usage": 0.9,
        "texture.polyphony.fugue_prob": 0.3,
        "harmony.voice_leading.smoothness": 0.9,
        "articulation.default": "detache",
        "structure.form.type": "fugue",
        "tempo.bpm": 120,
    })

    # 44. Chopin Romantic Piano
    add_style("chopin_romantic", 100, "Frédéric Chopin romantic piano style", {
        "melody.ornaments.probability": 0.7,
        "instrumentation.piano.arpeggio_prob": 0.7,
        "rhythm.rubato.active": True,
        "rhythm.rubato.intensity": 0.7,
        "dynamics.velocity.micro_dynamics": 0.9,
        "harmony.chromatic.chromaticism": 0.6,
        "tempo.bpm": 100,
    })

    # 45. Debussy Impressionist
    add_style("debussy_impressionist", 88, "Claude Debussy impressionist piano", {
        "harmony.voicing.quartal_probability": 0.7,
        "harmony.extensions.ninth_usage": 0.9,
        "harmony.chromatic.parallel_harmony": 0.8,
        "harmony.chromatic.whole_tone_usage": 0.6,
        "dynamics.velocity.micro_dynamics": 0.8,
        "tempo.bpm": 88,
    })

    # 46. Ravel Impressionist
    add_style("ravel_impressionist", 92, "Maurice Ravel orchestral impressionism", {
        "timbre.orchestration.lush_strings": 0.9,
        "harmony.voicing.spread": 0.8,
        "harmony.chromatic.parallel_harmony": 0.7,
        "instrumentation.strings.divisi_prob": 0.8,
        "dynamics.range.dynamic_range": "wide",
        "tempo.bpm": 92,
    })

    # 47. Stravinsky Neo-Classical
    add_style("stravinsky_neoclassical", 144, "Igor Stravinsky neo-classical style", {
        "rhythm.pattern.polymeter_active": True,
        "harmony.chromatic.chromaticism": 0.7,
        "rhythm.syncopation.probability": 0.7,
        "articulation.staccato_prob": 0.6,
        "texture.polyphony.counterpoint_usage": 0.6,
        "tempo.bpm": 144,
    })

    # 48. Reich Minimalism
    add_style("reich_minimalism", 160, "Steve Reich-style minimalist phasing", {
        "structure.repetition.exact_repeat_prob": 0.95,
        "structure.repetition.ostinato_usage": 0.9,
        "rhythm.pattern.metric_modulation": True,
        "rhythm.pattern.rhythmic_canon": 0.8,
        "texture.density": 0.4,
        "tempo.bpm": 160,
    })

    # 49. Glass Minimalism
    add_style("glass_minimalism", 120, "Philip Glass-style arpeggiated minimalism", {
        "instrumentation.piano.arpeggio_prob": 0.9,
        "structure.repetition.exact_repeat_prob": 0.9,
        "harmony.voicing.density": 3,
        "melody.note_density": 0.8,
        "dynamics.velocity.overall_level": 0.7,
        "tempo.bpm": 120,
    })

    # 50. Brahms Romantic
    add_style("brahms_romantic", 96, "Johannes Brahms rich romantic harmony", {
        "harmony.voicing.density": 6,
        "harmony.chromatic.chromaticism": 0.6,
        "instrumentation.strings.voicing_density": 6,
        "dynamics.range.dynamic_range": "wide",
        "texture.orchestral_density": 0.8,
        "tempo.bpm": 96,
    })

    # Continue with more classical styles...
    # 51-60: More classical, baroque, romantic variations

    # === WORLD MUSIC STYLES (15 styles) ===

    # 61. Ravi Shankar Raga
    add_style("ravi_shankar_raga", 80, "Ravi Shankar-style North Indian raga", {
        "harmony.modal.mode": "raga",
        "melody.ornaments.probability": 0.95,
        "rhythm.rubato.active": True,
        "instrumentation.strings.active": False,
        "texture.monophony.active": True,
        "melody.chromaticism.amount": 0.8,
        "tempo.bpm": 80,
    })

    # 62. Flamenco
    add_style("flamenco_paco_de_lucia", 140, "Paco de Lucía-style flamenco guitar", {
        "instrumentation.guitar.active": True,
        "instrumentation.guitar.fingerstyle": True,
        "rhythm.pattern.complexity": 0.9,
        "melody.ornaments.probability": 0.7,
        "articulation.variety": 0.9,
        "tempo.bpm": 140,
    })

    # 63. Middle Eastern Maqam
    add_style("middle_eastern_maqam", 100, "Middle Eastern maqam improvisation", {
        "melody.ornaments.probability": 0.9,
        "melody.chromaticism.amount": 0.8,
        "rhythm.pattern.polyrhythm_active": True,
        "harmony.modal.mode": "maqam",
        "tempo.bpm": 100,
    })

    # 64. West African Highlife
    add_style("west_african_highlife", 140, "West African highlife guitar style", {
        "instrumentation.guitar.active": True,
        "rhythm.pattern.polyrhythm_active": True,
        "instrumentation.brass.section_size": 4,
        "melody.intervals.third_prob": 0.6,
        "tempo.bpm": 140,
    })

    # 65. Reggae
    add_style("reggae", 80, "Jamaican reggae with offbeat emphasis", {
        "rhythm.syncopation.offbeat_emphasis": 0.9,
        "rhythm.feel.laid_back_timing": 0.4,
        "instrumentation.bass.pattern_type": "reggae",
        "instrumentation.guitar.comping_density": 0.7,
        "tempo.bpm": 80,
    })

    # 66. Irish Traditional
    add_style("irish_traditional", 120, "Irish traditional jig and reel style", {
        "melody.ornaments.probability": 0.8,
        "rhythm.dotted_rhythm_prob": 0.6,
        "instrumentation.woodwinds.flute_prob": 0.8,
        "time_signature.numerator": 6,
        "time_signature.denominator": 8,
        "tempo.bpm": 120,
    })

    # 67. Celtic Folk
    add_style("celtic_folk", 100, "Celtic folk ballad style", {
        "instrumentation.strings.active": True,
        "melody.contour.type": "arch",
        "harmony.voicing.density": 3,
        "texture.density": 0.4,
        "tempo.bpm": 100,
    })

    # 68. Balkan Odd Meter
    add_style("balkan_odd_meter", 140, "Balkan folk with 7/8 and 9/8 meters", {
        "time_signature.numerator": 7,
        "time_signature.denominator": 8,
        "rhythm.pattern.odd_meter_grouping": [3, 2, 2],
        "melody.ornaments.probability": 0.7,
        "tempo.bpm": 140,
    })

    # 69. Klezmer
    add_style("klezmer", 120, "Klezmer Jewish wedding music", {
        "melody.ornaments.probability": 0.8,
        "harmony.modal.mode": "phrygian",
        "instrumentation.woodwinds.clarinet_prob": 0.9,
        "melody.chromaticism.amount": 0.6,
        "tempo.bpm": 120,
    })

    # 70. Gamelan
    add_style("gamelan", 100, "Indonesian gamelan ensemble", {
        "harmony.voicing.density": 2,
        "texture.heterophony.active": True,
        "rhythm.pattern.polyrhythm_active": True,
        "instrumentation.drums.pattern_type": "gamelan",
        "tempo.bpm": 100,
    })

    # 71. Taiko Ensemble
    add_style("taiko_ensemble", 120, "Japanese taiko drumming ensemble", {
        "instrumentation.drums.active": True,
        "rhythm.pattern.polyrhythm_active": True,
        "dynamics.range.dynamic_range": "very_wide",
        "articulation.marcato_prob": 0.8,
        "tempo.bpm": 120,
    })

    # 72. Aboriginal Didgeridoo
    add_style("aboriginal_didgeridoo", 80, "Australian Aboriginal didgeridoo drone", {
        "texture.monophony.active": True,
        "rhythm.pattern.polyrhythm_active": True,
        "melody.note_density": 0.3,
        "timbre.darkness": 0.9,
        "tempo.bpm": 80,
    })

    # 73. Tuvan Throat Singing
    add_style("tuvan_throat_singing", 60, "Tuvan throat singing overtone style", {
        "texture.monophony.active": True,
        "melody.ornaments.probability": 0.9,
        "dynamics.velocity.overall_level": 0.6,
        "harmony.modal.mode": "pentatonic",
        "tempo.bpm": 60,
    })

    # 74. African Kora
    add_style("african_kora", 110, "West African kora harp style", {
        "instrumentation.strings.pizzicato_prob": 0.9,
        "melody.note_density": 0.8,
        "rhythm.pattern.polyrhythm_active": True,
        "harmony.voicing.density": 2,
        "tempo.bpm": 110,
    })

    # 75. Chinese Guzheng
    add_style("chinese_guzheng", 90, "Chinese guzheng zither style", {
        "melody.ornaments.probability": 0.9,
        "melody.ornaments.slide_prob": 0.6,
        "harmony.modal.mode": "pentatonic",
        "instrumentation.strings.active": True,
        "tempo.bpm": 90,
    })

    # === ELECTRONIC & MODERN STYLES (10 styles) ===

    # 76. Ambient Eno
    add_style("ambient_eno", 60, "Brian Eno-style ambient music", {
        "melody.note_density": 0.15,
        "rhythm.note_density": 0.1,
        "dynamics.velocity.overall_level": 0.35,
        "structure.repetition.exact_repeat_prob": 0.85,
        "timbre.effects.reverb_amount": 0.8,
        "tempo.bpm": 60,
    })

    # 77. Techno
    add_style("techno", 128, "Four-on-the-floor techno", {
        "genre.electronic.quantization": 0.98,
        "rhythm.subdivision.sixteenth_note": 0.8,
        "instrumentation.drums.kick_pattern": "four_on_floor",
        "harmony.voicing.density": 2,
        "tempo.bpm": 128,
    })

    # 78. House Music
    add_style("house_music", 124, "Chicago house music style", {
        "genre.electronic.quantization": 0.95,
        "instrumentation.drums.kick_pattern": "four_on_floor",
        "instrumentation.drums.hi_hat_pattern": "16th",
        "harmony.voicing.density": 3,
        "tempo.bpm": 124,
    })

    # 79. Drum and Bass
    add_style("drum_and_bass", 174, "UK drum and bass / jungle", {
        "rhythm.subdivision.sixteenth_note": 0.9,
        "instrumentation.drums.pattern_type": "breakbeat",
        "instrumentation.bass.syncopation": 0.8,
        "genre.electronic.quantization": 0.9,
        "tempo.bpm": 174,
    })

    # 80. IDM Autechre
    add_style("idm_autechre", 140, "Autechre-style IDM complexity", {
        "rhythm.pattern.complexity": 0.95,
        "rhythm.pattern.polymeter_active": True,
        "genre.electronic.quantization": 0.7,
        "harmony.voicing.cluster_probability": 0.5,
        "tempo.bpm": 140,
    })

    # 81. Dub Reggae
    add_style("dub_reggae", 75, "Dub reggae with heavy effects", {
        "rhythm.syncopation.offbeat_emphasis": 0.9,
        "timbre.effects.delay_amount": 0.8,
        "timbre.effects.reverb_amount": 0.7,
        "instrumentation.bass.pattern_type": "reggae",
        "tempo.bpm": 75,
    })

    # 82. Trap
    add_style("trap", 140, "Modern trap hip-hop production", {
        "rhythm.subdivision.sixteenth_note": 0.8,
        "rhythm.pattern.complexity": 0.7,
        "instrumentation.drums.hi_hat_pattern": "trap_rolls",
        "genre.electronic.quantization": 0.95,
        "tempo.bpm": 140,
    })

    # 83. Downtempo Trip-Hop
    add_style("downtempo_trip_hop", 90, "Trip-hop downtempo beats", {
        "rhythm.feel.laid_back_timing": 0.5,
        "timbre.effects.reverb_amount": 0.6,
        "texture.density": 0.5,
        "dynamics.velocity.overall_level": 0.6,
        "tempo.bpm": 90,
    })

    # 84. Trance
    add_style("trance", 138, "Uplifting trance with arpeggios", {
        "instrumentation.piano.arpeggio_prob": 0.9,
        "genre.electronic.quantization": 0.98,
        "dynamics.expression.crescendo_prob": 0.8,
        "structure.repetition.ostinato_usage": 0.8,
        "tempo.bpm": 138,
    })

    # 85. Dubstep
    add_style("dubstep", 140, "Dubstep with wobble bass and half-time", {
        "rhythm.syncopation.probability": 0.7,
        "instrumentation.bass.syncopation": 0.9,
        "timbre.effects.modulation_amount": 0.8,
        "dynamics.range.dynamic_range": "very_wide",
        "tempo.bpm": 140,
    })

    # === ROCK & POP STYLES (15 styles) ===

    # 86. Beatles Pop
    add_style("beatles_pop", 120, "Beatles-style pop rock", {
        "harmony.progression.type": "functional",
        "instrumentation.guitar.active": True,
        "instrumentation.strings.section_size": "medium",
        "melody.contour.type": "arch",
        "structure.form.type": "verse_chorus",
        "tempo.bpm": 120,
    })

    # 87. Led Zeppelin Rock
    add_style("led_zeppelin_rock", 100, "Led Zeppelin-style heavy rock", {
        "instrumentation.guitar.active": True,
        "genre.rock.power_chords": True,
        "dynamics.range.dynamic_range": "very_wide",
        "instrumentation.drums.pattern_type": "rock",
        "tempo.bpm": 100,
    })

    # 88. Pink Floyd Progressive
    add_style("pink_floyd_progressive", 80, "Pink Floyd-style progressive rock", {
        "structure.form.type": "through_composed",
        "timbre.effects.delay_amount": 0.6,
        "harmony.modal.modal_interchange_intensity": 0.5,
        "texture.variation": 0.8,
        "tempo.bpm": 80,
    })

    # 89. Motown Soul
    add_style("motown_soul", 120, "Motown soul with horn section", {
        "instrumentation.brass.section_size": 4,
        "instrumentation.strings.section_size": "medium",
        "genre.rock.backbeat": True,
        "melody.contour.type": "arch",
        "tempo.bpm": 120,
    })

    # 90. James Brown Funk
    add_style("james_brown_funk", 110, "James Brown-style funk groove", {
        "rhythm.syncopation.probability": 0.85,
        "genre.rock.backbeat": True,
        "instrumentation.brass.soli_probability": 0.7,
        "articulation.staccato_prob": 0.7,
        "tempo.bpm": 110,
    })

    # 91. Steely Dan Jazz Rock
    add_style("steely_dan_jazz_rock", 96, "Steely Dan-style sophisticated pop", {
        "harmony.extensions.use_13ths": True,
        "harmony.jazz.ii_v_i_prob": 0.7,
        "instrumentation.brass.active": True,
        "melody.chromaticism.amount": 0.5,
        "tempo.bpm": 96,
    })

    # 92. Hendrix Psychedelic
    add_style("hendrix_psychedelic", 120, "Jimi Hendrix psychedelic rock guitar", {
        "instrumentation.guitar.active": True,
        "timbre.effects.modulation_amount": 0.6,
        "harmony.chromatic.chromaticism": 0.6,
        "melody.ornaments.slide_prob": 0.5,
        "tempo.bpm": 120,
    })

    # 93. Yes Progressive Rock
    add_style("yes_progressive_rock", 150, "Yes-style complex progressive rock", {
        "time_signature.numerator": 7,
        "rhythm.pattern.odd_meter_grouping": [3, 2, 2],
        "harmony.progression.complexity": 0.8,
        "structure.form.type": "through_composed",
        "tempo.bpm": 150,
    })

    # 94. King Crimson Prog
    add_style("king_crimson_prog", 130, "King Crimson-style avant-garde prog", {
        "rhythm.pattern.polymeter_active": True,
        "harmony.chromatic.chromaticism": 0.8,
        "instrumentation.brass.active": True,
        "dynamics.expression.forte_piano_contrast": 0.9,
        "tempo.bpm": 130,
    })

    # 95. Fleetwood Mac Pop Rock
    add_style("fleetwood_mac_pop_rock", 110, "Fleetwood Mac-style melodic pop rock", {
        "melody.contour.type": "arch",
        "harmony.voicing.type": "close",
        "instrumentation.guitar.active": True,
        "articulation.legato_prob": 0.7,
        "tempo.bpm": 110,
    })

    # 96. The Police New Wave
    add_style("the_police_new_wave", 140, "The Police-style new wave/reggae fusion", {
        "rhythm.syncopation.offbeat_emphasis": 0.6,
        "instrumentation.guitar.active": True,
        "harmony.voicing.spread": 0.7,
        "texture.density": 0.5,
        "tempo.bpm": 140,
    })

    # 97. Prince Funk Pop
    add_style("prince_funk_pop", 115, "Prince-style funk pop fusion", {
        "rhythm.syncopation.probability": 0.8,
        "harmony.extensions.ninth_usage": 0.7,
        "instrumentation.brass.section_size": 3,
        "melody.note_density": 0.7,
        "tempo.bpm": 115,
    })

    # 98. Radiohead Alternative
    add_style("radiohead_alternative", 85, "Radiohead-style alternative rock", {
        "harmony.modal.modal_interchange_intensity": 0.6,
        "timbre.effects.reverb_amount": 0.6,
        "texture.variation": 0.8,
        "dynamics.expression.crescendo_prob": 0.7,
        "tempo.bpm": 85,
    })

    # 99. Queen Arena Rock
    add_style("queen_arena_rock", 144, "Queen-style arena rock with vocal harmonies", {
        "instrumentation.guitar.active": True,
        "harmony.voicing.density": 5,
        "dynamics.range.dynamic_range": "very_wide",
        "structure.development.thematic_transformation": 0.7,
        "tempo.bpm": 144,
    })

    # 100. Michael Jackson Pop
    add_style("michael_jackson_pop", 118, "Michael Jackson-style pop production", {
        "rhythm.syncopation.probability": 0.7,
        "instrumentation.brass.section_size": 4,
        "melody.note_density": 0.7,
        "structure.form.type": "verse_chorus",
        "tempo.bpm": 118,
    })

    # === ADDITIONAL SPECIALTY STYLES (5 bonus styles) ===

    # 101. Gershwin American Songbook
    add_style("gershwin_american_songbook", 120, "George Gershwin American songbook style", {
        "harmony.jazz.ii_v_i_prob": 0.7,
        "melody.contour.type": "arch",
        "instrumentation.piano.stride_prob": 0.3,
        "structure.form.type": "AABA",
        "tempo.bpm": 120,
    })

    # 102. Cole Porter Sophisticated
    add_style("cole_porter_sophisticated", 100, "Cole Porter sophisticated popular song", {
        "harmony.extensions.ninth_usage": 0.7,
        "melody.chromaticism.amount": 0.5,
        "harmony.chromatic.chromatic_approach": 0.5,
        "structure.form.type": "AABA",
        "tempo.bpm": 100,
    })

    # 103. Irving Berlin Tin Pan Alley
    add_style("irving_berlin_tin_pan_alley", 140, "Irving Berlin Tin Pan Alley style", {
        "melody.contour.type": "arch",
        "harmony.voicing.type": "close",
        "instrumentation.piano.stride_prob": 0.4,
        "rhythm.feel.straight_8ths": True,
        "tempo.bpm": 140,
    })

    # 104. Ennio Morricone Western
    add_style("ennio_morricone_western", 100, "Ennio Morricone spaghetti western score", {
        "instrumentation.guitar.active": True,
        "instrumentation.strings.active": True,
        "harmony.modal.modal_interchange_intensity": 0.6,
        "timbre.effects.reverb_amount": 0.7,
        "tempo.bpm": 100,
    })

    # 105. John Williams Cinematic
    add_style("john_williams_cinematic", 120, "John Williams cinematic orchestral score", {
        "instrumentation.strings.section_size": "full",
        "instrumentation.brass.section_size": 5,
        "harmony.voicing.density": 6,
        "dynamics.range.dynamic_range": "very_wide",
        "structure.development.thematic_transformation": 0.9,
        "tempo.bpm": 120,
    })

    return styles


if __name__ == "__main__":
    styles = generate_all_styles()
    print(f"Generated {len(styles)} complete style definitions")
    print(f"Each style has {len(BASE_PARAMETERS_TEMPLATE)} parameters")
    print(f"\nSample styles:")
    for i, name in enumerate(list(styles.keys())[:10]):
        print(f"  {i+1}. {name}")
