#!/usr/bin/env python3
"""
Structure & Form Expansion - Agent 2
=====================================

Expands structure parameters from 4 to 60 comprehensive parameters covering:
- Form & Architecture (20 parameters)
- Transitions & Development (20 parameters)
- Repetition & Memory (20 parameters)

This module is part of the self-expanding inverse music generation system,
enabling precise control over musical form, structural development, and
compositional architecture.

Musical Theory Foundation:
--------------------------
- William Caplin: "Classical Form" (phrase structure, cadences)
- James Hepokoski & Warren Darcy: "Elements of Sonata Theory"
- Arnold Schoenberg: "Fundamentals of Musical Composition"
- Leonard Meyer: "Explaining Music" (implication-realization)
- Fred Lerdahl & Ray Jackendoff: "A Generative Theory of Tonal Music" (grouping)

Author: Agent 2 - Structure & Form Expansion Specialist
Date: 2025
License: MIT
"""

import sys
from pathlib import Path

# Handle both relative and absolute imports
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from parameters.universal_registry import (
        ParameterDefinition, ParameterType, ParameterCategory,
        MusicalImpact, REGISTRY
    )
else:
    from .universal_registry import (
        ParameterDefinition, ParameterType, ParameterCategory,
        MusicalImpact, REGISTRY
    )

from typing import Dict, Any


# ============================================================================
# GENRE-SPECIFIC DEFAULT PRESETS
# ============================================================================

def get_jazz_structure_defaults() -> Dict[str, Any]:
    """
    Jazz structure defaults (AABA, 12-bar blues, rhythm changes)

    Musical Rationale:
    - Standard 32-bar AABA form predominant
    - 12-bar blues with turnarounds
    - Solo trading (4s, 8s)
    - Shout chorus for climax
    - Minimal intros/outros (common to start "in one")
    """
    return {
        "structure.form.type": "AABA",
        "structure.form.length_bars": 32,
        "structure.form.intro_bars": 4,
        "structure.form.outro_bars": 4,
        "structure.form.verse_length": 8,
        "structure.form.chorus_length": 8,
        "structure.form.bridge_length": 8,
        "structure.form.solo_section_length": 32,
        "structure.form.vamp_probability": 0.3,
        "structure.form.coda_type": "ritardando",
        "structure.form.tag_repetitions": 2,
        "structure.form.interlude_prob": 0.2,
        "structure.form.solo_trading": True,
        "structure.form.trading_length": 8,
        "structure.form.shout_chorus": True,
        "structure.form.stop_time_prob": 0.4,
        "structure.form.breakdown_prob": 0.1,
        "structure.form.grand_pause_prob": 0.2,
        "structure.form.section_markers": True,
        "structure.form.asymmetric_sections": False,

        "structure.transition.type": "pivot_chord",
        "structure.transition.smoothness": 0.8,
        "structure.transition.drum_fill_prob": 0.7,
        "structure.transition.key_change_prob": 0.3,
        "structure.transition.tempo_change_prob": 0.1,
        "structure.transition.meter_change_prob": 0.05,
        "structure.transition.texture_change_prob": 0.6,
        "structure.transition.dynamic_change_prob": 0.7,
        "structure.transition.turnaround_complexity": 5,
        "structure.transition.pickup_notes": True,
        "structure.development.variation_intensity": 0.7,
        "structure.development.motivic_transformation": 0.6,
        "structure.development.fragmentation_prob": 0.4,
        "structure.development.augmentation_prob": 0.2,
        "structure.development.diminution_prob": 0.3,
        "structure.development.inversion_prob": 0.3,
        "structure.development.retrograde_prob": 0.1,
        "structure.development.sequence_prob": 0.5,
        "structure.development.sequence_interval": 2,
        "structure.development.climax_placement": 0.75,

        "structure.repetition.exact_repeat_prob": 0.3,
        "structure.repetition.varied_repeat_prob": 0.7,
        "structure.repetition.variation_amount": 0.6,
        "structure.repetition.motif_recall_prob": 0.8,
        "structure.repetition.theme_return_prob": 0.9,
        "structure.repetition.ostinato_usage": 0.4,
        "structure.repetition.pedal_point_duration": 8,
        "structure.repetition.riff_based": False,
        "structure.repetition.hook_emphasis": 0.5,
        "structure.repetition.call_response": True,
        "structure.repetition.phrase_length": 4,
        "structure.repetition.phrase_grouping": "regular",
        "structure.repetition.antecedent_consequent": True,
        "structure.repetition.parallel_period": True,
        "structure.repetition.contrast_middle": True,
        "structure.repetition.return_variation": 0.6,
        "structure.repetition.thematic_unity": 0.8,
        "structure.repetition.motivic_consistency": 0.7,
        "structure.repetition.rhythmic_motif_recurrence": 0.8,
        "structure.repetition.melodic_cell_recurrence": 0.7,
    }


def get_blues_structure_defaults() -> Dict[str, Any]:
    """
    Blues structure defaults (12-bar form, AAB lyric structure)

    Musical Rationale:
    - 12-bar blues form is fundamental
    - AAB phrase structure (call and response)
    - Minimal development (statement/restatement aesthetic)
    - Turnarounds are essential
    - Repetition over variation
    """
    return {
        "structure.form.type": "blues",
        "structure.form.length_bars": 12,
        "structure.form.intro_bars": 0,
        "structure.form.outro_bars": 4,
        "structure.form.verse_length": 12,
        "structure.form.chorus_length": 12,
        "structure.form.bridge_length": 8,
        "structure.form.solo_section_length": 12,
        "structure.form.vamp_probability": 0.5,
        "structure.form.coda_type": "fade",
        "structure.form.tag_repetitions": 3,
        "structure.form.interlude_prob": 0.1,
        "structure.form.solo_trading": False,
        "structure.form.trading_length": 4,
        "structure.form.shout_chorus": False,
        "structure.form.stop_time_prob": 0.2,
        "structure.form.breakdown_prob": 0.05,
        "structure.form.grand_pause_prob": 0.1,
        "structure.form.section_markers": False,
        "structure.form.asymmetric_sections": False,

        "structure.transition.type": "direct",
        "structure.transition.smoothness": 0.6,
        "structure.transition.drum_fill_prob": 0.4,
        "structure.transition.key_change_prob": 0.1,
        "structure.transition.tempo_change_prob": 0.05,
        "structure.transition.meter_change_prob": 0.0,
        "structure.transition.texture_change_prob": 0.3,
        "structure.transition.dynamic_change_prob": 0.5,
        "structure.transition.turnaround_complexity": 4,
        "structure.transition.pickup_notes": True,
        "structure.development.variation_intensity": 0.3,
        "structure.development.motivic_transformation": 0.2,
        "structure.development.fragmentation_prob": 0.1,
        "structure.development.augmentation_prob": 0.05,
        "structure.development.diminution_prob": 0.1,
        "structure.development.inversion_prob": 0.05,
        "structure.development.retrograde_prob": 0.0,
        "structure.development.sequence_prob": 0.2,
        "structure.development.sequence_interval": 2,
        "structure.development.climax_placement": 0.7,

        "structure.repetition.exact_repeat_prob": 0.7,
        "structure.repetition.varied_repeat_prob": 0.3,
        "structure.repetition.variation_amount": 0.3,
        "structure.repetition.motif_recall_prob": 0.9,
        "structure.repetition.theme_return_prob": 1.0,
        "structure.repetition.ostinato_usage": 0.6,
        "structure.repetition.pedal_point_duration": 4,
        "structure.repetition.riff_based": True,
        "structure.repetition.hook_emphasis": 0.8,
        "structure.repetition.call_response": True,
        "structure.repetition.phrase_length": 4,
        "structure.repetition.phrase_grouping": "regular",
        "structure.repetition.antecedent_consequent": True,
        "structure.repetition.parallel_period": False,
        "structure.repetition.contrast_middle": False,
        "structure.repetition.return_variation": 0.2,
        "structure.repetition.thematic_unity": 0.9,
        "structure.repetition.motivic_consistency": 0.9,
        "structure.repetition.rhythmic_motif_recurrence": 0.9,
        "structure.repetition.melodic_cell_recurrence": 0.8,
    }


def get_rock_structure_defaults() -> Dict[str, Any]:
    """
    Classic rock structure defaults (verse-chorus form)

    Musical Rationale:
    - Verse-Chorus-Bridge form predominant
    - 4 and 8 bar phrases
    - Extended guitar solos
    - Big drum fills on transitions
    - Power chord riff-based construction
    """
    return {
        "structure.form.type": "ABAB",
        "structure.form.length_bars": 64,
        "structure.form.intro_bars": 4,
        "structure.form.outro_bars": 8,
        "structure.form.verse_length": 16,
        "structure.form.chorus_length": 8,
        "structure.form.bridge_length": 8,
        "structure.form.solo_section_length": 16,
        "structure.form.vamp_probability": 0.2,
        "structure.form.coda_type": "fade",
        "structure.form.tag_repetitions": 4,
        "structure.form.interlude_prob": 0.3,
        "structure.form.solo_trading": False,
        "structure.form.trading_length": 4,
        "structure.form.shout_chorus": False,
        "structure.form.stop_time_prob": 0.15,
        "structure.form.breakdown_prob": 0.3,
        "structure.form.grand_pause_prob": 0.1,
        "structure.form.section_markers": True,
        "structure.form.asymmetric_sections": False,

        "structure.transition.type": "direct",
        "structure.transition.smoothness": 0.4,
        "structure.transition.drum_fill_prob": 0.9,
        "structure.transition.key_change_prob": 0.05,
        "structure.transition.tempo_change_prob": 0.02,
        "structure.transition.meter_change_prob": 0.01,
        "structure.transition.texture_change_prob": 0.7,
        "structure.transition.dynamic_change_prob": 0.8,
        "structure.transition.turnaround_complexity": 2,
        "structure.transition.pickup_notes": False,
        "structure.development.variation_intensity": 0.4,
        "structure.development.motivic_transformation": 0.3,
        "structure.development.fragmentation_prob": 0.2,
        "structure.development.augmentation_prob": 0.1,
        "structure.development.diminution_prob": 0.1,
        "structure.development.inversion_prob": 0.1,
        "structure.development.retrograde_prob": 0.05,
        "structure.development.sequence_prob": 0.3,
        "structure.development.sequence_interval": 2,
        "structure.development.climax_placement": 0.8,

        "structure.repetition.exact_repeat_prob": 0.6,
        "structure.repetition.varied_repeat_prob": 0.4,
        "structure.repetition.variation_amount": 0.4,
        "structure.repetition.motif_recall_prob": 0.8,
        "structure.repetition.theme_return_prob": 0.9,
        "structure.repetition.ostinato_usage": 0.7,
        "structure.repetition.pedal_point_duration": 8,
        "structure.repetition.riff_based": True,
        "structure.repetition.hook_emphasis": 0.9,
        "structure.repetition.call_response": False,
        "structure.repetition.phrase_length": 4,
        "structure.repetition.phrase_grouping": "regular",
        "structure.repetition.antecedent_consequent": False,
        "structure.repetition.parallel_period": False,
        "structure.repetition.contrast_middle": True,
        "structure.repetition.return_variation": 0.3,
        "structure.repetition.thematic_unity": 0.8,
        "structure.repetition.motivic_consistency": 0.8,
        "structure.repetition.rhythmic_motif_recurrence": 0.9,
        "structure.repetition.melodic_cell_recurrence": 0.7,
    }


def get_pop_structure_defaults() -> Dict[str, Any]:
    """
    Pop music structure defaults (verse-chorus with intro/outro)

    Musical Rationale:
    - Verse-Chorus form with pre-chorus optional
    - Short intro (4-8 bars)
    - Bridge for contrast at 2/3 point
    - Hook repetition essential
    - Clear section markers
    """
    return {
        "structure.form.type": "ABAB",
        "structure.form.length_bars": 48,
        "structure.form.intro_bars": 8,
        "structure.form.outro_bars": 8,
        "structure.form.verse_length": 8,
        "structure.form.chorus_length": 8,
        "structure.form.bridge_length": 8,
        "structure.form.solo_section_length": 8,
        "structure.form.vamp_probability": 0.1,
        "structure.form.coda_type": "fade",
        "structure.form.tag_repetitions": 2,
        "structure.form.interlude_prob": 0.2,
        "structure.form.solo_trading": False,
        "structure.form.trading_length": 4,
        "structure.form.shout_chorus": False,
        "structure.form.stop_time_prob": 0.1,
        "structure.form.breakdown_prob": 0.4,
        "structure.form.grand_pause_prob": 0.15,
        "structure.form.section_markers": True,
        "structure.form.asymmetric_sections": False,

        "structure.transition.type": "common_tone",
        "structure.transition.smoothness": 0.7,
        "structure.transition.drum_fill_prob": 0.6,
        "structure.transition.key_change_prob": 0.2,
        "structure.transition.tempo_change_prob": 0.05,
        "structure.transition.meter_change_prob": 0.02,
        "structure.transition.texture_change_prob": 0.8,
        "structure.transition.dynamic_change_prob": 0.8,
        "structure.transition.turnaround_complexity": 3,
        "structure.transition.pickup_notes": True,
        "structure.development.variation_intensity": 0.5,
        "structure.development.motivic_transformation": 0.4,
        "structure.development.fragmentation_prob": 0.2,
        "structure.development.augmentation_prob": 0.1,
        "structure.development.diminution_prob": 0.2,
        "structure.development.inversion_prob": 0.1,
        "structure.development.retrograde_prob": 0.05,
        "structure.development.sequence_prob": 0.3,
        "structure.development.sequence_interval": 2,
        "structure.development.climax_placement": 0.75,

        "structure.repetition.exact_repeat_prob": 0.5,
        "structure.repetition.varied_repeat_prob": 0.5,
        "structure.repetition.variation_amount": 0.4,
        "structure.repetition.motif_recall_prob": 0.8,
        "structure.repetition.theme_return_prob": 0.9,
        "structure.repetition.ostinato_usage": 0.5,
        "structure.repetition.pedal_point_duration": 4,
        "structure.repetition.riff_based": False,
        "structure.repetition.hook_emphasis": 1.0,
        "structure.repetition.call_response": False,
        "structure.repetition.phrase_length": 4,
        "structure.repetition.phrase_grouping": "regular",
        "structure.repetition.antecedent_consequent": True,
        "structure.repetition.parallel_period": True,
        "structure.repetition.contrast_middle": True,
        "structure.repetition.return_variation": 0.4,
        "structure.repetition.thematic_unity": 0.8,
        "structure.repetition.motivic_consistency": 0.7,
        "structure.repetition.rhythmic_motif_recurrence": 0.8,
        "structure.repetition.melodic_cell_recurrence": 0.9,
    }


def get_country_structure_defaults() -> Dict[str, Any]:
    """
    Country music structure defaults (verse-chorus with storytelling)

    Musical Rationale:
    - Narrative verse-chorus structure
    - Instrumental breaks (fiddle, steel guitar)
    - Pickup notes common
    - Tag endings
    - Simple, direct transitions
    """
    return {
        "structure.form.type": "ABAB",
        "structure.form.length_bars": 64,
        "structure.form.intro_bars": 4,
        "structure.form.outro_bars": 8,
        "structure.form.verse_length": 16,
        "structure.form.chorus_length": 8,
        "structure.form.bridge_length": 8,
        "structure.form.solo_section_length": 16,
        "structure.form.vamp_probability": 0.1,
        "structure.form.coda_type": "tag",
        "structure.form.tag_repetitions": 3,
        "structure.form.interlude_prob": 0.4,
        "structure.form.solo_trading": False,
        "structure.form.trading_length": 4,
        "structure.form.shout_chorus": False,
        "structure.form.stop_time_prob": 0.05,
        "structure.form.breakdown_prob": 0.2,
        "structure.form.grand_pause_prob": 0.05,
        "structure.form.section_markers": True,
        "structure.form.asymmetric_sections": False,

        "structure.transition.type": "direct",
        "structure.transition.smoothness": 0.6,
        "structure.transition.drum_fill_prob": 0.5,
        "structure.transition.key_change_prob": 0.3,
        "structure.transition.tempo_change_prob": 0.05,
        "structure.transition.meter_change_prob": 0.05,
        "structure.transition.texture_change_prob": 0.6,
        "structure.transition.dynamic_change_prob": 0.6,
        "structure.transition.turnaround_complexity": 3,
        "structure.transition.pickup_notes": True,
        "structure.development.variation_intensity": 0.4,
        "structure.development.motivic_transformation": 0.3,
        "structure.development.fragmentation_prob": 0.2,
        "structure.development.augmentation_prob": 0.1,
        "structure.development.diminution_prob": 0.1,
        "structure.development.inversion_prob": 0.1,
        "structure.development.retrograde_prob": 0.0,
        "structure.development.sequence_prob": 0.2,
        "structure.development.sequence_interval": 2,
        "structure.development.climax_placement": 0.75,

        "structure.repetition.exact_repeat_prob": 0.6,
        "structure.repetition.varied_repeat_prob": 0.4,
        "structure.repetition.variation_amount": 0.3,
        "structure.repetition.motif_recall_prob": 0.7,
        "structure.repetition.theme_return_prob": 0.9,
        "structure.repetition.ostinato_usage": 0.3,
        "structure.repetition.pedal_point_duration": 4,
        "structure.repetition.riff_based": False,
        "structure.repetition.hook_emphasis": 0.9,
        "structure.repetition.call_response": False,
        "structure.repetition.phrase_length": 4,
        "structure.repetition.phrase_grouping": "regular",
        "structure.repetition.antecedent_consequent": True,
        "structure.repetition.parallel_period": True,
        "structure.repetition.contrast_middle": True,
        "structure.repetition.return_variation": 0.3,
        "structure.repetition.thematic_unity": 0.8,
        "structure.repetition.motivic_consistency": 0.7,
        "structure.repetition.rhythmic_motif_recurrence": 0.7,
        "structure.repetition.melodic_cell_recurrence": 0.8,
    }


def get_gospel_structure_defaults() -> Dict[str, Any]:
    """
    Gospel music structure defaults (verse-chorus with vamp)

    Musical Rationale:
    - Call-and-response fundamental
    - Extended vamps for emotional intensity
    - Modulations upward for climax
    - Repetitive hook emphasis
    - Thematic unity through restatement
    """
    return {
        "structure.form.type": "ABAB",
        "structure.form.length_bars": 48,
        "structure.form.intro_bars": 8,
        "structure.form.outro_bars": 16,
        "structure.form.verse_length": 16,
        "structure.form.chorus_length": 8,
        "structure.form.bridge_length": 8,
        "structure.form.solo_section_length": 8,
        "structure.form.vamp_probability": 0.8,
        "structure.form.coda_type": "fade",
        "structure.form.tag_repetitions": 4,
        "structure.form.interlude_prob": 0.3,
        "structure.form.solo_trading": False,
        "structure.form.trading_length": 4,
        "structure.form.shout_chorus": True,
        "structure.form.stop_time_prob": 0.2,
        "structure.form.breakdown_prob": 0.3,
        "structure.form.grand_pause_prob": 0.2,
        "structure.form.section_markers": True,
        "structure.form.asymmetric_sections": False,

        "structure.transition.type": "chromatic",
        "structure.transition.smoothness": 0.6,
        "structure.transition.drum_fill_prob": 0.7,
        "structure.transition.key_change_prob": 0.5,
        "structure.transition.tempo_change_prob": 0.1,
        "structure.transition.meter_change_prob": 0.05,
        "structure.transition.texture_change_prob": 0.8,
        "structure.transition.dynamic_change_prob": 0.9,
        "structure.transition.turnaround_complexity": 4,
        "structure.transition.pickup_notes": True,
        "structure.development.variation_intensity": 0.6,
        "structure.development.motivic_transformation": 0.5,
        "structure.development.fragmentation_prob": 0.3,
        "structure.development.augmentation_prob": 0.2,
        "structure.development.diminution_prob": 0.2,
        "structure.development.inversion_prob": 0.2,
        "structure.development.retrograde_prob": 0.1,
        "structure.development.sequence_prob": 0.4,
        "structure.development.sequence_interval": 1,
        "structure.development.climax_placement": 0.85,

        "structure.repetition.exact_repeat_prob": 0.4,
        "structure.repetition.varied_repeat_prob": 0.6,
        "structure.repetition.variation_amount": 0.5,
        "structure.repetition.motif_recall_prob": 0.9,
        "structure.repetition.theme_return_prob": 0.95,
        "structure.repetition.ostinato_usage": 0.7,
        "structure.repetition.pedal_point_duration": 8,
        "structure.repetition.riff_based": False,
        "structure.repetition.hook_emphasis": 0.95,
        "structure.repetition.call_response": True,
        "structure.repetition.phrase_length": 4,
        "structure.repetition.phrase_grouping": "regular",
        "structure.repetition.antecedent_consequent": True,
        "structure.repetition.parallel_period": True,
        "structure.repetition.contrast_middle": True,
        "structure.repetition.return_variation": 0.5,
        "structure.repetition.thematic_unity": 0.9,
        "structure.repetition.motivic_consistency": 0.8,
        "structure.repetition.rhythmic_motif_recurrence": 0.85,
        "structure.repetition.melodic_cell_recurrence": 0.9,
    }


def get_funk_soul_structure_defaults() -> Dict[str, Any]:
    """
    Funk/Soul structure defaults (groove-based with breakdowns)

    Musical Rationale:
    - Riff and groove-based construction
    - Extended breakdowns for rhythmic focus
    - Minimal harmonic development
    - Heavy use of ostinato
    - Stop-time for dramatic effect
    """
    return {
        "structure.form.type": "ABAB",
        "structure.form.length_bars": 64,
        "structure.form.intro_bars": 4,
        "structure.form.outro_bars": 8,
        "structure.form.verse_length": 16,
        "structure.form.chorus_length": 8,
        "structure.form.bridge_length": 8,
        "structure.form.solo_section_length": 16,
        "structure.form.vamp_probability": 0.6,
        "structure.form.coda_type": "fade",
        "structure.form.tag_repetitions": 3,
        "structure.form.interlude_prob": 0.4,
        "structure.form.solo_trading": False,
        "structure.form.trading_length": 4,
        "structure.form.shout_chorus": False,
        "structure.form.stop_time_prob": 0.5,
        "structure.form.breakdown_prob": 0.7,
        "structure.form.grand_pause_prob": 0.15,
        "structure.form.section_markers": True,
        "structure.form.asymmetric_sections": True,

        "structure.transition.type": "direct",
        "structure.transition.smoothness": 0.5,
        "structure.transition.drum_fill_prob": 0.8,
        "structure.transition.key_change_prob": 0.1,
        "structure.transition.tempo_change_prob": 0.05,
        "structure.transition.meter_change_prob": 0.1,
        "structure.transition.texture_change_prob": 0.7,
        "structure.transition.dynamic_change_prob": 0.8,
        "structure.transition.turnaround_complexity": 3,
        "structure.transition.pickup_notes": True,
        "structure.development.variation_intensity": 0.5,
        "structure.development.motivic_transformation": 0.4,
        "structure.development.fragmentation_prob": 0.3,
        "structure.development.augmentation_prob": 0.15,
        "structure.development.diminution_prob": 0.2,
        "structure.development.inversion_prob": 0.15,
        "structure.development.retrograde_prob": 0.1,
        "structure.development.sequence_prob": 0.3,
        "structure.development.sequence_interval": 2,
        "structure.development.climax_placement": 0.75,

        "structure.repetition.exact_repeat_prob": 0.7,
        "structure.repetition.varied_repeat_prob": 0.3,
        "structure.repetition.variation_amount": 0.3,
        "structure.repetition.motif_recall_prob": 0.9,
        "structure.repetition.theme_return_prob": 0.95,
        "structure.repetition.ostinato_usage": 0.9,
        "structure.repetition.pedal_point_duration": 16,
        "structure.repetition.riff_based": True,
        "structure.repetition.hook_emphasis": 0.9,
        "structure.repetition.call_response": True,
        "structure.repetition.phrase_length": 2,
        "structure.repetition.phrase_grouping": "irregular",
        "structure.repetition.antecedent_consequent": False,
        "structure.repetition.parallel_period": False,
        "structure.repetition.contrast_middle": False,
        "structure.repetition.return_variation": 0.2,
        "structure.repetition.thematic_unity": 0.85,
        "structure.repetition.motivic_consistency": 0.9,
        "structure.repetition.rhythmic_motif_recurrence": 0.95,
        "structure.repetition.melodic_cell_recurrence": 0.8,
    }


def get_electronic_structure_defaults() -> Dict[str, Any]:
    """
    Electronic music structure defaults (build-drop structure)

    Musical Rationale:
    - Build-drop architecture (tension-release)
    - Extended breakdowns for DJ mixing
    - Minimal melodic development
    - Section markers for structure
    - Asymmetric phrases common
    """
    return {
        "structure.form.type": "through_composed",
        "structure.form.length_bars": 64,
        "structure.form.intro_bars": 16,
        "structure.form.outro_bars": 16,
        "structure.form.verse_length": 16,
        "structure.form.chorus_length": 16,
        "structure.form.bridge_length": 8,
        "structure.form.solo_section_length": 16,
        "structure.form.vamp_probability": 0.7,
        "structure.form.coda_type": "fade",
        "structure.form.tag_repetitions": 1,
        "structure.form.interlude_prob": 0.5,
        "structure.form.solo_trading": False,
        "structure.form.trading_length": 4,
        "structure.form.shout_chorus": False,
        "structure.form.stop_time_prob": 0.1,
        "structure.form.breakdown_prob": 0.9,
        "structure.form.grand_pause_prob": 0.3,
        "structure.form.section_markers": True,
        "structure.form.asymmetric_sections": True,

        "structure.transition.type": "sequential",
        "structure.transition.smoothness": 0.3,
        "structure.transition.drum_fill_prob": 0.5,
        "structure.transition.key_change_prob": 0.2,
        "structure.transition.tempo_change_prob": 0.1,
        "structure.transition.meter_change_prob": 0.05,
        "structure.transition.texture_change_prob": 0.9,
        "structure.transition.dynamic_change_prob": 0.9,
        "structure.transition.turnaround_complexity": 2,
        "structure.transition.pickup_notes": False,
        "structure.development.variation_intensity": 0.6,
        "structure.development.motivic_transformation": 0.5,
        "structure.development.fragmentation_prob": 0.5,
        "structure.development.augmentation_prob": 0.3,
        "structure.development.diminution_prob": 0.3,
        "structure.development.inversion_prob": 0.2,
        "structure.development.retrograde_prob": 0.2,
        "structure.development.sequence_prob": 0.6,
        "structure.development.sequence_interval": 1,
        "structure.development.climax_placement": 0.7,

        "structure.repetition.exact_repeat_prob": 0.8,
        "structure.repetition.varied_repeat_prob": 0.2,
        "structure.repetition.variation_amount": 0.3,
        "structure.repetition.motif_recall_prob": 0.9,
        "structure.repetition.theme_return_prob": 0.8,
        "structure.repetition.ostinato_usage": 0.9,
        "structure.repetition.pedal_point_duration": 32,
        "structure.repetition.riff_based": True,
        "structure.repetition.hook_emphasis": 0.85,
        "structure.repetition.call_response": False,
        "structure.repetition.phrase_length": 8,
        "structure.repetition.phrase_grouping": "overlapping",
        "structure.repetition.antecedent_consequent": False,
        "structure.repetition.parallel_period": False,
        "structure.repetition.contrast_middle": False,
        "structure.repetition.return_variation": 0.4,
        "structure.repetition.thematic_unity": 0.7,
        "structure.repetition.motivic_consistency": 0.8,
        "structure.repetition.rhythmic_motif_recurrence": 0.9,
        "structure.repetition.melodic_cell_recurrence": 0.7,
    }


# ============================================================================
# PARAMETER REGISTRATION
# ============================================================================

def register_form_architecture_parameters():
    """
    Register 20 Form & Architecture parameters

    These parameters control the overall formal structure of compositions:
    - Form types (AABA, ABAB, blues, sonata, etc.)
    - Section lengths (verse, chorus, bridge, intro, outro)
    - Structural probabilities (vamps, codas, tags)
    - Special sections (solo trading, shout chorus, breakdowns)
    """

    # ========================================================================
    # Basic Form Type and Length
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="form_type",
        full_path="structure.form.type",
        description="Primary form structure type (AABA, ABAB, blues, sonata, etc.)",
        param_type=ParameterType.CATEGORICAL,
        options=["AABA", "ABAB", "ABAC", "AAA", "through_composed", "binary",
                 "ternary", "rondo", "sonata", "blues", "rhythm_changes"],
        default_value="AABA",
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "pop", "rock", "classical", "blues"],
        module_file="generators/form_generator.py"
    ))

    REGISTRY.register(ParameterDefinition(
        name="form_length_bars",
        full_path="structure.form.length_bars",
        description="Total form length in bars (standard: 12, 16, 24, 32, 48, 64)",
        param_type=ParameterType.CATEGORICAL,
        options=[12, 16, 24, 32, 48, 64],
        default_value=32,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "pop", "rock", "blues"]
    ))

    # ========================================================================
    # Section Lengths
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="intro_bars",
        full_path="structure.form.intro_bars",
        description="Introduction length in bars (0 = no intro)",
        param_type=ParameterType.CATEGORICAL,
        options=[0, 2, 4, 8, 16],
        default_value=4,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["pop", "rock", "electronic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="outro_bars",
        full_path="structure.form.outro_bars",
        description="Outro/coda length in bars (0 = direct ending)",
        param_type=ParameterType.CATEGORICAL,
        options=[0, 2, 4, 8, 16],
        default_value=4,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["pop", "rock", "jazz"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="verse_length",
        full_path="structure.form.verse_length",
        description="Verse section length in bars",
        param_type=ParameterType.CATEGORICAL,
        options=[8, 16, 24, 32],
        default_value=8,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["pop", "rock", "country", "gospel"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="chorus_length",
        full_path="structure.form.chorus_length",
        description="Chorus section length in bars",
        param_type=ParameterType.CATEGORICAL,
        options=[8, 16, 24, 32],
        default_value=8,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["pop", "rock", "country", "gospel"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="bridge_length",
        full_path="structure.form.bridge_length",
        description="Bridge/middle eight section length in bars",
        param_type=ParameterType.CATEGORICAL,
        options=[4, 8, 16],
        default_value=8,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["pop", "rock", "jazz"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="solo_section_length",
        full_path="structure.form.solo_section_length",
        description="Solo/improvisation section length in bars",
        param_type=ParameterType.CATEGORICAL,
        options=[12, 16, 24, 32, 64],
        default_value=32,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "blues", "rock"]
    ))

    # ========================================================================
    # Structural Probabilities and Special Sections
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="vamp_probability",
        full_path="structure.form.vamp_probability",
        description="Probability of extended vamp sections for improvisation/intensity",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "funk", "gospel", "electronic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="coda_type",
        full_path="structure.form.coda_type",
        description="Type of ending/coda",
        param_type=ParameterType.CATEGORICAL,
        options=["none", "fade", "ritardando", "fermata", "tag"],
        default_value="fade",
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["pop", "rock", "jazz", "classical"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="tag_repetitions",
        full_path="structure.form.tag_repetitions",
        description="Number of tag ending repetitions (country, gospel style)",
        param_type=ParameterType.CATEGORICAL,
        options=[0, 1, 2, 3, 4],
        default_value=2,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["country", "gospel", "blues"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="interlude_prob",
        full_path="structure.form.interlude_prob",
        description="Probability of instrumental interlude sections",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["country", "funk", "electronic"]
    ))

    # ========================================================================
    # Jazz-Specific Structural Elements
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="solo_trading",
        full_path="structure.form.solo_trading",
        description="Enable solo trading (4s, 8s, 12s) between instruments",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "bebop"],
        constraint_description="Typically used in jazz contexts for dynamic solo exchanges"
    ))

    REGISTRY.register(ParameterDefinition(
        name="trading_length",
        full_path="structure.form.trading_length",
        description="Length of solo trading exchanges in bars (4s, 8s, etc.)",
        param_type=ParameterType.CATEGORICAL,
        options=[2, 4, 8, 12, 16],
        default_value=8,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "bebop"],
        depends_on=["structure.form.solo_trading"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="shout_chorus",
        full_path="structure.form.shout_chorus",
        description="Include shout chorus (climactic ensemble section)",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "big_band", "gospel"]
    ))

    # ========================================================================
    # Rhythmic and Textural Structural Elements
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="stop_time_prob",
        full_path="structure.form.stop_time_prob",
        description="Probability of stop-time sections (rhythm section drops out)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "funk", "blues"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="breakdown_prob",
        full_path="structure.form.breakdown_prob",
        description="Probability of breakdown sections (reduced texture/dynamics)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["electronic", "funk", "pop", "rock"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="grand_pause_prob",
        full_path="structure.form.grand_pause_prob",
        description="Probability of grand pause (complete silence for dramatic effect)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "pop", "electronic"]
    ))

    # ========================================================================
    # Formal Organization
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="section_markers",
        full_path="structure.form.section_markers",
        description="Use explicit rehearsal marks/section boundaries in output",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["classical", "jazz", "big_band"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="asymmetric_sections",
        full_path="structure.form.asymmetric_sections",
        description="Allow asymmetric phrase lengths (7, 11, 13 bars)",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["progressive", "electronic", "funk", "contemporary"]
    ))


def register_transition_development_parameters():
    """
    Register 20 Transitions & Development parameters

    These parameters control how sections connect and themes develop:
    - Transition types (pivot chords, common tones, chromatic)
    - Transition smoothness and complexity
    - Key/tempo/meter changes
    - Motivic development techniques
    """

    # ========================================================================
    # Transition Types and Quality
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="transition_type",
        full_path="structure.transition.type",
        description="Primary transition technique between sections",
        param_type=ParameterType.CATEGORICAL,
        options=["direct", "pivot_chord", "common_tone", "chromatic", "sequential", "elision"],
        default_value="pivot_chord",
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "pop"],
        constraint_description="Direct: no preparation; Pivot: shared chord; Common tone: held note; "
                              "Chromatic: chromatic motion; Sequential: pattern repetition; Elision: overlap"
    ))

    REGISTRY.register(ParameterDefinition(
        name="transition_smoothness",
        full_path="structure.transition.smoothness",
        description="How smooth transitions are (0.0=abrupt/dramatic, 1.0=seamless)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.7,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "pop"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="drum_fill_prob",
        full_path="structure.transition.drum_fill_prob",
        description="Probability of drum fill at section transitions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.7,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["rock", "funk", "pop", "jazz"]
    ))

    # ========================================================================
    # Structural Change Probabilities
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="key_change_prob",
        full_path="structure.transition.key_change_prob",
        description="Probability of modulation/key change at transitions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["pop", "gospel", "classical", "country"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="tempo_change_prob",
        full_path="structure.transition.tempo_change_prob",
        description="Probability of tempo change at transitions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "progressive", "jazz"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="meter_change_prob",
        full_path="structure.transition.meter_change_prob",
        description="Probability of meter/time signature change at transitions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["progressive", "classical", "jazz"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="texture_change_prob",
        full_path="structure.transition.texture_change_prob",
        description="Probability of significant texture change at transitions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.6,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["pop", "electronic", "rock"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="dynamic_change_prob",
        full_path="structure.transition.dynamic_change_prob",
        description="Probability of dynamic level change at transitions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.7,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "pop", "rock"]
    ))

    # ========================================================================
    # Turnarounds and Pickups
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="turnaround_complexity",
        full_path="structure.transition.turnaround_complexity",
        description="Harmonic complexity of turnarounds (1=simple I-V, 7=complex jazz)",
        param_type=ParameterType.CONTINUOUS,
        default_value=3.0,
        min_value=1.0,
        max_value=7.0,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "blues", "country"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="pickup_notes",
        full_path="structure.transition.pickup_notes",
        description="Use pickup/anacrusis notes before sections",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "country", "pop"]
    ))

    # ========================================================================
    # Motivic Development Techniques
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="variation_intensity",
        full_path="structure.development.variation_intensity",
        description="Overall intensity of thematic variation (0.0=literal, 1.0=free)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["classical", "jazz", "progressive"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="motivic_transformation",
        full_path="structure.development.motivic_transformation",
        description="Amount of motivic transformation across sections (0.0=none, 1.0=maximum)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "progressive"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="fragmentation_prob",
        full_path="structure.development.fragmentation_prob",
        description="Probability of breaking themes into smaller fragments",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="augmentation_prob",
        full_path="structure.development.augmentation_prob",
        description="Probability of rhythmic augmentation (lengthening note values)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="diminution_prob",
        full_path="structure.development.diminution_prob",
        description="Probability of rhythmic diminution (shortening note values)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz", "bebop"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="inversion_prob",
        full_path="structure.development.inversion_prob",
        description="Probability of melodic inversion (mirror contour)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz", "progressive"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="retrograde_prob",
        full_path="structure.development.retrograde_prob",
        description="Probability of retrograde (theme in reverse)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "avant_garde"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="sequence_prob",
        full_path="structure.development.sequence_prob",
        description="Probability of sequential repetition at different pitch levels",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "baroque"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="sequence_interval",
        full_path="structure.development.sequence_interval",
        description="Interval for sequences in semitones (2=whole step, 5=fourth, etc.)",
        param_type=ParameterType.CATEGORICAL,
        options=[1, 2, 3, 4, 5, 7],
        default_value=2,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz"],
        depends_on=["structure.development.sequence_prob"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="climax_placement",
        full_path="structure.development.climax_placement",
        description="Where to place structural climax (0.0=beginning, 1.0=end)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.75,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["classical", "pop", "rock", "gospel"],
        constraint_description="Golden ratio (~0.618) often used in classical music, "
                              "pop typically at 0.75-0.85"
    ))


def register_repetition_memory_parameters():
    """
    Register 20 Repetition & Memory parameters

    These parameters control thematic unity, recall, and repetition strategies:
    - Exact vs varied repetition
    - Motif and theme recall
    - Ostinato and pedal points
    - Phrase structure and grouping
    - Thematic consistency
    """

    # ========================================================================
    # Repetition Strategies
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="exact_repeat_prob",
        full_path="structure.repetition.exact_repeat_prob",
        description="Probability of exact repetition of sections",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["blues", "rock", "electronic"],
        mutually_exclusive_with=["structure.repetition.varied_repeat_prob"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="varied_repeat_prob",
        full_path="structure.repetition.varied_repeat_prob",
        description="Probability of varied repetition (with embellishments/changes)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "classical", "gospel"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="variation_amount",
        full_path="structure.repetition.variation_amount",
        description="How much variation on repeated sections (0.0=minimal, 1.0=extensive)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "classical"],
        depends_on=["structure.repetition.varied_repeat_prob"]
    ))

    # ========================================================================
    # Thematic Recall and Memory
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="motif_recall_prob",
        full_path="structure.repetition.motif_recall_prob",
        description="Probability of recalling earlier motifs in later sections",
        param_type=ParameterType.PROBABILITY,
        default_value=0.7,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "gospel"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="theme_return_prob",
        full_path="structure.repetition.theme_return_prob",
        description="Probability of returning to opening theme (recapitulation)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.9,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["classical", "jazz", "pop"]
    ))

    # ========================================================================
    # Ostinato and Pedal Points
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="ostinato_usage",
        full_path="structure.repetition.ostinato_usage",
        description="Amount of ostinato (repeated pattern) usage (0.0=none, 1.0=extensive)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["electronic", "funk", "minimalist", "baroque"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="pedal_point_duration",
        full_path="structure.repetition.pedal_point_duration",
        description="Duration of pedal points (sustained bass notes) in bars",
        param_type=ParameterType.CATEGORICAL,
        options=[2, 4, 8, 16, 32],
        default_value=8,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "baroque", "electronic"]
    ))

    # ========================================================================
    # Riff and Hook Construction
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="riff_based",
        full_path="structure.repetition.riff_based",
        description="Construct piece around repeated riffs (blues/rock style)",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["blues", "rock", "funk", "electronic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="hook_emphasis",
        full_path="structure.repetition.hook_emphasis",
        description="Emphasis on catchy hooks/memorable phrases (0.0=none, 1.0=maximum)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.7,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["pop", "rock", "gospel", "country"]
    ))

    # ========================================================================
    # Call and Response
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="call_response",
        full_path="structure.repetition.call_response",
        description="Use call-and-response phrase structure",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "blues", "gospel", "funk", "african"]
    ))

    # ========================================================================
    # Phrase Structure
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="phrase_length",
        full_path="structure.repetition.phrase_length",
        description="Standard phrase length in bars",
        param_type=ParameterType.CATEGORICAL,
        options=[2, 4, 8, 16],
        default_value=4,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["classical", "jazz", "pop", "rock"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="phrase_grouping",
        full_path="structure.repetition.phrase_grouping",
        description="How phrases are grouped and related",
        param_type=ParameterType.CATEGORICAL,
        options=["regular", "irregular", "overlapping"],
        default_value="regular",
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "progressive"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="antecedent_consequent",
        full_path="structure.repetition.antecedent_consequent",
        description="Use antecedent-consequent phrase pairs (question-answer)",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "pop"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="parallel_period",
        full_path="structure.repetition.parallel_period",
        description="Use parallel period construction (similar phrase beginnings)",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "pop"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="contrast_middle",
        full_path="structure.repetition.contrast_middle",
        description="Create contrast in middle sections (B section different from A)",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "pop", "classical"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="return_variation",
        full_path="structure.repetition.return_variation",
        description="Amount of variation when themes return (0.0=exact, 1.0=extensively varied)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz"]
    ))

    # ========================================================================
    # Thematic Unity and Consistency
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="thematic_unity",
        full_path="structure.repetition.thematic_unity",
        description="Overall thematic unity/coherence (0.0=episodic, 1.0=highly unified)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.7,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["classical", "jazz", "progressive"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="motivic_consistency",
        full_path="structure.repetition.motivic_consistency",
        description="Consistency of motivic material across sections (0.0=varied, 1.0=consistent)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.7,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "bebop"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythmic_motif_recurrence",
        full_path="structure.repetition.rhythmic_motif_recurrence",
        description="How often rhythmic motifs recur (0.0=rarely, 1.0=constantly)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.6,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["funk", "electronic", "minimalist"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melodic_cell_recurrence",
        full_path="structure.repetition.melodic_cell_recurrence",
        description="How often melodic cells/fragments recur (0.0=rarely, 1.0=constantly)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.6,
        category=ParameterCategory.STRUCTURE,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "pop"]
    ))


# ============================================================================
# MAIN REGISTRATION FUNCTION
# ============================================================================

def register_all_structure_parameters():
    """
    Register all 60 structure expansion parameters

    This function adds:
    - 20 Form & Architecture parameters
    - 20 Transitions & Development parameters
    - 20 Repetition & Memory parameters

    Total: 60 new structure parameters
    """
    print("🏗️  Agent 2: Structure & Form Expansion")
    print("=" * 80)

    print("\n📐 Registering Form & Architecture parameters (20)...")
    register_form_architecture_parameters()

    print("🔄 Registering Transitions & Development parameters (20)...")
    register_transition_development_parameters()

    print("🔁 Registering Repetition & Memory parameters (20)...")
    register_repetition_memory_parameters()

    stats = REGISTRY.get_statistics()
    structure_params = REGISTRY.get_by_category(ParameterCategory.STRUCTURE)

    print("\n" + "=" * 80)
    print(f"✅ Structure expansion complete!")
    print(f"   Structure parameters: {len(structure_params)}")
    print(f"   Total registry parameters: {stats['total_parameters']}")
    print("=" * 80)


# ============================================================================
# GENRE PRESET RETRIEVAL
# ============================================================================

GENRE_PRESETS = {
    "jazz": get_jazz_structure_defaults,
    "blues": get_blues_structure_defaults,
    "rock": get_rock_structure_defaults,
    "classic_rock": get_rock_structure_defaults,
    "pop": get_pop_structure_defaults,
    "country": get_country_structure_defaults,
    "gospel": get_gospel_structure_defaults,
    "funk": get_funk_soul_structure_defaults,
    "soul": get_funk_soul_structure_defaults,
    "funk_soul": get_funk_soul_structure_defaults,
    "electronic": get_electronic_structure_defaults,
}


def get_structure_defaults(genre: str) -> Dict[str, Any]:
    """
    Get structure parameter defaults for a specific genre

    Args:
        genre: Genre name (jazz, blues, rock, pop, country, gospel, funk, soul, electronic)

    Returns:
        Dictionary of parameter paths to default values

    Example:
        >>> defaults = get_structure_defaults("jazz")
        >>> print(defaults["structure.form.type"])
        'AABA'
    """
    genre_lower = genre.lower().replace(" ", "_")

    if genre_lower in GENRE_PRESETS:
        return GENRE_PRESETS[genre_lower]()
    else:
        # Return jazz defaults as fallback
        print(f"⚠️  Warning: Genre '{genre}' not found, using jazz defaults")
        return get_jazz_structure_defaults()


# ============================================================================
# MODULE TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STRUCTURE & FORM EXPANSION - AGENT 2")
    print("=" * 80)

    # Register all parameters
    register_all_structure_parameters()

    # Show statistics
    print("\n📊 Parameter Statistics:")
    stats = REGISTRY.get_statistics()
    print(f"   Total parameters in registry: {stats['total_parameters']}")

    structure_params = REGISTRY.get_by_category(ParameterCategory.STRUCTURE)
    print(f"   Structure parameters: {len(structure_params)}")

    # Show genre defaults
    print("\n🎵 Genre Presets Available:")
    for genre_name in sorted(GENRE_PRESETS.keys()):
        defaults = GENRE_PRESETS[genre_name]()
        print(f"   - {genre_name.title()}: {len(defaults)} parameters")

    # Test validation
    print("\n✅ Validation Tests:")
    test_cases = [
        ("structure.form.type", "AABA", True),
        ("structure.form.type", "invalid_form", False),
        ("structure.form.length_bars", 32, True),
        ("structure.form.length_bars", 17, False),
        ("structure.transition.smoothness", 0.5, True),
        ("structure.transition.smoothness", 1.5, False),
    ]

    for path, value, expected_valid in test_cases:
        valid, msg = REGISTRY.validate_parameter(path, value)
        status = "✅" if valid == expected_valid else "❌"
        result = "VALID" if valid else f"INVALID: {msg}"
        print(f"   {status} {path} = {value}: {result}")

    # Export
    print("\n💾 Exporting registry...")
    REGISTRY.export_to_json("/home/user/Do/midi_generator/parameters/registry_with_structure.json")
    REGISTRY.generate_documentation("/home/user/Do/midi_generator/parameters/PARAMETERS_WITH_STRUCTURE.md")

    print("\n" + "=" * 80)
    print("✅ Agent 2 Structure Expansion Complete!")
    print("   60 new structure parameters added to registry")
    print("   8 genre-specific preset collections created")
    print("=" * 80)
