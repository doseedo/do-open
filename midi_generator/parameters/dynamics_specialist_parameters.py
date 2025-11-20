"""
Dynamics Specialist Parameters - Agent 22
==========================================

Advanced dynamics parameters for Agent 22 (Dynamics Specialist).
Adds 40+ specialized parameters for:
- ADSR envelopes (10 params)
- Dynamic curves (15 params)
- Humanization (10 params)
- Voice balancing (5 params)

These parameters extend the basic dynamics system with expressive control.

Author: Agent 22 - Dynamics Specialist
License: MIT
Date: 2025-11-20
"""

from typing import List, Dict, Any
from .universal_registry import (
    ParameterDefinition,
    ParameterType,
    ParameterCategory,
    MusicalImpact,
    REGISTRY
)


# ============================================================================
# ADSR ENVELOPE PARAMETERS (10 parameters)
# ============================================================================

def register_adsr_parameters():
    """
    Register ADSR envelope parameters (10 parameters)

    Controls attack, decay, sustain, and release characteristics for
    expressive note shaping.
    """

    REGISTRY.register(ParameterDefinition(
        name="attack_time",
        full_path="dynamics.adsr.attack_time",
        description="ADSR attack time in seconds (0=instant, higher=slower attack)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.05,
        min_value=0.0,
        max_value=2.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz", "ambient", "electronic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="decay_time",
        full_path="dynamics.adsr.decay_time",
        description="ADSR decay time in seconds (transition from attack to sustain)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.1,
        min_value=0.0,
        max_value=2.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz", "electronic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="sustain_level",
        full_path="dynamics.adsr.sustain_level",
        description="ADSR sustain level (0.0-1.0, relative to peak)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="release_time",
        full_path="dynamics.adsr.release_time",
        description="ADSR release time in seconds (note-off to silence)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.2,
        min_value=0.0,
        max_value=5.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz", "ambient"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="envelope_enabled",
        full_path="dynamics.adsr.envelope_enabled",
        description="Enable ADSR envelope shaping",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="attack_curve",
        full_path="dynamics.adsr.attack_curve",
        description="Attack curve shape (linear, exponential, logarithmic)",
        param_type=ParameterType.CATEGORICAL,
        default_value="linear",
        options=["linear", "exponential", "logarithmic"],
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["electronic", "ambient"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="release_curve",
        full_path="dynamics.adsr.release_curve",
        description="Release curve shape (linear, exponential, logarithmic)",
        param_type=ParameterType.CATEGORICAL,
        default_value="exponential",
        options=["linear", "exponential", "logarithmic"],
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["electronic", "ambient"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="envelope_variation",
        full_path="dynamics.adsr.envelope_variation",
        description="Random variation in envelope parameters (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["jazz", "classical"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="velocity_envelope_coupling",
        full_path="dynamics.adsr.velocity_envelope_coupling",
        description="How much note velocity affects envelope (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="duration_envelope_scaling",
        full_path="dynamics.adsr.duration_envelope_scaling",
        description="Scale envelope times with note duration (True/False)",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["all"],
        learnable=True
    ))


# ============================================================================
# DYNAMIC CURVE PARAMETERS (15 parameters)
# ============================================================================

def register_dynamic_curve_parameters():
    """
    Register dynamic curve parameters (15 parameters)

    Controls crescendo, diminuendo, and custom dynamic curves for
    phrase-level expression.
    """

    REGISTRY.register(ParameterDefinition(
        name="crescendo_enabled",
        full_path="dynamics.curves.crescendo_enabled",
        description="Enable automatic crescendo detection/application",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "orchestral", "dramatic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="crescendo_intensity",
        full_path="dynamics.curves.crescendo_intensity",
        description="Intensity of crescendo curves (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "orchestral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="crescendo_curve_type",
        full_path="dynamics.curves.crescendo_curve_type",
        description="Shape of crescendo curve",
        param_type=ParameterType.CATEGORICAL,
        default_value="exponential",
        options=["linear", "exponential", "logarithmic", "sigmoid", "parabolic"],
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["classical", "orchestral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="diminuendo_enabled",
        full_path="dynamics.curves.diminuendo_enabled",
        description="Enable automatic diminuendo detection/application",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "orchestral", "ambient"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="diminuendo_intensity",
        full_path="dynamics.curves.diminuendo_intensity",
        description="Intensity of diminuendo curves (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "orchestral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="diminuendo_curve_type",
        full_path="dynamics.curves.diminuendo_curve_type",
        description="Shape of diminuendo curve",
        param_type=ParameterType.CATEGORICAL,
        default_value="logarithmic",
        options=["linear", "exponential", "logarithmic", "sigmoid", "parabolic"],
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["classical", "orchestral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="phrase_arch_shaping",
        full_path="dynamics.curves.phrase_arch_shaping",
        description="Apply arch-shaped dynamics to phrases (crescendo→diminuendo)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "vocal"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="curve_smoothness",
        full_path="dynamics.curves.curve_smoothness",
        description="Smoothness of dynamic curves (0.0=stepped, 1.0=smooth)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.8,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="curve_shape_factor",
        full_path="dynamics.curves.curve_shape_factor",
        description="Exponential/logarithmic curve shape factor (1.0-5.0)",
        param_type=ParameterType.CONTINUOUS,
        default_value=2.0,
        min_value=0.5,
        max_value=5.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="dynamic_peaks_per_phrase",
        full_path="dynamics.curves.dynamic_peaks_per_phrase",
        description="Number of dynamic peaks within a phrase",
        param_type=ParameterType.INTEGER,
        default_value=1,
        min_value=0,
        max_value=5,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "dramatic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="terraced_dynamics",
        full_path="dynamics.curves.terraced_dynamics",
        description="Use terraced dynamics (baroque style, sudden changes)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["baroque", "classical"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="dynamic_plateau_duration",
        full_path="dynamics.curves.dynamic_plateau_duration",
        description="Duration of stable dynamics between curves (beats)",
        param_type=ParameterType.CONTINUOUS,
        default_value=4.0,
        min_value=0.0,
        max_value=16.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="subito_change_probability",
        full_path="dynamics.curves.subito_change_probability",
        description="Probability of sudden dynamic changes (subito forte/piano)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "dramatic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="climax_intensity",
        full_path="dynamics.curves.climax_intensity",
        description="Intensity of structural climax points (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.9,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "dramatic", "film_score"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="echo_diminuendo",
        full_path="dynamics.curves.echo_diminuendo",
        description="Apply echo-like diminuendo to repeated phrases",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz"],
        learnable=True
    ))


# ============================================================================
# HUMANIZATION PARAMETERS (10 parameters)
# ============================================================================

def register_humanization_parameters():
    """
    Register humanization parameters (10 parameters)

    Controls natural velocity and timing variations to avoid
    mechanical/robotic performance.
    """

    REGISTRY.register(ParameterDefinition(
        name="velocity_humanization",
        full_path="dynamics.humanization.velocity_humanization",
        description="Amount of velocity randomization (0.0=mechanical, 1.0=very human)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="preserve_accents",
        full_path="dynamics.humanization.preserve_accents",
        description="Keep loud notes loud during humanization",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="micro_dynamics_variance",
        full_path="dynamics.humanization.micro_dynamics_variance",
        description="Subtle phrase-level dynamics variation (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz", "expressive"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="phrase_dynamics_length",
        full_path="dynamics.humanization.phrase_dynamics_length",
        description="Notes per phrase for micro-dynamics shaping",
        param_type=ParameterType.INTEGER,
        default_value=4,
        min_value=2,
        max_value=16,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="timing_humanization",
        full_path="dynamics.humanization.timing_humanization",
        description="Micro-timing variance in seconds (0.0=perfect, higher=human)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.02,
        min_value=0.0,
        max_value=0.1,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "classical", "acoustic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="groove_consistency",
        full_path="dynamics.humanization.groove_consistency",
        description="Consistency of timing (0.0=loose, 1.0=tight)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["funk", "jazz", "groove"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="natural_variation_seed",
        full_path="dynamics.humanization.natural_variation_seed",
        description="Random seed for reproducible humanization (-1=random)",
        param_type=ParameterType.INTEGER,
        default_value=-1,
        min_value=-1,
        max_value=10000,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["all"],
        learnable=False
    ))

    REGISTRY.register(ParameterDefinition(
        name="accent_randomization",
        full_path="dynamics.humanization.accent_randomization",
        description="Randomize accent placement and intensity (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "classical"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="velocity_drift",
        full_path="dynamics.humanization.velocity_drift",
        description="Gradual velocity drift over time (simulates fatigue/excitement)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        min_value=0.0,
        max_value=0.5,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["live_performance", "jazz"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="mechanical_consistency_score",
        full_path="dynamics.humanization.mechanical_consistency_score",
        description="Target consistency (0.0=human, 1.0=mechanical)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["electronic", "minimal", "mechanical"],
        learnable=True
    ))


# ============================================================================
# VOICE BALANCING PARAMETERS (5 parameters)
# ============================================================================

def register_voice_balancing_parameters():
    """
    Register voice balancing parameters (5 parameters)

    Controls relative dynamics across multiple voices/layers for
    proper musical balance.
    """

    REGISTRY.register(ParameterDefinition(
        name="melody_emphasis_amount",
        full_path="dynamics.balance.melody_emphasis_amount",
        description="Amount to emphasize melody voice (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["pop", "vocal", "melodic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="bass_boost",
        full_path="dynamics.balance.bass_boost",
        description="Bass frequency emphasis (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["funk", "R&B", "dance", "hip_hop"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="inner_voice_reduction",
        full_path="dynamics.balance.inner_voice_reduction",
        description="Reduce inner harmony voices (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "orchestral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="voice_balance_ratios",
        full_path="dynamics.balance.voice_balance_ratios",
        description="Custom balance ratios for voices [melody, harmony, bass, drums]",
        param_type=ParameterType.ARRAY_FLOAT,
        default_value=[1.0, 0.7, 0.8, 0.9],
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"],
        learnable=True,
        constraint_description="Array of 4 floats in range [0.0, 1.5]"
    ))

    REGISTRY.register(ParameterDefinition(
        name="adaptive_balance",
        full_path="dynamics.balance.adaptive_balance",
        description="Automatically adjust balance based on texture density",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["orchestral", "complex_arrangements"],
        learnable=True
    ))


# ============================================================================
# REGISTRATION FUNCTION
# ============================================================================

def register_all_dynamics_specialist_parameters():
    """Register all 40 dynamics specialist parameters"""
    register_adsr_parameters()           # 10 parameters
    register_dynamic_curve_parameters()  # 15 parameters
    register_humanization_parameters()   # 10 parameters
    register_voice_balancing_parameters()# 5 parameters
    print("✅ Registered 40 Dynamics Specialist parameters")


# Auto-register when module is imported
if __name__ != "__main__":
    try:
        register_all_dynamics_specialist_parameters()
    except Exception as e:
        print(f"⚠️ Could not auto-register dynamics specialist parameters: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DYNAMICS SPECIALIST PARAMETERS - AGENT 22")
    print("=" * 80)
    print()
    print("Registering 40+ advanced dynamics parameters...")
    print()

    register_all_dynamics_specialist_parameters()

    print()
    print("Parameter Categories:")
    print("  • ADSR Envelopes:    10 parameters")
    print("  • Dynamic Curves:    15 parameters")
    print("  • Humanization:      10 parameters")
    print("  • Voice Balancing:    5 parameters")
    print("  ─" * 40)
    print("  TOTAL:               40 parameters")
    print()
    print("=" * 80)
    print("✅ REGISTRATION COMPLETE")
    print("=" * 80)
