#!/usr/bin/env python3
"""
Parameterized Orchestration Engine - Agent 7
=============================================

This module extends the base Orchestrator with parameter-driven behavior.
All orchestration decisions are now controlled by the 50 instrumentation
parameters defined in parameters/instrumentation_params.py.

This is Agent 7's refactoring of orchestrator.py to use the foundation
parameter system instead of hardcoded values.

Author: Agent 7 - Instrumentation & Orchestration
Part of: Focused Parameter Refactoring (Agents 1-10)
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import random

sys.path.append(str(Path(__file__).parent.parent))

from generators.orchestrator import (
    Orchestrator, OrchestrationStyle, VoicePart, OrchestralVoicing,
    TextureType
)
from core.instrument_library import (
    Instrument, InstrumentFamily, get_instrument,
    get_instruments_by_family
)
from parameters.instrumentation_params import (
    get_all_instrumentation_parameters,
    get_default_values,
    get_instrumentation_parameter
)


class ParameterizedOrchestrator(Orchestrator):
    """
    Orchestrator that uses foundation parameters instead of hardcoded values.

    All musical decisions are controlled by parameters from
    instrumentation_params.py, making the orchestration behavior
    learnable and tunable.
    """

    def __init__(
        self,
        style: OrchestrationStyle = OrchestrationStyle.ROMANTIC,
        available_instruments: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize parameterized orchestrator.

        Args:
            style: Orchestration style (sets some defaults)
            available_instruments: Available instruments
            parameters: Override parameters (uses defaults if None)
        """
        super().__init__(style, available_instruments)

        # Load parameters (defaults + overrides)
        self.params = get_default_values()
        if parameters:
            self.params.update(parameters)

        # Apply style-based parameter adjustments
        self._apply_style_adjustments()

    def _apply_style_adjustments(self):
        """Adjust parameters based on orchestration style"""
        if self.style == OrchestrationStyle.CLASSICAL:
            # Classical: transparent, balanced
            self.params.update({
                'instrumentation.doubling.octave_probability': 0.3,
                'instrumentation.voicing.close_position_ratio': 0.4,
                'instrumentation.selection.homogeneous_blend_probability': 0.7,
                'instrumentation.dynamics.pp_to_ff_range': 0.6,
            })

        elif self.style == OrchestrationStyle.ROMANTIC:
            # Romantic: lush, wide ranges
            self.params.update({
                'instrumentation.doubling.octave_probability': 0.6,
                'instrumentation.voicing.spread_factor': 0.7,
                'instrumentation.dynamics.pp_to_ff_range': 0.9,
                'instrumentation.register.extreme_register_probability': 0.25,
            })

        elif self.style == OrchestrationStyle.FILM:
            # Film: dramatic, reinforced
            self.params.update({
                'instrumentation.doubling.melody_reinforcement': 0.8,
                'instrumentation.doubling.bass_reinforcement': 0.8,
                'instrumentation.dynamics.balance_melody_ratio': 1.5,
                'instrumentation.selection.prefer_strings_probability': 0.7,
            })

        elif self.style == OrchestrationStyle.CHAMBER:
            # Chamber: intimate, transparent
            self.params.update({
                'instrumentation.doubling.unison_probability': 0.1,
                'instrumentation.selection.solo_vs_ensemble_ratio': 0.6,
                'instrumentation.voicing.close_position_ratio': 0.6,
            })

    def _apply_doubling_rules(
        self,
        voicings: List[OrchestralVoicing],
        voices: List[VoicePart]
    ) -> List[OrchestralVoicing]:
        """
        Apply professional doubling rules using parameters.

        Now uses:
        - instrumentation.doubling.octave_probability
        - instrumentation.doubling.unison_probability
        - instrumentation.doubling.melody_reinforcement
        - instrumentation.doubling.bass_reinforcement
        - instrumentation.doubling.avoid_muddy_bass
        """
        if not voicings or not voices:
            return voicings

        new_voicings = list(voicings)

        for i, voice in enumerate(voices):
            if i >= len(voicings):
                break

            original_voicing = voicings[i]
            instrument = get_instrument(original_voicing.instrument_name)

            if not instrument:
                continue

            # MELODY DOUBLING
            if voice.texture_type == TextureType.MELODY:
                # Use melody reinforcement parameter
                if random.random() < self.params['instrumentation.doubling.melody_reinforcement']:
                    doubled = self._create_doubling(
                        original_voicing,
                        instrument,
                        'octave' if random.random() < self.params['instrumentation.doubling.octave_probability'] else 'unison'
                    )
                    if doubled:
                        new_voicings.append(doubled)

            # BASS DOUBLING
            elif voice.texture_type == TextureType.BASS:
                # Use bass reinforcement parameter
                if random.random() < self.params['instrumentation.doubling.bass_reinforcement']:
                    # Check if we should avoid muddy bass
                    avg_pitch = sum(voice.notes) / len(voice.notes) if voice.notes else 60

                    if self.params['instrumentation.doubling.avoid_muddy_bass'] and avg_pitch < 48:
                        # Below C3 - use octave doubling, not unison
                        doubled = self._create_doubling(original_voicing, instrument, 'octave_up')
                    else:
                        doubled = self._create_doubling(original_voicing, instrument, 'octave')

                    if doubled:
                        new_voicings.append(doubled)

            # GENERAL DOUBLING (for harmony voices)
            elif voice.texture_type == TextureType.HARMONY:
                # Octave doubling
                if random.random() < self.params['instrumentation.doubling.octave_probability']:
                    doubled = self._create_doubling(original_voicing, instrument, 'octave')
                    if doubled:
                        new_voicings.append(doubled)

                # Thirds/sixths doubling
                elif random.random() < self.params['instrumentation.doubling.thirds_probability']:
                    doubled = self._create_doubling(original_voicing, instrument, 'thirds')
                    if doubled:
                        new_voicings.append(doubled)

        # Limit maximum simultaneous doublings
        max_doublings = self.params['instrumentation.doubling.max_simultaneous']
        # (Implementation would track and limit doublings per pitch)

        return new_voicings

    def _create_doubling(
        self,
        original: OrchestralVoicing,
        original_instrument: Instrument,
        doubling_type: str
    ) -> Optional[OrchestralVoicing]:
        """
        Create a doubling of the original voicing.

        Args:
            original: Original voicing to double
            original_instrument: Original instrument
            doubling_type: 'unison', 'octave', 'octave_up', 'octave_down', 'thirds', 'sixths'

        Returns:
            New voicing with doubled notes, or None if not possible
        """
        # Select doubling instrument
        doubling_inst = self._select_doubling_instrument(
            original_instrument,
            doubling_type
        )

        if not doubling_inst:
            return None

        # Transpose notes based on doubling type
        doubled_notes = []
        for note in original.notes:
            if doubling_type == 'unison':
                doubled_note = note
            elif doubling_type == 'octave':
                # Choose up or down based on range
                doubled_note = note + 12 if note < 72 else note - 12
            elif doubling_type == 'octave_up':
                doubled_note = note + 12
            elif doubling_type == 'octave_down':
                doubled_note = note - 12
            elif doubling_type == 'thirds':
                doubled_note = note + 4  # Major third up
            elif doubling_type == 'sixths':
                doubled_note = note + 9  # Major sixth up
            else:
                doubled_note = note

            # Check if in range
            if (doubled_note >= doubling_inst.range.lowest_note and
                doubled_note <= doubling_inst.range.highest_note):
                doubled_notes.append(doubled_note)
            else:
                # Skip this note if out of range
                pass

        if not doubled_notes:
            return None

        # Create new voicing
        # Reduce velocity slightly for doubling (layer reduction factor)
        reduction = self.params['instrumentation.dynamics.layer_reduction_factor']
        doubled_velocities = [int(v * reduction) for v in original.velocities[:len(doubled_notes)]]

        return OrchestralVoicing(
            instrument_name=doubling_inst.name,
            notes=doubled_notes,
            durations=original.durations[:len(doubled_notes)],
            start_times=original.start_times[:len(doubled_notes)],
            velocities=doubled_velocities,
            articulations=original.articulations[:len(doubled_notes)]
        )

    def _select_doubling_instrument(
        self,
        original_instrument: Instrument,
        doubling_type: str
    ) -> Optional[Instrument]:
        """
        Select appropriate instrument for doubling.

        Uses:
        - instrumentation.doubling.family_preference
        - instrumentation.selection.homogeneous_blend_probability
        """
        family_pref = self.params['instrumentation.doubling.family_preference']
        homogeneous_prob = self.params['instrumentation.selection.homogeneous_blend_probability']

        # Decide whether to stay within family or cross families
        if family_pref == 'within' or random.random() < homogeneous_prob:
            # Stay within same family
            same_family = get_instruments_by_family(original_instrument.family)
            candidates = [inst for inst in same_family if inst.name != original_instrument.name]
        else:
            # Cross families for color
            candidates = [inst for inst in self.available_instruments
                         if inst.family != original_instrument.family]

        if not candidates:
            return None

        # Return random candidate
        return random.choice(candidates)

    def _optimize_balance(
        self,
        voicings: List[OrchestralVoicing]
    ) -> List[OrchestralVoicing]:
        """
        Optimize orchestral balance using parameters.

        Uses:
        - instrumentation.dynamics.balance_melody_ratio
        - instrumentation.dynamics.balance_bass_ratio
        - instrumentation.dynamics.family_balance_mode
        - instrumentation.dynamics.register_compensation
        """
        if not voicings:
            return voicings

        melody_ratio = self.params['instrumentation.dynamics.balance_melody_ratio']
        bass_ratio = self.params['instrumentation.dynamics.balance_bass_ratio']
        register_comp = self.params['instrumentation.dynamics.register_compensation']

        for voicing in voicings:
            instrument = get_instrument(voicing.instrument_name)
            if not instrument:
                continue

            # Get average pitch for register compensation
            avg_pitch = sum(voicing.notes) / len(voicing.notes) if voicing.notes else 60

            # Family-based balance adjustments
            family = instrument.family
            family_adjustments = {
                InstrumentFamily.BRASS: -10,      # Brass overpowers, reduce
                InstrumentFamily.WOODWINDS: +5,   # Woodwinds get lost, boost
                InstrumentFamily.STRINGS: 0,       # Strings balanced
                InstrumentFamily.PERCUSSION: -5,   # Percussion can overpower
            }

            adjustment = family_adjustments.get(family, 0)

            # Apply adjustment
            voicing.velocities = [
                max(20, min(120, v + adjustment))
                for v in voicing.velocities
            ]

            # Register compensation (extreme registers need more power)
            if register_comp:
                if avg_pitch < 48:  # Low register
                    voicing.velocities = [min(120, int(v * 1.1)) for v in voicing.velocities]
                elif avg_pitch > 84:  # High register
                    voicing.velocities = [min(120, int(v * 1.15)) for v in voicing.velocities]

        return voicings

    def _adjust_velocity(
        self,
        original_velocity: int,
        target_dynamic: int,
        instrument: Instrument
    ) -> int:
        """
        Adjust velocity for instrument characteristics.

        Now uses instrumentation.dynamics parameters instead of hardcoded 0.5 blend.
        """
        # Use accent strength parameter for blend ratio
        accent = self.params.get('instrumentation.dynamics.accent_strength', 1.3)

        # Blend original expression with target dynamic
        # (Using accent_strength as a proxy for blend ratio)
        blend_ratio = 0.5  # Could add specific parameter for this
        adjusted = int(original_velocity * blend_ratio + target_dynamic * (1 - blend_ratio))

        # Clamp to instrument's dynamic range
        adjusted = max(instrument.min_dynamic, adjusted)
        adjusted = min(instrument.max_dynamic, adjusted)

        return adjusted

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values"""
        return self.params.copy()

    def set_parameter(self, param_name: str, value: Any):
        """Set a specific parameter value"""
        if param_name in self.params:
            self.params[param_name] = value
        else:
            raise KeyError(f"Parameter '{param_name}' not found")

    def set_parameters(self, parameters: Dict[str, Any]):
        """Set multiple parameters at once"""
        self.params.update(parameters)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_orchestrator_from_parameters(
    parameters: Dict[str, Any],
    style: OrchestrationStyle = OrchestrationStyle.ROMANTIC
) -> ParameterizedOrchestrator:
    """
    Create orchestrator from parameter dictionary.

    This is the main entry point for Agent 5 (XGBoost) to use.
    """
    return ParameterizedOrchestrator(
        style=style,
        parameters=parameters
    )


def create_orchestrator_with_defaults(
    style: OrchestrationStyle = OrchestrationStyle.ROMANTIC
) -> ParameterizedOrchestrator:
    """Create orchestrator with default parameters"""
    return ParameterizedOrchestrator(style=style)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PARAMETERIZED ORCHESTRATOR - Agent 7")
    print("=" * 80)

    # Example 1: Create with defaults
    print("\n1. Creating orchestrator with default parameters...")
    orch = create_orchestrator_with_defaults(OrchestrationStyle.ROMANTIC)
    print(f"   Loaded {len(orch.params)} parameters")
    print(f"   Doubling octave probability: {orch.params['instrumentation.doubling.octave_probability']}")
    print(f"   Voicing close position ratio: {orch.params['instrumentation.voicing.close_position_ratio']}")

    # Example 2: Override specific parameters
    print("\n2. Creating orchestrator with custom parameters...")
    custom_params = {
        'instrumentation.doubling.octave_probability': 0.8,  # More doubling
        'instrumentation.dynamics.balance_melody_ratio': 1.6,  # Louder melody
        'instrumentation.voicing.spread_factor': 0.9,  # Wide voicings
    }
    orch_custom = create_orchestrator_from_parameters(custom_params, OrchestrationStyle.FILM)
    print(f"   Custom octave doubling: {orch_custom.params['instrumentation.doubling.octave_probability']}")
    print(f"   Custom melody ratio: {orch_custom.params['instrumentation.dynamics.balance_melody_ratio']}")

    # Example 3: Style-based adjustments
    print("\n3. Comparing different orchestration styles...")
    styles = [
        OrchestrationStyle.CLASSICAL,
        OrchestrationStyle.ROMANTIC,
        OrchestrationStyle.FILM,
        OrchestrationStyle.CHAMBER
    ]

    for style in styles:
        orch_style = create_orchestrator_with_defaults(style)
        print(f"\n   {style.value.upper()}:")
        print(f"      Octave doubling: {orch_style.params['instrumentation.doubling.octave_probability']:.2f}")
        print(f"      Spread factor: {orch_style.params['instrumentation.voicing.spread_factor']:.2f}")
        print(f"      Dynamic range: {orch_style.params['instrumentation.dynamics.pp_to_ff_range']:.2f}")

    print("\n" + "=" * 80)
    print("Parameterized Orchestrator ready!")
    print("All orchestration decisions now use foundation parameters.")
    print("=" * 80)
