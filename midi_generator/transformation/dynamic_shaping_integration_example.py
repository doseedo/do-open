#!/usr/bin/env python3
"""
Dynamic Shaping Integration Example
====================================

This example demonstrates how to integrate the DynamicShaping module
with the existing BigBandArranger to create musically expressive
arrangements with proper dynamics and phrasing.

Shows:
------
1. How to apply dynamics to a complete big band arrangement
2. How to use form-based dynamic mapping
3. How to apply section-specific dynamics
4. How to balance sections (lead, brass, saxes, rhythm)
5. How to create shout chorus with climactic dynamics

Author: Agent 9 - Dynamic Shaping & Phrasing Master
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock imports for demonstration (replace with actual when dependencies available)
from typing import List, Dict, Union


# ============================================================================
# INTEGRATION PATTERN 1: Apply Dynamics to Existing Arrangement
# ============================================================================

def example_1_basic_integration():
    """
    Example 1: Apply dynamics to an existing static-velocity arrangement
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Dynamic Shaping Integration")
    print("=" * 80)

    print("""
    # Pseudo-code showing integration pattern:

    from transformation.arrangement_engine import BigBandArranger
    from transformation.dynamic_shaping import DynamicShaping, PhraseContour
    from generators.form_generator import AABAGenerator

    # Step 1: Create arrangement (currently has static velocities)
    melody = [...]  # Your melody notes
    chords = [...]  # Your chord progression
    arrangement = BigBandArranger.arrange(melody, chords)

    # Step 2: Apply dynamics to each section
    for section_name, notes in arrangement.items():
        if 'lead' in section_name:
            # Lead melody: arch contour, louder
            arrangement[section_name] = DynamicShaping.apply_phrase_contour(
                notes,
                contour=PhraseContour.ARCH,
                base_velocity=85,
                variation_range=20
            )

        elif 'sax' in section_name:
            # Saxes: blend, medium dynamics
            arrangement[section_name] = DynamicShaping.apply_phrase_contour(
                notes,
                contour=PhraseContour.ARCH,
                base_velocity=75,
                variation_range=15
            )

        elif 'brass' in section_name:
            # Brass: powerful, accented
            shaped = DynamicShaping.apply_phrase_contour(
                notes,
                contour=PhraseContour.ARCH,
                base_velocity=85,
                variation_range=18
            )
            arrangement[section_name] = DynamicShaping.apply_accent_pattern(
                shaped,
                pattern=AccentPattern.STRONG_WEAK,
                accent_amount=15
            )

    # Step 3: Export to MIDI
    # (arrangement now has expressive dynamics!)
    """)

    print("✅ Basic integration: Apply dynamics after arrangement creation")


# ============================================================================
# INTEGRATION PATTERN 2: Form-Based Dynamic Mapping
# ============================================================================

def example_2_form_based_dynamics():
    """
    Example 2: Use form structure to apply appropriate dynamics to each section
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Form-Based Dynamic Mapping")
    print("=" * 80)

    print("""
    # Generate AABA form with proper dynamics for each section:

    from generators.form_generator import FormGenerator, FormType
    from transformation.dynamic_shaping import (
        generate_dynamic_map_for_form,
        apply_dynamics_to_section
    )

    # Step 1: Generate form structure
    form = FormGenerator.generate_form(
        FormType.AABA,
        tonic_key=60,
        tempo=140
    )

    # Step 2: Get dynamic map for entire form
    dynamic_map = generate_dynamic_map_for_form(form)
    # Returns: {"A1": 0.65, "A2": 0.70, "B": 0.60, "A3": 0.85}

    # Step 3: Generate melody and arrangement
    melody = generate_melody_for_form(form)
    chords = generate_chords_for_form(form)
    arrangement = BigBandArranger.arrange(melody, chords)

    # Step 4: Apply section-specific dynamics
    timeline = form.get_section_timeline()

    for start_bar, end_bar, section in timeline:
        # Get notes in this section (pseudo-code)
        section_notes = extract_notes_in_range(arrangement, start_bar, end_bar)

        # Apply dynamics based on section
        shaped_notes = apply_dynamics_to_section(section_notes, section, form)

        # Update arrangement
        update_arrangement(arrangement, start_bar, shaped_notes)

    # Result:
    #   A1 section: mf (medium dynamics)
    #   A2 section: slightly louder
    #   B (Bridge): mp (softer for contrast)
    #   A3 (Shout): ff (loud, climactic!)
    """)

    print("✅ Form-based: Automatic dynamics based on musical form")


# ============================================================================
# INTEGRATION PATTERN 3: Shout Chorus
# ============================================================================

def example_3_shout_chorus():
    """
    Example 3: Apply shout chorus dynamics (climactic final A section)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Shout Chorus Dynamics")
    print("=" * 80)

    print("""
    # Make the final A section POWERFUL:

    from transformation.dynamic_shaping import BigBandDynamics

    # Get notes for final A section (shout chorus)
    shout_chorus_notes = arrangement_a3

    # Apply shout chorus treatment
    shout_chorus_notes = BigBandDynamics.apply_shout_chorus_dynamics(
        shout_chorus_notes,
        intensity=0.95  # Very loud!
    )

    # Result:
    #   - Base velocity: 120-125 (fff)
    #   - Strong downbeat accents
    #   - Building energy (crescendo throughout)
    #   - This is the CLIMAX of the arrangement!
    """)

    print("✅ Shout chorus: Climactic big band finale")


# ============================================================================
# INTEGRATION PATTERN 4: Section Balance
# ============================================================================

def example_4_section_balance():
    """
    Example 4: Balance sections properly (lead louder, rhythm softer)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Section Balance")
    print("=" * 80)

    print("""
    # Apply proper big band balance:

    from transformation.dynamic_shaping import BigBandDynamics

    # Balance entire arrangement
    balanced_arrangement = BigBandDynamics.apply_section_balance(
        arrangement,
        lead_boost=12,      # Lead melody +12 velocity
        brass_power=8,      # Brass +8 velocity
        sax_blend=0,        # Saxes unchanged (blend)
        rhythm_reduction=-8  # Rhythm section -8 (support role)
    )

    # Result:
    #   Lead melody: Clearly audible on top
    #   Brass: Powerful but not overpowering
    #   Saxes: Blended harmony
    #   Rhythm: Supportive, not too loud
    """)

    print("✅ Section balance: Professional big band mix")


# ============================================================================
# COMPLETE WORKFLOW
# ============================================================================

def example_5_complete_workflow():
    """
    Example 5: Complete workflow from form to dynamics
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Complete Workflow")
    print("=" * 80)

    print("""
    # Complete integration workflow:

    from generators.form_generator import FormGenerator, FormType
    from transformation.arrangement_engine import BigBandArranger
    from transformation.dynamic_shaping import (
        DynamicShaping,
        BigBandDynamics,
        generate_dynamic_map_for_form,
        apply_dynamics_to_section,
        PhraseContour,
        AccentPattern
    )

    # STEP 1: Generate form
    form = FormGenerator.generate_form(FormType.AABA, tonic_key=60, tempo=140)

    # STEP 2: Generate harmony and melody (simplified)
    chords = generate_chord_progression(form)
    melody = generate_melody(chords)

    # STEP 3: Create arrangement (static velocities)
    arrangement = BigBandArranger.arrange(melody, chords)

    # STEP 4: Apply form-based dynamics
    timeline = form.get_section_timeline()
    for start_bar, end_bar, section in timeline:
        # Extract notes for this section
        for section_name in arrangement:
            section_notes = get_notes_in_bars(
                arrangement[section_name],
                start_bar,
                end_bar
            )

            # Apply section-appropriate dynamics
            shaped = apply_dynamics_to_section(section_notes, section, form)

            # Update arrangement
            set_notes_in_bars(arrangement[section_name], start_bar, shaped)

    # STEP 5: Apply shout chorus boost to final A
    if has_shout_chorus(form):
        shout_start, shout_end = get_shout_chorus_bars(form)
        for section_name in arrangement:
            shout_notes = get_notes_in_bars(
                arrangement[section_name],
                shout_start,
                shout_end
            )
            boosted = BigBandDynamics.apply_shout_chorus_dynamics(
                shout_notes,
                intensity=0.9
            )
            set_notes_in_bars(arrangement[section_name], shout_start, boosted)

    # STEP 6: Balance sections
    arrangement = BigBandDynamics.apply_section_balance(
        arrangement,
        lead_boost=10,
        brass_power=6,
        sax_blend=0,
        rhythm_reduction=-6
    )

    # STEP 7: Export to MIDI
    export_to_midi(arrangement, "output_with_dynamics.mid")

    # Result: Professional sounding arrangement with:
    #   ✓ Proper phrase shaping (arch contours)
    #   ✓ Section-appropriate dynamics (intro soft, shout loud)
    #   ✓ Balanced mix (lead on top, rhythm supporting)
    #   ✓ Musical accents and phrasing
    """)

    print("✅ Complete workflow: Form → Arrangement → Dynamics → MIDI")


# ============================================================================
# ADDITIONAL TECHNIQUES
# ============================================================================

def example_6_advanced_techniques():
    """
    Example 6: Advanced dynamic techniques
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Advanced Techniques")
    print("=" * 80)

    print("""
    # Advanced dynamic shaping techniques:

    # 1. CRESCENDO with different curves
    notes_linear = DynamicShaping.apply_crescendo(
        notes, 50, 110, curve="linear"
    )
    notes_exponential = DynamicShaping.apply_crescendo(
        notes, 50, 110, curve="exponential"  # Starts slow, builds faster
    )
    notes_logarithmic = DynamicShaping.apply_crescendo(
        notes, 50, 110, curve="logarithmic"  # Starts fast, builds slower
    )

    # 2. BREATH MARKS for wind instruments
    sax_notes = DynamicShaping.mark_breath_points(
        sax_notes,
        phrase_length_bars=4,
        breath_gap=0.15  # 0.15 beat gap at phrase boundaries
    )

    # 3. CUSTOM CONTOURS for specific effects
    # Bridge: descending (calming down)
    bridge_notes = DynamicShaping.apply_phrase_contour(
        bridge_notes,
        contour=PhraseContour.DESCENDING
    )

    # Intro: ascending (building anticipation)
    intro_notes = DynamicShaping.apply_phrase_contour(
        intro_notes,
        contour=PhraseContour.ASCENDING
    )

    # Shout chorus: peak early then sustain
    shout_notes = DynamicShaping.apply_phrase_contour(
        shout_notes,
        contour=PhraseContour.PEAK_EARLY,
        base_velocity=110,
        variation_range=15
    )

    # 4. ACCENT PATTERNS for rhythmic interest
    # Syncopated accents for jazz feel
    jazz_notes = DynamicShaping.apply_accent_pattern(
        notes,
        pattern=AccentPattern.SYNCOPATED,
        accent_amount=15
    )

    # Strong downbeats for march
    march_notes = DynamicShaping.apply_accent_pattern(
        notes,
        pattern=AccentPattern.DOWNBEAT,
        accent_amount=20
    )
    """)

    print("✅ Advanced techniques: Curves, breath marks, custom contours")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all integration examples"""
    print("\n🎵 DYNAMIC SHAPING - INTEGRATION EXAMPLES")
    print("=" * 80)
    print("Agent 9: Dynamic Shaping & Phrasing Master")
    print("=" * 80)
    print("\nThese examples show how to integrate DynamicShaping with BigBandArranger")
    print("to create musically expressive arrangements.\n")

    example_1_basic_integration()
    example_2_form_based_dynamics()
    example_3_shout_chorus()
    example_4_section_balance()
    example_5_complete_workflow()
    example_6_advanced_techniques()

    print("\n" + "=" * 80)
    print("✅ ALL INTEGRATION EXAMPLES SHOWN")
    print("=" * 80)

    print("\n📚 KEY TAKEAWAYS:")
    print("=" * 80)
    print("""
1. BASIC: Apply dynamics after arrangement creation
   → Use DynamicShaping.apply_phrase_contour() on each section

2. FORM-BASED: Let form structure guide dynamics
   → Use generate_dynamic_map_for_form() for automatic section dynamics

3. SHOUT CHORUS: Make climactic sections POWERFUL
   → Use BigBandDynamics.apply_shout_chorus_dynamics()

4. BALANCE: Mix sections properly
   → Use BigBandDynamics.apply_section_balance()

5. CUSTOM: Fine-tune with advanced techniques
   → Crescendo curves, breath marks, accent patterns

RESULT: Arrangements that sound HUMAN and MUSICAL!
    """)

    print("=" * 80)
    print("\n🎺 Ready to make big band arrangements that SWING with dynamics!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
