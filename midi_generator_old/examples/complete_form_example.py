#!/usr/bin/env python3
"""
Complete Form & Structure Example
Demonstrates integration of all three Agent 5 modules:
- Form Generator
- Transition Engine
- Development Engine

This example shows how to create a complete musical composition
with proper form, transitions, and motivic development.

Author: Agent 5 - Form & Structure Engine
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generators.form_generator import (
    FormGenerator, FormType, MusicalForm, FormSection
)
from generators.transition_engine import (
    TransitionEngine, ModulationType, TransitionType, TurnaroundGenerator
)
from generators.development_engine import (
    DevelopmentEngine, Motif, DevelopmentTechnique,
    ThematicTransformation, AdvancedDevelopment
)


def example_1_sonata_with_development():
    """
    Example 1: Complete Sonata Form with Motivic Development

    Demonstrates:
    - Sonata form structure
    - Thematic development in development section
    - Modulations between sections
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Sonata Form in C Major with Motivic Development")
    print("=" * 80)

    # Generate sonata form structure
    sonata = FormGenerator.generate_form(
        FormType.SONATA,
        tonic_key=60,  # C
        is_major=True,
        tempo=140,
        exposition_length=32,
        development_length=24,
        recapitulation_length=32,
        include_introduction=True,
        include_coda=True
    )

    print("\n📋 FORM STRUCTURE:")
    print(FormGenerator.print_form_analysis(sonata))

    # Create themes for the sonata
    print("\n🎵 THEMATIC MATERIAL:")

    # First theme (heroic)
    theme_1 = Motif(
        pitches=[60, 64, 67, 72, 67, 64, 60],  # C E G C G E C (arpeggio)
        durations=[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        name="first_theme",
        character="heroic, assertive"
    )
    print(f"\nFirst Theme: {theme_1.pitches}")
    print(f"  Character: {theme_1.character}")

    # Second theme (lyrical)
    theme_2 = Motif(
        pitches=[67, 69, 71, 72, 74, 72, 71, 69],  # G A B C D C B A
        durations=[1.5, 0.5, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        name="second_theme",
        character="lyrical, singing"
    )
    print(f"\nSecond Theme: {theme_2.pitches}")
    print(f"  Character: {theme_2.character}")

    # Generate development section variations
    print("\n🔄 DEVELOPMENT SECTION:")
    development_variations = DevelopmentEngine.create_development_section(
        theme_1,
        num_variations=6
    )

    for i, dev in enumerate(development_variations):
        print(f"  Variation {i+1} ({dev.technique.value}): {dev.developed_motif.pitches[:5]}...")

    # Generate modulations
    print("\n🔀 MODULATIONS:")

    # Exposition: C major → G major (dominant)
    mod_to_dominant = TransitionEngine.generate_modulation(
        from_key=60,  # C
        from_major=True,
        to_key=67,  # G
        to_major=True,
        technique=ModulationType.COMMON_CHORD
    )
    print(f"\nExposition transition: C major → G major")
    print(f"  Technique: {mod_to_dominant.technique.value}")
    print(f"  Pivot chords: {mod_to_dominant.pivot_chords}")

    # Back to tonic for recapitulation
    mod_to_tonic = TransitionEngine.generate_modulation(
        from_key=67,  # G
        from_major=True,
        to_key=60,  # C
        to_major=True,
        technique=ModulationType.COMMON_CHORD
    )
    print(f"\nRecapitulation return: G major → C major")
    print(f"  Technique: {mod_to_tonic.technique.value}")

    print("\n✅ Complete sonata form with development and modulations generated!")


def example_2_rondo_with_transitions():
    """
    Example 2: Rondo Form with Section Transitions

    Demonstrates:
    - Rondo form (ABACA)
    - Build-up and breakdown transitions
    - Turnarounds
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Rondo Form with Section Transitions")
    print("=" * 80)

    # Generate rondo form
    rondo = FormGenerator.generate_form(
        FormType.RONDO,
        tonic_key=67,  # G
        is_major=True,
        tempo=120,
        pattern="ABACA",
        section_length=8
    )

    print("\n📋 FORM STRUCTURE:")
    print(FormGenerator.print_form_analysis(rondo))

    # Generate transitions between sections
    print("\n🌊 SECTION TRANSITIONS:")

    # Transition A → B (build-up)
    transition_ab = TransitionEngine.generate_transition(
        TransitionType.BUILD_UP,
        length_bars=2,
        intensity=0.7
    )
    print(f"\nA → B transition: {transition_ab.transition_type.value}")
    print(f"  Dynamic: {transition_ab.start_dynamic:.2f} → {transition_ab.end_dynamic:.2f}")
    print(f"  Texture: {transition_ab.start_texture:.2f} → {transition_ab.end_texture:.2f}")

    # Transition B → A (return, breakdown)
    transition_ba = TransitionEngine.generate_transition(
        TransitionType.BREAKDOWN,
        length_bars=2,
        final_texture=0.4
    )
    print(f"\nB → A transition: {transition_ba.transition_type.value}")
    print(f"  Dynamic: {transition_ba.start_dynamic:.2f} → {transition_ba.end_dynamic:.2f}")

    # Drum fill before C section
    drum_fill = TransitionEngine.generate_transition(
        TransitionType.FILL,
        length_bars=1,
        fill_style="linear"
    )
    print(f"\nBefore C section: {drum_fill.transition_type.value}")
    print(f"  Style: {drum_fill.rhythmic_pattern}")

    print("\n✅ Rondo form with smooth transitions generated!")


def example_3_theme_and_variations_with_transformations():
    """
    Example 3: Theme and Variations with Thematic Transformations

    Demonstrates:
    - Theme and variations form
    - Liszt-style thematic transformations
    - Character variations
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Theme and Variations with Transformations")
    print("=" * 80)

    # Generate theme and variations form
    variations_form = FormGenerator.generate_form(
        FormType.THEME_AND_VARIATIONS,
        tonic_key=62,  # D
        is_major=False,
        tempo=100,
        theme_length=16,
        num_variations=6
    )

    print("\n📋 FORM STRUCTURE:")
    print(FormGenerator.print_form_analysis(variations_form))

    # Create original theme
    original_theme = Motif(
        pitches=[62, 65, 67, 69, 70, 69, 67, 65],  # D F G A Bb A G F
        durations=[1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        name="original_theme",
        character="simple, melancholic"
    )

    print("\n🎵 ORIGINAL THEME:")
    print(f"Pitches: {original_theme.pitches}")
    print(f"Character: {original_theme.character}")

    # Generate thematic transformations
    print("\n🎭 THEMATIC TRANSFORMATIONS:")

    # Variation 1: Heroic transformation
    var1 = ThematicTransformation.heroic_transformation(original_theme)
    print(f"\nVariation 1 (Heroic): {var1.pitches[:5]}...")
    print(f"  Character: {var1.character}")

    # Variation 2: Lyrical transformation
    var2 = ThematicTransformation.lyrical_transformation(original_theme)
    print(f"\nVariation 2 (Lyrical): {var2.pitches[:5]}...")
    print(f"  Character: {var2.character}")

    # Variation 3: March transformation
    var3 = ThematicTransformation.march_transformation(original_theme)
    print(f"\nVariation 3 (March): {var3.pitches[:5]}...")
    print(f"  Character: {var3.character}")
    print(f"  Durations: {var3.durations} (all equal for march feel)")

    # Variation 4: Pastoral transformation
    var4 = ThematicTransformation.pastoral_transformation(original_theme)
    print(f"\nVariation 4 (Pastoral): {var4.pitches[:5]}...")
    print(f"  Character: {var4.character}")

    # Variation 5: Dramatic transformation
    var5 = ThematicTransformation.dramatic_transformation(original_theme)
    print(f"\nVariation 5 (Dramatic): {var5.pitches[:5]}...")
    print(f"  Character: {var5.character}")

    print("\n✅ Theme and variations with transformations generated!")


def example_4_pop_song_with_turnarounds():
    """
    Example 4: Pop Song with Turnarounds and Fills

    Demonstrates:
    - Verse-Chorus form
    - Jazz and blues turnarounds
    - Drum fills between sections
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Pop Song with Turnarounds")
    print("=" * 80)

    # Generate verse-chorus form
    pop_song = FormGenerator.generate_form(
        FormType.VERSE_CHORUS,
        tonic_key=69,  # A
        is_major=False,
        tempo=128,
        verse_length=8,
        chorus_length=8,
        include_bridge=True,
        num_verses=2
    )

    print("\n📋 FORM STRUCTURE:")
    print(FormGenerator.print_form_analysis(pop_song))

    # Generate turnarounds
    print("\n🔁 TURNAROUNDS:")

    # Jazz turnaround (end of verse)
    jazz_turn = TransitionEngine.generate_turnaround(
        key=69,  # A
        is_major=False,
        style="jazz"
    )
    print(f"\nJazz turnaround (end of verse): {' → '.join(jazz_turn)}")

    # Blues turnaround (end of chorus)
    blues_turn = TransitionEngine.generate_turnaround(
        key=69,  # A
        is_major=False,
        style="blues"
    )
    print(f"Blues turnaround (end of chorus): {' → '.join(blues_turn)}")

    # Drum fills
    print("\n🥁 DRUM FILLS:")

    fill_1 = TransitionEngine.generate_transition(
        TransitionType.FILL,
        length_bars=1,
        fill_style="linear"
    )
    print(f"Linear fill (verse → chorus)")

    fill_2 = TransitionEngine.generate_transition(
        TransitionType.FILL,
        length_bars=1,
        fill_style="triplet"
    )
    print(f"Triplet fill (chorus → verse)")

    print("\n✅ Pop song with turnarounds and fills generated!")


def example_5_complete_integrated_piece():
    """
    Example 5: Complete Integrated Composition

    Demonstrates full integration of all three modules:
    - Form structure
    - Thematic development
    - Modulations
    - Transitions
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Complete Integrated Composition")
    print("=" * 80)
    print("This example shows how all modules work together to create")
    print("a complete, coherent musical composition.\n")

    # Step 1: Choose form
    print("STEP 1: Generate Form Structure")
    print("-" * 40)
    form = FormGenerator.generate_form(
        FormType.SONATA,
        tonic_key=60,  # C
        is_major=True,
        tempo=140,
        exposition_length=24,
        development_length=16,
        include_coda=True
    )
    print(f"✓ Generated {form.form_type.value} form")
    print(f"  Total length: {form.total_bars} bars")
    print(f"  Sections: {len(form.sections)}")

    # Step 2: Create thematic material
    print("\nSTEP 2: Create Thematic Material")
    print("-" * 40)
    main_theme = Motif(
        pitches=[60, 62, 64, 65, 67, 65, 64, 62],
        durations=[1.0, 0.5, 0.5, 1.0, 2.0, 0.5, 0.5, 2.0],
        name="main_theme",
        character="bold, energetic"
    )
    print(f"✓ Created main theme: {main_theme.pitches}")

    # Step 3: Develop theme for development section
    print("\nSTEP 3: Generate Development Section")
    print("-" * 40)
    developments = DevelopmentEngine.create_development_section(
        main_theme,
        num_variations=4
    )
    print(f"✓ Generated {len(developments)} development variations")
    for i, dev in enumerate(developments):
        print(f"  Var {i+1}: {dev.technique.value}")

    # Step 4: Plan modulations
    print("\nSTEP 4: Plan Modulations")
    print("-" * 40)

    # Find sections that need modulation
    timeline = form.get_section_timeline()
    modulations = []

    for i, (start, end, section) in enumerate(timeline[:-1]):
        next_section = timeline[i+1][2]

        # Check if key changes
        from_key = 60  # Would calculate from section.key_relationship
        to_key = 67 if "Second Theme" in next_section.name else 60

        if from_key != to_key:
            mod = TransitionEngine.generate_modulation(
                from_key=from_key,
                from_major=True,
                to_key=to_key,
                to_major=True,
                technique=ModulationType.COMMON_CHORD
            )
            modulations.append((section.name, next_section.name, mod))

    print(f"✓ Planned {len(modulations)} modulations")

    # Step 5: Add transitions
    print("\nSTEP 5: Add Section Transitions")
    print("-" * 40)

    transitions = []

    # Build-up to development
    trans1 = TransitionEngine.generate_transition(
        TransitionType.BUILD_UP,
        length_bars=2,
        intensity=0.8
    )
    transitions.append(("Exposition → Development", trans1))

    # Return transition to recapitulation
    trans2 = TransitionEngine.generate_transition(
        TransitionType.BREAKDOWN,
        length_bars=2,
        final_texture=0.5
    )
    transitions.append(("Development → Recapitulation", trans2))

    print(f"✓ Created {len(transitions)} section transitions")

    # Summary
    print("\n" + "=" * 80)
    print("✅ COMPLETE COMPOSITION GENERATED!")
    print("=" * 80)
    print(f"Form: {form.form_type.value} in C major")
    print(f"Total length: {form.total_bars} bars at {form.tempo} BPM")
    print(f"Sections: {len(form.sections)}")
    print(f"Thematic developments: {len(developments)}")
    print(f"Modulations: {len(modulations)}")
    print(f"Transitions: {len(transitions)}")
    print("\nThis composition demonstrates complete integration of:")
    print("  • Form structure (sonata form)")
    print("  • Motivic development (variations in development section)")
    print("  • Modulations (between sections)")
    print("  • Smooth transitions (build-ups, breakdowns)")
    print("\n" + "=" * 80)


def main():
    """Run all examples"""
    print("\n" + "🎵" * 40)
    print("AGENT 5: FORM & STRUCTURE ENGINE")
    print("Complete Integration Examples")
    print("🎵" * 40)

    example_1_sonata_with_development()
    example_2_rondo_with_transitions()
    example_3_theme_and_variations_with_transformations()
    example_4_pop_song_with_turnarounds()
    example_5_complete_integrated_piece()

    print("\n" + "🎵" * 40)
    print("ALL EXAMPLES COMPLETE!")
    print("🎵" * 40)
    print("\nThese modules provide the structural foundation for")
    print("sophisticated musical composition with proper form,")
    print("development, and transitions.")
    print("\n")


if __name__ == "__main__":
    main()
