#!/usr/bin/env python3
"""
Component System Usage Examples

Demonstrates all capabilities of the Component Abstraction Layer including:
- Simple genre fusion
- Multi-component mixing
- Custom parameters
- Factory usage
- Custom components
- Advanced patterns

Author: Agent 2 - Component Abstraction Layer
Date: 2025
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_generator.core.component_system import (
    ComponentType,
    ComponentSpec,
    MusicalComponent,
    GenerationContext,
    ComponentFactory,
    CompositionBuilder,
    global_factory,
    register_component,
    list_available_components,
    create_simple_composition,
)


def example_1_simple_fusion():
    """
    Example 1: Simple Jazz-Funk Fusion

    Mix jazz harmony with funk rhythm in the simplest way possible.
    """
    print("=" * 70)
    print("EXAMPLE 1: Simple Jazz-Funk Fusion")
    print("=" * 70)

    composition = (CompositionBuilder()
        .set_tempo(120)
        .set_key("C")
        .set_length(8)
        .add_component(ComponentType.HARMONY, genre="jazz")
        .add_component(ComponentType.RHYTHM, genre="funk")
        .build())

    print(f"✓ Created composition:")
    print(f"  Tempo: {composition.get_tempo()} BPM")
    print(f"  Key: {composition.get_key()}")
    print(f"  Length: {composition.get_length()} measures")

    harmony = composition.get_component(ComponentType.HARMONY)
    rhythm = composition.get_component(ComponentType.RHYTHM)

    print(f"\n✓ Generated components:")
    print(f"  Harmony (jazz): {harmony}")
    print(f"  Rhythm (funk): {rhythm}")

    print("\n✓ Result: Jazz chord progression with funk groove!")
    print()


def example_2_three_way_fusion():
    """
    Example 2: Three-Way Fusion

    Bebop melody + Reggae rhythm + Funk bass
    """
    print("=" * 70)
    print("EXAMPLE 2: Three-Way Fusion (Bebop + Reggae + Funk)")
    print("=" * 70)

    composition = (CompositionBuilder()
        .set_tempo(90)
        .set_key("Gm")
        .set_time_signature(4, 4)
        .set_length(16)
        .add_component(ComponentType.RHYTHM, genre="reggae")
        .add_component(ComponentType.HARMONY, genre="jazz")
        .add_component(ComponentType.BASS, genre="funk")
        .build())

    print(f"✓ Created three-way fusion:")
    print(f"  Tempo: {composition.get_tempo()} BPM (slow reggae feel)")
    print(f"  Key: {composition.get_key()}")

    rhythm = composition.get_component(ComponentType.RHYTHM)
    harmony = composition.get_component(ComponentType.HARMONY)
    bass = composition.get_component(ComponentType.BASS)

    print(f"\n✓ Components:")
    print(f"  Rhythm: {rhythm['groove_type']} (reggae one-drop)")
    print(f"  Harmony: {len(harmony)} chords (jazz progression)")
    print(f"  Bass: {len(bass)} notes (funk slap bass)")

    print("\n✓ Result: Reggae-Jazz-Funk hybrid!")
    print()


def example_3_custom_parameters():
    """
    Example 3: Custom Parameters

    Pass custom parameters to components for fine control.
    """
    print("=" * 70)
    print("EXAMPLE 3: Custom Parameters")
    print("=" * 70)

    composition = (CompositionBuilder()
        .set_tempo(140)
        .set_key("Dm")
        .set_length(16)
        .add_component(
            ComponentType.HARMONY,
            genre="jazz",
            complexity=0.9,
            use_extensions=True,
            use_substitutions=True
        )
        .add_component(
            ComponentType.RHYTHM,
            genre="funk",
            syncopation=0.85
        )
        .build())

    print("✓ Created composition with custom parameters:")
    print("  Harmony: jazz, complexity=0.9, extensions=True")
    print("  Rhythm: funk, syncopation=0.85")

    harmony = composition.get_component(ComponentType.HARMONY)
    rhythm = composition.get_component(ComponentType.RHYTHM)

    print(f"\n✓ Results:")
    print(f"  Harmony: {len(harmony)} chords")
    print(f"  Rhythm syncopation: {rhythm['syncopation']}")

    print("\n✓ Result: Highly complex jazz harmony with heavy funk syncopation!")
    print()


def example_4_list_available():
    """
    Example 4: List Available Components

    Discover what components are available.
    """
    print("=" * 70)
    print("EXAMPLE 4: List Available Components")
    print("=" * 70)

    all_components = list_available_components()

    print("✓ Available components:\n")
    for comp_type, genres in all_components.items():
        print(f"  {comp_type.value.upper()}:")
        print(f"    Genres: {', '.join(genres)}")
        print()

    # Check specific component
    harmony_genres = global_factory.list_available(ComponentType.HARMONY)
    print(f"✓ Harmony genres: {harmony_genres}")

    has_jazz = global_factory.is_registered(ComponentType.HARMONY, "jazz")
    print(f"✓ Has jazz harmony: {has_jazz}")
    print()


def example_5_helper_function():
    """
    Example 5: Helper Function

    Use create_simple_composition for quick compositions.
    """
    print("=" * 70)
    print("EXAMPLE 5: Helper Function")
    print("=" * 70)

    composition = create_simple_composition(
        tempo=120,
        key="C",
        harmony_genre="blues",
        rhythm_genre="funk",
        length_measures=12
    )

    print("✓ Created composition with helper function:")
    print(f"  Tempo: {composition.get_tempo()} BPM")
    print(f"  Key: {composition.get_key()}")
    print(f"  Length: {composition.get_length()} measures")

    harmony = composition.get_component(ComponentType.HARMONY)
    print(f"\n✓ Harmony: {harmony}")

    print("\n✓ Result: Quick blues-funk composition in one line!")
    print()


def example_6_context_inspection():
    """
    Example 6: Context Inspection

    Inspect the generation context and component outputs.
    """
    print("=" * 70)
    print("EXAMPLE 6: Context Inspection")
    print("=" * 70)

    composition = (CompositionBuilder()
        .set_tempo(110)
        .set_key("Am")
        .add_component(ComponentType.RHYTHM, "funk")
        .add_component(ComponentType.HARMONY, "jazz")
        .add_component(ComponentType.BASS, "funk")
        .build())

    context = composition.context

    print("✓ Context properties:")
    print(f"  Tempo: {context.tempo}")
    print(f"  Key: {context.key}")
    print(f"  Time signature: {context.time_signature}")
    print(f"  Length: {context.length_measures} measures")

    print("\n✓ Generated components:")
    print(f"  Has rhythm: {context.has_component(ComponentType.RHYTHM)}")
    print(f"  Has harmony: {context.has_component(ComponentType.HARMONY)}")
    print(f"  Has bass: {context.has_component(ComponentType.BASS)}")
    print(f"  Has drums: {context.has_component(ComponentType.DRUMS)}")

    print("\n✓ Component outputs:")
    if context.rhythm:
        print(f"  Rhythm: {context.rhythm}")
    if context.harmony:
        print(f"  Harmony: {context.harmony[:4]}...")  # First 4 chords
    if context.bass:
        print(f"  Bass: {len(context.bass)} notes")

    print()


def example_7_custom_component():
    """
    Example 7: Creating a Custom Component

    Define and use a custom component.
    """
    print("=" * 70)
    print("EXAMPLE 7: Custom Component")
    print("=" * 70)

    # Define custom component
    @register_component(ComponentType.HARMONY, "custom")
    class CustomHarmonyComponent(MusicalComponent):
        """Custom harmony generator"""

        def get_type(self):
            return ComponentType.HARMONY

        def get_compatible_genres(self):
            return ["custom"]

        def generate(self, context):
            """Generate simple custom progression"""
            # I - IV - V - I progression
            return ["C", "F", "G", "C"] * (context.length_measures // 4)

    print("✓ Defined custom harmony component")

    # Use custom component
    composition = (CompositionBuilder()
        .set_tempo(120)
        .set_key("C")
        .set_length(8)
        .add_component(ComponentType.HARMONY, "custom")
        .add_component(ComponentType.RHYTHM, "funk")
        .build())

    harmony = composition.get_component(ComponentType.HARMONY)
    print(f"\n✓ Generated with custom component:")
    print(f"  Harmony: {harmony}")

    print("\n✓ Result: Custom I-IV-V-I progression!")
    print()


def example_8_multiple_compositions():
    """
    Example 8: Multiple Compositions

    Create multiple compositions with different settings.
    """
    print("=" * 70)
    print("EXAMPLE 8: Multiple Compositions")
    print("=" * 70)

    # Composition 1: Uptempo jazz-funk
    comp1 = (CompositionBuilder()
        .set_tempo(140)
        .set_key("F")
        .add_component(ComponentType.HARMONY, "jazz")
        .add_component(ComponentType.RHYTHM, "funk")
        .build())

    # Composition 2: Slow reggae-jazz
    comp2 = (CompositionBuilder()
        .set_tempo(85)
        .set_key("Gm")
        .add_component(ComponentType.HARMONY, "jazz")
        .add_component(ComponentType.RHYTHM, "reggae")
        .build())

    # Composition 3: Medium blues-funk
    comp3 = (CompositionBuilder()
        .set_tempo(110)
        .set_key("Bb")
        .add_component(ComponentType.HARMONY, "blues")
        .add_component(ComponentType.RHYTHM, "funk")
        .build())

    print("✓ Created 3 compositions:\n")
    print(f"  1. Uptempo jazz-funk: {comp1.get_tempo()} BPM, {comp1.get_key()}")
    print(f"  2. Slow reggae-jazz: {comp2.get_tempo()} BPM, {comp2.get_key()}")
    print(f"  3. Medium blues-funk: {comp3.get_tempo()} BPM, {comp3.get_key()}")

    print("\n✓ Result: Three different styles, ready for comparison!")
    print()


def example_9_factory_usage():
    """
    Example 9: Direct Factory Usage

    Use the ComponentFactory directly for advanced control.
    """
    print("=" * 70)
    print("EXAMPLE 9: Direct Factory Usage")
    print("=" * 70)

    # Create a separate factory
    custom_factory = ComponentFactory()

    # Import component class
    from midi_generator.core.component_system import JazzHarmonyComponent

    # Register components
    custom_factory.register(ComponentType.HARMONY, "jazz", JazzHarmonyComponent)

    print("✓ Created custom factory and registered jazz harmony")

    # Create component instance
    component = custom_factory.create(
        ComponentType.HARMONY,
        "jazz",
        complexity=0.8
    )

    print(f"✓ Created component: {component.__class__.__name__}")

    # Create context and generate
    context = GenerationContext(
        tempo=120,
        key="C",
        time_signature=(4, 4),
        length_measures=8
    )

    harmony = component.generate(context)
    print(f"\n✓ Generated harmony: {harmony}")

    print("\n✓ Result: Direct factory control for advanced use cases!")
    print()


def example_10_component_dependencies():
    """
    Example 10: Component Dependencies

    Demonstrate how dependencies are automatically resolved.
    """
    print("=" * 70)
    print("EXAMPLE 10: Component Dependencies")
    print("=" * 70)

    # Add components in WRONG order - builder will fix it
    composition = (CompositionBuilder()
        .set_tempo(120)
        .set_key("C")
        .add_component(ComponentType.BASS, "funk")     # Needs harmony + rhythm
        .add_component(ComponentType.HARMONY, "jazz")  # Needs rhythm
        .add_component(ComponentType.RHYTHM, "funk")   # No dependencies
        .build())

    print("✓ Added components in wrong order:")
    print("  1. BASS (depends on harmony + rhythm)")
    print("  2. HARMONY (depends on rhythm)")
    print("  3. RHYTHM (no dependencies)")

    print("\n✓ Builder automatically reordered to:")
    print("  1. RHYTHM (generated first)")
    print("  2. HARMONY (generated second, uses rhythm)")
    print("  3. BASS (generated last, uses harmony + rhythm)")

    print(f"\n✓ All components generated successfully:")
    print(f"  Has rhythm: {composition.has_component(ComponentType.RHYTHM)}")
    print(f"  Has harmony: {composition.has_component(ComponentType.HARMONY)}")
    print(f"  Has bass: {composition.has_component(ComponentType.BASS)}")

    print("\n✓ Result: Dependency resolution handled automatically!")
    print()


def run_all_examples():
    """Run all examples"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "COMPONENT SYSTEM EXAMPLES" + " " * 28 + "║")
    print("║" + " " * 20 + "Agent 2 Demo Suite" + " " * 29 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    examples = [
        example_1_simple_fusion,
        example_2_three_way_fusion,
        example_3_custom_parameters,
        example_4_list_available,
        example_5_helper_function,
        example_6_context_inspection,
        example_7_custom_component,
        example_8_multiple_compositions,
        example_9_factory_usage,
        example_10_component_dependencies,
    ]

    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"✗ Example {i} failed: {e}\n")

    print("=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  ✓ Mix ANY component from ANY genre")
    print("  ✓ Automatic dependency resolution")
    print("  ✓ Fluent builder API")
    print("  ✓ Custom parameters for fine control")
    print("  ✓ Extensible with custom components")
    print("  ✓ Factory pattern for advanced usage")
    print("\n💡 The Component System brings Photoshop-level modularity to music!")
    print()


if __name__ == "__main__":
    run_all_examples()
