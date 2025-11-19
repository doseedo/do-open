#!/usr/bin/env python3
"""
Groove Quantization Integration Examples

Demonstrates integration with other modules in the harmonymodule library.

Author: Agent 7
Date: 2025
"""

from groove_quantization import GrooveQuantization, Note

def example_1_basic_groove_chain():
    """Example 1: Basic groove processing chain."""
    print("=" * 70)
    print("Example 1: Basic Groove Processing Chain")
    print("=" * 70)

    gq = GrooveQuantization()

    # Create a simple drum pattern
    kick = [Note(36, i * 480, 0.5, velocity=100) for i in range(4)]
    snare = [Note(38, 480 + i * 960, 0.5, velocity=90) for i in range(2)]
    hihat = [Note(42, i * 120, 0.1, velocity=70) for i in range(16)]

    print(f"\nOriginal pattern:")
    print(f"  Kicks: {len(kick)} notes")
    print(f"  Snares: {len(snare)} notes")
    print(f"  Hi-hats: {len(hihat)} notes")

    # Apply 60% swing to hi-hats
    hihat_swung = gq.apply_swing(hihat, swing_percent=60.0)

    # Apply J Dilla feel to kick and snare
    kick_dilla = gq.create_j_dilla_swing(kick, drunk_factor=0.7)
    snare_dilla = gq.create_j_dilla_swing(snare, drunk_factor=0.7)

    # Add microtiming to everything
    kick_final = gq.apply_microtiming(kick_dilla, variance_ms=5.0)
    snare_final = gq.apply_microtiming(snare_dilla, variance_ms=5.0)
    hihat_final = gq.apply_microtiming(hihat_swung, variance_ms=3.0)

    # Humanize velocities
    kick_final = gq.humanize_velocities(kick_final, variance=5)
    snare_final = gq.humanize_velocities(snare_final, variance=5)
    hihat_final = gq.humanize_velocities(hihat_final, variance=3)

    print(f"\nProcessed pattern with groove:")
    print(f"  Kick timing sample: {[n.start_time for n in kick_final[:2]]}")
    print(f"  Kick velocity sample: {[n.velocity for n in kick_final[:2]]}")
    print(f"  Snare timing sample: {[n.start_time for n in snare_final]}")
    print(f"  Hi-hat timing sample: {[n.start_time for n in hihat_final[:4]]}")

    print("✓ Applied: 60% swing, J Dilla feel, microtiming, velocity humanization")


def example_2_per_instrument_timing():
    """Example 2: Per-instrument groove offsets."""
    print("\n" + "=" * 70)
    print("Example 2: Per-Instrument Groove Offsets")
    print("=" * 70)

    gq = GrooveQuantization()

    # Create tracks
    tracks = {
        "kick": [Note(36, i * 480, 0.5, velocity=100) for i in range(4)],
        "snare": [Note(38, 480 + i * 960, 0.5, velocity=90) for i in range(2)],
        "hihat": [Note(42, i * 120, 0.1, velocity=70) for i in range(16)],
        "bass": [Note(40, i * 480, 1.0, velocity=85) for i in range(4)],
    }

    print(f"\nOriginal track timings:")
    for name, notes in tracks.items():
        print(f"  {name}: {notes[0].start_time}")

    # Define offsets (in milliseconds)
    # Hi-hat slightly ahead, snare slightly behind, kick and bass on time
    offsets = {
        "hihat": +3.0,   # Hi-hat 3ms ahead
        "kick": 0.0,     # Kick on time
        "snare": -2.0,   # Snare 2ms behind
        "bass": -1.0,    # Bass 1ms behind
    }

    grooved_tracks = gq.per_instrument_offset(tracks, offsets)

    print(f"\nOffset track timings:")
    for name, notes in grooved_tracks.items():
        original = tracks[name][0].start_time
        offset = notes[0].start_time
        diff = offset - original
        print(f"  {name}: {offset} (offset: {diff:+.2f} ticks)")

    print("✓ Applied per-instrument timing offsets for ensemble feel")


def example_3_groove_template_workflow():
    """Example 3: Extract groove from reference and apply to new parts."""
    print("\n" + "=" * 70)
    print("Example 3: Groove Template Extraction & Application")
    print("=" * 70)

    gq = GrooveQuantization()

    # Simulate a "reference performance" with human-like timing
    # (In real use, this would come from MIDI file)
    reference = [
        Note(36, 5, 0.5, velocity=102),      # Slightly early, loud
        Note(42, 128, 0.1, velocity=68),     # Slightly late, soft
        Note(38, 485, 0.5, velocity=88),     # Slightly late, medium
        Note(42, 608, 0.1, velocity=71),     # Slightly late, soft
        Note(36, 965, 0.5, velocity=98),     # Slightly late, loud
    ]

    print(f"\nReference performance: {len(reference)} notes")
    print(f"  Timing variations from grid:")
    expected_times = [0, 120, 480, 600, 960]
    for i, (note, expected) in enumerate(zip(reference, expected_times)):
        deviation = note.start_time - expected
        print(f"    Note {i+1}: {deviation:+.1f} ticks")

    # Extract groove template
    template = gq.extract_groove_template(
        reference,
        resolution=16,
        name="custom_drummer"
    )

    print(f"\nExtracted template '{template.name}':")
    print(f"  Resolution: {template.resolution}")
    print(f"  Timing map entries: {len(template.timing_map)}")
    print(f"  Velocity map entries: {len(template.velocity_map)}")

    # Create a new, mechanical bass line
    mechanical_bass = [Note(40, i * 240, 0.5, velocity=80) for i in range(8)]

    print(f"\nMechanical bass line: {len(mechanical_bass)} notes")
    print(f"  Original timings: {[n.start_time for n in mechanical_bass[:4]]}")

    # Apply the extracted groove
    grooved_bass = gq.quantize_to_groove(
        mechanical_bass,
        template,
        amount=0.8  # 80% of the groove
    )

    print(f"  Grooved timings: {[n.start_time for n in grooved_bass[:4]]}")
    print(f"  Grooved velocities: {[n.velocity for n in grooved_bass[:4]]}")

    print("✓ Extracted human groove and applied to mechanical MIDI")


def example_4_builtin_templates():
    """Example 4: Using built-in groove templates."""
    print("\n" + "=" * 70)
    print("Example 4: Built-in Groove Templates")
    print("=" * 70)

    gq = GrooveQuantization()

    # Show available templates
    print(f"\nAvailable templates: {list(gq.groove_templates.keys())}")

    # Create a test pattern
    pattern = [Note(60, i * 120, 0.25, velocity=64) for i in range(8)]
    print(f"\nOriginal pattern timings: {[n.start_time for n in pattern[:4]]}")

    # Try each built-in template
    for template_name in gq.groove_templates.keys():
        template = gq.groove_templates[template_name]
        grooved = gq.quantize_to_groove(pattern, template, amount=1.0)
        print(f"\n{template_name} template:")
        print(f"  Description: {template.description}")
        print(f"  Grooved timings: {[n.start_time for n in grooved[:4]]}")

    print("\n✓ Demonstrated all built-in groove templates")


def example_5_complete_production_workflow():
    """Example 5: Complete production workflow combining multiple techniques."""
    print("\n" + "=" * 70)
    print("Example 5: Complete Production Workflow")
    print("=" * 70)

    gq = GrooveQuantization()

    # Step 1: Generate basic patterns
    print("\n[Step 1] Generate basic patterns")
    kick = [Note(36, i * 480, 0.5, velocity=100) for i in range(8)]
    snare = [Note(38, 480 + i * 960, 0.5, velocity=90) for i in range(4)]
    hihat = [Note(42, i * 120, 0.1, velocity=70) for i in range(32)]
    bass = [Note(40, i * 240, 0.5, velocity=80) for i in range(16)]

    print(f"  Generated: {len(kick)} kicks, {len(snare)} snares, "
          f"{len(hihat)} hi-hats, {len(bass)} bass notes")

    # Step 2: Apply MPC swing to hi-hats
    print("\n[Step 2] Apply 62% MPC swing to hi-hats")
    hihat = gq.apply_swing(hihat, swing_percent=62.0)

    # Step 3: Apply J Dilla feel to kick and snare
    print("\n[Step 3] Apply J Dilla 'drunk' feel to kick and snare")
    kick = gq.create_j_dilla_swing(kick, drunk_factor=0.7)
    snare = gq.create_j_dilla_swing(snare, drunk_factor=0.7)

    # Step 4: Apply samba groove to bass
    print("\n[Step 4] Apply Samba groove template to bass")
    samba = gq.groove_templates["Samba"]
    bass = gq.quantize_to_groove(bass, samba, amount=0.6)

    # Step 5: Add microtiming to everything
    print("\n[Step 5] Add microtiming humanization")
    kick = gq.apply_microtiming(kick, variance_ms=8.0)
    snare = gq.apply_microtiming(snare, variance_ms=8.0)
    hihat = gq.apply_microtiming(hihat, variance_ms=5.0)
    bass = gq.apply_microtiming(bass, variance_ms=6.0)

    # Step 6: Humanize velocities
    print("\n[Step 6] Humanize velocities")
    kick = gq.humanize_velocities(kick, variance=8)
    snare = gq.humanize_velocities(snare, variance=10)
    hihat = gq.humanize_velocities(hihat, variance=5)
    bass = gq.humanize_velocities(bass, variance=7)

    # Step 7: Apply per-instrument offsets
    print("\n[Step 7] Apply per-instrument timing offsets")
    tracks = {
        "kick": kick,
        "snare": snare,
        "hihat": hihat,
        "bass": bass,
    }
    offsets = {
        "hihat": +3.0,
        "kick": 0.0,
        "snare": -2.0,
        "bass": -1.0,
    }
    final_tracks = gq.per_instrument_offset(tracks, offsets)

    # Summary
    print("\n[Summary] Final groove-processed tracks:")
    for name, notes in final_tracks.items():
        print(f"  {name}:")
        print(f"    Timing sample: {[round(n.start_time, 1) for n in notes[:3]]}")
        print(f"    Velocity sample: {[n.velocity for n in notes[:3]]}")

    print("\n✓ Complete production workflow: swing → Dilla → groove template → "
          "microtiming → velocity → offsets")


def example_6_comparison_swing_amounts():
    """Example 6: Compare different swing amounts."""
    print("\n" + "=" * 70)
    print("Example 6: Swing Amount Comparison")
    print("=" * 70)

    gq = GrooveQuantization()

    # Create straight pattern
    straight = [Note(42, i * 120, 0.1, velocity=70) for i in range(8)]

    print(f"\nOriginal (straight) timings: {[n.start_time for n in straight]}")

    # Try different swing amounts
    swing_amounts = [50, 54, 60, 62, 66, 70]

    print("\nSwing amount comparison:")
    for swing in swing_amounts:
        swung = gq.apply_swing(straight, swing_percent=float(swing))
        # Show how the second note (index 1) is affected
        offset = swung[1].start_time - straight[1].start_time
        print(f"  {swing}% swing: 2nd note offset = {offset:+.1f} ticks "
              f"({swung[1].start_time:.1f})")

    print("\n✓ Demonstrated range of swing feels from 50% (straight) to 70% (very loose)")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GROOVE QUANTIZATION - INTEGRATION EXAMPLES")
    print("=" * 70)

    # Run all examples
    example_1_basic_groove_chain()
    example_2_per_instrument_timing()
    example_3_groove_template_workflow()
    example_4_builtin_templates()
    example_5_complete_production_workflow()
    example_6_comparison_swing_amounts()

    print("\n" + "=" * 70)
    print("ALL INTEGRATION EXAMPLES COMPLETE ✓")
    print("=" * 70)
    print("\nThese examples demonstrate:")
    print("  ✓ Basic groove processing chains")
    print("  ✓ Per-instrument timing offsets")
    print("  ✓ Groove template extraction and application")
    print("  ✓ Built-in template usage")
    print("  ✓ Complete production workflows")
    print("  ✓ Swing amount comparisons")
    print("\nReady for integration with:")
    print("  • MIDI Generator system")
    print("  • Bass Engine")
    print("  • Drum Pattern Engine")
    print("  • Film Scoring Engine")
    print("  • Any MIDI-generating module")
    print()
