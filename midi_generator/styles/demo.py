#!/usr/bin/env python3
"""
Modern Big Band Style Profiles - Interactive Demo
==================================================

This demo script showcases the modern big band style profiles and their usage.

Run this script to:
1. Compare all three modern arranger styles
2. Generate example voicings in each style
3. See style-specific recommendations
4. Explore the differences between arrangers

Author: Agent 15 - Modern Big Band Style Analyzer
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from styles.modern_profiles import (
    THAD_JONES_STYLE,
    MARIA_SCHNEIDER_STYLE,
    GORDON_GOODWIN_STYLE,
    ModernBigBandArranger,
    get_style_profile,
)


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_style_overview():
    """Demonstrate style profile overview."""
    print_header("MODERN BIG BAND STYLE PROFILES OVERVIEW")

    styles = [
        ("Thad Jones", THAD_JONES_STYLE),
        ("Maria Schneider", MARIA_SCHNEIDER_STYLE),
        ("Gordon Goodwin", GORDON_GOODWIN_STYLE),
    ]

    for name, style in styles:
        print(f"🎺 {name} ({style.era})")
        print("-" * 80)
        print(f"Mood: {style.characteristics.get('mood', 'N/A')}")
        print(f"Signature Techniques: {', '.join(style.characteristics.get('signature_techniques', []))}")
        print(f"Tempo Range: {style.typical_tempo_range[0]}-{style.typical_tempo_range[1]} BPM")
        print(f"Harmony Complexity: {style.harmony_complexity * 100:.0f}%")
        print(f"Rhythmic Complexity: {style.rhythmic_complexity * 100:.0f}%")
        print()


def demo_quartal_voicings():
    """Demonstrate quartal voicing generation."""
    print_header("QUARTAL VOICING GENERATION (Thad Jones Signature)")

    arranger = ModernBigBandArranger(THAD_JONES_STYLE)

    print(f"Style: {arranger.style.name}")
    print(f"Quartal Usage: {arranger.style.use_quartal * 100:.0f}%")
    print()

    # Generate examples
    print("Examples of quartal voicings (stacked 4ths):")
    print("-" * 80)

    chords = [
        ("C", 60),
        ("F", 65),
        ("G", 67),
    ]

    for chord_name, root in chords:
        voicing = arranger.generate_quartal_voicing(root, 4)
        intervals = [voicing[i+1] - voicing[i] for i in range(len(voicing)-1)]

        print(f"{chord_name} Quartal Voicing:")
        print(f"  MIDI Notes: {voicing}")
        print(f"  Intervals: {intervals} semitones")
        print(f"  Voicing Type: {'Perfect 4ths' if all(i == 5 for i in intervals) else 'Mixed 4ths/Aug4ths'}")
        print()


def demo_wide_spacing():
    """Demonstrate wide spacing voicings."""
    print_header("WIDE SPACING VOICINGS (Modern Technique)")

    arranger = ModernBigBandArranger(THAD_JONES_STYLE)

    print(f"Style: {arranger.style.name}")
    print(f"Spacing Preference: {arranger.style.voicing_spacing}")
    print()

    # Cmaj7 chord
    chord_tones = [60, 64, 67, 71]  # C, E, G, B
    print("Original Cmaj7 (close spacing):")
    print(f"  Notes: {chord_tones}")
    print(f"  Intervals: {[chord_tones[i+1] - chord_tones[i] for i in range(len(chord_tones)-1)]}")
    print()

    # Generate wide spacing
    wide_voicing = arranger.generate_wide_spacing_voicing(chord_tones, min_spacing=7)
    wide_intervals = [wide_voicing[i+1] - wide_voicing[i] for i in range(len(wide_voicing)-1)]

    print("Wide-Spaced Cmaj7 (Thad Jones style):")
    print(f"  Notes: {wide_voicing}")
    print(f"  Intervals: {wide_intervals}")
    print(f"  Average Spacing: {sum(wide_intervals) / len(wide_intervals):.1f} semitones")
    print()

    print("Compare:")
    print(f"  Close Spacing: Notes within {chord_tones[-1] - chord_tones[0]} semitones")
    print(f"  Wide Spacing: Notes spanning {wide_voicing[-1] - wide_voicing[0]} semitones")
    print(f"  Difference: {(wide_voicing[-1] - wide_voicing[0]) - (chord_tones[-1] - chord_tones[0])} semitones wider")


def demo_style_recommendations():
    """Demonstrate style-specific recommendations."""
    print_header("STYLE-SPECIFIC RECOMMENDATIONS")

    styles = [
        ("Thad Jones", THAD_JONES_STYLE),
        ("Maria Schneider", MARIA_SCHNEIDER_STYLE),
        ("Gordon Goodwin", GORDON_GOODWIN_STYLE),
    ]

    for name, style_profile in styles:
        arranger = ModernBigBandArranger(style_profile)

        print(f"🎼 {name} Arrangement")
        print("-" * 80)
        print(f"Recommended Intro: {arranger.suggest_intro_type()}")
        print(f"Recommended Ending: {arranger.suggest_ending_type()}")
        print(f"Typical Tempo: {arranger.get_typical_tempo()} BPM")
        print(f"Voicing Preference: {style_profile.voicing_preference}")
        print(f"Dynamic Range: {style_profile.dynamic_range}")
        print(f"Texture Density: {style_profile.texture_density}")
        print()


def demo_harmonic_comparison():
    """Compare harmonic complexity between styles."""
    print_header("HARMONIC COMPLEXITY COMPARISON")

    styles = [
        ("Thad Jones", THAD_JONES_STYLE),
        ("Maria Schneider", MARIA_SCHNEIDER_STYLE),
        ("Gordon Goodwin", GORDON_GOODWIN_STYLE),
    ]

    metrics = [
        ("Overall Complexity", "harmony_complexity"),
        ("Quartal Usage", "use_quartal"),
        ("Cluster Usage", "use_clusters"),
        ("Polychord Usage", "use_polychords"),
        ("Altered Dominants", "use_altered_dominants"),
    ]

    # Print header
    print(f"{'Metric':<25} {'Thad Jones':>15} {'Schneider':>15} {'Goodwin':>15}")
    print("-" * 80)

    # Print each metric
    for metric_name, attr_name in metrics:
        values = []
        for _, style in styles:
            value = getattr(style, attr_name)
            if isinstance(value, float):
                values.append(f"{value * 100:.0f}%")
            else:
                values.append(str(value))

        print(f"{metric_name:<25} {values[0]:>15} {values[1]:>15} {values[2]:>15}")

    print()
    print("Key Insights:")
    print("  • Maria Schneider has highest overall harmonic complexity (90%)")
    print("  • Thad Jones has highest quartal usage (60%) - signature technique")
    print("  • All three use rich chord extensions (7ths, 9ths, 11ths, 13ths)")


def demo_rhythmic_comparison():
    """Compare rhythmic characteristics."""
    print_header("RHYTHMIC CHARACTERISTICS COMPARISON")

    styles = [
        ("Thad Jones", THAD_JONES_STYLE),
        ("Maria Schneider", MARIA_SCHNEIDER_STYLE),
        ("Gordon Goodwin", GORDON_GOODWIN_STYLE),
    ]

    metrics = [
        ("Rhythmic Complexity", "rhythmic_complexity"),
        ("Use of Odd Meters", "use_odd_meters"),
        ("Syncopation Level", "syncopation_level"),
        ("Metric Modulation", "use_metric_modulation"),
    ]

    # Print header
    print(f"{'Metric':<25} {'Thad Jones':>15} {'Schneider':>15} {'Goodwin':>15}")
    print("-" * 80)

    # Print each metric
    for metric_name, attr_name in metrics:
        values = []
        for _, style in styles:
            value = getattr(style, attr_name)
            if isinstance(value, float):
                values.append(f"{value * 100:.0f}%")
            elif isinstance(value, bool):
                values.append("Yes" if value else "No")
            else:
                values.append(str(value))

        print(f"{metric_name:<25} {values[0]:>15} {values[1]:>15} {values[2]:>15}")

    print()
    print("Key Insights:")
    print("  • Gordon Goodwin has highest rhythmic complexity (90%)")
    print("  • Gordon Goodwin has highest syncopation (90%) - high energy style")
    print("  • Thad Jones uses odd meters (30%) - modern approach")


def demo_tempo_analysis():
    """Analyze tempo ranges for each style."""
    print_header("TEMPO RANGE ANALYSIS")

    styles = [
        ("Maria Schneider", MARIA_SCHNEIDER_STYLE, "Ballads to medium swing"),
        ("Thad Jones", THAD_JONES_STYLE, "Medium to up-tempo"),
        ("Gordon Goodwin", GORDON_GOODWIN_STYLE, "Fast, high-energy"),
    ]

    print("Tempo Ranges by Style:")
    print("-" * 80)
    print()

    for name, style, description in styles:
        min_t, max_t = style.typical_tempo_range
        print(f"{name}:")
        print(f"  Range: {min_t}-{max_t} BPM")
        print(f"  Description: {description}")
        print(f"  Average: {(min_t + max_t) / 2:.0f} BPM")
        print()

    # Visualization
    print("\nTempo Range Visualization:")
    print("-" * 80)
    print("50 BPM    100 BPM   150 BPM   200 BPM   250 BPM")
    print("|---------|---------|---------|---------|")

    for name, style, _ in styles:
        min_t, max_t = style.typical_tempo_range
        # Create visual bar
        start_pos = int((min_t - 50) / 2)
        length = int((max_t - min_t) / 2)
        bar = " " * start_pos + "█" * length
        print(f"{bar:<50} {name}")

    print()
    print("Notice:")
    print("  • Maria Schneider favors slower tempos (ballads, atmospheric)")
    print("  • Gordon Goodwin's minimum (160 BPM) > Schneider's maximum (140 BPM)")
    print("  • Thad Jones covers widest range (100-220 BPM) - versatile")


def interactive_menu():
    """Interactive menu for exploring styles."""
    print_header("MODERN BIG BAND STYLE PROFILES - INTERACTIVE DEMO")

    print("Welcome to the Modern Big Band Style Profiles Demo!")
    print()
    print("This demo showcases three groundbreaking arrangers:")
    print("  1. Thad Jones - Angular, modern, quartal harmony")
    print("  2. Maria Schneider - Orchestral, impressionistic, cinematic")
    print("  3. Gordon Goodwin - High energy, complex rhythms, virtuosic")
    print()

    demos = [
        ("Style Overview", demo_style_overview),
        ("Quartal Voicing Generation", demo_quartal_voicings),
        ("Wide Spacing Voicings", demo_wide_spacing),
        ("Style Recommendations", demo_style_recommendations),
        ("Harmonic Comparison", demo_harmonic_comparison),
        ("Rhythmic Comparison", demo_rhythmic_comparison),
        ("Tempo Analysis", demo_tempo_analysis),
    ]

    print("Available Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  {len(demos) + 1}. Run All Demos")
    print("  0. Exit")
    print()

    while True:
        try:
            choice = input("Select a demo (0-{0}): ".format(len(demos) + 1))
            choice = int(choice)

            if choice == 0:
                print("\nThank you for exploring Modern Big Band Styles!")
                break
            elif 1 <= choice <= len(demos):
                demos[choice - 1][1]()
                input("\nPress Enter to continue...")
            elif choice == len(demos) + 1:
                for _, demo_func in demos:
                    demo_func()
                    print()
                input("\nAll demos complete. Press Enter to continue...")
            else:
                print("Invalid choice. Please try again.")

        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            break


def run_all_demos():
    """Run all demos sequentially."""
    print_header("MODERN BIG BAND STYLE PROFILES - COMPLETE DEMO")

    demos = [
        demo_style_overview,
        demo_quartal_voicings,
        demo_wide_spacing,
        demo_style_recommendations,
        demo_harmonic_comparison,
        demo_rhythmic_comparison,
        demo_tempo_analysis,
    ]

    for demo in demos:
        demo()
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Modern Big Band Style Profiles Demo"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all demos sequentially"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive menu (default)"
    )

    args = parser.parse_args()

    if args.all:
        run_all_demos()
    else:
        interactive_menu()
