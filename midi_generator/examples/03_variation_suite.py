#!/usr/bin/env python3
"""
Example 03: Generate Theme and Variations

Demonstrates automatic variation generation in the classical tradition:
- Paraphrase variations (ornamentation)
- Character variations (major/minor, staccato/legato)
- Rhythmic variations (augmentation, diminution, dotted rhythms)
- Textural variations (Alberti bass, etc.)

Generates a complete variation suite like:
- Bach: Goldberg Variations
- Beethoven: Diabelli Variations
- Brahms: Handel Variations
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformation.variation_generator import VariationSuiteGenerator


def main():
    """Generate theme and variations."""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   THEME AND VARIATIONS GENERATOR                            ║
║                                                                             ║
║  Generate complete variation suites in the classical tradition.            ║
║                                                                             ║
║  Variation Techniques:                                                      ║
║    1. Paraphrase      - Baroque ornamentation (trills, turns, mordents)   ║
║    2. Minor Mode      - Transform major theme to minor (or vice versa)    ║
║    3. Staccato        - Short, detached articulation                       ║
║    4. Augmentation    - Slower, doubled durations                          ║
║    5. Diminution      - Faster, halved durations                           ║
║    6. Dotted Rhythm   - Long-short rhythmic pattern                        ║
║    7. Romantic        - Romantic-style chromatic ornamentation             ║
║    8. Legato          - Smooth, connected notes                            ║
║    9. Alberti Bass    - Broken chord accompaniment pattern                 ║
║   10. Finale          - Virtuosic conclusion with dense ornamentation      ║
║                                                                             ║
║  Usage:                                                                     ║
║    python 03_variation_suite.py <theme.mid> [num_variations]              ║
║                                                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Check arguments
    if len(sys.argv) < 2:
        print("❌ Error: Please provide a theme MIDI file")
        print("\nExample:")
        print("  python 03_variation_suite.py theme.mid 10")
        return

    theme_file = sys.argv[1]
    num_variations = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # Verify file exists
    if not Path(theme_file).exists():
        print(f"❌ Error: File not found: {theme_file}")
        return

    # Generate variation suite
    print(f"\n🎼 Generating variation suite from: {Path(theme_file).name}\n")

    generator = VariationSuiteGenerator(theme_file)

    output_files = generator.generate_suite(
        num_variations=num_variations
    )

    print("\n🎵 Variation Suite Generated:")
    for i, filepath in enumerate(output_files):
        filename = Path(filepath).name
        if i == 0:
            print(f"  0. {filename} (THEME)")
        else:
            print(f"  {i}. {filename}")

    print(f"\n✅ Complete! Generated {len(output_files)} files\n")


if __name__ == "__main__":
    main()
