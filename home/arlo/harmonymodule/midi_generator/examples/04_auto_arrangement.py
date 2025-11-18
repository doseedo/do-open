#!/usr/bin/env python3
"""
Example 04: Auto-Arrangement

Demonstrates automatic arrangement of lead sheets to full ensemble:
- Big Band (saxes, brass, rhythm section)
- String Quartet (2 violins, viola, cello)
- Solo Piano (complete piano arrangement)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformation.arrangement_engine import ArrangementEngine


def main():
    """Auto-arrange a lead sheet."""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                      AUTO-ARRANGEMENT ENGINE                                ║
║                                                                             ║
║  Transform simple lead sheets into full ensemble arrangements.             ║
║                                                                             ║
║  Arrangement Styles:                                                        ║
║                                                                             ║
║  • big_band         - Full jazz big band                                   ║
║      - 5 saxophones (2 alto, 2 tenor, 1 bari)                             ║
║      - 4 trumpets                                                           ║
║      - 4 trombones                                                          ║
║      - Piano, bass, drums (rhythm section)                                 ║
║      - Sax soli, brass stabs, swing feel                                   ║
║                                                                             ║
║  • string_quartet   - Classical chamber ensemble                           ║
║      - Violin I (melody)                                                    ║
║      - Violin II (harmony)                                                  ║
║      - Viola (inner voice)                                                  ║
║      - Cello (bass line)                                                    ║
║                                                                             ║
║  • solo_piano       - Complete piano arrangement                           ║
║      - Right hand: Melody with fills                                       ║
║      - Left hand: Bass line + chord voicings                               ║
║                                                                             ║
║  Usage:                                                                     ║
║    python 04_auto_arrangement.py <leadsheet.mid> <style>                   ║
║                                                                             ║
║  Example:                                                                   ║
║    python 04_auto_arrangement.py melody.mid big_band                       ║
║                                                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Check arguments
    if len(sys.argv) < 3:
        print("❌ Error: Please provide input file and arrangement style")
        print("\nExample:")
        print("  python 04_auto_arrangement.py leadsheet.mid big_band")
        print("\nAvailable styles:")
        print("  - big_band")
        print("  - string_quartet")
        print("  - solo_piano")
        return

    input_file = sys.argv[1]
    style = sys.argv[2]

    # Verify file exists
    if not Path(input_file).exists():
        print(f"❌ Error: File not found: {input_file}")
        return

    # Verify style
    valid_styles = ['big_band', 'string_quartet', 'solo_piano']
    if style not in valid_styles:
        print(f"❌ Error: Unknown style: {style}")
        print(f"Available: {valid_styles}")
        return

    # Create arrangement
    print(f"\n🎺 Creating {style} arrangement...\n")

    engine = ArrangementEngine(input_file)

    output_path = engine.arrange(style)

    print(f"\n✅ Arrangement complete!")
    print(f"📁 Output: {output_path}")
    print("\nYou can now:")
    print("  • Open in a DAW (Logic, Ableton, FL Studio, etc.)")
    print("  • Import into notation software (Finale, Sibelius, MuseScore)")
    print("  • Render with a high-quality soundfont\n")


if __name__ == "__main__":
    main()
