#!/usr/bin/env python3
"""
Example 02: Style Transfer

Demonstrates MIDI style transfer across musical styles:
- Classical → Jazz
- Jazz → Classical
- Pop → Baroque
- Classical → Minimalist
- And more!
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformation.style_transfer import StyleTransfer, STYLE_PROFILES


def main():
    """Demonstrate style transfer."""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                       STYLE TRANSFER EXAMPLE                                ║
║                                                                             ║
║  Transform MIDI files between different musical styles.                    ║
║                                                                             ║
║  Available Styles:                                                          ║
║    • classical  - Traditional classical style                              ║
║    • jazz       - Jazz with swing, extensions, alterations                 ║
║    • pop        - Modern pop style                                         ║
║    • baroque    - Baroque ornamentation and counterpoint                   ║
║    • romantic   - Chromatic romantic style                                 ║
║    • minimalist - Minimalist repetition and simplicity                     ║
║                                                                             ║
║  Transformation Dimensions:                                                 ║
║    • Harmonic   - Reharmonization, chord substitution                      ║
║    • Rhythmic   - Swing/straight, syncopation, quantization                ║
║    • Melodic    - Ornamentation, interval preference                       ║
║                                                                             ║
║  Usage:                                                                     ║
║    python 02_style_transfer.py <input.mid> <target_style> [source_style]  ║
║                                                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Check arguments
    if len(sys.argv) < 3:
        print("❌ Error: Please provide input file and target style")
        print("\nExample:")
        print("  python 02_style_transfer.py input.mid jazz classical")
        print("\nAvailable styles:")
        for style in STYLE_PROFILES.keys():
            print(f"  - {style}")
        return

    input_file = sys.argv[1]
    target_style = sys.argv[2]
    source_style = sys.argv[3] if len(sys.argv) > 3 else 'classical'

    # Verify file exists
    if not Path(input_file).exists():
        print(f"❌ Error: File not found: {input_file}")
        return

    # Verify styles
    if target_style not in STYLE_PROFILES:
        print(f"❌ Error: Unknown target style: {target_style}")
        print(f"Available: {list(STYLE_PROFILES.keys())}")
        return

    if source_style not in STYLE_PROFILES:
        print(f"❌ Error: Unknown source style: {source_style}")
        print(f"Available: {list(STYLE_PROFILES.keys())}")
        return

    # Perform style transfer
    print(f"\n🎨 Transferring style: {source_style} → {target_style}\n")

    engine = StyleTransfer(input_file, source_style)

    output_path = engine.transfer(
        target_style,
        transform_harmony=True,
        transform_rhythm=True,
        transform_melody=True
    )

    print(f"\n✅ Style transfer complete!")
    print(f"📁 Output: {output_path}\n")

    # Show style profile details
    profile = STYLE_PROFILES[target_style]
    print(f"\n{target_style.upper()} Style Profile:")
    print(f"  Swing ratio: {profile.swing_ratio:.2f}")
    print(f"  Syncopation: {profile.syncopation_level:.1f}")
    print(f"  Ornamentation: {profile.ornamentation_density:.1f}")
    print(f"  Voice leading: {profile.voice_leading_style}")
    print(f"  Texture: {profile.texture}")


if __name__ == "__main__":
    main()
