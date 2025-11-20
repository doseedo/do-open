#!/usr/bin/env python3
"""
Example 3: World Music Scales and Microtonal Systems
====================================================

Demonstrates microtonal and non-Western musical systems:
- Arabic maqam system (24-TET quarter tones)
- Indian raga system
- Turkish makam system (53-TET)
- Persian dastgah system
- Just intonation

Musical context: These systems represent thousands of years of
musical tradition outside Western equal temperament.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.microtonality import ArabicMaqam
from generators.advanced_harmony_generator import AdvancedHarmonyGenerator


def main():
    print("=" * 70)
    print("WORLD MUSIC SCALES & MICROTONAL SYSTEMS")
    print("=" * 70)

    gen = AdvancedHarmonyGenerator(root=0, octave=4)

    # ARABIC MAQAM SYSTEM
    print("\n" + "=" * 70)
    print("ARABIC MAQAM SYSTEM (24-TET Quarter Tones)")
    print("=" * 70)

    print("\n1. MAQAM RAST (Joyful, major-like)")
    print("-" * 70)
    rast = gen.generate_arabic_maqam(ArabicMaqam.RAST)
    print(f"Maqam: {rast['maqam']}")
    print(f"Description: {rast['description']}")
    print(f"Structure: {rast['lower_jins']} (lower) + {rast['upper_jins']} (upper)")
    print(f"Intervals (cents): {[f'{c:.0f}' for c in rast['intervals_cents']]}")
    print(f"MIDI notes with bends: {len(rast['midi_with_bends'])} notes")

    print("\n2. MAQAM BAYATI (Folk, minor-like)")
    print("-" * 70)
    bayati = gen.generate_arabic_maqam(ArabicMaqam.BAYATI)
    print(f"Maqam: {bayati['maqam']}")
    print(f"Description: {bayati['description']}")
    print(f"Structure: {bayati['lower_jins']} + {bayati['upper_jins']}")
    print(f"Characteristic: 3-quarter tone intervals")

    print("\n3. MAQAM HIJAZ (Dramatic, augmented 2nd)")
    print("-" * 70)
    hijaz = gen.generate_arabic_maqam(ArabicMaqam.HIJAZ)
    print(f"Maqam: {hijaz['maqam']}")
    print(f"Description: {hijaz['description']}")
    print(f"Intervals (cents): {[f'{c:.0f}' for c in hijaz['intervals_cents']]}")
    print(f"Characteristic: Large augmented 2nd (300 cents)")
    print(f"Used in: Middle Eastern pop, film music")

    print("\n4. MAQAM SABA (Melancholic, complex)")
    print("-" * 70)
    saba = gen.generate_arabic_maqam(ArabicMaqam.SABA)
    print(f"Maqam: {saba['maqam']}")
    print(f"Description: {saba['description']}")
    print(f"Characteristic: Most complex, very melancholic")

    # INDIAN RAGA SYSTEM
    print("\n" + "=" * 70)
    print("INDIAN RAGA SYSTEM")
    print("=" * 70)

    print("\n5. RAGA BHAIRAV (Morning, devotional)")
    print("-" * 70)
    bhairav = gen.generate_indian_raga("Bhairav", ascending=True)
    print(f"Raga: {bhairav['raga']}")
    print(f"Time: {bhairav['time']} raga")
    print(f"Rasa (mood): {bhairav['rasa']}")
    print(f"Vadi (main note): Scale degree {bhairav['vadi']}")
    print(f"Direction: {bhairav['direction']}")
    print(f"Intervals: {[f'{c:.0f}' for c in bhairav['intervals_cents']]}")
    print("\nNote: In Indian classical, timing and mood are essential")

    print("\n6. RAGA YAMAN (Evening, romantic)")
    print("-" * 70)
    yaman = gen.generate_indian_raga("Yaman", ascending=True)
    print(f"Raga: {yaman['raga']}")
    print(f"Time: {yaman['time']} raga")
    print(f"Rasa: {yaman['rasa']}")
    print(f"Vadi: {yaman['vadi']}, Samvadi: {IndianRaga.get_raga('Yaman')['samvadi']}")
    print(f"Characteristic: Tivra Ma (sharp 4th), similar to Lydian")

    print("\n7. RAGA KAFI (Night, peaceful)")
    print("-" * 70)
    kafi = gen.generate_indian_raga("Kafi", ascending=True)
    print(f"Raga: {kafi['raga']}")
    print(f"Time: {kafi['time']} raga")
    print(f"Rasa: {kafi['rasa']}")
    print(f"Similar to: Natural minor (Aeolian)")

    # TURKISH MAKAM SYSTEM
    print("\n" + "=" * 70)
    print("TURKISH MAKAM SYSTEM (53-TET)")
    print("=" * 70)

    print("\n8. MAKAM HICAZ (Phrygian dominant-like)")
    print("-" * 70)
    hicaz = gen.generate_turkish_makam("Hicaz")
    print(f"Makam: {hicaz['makam']}")
    print(f"System: {hicaz['system']} (Holdrian comma)")
    print(f"Description: {hicaz['description']}")
    print(f"Intervals (cents): {[f'{c:.1f}' for c in hicaz['intervals_cents']]}")

    print("\n9. MAKAM RAST (Major-like)")
    print("-" * 70)
    rast_tr = gen.generate_turkish_makam("Rast")
    print(f"Makam: {rast_tr['makam']}")
    print(f"Description: {rast_tr['description']}")
    print(f"Note: Similar to Arabic Rast but uses 53-TET")

    print("\n10. MAKAM KÜRDI (Minor with quarter tones)")
    print("-" * 70)
    kurdi = gen.generate_turkish_makam("Kürdi")
    print(f"Makam: {kurdi['makam']}")
    print(f"Description: {kurdi['description']}")

    # PERSIAN DASTGAH SYSTEM
    print("\n" + "=" * 70)
    print("PERSIAN DASTGAH SYSTEM")
    print("=" * 70)

    print("\n11. DASTGAH SHUR (Most common, melancholic)")
    print("-" * 70)
    shur = gen.generate_persian_dastgah("Shur")
    print(f"Dastgah: {shur['dastgah']}")
    print(f"Description: {shur['description']}")
    print(f"Gooshe (melodic motifs): {shur['gooshe_count']}")
    print(f"Intervals (cents): {[f'{c:.0f}' for c in shur['intervals_cents']]}")

    print("\n12. DASTGAH MAHUR (Joyful)")
    print("-" * 70)
    mahur = gen.generate_persian_dastgah("Mahur")
    print(f"Dastgah: {mahur['dastgah']}")
    print(f"Description: {PersianDastgah.DASTGAH_HA['Mahur']['description']}")
    print(f"Similar to: Western major scale")

    # MICROTONAL SYSTEMS
    print("\n" + "=" * 70)
    print("MICROTONAL EQUAL TEMPERAMENTS")
    print("=" * 70)

    print("\n13. 24-TET (Quarter-tone system)")
    print("-" * 70)
    tet24 = gen.generate_microtonal_scale("24-TET")
    print(f"System: {tet24['system']}")
    print(f"Step size: 50 cents")
    print(f"Uses: Arabic music, contemporary classical")
    print(f"Composers: Alois Hába, Julián Carrillo")

    print("\n14. 19-TET (1/3-comma meantone)")
    print("-" * 70)
    tet19 = gen.generate_microtonal_scale("19-TET")
    print(f"System: {tet19['system']}")
    print(f"Step size: ~63.16 cents")
    print(f"Characteristic: Better thirds than 12-TET")

    print("\n15. 31-TET (1/4-comma meantone)")
    print("-" * 70)
    tet31 = gen.generate_microtonal_scale("31-TET")
    print(f"System: {tet31['system']}")
    print(f"Step size: ~38.71 cents")
    print(f"Characteristic: Very close to just intonation")

    print("\n16. 53-TET (Pythagorean comma)")
    print("-" * 70)
    tet53 = gen.generate_microtonal_scale("53-TET")
    print(f"System: {tet53['system']}")
    print(f"Step size: ~22.64 cents")
    print(f"Uses: Turkish makam music")

    # JUST INTONATION
    print("\n" + "=" * 70)
    print("JUST INTONATION (Pure Frequency Ratios)")
    print("=" * 70)

    print("\n17. JUST INTONATION MAJOR SCALE")
    print("-" * 70)
    just_major = gen.generate_just_intonation_scale("major")
    print(f"System: {just_major['style']}")
    print(f"Scale: {just_major['scale_name']}")
    print(f"Intervals (cents):")
    for i, cents in enumerate(just_major['intervals_cents'], 1):
        print(f"  Degree {i}: {cents:.2f} cents")
    print("\nNote: Differs from 12-TET in thirds and sixths")

    print("\n18. HARMONIC SERIES")
    print("-" * 70)
    harmonic = gen.generate_just_intonation_scale("harmonic_series")
    print(f"Scale: {harmonic['scale_name']}")
    print(f"First 8 harmonics (cents):")
    for i, cents in enumerate(harmonic['intervals_cents'][:8], 1):
        print(f"  Harmonic {i}: {cents:.2f} cents")
    print("\nUses: Spectral music, natural trumpet/horn tones")

    print("\n" + "=" * 70)
    print("IMPLEMENTATION NOTES:")
    print("- MIDI pitch bend used for microtones")
    print("- Typical bend range: ±2 semitones")
    print("- Quarter tones require 2048 bend units")
    print("- Microtonal MIDI may need multiple channels")
    print("- Some DAWs support MPE for better microtonal control")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("CULTURAL CONTEXT:")
    print("- Arabic maqam: 1000+ years of tradition")
    print("- Indian raga: Strict time and mood associations")
    print("- Turkish makam: Synthesis of Arabic and Byzantine")
    print("- Persian dastgah: Complex modal system with improvisation")
    print("- Each system has performance practices beyond just notes")
    print("=" * 70)


# Import for example display
from core.microtonality import IndianRaga, PersianDastgah


if __name__ == "__main__":
    main()
