#!/usr/bin/env python3
"""
Genre Detection - Comprehensive Usage Examples

Demonstrates all capabilities of the genre detection module:
- Basic genre classification
- Feature extraction
- Swing detection
- Chord progression extraction
- Per-track analysis
- Per-section analysis
- Integration with style fusion

Author: Agent 1 - Genre Detection & Feature Extraction
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from midi_generator.analysis.genre_detector import (
    GenreDetector,
    SwingDetector,
    ChordProgressionExtractor,
    load_genre_database
)

from midi_generator.generators.style_fusion import StyleFusion

# For creating example MIDI files
from mido import MidiFile, MidiTrack, Message, MetaMessage
import tempfile
import os


# ==============================================================================
# EXAMPLE MIDI FILE CREATION
# ==============================================================================

def create_example_jazz_file(filename: str) -> str:
    """
    Create an example jazz MIDI file for demonstration

    Features:
    - 140 BPM
    - Swing feel (0.67)
    - ii-V-I progression in C
    - Extended chords (maj7, min7, dom7)
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo (140 BPM)
    tempo = 428571
    track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track.append(Message('program_change', program=0, time=0))  # Piano

    tpb = mid.ticks_per_beat

    # ii-V-I progression with swing feel
    # Dm7 (ii)
    swing_delay = int(tpb * 0.33)
    track.append(Message('note_on', note=62, velocity=80, time=0))
    track.append(Message('note_on', note=65, velocity=75, time=0))
    track.append(Message('note_on', note=69, velocity=75, time=0))
    track.append(Message('note_on', note=72, velocity=70, time=0))
    track.append(Message('note_off', note=62, velocity=0, time=tpb))
    track.append(Message('note_off', note=65, velocity=0, time=0))
    track.append(Message('note_off', note=69, velocity=0, time=0))
    track.append(Message('note_off', note=72, velocity=0, time=0))

    # Swing eighth
    track.append(Message('note_on', note=64, velocity=70, time=swing_delay))
    track.append(Message('note_off', note=64, velocity=0, time=tpb - swing_delay))

    # G7 (V)
    track.append(Message('note_on', note=55, velocity=80, time=0))
    track.append(Message('note_on', note=59, velocity=75, time=0))
    track.append(Message('note_on', note=62, velocity=75, time=0))
    track.append(Message('note_on', note=65, velocity=70, time=0))
    track.append(Message('note_off', note=55, velocity=0, time=tpb))
    track.append(Message('note_off', note=59, velocity=0, time=0))
    track.append(Message('note_off', note=62, velocity=0, time=0))
    track.append(Message('note_off', note=65, velocity=0, time=0))

    # Swing eighth
    track.append(Message('note_on', note=57, velocity=70, time=swing_delay))
    track.append(Message('note_off', note=57, velocity=0, time=tpb - swing_delay))

    # Cmaj7 (I)
    track.append(Message('note_on', note=60, velocity=85, time=0))
    track.append(Message('note_on', note=64, velocity=80, time=0))
    track.append(Message('note_on', note=67, velocity=80, time=0))
    track.append(Message('note_on', note=71, velocity=75, time=0))
    track.append(Message('note_off', note=60, velocity=0, time=tpb * 2))
    track.append(Message('note_off', note=64, velocity=0, time=0))
    track.append(Message('note_off', note=67, velocity=0, time=0))
    track.append(Message('note_off', note=71, velocity=0, time=0))

    mid.save(filename)
    return filename


# ==============================================================================
# EXAMPLE 1: BASIC GENRE DETECTION
# ==============================================================================

def example_1_basic_genre_detection():
    """
    Example 1: Basic Genre Detection

    Demonstrates:
    - Loading a MIDI file
    - Classifying genre
    - Getting top N matches
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Genre Detection")
    print("=" * 80)

    # Create example file
    temp_file = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
    filename = temp_file.name
    temp_file.close()

    create_example_jazz_file(filename)

    try:
        # Create detector
        detector = GenreDetector(filename)

        # Classify genre (top 3 matches)
        print("\nAnalyzing MIDI file...")
        classifications = detector.classify_genre(top_n=3)

        print("\nTop genre matches:")
        for i, (genre, confidence) in enumerate(classifications, 1):
            print(f"  {i}. {genre.capitalize()}: {confidence:.1%} confidence")

        # Show top match
        top_genre, top_confidence = classifications[0]
        print(f"\n✓ Best match: {top_genre.capitalize()} ({top_confidence:.1%} confidence)")

    finally:
        os.unlink(filename)


# ==============================================================================
# EXAMPLE 2: FEATURE EXTRACTION
# ==============================================================================

def example_2_feature_extraction():
    """
    Example 2: Comprehensive Feature Extraction

    Demonstrates:
    - Extracting rhythmic features
    - Extracting harmonic features
    - Extracting melodic features
    - Extracting instrumentation features
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Comprehensive Feature Extraction")
    print("=" * 80)

    temp_file = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
    filename = temp_file.name
    temp_file.close()

    create_example_jazz_file(filename)

    try:
        detector = GenreDetector(filename)

        # Extract all feature categories
        print("\n📊 RHYTHMIC FEATURES")
        print("-" * 80)
        rhythmic = detector.extract_rhythmic_features()

        print(f"Tempo:               {rhythmic['tempo_bpm']:.1f} BPM")
        print(f"Swing factor:        {rhythmic['swing_factor']:.3f}")
        print(f"  (0.5 = straight, 0.67 = triplet swing)")
        print(f"Syncopation:         {rhythmic['syncopation']:.2f}")
        print(f"  (0 = none, 1 = heavy)")
        print(f"Rhythmic complexity: {rhythmic['rhythmic_complexity']:.2f}")
        print(f"  (0 = simple, 1 = complex)")
        print(f"Note density:        {rhythmic['note_density']:.1f} notes/beat")
        print(f"Groove type:         {rhythmic['groove_type']}")

        print("\n🎵 HARMONIC FEATURES")
        print("-" * 80)
        harmonic = detector.extract_harmonic_features()

        print(f"Key:                 {harmonic['key']}")
        print(f"Chord types:         {', '.join(harmonic['chord_types'][:5])}")
        print(f"Harmonic rhythm:     {harmonic['harmonic_rhythm']:.1f} chords/measure")
        print(f"Chromaticism:        {harmonic['chromaticism']:.2f}")
        print(f"  (0 = diatonic, 1 = chromatic)")
        print(f"Uses extensions:     {harmonic['use_extensions']}")
        print(f"  (9ths, 11ths, 13ths)")

        print("\n🎶 MELODIC FEATURES")
        print("-" * 80)
        melodic = detector.extract_melodic_features()

        dist = melodic['interval_distribution']
        print(f"Interval distribution:")
        print(f"  Stepwise:          {dist['step']:.1%}")
        print(f"  Thirds:            {dist['third']:.1%}")
        print(f"  Leaps:             {dist['leap']:.1%}")
        print(f"Contour type:        {melodic['contour_type']}")
        print(f"Ornamentation:       {melodic['ornamentation_density']:.2f}")
        print(f"  (0 = none, 1 = heavy)")
        print(f"Range:               {melodic['range_semitones']} semitones")

        print("\n🎹 INSTRUMENTATION FEATURES")
        print("-" * 80)
        inst = detector.extract_instrumentation_features()

        print(f"Instruments:         {inst['instruments']}")
        print(f"  (MIDI program numbers)")
        print(f"Texture:             {inst['texture']}")
        print(f"Register distribution:")
        reg = inst['register_distribution']
        print(f"  Low (< C3):        {reg['low']:.1%}")
        print(f"  Mid (C3-C5):       {reg['mid']:.1%}")
        print(f"  High (> C5):       {reg['high']:.1%}")

    finally:
        os.unlink(filename)


# ==============================================================================
# EXAMPLE 3: SWING DETECTION
# ==============================================================================

def example_3_swing_detection():
    """
    Example 3: Detailed Swing Detection

    Demonstrates:
    - Detecting swing factor
    - Interpreting swing values
    - Classifying groove types
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Swing Detection")
    print("=" * 80)

    temp_file = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
    filename = temp_file.name
    temp_file.close()

    create_example_jazz_file(filename)

    try:
        detector = GenreDetector(filename)

        # Extract rhythmic features (includes swing)
        features = detector.extract_rhythmic_features()

        swing = features['swing_factor']
        syncopation = features['syncopation']
        density = features['note_density']

        print(f"\n🎵 Swing Analysis")
        print("-" * 80)
        print(f"Swing factor: {swing:.3f}")

        # Interpret swing factor
        if swing >= 0.65:
            feel = "Strong triplet swing (bebop/jazz)"
        elif swing >= 0.60:
            feel = "Medium swing (shuffle)"
        elif swing >= 0.55:
            feel = "Slight shuffle"
        elif swing >= 0.53:
            feel = "Straight with minimal swing"
        else:
            feel = "Perfectly straight eighth notes"

        print(f"Feel: {feel}")

        # Classify groove
        groove = SwingDetector.classify_groove_type(swing, syncopation, density)

        print(f"\nGroove classification: {groove}")

        print(f"\nInterpretation:")
        print(f"  Syncopation: {syncopation:.2f} (0=none, 1=heavy)")
        print(f"  Note density: {density:.1f} notes/beat")

        # Show reference values
        print(f"\n📚 Reference Values:")
        print(f"  0.500 = Straight eighth notes")
        print(f"  0.550 = Slight shuffle")
        print(f"  0.600 = Shuffle")
        print(f"  0.667 = Triplet swing (jazz)")

    finally:
        os.unlink(filename)


# ==============================================================================
# EXAMPLE 4: CHORD PROGRESSION EXTRACTION
# ==============================================================================

def example_4_chord_progression():
    """
    Example 4: Chord Progression Extraction

    Demonstrates:
    - Extracting chord progression
    - Converting to chord symbols
    - Analyzing harmonic patterns
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Chord Progression Extraction")
    print("=" * 80)

    temp_file = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
    filename = temp_file.name
    temp_file.close()

    create_example_jazz_file(filename)

    try:
        detector = GenreDetector(filename)
        detector.analyze()  # Run analysis

        # Extract chord progression
        progression = ChordProgressionExtractor.extract_chord_progression(
            detector.analysis_result.chords
        )

        print(f"\n🎸 Detected Chord Progression:")
        print("-" * 80)

        if progression:
            print("  " + " → ".join(progression))

            # Analyze progression
            print(f"\nAnalysis:")
            print(f"  Total chords: {len(progression)}")
            print(f"  Unique chords: {len(set(progression))}")

            # Detect common patterns
            prog_str = " ".join(progression)

            if "Dm7" in prog_str and "G7" in prog_str and "Cmaj7" in prog_str:
                print(f"  Pattern: ii-V-I in C major (classic jazz progression)")
            elif all(c.endswith('7') for c in progression):
                print(f"  Pattern: All seventh chords (jazz/sophisticated harmony)")
        else:
            print("  No chords detected")

        # Show harmonic features
        harmonic = detector.extract_harmonic_features()
        print(f"\nHarmonic context:")
        print(f"  Key: {harmonic['key']}")
        print(f"  Harmonic rhythm: {harmonic['harmonic_rhythm']:.1f} chords/measure")

    finally:
        os.unlink(filename)


# ==============================================================================
# EXAMPLE 5: GENRE FEATURES CONVERSION
# ==============================================================================

def example_5_genre_features_conversion():
    """
    Example 5: Converting to GenreFeatures

    Demonstrates:
    - Converting extracted features to GenreFeatures
    - Integration with style_fusion.py
    - Using detected features for generation
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: GenreFeatures Conversion & Integration")
    print("=" * 80)

    temp_file = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
    filename = temp_file.name
    temp_file.close()

    create_example_jazz_file(filename)

    try:
        detector = GenreDetector(filename)

        # Convert to GenreFeatures
        print("\n📦 Converting to GenreFeatures dataclass...")
        genre_features = detector.to_genre_features()

        print(f"\nGenreFeatures object created:")
        print(f"  Name: {genre_features.name}")
        print(f"  Tempo range: {genre_features.tempo_range[0]}-{genre_features.tempo_range[1]} BPM")
        print(f"  Swing factor: {genre_features.swing_factor:.3f}")
        print(f"  Syncopation: {genre_features.syncopation:.2f}")
        print(f"  Chord types: {', '.join(genre_features.chord_types[:5])}")
        print(f"  Groove type: {genre_features.groove_type}")
        print(f"  Texture: {genre_features.texture}")

        # Integration with StyleFusion
        print(f"\n🔗 Integration with StyleFusion:")
        print("-" * 80)

        fusion = StyleFusion()

        # Can now use detected features with style fusion
        print(f"✓ GenreFeatures object is compatible with:")
        print(f"  - StyleFusion.blend_genres()")
        print(f"  - GenreBlender.blend_features()")
        print(f"  - All component generators (Agent 2+)")

        # Example: Blend detected style with another genre
        print(f"\nExample blend: Detected style + Electronic (50/50)")
        blended = fusion.blend_genres(
            detector.classify_genre(top_n=1)[0][0],  # Top detected genre
            'electronic',
            weight_a=0.5
        )
        print(f"  Result: {blended.name}")
        print(f"  Tempo: {blended.tempo_range}")
        print(f"  Swing: {blended.swing_factor:.3f}")

    finally:
        os.unlink(filename)


# ==============================================================================
# EXAMPLE 6: GENRE DATABASE
# ==============================================================================

def example_6_genre_database():
    """
    Example 6: Working with Genre Database

    Demonstrates:
    - Loading genre profiles
    - Comparing features
    - Understanding genre characteristics
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Genre Database")
    print("=" * 80)

    # Load genre database
    database = load_genre_database()

    print(f"\n📚 Available Genre Profiles: {len(database)}")
    print("-" * 80)

    for genre_name, profile in database.items():
        print(f"\n{genre_name.upper()}:")
        print(f"  Tempo: {profile.tempo_range[0]}-{profile.tempo_range[1]} BPM")
        print(f"  Swing: {profile.swing_factor:.2f}")
        print(f"  Syncopation: {profile.syncopation:.2f}")
        print(f"  Groove: {profile.groove_type}")
        print(f"  Chord types: {', '.join(profile.chord_types[:3])}")
        print(f"  Cultural origin: {profile.cultural_origin}")

    print(f"\n💡 Use these profiles for:")
    print(f"  - Genre classification (distance comparison)")
    print(f"  - Style-based generation")
    print(f"  - Genre fusion")


# ==============================================================================
# EXAMPLE 7: COMPLETE WORKFLOW
# ==============================================================================

def example_7_complete_workflow():
    """
    Example 7: Complete Analysis Workflow

    Demonstrates:
    - Full pipeline from MIDI to classification
    - Extracting all information
    - Practical usage pattern
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Complete Analysis Workflow")
    print("=" * 80)

    temp_file = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
    filename = temp_file.name
    temp_file.close()

    create_example_jazz_file(filename)

    try:
        print("\n📁 Step 1: Load MIDI file")
        print("-" * 80)
        detector = GenreDetector(filename)
        print(f"✓ Loaded: {filename}")

        print("\n🔍 Step 2: Extract Features")
        print("-" * 80)
        rhythmic = detector.extract_rhythmic_features()
        harmonic = detector.extract_harmonic_features()
        melodic = detector.extract_melodic_features()
        inst = detector.extract_instrumentation_features()
        print(f"✓ Extracted all features")

        print("\n🎯 Step 3: Classify Genre")
        print("-" * 80)
        classifications = detector.classify_genre(top_n=3)
        top_genre, confidence = classifications[0]
        print(f"✓ Top match: {top_genre} ({confidence:.1%})")

        print("\n🎸 Step 4: Extract Chord Progression")
        print("-" * 80)
        detector.analyze()
        progression = ChordProgressionExtractor.extract_chord_progression(
            detector.analysis_result.chords
        )
        if progression:
            print(f"✓ Progression: {' → '.join(progression)}")
        else:
            print(f"✓ No clear progression detected")

        print("\n📦 Step 5: Convert to GenreFeatures")
        print("-" * 80)
        genre_features = detector.to_genre_features()
        print(f"✓ Created GenreFeatures for: {genre_features.name}")

        print("\n✅ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Genre: {top_genre.capitalize()}")
        print(f"  Tempo: {rhythmic['tempo_bpm']:.0f} BPM")
        print(f"  Feel: {rhythmic['groove_type']}")
        print(f"  Key: {harmonic['key']}")
        print(f"  Texture: {inst['texture']}")

    finally:
        os.unlink(filename)


# ==============================================================================
# MAIN - RUN ALL EXAMPLES
# ==============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print(" GENRE DETECTION MODULE - COMPREHENSIVE EXAMPLES")
    print("=" * 80)
    print("\nThis script demonstrates all capabilities of the genre detection module.")
    print("Creating temporary MIDI files for demonstration...\n")

    try:
        # Run all examples
        example_1_basic_genre_detection()
        example_2_feature_extraction()
        example_3_swing_detection()
        example_4_chord_progression()
        example_5_genre_features_conversion()
        example_6_genre_database()
        example_7_complete_workflow()

        print("\n" + "=" * 80)
        print(" ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print("\n💡 Next Steps:")
        print("  1. Try analyzing your own MIDI files")
        print("  2. Integrate with Agent 2-10 modules (coming soon)")
        print("  3. Use detected features for style-based generation")
        print("  4. Experiment with genre fusion")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
