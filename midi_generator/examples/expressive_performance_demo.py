#!/usr/bin/env python3
"""
Expressive Performance Module - Comprehensive Demo

This demo showcases practical applications of the ExpressivePerformance
module with integration to the existing harmonymodule components.

Demonstrations:
1. Classical piano phrase with rubato and dynamics
2. Jazz combo with swing and microtiming
3. Romantic orchestral passage with dramatic expression
4. Pop melody with tight timing
5. Electronic music with creative velocity shaping
6. Film scoring cue with dynamic curves

Author: Agent 2 - Expressive Performance Modeling
Date: 2025
"""

import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../advanced_modules/'))

from expressive_performance import (
    Note, ExpressivePerformance, DynamicsEngine, VelocityHumanizer,
    MicrotimingEngine, RubatoEngine, ArticulationEngine, StyleEngine,
    DynamicsCurveType, ArticulationType, ExpressionStyle
)


# ============================================================================
# DEMO 1: Classical Piano Phrase with Rubato
# ============================================================================

def demo_classical_piano():
    """
    Classical piano performance:
    - Subtle velocity variations
    - Romantic rubato (slow-fast-slow)
    - Legato articulation
    - Crescendo into climax
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Classical Piano Phrase (C Major Scale with Expression)")
    print("=" * 70)

    # Create a simple C major scale phrase (2 octaves up and down)
    scale_up = [60, 62, 64, 65, 67, 69, 71, 72]  # C to C
    scale_down = [72, 71, 69, 67, 65, 64, 62, 60]  # C back to C
    scale_notes = scale_up + scale_down

    notes = []
    for i, pitch in enumerate(scale_notes):
        note = Note(
            pitch=pitch,
            start_time=i * 480,  # Quarter notes at 480 TPQN
            duration=400,
            velocity=70
        )
        notes.append(note)

    print(f"\nOriginal phrase: {len(notes)} notes (C major scale up and down)")
    print(f"Original velocities: {[n.velocity for n in notes[:8]]}")

    # Apply classical expression
    perf = ExpressivePerformance(ticks_per_beat=480)

    # 1. Crescendo to climax (note 7, high C)
    notes_half1 = notes[:8]
    notes_half2 = notes[8:]

    notes_half1 = DynamicsEngine.apply_dynamics_curve(
        notes_half1, "crescendo", start_vel=60, end_vel=100,
        curve_shape=DynamicsCurveType.EXPONENTIAL
    )

    # 2. Diminuendo back down
    notes_half2 = DynamicsEngine.apply_dynamics_curve(
        notes_half2, "diminuendo", start_vel=100, end_vel=60,
        curve_shape=DynamicsCurveType.EXPONENTIAL
    )

    notes = notes_half1 + notes_half2

    # 3. Apply rubato (romantic style)
    notes = RubatoEngine.apply_rubato(notes, intensity=0.35, style="romantic")

    # 4. Legato articulation
    notes = ArticulationEngine.render_articulation(notes, ArticulationType.LEGATO, overlap=0.15)

    # 5. Subtle velocity humanization
    notes = VelocityHumanizer.humanize_velocities(notes, variance=5, preserve_accents=True)

    print(f"\nExpressive phrase:")
    print(f"Velocities: {[n.velocity for n in notes[:8]]}")
    print(f"Timings (first 8): {[f'{n.start_time:.1f}' for n in notes[:8]]}")
    print(f"Durations (first 8): {[f'{n.duration:.1f}' for n in notes[:8]]}")
    print("\n✓ Classical piano phrase ready for export to MIDI")


# ============================================================================
# DEMO 2: Jazz Combo with Swing
# ============================================================================

def demo_jazz_swing():
    """
    Jazz combo performance:
    - 60% swing on 16th notes
    - Microtiming for groove feel
    - Moderate dynamics humanization
    - Syncopated accents
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Jazz Walking Bass Line with Swing")
    print("=" * 70)

    # Create walking bass line (quarter notes walking up/down)
    bass_pattern = [41, 43, 45, 48, 50, 48, 46, 43, 41, 39, 36, 39]  # F walking bass

    notes = []
    for i, pitch in enumerate(bass_pattern):
        note = Note(
            pitch=pitch,
            start_time=i * 480,  # Quarter notes
            duration=400,
            velocity=75
        )
        notes.append(note)

    print(f"\nOriginal bass line: {len(notes)} quarter notes")
    print(f"Original timings: {[n.start_time for n in notes[:6]]}")

    # Apply jazz expression
    perf = ExpressivePerformance(ticks_per_beat=480)

    # 1. Medium swing (60%)
    notes = MicrotimingEngine.apply_swing(notes, swing_percent=60)

    # 2. Microtiming for groove
    notes = MicrotimingEngine.apply_microtiming(notes, variance_ms=8, groove_type="jazz", seed=42)

    # 3. Velocity humanization
    notes = VelocityHumanizer.humanize_velocities(notes, variance=12, seed=42)

    # 4. Accent downbeats (beats 1 and 3)
    accent_positions = [0, 2, 4, 6, 8, 10]
    notes = DynamicsEngine.add_dynamic_accents(notes, accent_positions, accent_amount=15)

    print(f"\nSwung bass line:")
    print(f"Timings (first 6): {[f'{n.start_time:.1f}' for n in notes[:6]]}")
    print(f"Velocities: {[n.velocity for n in notes]}")
    print("\n✓ Jazz walking bass with authentic swing feel")


# ============================================================================
# DEMO 3: Romantic Orchestral Passage
# ============================================================================

def demo_romantic_orchestral():
    """
    Romantic orchestral climax:
    - Heavy rubato
    - Dramatic crescendo
    - Sforzando accents
    - Expressive timing deviations
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Romantic Orchestral Climax (Strings)")
    print("=" * 70)

    # Create ascending melodic phrase (whole notes)
    melody = [60, 64, 67, 71, 72, 76, 79, 84]  # Rising dramatic melody

    notes = []
    for i, pitch in enumerate(melody):
        note = Note(
            pitch=pitch,
            start_time=i * 1920,  # Whole notes (4 beats * 480)
            duration=1800,
            velocity=70
        )
        notes.append(note)

    print(f"\nOriginal melodic phrase: {len(notes)} whole notes")
    print(f"Original velocities: {[n.velocity for n in notes]}")

    # Apply romantic expression
    perf = ExpressivePerformance(ticks_per_beat=480)

    # 1. Heavy rubato (slow start, accelerate, slow at climax)
    notes = RubatoEngine.apply_rubato(notes, intensity=0.6, style="romantic")

    # 2. Dramatic crescendo
    notes = DynamicsEngine.apply_dynamics_curve(
        notes, "crescendo", start_vel=45, end_vel=115,
        curve_shape=DynamicsCurveType.EXPONENTIAL
    )

    # 3. Sforzando on climax (last note)
    notes = DynamicsEngine.add_dynamic_accents(notes, [7], accent_type="sforzando")

    # 4. Strong velocity humanization
    notes = VelocityHumanizer.humanize_velocities(notes, variance=15, preserve_accents=True)

    # 5. Tenuto articulation
    notes = ArticulationEngine.render_articulation(notes, ArticulationType.TENUTO)

    print(f"\nExpressive orchestral phrase:")
    print(f"Velocities: {[n.velocity for n in notes]}")
    print(f"Timings (first 4): {[f'{n.start_time:.1f}' for n in notes[:4]]}")
    print("\n✓ Romantic orchestral climax with dramatic expression")


# ============================================================================
# DEMO 4: Pop Melody (Tight Timing)
# ============================================================================

def demo_pop_melody():
    """
    Modern pop production:
    - Tight timing (minimal microtiming)
    - Moderate dynamics
    - Consistent velocity
    - Staccato sections
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Pop Melody Hook (Synth Lead)")
    print("=" * 70)

    # Create catchy pop hook (16th note pattern)
    melody = [67, 69, 72, 71, 69, 67, 64, 62] * 2  # Repeated hook

    notes = []
    for i, pitch in enumerate(melody):
        note = Note(
            pitch=pitch,
            start_time=i * 120,  # 16th notes
            duration=100,
            velocity=85
        )
        notes.append(note)

    print(f"\nOriginal pop hook: {len(notes)} 16th notes")
    print(f"Original velocities: all {notes[0].velocity}")

    # Apply pop expression
    perf = ExpressivePerformance(ticks_per_beat=480)

    # 1. Tight timing (minimal microtiming)
    notes = MicrotimingEngine.apply_microtiming(notes, variance_ms=3, groove_type="straight")

    # 2. Moderate velocity humanization
    notes = VelocityHumanizer.humanize_velocities(notes, variance=8, seed=42)

    # 3. Staccato on certain notes (every 4th note)
    staccato_notes = notes[3::4]
    for note in staccato_notes:
        note.duration *= 0.5

    # 4. Slight crescendo in second half
    notes_half2 = notes[8:]
    notes_half2 = DynamicsEngine.apply_dynamics_curve(
        notes_half2, "crescendo", start_vel=85, end_vel=95,
        curve_shape=DynamicsCurveType.LINEAR
    )

    print(f"\nExpressive pop hook:")
    print(f"Velocities (first 8): {[n.velocity for n in notes[:8]]}")
    print(f"Durations (first 8): {[f'{n.duration:.1f}' for n in notes[:8]]}")
    print("\n✓ Pop melody with tight modern production feel")


# ============================================================================
# DEMO 5: Electronic Music - Creative Velocity Shaping
# ============================================================================

def demo_electronic_creative():
    """
    Electronic/EDM production:
    - Creative velocity contours
    - Minimal timing variation
    - Patterned dynamics
    - Marcato accents on beats
    """
    print("\n" + "=" * 70)
    print("DEMO 5: Electronic Arpeggio Pattern")
    print("=" * 70)

    # Create ascending arpeggio (16 notes)
    arpeggio = [48, 52, 55, 60, 52, 55, 60, 64, 55, 60, 64, 67, 60, 64, 67, 72]

    notes = []
    for i, pitch in enumerate(arpeggio):
        note = Note(
            pitch=pitch,
            start_time=i * 120,  # 16th notes
            duration=110,
            velocity=80
        )
        notes.append(note)

    print(f"\nOriginal arpeggio: {len(notes)} notes")

    # Apply electronic expression
    perf = ExpressivePerformance(ticks_per_beat=480)

    # 1. Very tight timing (almost quantized)
    notes = MicrotimingEngine.apply_microtiming(notes, variance_ms=2, groove_type="straight")

    # 2. Creative velocity contour (pulsing pattern)
    contour = [0.3, 0.6, 0.4, 1.0, 0.3, 0.6, 0.4, 0.9,
               0.3, 0.6, 0.5, 0.95, 0.4, 0.7, 0.6, 1.0]
    notes = VelocityHumanizer.add_velocity_contour(notes, contour, scale=35)

    # 3. Marcato on every 4th note (beat markers)
    beat_positions = [0, 4, 8, 12]
    notes = DynamicsEngine.add_dynamic_accents(notes, beat_positions, accent_type="marcato")

    print(f"\nExpressive arpeggio:")
    print(f"Velocities: {[n.velocity for n in notes]}")
    print(f"Timings (first 8): {[f'{n.start_time:.1f}' for n in notes[:8]]}")
    print("\n✓ Electronic arpeggio with creative velocity pulsing")


# ============================================================================
# DEMO 6: Film Scoring - Tension Build
# ============================================================================

def demo_film_scoring_tension():
    """
    Film scoring cue (building tension):
    - Accelerando (building tempo)
    - Crescendo (building dynamics)
    - Increasing microtiming chaos
    - Staccato to marcato transition
    """
    print("\n" + "=" * 70)
    print("DEMO 6: Film Scoring - Tension Build Cue")
    print("=" * 70)

    # Create ostinato pattern (repeated tense motif)
    ostinato = [48, 50, 51, 50] * 4  # Repeated half-step tension

    notes = []
    for i, pitch in enumerate(ostinato):
        note = Note(
            pitch=pitch,
            start_time=i * 240,  # 8th notes
            duration=200,
            velocity=60
        )
        notes.append(note)

    print(f"\nOriginal ostinato: {len(notes)} 8th notes")

    # Apply film scoring expression
    perf = ExpressivePerformance(ticks_per_beat=480)

    # 1. Accelerando (building tension through tempo)
    notes = RubatoEngine.apply_accelerando(
        notes, start_tempo_ratio=1.0, end_tempo_ratio=1.6,
        curve_type=DynamicsCurveType.EXPONENTIAL
    )

    # 2. Crescendo (building dynamics)
    notes = DynamicsEngine.apply_dynamics_curve(
        notes, "crescendo", start_vel=45, end_vel=110,
        curve_shape=DynamicsCurveType.EXPONENTIAL
    )

    # 3. Increasing microtiming chaos
    for i, note in enumerate(notes):
        # More chaos as pattern progresses
        chaos_factor = (i / len(notes)) * 20  # 0 to 20ms variance
        if i > 0:
            import random
            random.seed(42 + i)
            offset = random.gauss(0, chaos_factor)
            note.start_time += offset * 480 / 500  # Convert ms to ticks

    # 4. Transition from staccato to marcato
    for i, note in enumerate(notes):
        progress = i / len(notes)
        if progress < 0.5:
            # Start staccato
            note.duration *= 0.6
        else:
            # End marcato (short but accented)
            note.duration *= 0.7

    print(f"\nExpressive tension cue:")
    print(f"Velocities (start): {[n.velocity for n in notes[:4]]}")
    print(f"Velocities (end): {[n.velocity for n in notes[-4:]]}")
    print(f"Intervals (start): {[f'{notes[i+1].start_time - notes[i].start_time:.1f}' for i in range(3)]}")
    print(f"Intervals (end): {[f'{notes[i+1].start_time - notes[i].start_time:.1f}' for i in range(-4, -1)]}")
    print("\n✓ Film scoring tension cue with accelerando and crescendo")


# ============================================================================
# DEMO 7: J Dilla Style Hip-Hop Beat
# ============================================================================

def demo_j_dilla_hiphop():
    """
    J Dilla style hip-hop production:
    - Drunk drumming swing (variable)
    - Laid-back groove
    - Velocity variations
    - Ghost notes
    """
    print("\n" + "=" * 70)
    print("DEMO 7: J Dilla Style Hip-Hop Beat")
    print("=" * 70)

    # Create simple kick-snare pattern (16 16th notes)
    # 1=kick, 0=snare, -1=ghost/hi-hat
    pattern = [1, -1, -1, -1, 0, -1, -1, -1, 1, -1, 0, -1, 1, -1, -1, -1]
    pitches = {1: 36, 0: 38, -1: 42}  # Kick, snare, hi-hat

    notes = []
    for i, drum_type in enumerate(pattern):
        velocity = 85 if drum_type in [1, 0] else 55  # Ghost notes quieter
        note = Note(
            pitch=pitches[drum_type],
            start_time=i * 120,  # 16th notes
            duration=100,
            velocity=velocity
        )
        notes.append(note)

    print(f"\nOriginal beat: {len(notes)} 16th notes")
    print(f"Pattern: K=kick, S=snare, H=hihat")

    # Apply J Dilla expression
    perf = ExpressivePerformance(ticks_per_beat=480)

    # 1. J Dilla swing (drunk drumming)
    notes = MicrotimingEngine.create_j_dilla_swing(notes, drunk_factor=0.75)

    # 2. Additional microtiming for organic feel
    notes = MicrotimingEngine.apply_microtiming(notes, variance_ms=12, groove_type="funk", seed=42)

    # 3. Velocity humanization
    notes = VelocityHumanizer.humanize_velocities(notes, variance=10, preserve_accents=True, seed=42)

    print(f"\nJ Dilla style beat:")
    print(f"Timings (first 8): {[f'{n.start_time:.1f}' for n in notes[:8]]}")
    print(f"Velocities (first 8): {[n.velocity for n in notes[:8]]}")
    print("\n✓ J Dilla style hip-hop beat with drunk drumming feel")


# ============================================================================
# MAIN DEMO RUNNER
# ============================================================================

def run_all_demos():
    """Run all demonstration examples"""
    print("\n" + "=" * 70)
    print("EXPRESSIVE PERFORMANCE MODULE - COMPREHENSIVE DEMO SUITE")
    print("=" * 70)
    print("\nDemonstrating advanced MIDI expression techniques:")
    print("- Dynamics curves (crescendo, diminuendo)")
    print("- Velocity humanization (Gaussian variation)")
    print("- Microtiming and swing (Roger Linn algorithm)")
    print("- Rubato and tempo curves (accelerando, ritardando)")
    print("- Articulation rendering (staccato, legato, marcato)")
    print("- Style-specific expression (classical, jazz, pop, etc.)")

    # Run all demos
    demo_classical_piano()
    demo_jazz_swing()
    demo_romantic_orchestral()
    demo_pop_melody()
    demo_electronic_creative()
    demo_film_scoring_tension()
    demo_j_dilla_hiphop()

    # Summary
    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY! ✓")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Export these expressive phrases to MIDI files")
    print("2. Integrate with existing melody/harmony generators")
    print("3. Use in film scoring, game audio, or music production")
    print("4. Combine with other advanced_modules for complete compositions")
    print("\nResearch foundation:")
    print("  - Nature Scientific Reports 2025 (Transformer models)")
    print("  - MAESTRO Dataset (200 hours piano performance)")
    print("  - GigaMIDI (1.4M files, micro-timing analysis)")
    print("  - Roger Linn MPC swing algorithm (50-75%)")
    print("  - PMC participatory discrepancies (±50ms groove)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_demos()
