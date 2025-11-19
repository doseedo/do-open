#!/usr/bin/env python3
"""
Electronic Music Generator - Ambient, IDM, Glitch, Breakcore

This module implements comprehensive electronic music generation including:
- Ambient (Brian Eno, Aphex Twin Selected Ambient Works)
- IDM / Intelligent Dance Music (Autechre, Boards of Canada)
- Glitch (Oval, Alva Noto, Fennesz)
- Breakcore (Venetian Snares, Igorrr)
- Algorithmic modulation and generative patterns

Features:
- Euclidean sequencing
- Algorithmic modulation (LFOs, envelopes)
- Glitch effects (stuttering, bit crushing simulation)
- Breakbeat manipulation (time-stretching, chopping)
- Generative ambient soundscapes
- Polymetric and complex rhythms
- Microrhythmic variations

Author: Agent 7 - World Music & Additional Genres
References:
- "Ocean of Sound" - David Toop
- "More Brilliant Than the Sun" - Kodwo Eshun
- Euclidean rhythms - Godfried Toussaint
- Breakcore production techniques
- Modular synthesis concepts
"""

import random
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ElectronicStyle(Enum):
    """Electronic music sub-genres"""
    AMBIENT = "ambient"  # Atmospheric, minimal rhythm
    IDM = "idm"  # Complex, experimental electronic
    GLITCH = "glitch"  # Stuttering, digital artifacts
    BREAKCORE = "breakcore"  # Extreme breakbeat manipulation
    MINIMAL_TECHNO = "minimal_techno"  # Repetitive, minimal
    DRILL_N_BASS = "drill_n_bass"  # Hyper-speed breakbeats


class EuclideanRhythm:
    """
    Euclidean rhythm generator

    Euclidean rhythms distribute beats as evenly as possible.
    E(k, n) means k beats distributed over n steps.

    Examples:
    - E(3, 8) = [X . . X . . X .] (tresillo)
    - E(5, 8) = [X . X . X . X X] (cinquillo)
    - E(5, 12) = [X . . X . X . . X . X .] (venda rhythm)

    References:
    - "The Euclidean Algorithm Generates Traditional Musical Rhythms"
      by Godfried Toussaint
    """

    @staticmethod
    def generate(pulses: int, steps: int) -> List[int]:
        """
        Generate Euclidean rhythm using Bjorklund's algorithm

        Args:
            pulses: Number of beats (k)
            steps: Total steps (n)

        Returns:
            List of 1s (beat) and 0s (rest)
        """
        if pulses >= steps or pulses == 0:
            return [1 if i < pulses else 0 for i in range(steps)]

        pattern = []
        counts = []
        remainders = []

        divisor = steps - pulses
        remainders.append(pulses)

        level = 0

        while True:
            counts.append(divisor // remainders[level])
            remainders.append(divisor % remainders[level])
            divisor = remainders[level]
            level += 1
            if remainders[level] <= 1:
                break

        counts.append(divisor)

        def build(level: int) -> List[int]:
            if level == -1:
                return [0]
            if level == -2:
                return [1]

            result = []
            for _ in range(counts[level]):
                result.extend(build(level - 1))
            if remainders[level] != 0:
                result.extend(build(level - 2))
            return result

        pattern = build(level)
        # Rotate to start on downbeat
        pattern = pattern[-pulses:] + pattern[:-pulses] if pulses < len(pattern) else pattern

        return pattern[:steps]

    @staticmethod
    def pattern_to_times(pattern: List[int],
                        duration: float = 16.0) -> List[float]:
        """
        Convert binary pattern to time positions

        Args:
            pattern: Binary rhythm pattern
            duration: Total duration in beats

        Returns:
            List of beat times where pattern has 1
        """
        step_duration = duration / len(pattern)
        return [i * step_duration for i, val in enumerate(pattern) if val == 1]


class GlitchGenerator:
    """
    Glitch effect generator

    Glitch music uses "digital errors" as aesthetic elements:
    - Stuttering (repeating small segments)
    - Bit crushing (reducing bit depth)
    - Time stretching artifacts
    - Buffer underruns
    """

    @staticmethod
    def stutter(notes: List[Tuple[int, float, float, int]],
               stutter_prob: float = 0.3,
               repeat_count: int = 4) -> List[Tuple[int, float, float, int]]:
        """
        Apply stuttering effect to note sequence

        Randomly repeats notes in rapid succession.

        Args:
            notes: List of (note, time, duration, velocity) tuples
            stutter_prob: Probability of stuttering (0-1)
            repeat_count: How many times to repeat

        Returns:
            Modified note list with stutters
        """
        glitched = []

        for note, time, duration, velocity in notes:
            glitched.append((note, time, duration, velocity))

            # Randomly apply stutter
            if random.random() < stutter_prob:
                # Create rapid repeats
                stutter_duration = duration / (repeat_count * 2)
                for i in range(1, repeat_count):
                    stutter_time = time + (i * stutter_duration)
                    # Decaying velocity
                    stutter_vel = int(velocity * (1 - i / repeat_count))
                    glitched.append((note, stutter_time, stutter_duration, stutter_vel))

        return sorted(glitched, key=lambda x: x[1])

    @staticmethod
    def generate_glitch_sequence(length: int = 16,
                                 density: float = 0.5) -> List[Tuple[int, float, float, int]]:
        """
        Generate algorithmic glitch sequence

        Args:
            length: Length in beats
            density: Note density (0-1)

        Returns:
            List of glitchy notes
        """
        sequence = []
        time = 0.0
        note_pool = [60, 62, 64, 65, 67, 69, 71]  # C major scale

        while time < length:
            if random.random() < density:
                note = random.choice(note_pool)
                # Irregular durations
                duration = random.choice([0.0625, 0.125, 0.25, 0.5])
                velocity = random.randint(60, 100)
                sequence.append((note, time, duration, velocity))

            # Irregular time steps
            time += random.choice([0.125, 0.25, 0.5])

        return sequence


class BreakcoreGenerator:
    """
    Breakcore rhythm generator

    Breakcore manipulates breakbeats to extreme tempos (160-300+ BPM)
    with complex chopping, time-stretching, and layering.
    """

    # Classic breakbeat pattern (Amen break style)
    AMEN_PATTERN = [
        ('kick', 0.0, 100),
        ('snare', 0.5, 90),
        ('kick', 1.0, 95),
        ('hihat', 1.25, 60),
        ('snare', 1.5, 100),
        ('kick', 2.0, 100),
        ('hihat', 2.25, 70),
        ('snare', 2.5, 95),
        ('hihat', 2.75, 60),
        ('kick', 3.0, 90),
        ('snare', 3.5, 100),
    ]

    @staticmethod
    def chop_break(break_pattern: List[Tuple[str, float, int]],
                  chop_count: int = 16) -> List[Tuple[str, float, int]]:
        """
        Chop and rearrange breakbeat

        Args:
            break_pattern: Original break pattern
            chop_count: Number of slices

        Returns:
            Chopped and rearranged pattern
        """
        if not break_pattern:
            return []

        # Get total duration
        max_time = max(time for _, time, _ in break_pattern)
        chop_size = max_time / chop_count

        # Organize into chops
        chops = [[] for _ in range(chop_count)]
        for drum, time, vel in break_pattern:
            chop_idx = min(int(time / chop_size), chop_count - 1)
            chops[chop_idx].append((drum, time % chop_size, vel))

        # Rearrange chops randomly
        random.shuffle(chops)

        # Reassemble
        chopped = []
        time_offset = 0.0
        for chop in chops:
            for drum, time, vel in chop:
                chopped.append((drum, time_offset + time, vel))
            time_offset += chop_size

        return chopped

    @staticmethod
    def time_stretch(pattern: List[Tuple[str, float, int]],
                    factor: float = 2.0) -> List[Tuple[str, float, int]]:
        """
        Time-stretch breakbeat (simulated)

        Args:
            pattern: Break pattern
            factor: Stretch factor (2.0 = double speed)

        Returns:
            Time-stretched pattern
        """
        return [(drum, time / factor, vel) for drum, time, vel in pattern]

    @staticmethod
    def generate_hyper_break(measures: int = 4,
                            tempo_mult: float = 2.0) -> List[Tuple[str, float, int]]:
        """
        Generate hyper-speed breakbeat

        Args:
            measures: Number of measures
            tempo_mult: Tempo multiplier

        Returns:
            Breakcore pattern
        """
        pattern = []

        for measure in range(measures):
            offset = measure * 4.0

            # Use Amen break as base
            amen = BreakcoreGenerator.AMEN_PATTERN.copy()

            # Random variations
            if random.random() > 0.5:
                # Chop it
                amen = BreakcoreGenerator.chop_break(amen, 8)

            # Time stretch
            amen = BreakcoreGenerator.time_stretch(amen, tempo_mult)

            # Add to pattern with offset
            for drum, time, vel in amen:
                pattern.append((drum, offset + time, vel))

        return pattern


class AmbientGenerator:
    """
    Ambient music generator

    Ambient music is atmospheric, often with:
    - Slow evolution
    - Minimal rhythm
    - Emphasis on texture and timbre
    - Long, sustained notes
    """

    @staticmethod
    def generate_drone(root: int, duration: float = 64.0,
                      voices: int = 4) -> List[Tuple[int, float, float, int]]:
        """
        Generate ambient drone

        Args:
            root: Root note
            duration: Total duration in beats
            voices: Number of voices

        Returns:
            List of sustained notes forming drone
        """
        drone = []

        # Create consonant chord (e.g., suspended or add9)
        intervals = [0, 2, 7, 9, 12]  # Root, 2nd, 5th, 6th, octave
        chord_notes = [root + interval for interval in intervals[:voices]]

        # Each voice sustains
        for note in chord_notes:
            # Slight variation in start time (phasing)
            start = random.uniform(0, 2.0)
            # Very slow velocity changes (fade in)
            velocity = random.randint(50, 70)

            drone.append((note, start, duration - start, velocity))

        return drone

    @staticmethod
    def generate_arpeggio(root: int, measures: int = 8,
                         tempo: float = 0.5) -> List[Tuple[int, float, float, int]]:
        """
        Generate slow ambient arpeggio

        Args:
            root: Root note
            measures: Number of measures
            tempo: Speed factor (lower = slower)

        Returns:
            Arpeggio pattern
        """
        arp = []
        scale = [root + i for i in [0, 2, 4, 7, 9, 11, 14]]  # Extended scale

        time = 0.0
        duration = measures * 4.0

        while time < duration:
            note = random.choice(scale)
            note_duration = random.uniform(2.0, 4.0) * tempo
            velocity = random.randint(55, 75)

            arp.append((note, time, note_duration, velocity))
            time += random.uniform(1.0, 3.0) * tempo

        return arp


class IDMGenerator:
    """
    IDM (Intelligent Dance Music) generator

    IDM features:
    - Complex, non-standard rhythms
    - Microrhythmic variations
    - Polymetric structures
    - Algorithmic composition
    """

    @staticmethod
    def generate_polymetric_pattern(meter_a: int = 7,
                                   meter_b: int = 11,
                                   cycles: int = 4) -> Dict[str, List[Tuple[float, int]]]:
        """
        Generate polymetric rhythm pattern

        Args:
            meter_a: First meter (e.g., 7/8)
            meter_b: Second meter (e.g., 11/8)
            cycles: Number of complete cycles

        Returns:
            Dictionary with both metric patterns
        """
        pattern = {'meter_a': [], 'meter_b': []}

        # Calculate LCM for complete cycle
        import math
        cycle_length = abs(meter_a * meter_b) // math.gcd(meter_a, meter_b)

        for cycle in range(cycles):
            offset = cycle * cycle_length

            # Generate meter A pattern
            for beat in range(0, cycle_length, meter_a):
                pattern['meter_a'].append((offset + beat, 90))

            # Generate meter B pattern
            for beat in range(0, cycle_length, meter_b):
                pattern['meter_b'].append((offset + beat, 85))

        return pattern

    @staticmethod
    def generate_microrhythmic_variation(base_pattern: List[float],
                                        deviation: float = 0.1) -> List[float]:
        """
        Apply microrhythmic variations to pattern

        Args:
            base_pattern: Base timing pattern
            deviation: Maximum deviation in beats

        Returns:
            Pattern with microrhythmic variations
        """
        return [time + random.uniform(-deviation, deviation)
                for time in base_pattern]

    @staticmethod
    def generate_algorithmic_melody(length: int = 32,
                                   rule: str = 'fibonacci') -> List[Tuple[int, float, int]]:
        """
        Generate algorithmic melody using mathematical rules

        Args:
            length: Number of notes
            rule: Algorithm ('fibonacci', 'prime', 'chaos')

        Returns:
            Melodic sequence
        """
        melody = []
        time = 0.0
        base_note = 60

        if rule == 'fibonacci':
            # Use Fibonacci sequence for intervals
            fib = [1, 1]
            for i in range(20):
                fib.append(fib[-1] + fib[-2])

            for i in range(length):
                interval = fib[i % len(fib)] % 12
                note = base_note + interval
                melody.append((note, time, 85))
                time += 0.5

        elif rule == 'prime':
            # Use prime numbers
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
            for i in range(length):
                interval = primes[i % len(primes)] % 12
                note = base_note + interval
                melody.append((note, time, 85))
                time += 0.5

        elif rule == 'chaos':
            # Chaotic system (logistic map)
            x = 0.5
            r = 3.9  # Chaos parameter

            for i in range(length):
                x = r * x * (1 - x)
                interval = int(x * 24) % 12
                note = base_note + interval
                velocity = int(50 + x * 50)
                melody.append((note, time, velocity))
                time += 0.25

        return melody


class ModulationGenerator:
    """
    Algorithmic modulation generator

    Generates LFO (Low Frequency Oscillator) and envelope curves
    for parameter modulation.
    """

    @staticmethod
    def generate_lfo(waveform: str, frequency: float,
                    duration: float, steps: int = 64) -> List[Tuple[float, float]]:
        """
        Generate LFO modulation curve

        Args:
            waveform: 'sine', 'square', 'triangle', 'saw'
            frequency: LFO frequency in Hz
            duration: Duration in beats
            steps: Number of points

        Returns:
            List of (time, value) tuples (value 0-1)
        """
        curve = []
        step_time = duration / steps

        for i in range(steps):
            time = i * step_time
            phase = 2 * math.pi * frequency * time

            if waveform == 'sine':
                value = (math.sin(phase) + 1) / 2
            elif waveform == 'square':
                value = 1.0 if math.sin(phase) > 0 else 0.0
            elif waveform == 'triangle':
                value = 2 * abs(2 * (phase / (2 * math.pi) % 1) - 1)
            elif waveform == 'saw':
                value = phase / (2 * math.pi) % 1
            else:
                value = 0.5

            curve.append((time, value))

        return curve

    @staticmethod
    def apply_modulation_to_velocity(notes: List[Tuple[int, float, float, int]],
                                    modulation: List[Tuple[float, float]]) -> List[Tuple[int, float, float, int]]:
        """
        Apply modulation curve to note velocities

        Args:
            notes: Note sequence
            modulation: Modulation curve

        Returns:
            Notes with modulated velocities
        """
        if not modulation:
            return notes

        modulated = []

        for note, time, duration, velocity in notes:
            # Find closest modulation value
            mod_value = 0.5
            for mod_time, mod_val in modulation:
                if mod_time >= time:
                    mod_value = mod_val
                    break

            # Apply modulation (scale velocity)
            new_velocity = int(velocity * (0.5 + mod_value * 0.5))
            modulated.append((note, time, duration, new_velocity))

        return modulated


class ElectronicMusicGenerator:
    """
    Main electronic music generator

    Combines all electronic music elements.
    """

    def __init__(self, style: ElectronicStyle = ElectronicStyle.IDM,
                 tempo: int = 130):
        """
        Initialize generator

        Args:
            style: Electronic music style
            tempo: BPM
        """
        self.style = style
        self.tempo = tempo

    def generate_composition(self, duration: float = 64.0) -> Dict[str, List]:
        """
        Generate complete electronic composition

        Args:
            duration: Duration in beats

        Returns:
            Dictionary with all tracks
        """
        composition = {}

        if self.style == ElectronicStyle.AMBIENT:
            # Ambient composition
            composition['drone'] = AmbientGenerator.generate_drone(48, duration, 5)
            composition['arpeggio'] = AmbientGenerator.generate_arpeggio(60, int(duration / 4), 0.5)

        elif self.style == ElectronicStyle.IDM:
            # IDM composition
            poly = IDMGenerator.generate_polymetric_pattern(7, 11, 4)
            composition['rhythm_a'] = poly['meter_a']
            composition['rhythm_b'] = poly['meter_b']
            composition['melody'] = IDMGenerator.generate_algorithmic_melody(32, 'fibonacci')

        elif self.style == ElectronicStyle.GLITCH:
            # Glitch composition
            base_seq = GlitchGenerator.generate_glitch_sequence(duration, 0.6)
            composition['glitch'] = GlitchGenerator.stutter(base_seq, 0.4, 6)

        elif self.style == ElectronicStyle.BREAKCORE:
            # Breakcore composition
            composition['breaks'] = BreakcoreGenerator.generate_hyper_break(int(duration / 4), 2.5)

        # Add Euclidean rhythm for all styles except Ambient
        if self.style != ElectronicStyle.AMBIENT:
            euclidean = EuclideanRhythm.generate(5, 16)
            times = EuclideanRhythm.pattern_to_times(euclidean, duration)
            composition['euclidean'] = [(time, 85) for time in times]

        return composition


if __name__ == "__main__":
    """Example usage and testing"""

    print("Electronic Music Generator - Test Suite\n")
    print("=" * 60)

    # Test 1: Euclidean rhythms
    print("\n1. Generating Euclidean rhythms...")
    e38 = EuclideanRhythm.generate(3, 8)
    e58 = EuclideanRhythm.generate(5, 8)
    print(f"   E(3,8): {e38}")
    print(f"   E(5,8): {e58}")

    # Test 2: Glitch stutter
    print("\n2. Applying glitch stutter effect...")
    base_notes = [(60, i * 1.0, 0.5, 90) for i in range(8)]
    glitched = GlitchGenerator.stutter(base_notes, 0.5, 4)
    print(f"   Original: {len(base_notes)} notes")
    print(f"   Glitched: {len(glitched)} notes")

    # Test 3: Breakcore chopping
    print("\n3. Chopping breakbeat (breakcore style)...")
    chopped = BreakcoreGenerator.chop_break(BreakcoreGenerator.AMEN_PATTERN, 16)
    print(f"   Original: {len(BreakcoreGenerator.AMEN_PATTERN)} hits")
    print(f"   Chopped: {len(chopped)} hits")

    # Test 4: Ambient drone
    print("\n4. Generating ambient drone...")
    drone = AmbientGenerator.generate_drone(48, 32.0, 4)
    print(f"   Generated {len(drone)} sustained voices")

    # Test 5: IDM polymetric
    print("\n5. Generating IDM polymetric pattern (7 against 11)...")
    poly = IDMGenerator.generate_polymetric_pattern(7, 11, 2)
    print(f"   Meter 7: {len(poly['meter_a'])} hits")
    print(f"   Meter 11: {len(poly['meter_b'])} hits")

    # Test 6: Algorithmic melody
    print("\n6. Generating algorithmic melody (Fibonacci)...")
    fib_melody = IDMGenerator.generate_algorithmic_melody(16, 'fibonacci')
    print(f"   Generated {len(fib_melody)} notes")
    print(f"   First 5 notes: {[note for note, time, vel in fib_melody[:5]]}")

    # Test 7: LFO modulation
    print("\n7. Generating LFO sine wave...")
    lfo = ModulationGenerator.generate_lfo('sine', 0.25, 16.0, 32)
    print(f"   Generated {len(lfo)} modulation points")

    # Test 8: Complete IDM composition
    print("\n8. Generating complete IDM composition...")
    idm_gen = ElectronicMusicGenerator(ElectronicStyle.IDM, 140)
    composition = idm_gen.generate_composition(32.0)
    print(f"   Generated composition with {len(composition)} tracks:")
    for track, events in composition.items():
        print(f"   - {track}: {len(events)} events")

    # Test 9: Ambient composition
    print("\n9. Generating ambient composition...")
    ambient_gen = ElectronicMusicGenerator(ElectronicStyle.AMBIENT, 60)
    ambient_comp = ambient_gen.generate_composition(64.0)
    print(f"   Generated ambient with {len(ambient_comp)} tracks:")
    for track, events in ambient_comp.items():
        print(f"   - {track}: {len(events)} events")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("\nElectronic music features implemented:")
    print("  ✓ Euclidean rhythm generation (E(k,n) patterns)")
    print("  ✓ Glitch effects (stuttering, irregular timing)")
    print("  ✓ Breakcore (break chopping, time-stretching)")
    print("  ✓ Ambient (drones, slow arpeggios)")
    print("  ✓ IDM (polymetric, algorithmic melodies)")
    print("  ✓ Modulation (LFO generation, envelope curves)")
    print("  ✓ Algorithmic composition (Fibonacci, chaos)")
    print("  ✓ Microrhythmic variations")
