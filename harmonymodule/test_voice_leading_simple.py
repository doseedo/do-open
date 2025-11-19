#!/usr/bin/env python3
"""
Simple test of the voice leading algorithm with sample chord progressions
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def solve_voice_assignment(current_pitches, prev_assignments, voice_identities, time_step):
    """
    Solve voice assignment using Hungarian algorithm with strict register boundaries.
    """
    num_voices = 7  # Standard number of voices
    num_pitches = len(current_pitches)

    if num_pitches == 0:
        return {i: None for i in range(num_voices)}

    # Create cost matrix
    cost_matrix = np.zeros((num_voices, num_pitches))

    for voice_idx in range(num_voices):
        prev_pitch = prev_assignments.get(voice_idx)

        if prev_pitch is None:
            # No previous assignment - use strict register-based assignment
            for pitch_idx, pitch in enumerate(current_pitches):
                if pitch >= 82:      # A#5+ (Very High)
                    if voice_idx == 0:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 1000  # Prevent cross-register
                elif pitch >= 80:    # A5-G#5 (High)
                    if voice_idx == 0:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 1000
                elif pitch >= 70:    # A#4-G#5 (Upper)
                    if voice_idx == 1:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 1000
                elif pitch >= 65:    # F4-A4 (Mid-Upper)
                    if voice_idx == 2:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 1000
                elif pitch >= 60:    # C4-E4 (Mid)
                    if voice_idx == 3:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 1000
                elif pitch >= 55:    # G3-B3 (Lower)
                    if voice_idx == 4:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 1000
                elif pitch >= 50:    # D3-F#3 (Low)
                    if voice_idx == 5:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 1000
                else:                # C3 and below (Very Low)
                    if voice_idx == 6:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 1000
        else:
            # Has previous assignment - favor closest pitches with strict register enforcement
            for pitch_idx, pitch in enumerate(current_pitches):
                distance = abs(pitch - prev_pitch)
                cost = distance

                # Add historical affinity bonus
                identity_key = f"voice_{voice_idx}"
                if identity_key in voice_identities:
                    pitch_history = voice_identities[identity_key]
                    if pitch in pitch_history:
                        affinity_bonus = min(50, pitch_history[pitch] * 5)  # Cap at 50
                        cost = max(0, cost - affinity_bonus)

                # Virtually impossible penalty for jumps >= 12 semitones (octave or more)
                if distance >= 12:
                    cost += 10000  # Make octave jumps virtually impossible

                # Heavy penalty for jumps >= 7 semitones (perfect 5th or more)
                elif distance >= 7:
                    cost += 50 * (distance - 6)

                # Medium penalty for jumps >= 4 semitones (major 3rd or more)
                elif distance >= 4:
                    cost += 10 * (distance - 3)

                cost_matrix[voice_idx, pitch_idx] = cost

    # Solve assignment using Hungarian algorithm
    if num_voices <= num_pitches:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        assignment = {row_indices[i]: current_pitches[col_indices[i]]
                     for i in range(len(row_indices))}
    else:
        # More voices than pitches - assign best matches first
        assignment = {}
        used_pitches = set()

        for _ in range(num_pitches):
            # Find the minimum cost assignment among unused pitches
            min_cost = float('inf')
            best_voice = None
            best_pitch_idx = None

            for voice_idx in range(num_voices):
                if voice_idx in assignment:
                    continue

                for pitch_idx, pitch in enumerate(current_pitches):
                    if pitch_idx in used_pitches:
                        continue

                    if cost_matrix[voice_idx, pitch_idx] < min_cost:
                        min_cost = cost_matrix[voice_idx, pitch_idx]
                        best_voice = voice_idx
                        best_pitch_idx = pitch_idx

            if best_voice is not None and best_pitch_idx is not None:
                assignment[best_voice] = current_pitches[best_pitch_idx]
                used_pitches.add(best_pitch_idx)

    # Fill in None for voices without assignments
    for voice_idx in range(num_voices):
        if voice_idx not in assignment:
            assignment[voice_idx] = None

    return assignment

def test_chord_progression():
    """Test the algorithm with a sample chord progression"""

    # Sample chord progression similar to the problematic one
    chord_progression = [
        [82, 70, 63, 57, 53, 50, 81],  # First chord with A#5=82, A3=57 (problematic)
        [82, 70, 62, 57, 53, 50],      # Second chord
        [81, 69, 62, 57, 53, 50],      # Third chord
    ]

    prev_assignments = {}
    voice_identities = defaultdict(lambda: defaultdict(int))

    print("Testing chord progression with strict register boundaries:")
    print("=" * 60)

    for i, pitches in enumerate(chord_progression):
        print(f"\nChord {i+1}: {pitches}")

        assignment = solve_voice_assignment(pitches, prev_assignments, voice_identities, i)

        print("Voice assignments:")
        octave_jumps = 0
        max_interval = 0

        for voice_idx in sorted(assignment.keys()):
            current_pitch = assignment[voice_idx]
            prev_pitch = prev_assignments.get(voice_idx)

            if current_pitch is not None and prev_pitch is not None:
                interval = abs(current_pitch - prev_pitch)
                max_interval = max(max_interval, interval)

                if interval >= 12:
                    octave_jumps += 1
                    jump_status = f" [OCTAVE JUMP: {interval} semitones]"
                else:
                    jump_status = f" [interval: {interval}]"
            else:
                jump_status = " [new voice]"

            print(f"  Voice {voice_idx}: {prev_pitch} -> {current_pitch}{jump_status}")

        # Update voice identities
        for voice_idx, pitch in assignment.items():
            if pitch is not None:
                identity_key = f"voice_{voice_idx}"
                voice_identities[identity_key][pitch] += 1

        prev_assignments = assignment.copy()

        print(f"Octave jumps in this chord: {octave_jumps}")
        print(f"Max interval: {max_interval}")

if __name__ == "__main__":
    test_chord_progression()