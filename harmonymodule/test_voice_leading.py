#!/usr/bin/env python3
"""
Simple test script to verify voice leading logic without dependencies
"""
import pretty_midi
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque
import os

# Extract just the voice separation functions from genfromweb5.py
def solve_voice_assignment(current_pitches, prev_assignments, voice_identities, time_step):
    """Solve voice assignment using Hungarian algorithm with continuity tracking."""
    num_voices = len(prev_assignments)
    num_pitches = len(current_pitches)

    if num_pitches == 0:
        return {}

    # Create cost matrix
    cost_matrix = np.zeros((num_voices, num_pitches))

    for voice_idx in range(num_voices):
        prev_pitch = prev_assignments.get(voice_idx)

        if prev_pitch is None:
            # No previous assignment - use register-based assignment
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
            # Has previous assignment - favor closest pitches
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
                    cost += 1000 * (distance - 11)  # Make octave jumps virtually impossible

                # Medium penalty for jumps >= 7 semitones (perfect 5th or more)
                elif distance >= 7:
                    cost += 20 * (distance - 6)

                # Small penalty for jumps >= 4 semitones (major 3rd or more)
                elif distance >= 4:
                    cost += 5 * (distance - 3)

                cost_matrix[voice_idx, pitch_idx] = cost

    # Solve assignment
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

    return assignment

def assign_pitches_to_voices_with_continuity(pitch_arrays, num_voices=7):
    """
    Assign pitches to voices ensuring continuity and voice identity.
    """
    voice_assignments = []
    prev_assignments = {}
    voice_identities = defaultdict(lambda: defaultdict(int))

    for time_step, pitches in enumerate(pitch_arrays):
        if len(pitches) == 0:
            voice_assignments.append({})
            continue

        current_pitches = sorted(list(set(pitches)), reverse=True)

        # Solve assignment using Hungarian algorithm with continuity
        assignment = solve_voice_assignment(current_pitches, prev_assignments, voice_identities, time_step)

        # Update voice identities
        for voice_idx, pitch in assignment.items():
            identity_key = f"voice_{voice_idx}"
            voice_identities[identity_key][pitch] += 1

        voice_assignments.append(assignment)
        prev_assignments = assignment.copy()

    return voice_assignments

def analyze_midi_voice_leading(midi_file_path):
    """Analyze voice leading in a MIDI file."""
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    print(f"\nAnalyzing: {os.path.basename(midi_file_path)}")
    print(f"Number of tracks: {len(midi_data.instruments)}")

    voice_stats = {}
    total_octave_jumps = 0
    max_interval = 0

    for i, instrument in enumerate(midi_data.instruments):
        if instrument.is_drum:
            continue

        notes = sorted(instrument.notes, key=lambda x: x.start)
        if len(notes) < 2:
            continue

        intervals = []
        octave_jumps = 0

        for j in range(1, len(notes)):
            interval = abs(notes[j].pitch - notes[j-1].pitch)
            intervals.append(interval)
            max_interval = max(max_interval, interval)

            if interval >= 12:
                octave_jumps += 1
                total_octave_jumps += 1

        if intervals:
            voice_stats[f"Voice {i+1}"] = {
                'notes': len(notes),
                'avg_interval': np.mean(intervals),
                'max_interval': max(intervals),
                'octave_jumps': octave_jumps,
                'pitch_range': f"{min(n.pitch for n in notes)}-{max(n.pitch for n in notes)}"
            }

    print("\nVoice Statistics:")
    for voice, stats in voice_stats.items():
        print(f"{voice}: {stats['notes']} notes, "
              f"avg interval: {stats['avg_interval']:.1f}, "
              f"max interval: {stats['max_interval']}, "
              f"octave jumps: {stats['octave_jumps']}, "
              f"range: {stats['pitch_range']}")

    print(f"\nTotal octave jumps: {total_octave_jumps}")
    print(f"Max interval across all voices: {max_interval}")

    return voice_stats, total_octave_jumps, max_interval

if __name__ == "__main__":
    # Test the voice leading analysis
    test_files = [
        "/home/arlo/Data/miditest/ChordProg3_combined_voices_20250925-195040 (1).mid",
        "/home/arlo/Data/miditest/myarr.mid"
    ]

    for midi_file in test_files:
        if os.path.exists(midi_file):
            analyze_midi_voice_leading(midi_file)
        else:
            print(f"File not found: {midi_file}")