#!/usr/bin/env python3
"""
Generate improved voice separation MIDI using the fixed algorithm
"""
import pretty_midi
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import os

def solve_voice_assignment(current_pitches, prev_assignments, voice_identities, time_step):
    """Voice assignment with overlapping register boundaries"""
    num_voices = 7
    num_pitches = len(current_pitches)

    if num_pitches == 0:
        return {i: None for i in range(num_voices)}

    cost_matrix = np.zeros((num_voices, num_pitches))

    for voice_idx in range(num_voices):
        prev_pitch = prev_assignments.get(voice_idx)

        if prev_pitch is None:
            for pitch_idx, pitch in enumerate(current_pitches):
                # Overlapping register boundaries
                if pitch >= 80:
                    if voice_idx == 0:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx == 1:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 70:
                    if voice_idx == 1:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [0, 2]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 65:
                    if voice_idx == 2:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [1, 3]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 60:
                    if voice_idx == 3:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [2, 4]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 55:
                    if voice_idx == 4:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [3, 5]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 50:
                    if voice_idx == 5:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [4, 6]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                else:
                    if voice_idx == 6:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx == 5:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
        else:
            for pitch_idx, pitch in enumerate(current_pitches):
                distance = abs(pitch - prev_pitch)
                cost = distance

                # Apply register preferences
                register_penalty = 0
                if pitch >= 80:
                    if voice_idx == 0:
                        register_penalty = 0
                    elif voice_idx == 1:
                        register_penalty = 50
                    else:
                        register_penalty = 200
                elif pitch >= 70:
                    if voice_idx == 1:
                        register_penalty = 0
                    elif voice_idx in [0, 2]:
                        register_penalty = 50
                    else:
                        register_penalty = 200
                elif pitch >= 65:
                    if voice_idx == 2:
                        register_penalty = 0
                    elif voice_idx in [1, 3]:
                        register_penalty = 50
                    else:
                        register_penalty = 200
                elif pitch >= 60:
                    if voice_idx == 3:
                        register_penalty = 0
                    elif voice_idx in [2, 4]:
                        register_penalty = 50
                    else:
                        register_penalty = 200
                elif pitch >= 55:
                    if voice_idx == 4:
                        register_penalty = 0
                    elif voice_idx in [3, 5]:
                        register_penalty = 50
                    else:
                        register_penalty = 200
                elif pitch >= 50:
                    if voice_idx == 5:
                        register_penalty = 0
                    elif voice_idx in [4, 6]:
                        register_penalty = 50
                    else:
                        register_penalty = 200
                else:
                    if voice_idx == 6:
                        register_penalty = 0
                    elif voice_idx == 5:
                        register_penalty = 50
                    else:
                        register_penalty = 200

                cost += register_penalty

                # Historical affinity
                identity_key = f"voice_{voice_idx}"
                if identity_key in voice_identities:
                    pitch_history = voice_identities[identity_key]
                    if pitch in pitch_history:
                        affinity_bonus = min(50, pitch_history[pitch] * 5)
                        cost = max(0, cost - affinity_bonus)

                # Jump penalties
                if distance >= 12:
                    cost += 100000
                elif distance >= 7:
                    cost += 1000 * (distance - 6)
                elif distance >= 4:
                    cost += 100 * (distance - 3)

                cost_matrix[voice_idx, pitch_idx] = cost

    # Solve assignment
    finite_assignments = []
    for voice_idx in range(num_voices):
        for pitch_idx in range(num_pitches):
            finite_assignments.append((voice_idx, pitch_idx, cost_matrix[voice_idx, pitch_idx]))

    finite_assignments.sort(key=lambda x: x[2])

    assignment = {i: [] for i in range(num_voices)}
    used_pitches = set()

    for voice_idx, pitch_idx, cost in finite_assignments:
        if pitch_idx not in used_pitches:
            assignment[voice_idx].append(current_pitches[pitch_idx])
            used_pitches.add(pitch_idx)

    # Convert to single pitches for compatibility
    final_assignment = {}
    for voice_idx in range(num_voices):
        if assignment[voice_idx]:
            final_assignment[voice_idx] = max(assignment[voice_idx])  # Take highest pitch
        else:
            final_assignment[voice_idx] = None

    return final_assignment

def generate_improved_voices():
    """Generate improved voice separation MIDI"""

    # Load original MIDI
    midi_file = "/home/arlo/Data/miditest/ChordProg3_basicpitch (2).mid"
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    piano_roll = midi_data.get_piano_roll(fs=43.066)

    print(f"Generating improved voices from {os.path.basename(midi_file)}")

    # Extract time frames
    time_frames = {}
    for t in range(piano_roll.shape[1]):
        active_pitches = np.where(piano_roll[:, t] > 0.1)[0]
        if len(active_pitches) > 0:
            time_frames[t] = list(active_pitches)

    # Create voice tracks
    voices = [np.zeros_like(piano_roll) for _ in range(7)]

    prev_assignments = {}
    voice_identities = defaultdict(lambda: defaultdict(int))
    total_assigned = 0
    total_input = 0

    # Process each time frame
    for t, current_pitches in sorted(time_frames.items()):
        current_pitches = sorted(current_pitches)
        total_input += len(current_pitches)

        if t == 0:
            # Initial assignment by register
            assignment = {}
            for voice_idx in range(7):
                assignment[voice_idx] = None

            sorted_pitches = sorted(current_pitches, reverse=True)
            for pitch in sorted_pitches:
                if pitch >= 80:
                    voice_idx = 0
                elif pitch >= 70:
                    voice_idx = 1
                elif pitch >= 65:
                    voice_idx = 2
                elif pitch >= 60:
                    voice_idx = 3
                elif pitch >= 55:
                    voice_idx = 4
                elif pitch >= 50:
                    voice_idx = 5
                else:
                    voice_idx = 6

                # Find available voice
                original_voice = voice_idx
                attempts = 0
                while assignment.get(voice_idx) is not None and attempts < 7:
                    voice_idx = (original_voice + attempts + 1) % 7
                    attempts += 1

                assignment[voice_idx] = pitch
        else:
            assignment = solve_voice_assignment(current_pitches, prev_assignments, voice_identities, t)

        # Apply assignment to voice tracks
        assigned_pitches = 0
        for voice_idx, pitch in assignment.items():
            if pitch is not None:
                voices[voice_idx][pitch, t] = piano_roll[pitch, t]
                assigned_pitches += 1

                # Update voice identities
                identity_key = f"voice_{voice_idx}"
                voice_identities[identity_key][pitch] += 1

        total_assigned += assigned_pitches
        prev_assignments = assignment.copy()

        # Debug dropped pitches
        assigned_pitch_set = {p for p in assignment.values() if p is not None}
        dropped_pitches = set(current_pitches) - assigned_pitch_set
        if dropped_pitches:
            print(f"Time {t:3d}: DROPPED {sorted(dropped_pitches)} from {current_pitches}")

    print(f"Coverage: {total_assigned}/{total_input} = {(total_assigned/total_input)*100:.1f}%")

    # Create MIDI file with voices
    output_midi = pretty_midi.PrettyMIDI()

    for voice_idx, voice_roll in enumerate(voices):
        instrument = pretty_midi.Instrument(program=0, name=f"Voice {voice_idx + 1}")

        # Convert piano roll back to notes
        for pitch in range(voice_roll.shape[0]):
            note_starts = []
            note_ends = []

            # Find note onsets and offsets
            is_playing = False
            for t in range(voice_roll.shape[1]):
                if voice_roll[pitch, t] > 0.1 and not is_playing:
                    note_starts.append(t / 43.066)  # Convert to seconds
                    is_playing = True
                elif voice_roll[pitch, t] <= 0.1 and is_playing:
                    note_ends.append(t / 43.066)
                    is_playing = False

            # Handle notes that don't end
            if is_playing:
                note_ends.append((voice_roll.shape[1] - 1) / 43.066)

            # Create notes
            for start, end in zip(note_starts, note_ends):
                if end > start:  # Valid note duration
                    note = pretty_midi.Note(
                        velocity=80,
                        pitch=pitch,
                        start=start,
                        end=end
                    )
                    instrument.notes.append(note)

        output_midi.instruments.append(instrument)

    # Save output
    output_path = "/home/arlo/Data/miditest/ChordProg3_improved_voices.mid"
    output_midi.write(output_path)
    print(f"Saved improved voices to: {output_path}")

    return output_path

if __name__ == "__main__":
    generate_improved_voices()