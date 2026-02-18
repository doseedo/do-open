#!/usr/bin/env python3
"""
Debug script to test voice separation logic directly - minimal version
"""
import numpy as np
import pretty_midi
from scipy.optimize import linear_sum_assignment
from collections import Counter

def separate_piano_roll_voices_debug(piano_roll):
    """
    Debug version of separate_piano_roll_voices with extensive logging
    """
    print(f"🎼 Input piano roll shape: {piano_roll.shape}")

    # Find all time frames with notes
    time_frames = {}
    for t in range(piano_roll.shape[1]):
        active_pitches = np.where(piano_roll[:, t] > 0.1)[0]
        if len(active_pitches) > 0:
            time_frames[t] = list(active_pitches)

    if len(time_frames) == 0:
        return [piano_roll]

    print(f"🎼 Found {len(time_frames)} time frames with notes")

    # Analyze chord structure
    chord_sizes = [len(pitches) for pitches in time_frames.values()]
    max_voices = max(chord_sizes)
    all_pitches = []
    for pitches in time_frames.values():
        all_pitches.extend(pitches)

    all_pitches = sorted(set(all_pitches))
    pitch_range = max(all_pitches) - min(all_pitches) if all_pitches else 0

    chord_counts = Counter(chord_sizes)
    common_chord_size = chord_counts.most_common(1)[0][0]

    print(f"🎼 Chord sizes: min={min(chord_sizes)}, max={max_voices}, common={common_chord_size}")
    print(f"🎼 Pitch range: {min(all_pitches) if all_pitches else 0}-{max(all_pitches) if all_pitches else 0} ({pitch_range} semitones)")

    # Use enough voices to handle all simultaneous notes
    target_voices = max(8, max_voices + 2)
    print(f"🎼 Using {target_voices} voices")

    # Initialize voices
    voices = [np.zeros_like(piano_roll) for _ in range(target_voices)]
    sorted_times = sorted(time_frames.keys())
    voice_assignments = {}

    print(f"\n🎼 Processing {len(sorted_times)} chord changes...")

    for i, current_time in enumerate(sorted_times):
        current_pitches = sorted(time_frames[current_time])
        print(f"\n--- Time {current_time}: pitches {current_pitches} ---")

        if i == 0:
            # First chord - assign by register
            assignments = {}
            for j, pitch in enumerate(current_pitches):
                voice_idx = j % target_voices
                voices[voice_idx][pitch, current_time] = piano_roll[pitch, current_time]
                assignments[voice_idx] = pitch
                print(f"   Initial assignment: Voice {voice_idx} <- Pitch {pitch}")
            voice_assignments[current_time] = assignments
        else:
            # Use Hungarian algorithm for voice leading
            prev_time = sorted_times[i-1]
            prev_assignments = voice_assignments.get(prev_time, {})

            best_assignment = solve_assignment_debug(current_pitches, prev_assignments, target_voices)

            # Check for dropped pitches
            assigned_pitches = {p for p in best_assignment.values() if p is not None}
            dropped_pitches = set(current_pitches) - assigned_pitches

            if dropped_pitches:
                print(f"   ❌ DROPPED PITCHES: {sorted(dropped_pitches)}")
                # Force assign dropped pitches
                available_voices = [v for v in range(target_voices) if best_assignment.get(v) is None]
                for j, pitch in enumerate(sorted(dropped_pitches)):
                    if j < len(available_voices):
                        voice_idx = available_voices[j]
                        best_assignment[voice_idx] = pitch
                        print(f"   🔧 FORCE ASSIGNED: Voice {voice_idx} <- Pitch {pitch}")
                    else:
                        print(f"   ❌ STILL CAN'T ASSIGN: Pitch {pitch}")

            # Apply assignments
            voice_assignments[current_time] = {}
            for voice_idx, pitch in best_assignment.items():
                if pitch is not None and voice_idx < len(voices):
                    voices[voice_idx][pitch, current_time] = piano_roll[pitch, current_time]
                    voice_assignments[current_time][voice_idx] = pitch
                    prev_pitch = prev_assignments.get(voice_idx)
                    if prev_pitch is not None:
                        jump = abs(pitch - prev_pitch)
                        print(f"   Voice {voice_idx}: {prev_pitch} -> {pitch} (jump: {jump})")
                    else:
                        print(f"   Voice {voice_idx}: new <- {pitch}")

    print(f"\n🎼 Filling sustained notes...")
    # Fill in sustained notes between chord changes
    for i in range(len(sorted_times) - 1):
        current_time = sorted_times[i]
        next_time = sorted_times[i + 1]
        current_assignments = voice_assignments.get(current_time, {})

        sustained_count = 0
        for voice_idx, pitch in current_assignments.items():
            if pitch is not None and voice_idx < len(voices):
                for t in range(current_time + 1, next_time):
                    if piano_roll[pitch, t] > 0.1:
                        voices[voice_idx][pitch, t] = piano_roll[pitch, t]
                        sustained_count += 1

        if sustained_count > 0:
            print(f"   Time {current_time}->{next_time}: sustained {sustained_count} notes")

    # Handle final chord sustains
    if sorted_times:
        last_time = sorted_times[-1]
        last_assignments = voice_assignments.get(last_time, {})
        final_sustained = 0

        for voice_idx, pitch in last_assignments.items():
            if pitch is not None and voice_idx < len(voices):
                for t in range(last_time + 1, piano_roll.shape[1]):
                    if piano_roll[pitch, t] > 0.1:
                        voices[voice_idx][pitch, t] = piano_roll[pitch, t]
                        final_sustained += 1

        if final_sustained > 0:
            print(f"   Final sustains: {final_sustained} notes")

    # Count results
    final_voices = []
    for i, voice in enumerate(voices):
        note_count = np.sum(voice > 0.1)
        if note_count > 0:
            final_voices.append(voice)
            print(f"🎼 Voice {len(final_voices)}: {note_count} note events")

    # Verification
    total_original = np.sum(piano_roll > 0.1)
    total_separated = sum(np.sum(voice > 0.1) for voice in final_voices)
    print(f"\n🔍 VERIFICATION:")
    print(f"   Original: {total_original} note events")
    print(f"   Separated: {total_separated} note events")

    if total_separated < total_original:
        print(f"   ❌ LOST {total_original - total_separated} note events!")
    else:
        print(f"   ✅ All note events preserved!")

    return final_voices

def solve_assignment_debug(current_pitches, prev_assignments, num_voices):
    """Debug version with detailed logging"""
    if not current_pitches:
        return {i: None for i in range(num_voices)}

    num_pitches = len(current_pitches)
    cost_matrix = np.full((num_voices, num_pitches), 1000.0)

    # Build cost matrix
    for voice_idx in range(num_voices):
        prev_pitch = prev_assignments.get(voice_idx)

        for pitch_idx, pitch in enumerate(current_pitches):
            if prev_pitch is None:
                # No previous pitch - use register-based cost
                cost = abs(pitch - 60)  # Distance from middle C
            else:
                # Has previous pitch - penalize large jumps
                distance = abs(pitch - prev_pitch)
                cost = distance
                if distance >= 12:  # Octave jump
                    cost += 1000
                elif distance >= 7:  # Fifth jump
                    cost += 100

            cost_matrix[voice_idx, pitch_idx] = cost

    # Solve with Hungarian algorithm
    if num_pitches <= num_voices:
        voice_indices, pitch_indices = linear_sum_assignment(cost_matrix)
    else:
        # More pitches than voices - expand matrix
        expanded_matrix = np.full((num_pitches, num_pitches), 1000.0)
        expanded_matrix[:num_voices, :] = cost_matrix
        voice_indices, pitch_indices = linear_sum_assignment(expanded_matrix)

    # Create assignment
    assignment = {i: None for i in range(num_voices)}
    for voice_idx, pitch_idx in zip(voice_indices, pitch_indices):
        if voice_idx < num_voices and pitch_idx < num_pitches:
            assignment[voice_idx] = current_pitches[pitch_idx]
            cost = cost_matrix[voice_idx, pitch_idx] if voice_idx < cost_matrix.shape[0] else 999
            print(f"      Hungarian: Voice {voice_idx} <- Pitch {current_pitches[pitch_idx]} (cost: {cost:.1f})")

    return assignment

def midi_to_piano_roll(midi_path, fps=43.066):
    """Convert MIDI file to piano roll"""
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    if not midi_data.instruments:
        return np.zeros((128, 100))

    duration = midi_data.get_end_time()
    time_steps = int(duration * fps) + 1
    piano_roll = np.zeros((128, time_steps))

    print(f"Converting MIDI: duration={duration:.3f}s, {len(midi_data.instruments[0].notes)} notes")

    for note in midi_data.instruments[0].notes:
        start_frame = int(note.start * fps)
        end_frame = int(note.end * fps)
        piano_roll[note.pitch, start_frame:end_frame] = 1.0
        print(f"  Note: {pretty_midi.note_number_to_name(note.pitch)} frames {start_frame}-{end_frame}")

    return piano_roll

def main():
    original_midi_path = "/home/arlo/Data/miditest/ChordProg3_basicpitch (2).mid"

    print("="*60)
    print("DEBUGGING VOICE SEPARATION")
    print("="*60)

    # Convert MIDI to piano roll
    piano_roll = midi_to_piano_roll(original_midi_path)

    # Test separation
    voices = separate_piano_roll_voices_debug(piano_roll)

    print(f"\n✅ Separation complete: {len(voices)} voices")

if __name__ == "__main__":
    main()