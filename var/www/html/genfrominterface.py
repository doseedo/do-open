"""
Audio Track Mixing/Summing Utility
Generates a master/summed track from multiple audio files
"""

import numpy as np
from pydub import AudioSegment
import os
from typing import List, Optional


def sum_audio_tracks(audio_file_paths: List[str], output_path: str, normalize: bool = True) -> str:
    """
    Sum multiple audio tracks into a single master track.

    Args:
        audio_file_paths: List of paths to audio files to sum
        output_path: Path where the summed/master track should be saved
        normalize: Whether to normalize the output to prevent clipping (default: True)

    Returns:
        Path to the generated master track

    Example:
        >>> tracks = ['/path/to/track1.wav', '/path/to/track2.wav', '/path/to/track3.wav']
        >>> master = sum_audio_tracks(tracks, '/path/to/master.wav')
    """
    if not audio_file_paths:
        raise ValueError("No audio files provided")

    if len(audio_file_paths) == 1:
        # If only one track, just copy it
        audio = AudioSegment.from_file(audio_file_paths[0])
        audio.export(output_path, format='wav')
        return output_path

    # Load all audio files
    audio_segments = []
    max_duration = 0

    for file_path in audio_file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        audio = AudioSegment.from_file(file_path)
        audio_segments.append(audio)
        max_duration = max(max_duration, len(audio))

    if not audio_segments:
        raise ValueError("No valid audio files found")

    # Ensure all tracks are the same length by padding with silence
    padded_segments = []
    for audio in audio_segments:
        if len(audio) < max_duration:
            # Pad with silence
            silence = AudioSegment.silent(duration=max_duration - len(audio))
            audio = audio + silence
        padded_segments.append(audio)

    # Sum all tracks by overlaying them
    master_track = padded_segments[0]
    for audio in padded_segments[1:]:
        master_track = master_track.overlay(audio)

    # Normalize if requested to prevent clipping
    if normalize:
        # Get the maximum possible sample value
        max_possible_value = master_track.max_possible_amplitude

        # Get current max value
        current_max = master_track.max

        # Calculate normalization factor
        if current_max > 0:
            # Leave some headroom (multiply by 0.95 to avoid clipping)
            normalization_factor = (max_possible_value * 0.95) / current_max

            # Apply normalization (convert to dB change)
            db_change = 20 * np.log10(normalization_factor)
            master_track = master_track + db_change

    # Export the master track
    master_track.export(output_path, format='wav')

    print(f"✅ Generated master track: {output_path}")
    print(f"   - Summed {len(audio_segments)} tracks")
    print(f"   - Duration: {len(master_track) / 1000:.2f} seconds")
    print(f"   - Normalized: {normalize}")

    return output_path


def mix_with_levels(audio_file_paths: List[str],
                    output_path: str,
                    levels: Optional[List[float]] = None,
                    normalize: bool = True) -> str:
    """
    Mix multiple audio tracks with individual level control.

    Args:
        audio_file_paths: List of paths to audio files to mix
        output_path: Path where the mixed track should be saved
        levels: List of volume levels (0.0 to 1.0) for each track. If None, all tracks at 1.0
        normalize: Whether to normalize the output (default: True)

    Returns:
        Path to the generated mixed track

    Example:
        >>> tracks = ['/path/to/track1.wav', '/path/to/track2.wav']
        >>> levels = [1.0, 0.5]  # Second track at 50% volume
        >>> mixed = mix_with_levels(tracks, '/path/to/mixed.wav', levels)
    """
    if not audio_file_paths:
        raise ValueError("No audio files provided")

    if levels is None:
        levels = [1.0] * len(audio_file_paths)

    if len(levels) != len(audio_file_paths):
        raise ValueError(f"Number of levels ({len(levels)}) must match number of tracks ({len(audio_file_paths)})")

    # Load and apply levels to all audio files
    audio_segments = []
    max_duration = 0

    for file_path, level in zip(audio_file_paths, levels):
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        audio = AudioSegment.from_file(file_path)

        # Apply level adjustment (convert to dB)
        if level != 1.0:
            db_change = 20 * np.log10(level) if level > 0 else -100
            audio = audio + db_change

        audio_segments.append(audio)
        max_duration = max(max_duration, len(audio))

    if not audio_segments:
        raise ValueError("No valid audio files found")

    # Ensure all tracks are the same length
    padded_segments = []
    for audio in audio_segments:
        if len(audio) < max_duration:
            silence = AudioSegment.silent(duration=max_duration - len(audio))
            audio = audio + silence
        padded_segments.append(audio)

    # Mix all tracks
    mixed_track = padded_segments[0]
    for audio in padded_segments[1:]:
        mixed_track = mixed_track.overlay(audio)

    # Normalize if requested
    if normalize:
        max_possible_value = mixed_track.max_possible_amplitude
        current_max = mixed_track.max

        if current_max > 0:
            normalization_factor = (max_possible_value * 0.95) / current_max
            db_change = 20 * np.log10(normalization_factor)
            mixed_track = mixed_track + db_change

    # Export
    mixed_track.export(output_path, format='wav')

    print(f"✅ Generated mixed track: {output_path}")
    print(f"   - Mixed {len(audio_segments)} tracks")
    print(f"   - Duration: {len(mixed_track) / 1000:.2f} seconds")

    return output_path


# Example usage for integration with backend API
if __name__ == "__main__":
    # Example: Sum all tracks in a directory
    import sys

    if len(sys.argv) < 3:
        print("Usage: python genfrominterface.py <output_path> <track1> <track2> ...")
        print("Example: python genfrominterface.py master.wav track1.wav track2.wav track3.wav")
        sys.exit(1)

    output = sys.argv[1]
    tracks = sys.argv[2:]

    print(f"Summing {len(tracks)} tracks into {output}...")
    result = sum_audio_tracks(tracks, output)
    print(f"Done! Master track saved to: {result}")
