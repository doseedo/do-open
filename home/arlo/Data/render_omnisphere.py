#!/usr/bin/env python3
"""
Omnisphere MIDI Rendering Helper Script

This script renders MIDI files through the Omnisphere VST3 instrument plugin.
It supports multiple rendering backends:
1. Carla (recommended) - Install with: sudo apt install carla
2. Python-based VST hosting (experimental)

Usage:
    python render_omnisphere.py input.mid output.wav --patch 0
"""

import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import mido
import numpy as np


# Default paths
OMNISPHERE_VST3_PATH = "/home/arlo/.vst3/yabridge/Omnisphere.vst3"
OMNISPHERE_VST2_PATH = "/home/arlo/.vst/yabridge/Omnisphere.so"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_PATCH = 0  # Default program/patch number


def get_midi_duration(midi_path: str) -> float:
    """
    Calculate the total duration of a MIDI file in seconds.

    Args:
        midi_path: Path to MIDI file

    Returns:
        Duration in seconds
    """
    mid = mido.MidiFile(midi_path)
    return mid.length


def set_midi_program(midi_path: str, output_path: str, program: int, channel: int = 0) -> str:
    """
    Add a program change message at the start of a MIDI file.

    Args:
        midi_path: Input MIDI file
        output_path: Output MIDI file
        program: Program/patch number (0-127)
        channel: MIDI channel (0-15)

    Returns:
        Path to modified MIDI file
    """
    mid = mido.MidiFile(midi_path)

    # Create new MIDI file with program change
    new_mid = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)

    for i, track in enumerate(mid.tracks):
        new_track = mido.MidiTrack()

        # Add program change at the very beginning (time=0)
        if i == 0 or mid.type == 1:  # First track or format 1 (each track is separate)
            new_track.append(mido.Message('program_change', program=program, channel=channel, time=0))

        # Copy all original messages
        for msg in track:
            new_track.append(msg.copy())

        new_mid.tracks.append(new_track)

    new_mid.save(output_path)
    return output_path


def render_midi_with_dawdreamer(
    midi_path: str,
    output_path: str,
    patch_number: int = DEFAULT_PATCH,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    tail_duration: float = 3.0,
    verbose: bool = True
) -> str:
    """
    Render a MIDI file through Omnisphere VST3 using DawDreamer.

    DawDreamer is a Python audio processing framework that can host VST plugins
    and properly route MIDI events to them.

    Args:
        midi_path: Path to input MIDI file
        output_path: Path to output WAV file
        patch_number: Program/patch number to use (0-127)
        sample_rate: Sample rate in Hz
        tail_duration: Extra seconds to render after MIDI ends (for reverb/delay tails)
        verbose: Print progress information

    Returns:
        Path to the output audio file
    """
    import dawdreamer as daw
    from scipy.io import wavfile

    midi_path = Path(midi_path).resolve()
    output_path = Path(output_path).resolve()

    if verbose:
        print(f"🎹 Omnisphere MIDI Renderer (DawDreamer Backend)")
        print(f"   MIDI: {midi_path.name}")
        print(f"   Output: {output_path.name}")
        print(f"   Patch: {patch_number}")
        print(f"   Sample Rate: {sample_rate}Hz")

    # Get MIDI duration
    midi_duration = get_midi_duration(str(midi_path))
    total_duration = midi_duration + tail_duration

    if verbose:
        print(f"   ⏱️  MIDI duration: {midi_duration:.2f}s")
        print(f"   ⏱️  Total render duration: {total_duration:.2f}s (includes {tail_duration}s tail)")

    # Create temporary MIDI file with program change
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mid', delete=False) as tmp_midi:
        tmp_midi_path = tmp_midi.name

    try:
        if verbose:
            print(f"   🎼 Adding program change to MIDI...")

        set_midi_program(str(midi_path), tmp_midi_path, patch_number)

        if verbose:
            print(f"   🔌 Loading Omnisphere with DawDreamer...")

        # Create DawDreamer engine
        engine = daw.RenderEngine(sample_rate, 512)  # block_size = 512

        # Load Omnisphere as a plugin processor (use VST2 for better compatibility)
        plugin_path = OMNISPHERE_VST2_PATH if Path(OMNISPHERE_VST2_PATH).exists() else OMNISPHERE_VST3_PATH
        synth = engine.make_plugin_processor("omnisphere", str(plugin_path))

        if verbose:
            print(f"   ✅ Omnisphere loaded successfully")
            # Get plugin info if available
            try:
                print(f"   📊 Plugin: {synth.get_name()}")
            except:
                pass

        # Load MIDI file
        synth.load_midi(tmp_midi_path)

        if verbose:
            print(f"   🎵 MIDI loaded, rendering audio...")

        # Load the graph (just the synth, no other processors)
        engine.load_graph([
            (synth, [])
        ])

        # Render audio
        engine.render(total_duration)

        # Get rendered audio
        audio = engine.get_audio()  # Returns numpy array (channels, samples)

        if verbose:
            print(f"   🔊 Rendered {audio.shape[1] / sample_rate:.2f}s of audio ({audio.shape[0]} channels)")

        # Downmix to stereo if needed (take first 2 channels or sum all channels)
        if audio.shape[0] > 2:
            if verbose:
                print(f"   🎚️  Downmixing from {audio.shape[0]} channels to stereo")
            # Use first 2 channels (main stereo output)
            audio = audio[:2, :]
        elif audio.shape[0] == 1:
            # Mono to stereo
            audio = np.vstack([audio, audio])

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize to prevent clipping
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95  # Leave some headroom

        # Convert to int16 for WAV file (DawDreamer returns float32)
        audio_int16 = (audio.T * 32767).astype(np.int16)

        # Save as WAV
        wavfile.write(str(output_path), sample_rate, audio_int16)

        if verbose:
            print(f"   ✅ Render complete!")
            print(f"   💾 Output: {output_path}")

        return str(output_path)

    finally:
        # Clean up temp file
        Path(tmp_midi_path).unlink(missing_ok=True)


def render_midi_python_fallback(
    midi_path: str,
    output_path: str,
    patch_number: int = DEFAULT_PATCH,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    tail_duration: float = 3.0,
    verbose: bool = True
) -> str:
    """
    Experimental Python-based MIDI rendering using pedalboard.

    WARNING: This is a fallback method and may not work correctly for all plugins.
    Omnisphere requires proper MIDI event routing which this simplified approach
    cannot provide. Use the Carla backend for reliable results.

    This creates audio by generating note-by-note using plugin presets,
    which won't capture the full expressiveness of Omnisphere.

    Args:
        midi_path: Path to input MIDI file
        output_path: Path to output WAV file
        patch_number: Program/patch number to use
        sample_rate: Sample rate in Hz
        tail_duration: Extra seconds to render after MIDI ends
        verbose: Print progress information

    Returns:
        Path to output audio file
    """
    from pedalboard import load_plugin
    from pedalboard.io import AudioFile

    if verbose:
        print(f"🎹 Omnisphere MIDI Renderer (Python Fallback - Experimental)")
        print(f"   ⚠️  This is a fallback method and may not work properly!")
        print(f"   ⚠️  For best results, install Carla: sudo apt install carla")
        print(f"   MIDI: {midi_path}")
        print(f"   Output: {output_path}")
        print(f"   Patch: {patch_number}")

    # Load plugin
    omnisphere = load_plugin(OMNISPHERE_VST3_PATH, initialization_timeout=30.0)

    # Set program if supported
    if hasattr(omnisphere, 'program'):
        omnisphere.program = patch_number
        if verbose:
            print(f"   ✅ Set program to: {patch_number}")

    # Get MIDI duration
    midi_duration = get_midi_duration(midi_path)
    total_duration = midi_duration + tail_duration

    if verbose:
        print(f"   ⏱️  Total duration: {total_duration:.2f}s")

    # Create silent audio buffer
    total_samples = int(total_duration * sample_rate)
    audio_buffer = np.zeros((2, total_samples), dtype=np.float32)

    # Process silent audio through plugin to generate sound
    # Note: This won't trigger MIDI events without proper VST MIDI routing
    if verbose:
        print(f"   🔊 Processing audio (note: MIDI events may not trigger correctly)...")

    processed = omnisphere.process(audio_buffer, sample_rate)

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with AudioFile(str(output_path), 'w', sample_rate, 2) as f:
        f.write(processed)

    if verbose:
        print(f"   ⚠️  Output created, but may be silent due to MIDI routing limitations")
        print(f"   💾 Output: {output_path}")

    return str(output_path)


def render_midi_auto(
    midi_path: str,
    output_path: str,
    patch_number: int = DEFAULT_PATCH,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    tail_duration: float = 3.0,
    verbose: bool = True,
    force_fallback: bool = False
) -> str:
    """
    Automatically choose the best available rendering backend.

    Tries backends in this order:
    1. DawDreamer (recommended - proper MIDI support)
    2. Python fallback (experimental, likely to produce silence)

    Args:
        midi_path: Path to input MIDI file
        output_path: Path to output WAV file
        patch_number: Program/patch number
        sample_rate: Sample rate in Hz
        tail_duration: Extra seconds to render after MIDI
        verbose: Print progress
        force_fallback: Force use of Python fallback (for testing)

    Returns:
        Path to output audio file
    """
    # Try DawDreamer first unless forced to use fallback
    if not force_fallback:
        try:
            import dawdreamer
            if verbose:
                print("   ✅ Using DawDreamer backend (recommended)")
            return render_midi_with_dawdreamer(
                midi_path, output_path, patch_number,
                sample_rate, tail_duration, verbose
            )
        except ImportError:
            if verbose:
                print("   ⚠️  DawDreamer not found")
                print("   ⚠️  Install with: pip install dawdreamer")

    # Fall back to Python method
    if verbose:
        print("   ⚠️  Using Python fallback (may not work properly)")

    return render_midi_python_fallback(
        midi_path, output_path, patch_number,
        sample_rate, tail_duration, verbose
    )


def main():
    """Command-line interface for MIDI rendering."""
    parser = argparse.ArgumentParser(
        description="Render MIDI files through Omnisphere VST3"
    )
    parser.add_argument(
        "midi_file",
        type=str,
        help="Input MIDI file path"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output WAV file path"
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=DEFAULT_PATCH,
        help=f"Patch/program number (default: {DEFAULT_PATCH})"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Sample rate in Hz (default: {DEFAULT_SAMPLE_RATE})"
    )
    parser.add_argument(
        "--tail",
        type=float,
        default=3.0,
        help="Extra seconds after MIDI ends for reverb/delay tails (default: 3.0)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Force use of Python fallback (for testing)"
    )

    args = parser.parse_args()

    try:
        render_midi_auto(
            midi_path=args.midi_file,
            output_path=args.output_file,
            patch_number=args.patch,
            sample_rate=args.sample_rate,
            tail_duration=args.tail,
            verbose=not args.quiet,
            force_fallback=args.force_fallback
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
