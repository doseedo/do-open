#!/usr/bin/env python3
"""
Per-Syllable ACE-Step Generation

Generates each MIDI note/syllable separately through ACE-Step diffusion,
then combines the final outputs. This gives more precise control over each
syllable's generation.

Usage:
    python3 generate_per_syllable_ace_step.py \
        --midi input.mid \
        --lyrics-map '{"0": "do", "1": "re", "2": "mi"}' \
        --prompt "choir vocals" \
        --output combined_output.wav \
        --steps 100 \
        --seed 42
"""

import sys
import os
import json
import argparse
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import pretty_midi
import librosa

sys.path.append('/home/arlo/Data')


def extract_single_note_midi(midi_path: str, note_index: int, output_path: str):
    """
    Extract a single note from MIDI file and save as new MIDI.

    Args:
        midi_path: Path to input MIDI file
        note_index: Index of note to extract (0-based)
        output_path: Path to save single-note MIDI
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted(midi.instruments[0].notes, key=lambda n: n.start)

    if note_index >= len(notes):
        raise ValueError(f"Note index {note_index} out of range (only {len(notes)} notes)")

    target_note = notes[note_index]

    # Create new MIDI with just this note
    new_midi = pretty_midi.PrettyMIDI()
    new_instrument = pretty_midi.Instrument(program=midi.instruments[0].program)

    # Add the single note, shifted to start at 0
    shifted_note = pretty_midi.Note(
        velocity=target_note.velocity,
        pitch=target_note.pitch,
        start=0.0,
        end=target_note.end - target_note.start
    )
    new_instrument.notes.append(shifted_note)
    new_midi.instruments.append(new_instrument)

    # Save
    new_midi.write(output_path)

    return {
        'start_time': target_note.start,
        'duration': target_note.end - target_note.start,
        'pitch': target_note.pitch
    }


def generate_syllable_with_ace_step(
    midi_path: str,
    syllable: str,
    prompt: str,
    output_path: str,
    steps: int = 100,
    seed: int = 0,
    noise_level: float = 0.4,
    pipeline=None
):
    """
    Generate audio for a single syllable using ACE-Step.

    Args:
        midi_path: Path to single-note MIDI file
        syllable: Syllable text (e.g., "do")
        prompt: Generation prompt
        output_path: Where to save output
        steps: Diffusion steps
        seed: Random seed
        noise_level: Noise level for audio2audio
        pipeline: Pre-loaded ACE-Step pipeline (if available)
    """
    print(f"  Generating syllable '{syllable}'...")

    # First, generate syllable conditioning audio with WORLD vocoder
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        conditioning_audio = tmp_audio.name

    # Use espeak + WORLD vocoder to create conditioning
    result = subprocess.run([
        'python3', '/home/arlo/Data/espeak_world_vocoder_aligned.py',
        '--midi', midi_path,
        '--lyrics', syllable,
        '--output', conditioning_audio,
        '--quiet'
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ❌ WORLD vocoder failed!")
        print(f"    STDOUT: {result.stdout}")
        print(f"    STDERR: {result.stderr}")
        raise RuntimeError(f"WORLD vocoder failed for syllable '{syllable}'")

    # Now run ACE-Step with this conditioning
    if pipeline is not None:
        # Use pre-loaded pipeline
        task = "audio2audio"
        audio2audio_enable = True
        ref_audio_strength = 1.0 - noise_level

        # Get duration from MIDI
        midi = pretty_midi.PrettyMIDI(midi_path)
        duration = midi.get_end_time()

        pipeline(
            prompt=prompt,
            lyrics=syllable,
            audio_duration=duration,
            infer_step=steps,
            manual_seeds=[seed],
            guidance_scale=15.0,
            save_path=output_path,
            task=task,
            audio2audio_enable=audio2audio_enable,
            ref_audio_input=conditioning_audio,
            ref_audio_strength=ref_audio_strength
        )
    else:
        # Use wrapper script
        subprocess.run([
            'bash', '-c',
            f"""
            eval "$(conda shell.bash hook)"
            conda activate ace_step
            cd /home/arlo/Data
            python3 ace_step_noise_wrapper.py \
                --prompt "{prompt}" \
                --lyrics "{syllable}" \
                --steps {steps} \
                --output "{output_path}" \
                --seed {seed} \
                --noise-level {noise_level} \
                --ref-audio "{conditioning_audio}"
            """
        ], check=True, capture_output=True, text=True)

    # Clean up temp file
    if os.path.exists(conditioning_audio):
        os.remove(conditioning_audio)

    print(f"    ✅ Generated: {output_path}")


def combine_syllable_outputs(
    syllable_files: list,
    syllable_timings: list,
    output_path: str,
    sample_rate: int = 44100
):
    """
    Combine individual syllable outputs into final audio.

    Args:
        syllable_files: List of paths to syllable WAV files
        syllable_timings: List of dicts with 'start_time' and 'duration'
        output_path: Path to save combined output
        sample_rate: Output sample rate
    """
    print(f"\n🎵 Combining {len(syllable_files)} syllables...")

    # Calculate total duration
    total_duration = max(t['start_time'] + t['duration'] for t in syllable_timings)
    total_samples = int(total_duration * sample_rate)

    # Create output buffer
    output_audio = np.zeros(total_samples, dtype=np.float32)

    # Load and place each syllable
    for i, (syl_file, timing) in enumerate(zip(syllable_files, syllable_timings)):
        print(f"  [{i+1}/{len(syllable_files)}] Placing syllable at {timing['start_time']:.2f}s...")

        # Load syllable audio
        syl_audio, sr = librosa.load(syl_file, sr=sample_rate, mono=True)

        # Calculate placement
        start_sample = int(timing['start_time'] * sample_rate)
        end_sample = min(start_sample + len(syl_audio), total_samples)
        syl_length = end_sample - start_sample

        # Place in output (with simple crossfade if overlapping)
        if i > 0:
            # Check for overlap with previous syllable
            prev_timing = syllable_timings[i-1]
            prev_end = prev_timing['start_time'] + prev_timing['duration']

            if timing['start_time'] < prev_end:
                # Overlapping - apply crossfade
                overlap_duration = prev_end - timing['start_time']
                overlap_samples = int(overlap_duration * sample_rate)

                if overlap_samples > 0 and overlap_samples < len(syl_audio):
                    # Create fade out for previous segment
                    fade_start = int((prev_end - overlap_duration) * sample_rate)
                    fade_end = int(prev_end * sample_rate)
                    fade_length = fade_end - fade_start

                    if fade_length > 0 and fade_end <= len(output_audio):
                        fade_out = np.linspace(1.0, 0.0, fade_length)
                        output_audio[fade_start:fade_end] *= fade_out

                    # Create fade in for current segment
                    fade_in = np.linspace(0.0, 1.0, overlap_samples)
                    syl_audio[:overlap_samples] *= fade_in

        # Add to output
        output_audio[start_sample:end_sample] += syl_audio[:syl_length]

    # Normalize
    max_val = np.max(np.abs(output_audio))
    if max_val > 0:
        output_audio = output_audio / (max_val * 1.1)

    # Save
    sf.write(output_path, output_audio, sample_rate)
    print(f"✅ Combined output saved: {output_path}")


def generate_per_syllable(
    midi_path: str,
    midi_lyric_map: dict,
    prompt: str,
    output_path: str,
    steps: int = 100,
    seed: int = 0,
    noise_level: float = 0.4,
    keep_intermediates: bool = False,
    pipeline=None
):
    """
    Main function: Generate each syllable separately and combine.

    Args:
        midi_path: Path to input MIDI file
        midi_lyric_map: Dict mapping note indices to syllables
        prompt: Generation prompt
        output_path: Path to save final output
        steps: Diffusion steps
        seed: Random seed
        noise_level: Noise level for audio2audio
        keep_intermediates: Keep individual syllable files
        pipeline: Pre-loaded ACE-Step pipeline (optional)
    """
    print(f"\n{'='*80}")
    print(f"🎵 PER-SYLLABLE ACE-STEP GENERATION")
    print(f"{'='*80}")
    print(f"MIDI: {midi_path}")
    print(f"Syllables: {len(midi_lyric_map)}")
    print(f"Prompt: {prompt}")
    print(f"Steps: {steps}, Seed: {seed}")
    print(f"Noise Level: {noise_level}")
    print(f"{'='*80}\n")

    # Create temp directory for intermediate files
    temp_dir = Path(tempfile.mkdtemp(prefix='ace_step_syllables_'))
    print(f"📁 Working directory: {temp_dir}\n")

    try:
        # Sort syllables by note index
        sorted_indices = sorted([int(k) for k in midi_lyric_map.keys()])

        syllable_files = []
        syllable_timings = []

        # Process each syllable
        for i, note_idx in enumerate(sorted_indices):
            syllable = midi_lyric_map[str(note_idx)]

            print(f"\n{'─'*80}")
            print(f"[{i+1}/{len(sorted_indices)}] Processing syllable '{syllable}' (note {note_idx})")
            print(f"{'─'*80}")

            # Extract single note MIDI
            single_note_midi = temp_dir / f"note_{note_idx}.mid"
            timing = extract_single_note_midi(midi_path, note_idx, str(single_note_midi))
            print(f"  📝 MIDI: {single_note_midi.name}")
            print(f"  ⏱️  Start: {timing['start_time']:.2f}s, Duration: {timing['duration']:.2f}s")
            print(f"  🎹 Pitch: {timing['pitch']} ({pretty_midi.note_number_to_name(timing['pitch'])})")

            # Generate with ACE-Step
            syllable_output = temp_dir / f"syllable_{note_idx}_{syllable}.wav"

            # Use per-syllable seed for variation
            syllable_seed = seed + i

            generate_syllable_with_ace_step(
                midi_path=str(single_note_midi),
                syllable=syllable,
                prompt=prompt,
                output_path=str(syllable_output),
                steps=steps,
                seed=syllable_seed,
                noise_level=noise_level,
                pipeline=pipeline
            )

            syllable_files.append(str(syllable_output))
            syllable_timings.append(timing)

        # Combine all syllables
        combine_syllable_outputs(
            syllable_files=syllable_files,
            syllable_timings=syllable_timings,
            output_path=output_path
        )

        # Optionally keep intermediate files
        if keep_intermediates:
            intermediate_dir = Path(output_path).parent / f"{Path(output_path).stem}_syllables"
            intermediate_dir.mkdir(exist_ok=True)

            for syl_file in syllable_files:
                import shutil
                shutil.copy(syl_file, intermediate_dir / Path(syl_file).name)

            print(f"\n📂 Intermediate files saved to: {intermediate_dir}")

    finally:
        # Clean up temp directory
        if not keep_intermediates:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temp directory")


def main():
    parser = argparse.ArgumentParser(
        description='Per-Syllable ACE-Step Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 generate_per_syllable_ace_step.py \\
      --midi input.mid \\
      --lyrics-map '{"0": "do", "1": "re", "2": "mi", "3": "fa", "4": "sol"}' \\
      --prompt "choir vocals" \\
      --output combined.wav

  # With custom parameters
  python3 generate_per_syllable_ace_step.py \\
      --midi input.mid \\
      --lyrics-map '{"0": "do", "1": "re"}' \\
      --prompt "soft vocals" \\
      --steps 150 \\
      --seed 42 \\
      --noise-level 0.3 \\
      --keep-intermediates \\
      --output output.wav
        """
    )

    parser.add_argument('--midi', required=True, help='Input MIDI file')
    parser.add_argument('--lyrics-map', required=True, help='JSON map of note index to syllable')
    parser.add_argument('--prompt', required=True, help='Generation prompt')
    parser.add_argument('--output', required=True, help='Output WAV file')
    parser.add_argument('--steps', type=int, default=100, help='Diffusion steps (default: 100)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--noise-level', type=float, default=0.4, help='Noise level for audio2audio (default: 0.4)')
    parser.add_argument('--keep-intermediates', action='store_true', help='Keep individual syllable files')

    args = parser.parse_args()

    # Parse lyrics map
    midi_lyric_map = json.loads(args.lyrics_map)

    # Run generation
    generate_per_syllable(
        midi_path=args.midi,
        midi_lyric_map=midi_lyric_map,
        prompt=args.prompt,
        output_path=args.output,
        steps=args.steps,
        seed=args.seed,
        noise_level=args.noise_level,
        keep_intermediates=args.keep_intermediates,
        pipeline=None  # Will use wrapper script
    )


if __name__ == '__main__':
    main()
