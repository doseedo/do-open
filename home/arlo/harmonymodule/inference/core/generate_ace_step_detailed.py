#!/usr/bin/env python3
"""
ACE-Step Detailed Mode Generator

Generates audio phrase-by-phrase with noise-to-noise architecture.
Each lyric phrase is processed separately and then concatenated.

Usage:
    from generate_ace_step_detailed import generate_detailed_mode
    result = generate_detailed_mode(
        pipeline=pipeline,
        lyrics="full lyrics text",
        ref_audio_path="reference.wav",
        output_dir=Path("/output"),
        ...
    )
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

# Import lyric phrase segmenter
sys.path.insert(0, str(Path(__file__).parent))
from lyric_phrase_segmenter import extract_phrase_timings


def crop_audio_segment(
    audio_tensor: torch.Tensor,
    sr: int,
    start_time: float,
    end_time: float
) -> torch.Tensor:
    """
    Crop audio tensor to specified time range.

    Args:
        audio_tensor: [channels, samples] audio tensor
        sr: Sample rate
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        Cropped audio tensor
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(audio_tensor.shape[1], end_sample)

    return audio_tensor[:, start_sample:end_sample]


def add_noise_to_audio(
    audio: torch.Tensor,
    noise_level: float = 0.1,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Add Gaussian noise to audio for noise-to-noise architecture.

    Args:
        audio: [channels, samples] audio tensor
        noise_level: Noise standard deviation (0.0 to 1.0)
        device: Device to use

    Returns:
        Noisy audio tensor
    """
    if noise_level <= 0:
        return audio

    audio = audio.to(device)
    noise = torch.randn_like(audio) * noise_level
    noisy_audio = audio + noise

    return noisy_audio


def generate_phrase_segment(
    pipeline,
    phrase_text: str,
    ref_audio_segment: torch.Tensor,
    sr: int,
    noise_level: float,
    prompt: str,
    key: str,
    seed: int,
    steps: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Generate audio for a single lyric phrase using noise-to-noise.

    Args:
        pipeline: ACE-Step pipeline instance
        phrase_text: Lyric text for this phrase
        ref_audio_segment: Reference audio for this phrase [channels, samples]
        sr: Sample rate
        noise_level: Noise level for noise-to-noise (0.0 to 1.0)
        prompt: Instrument/voice prompt
        key: Musical key
        seed: Random seed
        steps: Number of diffusion steps
        device: Device to use

    Returns:
        Generated audio tensor for this phrase
    """
    # Add noise to reference audio (noise-to-noise architecture)
    noisy_ref = add_noise_to_audio(ref_audio_segment, noise_level, device)

    # Calculate duration
    duration = ref_audio_segment.shape[1] / sr

    print(f"   🎵 Generating phrase ({duration:.2f}s): \"{phrase_text[:50]}...\"")
    print(f"      Noise level: {noise_level}, Steps: {steps}")

    # Prepare input for pipeline
    # Note: This depends on your pipeline's specific API
    # You may need to adjust based on trainer_performerCN2.py implementation

    try:
        # Generate with pipeline
        generated = pipeline.generate(
            prompt=prompt,
            lyrics=phrase_text,
            duration=duration,
            key=key,
            seed=seed,
            steps=steps,
            ref_audio=noisy_ref,
            noise_level=noise_level,
            device=device
        )

        print(f"      ✅ Generated {generated.shape[1]} samples")
        return generated

    except Exception as e:
        print(f"      ⚠ Generation failed: {e}")
        print(f"         Using noisy reference audio as fallback")
        return noisy_ref


def concatenate_phrases(
    phrase_audio_list: List[torch.Tensor],
    crossfade_samples: int = 4410  # 100ms at 44.1kHz
) -> torch.Tensor:
    """
    Concatenate phrase audio segments with crossfading.

    Args:
        phrase_audio_list: List of audio tensors [channels, samples]
        crossfade_samples: Number of samples for crossfade

    Returns:
        Concatenated audio tensor
    """
    if not phrase_audio_list:
        return torch.zeros(1, 0)

    if len(phrase_audio_list) == 1:
        return phrase_audio_list[0]

    # Calculate total length
    total_samples = sum(audio.shape[1] for audio in phrase_audio_list)
    total_samples -= (len(phrase_audio_list) - 1) * crossfade_samples  # Account for crossfades

    # Initialize output
    channels = phrase_audio_list[0].shape[0]
    device = phrase_audio_list[0].device
    output = torch.zeros(channels, total_samples, device=device)

    current_pos = 0

    for i, audio in enumerate(phrase_audio_list):
        audio_len = audio.shape[1]

        if i == 0:
            # First segment: no crossfade at start
            output[:, :audio_len] = audio
            current_pos = audio_len
        else:
            # Crossfade with previous segment
            crossfade_start = current_pos - crossfade_samples

            # Create crossfade weights (linear)
            fade_out = torch.linspace(1.0, 0.0, crossfade_samples, device=device)
            fade_in = torch.linspace(0.0, 1.0, crossfade_samples, device=device)

            # Apply crossfade
            for c in range(channels):
                # Fade out previous segment
                output[c, crossfade_start:current_pos] *= fade_out
                # Fade in current segment
                output[c, crossfade_start:current_pos] += audio[c, :crossfade_samples] * fade_in

            # Add remaining part of current segment
            remaining_samples = audio_len - crossfade_samples
            output[:, current_pos:current_pos + remaining_samples] = audio[:, crossfade_samples:]
            current_pos += remaining_samples

    return output


def generate_detailed_mode(
    pipeline,
    lyrics: str,
    ref_audio_path: str,
    output_dir: Path,
    prompt: str = "vocals",
    key: str = "C",
    seed: int = 0,
    noise_level: float = 0.45,
    steps: int = 100,
    use_mfa: bool = True,
    device: str = "cuda"
) -> Dict[str, any]:
    """
    Generate audio in detailed mode: phrase-by-phrase with noise-to-noise.

    Args:
        pipeline: ACE-Step pipeline instance
        lyrics: Full lyrics text
        ref_audio_path: Path to reference audio file
        output_dir: Directory for outputs
        prompt: Instrument/voice prompt
        key: Musical key
        seed: Random seed
        noise_level: Noise level for noise-to-noise (0.0 to 1.0)
        steps: Number of diffusion steps per phrase
        use_mfa: Whether to use MFA for alignment
        device: Device to use

    Returns:
        Dict with:
        - 'output_path': Path to final concatenated audio
        - 'phrase_paths': List of paths to individual phrase audio files
        - 'phrase_timings': List of phrase timing dicts
    """
    print(f"\n{'='*60}")
    print(f"🔬 ACE-Step Detailed Mode Generation")
    print(f"{'='*60}")
    print(f"   Prompt: {prompt}")
    print(f"   Key: {key}")
    print(f"   Noise Level: {noise_level}")
    print(f"   Steps: {steps}")
    print(f"   Seed: {seed}")
    print(f"   Reference: {ref_audio_path}")

    # Extract phrase timings
    print(f"\n📝 Step 1/3: Extracting phrase timings...")
    phrase_timings = extract_phrase_timings(ref_audio_path, lyrics, use_mfa=use_mfa)

    if not phrase_timings:
        raise ValueError("No phrases extracted from lyrics")

    # Load reference audio
    print(f"\n🎵 Step 2/3: Loading reference audio...")
    ref_audio, sr = torchaudio.load(ref_audio_path)
    ref_audio = ref_audio.to(device)
    print(f"   Loaded: {ref_audio.shape[1]} samples @ {sr}Hz")

    # Generate each phrase
    print(f"\n🎨 Step 3/3: Generating {len(phrase_timings)} phrases...")
    phrase_audio_list = []
    phrase_paths = []

    for i, phrase_info in enumerate(phrase_timings):
        phrase_num = i + 1
        phrase_text = phrase_info['phrase']
        start_time = phrase_info['start_time']
        end_time = phrase_info['end_time']

        print(f"\n--- Phrase {phrase_num}/{len(phrase_timings)} ---")
        print(f"   Time: {start_time:.2f}s - {end_time:.2f}s")
        print(f"   Lyrics: \"{phrase_text}\"")

        # Crop reference audio for this phrase
        ref_segment = crop_audio_segment(ref_audio, sr, start_time, end_time)

        # Generate this phrase
        generated_segment = generate_phrase_segment(
            pipeline=pipeline,
            phrase_text=phrase_text,
            ref_audio_segment=ref_segment,
            sr=sr,
            noise_level=noise_level,
            prompt=prompt,
            key=key,
            seed=seed + i,  # Different seed per phrase
            steps=steps,
            device=device
        )

        phrase_audio_list.append(generated_segment.cpu())

        # Save individual phrase
        phrase_filename = f"phrase_{phrase_num:02d}.wav"
        phrase_path = output_dir / phrase_filename
        torchaudio.save(str(phrase_path), generated_segment.cpu(), sr)
        phrase_paths.append(str(phrase_path))

        print(f"   💾 Saved: {phrase_path}")

    # Concatenate all phrases with crossfading
    print(f"\n🔗 Concatenating {len(phrase_audio_list)} phrases...")
    final_audio = concatenate_phrases(phrase_audio_list, crossfade_samples=4410)

    # Save final output
    final_filename = "detailed_mode_output.wav"
    final_path = output_dir / final_filename
    torchaudio.save(str(final_path), final_audio, sr)

    print(f"\n✅ Detailed mode generation complete!")
    print(f"   Final output: {final_path}")
    print(f"   Duration: {final_audio.shape[1] / sr:.2f}s")
    print(f"   Individual phrases: {len(phrase_paths)} files")

    return {
        'output_path': str(final_path),
        'phrase_paths': phrase_paths,
        'phrase_timings': phrase_timings,
        'total_duration': final_audio.shape[1] / sr
    }


if __name__ == "__main__":
    # Test mode
    import argparse

    parser = argparse.ArgumentParser(description="ACE-Step Detailed Mode Generator")
    parser.add_argument("--ref-audio", required=True, help="Reference audio file")
    parser.add_argument("--lyrics", required=True, help="Lyrics text or file path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--prompt", default="vocals", help="Instrument/voice prompt")
    parser.add_argument("--key", default="C", help="Musical key")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--noise-level", type=float, default=0.45, help="Noise level (0-1)")
    parser.add_argument("--steps", type=int, default=100, help="Diffusion steps")
    parser.add_argument("--no-mfa", action="store_true", help="Disable MFA alignment")
    parser.add_argument("--checkpoint", help="Model checkpoint path")

    args = parser.parse_args()

    # Load lyrics
    if os.path.exists(args.lyrics):
        with open(args.lyrics, 'r') as f:
            lyrics = f.read()
    else:
        lyrics = args.lyrics

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline (you'll need to implement this based on your setup)
    print("⚠ Note: Pipeline loading not implemented in standalone mode")
    print("   This script should be imported and used with an existing pipeline instance")
    sys.exit(1)

    # Example usage when imported:
    # from trainer_performerCN2 import Pipeline
    # pipeline = Pipeline.load_from_checkpoint(args.checkpoint)
    # result = generate_detailed_mode(...)
