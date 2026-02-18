#!/usr/bin/env python3
"""
ACE-Step Music Generation Wrapper

A clean, easy-to-use wrapper for the ACE-Step v1-3.5B music generation model.
This script provides a simple interface for generating music from text prompts
and lyrics, suitable for integration into backend services.

Example usage:
    # Basic text-to-music generation
    generator = ACEStepGenerator()
    audio_path = generator.generate(
        prompt="funk, pop, soul, 105 BPM, energetic",
        lyrics="[verse]\\nNeon lights they flicker bright",
        duration=30.0
    )

    # With custom parameters
    audio_path = generator.generate(
        prompt="rock, electric guitar, 130 bpm",
        duration=60.0,
        inference_steps=60,
        guidance_scale=15.0,
        seed=42
    )

    # Audio-to-audio generation
    audio_path = generator.audio_to_audio(
        reference_audio="input.wav",
        target_prompt="jazz, saxophone, smooth",
        strength=0.5
    )

Author: Auto-generated wrapper for ACE-Step
License: Apache 2.0
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, field

# Add ACE-Step to path
ACE_STEP_DIR = os.path.join(os.path.dirname(__file__), "ACE-Step")
sys.path.insert(0, ACE_STEP_DIR)

from acestep.pipeline_ace_step import ACEStepPipeline


@dataclass
class GenerationConfig:
    """Configuration for music generation with ACE-Step.

    Attributes:
        format: Output audio format ("wav", "mp3", "ogg", "flac")
        duration: Audio duration in seconds (-1 for random 30-240s)
        inference_steps: Number of diffusion steps (more = higher quality, slower)
        guidance_scale: Classifier-free guidance scale (higher = more prompt adherence)
        scheduler_type: Diffusion scheduler ("euler", "heun", "pingpong")
        cfg_type: CFG type ("apg" for Adaptive Projected Guidance, "cfg" for standard)
        omega_scale: APG omega scale parameter
        guidance_interval: Guidance interval for dynamic CFG
        guidance_interval_decay: Decay rate for guidance interval
        min_guidance_scale: Minimum guidance scale for dynamic CFG
        use_erg_tag: Use Empty Reference Guidance for tags
        use_erg_lyric: Use Empty Reference Guidance for lyrics
        use_erg_diffusion: Use Empty Reference Guidance during diffusion
        guidance_scale_text: Separate guidance scale for text
        guidance_scale_lyric: Separate guidance scale for lyrics
        device_id: CUDA device ID to use
        dtype: Model precision ("bfloat16" or "float32")
        torch_compile: Enable torch.compile for faster inference
        cpu_offload: Offload models to CPU when not in use (saves VRAM)
        overlapped_decode: Use overlapped decoding for faster processing
    """
    format: str = "wav"
    duration: float = 60.0
    inference_steps: int = 60
    guidance_scale: float = 15.0
    scheduler_type: str = "euler"
    cfg_type: str = "apg"
    omega_scale: float = 10.0
    guidance_interval: float = 0.5
    guidance_interval_decay: float = 0.0
    min_guidance_scale: float = 3.0
    use_erg_tag: bool = True
    use_erg_lyric: bool = True
    use_erg_diffusion: bool = True
    guidance_scale_text: float = 0.0
    guidance_scale_lyric: float = 0.0
    device_id: int = 0
    dtype: str = "bfloat16"
    torch_compile: bool = False
    cpu_offload: bool = False
    overlapped_decode: bool = False


class ACEStepGenerator:
    """Main wrapper class for ACE-Step music generation.

    This class provides a simplified interface to the ACE-Step model,
    handling model loading, parameter management, and audio generation.

    Example:
        generator = ACEStepGenerator(checkpoint_dir="./checkpoints")
        audio_path = generator.generate(
            prompt="pop, 120 bpm, upbeat",
            lyrics="[verse]\\nSinging in the rain",
            duration=30.0
        )
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        auto_load: bool = False
    ):
        """Initialize the ACE-Step generator.

        Args:
            checkpoint_dir: Path to model checkpoints. If None, downloads from HuggingFace.
            config: Default generation configuration. If None, uses defaults.
            auto_load: If True, loads the model immediately. Otherwise loads on first use.
        """
        self.config = config or GenerationConfig()
        self.checkpoint_dir = checkpoint_dir

        # Initialize pipeline
        self.pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint_dir,
            device_id=self.config.device_id,
            dtype=self.config.dtype,
            torch_compile=self.config.torch_compile,
            cpu_offload=self.config.cpu_offload,
            overlapped_decode=self.config.overlapped_decode,
        )

        if auto_load:
            self.load_model()

    def load_model(self):
        """Explicitly load the model into memory.

        The model will auto-load on first generation if not called explicitly.
        Call this if you want to pre-load the model to avoid first-call latency.
        """
        if not self.pipeline.loaded:
            print("Loading ACE-Step model (this may take a minute)...")
            self.pipeline.load_checkpoint(self.checkpoint_dir)
            print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        lyrics: str = "",
        duration: Optional[float] = None,
        output_path: Optional[str] = None,
        seed: Optional[int] = None,
        inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        batch_size: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate music from text prompt and optional lyrics.

        Args:
            prompt: Text description of the music (genre, instruments, BPM, mood, etc.)
                    Example: "funk, pop, soul, 105 BPM, energetic, guitar, drums"
            lyrics: Optional lyrics with structure tags like [verse], [chorus], etc.
                    Example: "[verse]\\nNeon lights they flicker bright\\n[chorus]\\nTurn it up"
            duration: Audio duration in seconds. If None, uses config default.
            output_path: Where to save the audio. If None, auto-generates filename.
            seed: Random seed for reproducibility. If None, uses random seed.
            inference_steps: Number of diffusion steps. If None, uses config default.
            guidance_scale: CFG guidance scale. If None, uses config default.
            batch_size: Number of variations to generate simultaneously.
            **kwargs: Additional parameters to override config defaults.

        Returns:
            Path to generated audio file(s). Returns string if batch_size=1,
            otherwise returns list of paths.

        Example:
            # Generate 30 seconds of funk music
            audio = generator.generate(
                prompt="funk, groovy, bass, 100 BPM",
                duration=30.0,
                seed=42
            )

            # Generate with lyrics
            audio = generator.generate(
                prompt="pop, female vocals, 120 BPM",
                lyrics="[verse]\\nDancing in the moonlight\\n[chorus]\\nAll night long",
                duration=60.0
            )
        """
        # Use provided params or fall back to config defaults
        duration = duration if duration is not None else self.config.duration
        inference_steps = inference_steps if inference_steps is not None else self.config.inference_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.config.guidance_scale

        # Generate output path if not provided
        if output_path is None:
            timestamp = int(time.time())
            output_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"generated_{timestamp}.{self.config.format}")

        # Prepare seeds
        manual_seeds = [seed] if seed is not None else None

        # Call pipeline
        result = self.pipeline(
            format=self.config.format,
            audio_duration=duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=inference_steps,
            guidance_scale=guidance_scale,
            scheduler_type=self.config.scheduler_type,
            cfg_type=self.config.cfg_type,
            omega_scale=self.config.omega_scale,
            manual_seeds=manual_seeds,
            guidance_interval=self.config.guidance_interval,
            guidance_interval_decay=self.config.guidance_interval_decay,
            min_guidance_scale=self.config.min_guidance_scale,
            use_erg_tag=self.config.use_erg_tag,
            use_erg_lyric=self.config.use_erg_lyric,
            use_erg_diffusion=self.config.use_erg_diffusion,
            guidance_scale_text=self.config.guidance_scale_text,
            guidance_scale_lyric=self.config.guidance_scale_lyric,
            save_path=output_path,
            batch_size=batch_size,
            **kwargs
        )

        return output_path if batch_size == 1 else result

    def audio_to_audio(
        self,
        reference_audio: str,
        target_prompt: str,
        target_lyrics: str = "",
        strength: float = 0.5,
        output_path: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """Transform existing audio based on text prompt (audio-to-audio generation).

        Args:
            reference_audio: Path to reference audio file
            target_prompt: Text description of desired output
            target_lyrics: Optional target lyrics
            strength: Reference audio influence (0.0-1.0). Higher = more similar to reference.
            output_path: Where to save output. If None, auto-generates filename.
            seed: Random seed for reproducibility
            **kwargs: Additional parameters to override config defaults

        Returns:
            Path to generated audio file

        Example:
            # Transform a pop song into jazz style
            audio = generator.audio_to_audio(
                reference_audio="pop_song.wav",
                target_prompt="jazz, saxophone, smooth, 90 BPM",
                strength=0.5
            )
        """
        if not os.path.exists(reference_audio):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        # Generate output path if not provided
        if output_path is None:
            timestamp = int(time.time())
            output_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"audio2audio_{timestamp}.{self.config.format}")

        # Prepare seeds
        manual_seeds = [seed] if seed is not None else None

        # Call pipeline with audio2audio enabled
        result = self.pipeline(
            format=self.config.format,
            prompt=target_prompt,
            lyrics=target_lyrics,
            infer_step=self.config.inference_steps,
            guidance_scale=self.config.guidance_scale,
            scheduler_type=self.config.scheduler_type,
            cfg_type=self.config.cfg_type,
            omega_scale=self.config.omega_scale,
            manual_seeds=manual_seeds,
            audio2audio_enable=True,
            ref_audio_input=reference_audio,
            ref_audio_strength=strength,
            save_path=output_path,
            **kwargs
        )

        return output_path

    def music_editing(
        self,
        source_audio: str,
        edit_target_prompt: str,
        edit_target_lyrics: str = "",
        n_min: float = 0.0,
        n_max: float = 1.0,
        n_avg: int = 1,
        output_path: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """Edit existing music with fine-grained control.

        Args:
            source_audio: Path to source audio file
            edit_target_prompt: Target description for editing
            edit_target_lyrics: Target lyrics for editing
            n_min: Start of edit range (0.0-1.0)
            n_max: End of edit range (0.0-1.0)
            n_avg: Number of averaging steps
            output_path: Where to save output
            seed: Random seed
            **kwargs: Additional parameters

        Returns:
            Path to edited audio file
        """
        if not os.path.exists(source_audio):
            raise FileNotFoundError(f"Source audio not found: {source_audio}")

        if output_path is None:
            timestamp = int(time.time())
            output_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"edited_{timestamp}.{self.config.format}")

        manual_seeds = [seed] if seed is not None else None

        result = self.pipeline(
            format=self.config.format,
            task="music_editing",
            src_audio_path=source_audio,
            edit_target_prompt=edit_target_prompt,
            edit_target_lyrics=edit_target_lyrics,
            edit_n_min=n_min,
            edit_n_max=n_max,
            edit_n_avg=n_avg,
            manual_seeds=manual_seeds,
            save_path=output_path,
            **kwargs
        )

        return output_path

    def set_config(self, config: GenerationConfig):
        """Update the default generation configuration.

        Args:
            config: New configuration to use as default
        """
        self.config = config

    def cleanup_memory(self):
        """Clean up GPU memory to prevent VRAM overflow.

        Call this between generations if you're running low on memory.
        """
        self.pipeline.cleanup_memory()


# Convenience presets for common music styles
STYLE_PRESETS = {
    "pop": "pop, synth, drums, guitar, 120 bpm, upbeat, catchy, vibrant",
    "rock": "rock, electric guitar, drums, bass, 130 bpm, energetic, rebellious, gritty",
    "hip_hop": "hip hop, 808 bass, hi-hats, synth, 90 bpm, bold, urban, intense",
    "country": "country, acoustic guitar, steel guitar, fiddle, 100 bpm, heartfelt, rustic, warm",
    "edm": "edm, synth, bass, kick drum, 128 bpm, euphoric, pulsating, energetic",
    "reggae": "reggae, guitar, bass, drums, 80 bpm, chill, soulful, positive",
    "classical": "classical, orchestral, strings, piano, 60 bpm, elegant, emotive, timeless",
    "jazz": "jazz, saxophone, piano, double bass, 110 bpm, smooth, improvisational, soulful",
    "metal": "metal, electric guitar, double kick drum, bass, 160 bpm, aggressive, intense, heavy",
    "rnb": "r&b, synth, bass, drums, 85 bpm, sultry, groovy, romantic",
    "funk": "funk, pop, soul, rock, melodic, guitar, drums, bass, keyboard, 105 BPM, energetic, groovy",
}


def get_style_preset(style: str) -> str:
    """Get a preset prompt for a music style.

    Args:
        style: Style name (e.g., "pop", "rock", "jazz")

    Returns:
        Preset prompt string

    Available styles:
        pop, rock, hip_hop, country, edm, reggae, classical, jazz, metal, rnb, funk
    """
    return STYLE_PRESETS.get(style.lower(), style)


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate music with ACE-Step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate funk music
  python ace_step_wrapper.py --prompt "funk, groovy, 100 BPM" --duration 30

  # Use a style preset
  python ace_step_wrapper.py --style jazz --duration 60

  # Generate with lyrics
  python ace_step_wrapper.py --prompt "pop, 120 BPM" --lyrics "[verse]\\nDancing tonight" --duration 30

  # Audio-to-audio transformation
  python ace_step_wrapper.py --audio2audio input.wav --target-prompt "jazz, smooth" --strength 0.5
        """
    )

    # Main generation args
    parser.add_argument("--prompt", type=str, help="Text description of music")
    parser.add_argument("--style", type=str, choices=list(STYLE_PRESETS.keys()),
                       help="Use a style preset instead of custom prompt")
    parser.add_argument("--lyrics", type=str, default="", help="Lyrics with structure tags")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    # Audio2audio args
    parser.add_argument("--audio2audio", type=str, help="Reference audio for audio2audio generation")
    parser.add_argument("--target-prompt", type=str, help="Target prompt for audio2audio")
    parser.add_argument("--strength", type=float, default=0.5, help="Reference audio strength (0.0-1.0)")

    # Model args
    parser.add_argument("--checkpoint-dir", type=str, help="Path to model checkpoints")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--steps", type=int, default=60, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=15.0, help="Guidance scale")
    parser.add_argument("--format", type=str, default="wav", choices=["wav", "mp3", "ogg", "flac"])

    args = parser.parse_args()

    # Validate args
    if args.audio2audio:
        if not args.target_prompt:
            parser.error("--target-prompt is required for audio2audio generation")
    else:
        if not args.prompt and not args.style:
            parser.error("Either --prompt or --style is required")

    # Create config
    config = GenerationConfig(
        format=args.format,
        duration=args.duration,
        inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        device_id=args.device,
        dtype=args.dtype,
    )

    # Create generator
    print("Initializing ACE-Step generator...")
    generator = ACEStepGenerator(
        checkpoint_dir=args.checkpoint_dir,
        config=config,
    )

    # Generate
    try:
        if args.audio2audio:
            print(f"Generating audio2audio from {args.audio2audio}...")
            output = generator.audio_to_audio(
                reference_audio=args.audio2audio,
                target_prompt=args.target_prompt,
                strength=args.strength,
                output_path=args.output,
                seed=args.seed,
            )
        else:
            prompt = get_style_preset(args.style) if args.style else args.prompt
            print(f"Generating music: {prompt[:50]}...")
            output = generator.generate(
                prompt=prompt,
                lyrics=args.lyrics,
                duration=args.duration,
                output_path=args.output,
                seed=args.seed,
            )

        print(f"\n✓ Generated successfully!")
        print(f"  Output: {output}")

    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
